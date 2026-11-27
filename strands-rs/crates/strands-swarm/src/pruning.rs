//! What to drop from an agent's context before each model call, beyond the SDK's pipeline.
//!
//! Reasoning: models that keep prior turns' thinking (Claude Opus 4.5+) bill it as input on every
//! later call, and its signatures alone are most of a long agent's reasoning bytes. A long tool
//! loop (one turn) re-sends every step's reasoning on every call. Bedrock requires the thinking
//! of "the last assistant message", whole and signed; Anthropic's prefix check lets the sent
//! thinking leave out "the oldest, the newest, or all of them, but not one in between". So all
//! reasoning goes but the latest assistant message's that has any: on Claude Haiku 5.5 dropping
//! an earlier step's thinking mid-loop was accepted with no prefix mismatch reported, while a
//! block without its signature was rejected. An edit to an earlier message (a trimmed result, a
//! summary) unbinds the thinking after it (the call then reported `thinking_mismatch_allowed`),
//! so that goes too, the latest included.
//!
//! Old tool output: a `shell`, `read` or MCP result is mostly read once, then re-sent on every call.
//! Older ones are cut to their head and tail; the full text stays in the stash under the
//! `retrieve_context` reference the preview ends with.
//!
//! Old images: a screenshot (from `read` or an MCP tool) is ~1-2k tokens and ~150 KB of base64 that
//! every later call re-sends and every save writes, yet the model has used it within a few steps.
//! Those more than [`KEEP_IMAGE_STEPS`] assistant messages back become a text stub naming where
//! they came from, so a re-read gets them back; they are not stashed, as that would keep the bytes.
//!
//! Old mail and replies: see [`crate::aging`].
//!
//! All four edit history, and the provider's prompt cache only matches an unchanged prefix: one edit
//! deep in a 150k-token history turns the next call's cache reads (0.1x) into cache writes
//! (1.25x) for everything after it. Run on every turn, they raised input cost per call by ~80% on
//! a live run. So [`CachePrune`] runs them only when the prefix is lost anyway, and then all at
//! once: see [`PruneReason`].

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use strands_agents::context_manager::stash::Stash;
use strands_agents::context_manager::strategies::{
    Offload, OffloadConditions, OffloadStrategy, OffloadTarget, TruncateConfig,
};
use strands_agents::context_manager::{ContextState, ContextStrategy};
use strands_agents::types::{
    ContentBlock, ImageBlock, ImageSource, Message, Role, ToolResultBlock, ToolResultContent, ToolUseBlock,
};
use strands_agents::{Agent, Result};

use parking_lot::Mutex;

use crate::compaction::CompactAt;
use crate::images::dimensions;
use crate::resume::NOTICE_PREFIX;

/// Stands in for an assistant message that held nothing but reasoning.
const REASONING_DROPPED: &str = "[reasoning dropped]";

/// Where the current turn starts: the last user message that carries no tool results.
/// A restart notice is not a new turn: the turn it interrupted goes on.
fn turn_start(messages: &[Message]) -> usize {
    messages.iter().rposition(starts_turn).unwrap_or(0)
}

/// Whether `message` starts a turn: a user message that carries no tool results and is no
/// restart notice.
fn starts_turn(message: &Message) -> bool {
    message.role == Role::User && message.tool_results().next().is_none() && !is_restart_notice(message)
}

fn is_restart_notice(message: &Message) -> bool {
    matches!(message.content.first(), Some(ContentBlock::Text(text)) if text.starts_with(NOTICE_PREFIX))
}

fn is_reasoning(block: &ContentBlock) -> bool {
    matches!(block, ContentBlock::Reasoning(_))
}

fn has_reasoning(message: &Message) -> bool {
    message.role == Role::Assistant && message.content.iter().any(is_reasoning)
}

/// The one message whose reasoning is kept: the current turn's latest assistant message that has
/// any, which in a tool loop is usually the one whose tool use is being answered.
fn kept_reasoning(messages: &[Message]) -> Option<usize> {
    let start = turn_start(messages);
    let latest = messages[start..].iter().rposition(has_reasoning)?;
    Some(start + latest)
}

/// Removes reasoning from every assistant message but [`kept_reasoning`]'s; `true` when any went.
/// A message left empty gets [`REASONING_DROPPED`] so the history stays valid.
fn drop_old_reasoning(messages: &mut [Message]) -> bool {
    let kept = kept_reasoning(messages);
    let mut acted = false;
    for (i, message) in messages.iter_mut().enumerate() {
        if Some(i) != kept {
            acted |= strip_reasoning(message);
        }
    }
    acted
}

/// Removes reasoning from every message after `edited`, the latest one's too; `true` when any
/// went. Anthropic's prefix check binds a thinking block to everything before it, so after an edit
/// ("Changing, inserting, moving, or deleting an earlier message counts") later thinking no longer
/// matches; leaving out the newest blocks is allowed.
fn drop_reasoning_after(messages: &mut [Message], edited: usize) -> bool {
    let mut acted = false;
    for message in messages.iter_mut().skip(edited + 1) {
        acted |= strip_reasoning(message);
    }
    acted
}

/// Removes `message`'s reasoning, leaving [`REASONING_DROPPED`] when nothing else is left.
fn strip_reasoning(message: &mut Message) -> bool {
    if !has_reasoning(message) {
        return false;
    }
    message.content.retain(|block| !is_reasoning(block));
    if message.content.is_empty() {
        message.content.push(ContentBlock::text(REASONING_DROPPED));
    }
    true
}

/// A hash of `message` without its reasoning: dropping thinking is not an edit to the prefix
/// check (a live call reported no mismatch), anything else is.
fn hash_without_reasoning(message: &Message) -> u64 {
    let mut message = message.clone();
    message.content.retain(|block| !is_reasoning(block));
    hash(&message)
}

/// The first of `messages` whose hash differs from `hashes`'; messages past either's end are not
/// compared (appended ones are not edits).
fn first_change(hashes: &[u64], messages: &[Message], hash: fn(&Message) -> u64) -> Option<usize> {
    hashes.iter().zip(messages).position(|(h, m)| *h != hash(m))
}

/// Tools whose older results stay whole: a loaded skill is the procedure the agent follows, and
/// mail (a child's report) arrives in `wait` results. `retrieve_context` reads the stash back,
/// and the SDK never trims what it returned.
const UNTRIMMED_TOOLS: [&str; 3] = ["skills", "wait", "retrieve_context"];
/// The latest this many trimmable results stay whole.
const KEEP_RECENT_RESULTS: usize = 6;
/// Older results longer than this (in characters) are trimmed.
const TRIM_ABOVE_CHARS: usize = 2_000;
/// What a trimmed result keeps, in tokens (about 800 characters, head and tail).
const TRIM_PREVIEW_TOKENS: usize = 200;

/// Whether `tool`'s older results are trimmed: any tool's but [`UNTRIMMED_TOOLS`]'. An MCP
/// listing (a code review's JSON, a search's hits) is read once like a `shell` result.
fn trimmed(tool: &str) -> bool {
    !UNTRIMMED_TOOLS.contains(&tool)
}

/// Trims results longer than [`TRIM_ABOVE_CHARS`] once [`KEEP_RECENT_RESULTS`] newer ones exist,
/// whenever it runs: gated by [`CachePrune`]. See [`trimmed`] for which.
fn trim_old_results() -> OffloadStrategy {
    let truncate = Offload::truncate(
        OffloadTarget::tools(UNTRIMMED_TOOLS.map(|tool| format!("!{tool}"))),
        TruncateConfig {
            preview_tokens: Some(TRIM_PREVIEW_TOKENS),
            preview: None,
        },
    );
    truncate.when(OffloadConditions {
        threshold: Some(TRIM_ABOVE_CHARS.div_ceil(4) as f64),
        preserve_recent: Some(KEEP_RECENT_RESULTS as f64),
        ..Default::default()
    })
}

/// Tool-result images stay whole while fewer than this many assistant messages follow them. A
/// screenshot is acted on in the next step or two and checked against the following one, so 4
/// covers that and a retry, while the latest step's images (0 messages after) always stay. Older
/// ones cost ~1.5k tokens each on every call for little use, and a re-read is cheap.
const KEEP_IMAGE_STEPS: usize = 4;

/// What an image in the tool result of `tool` (called with `input`) becomes: where it came from,
/// its size, and how to get it back.
fn image_stub(tool: &str, input: &serde_json::Value, image: &ImageBlock) -> String {
    let size = match &image.source {
        ImageSource::Bytes(bytes) => {
            dimensions(bytes).map_or("size unknown".into(), |(w, h)| format!("{w}x{h}"))
        }
        _ => "size unknown".into(),
    };
    match (tool, input.get("path").and_then(|path| path.as_str())) {
        ("read", Some(path)) => format!("[image omitted: {path}, {size}; read it again if needed]"),
        _ => format!("[image omitted: {tool} result, {size}; call the tool again if needed]"),
    }
}

/// The images in tool results that [`KEEP_IMAGE_STEPS`] assistant messages or more follow, as
/// (message, block, item, stub, base64 size). Results of tools that are not [`trimmed`] (mail's
/// attachments arrive in `wait`) keep theirs.
fn stale_images(messages: &[Message]) -> Vec<(usize, usize, usize, String, usize)> {
    let uses: HashMap<&str, &ToolUseBlock> = messages
        .iter()
        .flat_map(|m| m.tool_uses())
        .map(|u| (u.tool_use_id.as_str(), u))
        .collect();
    let mut stale = Vec::new();
    let mut steps = 0;
    for (m, message) in messages.iter().enumerate().rev() {
        if message.role == Role::Assistant {
            steps += 1;
        }
        if steps < KEEP_IMAGE_STEPS || message.role != Role::User {
            continue;
        }
        for (b, block) in message.content.iter().enumerate() {
            let ContentBlock::ToolResult(result) = block else {
                continue;
            };
            let Some(call) = uses.get(result.tool_use_id.as_str()).filter(|u| trimmed(&u.name)) else {
                continue;
            };
            for (i, item) in result.content.iter().enumerate() {
                if let ToolResultContent::Image(image) = item {
                    let size = match &image.source {
                        ImageSource::Bytes(bytes) => bytes.len() / 3 * 4,
                        _ => 0,
                    };
                    stale.push((m, b, i, image_stub(&call.name, &call.input, image), size));
                }
            }
        }
    }
    stale
}

/// Replaces [`stale_images`] by their stubs; `true` when any went.
fn drop_old_images(messages: &mut [Message]) -> bool {
    let stale = stale_images(messages);
    for (m, b, i, stub, _) in &stale {
        if let ContentBlock::ToolResult(result) = &mut messages[*m].content[*b] {
            result.content[*i] = ToolResultContent::Text(stub.clone());
        }
    }
    !stale.is_empty()
}

/// How long the provider keeps a cached prompt prefix by default (Anthropic and Bedrock: 5
/// minutes, refreshed on every hit).
pub const CACHE_TTL: Duration = Duration::from_secs(5 * 60);
/// The cache lifetime with `--cache-ttl 1h`.
pub const CACHE_TTL_1H: Duration = Duration::from_secs(60 * 60);
/// What a cached input token costs to read, in input tokens.
const READ_COST: f64 = 0.1;
/// What writing a token to the cache costs, in input tokens: 1.25 for a [`CACHE_TTL`] cache, 2
/// for [`CACHE_TTL_1H`].
fn write_cost(ttl: Duration) -> f64 {
    match ttl > CACHE_TTL {
        true => 2.0,
        false => 1.25,
    }
}
/// Prune for its own sake once the removable part of the history has cost this many times more
/// in reads than the prune's cache rewrite would. Each prune rewrites the cached conversation
/// (the write price instead of the read price per token) to save the read price of what it
/// removes on every later call; waiting costs the removable part's reads each call. Pruning when
/// those reads add up to the rewrite is never worse than twice the best time. A replay over the
/// live run's histories (51.9k calls) put the cheapest patience at 1.5 for both cache
/// lifetimes, under a fixed 40% share of the history by 2.7% (5-minute cache) and 0.7% (1 hour):
/// that share pruned every 12 calls where the rewrite took 29 to pay back, and so lost on small
/// histories where a prune frees little to read.
pub const PRUNE_PATIENCE: f64 = 1.5;
/// Agent state key: when the agent last called its model (Unix milliseconds). Kept in the
/// session, so [`PruneReason::Idle`] holds across a restart.
pub const LAST_CALL_KEY: &str = "swarm_last_model_call_ms";

/// Why [`CachePrune`] prunes on this call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneReason {
    /// The model rejected the context as too long.
    Overflow,
    /// Older turns are summarized in this same pipeline pass.
    Compaction,
    /// History changed since the last call (the `context` tool offloaded results, say).
    Edited,
    /// The system prompt (a `cd`) or the tool specs (a `find_tools` enable) changed since the
    /// last call: they precede the history in the cache.
    Prompt,
    /// The last call was longer ago than the cache lives ([`CACHE_TTL`] by default), so the cache
    /// has expired.
    Idle,
    /// Reading what pruning would remove has cost [`PRUNE_PATIENCE`] cache rewrites since the
    /// history was last rewritten.
    Large,
}

/// Drops reasoning the model no longer needs (all but [`kept_reasoning`]'s) and trims old
/// results, together and only for a [`PruneReason`]; on any other call history reaches the
/// model byte for byte as last time. Goes first in the pipeline, with [`CachePrune::mark`] last.
#[derive(Clone)]
pub struct CachePrune {
    gate: Arc<Gate>,
}

struct Gate {
    trim: OffloadStrategy,
    compaction: Option<Arc<CompactAt>>,
    /// How long the cache lives.
    ttl: Duration,
    /// A hash of each message as the last call sent it (this process only).
    sent: Mutex<Option<Vec<u64>>>,
    /// A hash of the system prompt and tool specs the last call sent (this process only).
    prompt: Mutex<Option<u64>>,
    /// What reading the removable part of the history has cost (in input characters) over the
    /// calls since the cache last held a rewritten history: [`PruneReason::Large`]'s meter.
    reads: Mutex<f64>,
}

impl CachePrune {
    /// A gate for a cache that lives `ttl` ([`CACHE_TTL`] or [`CACHE_TTL_1H`]), which also
    /// prunes when `compaction` will summarize in the same pass.
    pub fn new(compaction: Option<Arc<CompactAt>>, ttl: Duration) -> Self {
        let gate = Gate {
            trim: trim_old_results(),
            compaction,
            ttl,
            sent: Mutex::new(None),
            prompt: Mutex::new(None),
            reads: Mutex::new(0.0),
        };
        Self { gate: Arc::new(gate) }
    }

    /// The pipeline's last strategy: notes what this call sends and when, for the next call's
    /// [`PruneReason::Edited`], [`PruneReason::Prompt`] and [`PruneReason::Idle`]. It never edits
    /// the context.
    pub fn mark(&self) -> Arc<dyn ContextStrategy> {
        Arc::new(Mark(Arc::clone(&self.gate)))
    }

    /// Why to prune before this call, if at all. Arms the compaction it reports.
    pub fn reason(&self, ctx: &ContextState<'_>) -> Option<PruneReason> {
        let compacting = self.gate.compaction.as_ref().is_some_and(|c| c.arm(ctx));
        if ctx.overflow {
            return Some(PruneReason::Overflow);
        }
        if compacting {
            return Some(PruneReason::Compaction);
        }
        if self.edited(ctx.messages) {
            return Some(PruneReason::Edited);
        }
        if self.prompt_changed(ctx.agent) {
            return Some(PruneReason::Prompt);
        }
        let idle = last_call(ctx.agent)
            .is_some_and(|at| SystemTime::now().duration_since(at).unwrap_or_default() > self.gate.ttl);
        if idle {
            return Some(PruneReason::Idle);
        }
        let (removable, total) = removable_chars(ctx.messages, ctx.stash.is_some());
        let mut reads = self.gate.reads.lock();
        *reads += READ_COST * removable as f64;
        let rewrite = (write_cost(self.gate.ttl) - READ_COST) * (total - removable) as f64;
        (removable > 0 && *reads >= PRUNE_PATIENCE * rewrite).then_some(PruneReason::Large)
    }

    fn edited(&self, messages: &[Message]) -> bool {
        match &*self.gate.sent.lock() {
            None => false,
            Some(sent) => sent.len() > messages.len() || first_change(sent, messages, hash).is_some(),
        }
    }

    fn prompt_changed(&self, agent: &Agent) -> bool {
        self.gate
            .prompt
            .lock()
            .is_some_and(|sent| sent != prompt_hash(agent))
    }
}

/// A hash of what precedes the history in a request: `agent`'s system prompt and tool specs.
fn prompt_hash(agent: &Agent) -> u64 {
    let specs: Vec<_> = agent.tools().iter().map(|tool| tool.spec().clone()).collect();
    let mut hasher = DefaultHasher::new();
    serde_json::to_string(&(agent.system_prompt(), specs))
        .unwrap_or_default()
        .hash(&mut hasher);
    hasher.finish()
}

#[async_trait]
impl ContextStrategy for CachePrune {
    fn name(&self) -> &str {
        "swarm:cache-prune"
    }

    fn init(&self, agent: &Agent, stash: Option<Arc<Stash>>) {
        self.gate.trim.init(agent, stash);
    }

    async fn apply(&self, ctx: &mut ContextState<'_>) -> Result<bool> {
        let Some(reason) = self.reason(ctx) else {
            return Ok(false);
        };
        // Whatever the reason, the history is sent rewritten (or the cache is cold anyway).
        *self.gate.reads.lock() = 0.0;
        let before: Vec<u64> = ctx.messages.iter().map(hash_without_reasoning).collect();
        let since_sent =
            (self.gate.sent.lock().as_deref()).and_then(|sent| first_change(sent, ctx.messages, hash));
        let dropped = drop_old_reasoning(ctx.messages);
        let trimmed = self.gate.trim.apply(ctx).await?;
        // After `before`, so an image going counts as an edit to the prefix like a trimmed result.
        let trimmed = drop_old_images(ctx.messages) | trimmed;
        let trimmed = match ctx.stash.clone() {
            Some(stash) => crate::aging::age(ctx.messages, starts_turn, &stash).await | trimmed,
            None => trimmed,
        };
        // A summary (compaction, or the overflow it answers) rewrites everything before the tail;
        // a new system prompt or tool set precedes all of it, so every thinking block goes.
        let edited = match reason {
            _ if self.prompt_changed(ctx.agent) => Some(0),
            PruneReason::Compaction | PruneReason::Overflow => Some(0),
            _ => {
                let now = first_change(&before, ctx.messages, hash_without_reasoning);
                since_sent.into_iter().chain(now).min()
            }
        };
        let after_edit = edited.is_some_and(|at| drop_reasoning_after(ctx.messages, at));
        if dropped || trimmed || after_edit {
            tracing::debug!(
                ?reason,
                dropped,
                trimmed,
                ?edited,
                "pruned old reasoning and results"
            );
        }
        Ok(dropped || trimmed || after_edit)
    }
}

struct Mark(Arc<Gate>);

#[async_trait]
impl ContextStrategy for Mark {
    fn name(&self) -> &str {
        "swarm:cache-mark"
    }

    async fn apply(&self, ctx: &mut ContextState<'_>) -> Result<bool> {
        *self.0.sent.lock() = Some(ctx.messages.iter().map(hash).collect());
        *self.0.prompt.lock() = Some(prompt_hash(ctx.agent));
        note_model_call(ctx.agent);
        Ok(false)
    }
}

/// Record that `agent`'s prompt was just sent (and so its cache refreshed), for
/// [`PruneReason::Idle`]. Keep-warm refreshes call it too: otherwise the first call after a long
/// wait would find the cache "expired", prune, and so rewrite the cache keep-warm kept.
pub fn note_model_call(agent: &Agent) {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    agent.state().set(LAST_CALL_KEY, now.as_millis() as u64);
}

/// When `agent` last sent its prompt (a call or a keep-warm refresh), as [`note_model_call`] noted.
pub(crate) fn last_call(agent: &Agent) -> Option<SystemTime> {
    let millis = agent.state().get(LAST_CALL_KEY)?.as_u64()?;
    Some(UNIX_EPOCH + Duration::from_millis(millis))
}

fn hash(message: &Message) -> u64 {
    let mut hasher = DefaultHasher::new();
    serde_json::to_string(message)
        .unwrap_or_default()
        .hash(&mut hasher);
    hasher.finish()
}

/// How much of the history (JSON characters) pruning would remove now, and its whole length;
/// with a stash, old mail and replies ([`crate::aging`]) count too.
fn removable_chars(messages: &[Message], stash: bool) -> (usize, usize) {
    let total: usize = messages.iter().map(json_len).sum();
    if total == 0 {
        return (0, 0);
    }
    let kept = kept_reasoning(messages);
    let reasoning: usize = messages
        .iter()
        .enumerate()
        .filter(|(i, m)| Some(*i) != kept && m.role == Role::Assistant)
        .flat_map(|(_, m)| &m.content)
        .filter(|block| is_reasoning(block))
        .map(json_len)
        .sum();
    let preview = TRIM_PREVIEW_TOKENS * 4;
    let results: usize = old_trimmed_results(messages, KEEP_RECENT_RESULTS)
        .map(result_chars)
        .filter(|chars| *chars > TRIM_ABOVE_CHARS)
        .map(|chars| chars.saturating_sub(preview))
        .sum();
    let images: usize = stale_images(messages)
        .iter()
        .map(|(.., stub, size)| size.saturating_sub(stub.len()))
        .sum();
    let aged = match stash {
        true => crate::aging::removable(messages, starts_turn),
        false => 0,
    };
    ((reasoning + results + images + aged).min(total), total)
}

fn json_len<T: serde::Serialize>(value: &T) -> usize {
    serde_json::to_string(value).map_or(0, |json| json.len())
}

/// Trimmable results (see [`trimmed`]) with at least `keep` newer ones.
fn old_trimmed_results(messages: &[Message], keep: usize) -> impl Iterator<Item = &ToolResultBlock> {
    let names: HashMap<&str, &str> = messages
        .iter()
        .flat_map(|m| m.tool_uses())
        .map(|u| (u.tool_use_id.as_str(), u.name.as_str()))
        .collect();
    let is_trimmed = |r: &&ToolResultBlock| {
        names
            .get(r.tool_use_id.as_str())
            .is_some_and(|name| trimmed(name))
    };
    let matching: Vec<&Message> = messages
        .iter()
        .filter(|m| m.role == Role::User && m.tool_results().any(|r| is_trimmed(&r)))
        .collect();
    let old = matching.len().saturating_sub(keep);
    let results: Vec<&ToolResultBlock> = matching[..old]
        .iter()
        .flat_map(|m| m.tool_results().filter(is_trimmed))
        .collect();
    results.into_iter()
}

fn result_chars(result: &ToolResultBlock) -> usize {
    result
        .content
        .iter()
        .map(|item| match item {
            ToolResultContent::Text(text) => text.chars().count(),
            ToolResultContent::Json(json) => json.to_string().chars().count(),
            _ => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use strands_agents::context_manager::{ContextManagerConfig, ContextManagerOption};
    use strands_agents::models::{ContentBlockDelta, ModelEventStream, ModelStreamEvent, StreamOptions};
    use strands_agents::testing::{MockMessageModel, MockResponse};
    use strands_agents::tools::{FunctionTool, ToolContext, ToolRef};
    use strands_agents::types::{
        ReasoningBlock, SystemPrompt, ToolResultBlock, ToolResultContent, ToolUseBlock,
    };
    use strands_agents::{InMemoryStorage, Model};

    use crate::compaction::CompactAt;

    fn thinking(text: &str) -> ContentBlock {
        ContentBlock::Reasoning(ReasoningBlock {
            text: Some(text.to_owned()),
            signature: Some("sig".repeat(100)),
            ..Default::default()
        })
    }

    fn assistant(blocks: Vec<ContentBlock>) -> Message {
        Message::new(Role::Assistant, blocks)
    }

    fn tool_call(id: &str) -> ContentBlock {
        ContentBlock::ToolUse(ToolUseBlock::new("shell", id, json!({})))
    }

    fn result(id: &str, text: String) -> Message {
        Message::new(
            Role::User,
            vec![ContentBlock::ToolResult(ToolResultBlock::success(
                id,
                vec![ToolResultContent::Text(text)],
            ))],
        )
    }

    fn tool_result(id: &str) -> Message {
        result(id, "ok".into())
    }

    fn agent() -> Agent {
        Agent::builder()
            .model(MockMessageModel::shared(vec![]))
            .no_sandbox()
            .build()
            .unwrap()
    }

    fn ctx<'a>(messages: &'a mut Vec<Message>, agent: &'a Agent, utilization: f64) -> ContextState<'a> {
        let stash = Arc::new(Stash::new(Arc::new(InMemoryStorage::new()), "s", "a1", false));
        ContextState {
            messages,
            agent,
            utilization,
            overflow: false,
            stash: Some(stash),
        }
    }

    fn minutes_ago(minutes: u64) -> u64 {
        let at = SystemTime::now() - Duration::from_secs(minutes * 60);
        at.duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
    }

    fn reasoning_count(messages: &[Message]) -> usize {
        messages
            .iter()
            .flat_map(|m| &m.content)
            .filter(|b| matches!(b, ContentBlock::Reasoning(_)))
            .count()
    }

    #[test]
    fn drops_all_reasoning_but_the_tool_loop_s_latest() {
        let mut messages = vec![
            Message::user("first task"),
            assistant(vec![thinking("old"), tool_call("t1")]),
            tool_result("t1"),
            assistant(vec![thinking("old again"), ContentBlock::text("done")]),
            Message::user("second task"),
            assistant(vec![thinking("now"), tool_call("t2")]),
            tool_result("t2"),
            assistant(vec![thinking("still now"), tool_call("t3")]),
            tool_result("t3"),
        ];

        assert!(drop_old_reasoning(&mut messages));

        assert_eq!(messages[1].content, vec![tool_call("t1")]);
        assert_eq!(messages[3].content, vec![ContentBlock::text("done")]);
        assert_eq!(
            messages[5].content,
            vec![tool_call("t2")],
            "an earlier step of the loop"
        );
        assert_eq!(
            messages[7].content,
            vec![thinking("still now"), tool_call("t3")],
            "the message whose tool use is being answered keeps its reasoning whole"
        );
        assert_eq!(reasoning_count(&messages), 1);
        assert!(
            !drop_old_reasoning(&mut messages),
            "a second pass changes nothing"
        );
    }

    #[test]
    fn the_latest_reasoning_is_kept_when_the_last_steps_had_none() {
        let mut messages = vec![
            Message::user("task"),
            assistant(vec![thinking("first"), tool_call("t1")]),
            tool_result("t1"),
            assistant(vec![thinking("second"), tool_call("t2")]),
            tool_result("t2"),
            assistant(vec![ContentBlock::text("no thinking"), tool_call("t3")]),
            tool_result("t3"),
        ];
        assert_eq!(kept_reasoning(&messages), Some(3));

        assert!(drop_old_reasoning(&mut messages));

        assert_eq!(messages[1].content, vec![tool_call("t1")]);
        assert_eq!(messages[3].content, vec![thinking("second"), tool_call("t2")]);
    }

    #[test]
    fn a_new_turn_keeps_no_earlier_reasoning() {
        let mut messages = vec![
            Message::user("task"),
            assistant(vec![thinking("a"), ContentBlock::text("done")]),
            Message::user("next task"),
        ];
        assert_eq!(kept_reasoning(&messages), None);
        assert!(drop_old_reasoning(&mut messages));
        assert_eq!(reasoning_count(&messages), 0);
    }

    #[test]
    fn a_single_turn_keeps_everything() {
        let mut messages = vec![
            Message::user("task"),
            assistant(vec![thinking("a"), tool_call("t1")]),
            tool_result("t1"),
        ];
        assert!(!drop_old_reasoning(&mut messages));
        assert_eq!(reasoning_count(&messages), 1);
    }

    #[test]
    fn a_message_of_only_reasoning_keeps_a_placeholder() {
        let mut messages = vec![
            Message::user("task"),
            assistant(vec![thinking("cut off")]),
            Message::user("go on"),
        ];
        assert!(drop_old_reasoning(&mut messages));
        assert_eq!(messages[1].content, vec![ContentBlock::text(REASONING_DROPPED)]);
    }

    /// A task, then one `name` call and its result per entry.
    fn history(results: &[(&str, usize)]) -> Vec<Message> {
        let mut messages = vec![Message::user("task")];
        for (i, (name, chars)) in results.iter().enumerate() {
            let id = format!("t{i}");
            messages.push(assistant(vec![ContentBlock::ToolUse(ToolUseBlock::new(
                *name,
                &id,
                json!({}),
            ))]));
            messages.push(result(&id, "x".repeat(*chars)));
        }
        messages
    }

    fn result_text(message: &Message) -> String {
        message
            .tool_results()
            .next()
            .unwrap()
            .content
            .iter()
            .map(|item| match item {
                ToolResultContent::Text(text) => text.clone(),
                _ => String::new(),
            })
            .collect()
    }

    async fn trim(messages: &mut Vec<Message>) -> bool {
        let agent = agent();
        trim_old_results()
            .apply(&mut ctx(messages, &agent, 0.0))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn trims_older_large_tool_results_to_a_preview_with_a_reference() {
        let mcp = "builder-mcp_CodeReviewReadActions";
        let mut results = vec![
            ("read", 5_000),
            ("skills", 5_000),
            ("shell", 100),
            (mcp, 5_000),
            ("wait", 5_000),
        ];
        results.extend([("shell", 5_000); 2 * KEEP_RECENT_RESULTS]);
        let mut messages = history(&results);

        assert!(trim(&mut messages).await);

        let text = |i: usize| result_text(&messages[2 + 2 * i]);
        assert!(text(0).len() < TRIM_ABOVE_CHARS, "{}", text(0));
        assert!(
            text(0).contains("chars elided") && text(0).contains("[ref: t0_0]"),
            "{}",
            text(0)
        );
        assert_eq!(text(1).len(), 5_000, "a loaded skill is left alone");
        assert_eq!(text(2).len(), 100, "small results are left alone");
        assert!(text(3).len() < TRIM_ABOVE_CHARS, "an MCP result: {}", text(3));
        assert_eq!(text(4).len(), 5_000, "mail in a wait result is left alone");
        let last_trimmed = results.len() - KEEP_RECENT_RESULTS - 1;
        assert!(text(last_trimmed).len() < TRIM_ABOVE_CHARS);
        for i in last_trimmed + 1..results.len() {
            assert_eq!(text(i).len(), 5_000, "the latest results stay whole");
        }
        assert!(!trim(&mut messages).await, "nothing left to trim");
    }

    /// One pipeline pass: the gate, then the mark. `true` when the gate pruned.
    async fn pass(prune: &CachePrune, agent: &Agent, messages: &mut Vec<Message>, utilization: f64) -> bool {
        let mut ctx = ctx(messages, agent, utilization);
        let pruned = prune.apply(&mut ctx).await.unwrap();
        prune.mark().apply(&mut ctx).await.unwrap();
        pruned
    }

    /// A finished turn with old reasoning and one large old `shell` result, then a second turn
    /// whose recent results are large enough that pruning would save under a few per cent of the history.
    fn prunable() -> Vec<Message> {
        let mut messages = vec![Message::user("first task")];
        messages.push(assistant(vec![thinking("old"), tool_call("t0")]));
        messages.push(result("t0", "x".repeat(3_000)));
        messages.push(assistant(vec![ContentBlock::text("done")]));
        let mut second = history(&[("shell", 6_000); KEEP_RECENT_RESULTS]);
        second[0] = Message::user("second task");
        messages.extend(second);
        assert!(share(&messages, true) < 0.4, "{}", share(&messages, true));
        messages
    }

    /// The part of the history pruning would remove now.
    fn share(messages: &[Message], stash: bool) -> f64 {
        let (removable, total) = removable_chars(messages, stash);
        removable as f64 / total as f64
    }

    /// Passes (warm calls with nothing new) until the gate prunes, at most `limit`.
    async fn passes_until_pruned(
        prune: &CachePrune,
        agent: &Agent,
        messages: &mut Vec<Message>,
        limit: usize,
    ) -> Option<usize> {
        for n in 1..=limit {
            if pass(prune, agent, messages, 0.1).await {
                return Some(n);
            }
        }
        None
    }

    fn pruned(messages: &[Message]) -> bool {
        reasoning_count(&messages[..4]) == 0 && result_text(&messages[2]).len() < TRIM_ABOVE_CHARS
    }

    fn json(messages: &[Message]) -> String {
        serde_json::to_string(messages).unwrap()
    }

    #[tokio::test]
    async fn a_normal_next_call_leaves_history_byte_identical() {
        let (prune, agent) = (CachePrune::new(None, CACHE_TTL), agent());
        let mut messages = prunable();
        assert!(!pass(&prune, &agent, &mut messages, 0.1).await);
        let sent = json(&messages);

        let write = ToolUseBlock::new("write", "t9", json!({}));
        messages.push(assistant(vec![thinking("next"), ContentBlock::ToolUse(write)]));
        messages.push(tool_result("t9"));
        let before = messages.clone();
        assert!(!pass(&prune, &agent, &mut messages, 0.1).await);
        assert_eq!(messages, before);
        assert!(
            json(&messages).starts_with(&sent[..sent.len() - 1]),
            "the sent prefix is unchanged"
        );
        assert!(!pruned(&messages));
    }

    #[tokio::test]
    async fn prunes_in_the_same_pass_as_a_compaction_and_keeps_it_armed() {
        let model = MockMessageModel::shared(vec![
            MockResponse::message(Message::assistant("summary")),
            MockResponse::message(Message::assistant("unused")),
        ]);
        let agent = Agent::builder()
            .model(model.clone())
            .no_sandbox()
            .build()
            .unwrap();
        let compact = Arc::new(CompactAt::new(Some(1_000)));
        let prune = CachePrune::new(Some(compact.clone()), CACHE_TTL);
        let mut messages = prunable();
        let latest = messages.len() - 2;
        messages[latest].content.insert(0, thinking("latest"));
        let mut ctx = ctx(&mut messages, &agent, 0.5);
        assert_eq!(prune.reason(&ctx), Some(PruneReason::Compaction));
        assert!(prune.apply(&mut ctx).await.unwrap());
        assert_eq!(
            reasoning_count(ctx.messages),
            0,
            "the summary unbinds all thinking"
        );
        // Pruning lowered the count below the point (1k of the mock's assumed 200k window), though
        // not below the target; the armed compaction still runs, once.
        ctx.utilization = 800.0 / 200_000.0;
        assert!(compact.apply(&mut ctx).await.unwrap());
        assert!(!compact.apply(&mut ctx).await.unwrap());
        assert_eq!(model.call_count(), 1);
        assert!(
            messages[0].text().contains("summary"),
            "the summary follows the task"
        );
    }

    #[tokio::test]
    async fn prunes_on_the_first_call_after_the_cache_expired() {
        let (prune, agent) = (CachePrune::new(None, CACHE_TTL), agent());
        let mut messages = prunable();
        assert!(!pass(&prune, &agent, &mut messages, 0.1).await);

        agent.state().set(LAST_CALL_KEY, minutes_ago(4));
        assert!(!pass(&prune, &agent, &mut messages, 0.1).await, "still cached");
        agent.state().set(LAST_CALL_KEY, minutes_ago(6));
        assert!(pass(&prune, &agent, &mut messages, 0.1).await);
        assert!(pruned(&messages));
        assert!(
            !pass(&prune, &agent, &mut messages, 0.1).await,
            "the mark restarts the clock"
        );
    }

    #[tokio::test]
    async fn a_one_hour_cache_is_still_warm_after_five_idle_minutes() {
        let (prune, agent) = (CachePrune::new(None, CACHE_TTL_1H), agent());
        let mut messages = prunable();
        assert!(!pass(&prune, &agent, &mut messages, 0.1).await);
        agent.state().set(LAST_CALL_KEY, minutes_ago(30));
        assert!(!pass(&prune, &agent, &mut messages, 0.1).await, "still cached");
        agent.state().set(LAST_CALL_KEY, minutes_ago(61));
        assert!(pass(&prune, &agent, &mut messages, 0.1).await);
    }

    #[tokio::test]
    async fn the_idle_clock_survives_a_restart() {
        let agent = agent();
        let mut messages = prunable();
        assert!(!pass(&CachePrune::new(None, CACHE_TTL), &agent, &mut messages, 0.1).await);
        let at = agent.state().get(LAST_CALL_KEY).unwrap().as_u64().unwrap();
        agent.state().set(LAST_CALL_KEY, at - 6 * 60 * 1_000);
        // A new process: no record of what was sent, only the persisted time.
        assert!(pass(&CachePrune::new(None, CACHE_TTL), &agent, &mut messages, 0.1).await);
    }

    /// Six more large results age the second turn's ones out of the kept window.
    fn age_out(messages: &mut Vec<Message>) {
        for i in 10..10 + KEEP_RECENT_RESULTS {
            let id = format!("t{i}");
            messages.push(assistant(vec![tool_call(&id)]));
            messages.push(result(&id, "y".repeat(6_000)));
        }
    }

    #[tokio::test]
    async fn prunes_once_reading_what_it_would_remove_costs_more_than_a_rewrite() {
        let (prune, agent) = (CachePrune::new(None, CACHE_TTL), agent());
        let mut messages = prunable();
        assert!(!pass(&prune, &agent, &mut messages, 0.1).await);
        age_out(&mut messages);
        let share = share(&messages, true);
        assert!(share > 0.3, "{share}");
        // Reading the removable part costs 0.1 per call and the rewrite of the rest 1.15, to
        // be matched PRUNE_PATIENCE times over: tens of calls for a share of a third.
        let calls = passes_until_pruned(&prune, &agent, &mut messages, 100).await;
        let expected = PRUNE_PATIENCE * 1.15 * (1.0 - share) / (0.1 * share);
        let calls = calls.expect("pruned in the end") as f64;
        assert!(
            (calls - expected).abs() <= 1.0,
            "{calls} calls, expected {expected}"
        );
        assert!(pruned(&messages));
        assert!(removable_chars(&messages, true).0 * 100 < removable_chars(&messages, true).1);
        assert_eq!(
            passes_until_pruned(&prune, &agent, &mut messages, 100).await,
            None,
            "nothing left to remove"
        );
    }

    #[tokio::test]
    async fn a_one_hour_cache_waits_longer_for_a_prune_to_pay() {
        let wait = |ttl| async move {
            let (prune, agent) = (CachePrune::new(None, ttl), agent());
            let mut messages = prunable();
            assert!(!pass(&prune, &agent, &mut messages, 0.1).await);
            age_out(&mut messages);
            passes_until_pruned(&prune, &agent, &mut messages, 100)
                .await
                .unwrap()
        };
        let (short, long) = (wait(CACHE_TTL).await, wait(CACHE_TTL_1H).await);
        // Writes cost 2 instead of 1.25, so the rewrite is 1.9 instead of 1.15 reads' worth.
        assert!(
            long > short && long as f64 / short as f64 > 1.4,
            "{short} vs {long}"
        );
    }

    #[tokio::test]
    async fn any_rewrite_starts_the_meter_over() {
        let full = {
            let (prune, agent) = (CachePrune::new(None, CACHE_TTL), agent());
            let mut messages = prunable();
            assert!(!pass(&prune, &agent, &mut messages, 0.1).await);
            age_out(&mut messages);
            passes_until_pruned(&prune, &agent, &mut messages, 100)
                .await
                .unwrap()
        };
        assert!(full > 3, "{full}");

        // Nearly paid, then a cold cache prunes for free: what was read so far no longer counts.
        let (prune, agent) = (CachePrune::new(None, CACHE_TTL), agent());
        let mut messages = prunable();
        assert!(!pass(&prune, &agent, &mut messages, 0.1).await);
        age_out(&mut messages);
        assert!(passes_until_pruned(&prune, &agent, &mut messages, full - 1)
            .await
            .is_none());
        agent.state().set(LAST_CALL_KEY, minutes_ago(6));
        assert!(pass(&prune, &agent, &mut messages, 0.1).await);
        age_out(&mut messages);
        let again = passes_until_pruned(&prune, &agent, &mut messages, 100)
            .await
            .unwrap();
        assert!(
            again * 2 >= full,
            "{again} calls after a rewrite, {full} on a fresh meter"
        );
    }

    /// A coordinator's turns: a child's long report, then a long reply of its own.
    fn coordinator(turns: usize) -> Vec<Message> {
        let mut messages = Vec::new();
        for t in 0..turns {
            let report = format!(
                "<message from=\"parser (a2)\" kind=\"reply\">\n{}\n</message>",
                "r".repeat(6_000)
            );
            messages.push(Message::user(report).with_tracking_id(format!("u{t}")));
            messages.push(Message::assistant("a".repeat(6_000)));
        }
        messages
    }

    #[tokio::test]
    async fn old_mail_and_replies_age_only_when_the_gate_prunes() {
        let (prune, agent) = (CachePrune::new(None, CACHE_TTL), agent());
        let keep = crate::aging::KEEP_TURNS;
        let mut messages = coordinator(keep);
        assert!(
            !pass(&prune, &agent, &mut messages, 0.1).await,
            "every turn is recent"
        );
        // Old turns past a few: some to age, but not yet worth a rewrite.
        messages.splice(0..0, coordinator(3));
        prune
            .mark()
            .apply(&mut ctx(&mut messages, &agent, 0.1))
            .await
            .unwrap();
        assert!(share(&messages, true) < 0.4);
        assert!(
            !pass(&prune, &agent, &mut messages, 0.1).await,
            "warm, not worth a rewrite"
        );
        assert_eq!(
            removable_chars(&messages, false).0,
            0,
            "without a stash nothing ages"
        );
        // As many old turns as recent ones: about half the history.
        messages.splice(0..0, coordinator(keep));
        prune
            .mark()
            .apply(&mut ctx(&mut messages, &agent, 0.1))
            .await
            .unwrap();
        assert!(share(&messages, true) > 0.4);
        assert!(passes_until_pruned(&prune, &agent, &mut messages, 100)
            .await
            .is_some());
        let aged = messages
            .iter()
            .take_while(|m| m.text().contains("[Stashed:"))
            .count();
        assert_eq!(aged, 2 * (keep + 3), "every old turn's report and reply aged");
        assert!(share(&messages, true) < 0.01);
    }

    #[tokio::test]
    async fn a_new_system_prompt_or_tool_set_drops_all_thinking() {
        let (prune, agent) = (CachePrune::new(None, CACHE_TTL), agent());
        let mut messages = reasoned_loop(&[100; 3]);
        agent.state().set(LAST_CALL_KEY, minutes_ago(6));
        assert!(pass(&prune, &agent, &mut messages, 0.1).await);
        assert_eq!(reasoned(&messages), vec![5]);
        assert!(
            !pass(&prune, &agent, &mut messages, 0.1).await,
            "warm and unchanged"
        );
        assert_eq!(reasoned(&messages), vec![5]);

        // A lone `cd` rewrites the system prompt.
        agent.set_system_prompt(Some(SystemPrompt::Text("moved".into())));
        let mut ctx = ctx(&mut messages, &agent, 0.1);
        assert_eq!(prune.reason(&ctx), Some(PruneReason::Prompt));
        assert!(prune.apply(&mut ctx).await.unwrap());
        prune.mark().apply(&mut ctx).await.unwrap();
        assert_eq!(reasoned(&messages), Vec::<usize>::new());

        // A `find_tools` enable adds a tool.
        messages.push(assistant(vec![thinking("step 3"), tool_call("t3")]));
        messages.push(tool_result("t3"));
        assert!(!pass(&prune, &agent, &mut messages, 0.1).await, "warm again");
        assert_eq!(reasoned(&messages), vec![7]);
        let tool = FunctionTool::builder("enabled")
            .description("x")
            .handler(|_input, _ctx: ToolContext| async move { Result::Ok("ok") });
        let tool: ToolRef = Arc::new(tool);
        agent
            .with_tool_registry(|registry| registry.add_or_replace(vec![tool]))
            .unwrap();
        assert!(pass(&prune, &agent, &mut messages, 0.1).await);
        assert_eq!(reasoned(&messages), Vec::<usize>::new());
    }

    #[tokio::test]
    async fn prunes_when_history_was_edited_since_the_last_call() {
        let (prune, agent) = (CachePrune::new(None, CACHE_TTL), agent());
        let mut messages = prunable();
        assert!(!pass(&prune, &agent, &mut messages, 0.1).await);
        // The `context` tool offloaded a result between two calls: the prefix is gone already.
        messages[6] = tool_result("t2");
        assert!(pass(&prune, &agent, &mut messages, 0.1).await);
        assert!(pruned(&messages));
    }

    /// One tool loop of `results` (characters each), every step reasoned; the last step's result
    /// is still to come, so its reasoning is the latest.
    fn reasoned_loop(results: &[usize]) -> Vec<Message> {
        let mut messages = vec![Message::user("task")];
        for (i, chars) in results.iter().enumerate() {
            let id = format!("t{i}");
            messages.push(assistant(vec![thinking(&format!("step {i}")), tool_call(&id)]));
            messages.push(result(&id, "x".repeat(*chars)));
        }
        messages
    }

    /// Indexes of the messages that still carry reasoning.
    fn reasoned(messages: &[Message]) -> Vec<usize> {
        (0..messages.len())
            .filter(|&i| has_reasoning(&messages[i]))
            .collect()
    }

    /// The gate's pass on the first call after the cache expired.
    async fn cold_pass(messages: &mut Vec<Message>) -> bool {
        let (prune, agent) = (CachePrune::new(None, CACHE_TTL), agent());
        agent.state().set(LAST_CALL_KEY, minutes_ago(6));
        pass(&prune, &agent, messages, 0.1).await
    }

    #[tokio::test]
    async fn an_untouched_loop_keeps_the_latest_thinking() {
        let mut messages = reasoned_loop(&[100; 9]);
        assert!(cold_pass(&mut messages).await);
        assert_eq!(reasoned(&messages), vec![17]);
        assert_eq!(messages[17].content[0], thinking("step 8"));
    }

    #[tokio::test]
    async fn a_trimmed_result_leaves_no_thinking_after_it() {
        let mut steps = vec![100, 5_000];
        steps.extend([100; KEEP_RECENT_RESULTS + 2]);
        let mut messages = reasoned_loop(&steps);
        assert!(cold_pass(&mut messages).await);
        assert!(
            result_text(&messages[4]).len() < TRIM_ABOVE_CHARS,
            "t1 was trimmed"
        );
        assert_eq!(reasoned(&messages), Vec::<usize>::new(), "the latest went too");
        assert_eq!(messages[messages.len() - 2].content, vec![tool_call("t9")]);
    }

    #[tokio::test]
    async fn an_edit_since_the_last_call_drops_the_thinking_after_it() {
        let (prune, agent) = (CachePrune::new(None, CACHE_TTL), agent());
        let mut messages = reasoned_loop(&[100; 3]);
        messages.push(assistant(vec![ContentBlock::text("done")]));
        messages.push(Message::user("next task"));
        messages.extend(reasoned_loop(&[100; 3]).into_iter().skip(1));
        agent.state().set(LAST_CALL_KEY, minutes_ago(6));
        assert!(
            pass(&prune, &agent, &mut messages, 0.1).await,
            "earlier turns' thinking"
        );
        assert_eq!(reasoned(&messages), vec![13]);
        // The `context` tool offloads the last result but one; the latest thinking is after it.
        messages[12] = tool_result("t1");
        assert!(pass(&prune, &agent, &mut messages, 0.1).await);
        assert_eq!(reasoned(&messages), Vec::<usize>::new());
    }

    /// A real PNG of `width` x `height` noise-free pixels, a few KB at most.
    fn png(width: u32, height: u32) -> ImageBlock {
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x * 7) as u8, (y * 13) as u8, (x ^ y) as u8])
        }));
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        ImageBlock {
            format: strands_agents::types::ImageFormat::Png,
            source: ImageSource::Bytes(bytes.into_inner()),
        }
    }

    /// A reasoned tool loop of `steps` `read` calls, each returning an image: step `i` reads
    /// `/shots/i.png`, a 40x30 image.
    fn image_loop(steps: usize) -> Vec<Message> {
        let mut messages = vec![Message::user("task")];
        for i in 0..steps {
            let id = format!("t{i}");
            let call = ToolUseBlock::new("read", &id, json!({ "path": format!("/shots/{i}.png") }));
            messages.push(assistant(vec![
                thinking(&format!("step {i}")),
                ContentBlock::ToolUse(call),
            ]));
            messages.push(Message::new(
                Role::User,
                vec![ContentBlock::ToolResult(ToolResultBlock::success(
                    &id,
                    vec![ToolResultContent::Image(png(40, 30))],
                ))],
            ));
        }
        messages
    }

    /// Whether message `i`'s result still holds its image.
    fn has_image(messages: &[Message], i: usize) -> bool {
        messages[i]
            .tool_results()
            .flat_map(|r| &r.content)
            .any(|item| matches!(item, ToolResultContent::Image(_)))
    }

    #[tokio::test]
    async fn a_cold_prune_stubs_images_older_than_the_last_steps_and_keeps_the_latest() {
        let mut messages = image_loop(8);
        let before = json(&messages).len();
        assert!(cold_pass(&mut messages).await);
        // Results at 2, 4, ..., 16; a result stays while fewer than KEEP_IMAGE_STEPS assistant
        // messages follow it, so the last four steps' images stay.
        for i in (2..=16).step_by(2) {
            assert_eq!(
                has_image(&messages, i),
                i > 16 - 2 * KEEP_IMAGE_STEPS,
                "message {i}"
            );
        }
        assert!(
            has_image(&messages, 16),
            "the latest step's image is never dropped"
        );
        let ToolResultContent::Text(stub) = &messages[2].tool_results().next().unwrap().content[0] else {
            panic!("a stub")
        };
        assert_eq!(
            stub,
            "[image omitted: /shots/0.png, 40x30; read it again if needed]"
        );
        assert!(json(&messages).len() < before, "the saved conversation shrinks");
    }

    #[tokio::test]
    async fn images_stay_on_a_warm_call() {
        let (prune, agent) = (CachePrune::new(None, CACHE_TTL), agent());
        // A long first message keeps what pruning would save a small part of the history.
        let mut messages = image_loop(8);
        messages[0] = Message::user("task ".repeat(20_000));
        let before = messages.clone();
        assert!(!pass(&prune, &agent, &mut messages, 0.1).await);
        assert_eq!(messages, before);
    }

    #[tokio::test]
    async fn an_image_stub_drops_the_thinking_after_it() {
        let mut messages = image_loop(8);
        assert!(cold_pass(&mut messages).await);
        assert_eq!(reasoned(&messages), Vec::<usize>::new(), "the latest went too");
    }

    #[tokio::test]
    async fn an_mcp_image_stub_names_the_tool_and_mail_images_stay() {
        let mut messages = image_loop(8);
        let call = ToolUseBlock::new("browser_screenshot", "t0", json!({}));
        messages[1].content = vec![ContentBlock::ToolUse(call)];
        let wait = ToolUseBlock::new("wait", "t1", json!({}));
        messages[3].content = vec![ContentBlock::ToolUse(wait)];
        assert!(cold_pass(&mut messages).await);
        let ToolResultContent::Text(stub) = &messages[2].tool_results().next().unwrap().content[0] else {
            panic!("a stub")
        };
        assert_eq!(
            stub,
            "[image omitted: browser_screenshot result, 40x30; call the tool again if needed]"
        );
        assert!(has_image(&messages, 4), "mail attachments arrive in wait results");
    }

    #[tokio::test]
    async fn an_edit_after_the_latest_thinking_keeps_it() {
        let (prune, agent) = (CachePrune::new(None, CACHE_TTL), agent());
        let mut messages = reasoned_loop(&[100; 3]);
        agent.state().set(LAST_CALL_KEY, minutes_ago(6));
        assert!(pass(&prune, &agent, &mut messages, 0.1).await);
        assert_eq!(reasoned(&messages), vec![5]);
        messages[6] = tool_result("edited");
        assert!(!pass(&prune, &agent, &mut messages, 0.1).await, "nothing to drop");
        assert_eq!(reasoned(&messages), vec![5]);
    }

    #[test]
    fn a_restart_notice_does_not_start_a_new_turn() {
        let mut messages = vec![
            Message::user("task"),
            assistant(vec![thinking("mid-turn"), ContentBlock::text("working")]),
            Message::user(format!("{NOTICE_PREFIX} and your conversation restored.]")),
        ];
        assert!(!drop_old_reasoning(&mut messages));
        assert_eq!(reasoning_count(&messages), 1);
    }

    #[tokio::test]
    async fn prunes_on_overflow() {
        let (prune, agent) = (CachePrune::new(None, CACHE_TTL), agent());
        let mut messages = prunable();
        let mut ctx = ctx(&mut messages, &agent, 0.1);
        ctx.overflow = true;
        assert_eq!(prune.reason(&ctx), Some(PruneReason::Overflow));
        assert!(prune.apply(&mut ctx).await.unwrap());
    }

    /// `inner`'s replies, each opened by a signed reasoning block `step N` (the mock replays none).
    struct Thinking {
        inner: Arc<MockMessageModel>,
        step: AtomicUsize,
    }

    impl Model for Thinking {
        fn model_id(&self) -> Option<String> {
            self.inner.model_id()
        }

        fn stream<'a>(&'a self, messages: &'a [Message], options: &'a StreamOptions) -> ModelEventStream<'a> {
            let step = self.step.fetch_add(1, Ordering::SeqCst);
            let reasoning = vec![
                ModelStreamEvent::ContentBlockStart { start: None },
                ModelStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::ReasoningContent {
                        text: Some(format!("step {step}")),
                        signature: Some("sig".into()),
                        redacted_content: None,
                    },
                },
                ModelStreamEvent::ContentBlockStop {},
            ];
            let mut reasoning = Some(reasoning);
            // After the message start, which comes first.
            Box::pin(self.inner.stream(messages, options).flat_map(move |event| {
                let after = reasoning.take().unwrap_or_default().into_iter().map(Ok);
                futures::stream::iter(std::iter::once(event).chain(after))
            }))
        }
    }

    /// An agent behind the swarm's gate that runs a tool loop of `steps` `shell` calls, each
    /// reasoned, then ends its turn. The call answering `expire_at` finds the cache expired.
    async fn tool_loop(steps: usize, expire_at: Option<&'static str>) -> Vec<Vec<Message>> {
        let mut responses: Vec<MockResponse> = (0..steps)
            .map(|i| MockResponse::message(assistant(vec![tool_call(&format!("t{i}"))])))
            .collect();
        responses.push(MockResponse::message(Message::assistant("done")));
        let model = Arc::new(Thinking {
            inner: MockMessageModel::shared(responses),
            step: AtomicUsize::new(0),
        });
        let prune = CachePrune::new(None, CACHE_TTL);
        let config = ContextManagerConfig {
            strategies: Some(vec![Arc::new(prune.clone()), prune.mark()]),
            ..Default::default()
        };
        let shell = FunctionTool::builder("shell").description("x").handler(
            move |_input, ctx: ToolContext| async move {
                if Some(ctx.tool_use.tool_use_id.as_str()) == expire_at {
                    ctx.agent.state().set(LAST_CALL_KEY, minutes_ago(6));
                }
                Result::Ok("ok")
            },
        );
        let agent = Agent::builder()
            .model(Arc::clone(&model))
            .tool(shell)
            .context_manager(ContextManagerOption::Config(config))
            .no_sandbox()
            .build()
            .unwrap();
        agent.invoke("task").await.unwrap();
        model
            .inner
            .calls()
            .into_iter()
            .map(|(messages, _)| messages)
            .collect()
    }

    fn reasoning_texts(messages: &[Message]) -> Vec<(usize, String)> {
        let mut texts = Vec::new();
        for (i, message) in messages.iter().enumerate() {
            for block in &message.content {
                if let ContentBlock::Reasoning(reasoning) = block {
                    texts.push((i, reasoning.text.clone().unwrap_or_default()));
                }
            }
        }
        texts
    }

    #[tokio::test]
    async fn a_warm_tool_loop_sends_every_step_s_reasoning_unchanged() {
        let calls = tool_loop(4, None).await;
        assert_eq!(calls.len(), 5);
        for (k, messages) in calls.iter().enumerate() {
            let expected: Vec<(usize, String)> = (0..k).map(|i| (1 + 2 * i, format!("step {i}"))).collect();
            assert_eq!(reasoning_texts(messages), expected, "call {k}");
        }
    }

    #[tokio::test]
    async fn a_cold_cache_mid_loop_keeps_only_the_latest_step_s_reasoning() {
        let calls = tool_loop(4, Some("t2")).await;
        let texts: Vec<Vec<(usize, String)>> = calls.iter().map(|m| reasoning_texts(m)).collect();
        assert_eq!(texts[2].len(), 2, "warm until then: {texts:?}");
        assert_eq!(
            texts[3],
            vec![(5, "step 2".to_owned())],
            "the call answering t2 keeps t2's reasoning only"
        );
        assert_eq!(calls[3][1].content, vec![tool_call("t0")]);
        assert_eq!(
            texts[4],
            vec![(5, "step 2".to_owned()), (7, "step 3".to_owned())],
            "then warm again: only appended to"
        );
    }
}
