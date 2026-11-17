//! What to drop from an agent's context before each model call, beyond the SDK's pipeline.
//!
//! Reasoning: models that keep prior turns' thinking (Claude Opus 4.5+) bill it as input on every
//! later call, and its signatures alone are most of a long agent's reasoning bytes. Anthropic
//! requires thinking blocks within the turn in progress (a tool-use loop is one turn) and allows
//! omitting earlier turns' thinking, as long as no dropped block sits between two kept ones.
//!
//! Old tool output: a `shell` or `read` result is mostly read once, then re-sent on every call.
//! Older ones are cut to their head and tail; the full text stays in the stash under the
//! `retrieve_context` reference the preview ends with.
//!
//! Both edit history, and the provider's prompt cache only matches an unchanged prefix: one edit
//! deep in a 150k-token history turns the next call's cache reads (0.1x) into cache writes
//! (1.25x) for everything after it. Run on every turn, they raised input cost per call by ~80% on
//! a live run. So [`CachePrune`] runs them only when the prefix is lost anyway, and then both at
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
use strands_agents::types::{ContentBlock, Message, Role, ToolResultBlock, ToolResultContent};
use strands_agents::{Agent, Result};

use parking_lot::Mutex;

use crate::compaction::CompactAt;
use crate::resume::NOTICE_PREFIX;

/// Stands in for an assistant message that held nothing but reasoning.
const REASONING_DROPPED: &str = "[reasoning dropped]";

/// Where the current turn starts: the last user message that carries no tool results.
/// A restart notice is not a new turn: the turn it interrupted goes on.
fn turn_start(messages: &[Message]) -> usize {
    messages
        .iter()
        .rposition(|m| m.role == Role::User && m.tool_results().next().is_none() && !is_restart_notice(m))
        .unwrap_or(0)
}

fn is_restart_notice(message: &Message) -> bool {
    matches!(message.content.first(), Some(ContentBlock::Text(text)) if text.starts_with(NOTICE_PREFIX))
}

/// Removes reasoning from the assistant messages before the current turn; `true` when any went.
/// A message left empty gets [`REASONING_DROPPED`] so the history stays valid.
fn drop_old_reasoning(messages: &mut [Message]) -> bool {
    let start = turn_start(messages);
    let mut acted = false;
    for message in &mut messages[..start] {
        let is_reasoning = |block: &ContentBlock| matches!(block, ContentBlock::Reasoning(_));
        if message.role != Role::Assistant || !message.content.iter().any(is_reasoning) {
            continue;
        }
        message.content.retain(|block| !is_reasoning(block));
        if message.content.is_empty() {
            message.content.push(ContentBlock::text(REASONING_DROPPED));
        }
        acted = true;
    }
    acted
}

/// Tools whose older results are trimmed.
const TRIMMED_TOOLS: [&str; 2] = ["shell", "read"];
/// The latest this many [`TRIMMED_TOOLS`] results stay whole.
const KEEP_RECENT_RESULTS: usize = 6;
/// Older results longer than this (in characters) are trimmed.
const TRIM_ABOVE_CHARS: usize = 2_000;
/// What a trimmed result keeps, in tokens (about 800 characters, head and tail).
const TRIM_PREVIEW_TOKENS: usize = 200;

/// Trims [`TRIMMED_TOOLS`] results longer than [`TRIM_ABOVE_CHARS`] once
/// [`KEEP_RECENT_RESULTS`] newer ones exist, whenever it runs: gated by [`CachePrune`].
fn trim_old_results() -> OffloadStrategy {
    let truncate = Offload::truncate(
        OffloadTarget::tools(TRIMMED_TOOLS),
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

/// How long the provider keeps a cached prompt prefix by default (Anthropic and Bedrock: 5
/// minutes, refreshed on every hit).
pub const CACHE_TTL: Duration = Duration::from_secs(5 * 60);
/// The cache lifetime with `--cache-ttl 1h`.
pub const CACHE_TTL_1H: Duration = Duration::from_secs(60 * 60);
/// Prune for its own sake once that would remove this share of the history (in characters).
/// Each prune rewrites the cached conversation (1.25× instead of 0.1× per token) to save 0.1×
/// of what it removes on every later call. At 15% small agents with chatty shell output pruned
/// every 4-6 calls, which came to 15-40% of their cost; a simulation over result sizes and
/// rates puts the cheapest share at 0.4-0.5.
pub const PRUNE_SHARE: f64 = 0.4;
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
    /// The last call was longer ago than the cache lives ([`CACHE_TTL`] by default), so the cache
    /// has expired.
    Idle,
    /// Pruning would remove at least [`PRUNE_SHARE`] of the history, worth one cache rewrite.
    Large,
}

/// Drops earlier turns' reasoning and trims old results, together and only for a
/// [`PruneReason`]; on any other call history reaches the model byte for byte as last time. Goes
/// first in the pipeline, with [`CachePrune::mark`] last.
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
        };
        Self { gate: Arc::new(gate) }
    }

    /// The pipeline's last strategy: notes what this call sends and when, for the next call's
    /// [`PruneReason::Edited`] and [`PruneReason::Idle`]. It never edits the context.
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
        let idle = last_call(ctx.agent)
            .is_some_and(|at| SystemTime::now().duration_since(at).unwrap_or_default() > self.gate.ttl);
        if idle {
            return Some(PruneReason::Idle);
        }
        (removable_share(ctx.messages) >= PRUNE_SHARE).then_some(PruneReason::Large)
    }

    fn edited(&self, messages: &[Message]) -> bool {
        match &*self.gate.sent.lock() {
            None => false,
            Some(sent) => {
                sent.len() > messages.len() || sent.iter().zip(messages).any(|(h, m)| *h != hash(m))
            }
        }
    }
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
        let dropped = drop_old_reasoning(ctx.messages);
        let trimmed = self.gate.trim.apply(ctx).await?;
        if dropped || trimmed {
            tracing::debug!(?reason, dropped, trimmed, "pruned old reasoning and results");
        }
        Ok(dropped || trimmed)
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

fn last_call(agent: &Agent) -> Option<SystemTime> {
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

/// The share of the history (JSON characters) that pruning would remove now.
fn removable_share(messages: &[Message]) -> f64 {
    let total: usize = messages.iter().map(json_len).sum();
    if total == 0 {
        return 0.0;
    }
    let reasoning: usize = messages[..turn_start(messages)]
        .iter()
        .filter(|m| m.role == Role::Assistant)
        .flat_map(|m| &m.content)
        .filter(|block| matches!(block, ContentBlock::Reasoning(_)))
        .map(json_len)
        .sum();
    let preview = TRIM_PREVIEW_TOKENS * 4;
    let results: usize = old_trimmed_results(messages, KEEP_RECENT_RESULTS)
        .map(result_chars)
        .filter(|chars| *chars > TRIM_ABOVE_CHARS)
        .map(|chars| chars.saturating_sub(preview))
        .sum();
    (reasoning + results) as f64 / total as f64
}

fn json_len<T: serde::Serialize>(value: &T) -> usize {
    serde_json::to_string(value).map_or(0, |json| json.len())
}

/// [`TRIMMED_TOOLS`] results with at least `keep` newer ones.
fn old_trimmed_results(messages: &[Message], keep: usize) -> impl Iterator<Item = &ToolResultBlock> {
    let names: HashMap<&str, &str> = messages
        .iter()
        .flat_map(|m| m.tool_uses())
        .map(|u| (u.tool_use_id.as_str(), u.name.as_str()))
        .collect();
    let trimmed = |r: &&ToolResultBlock| {
        names
            .get(r.tool_use_id.as_str())
            .is_some_and(|name| TRIMMED_TOOLS.contains(name))
    };
    let matching: Vec<&Message> = messages
        .iter()
        .filter(|m| m.role == Role::User && m.tool_results().any(|r| trimmed(&r)))
        .collect();
    let old = matching.len().saturating_sub(keep);
    let results: Vec<&ToolResultBlock> = matching[..old]
        .iter()
        .flat_map(|m| m.tool_results().filter(trimmed))
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
    use serde_json::json;
    use strands_agents::testing::{MockMessageModel, MockResponse};
    use strands_agents::types::{ReasoningBlock, ToolResultBlock, ToolResultContent, ToolUseBlock};
    use strands_agents::InMemoryStorage;

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
    fn drops_earlier_turns_reasoning_and_keeps_the_whole_tool_loop_in_progress() {
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

        assert_eq!(reasoning_count(&messages[..4]), 0);
        assert_eq!(messages[1].content, vec![tool_call("t1")]);
        assert_eq!(messages[3].content, vec![ContentBlock::text("done")]);
        assert_eq!(
            reasoning_count(&messages[4..]),
            2,
            "the turn in progress is untouched"
        );
        assert!(
            !drop_old_reasoning(&mut messages),
            "a second pass changes nothing"
        );
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
    async fn trims_older_large_shell_and_read_results_to_a_preview_with_a_reference() {
        let mut results = vec![("read", 5_000), ("write", 5_000), ("shell", 100)];
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
        assert_eq!(text(1).len(), 5_000, "other tools are left alone");
        assert_eq!(text(2).len(), 100, "small results are left alone");
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
    /// whose recent results are large enough that pruning would save under [`PRUNE_SHARE`].
    fn prunable() -> Vec<Message> {
        let mut messages = vec![Message::user("first task")];
        messages.push(assistant(vec![thinking("old"), tool_call("t0")]));
        messages.push(result("t0", "x".repeat(3_000)));
        messages.push(assistant(vec![ContentBlock::text("done")]));
        let mut second = history(&[("shell", 6_000); KEEP_RECENT_RESULTS]);
        second[0] = Message::user("second task");
        messages.extend(second);
        assert!(
            removable_share(&messages) < PRUNE_SHARE,
            "{}",
            removable_share(&messages)
        );
        messages
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
        let compact = Arc::new(CompactAt::new(1_000));
        let prune = CachePrune::new(Some(compact.clone()), CACHE_TTL);
        let mut messages = prunable();
        let mut ctx = ctx(&mut messages, &agent, 0.5);
        assert_eq!(prune.reason(&ctx), Some(PruneReason::Compaction));
        assert!(prune.apply(&mut ctx).await.unwrap());
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

    #[tokio::test]
    async fn prunes_once_it_would_remove_15_percent_of_the_history() {
        let (prune, agent) = (CachePrune::new(None, CACHE_TTL), agent());
        let mut messages = prunable();
        assert!(!pass(&prune, &agent, &mut messages, 0.1).await);
        // Six more large results age the second turn's ones out of the kept window.
        for i in 10..10 + KEEP_RECENT_RESULTS {
            let id = format!("t{i}");
            messages.push(assistant(vec![tool_call(&id)]));
            messages.push(result(&id, "y".repeat(6_000)));
        }
        assert!(removable_share(&messages) >= PRUNE_SHARE);
        assert!(pass(&prune, &agent, &mut messages, 0.1).await);
        assert!(pruned(&messages));
        assert!(removable_share(&messages) < 0.01);
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
}
