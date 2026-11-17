//! When to summarize an agent's older turns: by default at 85% of its model's window (850k
//! tokens of a 1M window), or at a fixed token count (`--compact-at`, `spawn_agent`'s
//! `compact_at`) when that is smaller.

use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::dashboard::{ContextControl, SUMMARY_KEEPS};
use async_trait::async_trait;
use futures::StreamExt;
use strands_agents::context_manager::strategies::{
    flatten_messages_to_content, format_summarized, CHARS_PER_TOKEN, SUMMARIZATION_SYSTEM_PROMPT,
    SUMMARIZED_PREFIX,
};
use strands_agents::context_manager::{ContextState, ContextStrategy, AUTO_SUMMARIZE_UTILIZATION};
use strands_agents::models::{
    estimate_tokens_heuristic, AggregatedItem, CountTokensOptions, StreamOptions,
    DEFAULT_CONTEXT_WINDOW_LIMIT,
};
use strands_agents::types::{ContentBlock, Message, Role, SystemPrompt};
use strands_agents::{Error, Model, Result};

/// The input size (tokens) at which older turns are summarized: the context manager's own 85%
/// mark of `window`, or `compact_at` when given and smaller.
pub fn compaction_point(window: u64, compact_at: Option<u64>) -> u64 {
    let auto = (window as f64 * AUTO_SUMMARIZE_UTILIZATION) as u64;
    compact_at.map_or(auto, |at| auto.min(at)).max(1)
}

/// The utilization (share of `window`) at which to summarize:
/// `min(AUTO_SUMMARIZE_UTILIZATION, compact_at / window)`.
pub fn compaction_utilization(window: u64, compact_at: Option<u64>) -> f64 {
    compaction_point(window, compact_at) as f64 / window.max(1) as f64
}

/// The window the context manager assumes for `model`: its own, else the same default.
pub fn window_of(model: &dyn Model) -> u64 {
    model
        .context_window_limit()
        .filter(|window| *window > 0)
        .unwrap_or(DEFAULT_CONTEXT_WINDOW_LIMIT)
}

/// Most summarize passes in one model call: a second runs only when the first left the context
/// above the compaction point (a summary much longer than expected, say).
const MAX_PASSES: usize = 2;

/// After a compaction that changed nothing (the summarizer failed, or nothing could be cut), the
/// automatic retries wait until the input has grown by this share of what it was then…
const RETRY_GROWTH: f64 = 0.10;
/// …or this many model calls have passed. Overflow and `context` requests retry at once.
const RETRY_CALLS: u32 = 10;

/// Most of the summarizer's window one summarizer call's slice of the transcript may take,
/// leaving the rest for the prompt, the summary so far and the output. At 850k of a 1M window the
/// ~425k tokens to summarize go in one call; a 200k-window summarizer gets 100k slices.
const SLICE_SHARE: f64 = 0.5;
/// Smallest slice, in estimated tokens.
const MIN_SLICE: u64 = 1_000;
/// How often a slice the summarizer rejects as too long is halved before compaction fails.
const MAX_HALVINGS: u32 = 4;

/// Why automatic compaction is holding off: the input it last failed at, and the calls left.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Backoff {
    tokens: u64,
    calls_left: u32,
}

/// What [`CompactAt`] reports when it summarized, or tried to and failed.
#[derive(Debug, Clone, PartialEq)]
pub enum Compaction {
    /// Older turns were summarized.
    Done {
        /// The call's projected input that set it off, before cache pruning trimmed any of it.
        before: u64,
        /// Estimated input tokens after.
        after: u64,
    },
    /// The summarizer failed; the context was left as it was.
    Failed {
        /// Projected input tokens, still to be sent.
        tokens: u64,
        /// The summarizer's error.
        error: String,
    },
}

/// Told each [`Compaction`], so a failure is seen rather than only logged.
pub type CompactedSink = Arc<dyn Fn(Compaction) + Send + Sync>;

/// Summarizes an agent's older turns once the projected input reaches the compaction point of
/// its current model (so a `/model` switch moves the point with the window), on overflow, or
/// when the agent asked through its `context` tool.
///
/// It summarizes down to half the point, choosing how many leading messages go into the
/// summary from their token counts, so the next calls have room to grow before compacting again
/// (a fixed "all but the last few" or a share would leave a big tail above the point). The
/// first message (the task) stays verbatim; the summary goes after it in the same message and is
/// folded into the next summary. On request it keeps only the latest [`SUMMARY_KEEPS`] messages.
pub struct CompactAt {
    compact_at: Option<u64>,
    control: Option<Arc<ContextControl>>,
    sink: Option<CompactedSink>,
    /// Set by [`CompactAt::arm`] to `Some(requested)`: run on this pipeline pass even if an
    /// earlier strategy brought the context under the point (as long as it is still above the
    /// target).
    armed: Mutex<Option<bool>>,
    /// Set when a compaction changed nothing; cleared by the next one that does.
    backoff: Mutex<Option<Backoff>>,
    /// Set by [`CompactAt::arm`]: real tokens per estimated token of the messages, from the
    /// pass's projection before any strategy edited them. See [`CompactAt::tokens`].
    scale: Mutex<Option<f64>>,
    /// Set by [`CompactAt::arm`]: the pass's projected input before any strategy edited it, so a
    /// report says the size that set compaction off, not what was left after pruning.
    reached: Mutex<Option<u64>>,
}

impl CompactAt {
    /// Compact at `compact_at` tokens, or at 85% of the window when `None` or smaller.
    pub fn new(compact_at: Option<u64>) -> Self {
        Self {
            compact_at,
            control: None,
            sink: None,
            armed: Mutex::new(None),
            backoff: Mutex::new(None),
            scale: Mutex::new(None),
            reached: Mutex::new(None),
        }
    }

    /// Also summarize once after each request made through `control`.
    pub fn on_request(mut self, control: Arc<ContextControl>) -> Self {
        self.control = Some(control);
        self
    }

    /// Report each compaction, and each failed one, to `sink`.
    pub fn on_compacted(mut self, sink: Option<CompactedSink>) -> Self {
        self.sink = sink;
        self
    }

    /// Whether this pipeline pass will compact; if so, it will even if an earlier strategy
    /// brings utilization back under the point. Takes a pending summarize request.
    pub fn arm(&self, ctx: &ContextState<'_>) -> bool {
        let window = window_of(ctx.agent.model().as_ref());
        let estimated = heuristic(ctx.messages).max(1);
        *self.scale.lock() = Some(input_tokens(ctx, window) as f64 / estimated as f64);
        let due = self.due(ctx);
        if let Some(requested) = due {
            *self.reached.lock() = Some(input_tokens(ctx, window));
            let mut armed = self.armed.lock();
            *armed = Some(requested || *armed == Some(true));
        }
        due.is_some()
    }

    /// `Some(requested)` when this pass is to compact: on request, overflow, or at the point
    /// unless a recent failure is being backed off from.
    fn due(&self, ctx: &ContextState<'_>) -> Option<bool> {
        let requested = self.control.as_ref().is_some_and(|c| c.take_summarize_request());
        let window = window_of(ctx.agent.model().as_ref());
        let at_point = ctx.utilization >= compaction_utilization(window, self.compact_at)
            && !self.backing_off(input_tokens(ctx, window));
        (requested || ctx.overflow || at_point).then_some(requested)
    }

    fn backing_off(&self, tokens: u64) -> bool {
        self.backoff.lock().is_some_and(|backoff| {
            backoff.calls_left > 0 && (tokens as f64) < backoff.tokens as f64 * (1.0 + RETRY_GROWTH)
        })
    }

    /// One more model call towards the end of a backoff.
    fn tick(&self) {
        if let Some(backoff) = self.backoff.lock().as_mut() {
            backoff.calls_left = backoff.calls_left.saturating_sub(1);
        }
    }
}

#[async_trait]
impl ContextStrategy for CompactAt {
    fn name(&self) -> &str {
        "compact"
    }

    async fn apply(&self, ctx: &mut ContextState<'_>) -> Result<bool> {
        let compacted = self.compact(ctx).await;
        // After deciding, so `arm` and `apply` in one pass see the same backoff.
        self.tick();
        compacted
    }
}

impl CompactAt {
    fn report(&self, compaction: Compaction) {
        if let Some(sink) = &self.sink {
            sink(compaction);
        }
    }

    /// This pass's input in real tokens. The projection starts from the provider's count, but
    /// once a strategy edits the messages the SDK re-estimates them alone, by characters: no
    /// system prompt or tools, and code and JSON weigh more than four characters a token. That
    /// read 30-40% low (113k for a 186k-token input), so compaction stopped ~50k above its
    /// target. Scale the messages' estimate by the ratio seen before any edit instead.
    fn tokens(&self, ctx: &ContextState<'_>, window: u64) -> u64 {
        match self.scale.lock().take() {
            Some(scale) => (heuristic(ctx.messages) as f64 * scale).round() as u64,
            None => input_tokens(ctx, window),
        }
    }

    async fn compact(&self, ctx: &mut ContextState<'_>) -> Result<bool> {
        let armed = self.armed.lock().take();
        let due = self.due(ctx);
        if armed.is_none() && due.is_none() {
            return Ok(false);
        }
        let requested = armed == Some(true) || due == Some(true);
        let window = window_of(ctx.agent.model().as_ref());
        let point = compaction_point(window, self.compact_at);
        let before = self.tokens(ctx, window);
        let reached = self.reached.lock().take().map_or(before, |r| r.max(before));
        let target = if requested { 0 } else { point / 2 };
        // Armed, but an earlier strategy (pruning) already took it under the target.
        if !requested && !ctx.overflow && before <= target {
            return Ok(false);
        }
        let model = ctx.agent.aux_model();
        let mut tokens = before;
        let mut passes = 0;
        let mut failure = None;
        while passes < MAX_PASSES && (passes == 0 || tokens >= point) {
            match summarize_down_to(ctx.messages, tokens, target, &model).await {
                Ok(Some(after)) => tokens = after,
                Ok(None) => break,
                // A shutdown or reload cut the summary short: not a failure, nothing to back off.
                Err(Error::Cancelled) if passes == 0 => return Ok(false),
                Err(Error::Cancelled) => break,
                Err(error) => {
                    failure = Some(error.to_string());
                    break;
                }
            }
            passes += 1;
        }
        if let (0, Some(error)) = (passes, failure) {
            self.report(Compaction::Failed { tokens, error });
        }
        if passes == 0 {
            tracing::warn!(
                "tokens=<{before}>, retry_growth=<{RETRY_GROWTH}>, retry_calls=<{RETRY_CALLS}> | compaction changed nothing, backing off"
            );
            *self.backoff.lock() = Some(Backoff {
                tokens: before,
                calls_left: RETRY_CALLS,
            });
            return Ok(false);
        }
        *self.backoff.lock() = None;
        tracing::info!("before=<{before}>, after=<{tokens}>, passes=<{passes}> | compacted the context");
        self.report(Compaction::Done {
            before: reached,
            after: tokens,
        });
        Ok(true)
    }
}

/// The projected input of this pipeline pass, in tokens.
fn input_tokens(ctx: &ContextState<'_>, window: u64) -> u64 {
    (ctx.utilization * window as f64).round() as u64
}

/// One pass: summarize the leading messages (after the first) that bring `tokens` (the projected
/// input, system prompt and tools included) down to `target`, keeping at least the latest
/// [`SUMMARY_KEEPS`]. Returns the estimated input afterwards, `None` when nothing could be
/// summarized, or the summarizer's error; either way the context is then left as it was.
async fn summarize_down_to(
    messages: &mut Vec<Message>,
    tokens: u64,
    target: u64,
    model: &Arc<dyn Model>,
) -> Result<Option<u64>> {
    let Some(first) = messages.first().filter(|m| m.role == Role::User) else {
        return Ok(None);
    };
    let (task, earlier): (Vec<ContentBlock>, Vec<ContentBlock>) = first
        .content
        .iter()
        .cloned()
        .partition(|block| !is_summary(block));
    let sizes: Vec<u64> = messages
        .iter()
        .map(|m| heuristic(std::slice::from_ref(m)))
        .collect();
    // The projection counts the system prompt and tools too; scaling the per-message estimates
    // up to it overestimates what a message weighs, which errs towards summarizing more.
    let scale = tokens as f64 / sizes.iter().sum::<u64>().max(1) as f64;
    let excess = tokens.saturating_sub(target) as f64;
    let Some(last) = messages.len().checked_sub(SUMMARY_KEEPS) else {
        return Ok(None);
    };
    // `cut` is the first message kept after the summary; an assistant message, so the summary's
    // range ends with a user message and no tool call loses its result.
    let mut removed = heuristic_blocks(&earlier) as f64 * scale;
    let mut cut = None;
    for index in 2..=last {
        removed += sizes[index - 1] as f64 * scale;
        if messages[index].role == Role::Assistant && messages[index - 1].role == Role::User {
            cut = Some((index, removed));
            if removed >= excess {
                break;
            }
        }
    }
    let Some((cut, removed)) = cut else {
        return Ok(None);
    };
    let summary = summarize_in_slices(
        model,
        earlier,
        flatten_messages_to_content(&messages[1..cut]),
        scale,
    )
    .await
    .inspect_err(|error| {
        if !matches!(error, Error::Cancelled) {
            tracing::warn!("error=<{error}> | summarizing older turns failed");
        }
    })?;
    let text = format_summarized(
        &format!("{} messages", cut - 1),
        removed.round() as u64,
        Some(&summary),
    );
    let added = heuristic_blocks(std::slice::from_ref(&ContentBlock::Text(text.clone()))) as f64 * scale;
    let mut kept = task;
    kept.push(ContentBlock::Text(text));
    messages[0].content = kept;
    messages.drain(1..cut);
    Ok(Some((tokens as f64 - removed + added).max(0.0).round() as u64))
}

fn is_summary(block: &ContentBlock) -> bool {
    matches!(block, ContentBlock::Text(text) if text.starts_with(SUMMARIZED_PREFIX))
}

fn heuristic(messages: &[Message]) -> u64 {
    estimate_tokens_heuristic(messages, &CountTokensOptions::default())
}

fn heuristic_blocks(blocks: &[ContentBlock]) -> u64 {
    heuristic(&[Message::new(Role::User, blocks.to_vec())])
}

/// Summarizes `earlier` (the summary so far, if any) followed by `content` in calls that fit the
/// summarizer's window: each gets the summary so far and the next slice of `content`, of at most
/// [`SLICE_SHARE`] of that window, with `scale` real tokens per estimated one. A slice the
/// summarizer still rejects as too long is halved and sent again, up to [`MAX_HALVINGS`] times.
async fn summarize_in_slices(
    model: &Arc<dyn Model>,
    earlier: Vec<ContentBlock>,
    content: Vec<ContentBlock>,
    scale: f64,
) -> Result<String> {
    let window = window_of(model.as_ref()) as f64;
    let mut budget = ((window * SLICE_SHARE / scale.max(f64::EPSILON)) as u64).max(MIN_SLICE);
    let mut summary = earlier;
    let mut rest = VecDeque::from(content);
    let (mut halvings, mut summarized, mut calls) = (0, 0.0, 0);
    loop {
        let slice = take_slice(&mut rest, budget);
        let mut input = summary.clone();
        input.extend(slice.iter().cloned());
        match summarize(model, input).await {
            Ok(text) if rest.is_empty() => {
                if calls > 0 {
                    tracing::info!("calls=<{}>, budget=<{budget}> | summarized in slices", calls + 1);
                }
                return Ok(text);
            }
            Ok(text) => {
                calls += 1;
                summarized += heuristic_blocks(&slice) as f64 * scale;
                let header = format_summarized("the turns before", summarized.round() as u64, Some(&text));
                summary = vec![ContentBlock::Text(header)];
            }
            Err(Error::ContextWindowOverflow(error)) if halvings < MAX_HALVINGS => {
                tracing::warn!(
                    "budget=<{budget}>, error=<{error}> | summarizer input too long, halving the slice"
                );
                halvings += 1;
                budget = (budget / 2).max(MIN_SLICE);
                for block in slice.into_iter().rev() {
                    rest.push_front(block);
                }
            }
            Err(error) => return Err(error),
        }
    }
}

/// Takes the leading blocks of `rest` that estimate at most `budget` tokens, at least one; a text
/// block that does not fit is split to fill the slice.
fn take_slice(rest: &mut VecDeque<ContentBlock>, budget: u64) -> Vec<ContentBlock> {
    let mut slice = Vec::new();
    let mut size = 0;
    while let Some(block) = rest.pop_front() {
        let weight = heuristic_blocks(std::slice::from_ref(&block));
        if size + weight <= budget {
            size += weight;
            slice.push(block);
            continue;
        }
        let room = (budget - size) as usize * CHARS_PER_TOKEN;
        match block {
            // `room` is at least one token's characters, so at least one character fits.
            ContentBlock::Text(text) if room > 0 => {
                let mut at = room.min(text.len());
                while !text.is_char_boundary(at) {
                    at -= 1;
                }
                slice.push(ContentBlock::Text(text[..at].to_owned()));
                if at < text.len() {
                    rest.push_front(ContentBlock::Text(text[at..].to_owned()));
                }
            }
            block if slice.is_empty() => slice.push(block),
            block => rest.push_front(block),
        }
        break;
    }
    slice
}

/// One summarizer call over `content`, on the agent's auxiliary model.
async fn summarize(model: &Arc<dyn Model>, content: Vec<ContentBlock>) -> Result<String> {
    let mut blocks = vec![ContentBlock::text("<content>")];
    blocks.extend(content);
    blocks.push(ContentBlock::text("</content>"));
    let messages = [Message::new(Role::User, blocks)];
    let options = StreamOptions {
        system_prompt: Some(SystemPrompt::Text(SUMMARIZATION_SYSTEM_PROMPT.to_owned())),
        ..Default::default()
    };
    let mut stream = model.stream_aggregated(&messages, &options);
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if let AggregatedItem::Result(result) = item? {
            text = result.message.text();
        }
    }
    match text.trim() {
        "" => Err(Error::other("the summarizer returned no text")),
        summary => Ok(summary.to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use strands_agents::context_manager::{ContextManager, ContextManagerConfig, ContextManagerOption};
    use strands_agents::models::ModelEventStream;
    use strands_agents::testing::{MockMessageModel, MockResponse};
    use strands_agents::types::{ToolResultBlock, ToolResultContent, ToolUseBlock};
    use strands_agents::Agent;

    /// A task, then `turns` tool calls with ~300-token results.
    fn tool_loop(turns: usize) -> Vec<Message> {
        tool_loop_of(turns, 300)
    }

    /// A task, then `turns` tool calls with results of about `tokens` tokens each.
    fn tool_loop_of(turns: usize, tokens: usize) -> Vec<Message> {
        let mut messages = vec![Message::user("the task")];
        for i in 0..turns {
            let id = format!("t{i}");
            let call = ToolUseBlock::new("shell", id.clone(), serde_json::json!({ "command": "ls" }));
            messages.push(Message::new(Role::Assistant, vec![ContentBlock::ToolUse(call)]));
            let output = ToolResultContent::Text(format!("output {i} {}", "x".repeat(tokens * 4)));
            messages.push(Message::new(
                Role::User,
                vec![ContentBlock::ToolResult(ToolResultBlock::success(
                    id,
                    vec![output],
                ))],
            ));
        }
        messages
    }

    type Events = Arc<Mutex<Vec<Compaction>>>;

    fn compact(compact_at: Option<u64>, control: Option<Arc<ContextControl>>) -> (Arc<CompactAt>, Events) {
        let events: Events = Arc::default();
        let sink = Arc::clone(&events);
        let mut strategy = CompactAt::new(compact_at)
            .on_compacted(Some(Arc::new(move |compaction| sink.lock().push(compaction))));
        if let Some(control) = control {
            strategy = strategy.on_request(control);
        }
        (Arc::new(strategy), events)
    }

    fn pipeline(strategies: Vec<Arc<dyn ContextStrategy>>) -> Arc<ContextManager> {
        let config = ContextManagerConfig {
            strategies: Some(strategies),
            ..Default::default()
        };
        ContextManager::from(ContextManagerOption::Config(config)).unwrap()
    }

    fn manager(
        compact_at: Option<u64>,
        control: Option<Arc<ContextControl>>,
    ) -> (Arc<ContextManager>, Events) {
        let (compact, events) = compact(compact_at, control);
        (pipeline(vec![compact]), events)
    }

    /// One pipeline pass at a projected `tokens`; whether it changed the context.
    async fn pass(
        manager: &ContextManager,
        agent: &Agent,
        messages: &mut Vec<Message>,
        tokens: u64,
        overflow: bool,
    ) -> bool {
        manager
            .run_strategies_on(agent, messages, Some(tokens), overflow)
            .await
            .unwrap()
    }

    fn agent(summaries: &[&str]) -> (Agent, Arc<MockMessageModel>) {
        let responses = summaries
            .iter()
            .map(|text| MockResponse::message(Message::assistant(*text)))
            .collect();
        let model = MockMessageModel::shared(responses);
        (Agent::builder().model(model.clone()).build().unwrap(), model)
    }

    fn assert_well_formed(messages: &[Message]) {
        assert_eq!(messages[0].role, Role::User);
        for pair in messages.windows(2) {
            assert_ne!(pair[0].role, pair[1].role, "roles alternate");
        }
        let calls: Vec<String> = messages
            .iter()
            .flat_map(|m| m.tool_uses())
            .map(|c| c.tool_use_id.clone())
            .collect();
        for result in messages.iter().flat_map(|m| m.tool_results()) {
            assert!(
                calls.contains(&result.tool_use_id),
                "result {} has its call",
                result.tool_use_id
            );
        }
    }

    #[tokio::test]
    async fn a_conversation_at_1_4x_the_point_ends_below_half_of_it_after_one_call() {
        let (agent, model) = agent(&["SUMMARY one"]);
        // The mock has no window, so 200k is assumed: the point is `compact_at`.
        let (manager, events) = manager(Some(10_000), None);
        let mut messages = tool_loop(45);
        let before = heuristic(&messages);
        assert!((13_500..15_500).contains(&before), "{before} tokens");

        let changed = pass(&manager, &agent, &mut messages, before, false).await;

        assert!(changed);
        assert_eq!(model.call_count(), 1);
        let after = heuristic(&messages);
        assert!(after <= 5_000, "{after} tokens left");
        assert!(after > 2_500, "only what is needed is summarized: {after}");
        assert_well_formed(&messages);
        assert!(
            messages[0].text().starts_with("the task"),
            "the task stays verbatim"
        );
        assert!(messages[0].text().contains("SUMMARY one"));
        let reported = events.lock().clone();
        assert!(
            matches!(reported[..], [Compaction::Done { before: b, after }] if b == before && after <= 5_000),
            "{reported:?}"
        );
        // The summarizer saw the turns as text: Bedrock refuses tool calls in a user message.
        let (input, _) = &model.calls()[0];
        assert!(input[0]
            .content
            .iter()
            .all(|b| matches!(b, ContentBlock::Text(_))));
        assert!(input[0].text().contains("[tool call shell]"));
    }

    /// Arms `compact` and then edits the context, like the swarm's cache pruning, so the SDK
    /// re-estimates it from the messages alone.
    struct ArmThenEdit(Arc<CompactAt>);

    #[async_trait]
    impl ContextStrategy for ArmThenEdit {
        fn name(&self) -> &str {
            "arm-then-edit"
        }

        async fn apply(&self, ctx: &mut ContextState<'_>) -> Result<bool> {
            self.0.arm(ctx);
            ctx.messages.last_mut().unwrap().content =
                vec![ContentBlock::ToolResult(ToolResultBlock::success(
                    format!("t{}", 44),
                    vec![ToolResultContent::Text("trimmed".into())],
                ))];
            Ok(true)
        }
    }

    #[tokio::test]
    async fn after_an_edit_it_still_counts_what_the_estimate_misses() {
        let (agent, _) = agent(&["SUMMARY one"]);
        let (compact, events) = compact(Some(10_000), None);
        let manager = pipeline(vec![Arc::new(ArmThenEdit(Arc::clone(&compact))), compact]);
        let mut messages = tool_loop(45);
        // The provider counts twice the characters' estimate (system prompt, tools, a tokenizer
        // that weighs code more), as a long-running swarm agent's calls do.
        let real = |messages: &[Message]| 2 * heuristic(messages);
        let projected = real(&messages);

        pass(&manager, &agent, &mut messages, projected, false).await;

        let reported = events.lock().clone();
        let [Compaction::Done { before, after }] = reported[..] else {
            panic!("{reported:?}");
        };
        assert!(before > projected * 9 / 10, "before={before}, real={projected}");
        assert!(
            real(&messages) <= 6_000,
            "half the point, in real tokens: {}",
            real(&messages)
        );
        assert!(
            after.abs_diff(real(&messages)) < 1_000,
            "after={after}, real={}",
            real(&messages)
        );
    }

    /// Arms `compact`, then trims the older half of the tool results, as cache pruning does.
    struct ArmThenPrune(Arc<CompactAt>);

    #[async_trait]
    impl ContextStrategy for ArmThenPrune {
        fn name(&self) -> &str {
            "arm-then-prune"
        }

        async fn apply(&self, ctx: &mut ContextState<'_>) -> Result<bool> {
            self.0.arm(ctx);
            let half = ctx.messages.len() / 2;
            for message in &mut ctx.messages[1..half] {
                for block in &mut message.content {
                    if let ContentBlock::ToolResult(result) = block {
                        result.content = vec![ToolResultContent::Text("pruned".into())];
                    }
                }
            }
            Ok(true)
        }
    }

    #[tokio::test]
    async fn a_report_gives_the_size_that_set_it_off_not_what_pruning_left() {
        let (agent, _) = agent(&["SUMMARY one"]);
        let (compact, events) = compact(Some(10_000), None);
        let manager = pipeline(vec![Arc::new(ArmThenPrune(Arc::clone(&compact))), compact]);
        let mut messages = tool_loop(45);
        let projected = heuristic(&messages);

        pass(&manager, &agent, &mut messages, projected, false).await;

        let reported = events.lock().clone();
        let [Compaction::Done { before, .. }] = reported[..] else {
            panic!("{reported:?}");
        };
        assert_eq!(before, projected);
    }

    #[tokio::test]
    async fn stays_out_of_the_way_below_the_point() {
        let (agent, model) = agent(&["unused"]);
        let (manager, events) = manager(Some(10_000), None);
        let mut messages = tool_loop(10);
        let changed = pass(&manager, &agent, &mut messages, 9_999, false).await;
        assert!(!changed);
        assert_eq!(model.call_count(), 0);
        assert!(events.lock().is_empty());
    }

    #[tokio::test]
    async fn a_second_pass_runs_when_the_first_summary_leaves_it_above_the_point() {
        let long = format!("LONG {}", "s".repeat(48_000));
        let (agent, model) = agent(&[&long, "SHORT"]);
        let (manager, events) = manager(Some(10_000), None);
        let mut messages = tool_loop(45);
        let before = heuristic(&messages);

        pass(&manager, &agent, &mut messages, before, false).await;

        assert_eq!(model.call_count(), 2);
        // The first summary went into the second; one summary is left.
        assert!(model.calls()[1].0[0].text().contains("LONG"));
        let first = messages[0].text();
        assert!(
            first.contains("SHORT") && !first.contains("LONG"),
            "{}",
            &first[..200.min(first.len())]
        );
        assert!(heuristic(&messages) <= 5_000);
        assert_eq!(events.lock().len(), 1, "one event per compaction");
        assert_well_formed(&messages);
    }

    #[tokio::test]
    async fn a_failed_summary_leaves_the_context_alone_and_is_reported() {
        let (agent, _model) = agent(&[""]);
        let (manager, events) = manager(Some(10_000), None);
        let mut messages = tool_loop(45);
        let original = messages.clone();
        let before = heuristic(&messages);
        let changed = pass(&manager, &agent, &mut messages, before, false).await;
        assert!(!changed);
        assert_eq!(messages, original);
        assert_eq!(
            events.lock()[..],
            [Compaction::Failed {
                tokens: before,
                error: "the summarizer returned no text".into()
            }]
        );
    }

    #[tokio::test]
    async fn nothing_to_summarize_is_not_a_failure() {
        let (agent, model) = agent(&["unused"]);
        let (manager, events) = manager(Some(100), None);
        let mut messages = tool_loop(1);
        let changed = pass(&manager, &agent, &mut messages, 5_000, false).await;
        assert!(!changed);
        assert_eq!(model.call_count(), 0);
        assert!(events.lock().is_empty());
    }

    #[tokio::test]
    async fn after_a_failed_summary_it_backs_off_until_the_input_grows_or_calls_pass() {
        let (agent, model) = agent(&["", "", "", "SUMMARY at last", "SUMMARY again"]);
        let control = ContextControl::new(Some(10_000));
        let (manager, events) = manager(Some(10_000), Some(Arc::clone(&control)));
        let mut messages = tool_loop(45);
        let tokens = heuristic(&messages);

        // The first attempt fails; the next calls at about the same size do not retry.
        assert!(!pass(&manager, &agent, &mut messages, tokens, false).await);
        assert_eq!(model.call_count(), 1);
        for _ in 1..RETRY_CALLS {
            pass(&manager, &agent, &mut messages, tokens + 100, false).await;
        }
        assert_eq!(model.call_count(), 1, "backing off");
        // The RETRY_CALLS-th call after the failure retries (and fails again).
        pass(&manager, &agent, &mut messages, tokens, false).await;
        assert_eq!(model.call_count(), 2);
        // Growth past RETRY_GROWTH retries at once (and fails again).
        let grown = (tokens as f64 * (1.0 + RETRY_GROWTH)) as u64 + 1;
        pass(&manager, &agent, &mut messages, grown, false).await;
        assert_eq!(model.call_count(), 3);
        pass(&manager, &agent, &mut messages, grown, false).await;
        assert_eq!(model.call_count(), 3, "backing off again");
        // A request goes through, here the one that succeeds.
        control.request_summary();
        assert!(pass(&manager, &agent, &mut messages, grown, false).await);
        assert_eq!(model.call_count(), 4);
        // Each failed attempt is reported once, the backed-off calls not at all.
        let reported = events.lock().clone();
        assert_eq!(reported.len(), 4, "{reported:?}");
        assert!(reported[..3]
            .iter()
            .all(|c| matches!(c, Compaction::Failed { .. })));
        assert!(matches!(reported[3], Compaction::Done { .. }));
        assert!(messages[0].text().contains("SUMMARY at last"));
        // Success ends the backoff: the next call at the point compacts again (on a fresh history,
        // the request left nothing to cut).
        let mut messages = tool_loop(45);
        assert!(pass(&manager, &agent, &mut messages, tokens, false).await);
        assert_eq!(model.call_count(), 5);
    }

    #[tokio::test]
    async fn a_cancelled_summary_is_no_failure_and_starts_no_backoff() {
        let (_, model) = agent(&["SUMMARY after the reload"]);
        let cut_short = Arc::new(CancelledOnce {
            inner: Arc::clone(&model),
            cancelled: std::sync::atomic::AtomicBool::new(false),
        });
        let agent = Agent::builder().model(cut_short).build().unwrap();
        let (manager, events) = manager(Some(10_000), None);
        let mut messages = tool_loop(45);
        let tokens = heuristic(&messages);

        assert!(!pass(&manager, &agent, &mut messages, tokens, false).await);
        assert!(events.lock().is_empty(), "{:?}", events.lock());
        // The next call at the same size retries at once.
        assert!(pass(&manager, &agent, &mut messages, tokens, false).await);
        assert_eq!(model.call_count(), 1);
        assert!(matches!(events.lock()[..], [Compaction::Done { .. }]));
    }

    /// A summarizer whose first call a shutdown cuts short (as the swarm's side model does), then
    /// `inner`.
    struct CancelledOnce {
        inner: Arc<MockMessageModel>,
        cancelled: std::sync::atomic::AtomicBool,
    }

    impl Model for CancelledOnce {
        fn model_id(&self) -> Option<String> {
            self.inner.model_id()
        }

        fn stream<'a>(&'a self, messages: &'a [Message], options: &'a StreamOptions) -> ModelEventStream<'a> {
            if self.cancelled.swap(true, std::sync::atomic::Ordering::SeqCst) {
                return self.inner.stream(messages, options);
            }
            Box::pin(futures::stream::iter([Err(Error::Cancelled)]))
        }
    }

    /// `inner` behind a context window, as a provider enforces it: a call whose input (system
    /// prompt included) estimates over `rejects_over` tokens fails with an overflow and uses no
    /// response. It reports `window` as its window, which may be larger than what it accepts.
    struct Windowed {
        inner: Arc<MockMessageModel>,
        window: u64,
        rejects_over: u64,
        /// Each call's input, in estimated tokens, rejected or not.
        inputs: Mutex<Vec<u64>>,
    }

    impl Windowed {
        fn new(responses: &[&str], window: u64) -> Arc<Self> {
            Self::rejecting(responses, window, window)
        }

        fn rejecting(responses: &[&str], window: u64, rejects_over: u64) -> Arc<Self> {
            let responses = responses
                .iter()
                .map(|text| MockResponse::message(Message::assistant(*text)))
                .collect();
            Arc::new(Self {
                inner: MockMessageModel::shared(responses),
                window,
                rejects_over,
                inputs: Mutex::default(),
            })
        }

        fn inputs(&self) -> Vec<u64> {
            self.inputs.lock().clone()
        }
    }

    impl Model for Windowed {
        fn model_id(&self) -> Option<String> {
            self.inner.model_id()
        }

        fn context_window_limit(&self) -> Option<u64> {
            Some(self.window)
        }

        fn stream<'a>(&'a self, messages: &'a [Message], options: &'a StreamOptions) -> ModelEventStream<'a> {
            let count = CountTokensOptions {
                system_prompt: options.system_prompt.clone(),
                tool_specs: options.tool_specs.clone(),
            };
            let input = estimate_tokens_heuristic(messages, &count);
            self.inputs.lock().push(input);
            if input > self.rejects_over {
                let error = Error::ContextWindowOverflow(format!(
                    "Input is too long for requested model: {input} > {}",
                    self.rejects_over
                ));
                return Box::pin(futures::stream::iter([Err(error)]));
            }
            self.inner.stream(messages, options)
        }
    }

    /// An agent on a 1M-window model whose side calls run on `summarizer`.
    fn agent_with_summarizer(summarizer: &Arc<Windowed>) -> Agent {
        Agent::builder()
            .model(Windowed::new(&[], 1_000_000))
            .aux_model(Arc::clone(summarizer))
            .build()
            .unwrap()
    }

    /// The default point of a 1M window (850k) in ~2k-token tool results.
    fn at_850k() -> Vec<Message> {
        let messages = tool_loop_of(425, 2_000);
        let tokens = heuristic(&messages);
        assert!((850_000..860_000).contains(&tokens), "{tokens}");
        messages
    }

    #[tokio::test]
    async fn at_850k_a_200k_summarizer_gets_slices_that_fit_its_window() {
        let summaries: Vec<String> = (1..=8).map(|i| format!("SUMMARY {i}")).collect();
        let summaries: Vec<&str> = summaries.iter().map(String::as_str).collect();
        let summarizer = Windowed::new(&summaries, 200_000);
        let agent = agent_with_summarizer(&summarizer);
        let (manager, events) = manager(None, None);
        let mut messages = at_850k();
        let before = heuristic(&messages);

        assert!(pass(&manager, &agent, &mut messages, before, false).await);

        let inputs = summarizer.inputs();
        assert!(inputs.len() > 1, "{inputs:?}");
        assert!(inputs.iter().all(|&input| input <= 200_000), "{inputs:?}");
        let after = heuristic(&messages);
        assert!((300_000..=425_000).contains(&after), "{after} tokens left");
        // Each slice's summary went into the next; the last one is what is kept.
        let calls = summarizer.inner.calls();
        assert!(calls[1].0[0].text().contains("SUMMARY 1"));
        let last = format!("SUMMARY {}", inputs.len());
        assert!(
            messages[0].text().contains(&last),
            "{}",
            &messages[0].text()[..200]
        );
        assert!(!messages[0].text().contains("SUMMARY 1\n"));
        assert_well_formed(&messages);
        assert!(matches!(events.lock()[..], [Compaction::Done { after: a, .. }] if a <= 425_000));
    }

    #[tokio::test]
    async fn a_summarizer_that_rejects_a_slice_gets_smaller_ones() {
        // It claims a 1M window but takes no more than 150k.
        let summaries: Vec<String> = (1..=8).map(|i| format!("SUMMARY {i}")).collect();
        let summaries: Vec<&str> = summaries.iter().map(String::as_str).collect();
        let summarizer = Windowed::rejecting(&summaries, 1_000_000, 150_000);
        let agent = agent_with_summarizer(&summarizer);
        let (manager, events) = manager(None, None);
        let mut messages = at_850k();
        let before = heuristic(&messages);

        assert!(pass(&manager, &agent, &mut messages, before, false).await);

        let inputs = summarizer.inputs();
        assert!(inputs[0] > 150_000, "the first try was too big: {inputs:?}");
        let accepted = summarizer.inner.call_count();
        assert!(accepted >= 3, "{inputs:?}");
        assert!(heuristic(&messages) <= 425_000);
        assert!(messages[0].text().contains(&format!("SUMMARY {accepted}")));
        assert!(matches!(events.lock()[..], [Compaction::Done { .. }]));
    }

    #[tokio::test]
    async fn an_agent_over_its_window_with_a_200k_summarizer_finishes_its_turn() {
        let summaries: Vec<String> = (1..=10).map(|i| format!("SUMMARY {i}")).collect();
        let summaries: Vec<&str> = summaries.iter().map(String::as_str).collect();
        let summarizer = Windowed::new(&summaries, 200_000);
        let model = Windowed::new(&["done"], 1_000_000);
        let (compact, events) = compact(None, None);
        let config = ContextManagerConfig {
            strategies: Some(vec![compact]),
            ..Default::default()
        };
        // Past the 1M window, as after a resume or a burst of big tool results.
        let mut history = tool_loop_of(520, 2_000);
        history.push(Message::assistant("next"));
        let agent = Agent::builder()
            .model(Arc::clone(&model))
            .aux_model(Arc::clone(&summarizer))
            .messages(history)
            .context_manager(ContextManagerOption::Config(config))
            .build()
            .unwrap();

        agent.invoke("go on").await.unwrap();

        // One call, under the window: no overflow, retry or truncation.
        let inputs = model.inputs();
        assert_eq!(inputs.len(), 1, "{inputs:?}");
        assert!(inputs[0] <= 500_000, "{inputs:?}");
        assert!(summarizer.inputs().iter().all(|&input| input <= 200_000));
        let messages = agent.messages();
        assert!(messages[0].text().starts_with("the task"));
        assert!(messages[0].text().contains("SUMMARY"));
        assert!(matches!(events.lock()[..], [Compaction::Done { .. }]));
    }

    #[test]
    fn a_slice_splits_a_text_too_big_for_it_at_a_character() {
        let text = |s: &str| ContentBlock::Text(s.to_owned());
        let big = "é".repeat(30_000); // 60k bytes
        let mut rest = VecDeque::from([text("small"), text(&big), text("after")]);
        let budget = 5_000;
        let mut joined = String::new();
        let mut slices = 0;
        while !rest.is_empty() {
            let slice = take_slice(&mut rest, budget);
            assert!(!slice.is_empty());
            assert!(
                heuristic_blocks(&slice) <= budget + 10,
                "{}",
                heuristic_blocks(&slice)
            );
            for block in &slice {
                let ContentBlock::Text(part) = block else { panic!() };
                joined.push_str(part);
            }
            slices += 1;
        }
        assert_eq!(joined, format!("small{big}after"), "nothing lost or doubled");
        assert!((3..=5).contains(&slices), "{slices}");
        // A block that is not text goes alone when over the budget.
        let input = serde_json::json!({ "command": "x".repeat(40_000) });
        let call = ContentBlock::ToolUse(ToolUseBlock::new("shell", "id", input));
        let mut rest = VecDeque::from([call, text("after")]);
        assert_eq!(take_slice(&mut rest, 1_000).len(), 1);
        assert_eq!(rest.len(), 1);
    }

    #[tokio::test]
    async fn overflow_retries_during_a_backoff() {
        let (agent, model) = agent(&["", "SUMMARY"]);
        let (manager, _events) = manager(Some(10_000), None);
        let mut messages = tool_loop(45);
        let tokens = heuristic(&messages);
        pass(&manager, &agent, &mut messages, tokens, false).await;
        pass(&manager, &agent, &mut messages, tokens, false).await;
        assert_eq!(model.call_count(), 1);
        assert!(pass(&manager, &agent, &mut messages, tokens, true).await);
        assert_eq!(model.call_count(), 2);
    }

    #[tokio::test]
    async fn on_request_it_keeps_only_the_latest_messages() {
        let (agent, _model) = agent(&["SUMMARY on request"]);
        let control = ContextControl::new(Some(10_000));
        let (manager, events) = manager(Some(10_000), Some(Arc::clone(&control)));
        control.request_summary();
        let mut messages = tool_loop(10);
        let tokens = heuristic(&messages);
        let changed = pass(&manager, &agent, &mut messages, tokens, false).await;
        assert!(changed);
        assert!(
            messages.len() <= 1 + SUMMARY_KEEPS + 1,
            "{} messages",
            messages.len()
        );
        assert!(messages.len() > SUMMARY_KEEPS);
        assert_well_formed(&messages);
        assert_eq!(events.lock().len(), 1);
    }

    #[test]
    fn by_default_compacts_at_85_percent_of_the_window() {
        // A 1M window (Opus 5.5) is used to 850k, not cut at 200k as before.
        assert_eq!(compaction_point(1_000_000, None), 850_000);
        assert!((compaction_utilization(1_000_000, None) - AUTO_SUMMARIZE_UTILIZATION).abs() < 1e-9);
        assert_eq!(compaction_point(200_000, None), 170_000);
        assert_eq!(compaction_point(0, None), 1);
    }

    #[test]
    fn compacts_at_the_smaller_of_compact_at_and_85_percent() {
        assert_eq!(compaction_point(1_000_000, Some(200_000)), 200_000);
        assert!((compaction_utilization(1_000_000, Some(200_000)) - 0.2).abs() < 1e-9);
        assert_eq!(compaction_point(200_000, Some(200_000)), 170_000);
        assert!((compaction_utilization(200_000, Some(200_000)) - AUTO_SUMMARIZE_UTILIZATION).abs() < 1e-9);
        assert_eq!(compaction_point(200_000, Some(1_500_000)), 170_000);
        assert_eq!(compaction_point(1_000_000, Some(50_000)), 50_000);
    }
}
