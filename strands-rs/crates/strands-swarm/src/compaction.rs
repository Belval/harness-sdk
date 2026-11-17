//! When to summarize an agent's older turns: at a fixed token count (`--compact-at`), not only
//! near the end of the model's window. A 1M-token window would otherwise let every call carry
//! up to ~850k tokens before anything is compacted.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::dashboard::{ContextControl, SUMMARY_KEEPS};
use async_trait::async_trait;
use futures::StreamExt;
use strands_agents::context_manager::strategies::{
    flatten_messages_to_content, format_summarized, SUMMARIZATION_SYSTEM_PROMPT, SUMMARIZED_PREFIX,
};
use strands_agents::context_manager::{ContextState, ContextStrategy, AUTO_SUMMARIZE_UTILIZATION};
use strands_agents::models::{
    estimate_tokens_heuristic, AggregatedItem, CountTokensOptions, StreamOptions,
    DEFAULT_CONTEXT_WINDOW_LIMIT,
};
use strands_agents::types::{ContentBlock, Message, Role, SystemPrompt};
use strands_agents::{Error, Model, Result};

/// Default `--compact-at`: summarize once a call's input reaches 200k tokens.
pub const DEFAULT_COMPACT_AT: u64 = 200_000;

/// The input size (tokens) at which older turns are summarized: `compact_at`, or the context
/// manager's own 85% mark when the window is smaller.
pub fn compaction_point(window: u64, compact_at: u64) -> u64 {
    let auto = (window as f64 * AUTO_SUMMARIZE_UTILIZATION) as u64;
    auto.min(compact_at).max(1)
}

/// The utilization (share of `window`) at which to summarize:
/// `min(AUTO_SUMMARIZE_UTILIZATION, compact_at / window)`.
pub fn compaction_utilization(window: u64, compact_at: u64) -> f64 {
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
        /// Projected input tokens before.
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
    compact_at: u64,
    control: Option<Arc<ContextControl>>,
    sink: Option<CompactedSink>,
    /// Set by [`CompactAt::arm`]: run on this pipeline pass even if an earlier strategy brought
    /// the context under the point (as long as it is still above the target).
    armed: AtomicBool,
    /// Set by [`CompactAt::arm`] when it took a summarize request.
    armed_request: AtomicBool,
    /// Set when a compaction changed nothing; cleared by the next one that does.
    backoff: Mutex<Option<Backoff>>,
}

impl CompactAt {
    /// Compact at `compact_at` tokens.
    pub fn new(compact_at: u64) -> Self {
        Self {
            compact_at,
            control: None,
            sink: None,
            armed: AtomicBool::new(false),
            armed_request: AtomicBool::new(false),
            backoff: Mutex::new(None),
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
        let due = self.due(ctx);
        if let Some(requested) = due {
            self.armed.store(true, Ordering::SeqCst);
            self.armed_request.fetch_or(requested, Ordering::SeqCst);
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

    async fn compact(&self, ctx: &mut ContextState<'_>) -> Result<bool> {
        let armed = self.armed.swap(false, Ordering::SeqCst);
        let armed_request = self.armed_request.swap(false, Ordering::SeqCst);
        let due = self.due(ctx);
        if !armed && due.is_none() {
            return Ok(false);
        }
        let requested = armed_request || due == Some(true);
        let window = window_of(ctx.agent.model().as_ref());
        let point = compaction_point(window, self.compact_at);
        let before = input_tokens(ctx, window);
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
            before,
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
    let mut content = earlier;
    content.extend(flatten_messages_to_content(&messages[1..cut]));
    let summary = summarize(model, content)
        .await
        .inspect_err(|error| tracing::warn!("error=<{error}> | summarizing older turns failed"))?;
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
    use strands_agents::testing::{MockMessageModel, MockResponse};
    use strands_agents::types::{ToolResultBlock, ToolResultContent, ToolUseBlock};
    use strands_agents::Agent;

    /// A task, then `turns` tool calls with ~300-token results.
    fn tool_loop(turns: usize) -> Vec<Message> {
        let mut messages = vec![Message::user("the task")];
        for i in 0..turns {
            let id = format!("t{i}");
            let call = ToolUseBlock::new("shell", id.clone(), serde_json::json!({ "command": "ls" }));
            messages.push(Message::new(Role::Assistant, vec![ContentBlock::ToolUse(call)]));
            let output = ToolResultContent::Text(format!("output {i} {}", "x".repeat(1_200)));
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

    fn manager(compact_at: u64, control: Option<Arc<ContextControl>>) -> (Arc<ContextManager>, Events) {
        let events: Events = Arc::default();
        let sink = Arc::clone(&events);
        let mut strategy = CompactAt::new(compact_at)
            .on_compacted(Some(Arc::new(move |compaction| sink.lock().push(compaction))));
        if let Some(control) = control {
            strategy = strategy.on_request(control);
        }
        let config = ContextManagerConfig {
            strategies: Some(vec![Arc::new(strategy)]),
            ..Default::default()
        };
        (
            ContextManager::from(ContextManagerOption::Config(config)).unwrap(),
            events,
        )
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
        let (manager, events) = manager(10_000, None);
        let mut messages = tool_loop(45);
        let before = heuristic(&messages);
        assert!((13_500..15_500).contains(&before), "{before} tokens");

        let changed = manager
            .run_strategies_on(&agent, &mut messages, Some(before), false)
            .await
            .unwrap();

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

    #[tokio::test]
    async fn stays_out_of_the_way_below_the_point() {
        let (agent, model) = agent(&["unused"]);
        let (manager, events) = manager(10_000, None);
        let mut messages = tool_loop(10);
        let changed = manager
            .run_strategies_on(&agent, &mut messages, Some(9_999), false)
            .await
            .unwrap();
        assert!(!changed);
        assert_eq!(model.call_count(), 0);
        assert!(events.lock().is_empty());
    }

    #[tokio::test]
    async fn a_second_pass_runs_when_the_first_summary_leaves_it_above_the_point() {
        let long = format!("LONG {}", "s".repeat(48_000));
        let (agent, model) = agent(&[&long, "SHORT"]);
        let (manager, events) = manager(10_000, None);
        let mut messages = tool_loop(45);
        let before = heuristic(&messages);

        manager
            .run_strategies_on(&agent, &mut messages, Some(before), false)
            .await
            .unwrap();

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
        let (manager, events) = manager(10_000, None);
        let mut messages = tool_loop(45);
        let original = messages.clone();
        let before = heuristic(&messages);
        let changed = manager
            .run_strategies_on(&agent, &mut messages, Some(before), false)
            .await
            .unwrap();
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
        let (manager, events) = manager(100, None);
        let mut messages = tool_loop(1);
        let changed = manager
            .run_strategies_on(&agent, &mut messages, Some(5_000), false)
            .await
            .unwrap();
        assert!(!changed);
        assert_eq!(model.call_count(), 0);
        assert!(events.lock().is_empty());
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

    #[tokio::test]
    async fn after_a_failed_summary_it_backs_off_until_the_input_grows_or_calls_pass() {
        let (agent, model) = agent(&["", "", "", "SUMMARY at last", "SUMMARY again"]);
        let control = ContextControl::new(10_000);
        let (manager, events) = manager(10_000, Some(Arc::clone(&control)));
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
    async fn overflow_retries_during_a_backoff() {
        let (agent, model) = agent(&["", "SUMMARY"]);
        let (manager, _events) = manager(10_000, None);
        let mut messages = tool_loop(45);
        let tokens = heuristic(&messages);
        pass(&manager, &agent, &mut messages, tokens, false).await;
        pass(&manager, &agent, &mut messages, tokens, false).await;
        assert_eq!(model.call_count(), 1);
        assert!(pass(&manager, &agent, &mut messages, tokens, true).await);
        assert_eq!(model.call_count(), 2);
    }

    #[test]
    fn the_backoff_ends_after_its_calls_or_on_growth() {
        let compact = CompactAt::new(10_000);
        assert!(!compact.backing_off(9_000));
        *compact.backoff.lock() = Some(Backoff {
            tokens: 10_000,
            calls_left: 2,
        });
        assert!(compact.backing_off(10_999));
        assert!(!compact.backing_off(11_000));
        compact.tick();
        assert!(compact.backing_off(10_000));
        compact.tick();
        assert!(!compact.backing_off(10_000));
        compact.tick();
        assert_eq!(compact.backoff.lock().unwrap().calls_left, 0);
    }

    #[tokio::test]
    async fn on_request_it_keeps_only_the_latest_messages() {
        let (agent, _model) = agent(&["SUMMARY on request"]);
        let control = ContextControl::new(10_000);
        let (manager, events) = manager(10_000, Some(Arc::clone(&control)));
        control.request_summary();
        let mut messages = tool_loop(10);
        let tokens = heuristic(&messages);
        let changed = manager
            .run_strategies_on(&agent, &mut messages, Some(tokens), false)
            .await
            .unwrap();
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
    fn compacts_at_the_smaller_of_compact_at_and_85_percent() {
        assert_eq!(compaction_point(1_000_000, 200_000), 200_000);
        assert!((compaction_utilization(1_000_000, 200_000) - 0.2).abs() < 1e-9);
        assert_eq!(compaction_point(200_000, 200_000), 170_000);
        assert!((compaction_utilization(200_000, 200_000) - AUTO_SUMMARIZE_UTILIZATION).abs() < 1e-9);
        assert_eq!(compaction_point(200_000, 1_500_000), 170_000);
        assert_eq!(compaction_point(1_000_000, 50_000), 50_000);
    }
}
