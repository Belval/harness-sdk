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

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use strands_agents::context_manager::stash::Stash;
use strands_agents::context_manager::strategies::{
    Offload, OffloadConditions, OffloadStrategy, OffloadTarget, TruncateConfig,
};
use strands_agents::context_manager::{ContextState, ContextStrategy};
use strands_agents::types::{ContentBlock, Message, Role, ToolResultBlock, ToolResultContent};
use strands_agents::{Agent, Result};

/// Stands in for an assistant message that held nothing but reasoning.
pub const REASONING_DROPPED: &str = "[reasoning dropped]";

/// Drops reasoning blocks (text and signatures) from every turn before the current one.
pub struct DropOldReasoning;

#[async_trait]
impl ContextStrategy for DropOldReasoning {
    fn name(&self) -> &str {
        "swarm:drop-old-reasoning"
    }

    async fn apply(&self, ctx: &mut ContextState<'_>) -> Result<bool> {
        Ok(drop_old_reasoning(ctx.messages))
    }
}

/// Where the current turn starts: the last user message that carries no tool results.
fn turn_start(messages: &[Message]) -> usize {
    messages
        .iter()
        .rposition(|m| m.role == Role::User && m.tool_results().next().is_none())
        .unwrap_or(0)
}

/// Removes reasoning from the assistant messages before the current turn; `true` when any went.
/// A message left empty gets [`REASONING_DROPPED`] so the history stays valid.
pub fn drop_old_reasoning(messages: &mut [Message]) -> bool {
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
pub const TRIMMED_TOOLS: [&str; 2] = ["shell", "read"];
/// The latest this many [`TRIMMED_TOOLS`] results stay whole.
pub const KEEP_RECENT_RESULTS: usize = 6;
/// Older results longer than this (in characters) are trimmed.
pub const TRIM_ABOVE_CHARS: usize = 2_000;
/// What a trimmed result keeps, in tokens (about 800 characters, head and tail).
const TRIM_PREVIEW_TOKENS: usize = 200;

/// Trims [`TRIMMED_TOOLS`] results longer than [`TRIM_ABOVE_CHARS`] once
/// [`KEEP_RECENT_RESULTS`] newer ones exist. It runs in batches, only once such a result is twice
/// that old, because every edit to history invalidates the prompt cache from that point on.
pub struct TrimOldResults {
    inner: OffloadStrategy,
}

impl Default for TrimOldResults {
    fn default() -> Self {
        let truncate = Offload::truncate(
            OffloadTarget::tools(TRIMMED_TOOLS),
            TruncateConfig {
                preview_tokens: Some(TRIM_PREVIEW_TOKENS),
                preview: None,
            },
        );
        let inner = truncate.when(OffloadConditions {
            threshold: Some(TRIM_ABOVE_CHARS.div_ceil(4) as f64),
            preserve_recent: Some(KEEP_RECENT_RESULTS as f64),
            ..Default::default()
        });
        Self { inner }
    }
}

#[async_trait]
impl ContextStrategy for TrimOldResults {
    fn name(&self) -> &str {
        "swarm:trim-old-results"
    }

    fn init(&self, agent: &Agent, stash: Option<Arc<Stash>>) {
        self.inner.init(agent, stash);
    }

    async fn apply(&self, ctx: &mut ContextState<'_>) -> Result<bool> {
        if !trim_due(ctx.messages) {
            return Ok(false);
        }
        self.inner.apply(ctx).await
    }
}

/// Whether a [`TRIMMED_TOOLS`] result over [`TRIM_ABOVE_CHARS`] has `2 * KEEP_RECENT_RESULTS`
/// newer ones.
fn trim_due(messages: &[Message]) -> bool {
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
    let old = matching.len().saturating_sub(2 * KEEP_RECENT_RESULTS);
    matching[..old]
        .iter()
        .flat_map(|m| m.tool_results().filter(trimmed))
        .any(|r| result_chars(r) > TRIM_ABOVE_CHARS)
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
    use strands_agents::types::{ReasoningBlock, ToolResultBlock, ToolResultContent, ToolUseBlock};

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

    fn tool_result(id: &str) -> Message {
        Message::new(
            Role::User,
            vec![ContentBlock::ToolResult(ToolResultBlock::success(
                id,
                vec![ToolResultContent::Text("ok".into())],
            ))],
        )
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
            messages.push(Message::new(
                Role::User,
                vec![ContentBlock::ToolResult(ToolResultBlock::success(
                    &id,
                    vec![ToolResultContent::Text("x".repeat(*chars))],
                ))],
            ));
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
        use strands_agents::testing::MockMessageModel;
        use strands_agents::InMemoryStorage;
        let agent = Agent::builder()
            .model(MockMessageModel::shared(vec![]))
            .no_sandbox()
            .build()
            .unwrap();
        let stash = Arc::new(Stash::new(Arc::new(InMemoryStorage::new()), "s", "a1", false));
        let mut ctx = ContextState {
            messages,
            agent: &agent,
            utilization: 0.0,
            overflow: false,
            stash: Some(stash),
        };
        TrimOldResults::default().apply(&mut ctx).await.unwrap()
    }

    #[tokio::test]
    async fn waits_until_a_large_result_is_twice_the_kept_window_old() {
        let mut messages = history(&[("shell", 5_000); 2 * KEEP_RECENT_RESULTS]);
        let before = messages.clone();
        assert!(!trim(&mut messages).await);
        assert_eq!(messages, before);
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
}
