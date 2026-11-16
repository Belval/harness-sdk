//! What to drop from an agent's context before each model call, beyond the SDK's pipeline.
//!
//! Reasoning: models that keep prior turns' thinking (Claude Opus 4.5+) bill it as input on every
//! later call, and its signatures alone are most of a long agent's reasoning bytes. Anthropic
//! requires thinking blocks within the turn in progress (a tool-use loop is one turn) and allows
//! omitting earlier turns' thinking, as long as no dropped block sits between two kept ones.

use async_trait::async_trait;
use strands_agents::context_manager::{ContextState, ContextStrategy};
use strands_agents::types::{ContentBlock, Message, Role};
use strands_agents::Result;

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
}
