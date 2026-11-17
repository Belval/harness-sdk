//! The notice a restored agent resumes with: the facts it would otherwise re-derive by re-reading
//! its work after a restart. That covers the tool calls the restart cut short and the state of its
//! checkout. Its todo list needs no copy here, because the harness re-surfaces it before every
//! model call.

use std::path::Path;
use std::time::Duration;

use strands_agents::tools::EXECUTION_CANCELLED_MESSAGE;
use strands_agents::types::{ContentBlock, Message, Role, ToolResultBlock, ToolResultStatus, ToolUseBlock};
use strands_agents::InvokeInput;

/// How every restart notice starts, so context strategies can tell it from a new task.
pub const NOTICE_PREFIX: &str = "[The swarm was restarted";
/// Longest tool input shown, in characters.
const INPUT_CHARS: usize = 300;
/// Most `git status` lines shown.
const STATUS_LINES: usize = 30;

/// The calls of the last assistant message that never completed: unanswered, or cancelled.
pub fn in_flight(messages: &[Message]) -> Vec<ToolUseBlock> {
    let Some(at) = messages.iter().rposition(|m| m.role == Role::Assistant) else {
        return Vec::new();
    };
    let results: Vec<&ToolResultBlock> = messages[at + 1..].iter().flat_map(|m| m.tool_results()).collect();
    messages[at]
        .tool_uses()
        .filter(
            |call| match results.iter().find(|r| r.tool_use_id == call.tool_use_id) {
                None => true,
                Some(result) => {
                    result.status == ToolResultStatus::Error && result.text() == EXECUTION_CANCELLED_MESSAGE
                }
            },
        )
        .cloned()
        .collect()
}

/// `git status --short --branch` of `cwd`, capped; `None` outside a repository or on failure.
pub async fn git_status(cwd: &Path) -> Option<String> {
    // No optional locks: a read must not refresh the index of a checkout another process uses.
    let run = crate::worktree::git(cwd, &["--no-optional-locks", "status", "--short", "--branch"]);
    let text = tokio::time::timeout(Duration::from_secs(10), run)
        .await
        .ok()?
        .ok()?;
    let lines: Vec<&str> = text.lines().collect();
    let mut shown = lines[..lines.len().min(STATUS_LINES)].join("\n");
    if lines.len() > STATUS_LINES {
        shown.push_str(&format!("\n... {} more", lines.len() - STATUS_LINES));
    }
    Some(shown)
}

/// The restart notice: what was cut short and, when `cwd` is a checkout, its status.
pub fn notice(in_flight: &[ToolUseBlock], cwd: &Path, git_status: Option<&str>) -> String {
    let mut text = format!("{NOTICE_PREFIX} and your conversation restored.");
    if in_flight.is_empty() {
        text.push_str("\nNo tool call was in flight.");
    } else {
        text.push_str("\nInterrupted before finishing (any effects may be partial):");
        for call in in_flight {
            let input = crate::text::clip(&call.input.to_string(), INPUT_CHARS);
            text.push_str(&format!("\n- {} {input}", call.name));
        }
    }
    if let Some(status) = git_status {
        text.push_str(&format!(
            "\n`git status --short --branch` in {}:\n{status}",
            cwd.display()
        ));
    }
    text.push(']');
    text
}

/// Add `notice` to the end of a restored conversation without breaking role alternation:
/// answer dangling tool calls (with errors) and put the notice in the same user message, or
/// fold it into a trailing user message. Returns `false` when the conversation ends with a plain
/// assistant message (or is empty), so the notice can be sent as normal input.
pub fn append_notice(messages: &mut Vec<Message>, notice: &str) -> bool {
    let Some(last) = messages.last_mut() else {
        return false;
    };
    let notice = ContentBlock::Text(notice.to_owned());
    if last.role == Role::User {
        last.content.push(notice);
        return true;
    }
    let mut content: Vec<ContentBlock> = last
        .tool_uses()
        .map(|call| {
            ContentBlock::ToolResult(ToolResultBlock::error(
                call.tool_use_id.clone(),
                "Interrupted by a swarm restart.",
            ))
        })
        .collect();
    if content.is_empty() {
        return false;
    }
    content.push(notice);
    messages.push(Message::new(Role::User, content));
    true
}

/// A failed model call leaves the user turn open (the SDK closes it only on cancel): the
/// conversation ends with the turn's input or tool results and no reply. Fold the next `input`
/// into that message, as Bedrock rejects two user messages in a row; the agent then continues
/// from the conversation ([`InvokeInput::None`]).
pub fn fold_into_open_turn(messages: &mut [Message], input: InvokeInput) -> InvokeInput {
    let Some(last) = messages.last_mut().filter(|last| last.role == Role::User) else {
        return input;
    };
    match input {
        InvokeInput::Text(text) => last.content.push(ContentBlock::Text(text)),
        InvokeInput::Blocks(blocks) => last.content.extend(blocks),
        other => return other,
    }
    InvokeInput::None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use strands_agents::types::ToolResultContent;

    fn calls(ids: &[&str]) -> Message {
        Message::new(
            Role::Assistant,
            ids.iter()
                .map(|id| {
                    ContentBlock::ToolUse(ToolUseBlock::new(
                        "shell",
                        *id,
                        json!({"command": format!("run {id}")}),
                    ))
                })
                .collect(),
        )
    }

    fn results(results: Vec<ToolResultBlock>) -> Message {
        Message::new(
            Role::User,
            results.into_iter().map(ContentBlock::ToolResult).collect(),
        )
    }

    fn ok(id: &str) -> ToolResultBlock {
        ToolResultBlock::success(id, vec![ToolResultContent::Text("done".into())])
    }

    fn ids(calls: Vec<ToolUseBlock>) -> Vec<String> {
        calls.into_iter().map(|c| c.tool_use_id).collect()
    }

    #[test]
    fn in_flight_calls_are_the_unanswered_and_cancelled_ones_of_the_last_assistant_message() {
        let dangling = vec![Message::user("go"), calls(&["t1"])];
        assert_eq!(ids(in_flight(&dangling)), ["t1"]);

        let cancelled = vec![
            Message::user("go"),
            calls(&["t1", "t2"]),
            results(vec![
                ok("t1"),
                ToolResultBlock::error("t2", EXECUTION_CANCELLED_MESSAGE),
            ]),
        ];
        assert_eq!(ids(in_flight(&cancelled)), ["t2"]);

        let completed = vec![Message::user("go"), calls(&["t1"]), results(vec![ok("t1")])];
        assert!(in_flight(&completed).is_empty(), "cut off during the model call");
        let failed = vec![
            Message::user("go"),
            calls(&["t1"]),
            results(vec![ToolResultBlock::error("t1", "exit 1")]),
        ];
        assert!(in_flight(&failed).is_empty(), "a real failure completed");
        assert!(in_flight(&[Message::user("go"), Message::assistant("done")]).is_empty());
    }

    #[test]
    fn the_notice_lists_cut_short_calls_with_capped_input_and_the_checkout_status() {
        let long = ToolUseBlock::new("write", "t2", json!({"content": "x".repeat(1_000)}));
        let text = notice(
            &[calls(&["t1"]).tool_uses().next().unwrap().clone(), long],
            Path::new("/work"),
            Some("## swarm/a1\n M src/lib.rs"),
        );
        assert!(text.contains("\n- shell {\"command\":\"run t1\"}"), "{text}");
        assert!(text.contains("- write {\"content\":\"xxx"), "{text}");
        assert!(text.len() < 600, "{} chars", text.len());
        assert!(text.ends_with("in /work:\n## swarm/a1\n M src/lib.rs]"), "{text}");

        assert!(text.starts_with(NOTICE_PREFIX));
        let idle = notice(&[], Path::new("/work"), None);
        assert!(
            idle.contains("No tool call was in flight.") && !idle.contains("git"),
            "{idle}"
        );
    }

    #[tokio::test]
    async fn git_status_reports_the_branch_and_changes_and_nothing_outside_a_repo() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(git_status(dir.path()).await, None);
        crate::worktree::test_git(dir.path(), &["init", "-q", "-b", "main"]);
        std::fs::write(dir.path().join("new.txt"), "x").unwrap();
        let status = git_status(dir.path()).await.unwrap();
        assert!(status.starts_with("## ") && status.contains("main"), "{status}");
        assert!(status.contains("?? new.txt"), "{status}");
    }

    #[test]
    fn the_notice_answers_dangling_calls_or_joins_a_trailing_user_message() {
        let mut dangling = vec![Message::user("go"), calls(&["t1"])];
        assert!(append_notice(&mut dangling, "N"));
        assert_eq!(dangling.len(), 3);
        assert_eq!(dangling[2].tool_results().count(), 1);
        assert_eq!(dangling[2].content.last(), Some(&ContentBlock::text("N")));

        let mut answered = vec![Message::user("go"), calls(&["t1"]), results(vec![ok("t1")])];
        assert!(append_notice(&mut answered, "N"));
        assert_eq!(answered.len(), 3);

        let mut ended = vec![Message::user("go"), Message::assistant("done")];
        assert!(!append_notice(&mut ended, "N"));
        assert_eq!(ended.len(), 2);
    }

    #[test]
    fn input_after_a_failed_call_joins_the_open_user_turn() {
        let mut open = vec![Message::user("go"), calls(&["t1"]), results(vec![ok("t1")])];
        let input = fold_into_open_turn(&mut open, InvokeInput::Text("again".into()));
        assert!(matches!(input, InvokeInput::None));
        assert_eq!(open.len(), 3);
        assert!(matches!(open[2].content.last(), Some(ContentBlock::Text(t)) if t == "again"));

        let mut closed = vec![Message::user("go"), Message::assistant("done")];
        let input = fold_into_open_turn(&mut closed, InvokeInput::Text("next".into()));
        assert!(matches!(input, InvokeInput::Text(t) if t == "next"));
        assert_eq!(closed.len(), 2);
    }
}
