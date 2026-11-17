//! The notice a restored agent resumes with: the facts it would otherwise re-derive by re-reading
//! its work after a restart. That covers the tool calls the restart cut short and the state of its
//! checkout. Its todo list needs no copy here, because the harness re-surfaces it before every
//! model call.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use strands_agents::agent::CANCELLED_MESSAGE;
use strands_agents::tools::EXECUTION_CANCELLED_MESSAGE;
use strands_agents::types::{ContentBlock, Message, Role, ToolResultBlock, ToolResultStatus, ToolUseBlock};
use strands_agents::InvokeInput;

/// How every restart notice starts, so context strategies can tell it from a new task.
pub const NOTICE_PREFIX: &str = "[The swarm was restarted";
/// What a turn the restart cut ends with, in place of the SDK's [`CANCELLED_MESSAGE`], which
/// reads as if the user had stopped it.
pub const RESTARTED_MESSAGE: &str = "Interrupted: the swarm restarted.";
/// Longest tool input shown, in characters.
const INPUT_CHARS: usize = 300;
/// Most `git status` lines shown.
const STATUS_LINES: usize = 30;
/// A cut call that had run this long gets the hint to run slow commands as background jobs.
const SLOW_CALL_SECS: u64 = 60;
/// Said when a slow foreground call was cut: the restart would not have cost a job anything.
const SLOW_CALL_HINT: &str = "Foreground calls hold up a hot reload, which cuts them after at most 10 \
    minutes, and die with any restart. Background jobs survive both: for anything slow, use \
    run_background and wait for its message instead of re-running it in the foreground or polling.";
/// Said when the restart cut a `wait`. Agents whose long waits were cut by reloads took to
/// polling on a 10-minute schedule, each poll rewriting their whole cached context.
const WAIT_CUT: &str = "cut short, but nothing it waited for is lost: background jobs run on, and \
    their ends and any mail still reach you. Call `wait` again; no need to poll.";

/// The calls of the last assistant message that never completed: unanswered, or cancelled. The
/// SDK's closing message of a cancelled turn ([`CANCELLED_MESSAGE`]) does not count as the last.
pub fn in_flight(messages: &[Message]) -> Vec<ToolUseBlock> {
    let mut end = messages.len();
    if messages.last().is_some_and(is_cancel_marker) {
        end -= 1;
    }
    let messages = &messages[..end];
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
                    result.status == ToolResultStatus::Error
                        && (result.text() == EXECUTION_CANCELLED_MESSAGE
                            || crate::shell_text::is_cancelled(&result.text()))
                }
            },
        )
        .cloned()
        .collect()
}

/// The assistant message the SDK closes a cancelled turn with, reworded or not.
fn is_cancel_marker(message: &Message) -> bool {
    message.role == Role::Assistant
        && matches!(message.content.as_slice(), [ContentBlock::Text(text)]
            if text == CANCELLED_MESSAGE || text == RESTARTED_MESSAGE)
}

/// Reword the closing message of a turn the restart cut ([`RESTARTED_MESSAGE`]). It ends the
/// conversation and was never sent, so the cached prefix is unchanged.
pub fn reword_cancel(messages: &mut [Message]) {
    if let Some(last) = messages.last_mut().filter(|last| is_cancel_marker(last)) {
        last.content = vec![ContentBlock::text(RESTARTED_MESSAGE)];
    }
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

/// The restart notice: what was cut short (with how long it had run, from `ran`: seconds by
/// tool use id) and, when `cwd` is a checkout, its status.
pub fn notice(
    in_flight: &[ToolUseBlock],
    ran: &BTreeMap<String, u64>,
    cwd: &Path,
    git_status: Option<&str>,
) -> String {
    let mut text = format!("{NOTICE_PREFIX} and your conversation restored.");
    // A `wait` has no effects to be partial; what it waited for still arrives.
    let (waits, cut): (Vec<&ToolUseBlock>, Vec<&ToolUseBlock>) =
        in_flight.iter().partition(|call| call.name == "wait");
    if in_flight.is_empty() {
        text.push_str("\nNo tool call was in flight.");
    } else if !cut.is_empty() {
        text.push_str("\nInterrupted before finishing (any effects may be partial):");
        let mut slow = false;
        for call in cut {
            let input = crate::text::clip(&call.input.to_string(), INPUT_CHARS);
            let secs = ran.get(&call.tool_use_id).copied();
            slow |= secs.is_some_and(|secs| secs >= SLOW_CALL_SECS);
            let after = secs.map_or(String::new(), |secs| format!(" (cut after {})", duration(secs)));
            text.push_str(&format!("\n- {}{after} {input}", call.name));
        }
        if slow {
            text.push('\n');
            text.push_str(SLOW_CALL_HINT);
        }
    }
    if let Some(wait) = waits.first() {
        let input = crate::text::clip(&wait.input.to_string(), INPUT_CHARS);
        text.push_str(&format!("\n- wait {input}: {WAIT_CUT}"));
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

/// `secs` as `45s`, `9m 12s` or `1h 3m`.
fn duration(secs: u64) -> String {
    match secs {
        0..60 => format!("{secs}s"),
        60..3600 => format!("{}m {}s", secs / 60, secs % 60),
        _ => format!("{}h {}m", secs / 3600, secs % 3600 / 60),
    }
}

/// Add `notice` to the end of a restored conversation without breaking role alternation:
/// answer dangling tool calls (with errors) and put the notice in the same user message, or
/// fold it into a trailing user message. Returns `false` when the conversation ends with a plain
/// assistant message (or is empty), so the notice can be sent as normal input.
pub fn append_notice(messages: &mut Vec<Message>, notice: &str) -> bool {
    let notice = match fold_into_open_turn(messages, InvokeInput::Text(notice.to_owned())) {
        InvokeInput::Text(notice) => ContentBlock::Text(notice),
        _ => return true,
    };
    let Some(last) = messages.last() else {
        return false;
    };
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

/// File in an agent's session directory that holds its [`OpenBatch`].
pub const OPEN_BATCH_FILE: &str = "open-batch.json";

/// A batch of tool calls in progress. The SDK adds a batch to the conversation only once every
/// call in it has finished, so a hard crash (kill -9, OOM kill, power loss) would lose the calls
/// and the results already in: the agent would not know it had started a job or spawned an
/// agent. Written ahead while the batch runs, it puts them back on resume ([`restore_batch`]).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenBatch {
    /// The assistant message with the calls.
    pub message: Message,
    /// Results of the calls that finished, in the order they did.
    pub results: Vec<ToolResultBlock>,
}

impl OpenBatch {
    /// Write `batch` to `path`, or remove the file for `None`. Synced like `swarm.json`, so the
    /// batch survives a power loss, which is what it is written ahead for.
    pub fn save(path: &std::path::Path, batch: Option<&OpenBatch>) -> std::io::Result<()> {
        match batch {
            Some(open) => crate::files::replace(path, &serde_json::to_vec(open)?, true),
            None => std::fs::remove_file(path).or_else(|e| match e.kind() {
                std::io::ErrorKind::NotFound => Ok(()),
                _ => Err(e),
            }),
        }
    }

    /// Whether a `shell` call of the batch had not finished.
    pub fn cut_shell(&self) -> bool {
        self.message
            .tool_uses()
            .any(|call| call.name == "shell" && self.result(call).is_none())
    }

    fn result(&self, call: &ToolUseBlock) -> Option<&ToolResultBlock> {
        self.results
            .iter()
            .find(|result| result.tool_use_id == call.tool_use_id)
    }
}

/// Put a batch a hard crash cut short back at the end of `messages`: its calls, the results that
/// came in, and the SDK's cancellation for the rest, so [`in_flight`] lists those. Returns whether
/// anything changed: nothing does when the conversation already holds the batch's results.
pub fn restore_batch(messages: &mut Vec<Message>, batch: OpenBatch) -> bool {
    let calls: Vec<&ToolUseBlock> = batch.message.tool_uses().collect();
    let Some(first) = calls.first() else {
        return false;
    };
    let answered = messages
        .iter()
        .flat_map(|m| m.tool_results())
        .any(|result| calls.iter().any(|call| call.tool_use_id == result.tool_use_id));
    if answered {
        return false;
    }
    // The crash may have come between the SDK adding the calls and adding their results.
    let added = messages.last().is_some_and(|last| {
        last.role == Role::Assistant && last.tool_uses().any(|call| call.tool_use_id == first.tool_use_id)
    });
    if !added {
        if messages.last().is_some_and(|last| last.role == Role::Assistant) {
            return false;
        }
        messages.push(batch.message.clone());
    }
    let content = calls
        .iter()
        .map(|call| {
            let result = batch
                .result(call)
                .cloned()
                .unwrap_or_else(|| ToolResultBlock::error(&call.tool_use_id, EXECUTION_CANCELLED_MESSAGE));
            ContentBlock::ToolResult(result)
        })
        .collect();
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
        let mut closed = cancelled.clone();
        closed.push(Message::assistant(CANCELLED_MESSAGE));
        assert_eq!(
            ids(in_flight(&closed)),
            ["t2"],
            "the SDK's cancel marker is skipped"
        );
        reword_cancel(&mut closed);
        assert_eq!(closed.last().unwrap().text(), RESTARTED_MESSAGE);
        assert_eq!(ids(in_flight(&closed)), ["t2"], "and so is the reworded one");
        let mut done = vec![Message::user("go"), Message::assistant("done")];
        reword_cancel(&mut done);
        assert_eq!(done[1].text(), "done", "only the marker is reworded");

        let completed = vec![Message::user("go"), calls(&["t1"]), results(vec![ok("t1")])];
        assert!(in_flight(&completed).is_empty(), "cut off during the model call");
        let failed = vec![
            Message::user("go"),
            calls(&["t1"]),
            results(vec![ToolResultBlock::error("t1", "exit 1")]),
        ];
        assert!(in_flight(&failed).is_empty(), "a real failure completed");
        let killed = vec![
            Message::user("go"),
            calls(&["t1"]),
            results(vec![ToolResultBlock::error("t1", "half\n[cancelled after 2.0s]")]),
        ];
        assert_eq!(ids(in_flight(&killed)), ["t1"], "a cut shell call with output");
        assert!(in_flight(&[Message::user("go"), Message::assistant("done")]).is_empty());
    }

    #[test]
    fn the_notice_says_how_long_each_cut_call_ran_and_points_slow_ones_at_background_jobs() {
        let cut = calls(&["t1", "t2"]).tool_uses().cloned().collect::<Vec<_>>();
        let quick = notice(&cut, &BTreeMap::from([("t1".into(), 12)]), Path::new("/w"), None);
        assert!(
            quick.contains("\n- shell (cut after 12s) {\"command\":\"run t1\"}"),
            "{quick}"
        );
        assert!(quick.contains("\n- shell {\"command\":\"run t2\"}"), "{quick}");
        assert!(!quick.contains("run_background"), "{quick}");

        let ran = BTreeMap::from([("t1".into(), 12), ("t2".into(), 552)]);
        let slow = notice(&cut, &ran, Path::new("/w"), None);
        assert!(slow.contains("- shell (cut after 9m 12s)"), "{slow}");
        assert!(slow.contains(&format!("\n{SLOW_CALL_HINT}]")), "{slow}");
        assert_eq!(duration(3_780), "1h 3m");
    }

    #[test]
    fn a_cut_wait_is_not_an_effect_that_may_be_partial_and_need_not_become_polling() {
        let wait = ToolUseBlock::new("wait", "w1", json!({"from": ["jobs"], "timeout_seconds": 3600}));
        let alone = notice(
            std::slice::from_ref(&wait),
            &BTreeMap::new(),
            Path::new("/w"),
            None,
        );
        assert!(
            !alone.contains("may be partial") && !alone.contains("No tool call"),
            "{alone}"
        );
        assert!(
            alone.contains(&format!(
                "\n- wait {{\"from\":[\"jobs\"],\"timeout_seconds\":3600}}: {WAIT_CUT}]"
            )),
            "{alone}"
        );

        let shell = calls(&["t1"]).tool_uses().next().unwrap().clone();
        let both = notice(&[shell, wait], &BTreeMap::new(), Path::new("/w"), None);
        assert!(
            both.contains("may be partial):\n- shell {\"command\":\"run t1\"}\n- wait {"),
            "{both}"
        );
    }

    #[test]
    fn the_notice_lists_cut_short_calls_with_capped_input_and_the_checkout_status() {
        let long = ToolUseBlock::new("write", "t2", json!({"content": "x".repeat(1_000)}));
        let text = notice(
            &[calls(&["t1"]).tool_uses().next().unwrap().clone(), long],
            &BTreeMap::new(),
            Path::new("/work"),
            Some("## swarm/a1\n M src/lib.rs"),
        );
        assert!(text.contains("\n- shell {\"command\":\"run t1\"}"), "{text}");
        assert!(text.contains("- write {\"content\":\"xxx"), "{text}");
        assert!(text.len() < 600, "{} chars", text.len());
        assert!(text.ends_with("in /work:\n## swarm/a1\n M src/lib.rs]"), "{text}");

        assert!(text.starts_with(NOTICE_PREFIX));
        let idle = notice(&[], &BTreeMap::new(), Path::new("/work"), None);
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

    #[test]
    fn a_saved_batch_is_synced_and_removed_once_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(OPEN_BATCH_FILE);
        let batch = OpenBatch {
            message: calls(&["t1"]),
            results: Vec::new(),
        };
        let synced = || crate::files::SYNCED.with(std::cell::Cell::get);
        let before = synced();
        OpenBatch::save(&path, Some(&batch)).unwrap();
        assert_eq!(synced(), before + 1, "written durably, as for a power loss");
        let saved: OpenBatch = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved.message, batch.message);
        OpenBatch::save(&path, None).unwrap();
        assert!(!path.exists());
        OpenBatch::save(&path, None).unwrap();
    }

    #[test]
    fn a_batch_cut_by_a_hard_crash_comes_back_with_its_finished_results() {
        let batch = OpenBatch {
            message: calls(&["t1", "t2"]),
            results: vec![ok("t1")],
        };
        let message = batch.message.clone();
        // Crashed while t2 ran: the SDK had added nothing of the batch.
        let mut cut = vec![Message::user("go")];
        assert!(restore_batch(&mut cut, batch.clone()));
        assert_eq!(cut.len(), 3);
        assert_eq!(cut[1], message);
        let restored: Vec<_> = cut[2]
            .tool_results()
            .map(|r| (r.tool_use_id.clone(), r.text()))
            .collect();
        assert_eq!(
            restored,
            [
                ("t1".to_owned(), "done".to_owned()),
                ("t2".to_owned(), EXECUTION_CANCELLED_MESSAGE.to_owned())
            ]
        );
        assert_eq!(ids(in_flight(&cut)), ["t2"], "the notice lists the cut call");
        assert!(batch.cut_shell(), "t2 is a shell call that had not finished");
        let finished = OpenBatch {
            results: vec![ok("t1"), ok("t2")],
            ..batch.clone()
        };
        assert!(!finished.cut_shell());

        // Crashed between the SDK adding the calls and adding their results.
        let mut half = vec![cut[0].clone(), message.clone()];
        assert!(restore_batch(&mut half, batch.clone()));
        assert_eq!(half[..2], cut[..2]);
        assert_eq!(half[2].tool_results().count(), 2);

        // The batch was saved whole (the crash came before the file was removed).
        let mut done = vec![
            Message::user("go"),
            calls(&["t1", "t2"]),
            results(vec![ok("t1"), ok("t2")]),
        ];
        assert!(!restore_batch(&mut done, batch.clone()));
        assert_eq!(done.len(), 3);

        let mut closed = vec![Message::user("go"), Message::assistant("done")];
        assert!(
            !restore_batch(&mut closed, batch),
            "never two assistant messages in a row"
        );
        assert_eq!(closed.len(), 2);
    }
}
