//! Old tool-call inputs: what the model sent a tool stays in its context as long as the result.
//! On 113 live agents' saved conversations, call inputs (heredoc scripts and long commands
//! most) were 23% of the bytes, near the 29% of `shell` results, yet pruning trimmed only
//! results. A call that ran many steps ago has done its work: a script's output is in its
//! result, a `write`'s or `edit`'s text is on disk, a `send_message` was delivered, a
//! `spawn_agent` task is with the child.
//!
//! So once [`KEEP_RECENT_CALLS`] newer assistant messages with tool calls exist, each string in
//! a call's input longer than [`TRIM_ABOVE_CHARS`] keeps its head ([`HEAD_CHARS`]) and a
//! `retrieve_context` reference to the whole string in the stash. The input stays an object with
//! the same keys, other values untouched. No tool is exempt: the head says what the call did and
//! the stash holds the rest. Like trimmed results and [`crate::aging`], this runs only when
//! [`crate::pruning::CachePrune`] prunes, and only with a stash.
//!
//! An old `todo_write` that a later one replaced goes whole, short items included: each call
//! re-sends the full list, and the current one is in the agent's reminders on every call. On a
//! live coordinator's history, 42 such lists were 10% of the bytes, none over the string limit.

use strands_agents::context_manager::stash::Stash;
use strands_agents::types::{ContentBlock, JsonValue, Message, Role};

/// The latest this many assistant messages with tool calls keep their inputs whole (as many as
/// the tool results [`crate::pruning`] keeps whole).
pub const KEEP_RECENT_CALLS: usize = 6;
/// Older input strings longer than this (in characters) are trimmed: half a result's threshold.
/// A result's tail holds the error or the verdict; an input is the model's own, its head says
/// what it did, and its effect is in the result. Replayed on the 113 live agents' histories, a
/// prune removes 9.3% of the history at 1,000 and 400 characters, 5.3% at a result's 2,000 and
/// 800 (most long inputs are 1-2k characters).
pub const TRIM_ABOVE_CHARS: usize = 1_000;
/// What a trimmed string keeps of its head, in characters.
pub const HEAD_CHARS: usize = 400;
/// The tool whose every call replaces the list the previous call set.
const TODO_WRITE: &str = "todo_write";

/// One call input to replace: where it is, its trimmed form, and the strings it cut, in the
/// order of their stash indices.
struct Cut {
    message: usize,
    block: usize,
    id: String,
    input: JsonValue,
    originals: Vec<String>,
}

/// The stash id of `tool_use_id`'s input strings (its result's are `<tool_use_id>_<index>`).
fn stash_id(tool_use_id: &str) -> String {
    format!("{tool_use_id}-input")
}

/// Trims every long string in `value`, pushing each original to `originals`.
fn trim(value: &mut JsonValue, id: &str, originals: &mut Vec<String>) {
    match value {
        JsonValue::String(text) => {
            let total = text.chars().count();
            if total <= TRIM_ABOVE_CHARS {
                return;
            }
            let kept = crate::aging::head(text, HEAD_CHARS);
            let more = total - kept.chars().count();
            let key = format!("{id}_{}", originals.len());
            let short = format!(
                "{kept}\n… [{more} chars of this old call's input trimmed; it ran as sent] [ref: {key}]"
            );
            originals.push(std::mem::replace(text, short));
        }
        JsonValue::Array(items) => items.iter_mut().for_each(|item| trim(item, id, originals)),
        JsonValue::Object(map) => map.values_mut().for_each(|item| trim(item, id, originals)),
        _ => {}
    }
}

/// What trimming the calls before the latest [`KEEP_RECENT_CALLS`] would cut.
fn cuts(messages: &[Message]) -> Vec<Cut> {
    let calls: Vec<usize> = (messages.iter().enumerate())
        .filter(|(_, m)| m.role == Role::Assistant && m.tool_uses().next().is_some())
        .map(|(i, _)| i)
        .collect();
    let old = calls.len().saturating_sub(KEEP_RECENT_CALLS);
    let last_list = (messages.iter().enumerate())
        .flat_map(|(m, message)| message.tool_uses().map(move |call| (m, call)))
        .filter(|(_, call)| call.name == TODO_WRITE)
        .map(|(_, call)| call.tool_use_id.clone())
        .last();
    let mut cuts = Vec::new();
    for &m in &calls[..old] {
        for (b, block) in messages[m].content.iter().enumerate() {
            let ContentBlock::ToolUse(call) = block else {
                continue;
            };
            let id = stash_id(&call.tool_use_id);
            let mut input = call.input.clone();
            let mut originals = Vec::new();
            let superseded = call.name == TODO_WRITE
                && last_list.as_ref().is_some_and(|last| *last != call.tool_use_id)
                && !is_superseded_stub(&input);
            let stub = serde_json::json!({ "todos": format!("{SUPERSEDED} [ref: {id}_0]") });
            if superseded && json_len(&stub) < json_len(&input) {
                originals.push(serde_json::to_string(&input).unwrap_or_default());
                input = stub;
            } else {
                trim(&mut input, &id, &mut originals);
            }
            if !originals.is_empty() {
                cuts.push(Cut {
                    message: m,
                    block: b,
                    id,
                    input,
                    originals,
                });
            }
        }
    }
    cuts
}

/// What stands in for a superseded `todo_write`'s list.
const SUPERSEDED: &str = "[an earlier todo list, replaced by a later todo_write]";

/// Whether `input` is already a superseded list's stand-in.
fn is_superseded_stub(input: &JsonValue) -> bool {
    input["todos"]
        .as_str()
        .is_some_and(|todos| todos.starts_with(SUPERSEDED))
}

fn json_len(value: &JsonValue) -> usize {
    serde_json::to_string(value).map_or(0, |json| json.len())
}

/// The characters (of JSON) trimming would remove now.
pub(crate) fn removable(messages: &[Message]) -> usize {
    cuts(messages)
        .iter()
        .map(|cut| {
            let ContentBlock::ToolUse(call) = &messages[cut.message].content[cut.block] else {
                return 0;
            };
            json_len(&call.input).saturating_sub(json_len(&cut.input))
        })
        .sum()
}

/// Trims the old calls' long input strings, stashing each whole string first (a call whose
/// strings cannot all be stashed stays whole); `true` when any was trimmed.
pub(crate) async fn trim_old(messages: &mut [Message], stash: &Stash) -> bool {
    let mut acted = false;
    'cuts: for cut in cuts(messages) {
        for (index, original) in cut.originals.into_iter().enumerate() {
            if stash
                .store(&cut.id, index, &JsonValue::String(original))
                .await
                .is_err()
            {
                continue 'cuts;
            }
        }
        if let ContentBlock::ToolUse(call) = &mut messages[cut.message].content[cut.block] {
            call.input = cut.input;
            acted = true;
        }
    }
    acted
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;
    use strands_agents::storage::InMemoryStorage;
    use strands_agents::types::{ToolResultBlock, ToolResultContent, ToolUseBlock};

    fn stash() -> Stash {
        Stash::new(Arc::new(InMemoryStorage::new()), "s", "a1", false)
    }

    /// `calls` steps, each a call to `tool` with `input` and its result.
    fn history(calls: usize, tool: &str, input: JsonValue) -> Vec<Message> {
        let mut messages = vec![Message::user("go")];
        for c in 0..calls {
            let id = format!("t{c}");
            let call = ToolUseBlock::new(tool, &id, input.clone());
            messages.push(Message::new(Role::Assistant, vec![ContentBlock::ToolUse(call)]));
            let result = ToolResultBlock::success(&id, vec![ToolResultContent::Text("ok".into())]);
            messages.push(Message::new(Role::User, vec![ContentBlock::ToolResult(result)]));
        }
        messages
    }

    fn input(message: &Message) -> &JsonValue {
        &message.tool_uses().next().unwrap().input
    }

    fn script() -> String {
        format!(
            "cat > x.py <<'EOF'\n{}EOF\npython3 x.py",
            "print('hello world')\n".repeat(200)
        )
    }

    #[tokio::test]
    async fn an_old_long_command_keeps_its_head_its_keys_and_a_reference() {
        let original = json!({"command": script(), "timeout": 60});
        let mut messages = history(KEEP_RECENT_CALLS + 1, "shell", original.clone());
        let stash = stash();

        assert!(trim_old(&mut messages, &stash).await);

        let trimmed = input(&messages[1]);
        let object = trimmed.as_object().expect("still an object");
        assert_eq!(object.keys().collect::<Vec<_>>(), ["command", "timeout"]);
        assert_eq!(trimmed["timeout"], 60, "other values stay as they are");
        let command = trimmed["command"].as_str().unwrap();
        assert!(command.starts_with("cat > x.py <<'EOF'\nprint('hello world')"));
        assert!(command.len() < HEAD_CHARS + 120, "{command}");
        assert!(
            command.ends_with("chars of this old call's input trimmed; it ran as sent] [ref: t0-input_0]"),
            "{command}"
        );
        assert_eq!(
            stash.retrieve("t0-input_0").await,
            Some(original["command"].clone()),
            "the stash holds the whole command"
        );
        for message in messages[3..].iter().filter(|m| m.role == Role::Assistant) {
            assert_eq!(input(message), &original, "the latest calls stay whole");
        }
    }

    #[tokio::test]
    async fn nested_strings_are_trimmed_and_short_ones_kept() {
        let long = "x".repeat(TRIM_ABOVE_CHARS + 1);
        let original = json!({"to": "a2", "items": [{"text": long, "done": false}], "n": null});
        let mut messages = history(KEEP_RECENT_CALLS + 1, "send_message", original);

        assert!(trim_old(&mut messages, &stash()).await);

        let trimmed = input(&messages[1]);
        assert_eq!(trimmed["to"], "a2");
        assert_eq!(trimmed["items"][0]["done"], false);
        assert_eq!(trimmed["n"], JsonValue::Null);
        let text = trimmed["items"][0]["text"].as_str().unwrap();
        assert!(text.ends_with("[ref: t0-input_0]"), "{text}");
    }

    #[tokio::test]
    async fn counted_as_removable_then_not_trimmed_again() {
        let mut messages = history(
            KEEP_RECENT_CALLS + 2,
            "write",
            json!({"path": "a", "content": script()}),
        );
        let before = removable(&messages);
        assert!(before > 2 * (script().len() - HEAD_CHARS - 120), "{before}");

        assert!(trim_old(&mut messages, &stash()).await);
        let once = messages.clone();

        assert_eq!(removable(&messages), 0, "a trimmed input is not removable again");
        assert!(!trim_old(&mut messages, &stash()).await);
        assert_eq!(messages, once);
    }

    #[tokio::test]
    async fn recent_and_short_calls_stay_whole() {
        let recent = history(KEEP_RECENT_CALLS, "shell", json!({"command": script()}));
        let short = history(KEEP_RECENT_CALLS + 3, "shell", json!({"command": "ls"}));
        for mut messages in [recent, short] {
            let whole = messages.clone();
            assert_eq!(removable(&messages), 0);
            assert!(!trim_old(&mut messages, &stash()).await);
            assert_eq!(messages, whole);
        }
    }

    #[tokio::test]
    async fn an_old_todo_list_a_later_one_replaced_goes_whole() {
        let item = json!({"content": "Land it", "activeForm": "Landing it", "status": "pending"});
        let list = json!({"todos": vec![item; 4]});
        let mut messages = history(KEEP_RECENT_CALLS + 2, "todo_write", list.clone());
        let stash = stash();
        assert!(removable(&messages) > 0, "short items count too");

        assert!(trim_old(&mut messages, &stash).await);

        let stub = input(&messages[1])["todos"].as_str().unwrap();
        assert_eq!(stub, format!("{SUPERSEDED} [ref: t0-input_0]"));
        let whole = stash.retrieve("t0-input_0").await.unwrap();
        assert_eq!(
            serde_json::from_str::<JsonValue>(whole.as_str().unwrap()).unwrap(),
            list
        );
        assert_eq!(input(&messages[5]), &list, "the recent ones stay whole");
        assert_eq!(removable(&messages), 0, "a stand-in is not replaced again");
        assert!(!trim_old(&mut messages, &stash).await);
    }

    #[tokio::test]
    async fn the_latest_todo_list_stays_whole_however_old() {
        let item = json!({"content": "Land it", "activeForm": "Landing it", "status": "pending"});
        let list = json!({"todos": vec![item; 4]});
        let mut messages = history(1, "todo_write", list.clone());
        let rest = history(KEEP_RECENT_CALLS + 1, "shell", json!({"command": "ls"}));
        messages.extend(rest.into_iter().skip(1));

        assert!(!trim_old(&mut messages, &stash()).await);
        assert_eq!(input(&messages[1]), &list);
    }
}
