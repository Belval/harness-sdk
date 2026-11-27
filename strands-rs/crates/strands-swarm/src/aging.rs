//! Old mail and replies: a coordinator's context is mostly agent mail and its own reports, not tool
//! output. Over a day of a 30-child coordinator's history, 41% of its context was its own
//! replies (2.4k characters each, half of each restating the last ten), 37% was mail from its
//! children and 5% tool results; it compacted every 1-2 hours. Mail a few turns old has been
//! acted on, yet every later call re-sends all of it.
//!
//! So once [`KEEP_TURNS`] newer turns exist, a long agent mail body or own reply (over
//! [`AGE_ABOVE_CHARS`]) keeps its head ([`HEAD_CHARS`]) and a `retrieve_context` reference to
//! the whole text in the stash. The user's mail is never cut. Like trimmed tool results, this
//! runs only when [`crate::pruning::CachePrune`] prunes, so the cached prefix is not rewritten
//! for it alone.

use strands_agents::context_manager::stash::Stash;
use strands_agents::types::{ContentBlock, JsonValue, Message, Role, ToolResultContent};

use crate::runtime::USER;

/// The latest this many turns keep their mail and replies whole.
pub const KEEP_TURNS: usize = 20;
/// Older mail bodies and replies longer than this (in characters) are cut.
pub const AGE_ABOVE_CHARS: usize = 1_500;
/// What a cut text keeps of its head, in characters.
pub const HEAD_CHARS: usize = 600;

const OPEN: &str = "<message from=\"";
const CLOSE: &str = "\n</message>";

/// One text to replace: where it is, its stash key and what it becomes.
struct Cut {
    message: usize,
    block: usize,
    /// The item of a tool result block; `None` for a text block.
    item: Option<usize>,
    key: String,
    original: String,
    aged: String,
}

/// Where the old turns end: the start of the [`KEEP_TURNS`]-th latest turn (0 when there are
/// not that many). `turn_start` tells a turn's first message.
fn old_end(messages: &[Message], turn_start: impl Fn(&Message) -> bool) -> usize {
    messages
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, m)| turn_start(m))
        .nth(KEEP_TURNS - 1)
        .map_or(0, |(i, _)| i)
}

/// `text` cut to about `chars` characters, on a line or word end when one is near.
pub(crate) fn head(text: &str, chars: usize) -> &str {
    let end = text.char_indices().nth(chars).map_or(text.len(), |(at, _)| at);
    let cut = &text[..end];
    let near = |at: &usize| *at >= end * 2 / 3;
    match cut
        .rfind('\n')
        .filter(near)
        .or_else(|| cut.rfind(' ').filter(near))
    {
        Some(at) => &cut[..at],
        None => cut,
    }
}

/// `text`'s head and a reference to all of it under `key`, when it is over [`AGE_ABOVE_CHARS`].
fn shorten(text: &str, key: &str) -> Option<String> {
    let total = text.chars().count();
    if total <= AGE_ABOVE_CHARS {
        return None;
    }
    let kept = head(text, HEAD_CHARS);
    let more = total - kept.chars().count();
    Some(format!("{kept}\n[Stashed: {more} more chars] [ref: {key}]"))
}

/// `text` with each long agent mail body in it shortened ([`shorten`]); `None` when none is.
/// Mail bodies are [defanged](crate::text::defang), so none holds a closing tag of its own.
fn age_mail(text: &str, key: &str) -> Option<String> {
    let user = format!("{OPEN}{USER}\"");
    let mut out = String::new();
    let mut rest = text;
    let mut changed = false;
    while let Some(at) = rest.find(OPEN) {
        let Some(body) = rest[at..].find(">\n").map(|end| at + end + 2) else {
            break;
        };
        let Some(close) = rest[body..].find(CLOSE).map(|end| body + end) else {
            break;
        };
        out.push_str(&rest[..body]);
        let short = (!rest[at..].starts_with(&user))
            .then(|| shorten(&rest[body..close], key))
            .flatten();
        changed |= short.is_some();
        out.push_str(short.as_deref().unwrap_or(&rest[body..close]));
        rest = &rest[close..];
    }
    out.push_str(rest);
    changed.then_some(out)
}

/// What aging the turns before [`old_end`] would cut. Keys end in `-aged`, apart from the SDK's
/// own `<id>_<index>` (a summary stashes the message it replaces under those).
fn cuts(messages: &[Message], turn_start: impl Fn(&Message) -> bool) -> Vec<Cut> {
    let mut cuts = Vec::new();
    for (m, message) in messages[..old_end(messages, turn_start)].iter().enumerate() {
        for (b, block) in message.content.iter().enumerate() {
            let mut cut = |item: Option<usize>, id: &str, text: &str, mail: bool| {
                let key = format!("{id}-aged_{}", item.unwrap_or(b));
                let aged = match mail {
                    true => age_mail(text, &key),
                    false => shorten(text, &key),
                };
                if let Some(aged) = aged {
                    let original = text.to_owned();
                    cuts.push(Cut {
                        message: m,
                        block: b,
                        item,
                        key,
                        original,
                        aged,
                    });
                }
            };
            match block {
                ContentBlock::Text(text) => cut(None, &message.tracking_id, text, message.role == Role::User),
                ContentBlock::ToolResult(result) => {
                    for (i, item) in result.content.iter().enumerate() {
                        if let ToolResultContent::Text(text) = item {
                            cut(Some(i), &result.tool_use_id, text, true);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    cuts
}

/// The characters aging would remove now.
pub(crate) fn removable(messages: &[Message], turn_start: impl Fn(&Message) -> bool) -> usize {
    cuts(messages, turn_start)
        .iter()
        .map(|cut| cut.original.len().saturating_sub(cut.aged.len()))
        .sum()
}

/// Ages the old turns' long mail and replies, stashing each whole text first (one that cannot be
/// stashed stays whole); `true` when any was cut.
pub(crate) async fn age(
    messages: &mut [Message],
    turn_start: impl Fn(&Message) -> bool,
    stash: &Stash,
) -> bool {
    let mut acted = false;
    for cut in cuts(messages, turn_start) {
        let (id, index) = cut.key.rsplit_once('_').expect("a cut's key ends in _<index>");
        let index = index.parse().expect("a cut's key ends in _<index>");
        if stash
            .store(id, index, &JsonValue::String(cut.original))
            .await
            .is_err()
        {
            continue;
        }
        let text = match (&mut messages[cut.message].content[cut.block], cut.item) {
            (ContentBlock::Text(text), None) => text,
            (ContentBlock::ToolResult(result), Some(i)) => match &mut result.content[i] {
                ToolResultContent::Text(text) => text,
                _ => continue,
            },
            _ => continue,
        };
        *text = cut.aged;
        acted = true;
    }
    acted
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use strands_agents::storage::InMemoryStorage;
    use strands_agents::types::ToolResultBlock;

    fn mail(from: &str, kind: &str, body: &str) -> String {
        format!("{OPEN}{from}\" kind=\"{kind}\">\n{body}\n</message>")
    }

    fn turn_start(m: &Message) -> bool {
        m.role == Role::User && m.tool_results().next().is_none()
    }

    /// `turns` turns, each a mail from a child and a user message, then a long reply.
    fn history(turns: usize, body: &str) -> Vec<Message> {
        let mut messages = Vec::new();
        for t in 0..turns {
            let text = [mail("parser (a2)", "reply", body), mail(USER, "message", body)].join("\n\n");
            messages.push(Message::user(text).with_tracking_id(format!("u{t}")));
            messages.push(Message::assistant(body).with_tracking_id(format!("r{t}")));
        }
        messages
    }

    fn text(message: &Message) -> &str {
        match &message.content[0] {
            ContentBlock::Text(text) => text,
            _ => panic!("not text"),
        }
    }

    fn stash() -> Stash {
        Stash::new(Arc::new(InMemoryStorage::new()), "s", "a1", false)
    }

    #[tokio::test]
    async fn old_long_agent_mail_and_replies_keep_a_head_and_a_reference() {
        let body = "word ".repeat(1_000);
        let mut messages = history(KEEP_TURNS + 2, &body);
        let stash = stash();

        assert!(removable(&messages, turn_start) > 0);
        assert!(age(&mut messages, turn_start, &stash).await);

        let turn = text(&messages[0]);
        let child = &turn[..turn.find(&format!("{OPEN}{USER}")).unwrap()];
        assert!(child.len() < HEAD_CHARS + 200, "{child}");
        assert!(
            child.contains("more chars] [ref: u0-aged_0]\n</message>"),
            "{child}"
        );
        assert!(
            turn.ends_with(&mail(USER, "message", &body)),
            "the user's mail stays whole"
        );
        let reply = text(&messages[1]);
        assert!(reply.ends_with("more chars] [ref: r0-aged_0]"), "{reply}");
        let whole = stash.retrieve("u0-aged_0").await.unwrap();
        assert!(whole
            .as_str()
            .unwrap()
            .contains(&mail("parser (a2)", "reply", &body)));
        assert_eq!(
            stash.retrieve("r0-aged_0").await.unwrap(),
            JsonValue::String(body.clone())
        );

        assert!(
            text(&messages[3]).len() < HEAD_CHARS + 200,
            "the second turn aged too"
        );
        for message in &messages[4..] {
            assert!(text(message).len() >= body.len(), "the latest turns stay whole");
        }
        assert_eq!(removable(&messages, turn_start), 0);
        assert!(
            !age(&mut messages, turn_start, &stash).await,
            "nothing left to age"
        );
    }

    #[tokio::test]
    async fn short_texts_and_recent_turns_stay_whole() {
        let mut short = history(KEEP_TURNS + 2, &"w".repeat(AGE_ABOVE_CHARS));
        let mut recent = history(KEEP_TURNS, &"w".repeat(5_000));
        assert!(!age(&mut short, turn_start, &stash()).await);
        assert!(!age(&mut recent, turn_start, &stash()).await);
    }

    #[tokio::test]
    async fn mail_appended_to_a_tool_result_ages() {
        let body = "line\n".repeat(1_000);
        let arrived = format!(
            "done\n\n[New messages arrived while this tool ran]\n{}",
            mail("fuzz (a3)", "fyi", &body)
        );
        let result = ToolResultBlock::success("t1", vec![ToolResultContent::Text(arrived)]);
        let mut messages = vec![
            Message::user("go"),
            Message::new(Role::User, vec![ContentBlock::ToolResult(result)]),
        ];
        messages.extend(history(KEEP_TURNS, "ok"));

        assert!(age(&mut messages, turn_start, &stash()).await);

        let ContentBlock::ToolResult(result) = &messages[1].content[0] else {
            panic!()
        };
        let ToolResultContent::Text(appended) = &result.content[0] else {
            panic!()
        };
        assert!(appended
            .starts_with("done\n\n[New messages arrived while this tool ran]\n<message from=\"fuzz (a3)\""));
        assert!(appended.ends_with("[ref: t1-aged_0]\n</message>"), "{appended}");
        assert_eq!(text(&messages[0]), "go", "text that is not mail stays");
    }

    #[test]
    fn a_head_ends_on_a_line_or_word_and_on_a_char_boundary() {
        assert_eq!(head("one two three", 9), "one two");
        assert_eq!(head("one\ntwo three", 11), "one\ntwo");
        assert_eq!(head(&"é".repeat(10), 4), "éééé");
        assert_eq!(head("short", 600), "short");
    }
}
