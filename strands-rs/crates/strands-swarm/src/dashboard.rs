//! The `context` tool: an agent's own view of its context window (usage against the compaction
//! point, the biggest items, what is stashed) and two actions on it: offload chosen old tool
//! results to the stash, and summarize older turns now. A primitive next to the fixed thresholds
//! of the context pipeline ([`crate::factory`]), not a replacement for them.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};
use serde::Deserialize;
use strands_agents::context_manager::stash::{format_stash_refs, Stash};
use strands_agents::context_manager::{ContextState, ContextStrategy};
use strands_agents::tools::{FunctionTool, ToolContext, ToolRef};
use strands_agents::types::{ContentBlock, Message, Role, ToolResultContent};
use strands_agents::{Agent, Result};

use crate::compaction::{compaction_point, window_of};

/// Items the dashboard lists.
const TOP_ITEMS: usize = 10;
/// Tokens counted for an image, video, audio or document block.
const MEDIA_TOKENS: usize = 1_600;
/// Messages a summary always keeps verbatim (see [`crate::compaction::CompactAt`]).
pub const SUMMARY_KEEPS: usize = 4;

const DESCRIPTION: &str = concat!(
    "Your context window. No arguments: its size against the auto-summary point, the biggest items ",
    "(numbered tool results) and the stash. `offload` swaps listed results for a stub (the text stays ",
    "in `retrieve_context`); `summarize` summarizes all but the latest messages before your next call. ",
    "An edit makes your next call rewrite the prompt cache from it on (1.25x input, not 0.1x): ",
    "offload big results, in one batch."
);

/// Shared between an agent's `context` tool and its context pipeline: the stash, a pending
/// summarize request, and the tool results of the latest listing.
pub struct ContextControl {
    compact_at: Option<u64>,
    stash: RwLock<Option<Arc<Stash>>>,
    summarize: AtomicBool,
    listed: Mutex<Vec<String>>,
}

impl ContextControl {
    /// A control for an agent whose older turns are summarized at `compact_at` tokens, or at
    /// 85% of its window when `None` or smaller.
    pub fn new(compact_at: Option<u64>) -> Arc<Self> {
        Arc::new(Self {
            compact_at,
            stash: RwLock::new(None),
            summarize: AtomicBool::new(false),
            listed: Mutex::new(Vec::new()),
        })
    }

    /// Ask for older turns to be summarized before the next model call.
    pub fn request_summary(&self) {
        self.summarize.store(true, Ordering::SeqCst);
    }

    /// Take a pending summarize request (see [`crate::compaction::CompactAt`]).
    pub fn take_summarize_request(&self) -> bool {
        self.summarize.swap(false, Ordering::SeqCst)
    }

    /// The pipeline strategy that hands the agent's stash to this control. It never edits the
    /// context itself.
    pub fn strategy(self: &Arc<Self>) -> Arc<dyn ContextStrategy> {
        Arc::new(StashHandle(Arc::clone(self)))
    }

    /// The `context` tool.
    pub fn tool(self: &Arc<Self>) -> ToolRef {
        let control = Arc::clone(self);
        let tool = FunctionTool::builder("context")
            .description(DESCRIPTION)
            .input_schema(crate::tools::lean_schema::<ContextInput>())
            .typed_handler(move |input: ContextInput, ctx: ToolContext| {
                let control = Arc::clone(&control);
                async move { Result::Ok(control.run(&ctx.agent, input).await) }
            });
        Arc::new(tool)
    }

    async fn run(&self, agent: &Agent, input: ContextInput) -> String {
        let mut notes = Vec::new();
        if !input.offload.is_empty() {
            notes.push(self.offload(agent, &input.offload).await);
        }
        if input.summarize {
            self.request_summary();
            notes.push(format!(
                "Older turns will be summarized before your next model call (the latest {SUMMARY_KEEPS} messages stay verbatim)."
            ));
        }
        notes.push(self.dashboard(agent).await);
        notes.join("\n\n")
    }

    async fn dashboard(&self, agent: &Agent) -> String {
        let messages = agent.messages();
        let items = items(&messages);
        let estimate: usize = items.iter().map(|item| item.tokens).sum();
        let point = compaction_point(window_of(agent.model().as_ref()), self.compact_at);
        let measured = agent.meter().metrics().latest_context_size;
        let tokens = measured.unwrap_or(estimate as u64);
        let source = match measured {
            Some(_) => "your last model call's input",
            None => "estimated from the messages",
        };
        let share = tokens as f64 * 100.0 / point.max(1) as f64;
        let mut out = format!(
            "Context: {} tokens ({source}), {share:.0}% of the {}-token point where older turns are summarized. {} messages, ~{} tokens of them in tool results.",
            group(tokens),
            group(point),
            messages.len(),
            group(items.iter().filter(|i| i.tool_use_id.is_some()).map(|i| i.tokens as u64).sum()),
        );

        let mut biggest: Vec<&Item> = items.iter().collect();
        biggest.sort_by_key(|item| std::cmp::Reverse(item.tokens));
        biggest.truncate(TOP_ITEMS);
        let mut listed = Vec::new();
        if !biggest.is_empty() {
            out.push_str("\n\nBiggest items:");
        }
        for item in biggest {
            let number = match &item.tool_use_id {
                // A preview or stub frees next to nothing when offloaded.
                Some(id) if !item.stashed => {
                    listed.push(id.clone());
                    format!("#{}", listed.len())
                }
                _ => "  ".to_owned(),
            };
            out.push_str(&format!(
                "\n{number} ~{} tokens, {}: {}",
                group(item.tokens as u64),
                age(item.age),
                item.what
            ));
        }
        let offloadable = !listed.is_empty();
        *self.listed.lock() = listed;

        let previews: Vec<&Item> = items.iter().filter(|item| item.stashed).collect();
        let stash = self.stash.read().clone();
        out.push_str("\n\nStash: ");
        match stash {
            None => out.push_str("none; offloading is unavailable."),
            Some(stash) => {
                let entries = stash.list().await.map_or(0, |keys| keys.len());
                out.push_str(&format!(
                    "{entries} entries (every tool result is kept there). {} results in context are previews or stubs of stashed output (~{} tokens); `retrieve_context` with a ref reads one back.",
                    previews.len(),
                    group(previews.iter().map(|item| item.tokens as u64).sum()),
                ));
            }
        }
        if offloadable {
            out.push_str("\nOffload numbered results with `context {\"offload\": [1, 2]}`.");
        }
        out
    }

    async fn offload(&self, agent: &Agent, numbers: &[usize]) -> String {
        let Some(stash) = self.stash.read().clone() else {
            return "Nothing offloaded: this agent has no stash.".to_owned();
        };
        let listed = self.listed.lock().clone();
        if listed.is_empty() {
            return "Nothing offloaded: call `context` first to list the results to choose from.".to_owned();
        }
        let mut ids = Vec::new();
        let mut unknown = Vec::new();
        for &number in numbers {
            match number.checked_sub(1).and_then(|i| listed.get(i)) {
                Some(id) => ids.push(id.clone()),
                None => unknown.push(number.to_string()),
            }
        }
        let mut messages = agent.messages();
        let (mut done, mut freed, mut offloaded, mut first) = (0, 0, Vec::new(), None);
        for (at, message) in messages
            .iter_mut()
            .enumerate()
            .filter(|(_, m)| m.role == Role::User)
        {
            for block in &mut message.content {
                let ContentBlock::ToolResult(result) = block else {
                    continue;
                };
                if !ids.contains(&result.tool_use_id) {
                    continue;
                }
                if is_stub(&result.content) {
                    // Rewriting a stub frees nothing and costs the cached prefix after it.
                    continue;
                }
                let before = result_tokens(&result.content);
                let mut refs = stash_refs(&result.tool_use_id, &result.content);
                if refs.is_empty() {
                    // Stashed when the result arrived; stored again only if missing, so the
                    // original is never overwritten.
                    for (index, item) in result.content.iter().enumerate() {
                        let key = format!("{}_{index}", result.tool_use_id);
                        if stash.retrieve(&key).await.is_none() {
                            let value = serde_json::to_value(item).unwrap_or_default();
                            if stash.store(&result.tool_use_id, index, &value).await.is_err() {
                                continue;
                            }
                        }
                        refs.push(key);
                    }
                }
                let stub = format!(
                    "{STUB_PREFIX}~{} tokens]{}",
                    group(before as u64),
                    format_stash_refs(&refs)
                );
                result.content = vec![ToolResultContent::Text(stub)];
                freed += before.saturating_sub(result_tokens(&result.content));
                done += 1;
                first.get_or_insert(at);
                offloaded.extend(refs);
            }
        }
        let mut note = format!("Offloaded {done} tool results (~{} tokens).", group(freed as u64));
        if let Some(first) = first {
            // Everything from the first edit on is written to the cache again on the next call.
            let rewritten: usize = messages[first..]
                .iter()
                .flat_map(|m| &m.content)
                .map(block_tokens)
                .sum();
            note.push_str(&format!(
                " Your next call writes the ~{} tokens from the first of them on to the prompt cache again.",
                group(rewritten as u64)
            ));
            agent.set_messages(messages);
        }
        if !offloaded.is_empty() {
            note.push_str(&format!(
                " `retrieve_context` reads them back by reference: {}.",
                offloaded.join(", ")
            ));
        }
        if !unknown.is_empty() {
            note.push_str(&format!(" Not in the latest listing: {}.", unknown.join(", ")));
        }
        note
    }
}

/// How the stub of an offloaded result starts.
const STUB_PREFIX: &str = "[Offloaded by the agent: ";

/// Whether `content` is already the stub [`ContextControl::offload`] leaves.
fn is_stub(content: &[ToolResultContent]) -> bool {
    matches!(content, [ToolResultContent::Text(text)] if text.starts_with(STUB_PREFIX))
}

/// Input of the `context` tool.
#[derive(Deserialize, schemars::JsonSchema)]
pub struct ContextInput {
    /// Numbers from the latest listing.
    #[serde(default)]
    offload: Vec<usize>,
    #[serde(default)]
    summarize: bool,
}

struct StashHandle(Arc<ContextControl>);

#[async_trait]
impl ContextStrategy for StashHandle {
    fn name(&self) -> &str {
        "swarm:context-tool"
    }

    fn init(&self, _agent: &Agent, stash: Option<Arc<Stash>>) {
        *self.0.stash.write() = stash;
    }

    async fn apply(&self, _ctx: &mut ContextState<'_>) -> Result<bool> {
        Ok(false)
    }
}

/// One entry of the dashboard: a tool result, or the rest of a message.
#[derive(Debug)]
struct Item {
    tokens: usize,
    /// Model calls since it arrived, the one in progress included.
    age: usize,
    what: String,
    /// Set for tool results, which can be offloaded.
    tool_use_id: Option<String>,
    /// A preview or stub that names stashed content.
    stashed: bool,
}

fn items(messages: &[Message]) -> Vec<Item> {
    let mut calls: HashMap<&str, (&str, &serde_json::Value)> = HashMap::new();
    for message in messages {
        for block in &message.content {
            if let ContentBlock::ToolUse(call) = block {
                calls.insert(&call.tool_use_id, (&call.name, &call.input));
            }
        }
    }
    // The call in progress (the one running this tool) is not in `messages` yet.
    let mut later_calls = messages.iter().filter(|m| m.role == Role::Assistant).count() + 1;
    let mut items = Vec::new();
    for message in messages {
        if message.role == Role::Assistant {
            later_calls -= 1;
        }
        let mut rest = 0;
        for block in &message.content {
            match block {
                ContentBlock::ToolResult(result) => {
                    let refs = stash_refs(&result.tool_use_id, &result.content);
                    let what = match calls.get(result.tool_use_id.as_str()) {
                        Some((name, input)) => {
                            format!("{name} result `{}`", crate::runtime::summarize_input(input))
                        }
                        None => "tool result".to_owned(),
                    };
                    let stashed = !refs.is_empty() || is_stub(&result.content);
                    items.push(Item {
                        tokens: result_tokens(&result.content),
                        age: later_calls,
                        // A stub names its stash keys, for `retrieve_context`.
                        what: what + &format_stash_refs(&refs),
                        tool_use_id: Some(result.tool_use_id.clone()),
                        stashed,
                    });
                }
                other => rest += block_tokens(other),
            }
        }
        if rest > 0 {
            let role = match message.role {
                Role::User => "user message",
                Role::Assistant => "your message",
            };
            let calls: Vec<&str> = message
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolUse(call) => Some(call.name.as_str()),
                    _ => None,
                })
                .collect();
            let what = match (preview(&message.text()), calls.is_empty()) {
                (text, true) => format!("{role} \"{text}\""),
                (text, false) if text.is_empty() => format!("{role}, calling {}", calls.join(", ")),
                (text, false) => format!("{role} \"{text}\", calling {}", calls.join(", ")),
            };
            items.push(Item {
                tokens: rest,
                age: later_calls,
                what,
                tool_use_id: None,
                stashed: false,
            });
        }
    }
    items
}

fn block_tokens(block: &ContentBlock) -> usize {
    match block {
        ContentBlock::Text(text) => text.chars().count() / 4,
        ContentBlock::ToolUse(call) => (call.name.len() + call.input.to_string().len()) / 4,
        ContentBlock::ToolResult(result) => result_tokens(&result.content),
        ContentBlock::Image(_)
        | ContentBlock::Video(_)
        | ContentBlock::Audio(_)
        | ContentBlock::Document(_) => MEDIA_TOKENS,
        ContentBlock::CachePoint(_) => 0,
        other => serde_json::to_string(other).map_or(0, |json| json.len() / 4),
    }
}

fn result_tokens(content: &[ToolResultContent]) -> usize {
    content
        .iter()
        .map(|item| match item {
            ToolResultContent::Text(text) => text.chars().count() / 4,
            ToolResultContent::Json(json) => json.to_string().len() / 4,
            _ => MEDIA_TOKENS,
        })
        .sum()
}

/// The stash keys under which a preview or stub of result `tool_use_id` names its own output
/// (`[ref: <id>_0]` or `[refs: <id>_0, <id>_1]`), each once. Refs to other results, which a
/// `context` listing or a `retrieve_context` read quotes, don't count: such a result is not
/// stashed under them.
fn stash_refs(tool_use_id: &str, content: &[ToolResultContent]) -> Vec<String> {
    let own = format!("{tool_use_id}_");
    let mut refs: Vec<String> = Vec::new();
    for item in content {
        let ToolResultContent::Text(text) = item else {
            continue;
        };
        for marker in ["[ref: ", "[refs: "] {
            for (start, _) in text.match_indices(marker) {
                let rest = &text[start + marker.len()..];
                let Some(end) = rest.find(']') else {
                    continue;
                };
                for key in rest[..end].split(", ") {
                    let index = key.strip_prefix(&own);
                    if index.is_some_and(|i| i.parse::<usize>().is_ok()) && !refs.iter().any(|r| r == key) {
                        refs.push(key.to_owned());
                    }
                }
            }
        }
    }
    refs
}

fn preview(text: &str) -> String {
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or_default()
        .trim();
    crate::text::clip(line, 60)
}

fn age(calls: usize) -> String {
    match calls {
        0 | 1 => "1 call ago".to_owned(),
        n => format!("{n} calls ago"),
    }
}

/// `84213` as `84,213`.
fn group(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use strands_agents::context_manager::{ContextManagerConfig, ContextManagerOption, StashOption};
    use strands_agents::testing::{MockMessageModel, MockResponse};
    use strands_agents::types::ToolUseBlock;

    fn call(name: &str, id: &str, input: serde_json::Value) -> MockResponse {
        MockResponse::message(Message::new(
            Role::Assistant,
            vec![ContentBlock::ToolUse(ToolUseBlock::new(name, id, input))],
        ))
    }

    fn dump_tool() -> ToolRef {
        Arc::new(FunctionTool::builder("shell").description("print").handler(
            |input: serde_json::Value, _ctx| async move {
                let n = input["n"].as_u64().unwrap_or(0) as usize;
                Ok(serde_json::Value::String(format!("{}\nEND", "x".repeat(n))))
            },
        ))
    }

    fn result_text(model: &MockMessageModel, call: usize, tool_use_id: &str) -> String {
        model.calls()[call]
            .0
            .iter()
            .flat_map(|message| &message.content)
            .find_map(|block| match block {
                ContentBlock::ToolResult(result) if result.tool_use_id == tool_use_id => {
                    Some(format!("{:?}", result.content))
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("no result {tool_use_id} in call {call}"))
    }

    /// A mock with a 1M-token window, Opus 5.5's.
    struct Million(Arc<MockMessageModel>);

    impl strands_agents::Model for Million {
        fn model_id(&self) -> Option<String> {
            self.0.model_id()
        }

        fn context_window_limit(&self) -> Option<u64> {
            Some(1_000_000)
        }

        fn stream<'a>(
            &'a self,
            messages: &'a [Message],
            options: &'a strands_agents::models::StreamOptions,
        ) -> strands_agents::models::ModelEventStream<'a> {
            self.0.stream(messages, options)
        }
    }

    #[tokio::test]
    async fn the_point_is_85_percent_of_the_window_unless_compact_at_is_smaller() {
        for (compact_at, point) in [(None, "850,000"), (Some(300_000), "300,000")] {
            let model = MockMessageModel::shared(vec![
                call("context", "c1", serde_json::json!({})),
                MockResponse::message(Message::assistant("done")),
            ]);
            let control = ContextControl::new(compact_at);
            let agent = Agent::builder()
                .model(Million(model.clone()))
                .tools([control.tool()])
                .build()
                .unwrap();
            agent.invoke("go").await.unwrap();
            let listing = result_text(&model, 1, "c1");
            assert!(
                listing.contains(&format!("% of the {point}-token point")),
                "{listing}"
            );
        }
    }

    #[tokio::test]
    async fn the_agent_lists_its_biggest_results_and_offloads_one_it_can_read_back() {
        let model = MockMessageModel::shared(vec![
            call(
                "shell",
                "big",
                serde_json::json!({ "command": "cargo test", "n": 40_000 }),
            ),
            call(
                "shell",
                "small",
                serde_json::json!({ "command": "ls", "n": 4_000 }),
            ),
            call("context", "c1", serde_json::json!({})),
            call("context", "c2", serde_json::json!({ "offload": [1, 9] })),
            MockResponse::message(Message::assistant("done")),
        ]);
        let control = ContextControl::new(None);
        let manager = ContextManagerOption::Config(ContextManagerConfig {
            strategies: Some(vec![control.strategy()]),
            stash: StashOption::Enabled,
        });
        let agent = Agent::builder()
            .model(model.clone())
            .tools([dump_tool(), control.tool()])
            .context_manager(manager)
            .build()
            .unwrap();
        agent.invoke("go").await.unwrap();

        let listing = result_text(&model, 3, "c1");
        assert!(listing.contains("Context: "), "{listing}");
        // The mock has no known window: 85% of the assumed 200k.
        assert!(listing.contains("% of the 170,000-token point"), "{listing}");
        assert!(listing.contains("your message, calling shell"), "{listing}");
        assert!(
            listing.contains("#1 ~10,001 tokens, 2 calls ago: shell result `cargo test`"),
            "{listing}"
        );
        assert!(
            listing.contains("#2 ~1,001 tokens, 1 call ago: shell result `ls`"),
            "{listing}"
        );
        assert!(listing.contains("0 results in context are previews"), "{listing}");

        // The next model call sees the stub instead of the output.
        let report = result_text(&model, 4, "c2");
        assert!(report.contains("Offloaded 1 tool results"), "{report}");
        assert!(
            report.contains("Your next call writes the ~1,1")
                && report.contains("tokens from the first of them on to the prompt cache again"),
            "the stub, the small result and the listing after it: {report}"
        );
        assert!(report.contains("by reference: big_0."), "{report}");
        assert!(
            report.contains("shell result `cargo test` [ref: big_0]"),
            "{report}"
        );
        assert!(report.contains("Not in the latest listing: 9."), "{report}");
        assert!(report.contains("1 results in context are previews"), "{report}");
        let stub = result_text(&model, 4, "big");
        assert!(
            stub.contains("[Offloaded by the agent: ~10,001 tokens] [ref: big_0]"),
            "{stub}"
        );
        assert!(!stub.contains("xxxx"), "{stub}");
        assert!(
            result_text(&model, 4, "small").contains("xxxx"),
            "the other stays"
        );

        let stash = control.stash.read().clone().unwrap();
        let full = stash.retrieve("big_0").await.expect("still stashed");
        assert!(full.to_string().contains(&"x".repeat(40_000)));

        // Offloading the stub again keeps its ref and never overwrites the original.
        *control.listed.lock() = vec!["big".into()];
        control.offload(&agent, &[1]).await;
        let full = stash.retrieve("big_0").await.unwrap();
        assert!(full.to_string().contains(&"x".repeat(40_000)));
    }

    #[test]
    fn stash_refs_are_read_from_previews_and_stubs() {
        let content = |text: &str| vec![ToolResultContent::Text(text.into())];
        assert_eq!(stash_refs("t1", &content("head … tail [ref: t1_0]")), ["t1_0"]);
        assert_eq!(
            stash_refs("a", &content("[Offloaded] [refs: a_0, a_1, a_0]")),
            ["a_0", "a_1"]
        );
        assert!(stash_refs("t1", &content("plain output")).is_empty());
        // A listing or a read-back quotes other results' refs; this one is not stashed under them.
        assert!(stash_refs("c1", &content("#1 shell result `ls` [ref: big_0], [refs: c10_0]")).is_empty());
    }

    #[tokio::test]
    async fn offloading_a_listing_or_a_stub_keeps_refs_its_own_and_edits_nothing_twice() {
        let model = MockMessageModel::shared(vec![
            call(
                "shell",
                "big",
                serde_json::json!({ "command": "cargo test", "n": 40_000 }),
            ),
            call("context", "c1", serde_json::json!({})),
            call("context", "c2", serde_json::json!({ "offload": [1] })),
            // c2's listing numbers only c1 (big is a stub now); c3's then only c2.
            call("context", "c3", serde_json::json!({ "offload": [1] })),
            call("context", "c4", serde_json::json!({ "offload": [1] })),
            MockResponse::message(Message::assistant("done")),
        ]);
        let control = ContextControl::new(None);
        let manager = ContextManagerOption::Config(ContextManagerConfig {
            strategies: Some(vec![control.strategy()]),
            stash: StashOption::Enabled,
        });
        let agent = Agent::builder()
            .model(model.clone())
            .tools([dump_tool(), control.tool()])
            .context_manager(manager)
            .build()
            .unwrap();
        agent.invoke("go").await.unwrap();

        let listing = result_text(&model, 3, "c2");
        assert!(listing.contains("Offloaded 1 tool results"), "{listing}");
        assert!(
            !listing.contains("#1 ~10,001") && listing.contains("shell result `cargo test` [ref: big_0]"),
            "a stub is listed without a number: {listing}"
        );
        let after = result_text(&model, 5, "c4");
        assert!(after.contains("Offloaded 1 tool results"), "{after}");
        let stub = |id: &str| result_text(&model, 5, id);
        assert!(
            stub("big").contains("[ref: big_0]") && !stub("big").contains("big_0, "),
            "{}",
            stub("big")
        );
        for listing in ["c1", "c2"] {
            let text = stub(listing);
            assert!(text.contains(&format!("[ref: {listing}_0]")), "{text}");
            assert!(!text.contains("big_0"), "not another result's ref: {text}");
        }
        let stash = control.stash.read().clone().unwrap();
        let kept = stash
            .retrieve("c2_0")
            .await
            .expect("the listing itself is stashed");
        assert!(kept.to_string().contains("Biggest items"), "{kept}");

        // Offloading stubs again leaves the history (and the cached prefix) as it is.
        let before = agent.messages();
        *control.listed.lock() = vec!["big".into(), "c1".into()];
        let note = control.offload(&agent, &[1, 2]).await;
        assert!(note.starts_with("Offloaded 0 tool results"), "{note}");
        assert!(!note.contains("next call writes"), "{note}");
        assert_eq!(format!("{:?}", agent.messages()), format!("{before:?}"));
    }

    #[test]
    fn numbers_and_ages_read_well() {
        assert_eq!(group(0), "0");
        assert_eq!(group(1_234), "1,234");
        assert_eq!(group(200_000), "200,000");
        assert_eq!(age(1), "1 call ago");
        assert_eq!(age(3), "3 calls ago");
    }
}
