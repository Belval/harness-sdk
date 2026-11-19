//! Keep-warm holds no copy of a waiting agent's conversation: a refresh is built from the agent's
//! own history when it is sent, so an agent in a long tool call is in memory once, not twice.
//!
//! This binary has one test and counts every allocation, so it compares heap in use with
//! keep-warm on and off. `KEEP_WARM_AGENTS` and `KEEP_WARM_MB` (default 4 and 2) size the swarm;
//! `KEEP_WARM_ONLY=on|off` runs one alone and prints its resident set.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use peak_alloc::PeakAlloc;
use strands_agents::models::{ModelEventStream, StreamOptions};
use strands_agents::testing::events_for_message;
use strands_agents::tools::FunctionTool;
use strands_agents::types::{ContentBlock, Message, Role, StopReason, ToolUseBlock, Usage};
use strands_agents::{Agent, Model};
use strands_swarm::{AgentFactory, AgentSpec, SpawnRequest, Swarm, SwarmConfig};
use tokio::sync::Notify;

#[global_allocator]
static HEAP: PeakAlloc = PeakAlloc;

/// Asks for `pause` until it sees a tool result, with every response a cache hit.
struct Caller;

impl Model for Caller {
    fn model_id(&self) -> Option<String> {
        Some("caching".into())
    }

    fn stream<'a>(&'a self, messages: &'a [Message], _: &'a StreamOptions) -> ModelEventStream<'a> {
        let answered = messages
            .last()
            .is_some_and(|m| m.content.iter().any(|b| matches!(b, ContentBlock::ToolResult(_))));
        let (message, stop) = match answered {
            false => (
                Message::new(
                    Role::Assistant,
                    vec![ContentBlock::ToolUse(ToolUseBlock::new(
                        "pause",
                        "p1",
                        serde_json::json!({}),
                    ))],
                ),
                StopReason::ToolUse,
            ),
            true => (Message::assistant("done"), StopReason::EndTurn),
        };
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 1005,
            cache_read_input_tokens: Some(990),
            ..Default::default()
        };
        let events = events_for_message(&message, Some(stop), Some(usage));
        Box::pin(futures::stream::iter(events.into_iter().map(Ok)))
    }
}

/// About `bytes` of alternating turns, ending on the assistant's.
fn conversation(bytes: usize, salt: usize) -> Vec<Message> {
    let text = |i: usize| format!("{salt}/{i} {}", "lorem ipsum ".repeat(4096));
    let turns = (bytes / text(0).len()).max(1) / 2 * 2;
    (0..turns)
        .map(|i| match i % 2 {
            0 => Message::user(text(i)),
            _ => Message::assistant(text(i)),
        })
        .collect()
}

fn resident() -> usize {
    let pages = std::fs::read_to_string("/proc/self/statm").unwrap();
    let pages: usize = pages.split_whitespace().nth(1).unwrap().parse().unwrap();
    pages * 4096
}

/// Heap this run holds (what it allocated since it began) and the process's resident set while `agents` agents with `bytes`-sized conversations all wait
/// on a tool call.
async fn waiting(agents: usize, bytes: usize, keep_warm: bool) -> (usize, usize) {
    let dir = tempfile::tempdir().unwrap();
    let before = HEAP.current_usage();
    let waiting = Arc::new(AtomicUsize::new(0));
    let release = Arc::new(Notify::new());
    let (counter, gate) = (waiting.clone(), release.clone());
    let factory: AgentFactory = Arc::new(move |spec: AgentSpec| {
        let (counter, gate) = (counter.clone(), gate.clone());
        let pause = FunctionTool::builder("pause").description("Waits.").handler(
            move |_input: serde_json::Value, _ctx| {
                let (counter, gate) = (counter.clone(), gate.clone());
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    gate.notified().await;
                    strands_agents::Result::Ok("back".to_owned())
                }
            },
        );
        let mut tools = spec.tools;
        tools.push(Arc::new(pause));
        Box::pin(async move {
            let agent = Agent::builder()
                .id(spec.id.clone())
                .model(Arc::new(Caller))
                .tools(tools)
                .no_sandbox()
                .build()?;
            agent.set_messages(conversation(bytes, spec.id.len() + agents));
            Ok(agent)
        })
    });
    let mut config = SwarmConfig::new(factory, dir.path());
    config.keep_warm = keep_warm.then_some(Duration::from_secs(3600));
    let (swarm, _events) = Swarm::new(config);
    for i in 0..agents {
        let request = SpawnRequest {
            task: format!("go {i}"),
            name: Some(format!("w{i}")),
            ..Default::default()
        };
        swarm.spawn(strands_swarm::USER, request).await.unwrap();
    }
    while waiting.load(Ordering::SeqCst) < agents {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // Let the keep-warm tasks start and the turns' transient copies be freed.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let (heap, rss) = (HEAP.current_usage().saturating_sub(before), resident());
    release.notify_waiters();
    swarm.shutdown().await;
    drop(swarm);
    // Whatever the run still holds goes before the next one starts.
    tokio::time::sleep(Duration::from_secs(1)).await;
    (heap, rss)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_waiting_agent_s_conversation_is_not_held_twice() {
    let env = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let agents = env("KEEP_WARM_AGENTS", 4);
    let bytes = env("KEEP_WARM_MB", 2) << 20;
    // `KEEP_WARM_ONLY=on|off` runs one swarm alone, for a resident set that is its own.
    if let Ok(only) = std::env::var("KEEP_WARM_ONLY") {
        let (heap, rss) = waiting(agents, bytes, only == "on").await;
        eprintln!(
            "keep-warm {only}: heap {} MB, resident {} MB",
            heap >> 20,
            rss >> 20
        );
        return;
    }
    // A first swarm in the process allocates what every later one reuses.
    waiting(1, 1 << 16, false).await;
    let (cold, _) = waiting(agents, bytes, false).await;
    let (warm, _) = waiting(agents, bytes, true).await;
    eprintln!(
        "{agents} agents x {} MB waiting: heap {} MB without keep-warm, {} MB with",
        bytes >> 20,
        cold >> 20,
        warm >> 20
    );
    let extra = warm.saturating_sub(cold);
    let conversations = agents * bytes;
    assert!(
        extra < conversations / 10,
        "keep-warm holds {extra} bytes beside {conversations} bytes of conversations"
    );
}
