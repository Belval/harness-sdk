//! Prompt-cache stability through the whole swarm: real harness agents (every plugin, injector
//! and hook the swarm installs) driven by the runtime across turns, mid-turn mail, a child's
//! reply, the user's memory and a restart in the middle of a `wait`. Each call must re-send
//! what the previous call cached byte for byte: per-call text (todo list, environment) only
//! after the cache point, and nothing the runtime adds may edit an earlier message. Nothing here
//! is old or large enough for a pruning strategy to act on deliberately.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::json;
use strands_agents::models::StreamOptions;
use strands_agents::testing::{MockMessageModel, MockResponse};
use strands_agents::types::{ContentBlock, Message, Role, ToolUseBlock};
use strands_harness::ModelOption;

use super::{build, harness_options, HarnessFactoryOptions};
use crate::runtime::{AgentFactory, AgentSpec, SpawnRequest, Swarm, SwarmConfig, SwarmEvent, USER};

type Scripts = Arc<Mutex<HashMap<String, Arc<MockMessageModel>>>>;

fn factory(scripts: &Scripts) -> AgentFactory {
    let scripts = Arc::clone(scripts);
    Arc::new(move |spec: AgentSpec| {
        let model = scripts.lock().get(&spec.id).cloned().expect("a script per agent");
        Box::pin(async move {
            let options = HarnessFactoryOptions::default();
            let mut harness = harness_options(&options, spec)?;
            harness.model = Some(ModelOption::Instance(model));
            build(&options, harness).await
        })
    })
}

fn call(name: &str, id: &str, input: serde_json::Value) -> MockResponse {
    MockResponse::message(Message::new(
        Role::Assistant,
        vec![ContentBlock::ToolUse(ToolUseBlock::new(name, id, input))],
    ))
}

fn say(text: &str) -> MockResponse {
    MockResponse::message(Message::assistant(text))
}

fn todos(status: &str) -> serde_json::Value {
    json!({"todos": [
        {"content": "Do it", "activeForm": "Doing it", "status": status},
        {"content": "Check it", "activeForm": "Checking it", "status": "pending"},
    ]})
}

async fn reply(events: &mut tokio::sync::mpsc::UnboundedReceiver<SwarmEvent>) -> String {
    let wait = async {
        while let Some(event) = events.recv().await {
            if let SwarmEvent::Mail { to, mail } = event {
                if to == USER && mail.kind == "reply" {
                    return mail.text;
                }
            }
        }
        panic!("event stream closed");
    };
    tokio::time::timeout(Duration::from_secs(20), wait)
        .await
        .expect("a reply")
}

/// A message as the provider sees it: role and content, no metadata.
fn wire(message: &Message, blocks: usize) -> String {
    let content = serde_json::to_string(&message.content[..blocks]).unwrap();
    format!("{:?} {content}", message.role)
}

type Call = (Vec<Message>, StreamOptions);

/// Asserts `next` re-sends everything `previous` put before its cache point: the system prompt,
/// the tools, every earlier message and the last one up to its per-call trailing blocks.
fn assert_cached_prefix_kept(previous: &Call, next: &Call, at: usize) {
    let ((before, options), (after, next_options)) = (previous, next);
    assert_eq!(
        format!("{:?}", options.system_prompt),
        format!("{:?}", next_options.system_prompt),
        "call {at}: the system prompt changed"
    );
    assert_eq!(
        format!("{:?}", options.tool_specs),
        format!("{:?}", next_options.tool_specs),
        "call {at}: the tools changed"
    );
    assert!(after.len() >= before.len(), "call {at}: messages dropped");
    let last = before.len() - 1;
    for (index, (sent, resent)) in before.iter().zip(after).enumerate() {
        let blocks = match index == last {
            true => sent.content.len() - options.dynamic_trailing_blocks.unwrap_or(0),
            false => sent.content.len(),
        };
        assert!(
            resent.content.len() >= blocks,
            "call {at}: message {index} lost blocks"
        );
        assert_eq!(
            wire(sent, blocks),
            wire(resent, blocks),
            "call {at}: message {index} changed"
        );
    }
}

/// The journal line of the helper going idle, which follows its reply.
const HELPER_IDLE: &str = r#""id":"a2","status":"idle""#;

#[tokio::test]
async fn every_call_resends_the_cached_prefix_unchanged_across_turns_mail_and_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let state_dir = dir.path().join("run");
    let scripts: Scripts = Arc::default();
    let script = |id: &str, responses| {
        scripts
            .lock()
            .insert(id.to_owned(), MockMessageModel::shared(responses))
    };
    script(
        "a1",
        vec![
            call("todo_write", "t1", todos("in_progress")),
            call("spawn_agent", "t2", json!({"task": "help", "name": "helper"})),
            // Long enough to be trimmed once six newer results exist, if anything prunes.
            call(
                "shell",
                "t3",
                json!({"command": "head -c 3000 /dev/zero | tr '\\0' x"}),
            ),
            // Runs until the helper has replied (its reply comes before the journal shows it
            // idle; at most 5s), so the reply rides on this result or an earlier one.
            call(
                "shell",
                "t4",
                json!({"command": format!("for _ in $(seq 500); do grep -qs '{HELPER_IDLE}' run/events.jsonl && break; sleep 0.01; done")}),
            ),
        ]
        .into_iter()
        .chain((0..6).map(|i| call("shell", &format!("e{i}"), json!({"command": format!("echo {i}")}))))
        .chain([
            say("first done"),
            call("todo_write", "t5", todos("completed")),
            call("wait", "t6", json!({"timeout_seconds": 600})),
        ])
        .collect(),
    );
    script("a2", vec![say("helped")]);
    let mut config = SwarmConfig::new(factory(&scripts), dir.path());
    config.state_dir = Some(state_dir.clone());
    // Each turn re-reads the memory file into the system prompt.
    let memory = Arc::new(crate::memory::Memory::new(dir.path().join("memory.md")));
    memory.remember("Prefer tabs.", "2026-01-01").await.unwrap();
    config.memory = Some(memory);
    let (swarm, mut events) = Swarm::new(config.clone());
    let request = SpawnRequest {
        // A long task, so the trimmable result is too small a share to prune for its own sake.
        task: format!("go {}", "and keep going ".repeat(3_000)),
        ..Default::default()
    };
    swarm.spawn(USER, request).await.unwrap();
    assert_eq!(reply(&mut events).await, "first done");
    swarm.send(USER, "a1", "next").unwrap();
    let first = scripts.lock()["a1"].clone();
    let waiting = async {
        while first.call_count() < 13 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    tokio::time::timeout(Duration::from_secs(10), waiting)
        .await
        .unwrap();
    swarm.shutdown().await;

    // The restart cuts the wait; the resumed agent gets the notice and carries on.
    script("a1", vec![say("resumed"), say("bye")]);
    let saved = crate::state::load(&state_dir)
        .await
        .unwrap()
        .expect("state saved");
    let (swarm, mut events) = Swarm::resume(config, saved).await.unwrap();
    assert_eq!(reply(&mut events).await, "resumed");
    swarm.send(USER, "a1", "last").unwrap();
    assert_eq!(reply(&mut events).await, "bye");
    let second = scripts.lock()["a1"].clone();
    swarm.shutdown().await;

    let calls: Vec<Call> = first.calls().into_iter().chain(second.calls()).collect();
    assert_eq!(calls.len(), 15);
    let reminded = |call: &Call| call.0.last().unwrap().text().contains("current todo list");
    assert!(
        calls[1..].iter().all(reminded),
        "the todo list rides along on every call"
    );
    let system = format!("{:?}", calls[0].1.system_prompt);
    assert!(
        system.contains("Prefer tabs."),
        "the memory is in the system prompt"
    );
    let sent = serde_json::to_string(&calls[14].0).unwrap();
    assert!(
        sent.contains(crate::resume::RESTARTED_MESSAGE),
        "the cut turn says why"
    );
    assert!(
        !sent.contains(strands_agents::agent::CANCELLED_MESSAGE),
        "not as a user stop"
    );
    assert!(sent.contains("helped"), "the helper's reply reached a1");
    assert!(
        sent.contains("The swarm was restarted"),
        "the restart notice reached a1"
    );
    for (at, pair) in calls.windows(2).enumerate() {
        assert_cached_prefix_kept(&pair[0], &pair[1], at + 1);
    }
    let sent = &calls[14].0;
    let history: String = sent[..sent.len() - 1].iter().map(Message::text).collect();
    assert!(
        !history.contains("current todo list"),
        "per-call text stays out of the history"
    );
}
