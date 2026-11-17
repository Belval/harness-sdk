//! Swarm runtime behaviour against scripted mock models: spawning, mail delivery, cancellation,
//! limits and worktrees.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::json;
use strands_agents::testing::{MockMessageModel, MockResponse};
use strands_agents::types::{ContentBlock, Message, Role, ToolUseBlock};
use strands_agents::{Agent, Model};
use strands_swarm::compaction::Compaction;
use strands_swarm::status::Merge;
use strands_swarm::{
    AgentFactory, AgentSpec, AgentStatus, LandRequest, ModelResolver, ReloadWhen, SpawnRequest, Swarm,
    SwarmConfig, SwarmEvent, GOAL_MAX_ATTEMPTS, USER,
};
use tokio::sync::mpsc::UnboundedReceiver;

type Scripts = Arc<Mutex<HashMap<String, Arc<MockMessageModel>>>>;

fn tool_call(name: &str, id: &str, input: serde_json::Value) -> MockResponse {
    MockResponse::message(Message::new(
        Role::Assistant,
        vec![ContentBlock::ToolUse(ToolUseBlock::new(name, id, input))],
    ))
}

fn say(text: &str) -> MockResponse {
    MockResponse::message(Message::assistant(text))
}

/// A factory that gives agent `aN` the mock registered under `aN`.
fn mock_factory(scripts: &Scripts) -> AgentFactory {
    let scripts = scripts.clone();
    Arc::new(move |spec: AgentSpec| {
        let model = scripts
            .lock()
            .get(&spec.id)
            .cloned()
            .expect("no script for agent");
        Box::pin(async move {
            Agent::builder()
                .id(spec.id)
                .model(model)
                .system_prompt(spec.instructions)
                .tools(spec.tools)
                .no_sandbox()
                .build()
        })
    })
}

fn swarm_with(
    scripts: Vec<(&str, Vec<MockResponse>)>,
    cwd: &Path,
) -> (Swarm, UnboundedReceiver<SwarmEvent>, Scripts) {
    let map: Scripts = Arc::new(Mutex::new(HashMap::new()));
    for (id, responses) in scripts {
        map.lock()
            .insert(id.to_owned(), MockMessageModel::shared(responses));
    }
    let (swarm, events) = Swarm::new(SwarmConfig::new(mock_factory(&map), cwd));
    (swarm, events, map)
}

/// Collect events until `to` receives a reply, returning that reply's text.
async fn reply_to(events: &mut UnboundedReceiver<SwarmEvent>, to: &str) -> String {
    let wait = async {
        while let Some(event) = events.recv().await {
            if let SwarmEvent::Mail { to: receiver, mail } = event {
                if receiver == to && mail.kind == "reply" {
                    return mail.text;
                }
            }
        }
        panic!("event stream closed");
    };
    tokio::time::timeout(Duration::from_secs(10), wait)
        .await
        .expect("timed out waiting for a reply")
}

fn sent_text(model: &MockMessageModel, call: usize) -> String {
    let (messages, _) = &model.calls()[call];
    messages
        .iter()
        .map(|m| format!("{m:?}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[tokio::test]
async fn a_spawned_agents_reply_reaches_the_parent_through_wait() {
    let dir = tempfile::tempdir().unwrap();
    let (swarm, mut events, scripts) = swarm_with(
        vec![
            (
                "a1",
                vec![
                    tool_call(
                        "spawn_agent",
                        "t1",
                        json!({"task": "count to 3", "name": "counter"}),
                    ),
                    tool_call("wait", "t2", json!({"from": ["a2"]})),
                    say("child said it"),
                ],
            ),
            ("a2", vec![say("1 2 3")]),
        ],
        dir.path(),
    );

    swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "go".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(reply_to(&mut events, USER).await, "child said it");
    let root = scripts.lock()["a1"].clone();
    let third = sent_text(&root, 2);
    assert!(third.contains("counter (a2)"), "{third}");
    assert!(third.contains("1 2 3"), "{third}");
    let child = scripts.lock()["a2"].clone();
    assert!(sent_text(&child, 0).contains("count to 3"));
}

#[tokio::test]
async fn a_silent_or_empty_final_reply_reaches_no_agent_parent() {
    let dir = tempfile::tempdir().unwrap();
    let spawn = json!({"task": "poll", "name": "poller"});
    let (swarm, mut events, scripts) = swarm_with(
        vec![
            (
                "a1",
                vec![
                    tool_call("spawn_agent", "t1", spawn),
                    say("spawned"),
                    say("noted"),
                ],
            ),
            ("a2", vec![say("[silent]"), say(" \n"), say("CI is green")]),
        ],
        dir.path(),
    );
    spawn_root(&swarm, "go").await;
    assert_eq!(reply_to(&mut events, USER).await, "spawned");
    let child = scripts.lock()["a2"].clone();
    for (turn, poll) in ["poll 2", "poll 3"].into_iter().enumerate() {
        while child.call_count() <= turn {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        wait_for_status(&swarm, "a2", AgentStatus::Idle).await;
        swarm.send(USER, "a2", poll).unwrap();
    }
    let mut to_parent = Vec::new();
    while let Some(event) = events.recv().await {
        match event {
            SwarmEvent::Mail { to, mail } if to == "a1" && mail.from == "a2" => to_parent.push(mail.text),
            SwarmEvent::Mail { to, mail } if to == USER && mail.text == "noted" => break,
            _ => {}
        }
    }
    assert_eq!(to_parent, ["CI is green"]);
    let prompt = system_prompt(&child, 0);
    assert!(
        prompt.contains("end with just `[silent]` and nothing is sent"),
        "{prompt}"
    );
    let root = scripts.lock()["a1"].clone();
    assert!(
        !system_prompt(&root, 0).contains("[silent]"),
        "the user's root has no parent agent"
    );
}

#[tokio::test]
async fn an_idle_agent_wakes_up_for_a_new_message() {
    let dir = tempfile::tempdir().unwrap();
    let (swarm, mut events, scripts) =
        swarm_with(vec![("a1", vec![say("first"), say("second")])], dir.path());

    let id = swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "hello".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "first");

    swarm.send(USER, &id, "again please").unwrap();
    let mut seen = Vec::new();
    let delivered = async {
        while let Some(event) = events.recv().await {
            match event {
                SwarmEvent::Mail { to, mail } if to == id => seen.push(format!("mail {}", mail.text)),
                SwarmEvent::MailRead { id: reader } if reader == id => seen.push("read".to_owned()),
                SwarmEvent::Mail { to, mail } if to == USER && mail.kind == "reply" => return mail.text,
                _ => {}
            }
        }
        panic!("event stream closed");
    };
    let reply = tokio::time::timeout(Duration::from_secs(10), delivered)
        .await
        .unwrap();
    assert_eq!(reply, "second");
    assert_eq!(seen, ["mail again please", "read"]);
    let root = scripts.lock()["a1"].clone();
    assert!(sent_text(&root, 1).contains("again please"));
}

#[tokio::test]
async fn mail_that_arrives_mid_turn_is_appended_to_the_next_tool_result() {
    let dir = tempfile::tempdir().unwrap();
    let (swarm, mut events, scripts) = swarm_with(
        vec![(
            "a1",
            vec![tool_call("list_agents", "t1", json!({})), say("noted")],
        )],
        dir.path(),
    );
    // Queue the message before the agent's first tool call completes.
    let id = swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "work".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    swarm.send(USER, &id, "priority change").unwrap();

    assert_eq!(reply_to(&mut events, USER).await, "noted");
    let root = scripts.lock()["a1"].clone();
    let second = sent_text(&root, 1);
    assert!(
        second.contains("New messages arrived") && second.contains("priority change"),
        "{second}"
    );
}

#[tokio::test]
async fn cancelling_an_agent_cancels_its_descendants() {
    let dir = tempfile::tempdir().unwrap();
    let (swarm, mut events, _scripts) = swarm_with(
        vec![
            (
                "a1",
                vec![
                    tool_call("spawn_agent", "t1", json!({"task": "sub", "name": "sub"})),
                    say("spawned"),
                ],
            ),
            (
                "a2",
                vec![
                    tool_call("wait", "t1", json!({"timeout_seconds": 60})),
                    say("unreachable"),
                ],
            ),
        ],
        dir.path(),
    );
    let root = swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "go".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "spawned");

    swarm.cancel(&root).unwrap();
    let all_cancelled = async {
        while !swarm.agents().iter().all(|a| a.status == AgentStatus::Cancelled) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(5), all_cancelled)
        .await
        .expect("agents still live");
    assert_eq!(swarm.agents().len(), 2);
}

#[tokio::test]
async fn a_cancelled_agent_that_stopped_is_released_from_memory() {
    let dir = tempfile::tempdir().unwrap();
    let (swarm, mut events, scripts) = swarm_with(
        vec![("a1", vec![say("first")]), ("a2", vec![say("kept")])],
        dir.path(),
    );
    let spawn = |task: &str| SpawnRequest {
        task: task.into(),
        ..Default::default()
    };
    let gone = swarm.spawn(USER, spawn("go")).await.unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "first");
    swarm.spawn(USER, spawn("stay")).await.unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "kept");
    // Counted: the scripts map's and the one `model` hands out, plus the agent's while held.
    let model = |id: &str| scripts.lock()[id].clone();
    assert!(
        Arc::strong_count(&model("a1")) > 2,
        "the live agent holds its model"
    );

    swarm.cancel(&gone).unwrap();
    let released = async {
        while Arc::strong_count(&model("a1")) > 2 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(5), released)
        .await
        .expect("the cancelled agent is still held, and its conversation with it");
    assert!(Arc::strong_count(&model("a2")) > 2, "a live agent keeps its own");
    let listed = swarm.agents();
    let info = listed.iter().find(|a| a.id == gone).unwrap();
    assert_eq!(info.status, AgentStatus::Cancelled);
    assert_eq!(info.model.as_deref(), Some("test-model"), "its model id is kept");
    swarm.shutdown().await;
}

#[tokio::test]
async fn agents_get_unique_names_and_can_be_addressed_by_them() {
    let dir = tempfile::tempdir().unwrap();
    let (swarm, mut events, scripts) = swarm_with(
        vec![
            (
                "a1",
                vec![
                    tool_call("spawn_agent", "t1", json!({"task": "x", "name": "Worker"})),
                    tool_call(
                        "spawn_agent",
                        "t2",
                        json!({"task": "Fix the flaky tests", "name": " "}),
                    ),
                    tool_call("spawn_agent", "t3", json!({"task": "y", "name": "worker"})),
                    tool_call("send_message", "t4", json!({"to": "worker-2", "message": "ping"})),
                    tool_call("wait", "t5", json!({"from": ["worker-2"], "timeout_seconds": 10})),
                    tool_call("cancel_agent", "t6", json!({"id": "fix-flaky-tests"})),
                    tool_call("send_message", "t7", json!({"to": "nobody", "message": "x"})),
                    tool_call("list_agents", "t8", json!({})),
                    say("done"),
                    say("late mail"),
                    say("late mail"),
                ],
            ),
            ("a2", vec![say("w1")]),
            (
                "a3",
                vec![
                    tool_call("wait", "t1", json!({"timeout_seconds": 60})),
                    say("unreachable"),
                ],
            ),
            ("a4", vec![say("w2"), say("pong")]),
        ],
        dir.path(),
    );
    swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "Coordinate the workers".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "done");

    let names: Vec<(String, String)> = swarm.agents().into_iter().map(|a| (a.id, a.name)).collect();
    let expected = [
        ("a1", "coordinate-workers"),
        ("a2", "worker"),
        ("a3", "fix-flaky-tests"),
        ("a4", "worker-2"),
    ];
    assert_eq!(names, expected.map(|(id, name)| (id.to_owned(), name.to_owned())));
    assert_eq!(swarm.resolve("worker-2").unwrap(), "a4");
    assert_eq!(swarm.resolve("a4").unwrap(), "a4");
    assert!(swarm.resolve("nobody").is_err());
    let cancelled = async {
        while swarm.agent("a3").unwrap().status != AgentStatus::Cancelled {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(5), cancelled)
        .await
        .expect("fix-flaky-tests still live");

    let pinged = sent_text(&scripts.lock()["a4"].clone(), 1);
    assert!(
        pinged.contains("coordinate-workers (a1)") && pinged.contains("ping"),
        "{pinged}"
    );
    let root = scripts.lock()["a1"].clone();
    let after_unknown = sent_text(&root, 7);
    assert!(
        after_unknown.contains("No agent with id or name `nobody`"),
        "{after_unknown}"
    );
    let listed = sent_text(&root, 8);
    assert!(
        listed.contains("worker-2 (a4): ") && listed.contains("parent coordinate-workers (a1)"),
        "{listed}"
    );
}

#[tokio::test]
async fn a_supervisor_notice_reaches_the_live_root_once_on_resume() {
    let dir = tempfile::tempdir().unwrap();
    let state_dir = dir.path().join("run");
    std::fs::create_dir_all(&state_dir).unwrap();
    let notice = state_dir.join(strands_swarm::SUPERVISOR_NOTICE);
    std::fs::write(&notice, "2026-01-01T00:00:00Z The new binary exited with code 70 (deadlock detected) 12s after it started, so the supervisor rolled back.\n").unwrap();
    let saved: strands_swarm::state::SwarmState = serde_json::from_value(json!({
        "run_id": "old",
        "cwd": dir.path(),
        "next_id": 3,
        "agents": [
            {"id": "a1", "name": "lead", "parent": "user", "cwd": dir.path(), "branch": null, "task": "lead",
             "model": null, "goal": null, "status": "idle", "mailbox": []},
            {"id": "a2", "name": "old-lead", "parent": "user", "cwd": dir.path(), "branch": null, "task": "t",
             "model": null, "goal": null, "status": "cancelled", "mailbox": []},
        ],
    }))
    .unwrap();
    let scripts: Scripts = Arc::new(Mutex::new(HashMap::new()));
    scripts
        .lock()
        .insert("a1".into(), MockMessageModel::shared(vec![say("noted")]));
    let mut config = SwarmConfig::new(mock_factory(&scripts), dir.path());
    config.state_dir = Some(state_dir.clone());
    let (swarm, mut events) = Swarm::resume(config, saved).await.unwrap();

    let mut delivered = Vec::new();
    let reply = tokio::time::timeout(Duration::from_secs(10), async {
        while let Some(event) = events.recv().await {
            match event {
                SwarmEvent::Mail { to, mail } if mail.from == strands_swarm::SUPERVISOR_SENDER => {
                    delivered.push((to, mail.text));
                }
                SwarmEvent::Mail { to, mail } if to == USER && mail.kind == "reply" => return mail.text,
                _ => {}
            }
        }
        panic!("event stream closed");
    })
    .await
    .expect("the root never answered the notice");
    assert_eq!(reply, "noted");
    assert_eq!(delivered.len(), 1, "only the live root: {delivered:?}");
    let (to, text) = &delivered[0];
    assert_eq!(to, "a1");
    assert!(
        text.starts_with("[From the swarm's supervisor (swarmctl)]\n2026-01-01T00:00:00Z The new binary exited with code 70"),
        "{text}"
    );
    let sent = sent_text(&scripts.lock()["a1"].clone(), 0);
    assert!(sent.contains("rolled back"), "{sent}");
    assert!(!notice.exists(), "a notice is delivered once");
    swarm.shutdown().await;
}

#[tokio::test]
async fn a_saved_run_without_names_resumes_with_derived_ones() {
    let dir = tempfile::tempdir().unwrap();
    let saved: strands_swarm::state::SwarmState = serde_json::from_value(json!({
        "run_id": "old",
        "cwd": dir.path(),
        "next_id": 3,
        "agents": [
            {"id": "a1", "parent": "user", "cwd": dir.path(), "branch": null, "task": "Review the parser",
             "model": null, "goal": null, "status": "cancelled", "mailbox": []},
            {"id": "a2", "name": "parser", "parent": "a1", "cwd": dir.path(), "branch": null, "task": "t",
             "model": null, "goal": null, "status": "cancelled", "mailbox": []},
        ],
    }))
    .unwrap();
    let map: Scripts = Arc::new(Mutex::new(HashMap::new()));
    let config = SwarmConfig::new(mock_factory(&map), dir.path());
    let (swarm, _events) = Swarm::resume(config, saved).await.unwrap();
    let names: Vec<String> = swarm.agents().into_iter().map(|a| a.name).collect();
    assert_eq!(names, ["review-parser", "parser"]);
    assert_eq!(swarm.snapshot().agents[0].name.as_deref(), Some("review-parser"));
}

async fn wait_for_status(swarm: &Swarm, id: &str, wanted: AgentStatus) {
    let reached = async {
        while swarm.agent(id).unwrap().status != wanted {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(5), reached)
        .await
        .unwrap_or_else(|_| panic!("{id} never became {wanted}"));
}

fn status_saved(swarm: &Swarm, id: &str) -> String {
    let state = swarm.snapshot();
    state.agents.iter().find(|a| a.id == id).unwrap().status.clone()
}

#[tokio::test]
async fn spawns_beyond_the_agent_limit_queue_and_start_when_a_slot_frees() {
    let dir = tempfile::tempdir().unwrap();
    let scripts: Scripts = Arc::new(Mutex::new(HashMap::new()));
    let script = |id: &str, responses| {
        scripts
            .lock()
            .insert(id.to_owned(), MockMessageModel::shared(responses))
    };
    script(
        "a1",
        vec![
            tool_call("spawn_agent", "t1", json!({"task": "two", "name": "two"})),
            tool_call("spawn_agent", "t2", json!({"task": "three", "name": "three"})),
            say("spawned"),
            say("ok"),
            say("ok"),
            say("ok"),
        ],
    );
    script("a2", vec![say("two done")]);
    script("a3", vec![say("three done"), say("got mail")]);
    let mut config = SwarmConfig::new(mock_factory(&scripts), dir.path());
    config.max_agents = 2;
    let (swarm, mut events) = Swarm::new(config);

    swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "one".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "spawned");
    assert_eq!(swarm.agent("a3").unwrap().status, AgentStatus::Queued);
    assert_eq!(status_saved(&swarm, "a3"), "queued");
    let a1 = scripts.lock()["a1"].clone();
    let queued = sent_text(&a1, 2);
    assert!(
        queued.contains("Queued three (a3): queued") && queued.contains("agent limit"),
        "{queued}"
    );
    swarm.send(USER, "a3", "hello").unwrap();

    swarm.cancel("a2").unwrap();
    // a2's own reply ("two done") may land in between.
    let mut replies = Vec::new();
    while replies.last().map(String::as_str) != Some("got mail") {
        replies.push(reply_to(&mut events, "a1").await);
    }
    replies.retain(|reply| reply != "two done");
    assert_eq!(replies, ["three done", "got mail"]);
    let a3 = scripts.lock()["a3"].clone();
    assert!(sent_text(&a3, 0).contains("three"));
    assert!(sent_text(&a3, 1).contains("hello"));
}

#[tokio::test]
async fn cancelling_a_queued_agent_drops_it() {
    let dir = tempfile::tempdir().unwrap();
    let map: Scripts = Arc::new(Mutex::new(HashMap::new()));
    map.lock()
        .insert("a1".into(), MockMessageModel::shared(vec![say("one")]));
    map.lock()
        .insert("a3".into(), MockMessageModel::shared(vec![say("three")]));
    let mut config = SwarmConfig::new(mock_factory(&map), dir.path());
    config.max_agents = 1;
    let (swarm, mut events) = Swarm::new(config);
    let task = |task: &str| SpawnRequest {
        task: task.into(),
        ..Default::default()
    };

    swarm.spawn(USER, task("one")).await.unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "one");
    swarm.spawn(USER, task("two")).await.unwrap();
    assert_eq!(swarm.agent("a2").unwrap().status, AgentStatus::Queued);
    swarm.cancel("a2").unwrap();
    assert_eq!(swarm.agent("a2").unwrap().status, AgentStatus::Cancelled);

    // a2 has no script: it is never built, even once a slot frees.
    swarm.cancel("a1").unwrap();
    swarm.spawn(USER, task("three")).await.unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "three");
    assert_eq!(swarm.agent("a2").unwrap().status, AgentStatus::Cancelled);
}

#[tokio::test]
async fn a_queued_agent_stays_queued_across_a_resume_until_there_is_room() {
    let dir = tempfile::tempdir().unwrap();
    let saved = || -> strands_swarm::state::SwarmState {
        serde_json::from_value(json!({
            "run_id": "old",
            "cwd": dir.path(),
            "next_id": 3,
            "agents": [
                {"id": "a1", "parent": "user", "cwd": dir.path(), "branch": null, "task": "one",
                 "model": null, "goal": null, "status": "idle", "mailbox": []},
                {"id": "a2", "parent": "a1", "cwd": dir.path(), "branch": null, "task": "late",
                 "model": null, "goal": null, "status": "queued", "mailbox": []},
            ],
        }))
        .unwrap()
    };
    let map: Scripts = Arc::new(Mutex::new(HashMap::new()));
    map.lock()
        .insert("a1".into(), MockMessageModel::shared(vec![say("ok")]));
    map.lock()
        .insert("a2".into(), MockMessageModel::shared(vec![say("late start")]));

    let mut config = SwarmConfig::new(mock_factory(&map), dir.path());
    config.max_agents = 1;
    let (swarm, _events) = Swarm::resume(config.clone(), saved()).await.unwrap();
    assert_eq!(swarm.agent("a2").unwrap().status, AgentStatus::Queued);
    assert_eq!(status_saved(&swarm, "a2"), "queued");
    swarm.shutdown().await;

    config.max_agents = 2;
    let (swarm, mut events) = Swarm::resume(config, saved()).await.unwrap();
    assert_eq!(reply_to(&mut events, "a1").await, "late start");
    wait_for_status(&swarm, "a2", AgentStatus::Idle).await;
    assert!(sent_text(&map.lock()["a2"].clone(), 0).contains("late"));
}

#[tokio::test]
async fn raising_the_agent_limit_at_runtime_starts_queued_agents_and_survives_a_resume() {
    let dir = tempfile::tempdir().unwrap();
    let saved: strands_swarm::state::SwarmState = serde_json::from_value(json!({
        "run_id": "old",
        "cwd": dir.path(),
        "next_id": 3,
        "agents": [
            {"id": "a1", "parent": "user", "cwd": dir.path(), "branch": null, "task": "one",
             "model": null, "goal": null, "status": "idle", "mailbox": []},
            {"id": "a2", "parent": "a1", "cwd": dir.path(), "branch": null, "task": "late",
             "model": null, "goal": null, "status": "queued", "mailbox": []},
        ],
    }))
    .unwrap();
    let map: Scripts = Arc::new(Mutex::new(HashMap::new()));
    map.lock()
        .insert("a1".into(), MockMessageModel::shared(vec![say("ok")]));
    map.lock()
        .insert("a2".into(), MockMessageModel::shared(vec![say("late start")]));
    let mut config = SwarmConfig::new(mock_factory(&map), dir.path());
    config.max_agents = 1;
    let (swarm, mut events) = Swarm::resume(config.clone(), saved).await.unwrap();
    assert_eq!(swarm.agent("a2").unwrap().status, AgentStatus::Queued);
    assert!(swarm.set_max_agents(Some(0)).is_err());

    swarm.set_max_agents(Some(2)).unwrap();
    assert_eq!(swarm.max_agents(), 2);
    assert_eq!(reply_to(&mut events, "a1").await, "late start");
    let state = swarm.snapshot();
    assert_eq!(state.max_agents, Some(2));
    swarm.shutdown().await;

    // The saved override wins over the configured limit; `None` returns to it.
    let (swarm, _events) = Swarm::resume(config, state).await.unwrap();
    assert_eq!(swarm.max_agents(), 2);
    swarm.set_max_agents(None).unwrap();
    assert_eq!(swarm.max_agents(), 1);
    assert_eq!(swarm.snapshot().max_agents, None);
}

#[tokio::test]
async fn cancelling_frees_capacity_at_once_for_the_whole_subtree() {
    let dir = tempfile::tempdir().unwrap();
    let map: Scripts = Arc::new(Mutex::new(HashMap::new()));
    let script = |id: &str, responses| {
        map.lock()
            .insert(id.to_owned(), MockMessageModel::shared(responses))
    };
    script(
        "a1",
        vec![
            tool_call("spawn_agent", "t1", json!({"task": "sub", "name": "sub"})),
            say("spawned"),
        ],
    );
    script("a2", vec![say("never")]);
    script(
        "a3",
        vec![
            tool_call("list_agents", "l1", json!({})),
            tool_call("list_agents", "l2", json!({"include_cancelled": true})),
            say("three"),
        ],
    );
    // Every agent but a1 takes a while to build, so a2 is still starting when a1 is cancelled.
    let inner = mock_factory(&map);
    let factory: AgentFactory = Arc::new(move |spec: AgentSpec| {
        let inner = inner.clone();
        Box::pin(async move {
            if spec.id != "a1" {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            inner(spec).await
        })
    });
    let mut config = SwarmConfig::new(factory, dir.path());
    config.max_agents = 2;
    let (swarm, mut events) = Swarm::new(config);
    let task = |task: &str| SpawnRequest {
        task: task.into(),
        ..Default::default()
    };

    swarm.spawn(USER, task("one")).await.unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "spawned");
    assert_eq!(swarm.agent("a2").unwrap().status, AgentStatus::Starting);

    swarm.cancel("a1").unwrap();
    for id in ["a1", "a2"] {
        assert_eq!(swarm.agent(id).unwrap().status, AgentStatus::Cancelled, "{id}");
    }
    swarm.spawn(USER, task("three")).await.unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "three");
    let a3 = map.lock()["a3"].clone();
    let listed = sent_text(&a3, 1);
    assert!(listed.contains("2 cancelled agents not shown"), "{listed}");
    assert!(!listed.contains("(a2)"), "{listed}");
    let all = sent_text(&a3, 2);
    assert!(all.contains("(a2): cancelled"), "{all}");
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert_eq!(
        swarm.agent("a2").unwrap().status,
        AgentStatus::Cancelled,
        "stays cancelled"
    );
}

#[tokio::test]
async fn wait_returns_at_once_when_the_awaited_agents_are_cancelled() {
    let dir = tempfile::tempdir().unwrap();
    let wait = |id: &str, from: serde_json::Value| {
        tool_call("wait", id, json!({"from": from, "timeout_seconds": 600}))
    };
    let (swarm, mut events, scripts) = swarm_with(
        vec![
            (
                "a1",
                vec![
                    tool_call("spawn_agent", "t1", json!({"task": "sub", "name": "sub"})),
                    wait("w1", json!(["a2"])),
                    wait("w2", json!(["sub"])),
                    wait("w3", json!(["nobody"])),
                    say("done"),
                ],
            ),
            ("a2", vec![wait("w1", json!([])), say("unreachable")]),
        ],
        dir.path(),
    );
    swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "go".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let a1 = scripts.lock()["a1"].clone();
    eventually("a1 waits", || a1.calls().len() >= 2).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    swarm.cancel("a2").unwrap();

    assert_eq!(reply_to(&mut events, USER).await, "done");
    let notice = "No messages will come: sub (a2) cancelled.";
    assert!(sent_text(&a1, 2).contains(notice), "{}", sent_text(&a1, 2));
    assert_eq!(sent_text(&a1, 3).matches(notice).count(), 2, "already cancelled");
    assert!(sent_text(&a1, 4).contains("No agent with id or name `nobody`"));
}

#[tokio::test]
async fn wait_without_from_returns_once_every_child_is_cancelled() {
    let dir = tempfile::tempdir().unwrap();
    let (swarm, mut events, scripts) = swarm_with(
        vec![
            (
                "a1",
                vec![
                    tool_call("spawn_agent", "t1", json!({"task": "sub", "name": "sub"})),
                    tool_call("wait", "w1", json!({"timeout_seconds": 600})),
                    say("done"),
                ],
            ),
            (
                "a2",
                vec![tool_call("wait", "w1", json!({"timeout_seconds": 600}))],
            ),
        ],
        dir.path(),
    );
    swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "go".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let a1 = scripts.lock()["a1"].clone();
    eventually("a1 waits", || a1.calls().len() >= 2).await;
    swarm.cancel("a2").unwrap();

    assert_eq!(reply_to(&mut events, USER).await, "done");
    assert!(sent_text(&a1, 2).contains("Every agent you spawned is cancelled"));
}

#[tokio::test]
async fn an_agent_cannot_message_itself() {
    let dir = tempfile::tempdir().unwrap();
    let (swarm, mut events, scripts) = swarm_with(
        vec![(
            "a1",
            vec![
                tool_call(
                    "send_message",
                    "s1",
                    json!({"to": "a1", "message": "note to self"}),
                ),
                say("done"),
            ],
        )],
        dir.path(),
    );
    swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "go".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "done");
    let a1 = scripts.lock()["a1"].clone();
    assert!(sent_text(&a1, 1).contains("You cannot message yourself"));
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(a1.calls().len(), 2, "no extra turn for its own message");
    assert_eq!(swarm.agent("a1").unwrap().unread, 0);
}

#[tokio::test]
async fn an_agent_cancelled_in_wait_stops_and_its_child_stays_quiet() {
    let dir = tempfile::tempdir().unwrap();
    let (swarm, mut events, scripts) = swarm_with(
        vec![
            (
                "a1",
                vec![
                    tool_call("spawn_agent", "t1", json!({"task": "sub", "name": "sub"})),
                    tool_call("wait", "w1", json!({"from": ["user"], "timeout_seconds": 600})),
                    say("unreachable"),
                ],
            ),
            (
                "a2",
                vec![
                    tool_call("wait", "w1", json!({"from": ["parent"], "timeout_seconds": 600})),
                    say("late reply"),
                ],
            ),
        ],
        dir.path(),
    );
    swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "go".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let (a1, a2) = {
        let scripts = scripts.lock();
        (scripts["a1"].clone(), scripts["a2"].clone())
    };
    eventually("both wait", || a1.calls().len() >= 2 && !a2.calls().is_empty()).await;
    swarm.cancel("a1").unwrap();

    eventually("the actors finish", || !swarm.is_busy()).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        (a1.calls().len(), a2.calls().len()),
        (2, 1),
        "no turn after the cancel"
    );
    let mut mail = Vec::new();
    while let Ok(event) = events.try_recv() {
        if let SwarmEvent::Mail { to, mail: m } = event {
            mail.push((to, m.text));
        }
    }
    assert!(mail.is_empty(), "{mail:?}");
}

#[tokio::test]
async fn a_goal_the_judge_keeps_rejecting_ends_the_turn_after_the_attempt_cap() {
    use strands_agents::plugins::vended::{GoalLoop, ValidationOutcome};
    let dir = tempfile::tempdir().unwrap();
    let attempts: Vec<MockResponse> = (1..=GOAL_MAX_ATTEMPTS)
        .map(|i| say(&format!("attempt {i}")))
        .collect();
    let model = MockMessageModel::shared(attempts);
    let factory: AgentFactory = {
        let model = model.clone();
        Arc::new(move |spec: AgentSpec| {
            let model = model.clone();
            Box::pin(async move {
                let never = GoalLoop::sync_validator(|_| Ok(ValidationOutcome::failed("not yet")));
                Agent::builder()
                    .id(spec.id)
                    .model(model)
                    .system_prompt(spec.instructions)
                    .tools(spec.tools)
                    .plugin(strands_swarm::bounded_goal(never))
                    .no_sandbox()
                    .build()
            })
        })
    };
    let (swarm, mut events) = Swarm::new(SwarmConfig::new(factory, dir.path()));
    swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "impossible".into(),
                goal: Some("impossible".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let last = format!("attempt {GOAL_MAX_ATTEMPTS}");
    assert_eq!(reply_to(&mut events, USER).await, last);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(model.calls().len(), GOAL_MAX_ATTEMPTS);
    assert_eq!(swarm.agent("a1").unwrap().status, AgentStatus::Idle);
    let system = format!("{:?}", model.calls()[0].1.system_prompt);
    assert!(
        system.contains(&format!("After {GOAL_MAX_ATTEMPTS} rejected attempts")),
        "{system}"
    );
}

#[tokio::test(start_paused = true)]
async fn a_failing_goal_agent_backs_off_and_reports_a_streak_once() {
    use strands_agents::plugins::vended::{GoalLoop, ValidationOutcome};
    let dir = tempfile::tempdir().unwrap();
    let map: Scripts = Arc::new(Mutex::new(HashMap::new()));
    let fail = || MockResponse::Error(strands_agents::Error::other("throttled"));
    map.lock().insert(
        "a1".into(),
        MockMessageModel::shared(
            [tool_call(
                "spawn_agent",
                "t1",
                json!({"task": "sub", "name": "sub", "goal": "work"}),
            )]
            .into_iter()
            .chain((0..6).map(|_| say("noted")))
            .collect(),
        ),
    );
    map.lock().insert(
        "a2".into(),
        MockMessageModel::shared(vec![fail(), fail(), fail(), say("recovered")]),
    );
    let scripts = map.clone();
    let factory: AgentFactory = Arc::new(move |spec: AgentSpec| {
        let model = scripts.lock()[&spec.id].clone();
        Box::pin(async move {
            let mut builder = Agent::builder()
                .id(spec.id)
                .model(model)
                .system_prompt(spec.instructions)
                .tools(spec.tools)
                .no_sandbox();
            if spec.goal.is_some() {
                let pass = GoalLoop::sync_validator(|_| Ok(ValidationOutcome::passed()));
                builder = builder.plugin(strands_swarm::bounded_goal(pass));
            }
            builder.build()
        })
    });
    let (swarm, mut events) = Swarm::new(SwarmConfig::new(factory, dir.path()));
    swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "go".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let started = tokio::time::Instant::now();
    let mut errors = 0;
    let reply = loop {
        match events.recv().await.expect("events") {
            SwarmEvent::Mail { to, mail } if to == "a1" && mail.kind == "error" => errors += 1,
            SwarmEvent::Mail { to, mail } if to == "a1" && mail.kind == "reply" => break mail.text,
            _ => {}
        }
    };
    assert_eq!(reply, "recovered");
    assert_eq!(errors, 1, "one error mail for the whole streak");
    // Backoff before retries 1, 2 and 3: 15 s + 30 s + 60 s.
    assert!(
        started.elapsed() >= Duration::from_secs(105),
        "{:?}",
        started.elapsed()
    );
}

/// `git -C dir`, blind to repository variables inherited from e.g. `git rebase --exec`, which
/// would otherwise point these commands at the repository running the tests.
fn git_in(dir: &Path) -> std::process::Command {
    let mut command = std::process::Command::new("git");
    command.arg("-C").arg(dir);
    for var in strands_swarm::worktree::GIT_LOCATION_VARS {
        command.env_remove(var);
    }
    command
}

/// `git <args>` in `dir` with a throwaway identity: its trimmed stdout; panics when git fails.
fn git(dir: &Path, args: &[&str]) -> String {
    let ident = [
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "-c",
        "commit.gpgsign=false",
    ];
    let output = git_in(dir).args(ident).args(args).output().unwrap();
    assert!(output.status.success(), "git {args:?}: {output:?}");
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

/// A repo on `main` at `<dir>/repo` with `files` in one commit and a committer identity set
/// (for the swarm's own git calls).
fn init_repo(dir: &Path, files: &[(&str, &str)]) -> std::path::PathBuf {
    let repo = dir.join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    for (path, text) in files {
        let path = repo.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
    git(&repo, &["init", "-q", "-b", "main"]);
    git(&repo, &["config", "user.name", "t"]);
    git(&repo, &["config", "user.email", "t@t"]);
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-q", "--allow-empty", "-m", "init"]);
    repo
}

#[tokio::test]
async fn a_worktree_agent_gets_its_own_checkout_and_branch() {
    let dir = tempfile::tempdir().unwrap();
    let repo = init_repo(dir.path(), &[("sub/file.txt", "x")]);

    let (swarm, _events, _scripts) = swarm_with(vec![("a1", vec![say("ok")])], &repo.join("sub"));
    let id = swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "t".into(),
                worktree: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let info = swarm.agent(&id).unwrap();
    let branch = info.branch.unwrap();
    assert_eq!(branch, format!("swarm/{}/{id}", swarm.run_id()));
    let expected = dir
        .path()
        .join("repo.swarm")
        .join(swarm.run_id())
        .join(&id)
        .join("sub");
    assert_eq!(info.cwd, expected);
    assert!(expected.join("file.txt").exists());
}

#[tokio::test]
async fn worktree_agents_get_a_private_cargo_target_dir_that_their_children_share() {
    let dir = tempfile::tempdir().unwrap();
    let repo = init_repo(dir.path(), &[]);
    let map: Scripts = Arc::new(Mutex::new(HashMap::new()));
    for id in ["a1", "a2"] {
        map.lock()
            .insert(id.to_owned(), MockMessageModel::shared(vec![say("ok")]));
    }
    let envs: Arc<Mutex<HashMap<String, std::collections::BTreeMap<String, String>>>> = Arc::default();
    let inner = mock_factory(&map);
    let recorded = envs.clone();
    let factory: AgentFactory = Arc::new(move |spec: AgentSpec| {
        recorded.lock().insert(spec.id.clone(), spec.env.clone());
        inner(spec)
    });
    let mut config = SwarmConfig::new(factory, &repo);
    config.cargo_target_dir = Some(dir.path().join("target"));
    let (swarm, mut events) = Swarm::new(config);

    let worktree = SpawnRequest {
        task: "t".into(),
        worktree: true,
        ..Default::default()
    };
    let parent = swarm.spawn(USER, worktree).await.unwrap();
    let child = SpawnRequest {
        task: "t".into(),
        ..Default::default()
    };
    let child = swarm.spawn(&parent, child).await.unwrap();
    reply_to(&mut events, USER).await;
    reply_to(&mut events, &parent).await;

    let private = format!(
        "{}-worktrees/{}/{parent}",
        dir.path().join("target").display(),
        swarm.run_id()
    );
    let envs = envs.lock();
    assert_eq!(envs[&parent]["CARGO_TARGET_DIR"], private);
    assert_eq!(envs[&child]["CARGO_TARGET_DIR"], private);
}

/// A factory like [`mock_factory`] whose agents persist their conversation under `sessions`.
fn session_factory(scripts: &Scripts, sessions: std::path::PathBuf) -> AgentFactory {
    let scripts = scripts.clone();
    Arc::new(move |spec: AgentSpec| {
        let model = scripts
            .lock()
            .get(&spec.id)
            .cloned()
            .expect("no script for agent");
        let sessions = sessions.clone();
        Box::pin(async move {
            let manager = strands_swarm::agent_session_manager(&sessions, &spec.id)?;
            Agent::builder()
                .id(spec.id)
                .model(model)
                .system_prompt(spec.instructions)
                .tools(spec.tools)
                .session_manager(manager)
                .no_sandbox()
                .build()
        })
    })
}

#[tokio::test]
async fn a_resumed_run_restores_conversations_and_wakes_interrupted_agents() {
    let dir = tempfile::tempdir().unwrap();
    let state_dir = dir.path().join("run");
    let scripts: Scripts = Arc::new(Mutex::new(HashMap::new()));
    let script = |id: &str, responses| {
        scripts
            .lock()
            .insert(id.to_owned(), MockMessageModel::shared(responses))
    };
    script(
        "a1",
        vec![
            tool_call("spawn_agent", "t1", json!({"task": "sub", "name": "sub"})),
            say("spawned"),
        ],
    );
    script(
        "a2",
        vec![tool_call("wait", "w1", json!({"timeout_seconds": 600}))],
    );
    let mut config = SwarmConfig::new(session_factory(&scripts, state_dir.join("sessions")), dir.path());
    config.state_dir = Some(state_dir.clone());
    let (swarm, mut events) = Swarm::new(config.clone());

    swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "go".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "spawned");
    let a2_waiting = async {
        while scripts.lock()["a2"].call_count() < 1
            || swarm.agent("a2").unwrap().status != AgentStatus::Running
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    tokio::time::timeout(Duration::from_secs(5), a2_waiting)
        .await
        .unwrap();
    swarm.shutdown().await;

    let saved = strands_swarm::state::load(&state_dir)
        .await
        .unwrap()
        .expect("state saved");
    let statuses: Vec<(&str, &str)> = saved
        .agents
        .iter()
        .map(|a| (a.id.as_str(), a.status.as_str()))
        .collect();
    assert_eq!(statuses, vec![("a1", "idle"), ("a2", "running")]);

    script("a1", vec![say("got it")]);
    script("a2", vec![say("resumed")]);
    let (resumed, mut events) = Swarm::resume(config, saved).await.unwrap();

    let mut history = HashMap::new();
    let reply = tokio::time::timeout(Duration::from_secs(10), async {
        while let Some(event) = events.recv().await {
            match event {
                SwarmEvent::History {
                    id,
                    messages,
                    omitted,
                } => {
                    assert_eq!(omitted, 0);
                    history.insert(id, messages);
                }
                SwarmEvent::Mail { to, mail } if to == USER && mail.kind == "reply" => return mail.text,
                _ => {}
            }
        }
        panic!("event stream closed");
    })
    .await
    .expect("timed out waiting for a reply");
    assert_eq!(reply, "got it");
    let a1_history = &history["a1"];
    assert_eq!(a1_history.first().map(|m| m.role), Some(Role::User));
    assert!(a1_history.iter().any(|m| m
        .content
        .iter()
        .any(|b| matches!(b, ContentBlock::ToolUse(call) if call.name == "spawn_agent"))));
    assert!(
        history.contains_key("a2"),
        "interrupted agents report their history too"
    );
    let a2_call = sent_text(&scripts.lock()["a2"].clone(), 0);
    assert!(a2_call.contains("The swarm was restarted"), "{a2_call}");
    assert!(a2_call.contains("No tool call was in flight."), "{a2_call}");
    let a1_call = sent_text(&scripts.lock()["a1"].clone(), 0);
    assert!(
        a1_call.contains("spawned") && a1_call.contains("resumed"),
        "{a1_call}"
    );
    assert_eq!(resumed.agents().len(), 2);
}

/// The system prompt and tool specs an agent sends, serialized: the prompt cache is keyed on these
/// bytes, so a resumed agent must send exactly what it sent before the restart.
fn request_prefix(model: &MockMessageModel, call: usize) -> String {
    let (_, options) = &model.calls()[call];
    format!(
        "{}\n{}",
        serde_json::to_string(&options.system_prompt).unwrap(),
        serde_json::to_string(&options.tool_specs).unwrap()
    )
}

#[tokio::test]
async fn a_resumed_agent_sends_the_same_system_prompt_and_tools() {
    let dir = tempfile::tempdir().unwrap();
    let state_dir = dir.path().join("run");
    let memory = dir.path().join("memory.md");
    std::fs::write(&memory, "- 2026-01-01: the user likes short replies\n").unwrap();
    let scripts = scripts_of(vec![("a1", vec![say("done")])]);
    let mut config = SwarmConfig::new(session_factory(&scripts, state_dir.join("sessions")), dir.path());
    config.state_dir = Some(state_dir.clone());
    config.memory = Some(Arc::new(strands_swarm::Memory::new(&memory)));
    let (swarm, mut events) = Swarm::new(config.clone());
    spawn_root(&swarm, "hello").await;
    assert_eq!(reply_to(&mut events, USER).await, "done");
    let before = request_prefix(&scripts.lock()["a1"].clone(), 0);
    swarm.shutdown().await;

    let saved = strands_swarm::state::load(&state_dir).await.unwrap().unwrap();
    scripts
        .lock()
        .insert("a1".into(), MockMessageModel::shared(vec![say("again")]));
    let (resumed, mut events) = Swarm::resume(config, saved).await.unwrap();
    resumed.send(USER, "a1", "next").unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "again");
    let after = request_prefix(&scripts.lock()["a1"].clone(), 0);
    assert!(after.contains("likes short replies"), "{after}");
    assert_eq!(after, before);
}

#[tokio::test]
async fn reload_installs_the_binary_keeps_the_previous_one_and_signals_the_host() {
    let dir = tempfile::tempdir().unwrap();
    let map: Scripts = Arc::new(Mutex::new(HashMap::new()));
    let mut config = SwarmConfig::new(mock_factory(&map), dir.path());
    config.state_dir = Some(dir.path().join("run"));
    let (swarm, _events) = Swarm::new(config);
    let bin = dir.path().join("run/bin");

    swarm
        .reload(Path::new("/bin/true"), ReloadWhen::Idle)
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(1), swarm.reload_requested())
        .await
        .expect("reload not signalled");
    assert!(bin.join("current").exists());

    swarm
        .reload(Path::new("/bin/true"), ReloadWhen::Now)
        .await
        .unwrap();
    assert!(bin.join("previous").exists());

    let error = swarm
        .reload(Path::new("/bin/false"), ReloadWhen::Now)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("--version failed"), "{error}");
}

#[tokio::test]
async fn reload_needs_a_persistent_run() {
    let dir = tempfile::tempdir().unwrap();
    let (swarm, _events, _scripts) = swarm_with(vec![], dir.path());
    let error = swarm
        .reload(Path::new("/bin/true"), ReloadWhen::Now)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("--run"), "{error}");
}

/// Resolves model `id` to the mock registered under `model:<id>`; anything else is rejected.
fn mock_resolver(scripts: &Scripts) -> ModelResolver {
    let scripts = scripts.clone();
    Arc::new(move |model: Option<String>| {
        let found = model
            .as_ref()
            .and_then(|id| scripts.lock().get(&format!("model:{id}")).cloned());
        Box::pin(async move {
            found
                .map(|mock| mock as Arc<dyn Model>)
                .ok_or_else(|| strands_agents::Error::other(format!("unknown model {model:?}")))
        })
    })
}

fn scripts_of(scripts: Vec<(&str, Vec<MockResponse>)>) -> Scripts {
    let map: Scripts = Arc::new(Mutex::new(HashMap::new()));
    for (id, responses) in scripts {
        map.lock()
            .insert(id.to_owned(), MockMessageModel::shared(responses));
    }
    map
}

async fn spawn_root(swarm: &Swarm, task: &str) -> String {
    let request = SpawnRequest {
        task: task.into(),
        ..Default::default()
    };
    swarm.spawn(USER, request).await.unwrap()
}

#[tokio::test]
async fn switching_a_model_takes_effect_on_the_next_turn_and_rejects_unknown_ids() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![
        ("a1", vec![say("first"), say("still a")]),
        ("model:b", vec![say("from b")]),
    ]);
    let mut config = SwarmConfig::new(mock_factory(&scripts), dir.path());
    config.models = Some(mock_resolver(&scripts));
    let (swarm, mut events) = Swarm::new(config);
    let id = spawn_root(&swarm, "hello").await;
    assert_eq!(reply_to(&mut events, USER).await, "first");

    let error = swarm.set_model(&id, Some("bogus".into())).await.unwrap_err();
    assert!(error.to_string().contains("bogus"), "{error}");
    swarm.send(USER, &id, "and now").unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "still a");
    assert_eq!(swarm.snapshot().agents[0].model, None);

    let model = swarm.set_model(&id, Some("b".into())).await.unwrap();
    assert_eq!(model, "test-model");
    swarm.send(USER, &id, "switched").unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "from b");
    let b = scripts.lock()["model:b"].clone();
    let call = sent_text(&b, 0);
    assert!(call.contains("still a") && call.contains("switched"), "{call}");
    assert_eq!(swarm.snapshot().agents[0].model.as_deref(), Some("b"));
}

#[tokio::test]
async fn switching_needs_a_resolver_and_a_known_agent() {
    let dir = tempfile::tempdir().unwrap();
    let (swarm, mut events, _scripts) = swarm_with(vec![("a1", vec![say("hi")])], dir.path());
    let id = spawn_root(&swarm, "hello").await;
    assert_eq!(reply_to(&mut events, USER).await, "hi");
    let error = swarm.set_model(&id, Some("b".into())).await.unwrap_err();
    assert!(error.to_string().contains("not available"), "{error}");
    let error = swarm.set_model("a9", None).await.unwrap_err();
    assert!(error.to_string().contains("Unknown agent"), "{error}");
}

#[tokio::test]
async fn a_switched_model_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let state_dir = dir.path().join("run");
    let scripts = scripts_of(vec![("a1", vec![say("first")]), ("model:b", vec![])]);
    let built: Arc<Mutex<Vec<Option<String>>>> = Arc::default();
    let inner = mock_factory(&scripts);
    let recorded = built.clone();
    let factory: AgentFactory = Arc::new(move |spec: AgentSpec| {
        recorded.lock().push(spec.model.clone());
        inner(spec)
    });
    let mut config = SwarmConfig::new(factory, dir.path());
    config.state_dir = Some(state_dir.clone());
    config.models = Some(mock_resolver(&scripts));
    let (swarm, mut events) = Swarm::new(config.clone());
    let id = spawn_root(&swarm, "hello").await;
    assert_eq!(reply_to(&mut events, USER).await, "first");
    swarm.set_model(&id, Some("b".into())).await.unwrap();
    swarm.shutdown().await;

    let saved = strands_swarm::state::load(&state_dir)
        .await
        .unwrap()
        .expect("state saved");
    assert_eq!(saved.agents[0].model.as_deref(), Some("b"));
    let (resumed, _events) = Swarm::resume(config, saved).await.unwrap();
    let rebuilt = async {
        while built.lock().len() < 2 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), rebuilt)
        .await
        .expect("agent not rebuilt");
    assert_eq!(*built.lock(), vec![None, Some("b".to_owned())]);
    assert_eq!(resumed.snapshot().agents[0].model.as_deref(), Some("b"));
}

/// Many agents resuming at once while the swarm is snapshotted and listed concurrently: a lock
/// order inversion between agent startup and `snapshot` would hang here.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_large_run_resumes_without_deadlocking_under_concurrent_snapshots() {
    const AGENTS: usize = 24;
    let dir = tempfile::tempdir().unwrap();
    let state_dir = dir.path().join("run");
    let scripts: Scripts = Arc::new(Mutex::new(HashMap::new()));
    let mut agents = Vec::new();
    for n in 1..=AGENTS {
        let id = format!("a{n}");
        let parent = match n {
            1 => USER.to_owned(),
            n => format!("a{}", n / 2),
        };
        scripts.lock().insert(
            id.clone(),
            MockMessageModel::shared((0..50).map(|_| say("ok")).collect()),
        );
        agents.push(json!({
            "id": id, "name": format!("agent-{n}"), "parent": parent, "cwd": dir.path(),
            "branch": null, "task": "t", "model": null, "goal": null,
            "status": if n % 2 == 0 { "running" } else { "idle" },
            "mailbox": [{"from": "user", "from_label": "user", "kind": "message", "text": "hi"}],
        }));
    }
    let saved = json!({"run_id": "r", "cwd": dir.path(), "next_id": AGENTS + 1, "agents": agents});
    let saved = serde_json::from_value(saved).unwrap();
    let mut config = SwarmConfig::new(session_factory(&scripts, state_dir.join("sessions")), dir.path());
    config.state_dir = Some(state_dir);

    let (swarm, _events) = Swarm::resume(config, saved).await.unwrap();
    let hammer = {
        let swarm = swarm.clone();
        tokio::spawn(async move {
            loop {
                let _ = swarm.snapshot();
                let _ = swarm.agents();
                tokio::task::yield_now().await;
            }
        })
    };
    let all_ran = async {
        while scripts.lock().values().any(|model| model.call_count() == 0) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    let ran = tokio::time::timeout(Duration::from_secs(30), all_ran).await;
    hammer.abort();
    assert!(
        ran.is_ok(),
        "agents did not all run: {:?}",
        strands_swarm::watchdog::check()
    );
    swarm.shutdown().await;
}

fn usage(input: u64, cache_read: u64, output: u64) -> strands_agents::types::Usage {
    strands_agents::types::Usage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: input + cache_read + output,
        cache_read_input_tokens: Some(cache_read),
        ..Default::default()
    }
}

/// A mock with a known context window, so utilization can be reported.
struct Windowed(Arc<MockMessageModel>);

impl Model for Windowed {
    fn model_id(&self) -> Option<String> {
        self.0.model_id()
    }

    fn context_window_limit(&self) -> Option<u64> {
        Some(4000)
    }

    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        options: &'a strands_agents::models::StreamOptions,
    ) -> strands_agents::models::ModelEventStream<'a> {
        self.0.stream(messages, options)
    }
}

/// [`mock_factory`] with every model given a 4000-token context window.
fn windowed_factory(scripts: &Scripts) -> AgentFactory {
    let scripts = scripts.clone();
    Arc::new(move |spec: AgentSpec| {
        let model = Windowed(scripts.lock()[&spec.id].clone());
        Box::pin(async move {
            Agent::builder()
                .id(spec.id)
                .model(model)
                .tools(spec.tools)
                .no_sandbox()
                .build()
        })
    })
}

type Report = (u64, u64, Option<f64>);

/// Every `Usage` event for `id` until `id` replies to the user.
async fn usage_until_reply(events: &mut UnboundedReceiver<SwarmEvent>, id: &str) -> Vec<Report> {
    let mut seen = Vec::new();
    let wait = async {
        while let Some(event) = events.recv().await {
            match event {
                SwarmEvent::Usage {
                    id: agent,
                    input_tokens,
                    output_tokens,
                    context,
                    ..
                } if agent == id => seen.push((input_tokens, output_tokens, context)),
                SwarmEvent::Mail { to, mail } if to == USER && mail.kind == "reply" => return,
                _ => {}
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(10), wait)
        .await
        .expect("timed out waiting for a reply");
    seen
}

#[tokio::test]
async fn token_usage_and_context_are_reported_after_each_model_call_and_survive_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let state_dir = dir.path().join("run");
    let call = Message::new(
        Role::Assistant,
        vec![ContentBlock::ToolUse(ToolUseBlock::new(
            "list_agents",
            "t1",
            json!({}),
        ))],
    );
    let scripts = scripts_of(vec![(
        "a1",
        vec![
            MockResponse::with_usage(call, usage(100, 900, 10)),
            MockResponse::with_usage(Message::assistant("done"), usage(5, 0, 1)),
        ],
    )]);
    let mut config = SwarmConfig::new(windowed_factory(&scripts), dir.path());
    config.state_dir = Some(state_dir.clone());
    // The compaction point is min(85% of the 4000-token window, 2000) = 2000 tokens.
    config.compact_at = 2000;
    let (swarm, mut events) = Swarm::new(config.clone());
    let id = spawn_root(&swarm, "count").await;

    let seen = usage_until_reply(&mut events, &id).await;
    // Context is the last call's input over the compaction point.
    assert_eq!(
        seen.first(),
        Some(&(1000, 10, Some(0.5))),
        "mid-turn report: {seen:?}"
    );
    assert_eq!(seen.last(), Some(&(1005, 11, Some(5.0 / 2000.0))), "{seen:?}");
    // The cost report has the same tokens by model, and per-call stats.
    let report = swarm.cost();
    let ledger = &report.agents[0].ledger;
    let on_model = ledger.by_model["test-model"];
    assert_eq!(
        (on_model.calls, on_model.input, on_model.cache_read),
        (2, 1005, 900)
    );
    let stats = ledger.call_stats().unwrap();
    assert_eq!((stats.calls, stats.median_input), (2, 502));
    assert_eq!(report.total().render(), "unknown", "the mock model has no price");
    swarm.shutdown().await;

    let saved = strands_swarm::state::load(&state_dir)
        .await
        .unwrap()
        .expect("state saved");
    assert_eq!(
        (saved.agents[0].input_tokens, saved.agents[0].output_tokens),
        (1005, 11)
    );
    assert_eq!(saved.agents[0].context, Some(5.0 / 2000.0));
    assert_eq!(&saved.agents[0].ledger, ledger, "the cost ledger is saved");
    // 900 of the 1005 input tokens came from the prompt cache.
    assert_eq!(
        (
            saved.agents[0].cache_read_tokens,
            saved.agents[0].cache_write_tokens
        ),
        (900, 0)
    );
    let (resumed, mut events) = Swarm::resume(config, saved.clone()).await.unwrap();
    assert_eq!(resumed.cost().agents[0].ledger, saved.agents[0].ledger);
    let restored = async {
        loop {
            if let Some(SwarmEvent::Usage {
                input_tokens,
                output_tokens,
                context,
                ..
            }) = events.recv().await
            {
                return (input_tokens, output_tokens, context);
            }
        }
    };
    let restored = tokio::time::timeout(Duration::from_secs(10), restored)
        .await
        .expect("no usage after resume");
    assert_eq!(restored, (1005, 11, Some(5.0 / 2000.0)));
}

#[tokio::test]
async fn a_persistent_run_journals_user_traffic_and_lifecycle_events() {
    let dir = tempfile::tempdir().unwrap();
    let state_dir = dir.path().join("run");
    let map: Scripts = Arc::new(Mutex::new(HashMap::new()));
    map.lock().insert(
        "a1".into(),
        MockMessageModel::shared(vec![
            tool_call("list_agents", "t1", json!({})),
            say("hello"),
            say("bye"),
        ]),
    );
    let mut config = SwarmConfig::new(mock_factory(&map), dir.path());
    config.state_dir = Some(state_dir.clone());
    let (swarm, mut events) = Swarm::new(config.clone());
    let id = swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "go".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "hello");
    swarm.send(USER, &id, "again").unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "bye");
    swarm.shutdown().await;

    let journal = strands_swarm::journal::read(&state_dir).unwrap();
    let lines: Vec<(&str, &str, &str, &str)> = journal
        .iter()
        .map(|e| {
            (
                e.from_label.as_str(),
                e.to_label.as_str(),
                e.kind.as_str(),
                e.text.as_str(),
            )
        })
        .collect();
    assert_eq!(
        lines,
        [
            ("user", "go (a1)", "task", "go"),
            ("go (a1)", "user", "reply", "hello"),
            ("user", "go (a1)", "message", "again"),
            ("go (a1)", "user", "reply", "bye"),
        ]
    );

    let event_names = || -> Vec<String> {
        std::fs::read_to_string(state_dir.join(strands_swarm::journal::EVENTS_FILE))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap()["event"].to_string())
            .collect()
    };
    let names = event_names();
    for wanted in [
        "\"spawned\"",
        "\"status\"",
        "\"tool\"",
        "\"tool_done\"",
        "\"shutdown\"",
    ] {
        assert!(
            names.iter().any(|n| n == wanted),
            "{wanted} missing from {names:?}"
        );
    }

    let saved = strands_swarm::state::load(&state_dir).await.unwrap().unwrap();
    let (resumed, _events) = Swarm::resume(config, saved).await.unwrap();
    resumed.shutdown().await;
    let names = event_names();
    assert_eq!(
        names.iter().filter(|n| *n == "\"spawned\"").count(),
        1,
        "{names:?}"
    );
    assert!(names.iter().any(|n| n == "\"resume\""), "{names:?}");
}

/// Delegates to a mock after a delay, recording how many calls were in flight at once.
struct SlowModel {
    inner: MockMessageModel,
    in_flight: Arc<std::sync::atomic::AtomicUsize>,
    peak: Arc<std::sync::atomic::AtomicUsize>,
}

impl strands_agents::models::Model for SlowModel {
    fn model_id(&self) -> Option<String> {
        Some("slow".into())
    }

    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        options: &'a strands_agents::models::StreamOptions,
    ) -> futures::stream::BoxStream<'a, strands_agents::Result<strands_agents::models::ModelStreamEvent>>
    {
        use futures::StreamExt;
        use std::sync::atomic::Ordering::SeqCst;
        let start = futures::stream::once(async move {
            let now = self.in_flight.fetch_add(1, SeqCst) + 1;
            self.peak.fetch_max(now, SeqCst);
            tokio::time::sleep(Duration::from_millis(50)).await;
        });
        let end = futures::stream::once(async move {
            self.in_flight.fetch_sub(1, SeqCst);
        });
        let skip = |()| async { None };
        Box::pin(
            start
                .filter_map(skip)
                .chain(self.inner.stream(messages, options))
                .chain(end.filter_map(skip)),
        )
    }
}

#[tokio::test]
async fn model_calls_beyond_the_swarm_cap_queue() {
    let dir = tempfile::tempdir().unwrap();
    let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let factory: AgentFactory = {
        let (in_flight, peak) = (in_flight.clone(), peak.clone());
        Arc::new(move |spec: AgentSpec| {
            let model = SlowModel {
                inner: MockMessageModel::new(vec![say("hi")]),
                in_flight: in_flight.clone(),
                peak: peak.clone(),
            };
            Box::pin(async move {
                let agent = Agent::builder()
                    .id(spec.id)
                    .model(model)
                    .system_prompt(spec.instructions)
                    .tools(spec.tools)
                    .no_sandbox()
                    .build()?;
                Ok(agent)
            })
        })
    };
    let mut config = SwarmConfig::new(factory, dir.path());
    config.max_concurrent_calls = 1;
    let (swarm, mut events) = Swarm::new(config);
    for task in ["one", "two", "three"] {
        let request = SpawnRequest {
            task: task.into(),
            ..Default::default()
        };
        swarm.spawn(USER, request).await.unwrap();
    }
    let (mut replies, mut queued) = (0, 0);
    let collect = async {
        while replies < 3 {
            match events.recv().await.expect("event stream closed") {
                SwarmEvent::Mail { to, mail } if to == USER && mail.kind == "reply" => replies += 1,
                SwarmEvent::ModelQueued { queued: true, .. } => queued += 1,
                _ => {}
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(10), collect)
        .await
        .unwrap();
    assert_eq!(peak.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(queued >= 1, "no agent was queued");
}

#[tokio::test]
async fn a_spent_token_budget_stops_model_calls_until_it_is_raised() {
    let dir = tempfile::tempdir().unwrap();
    let usage = strands_agents::types::Usage {
        input_tokens: 15,
        output_tokens: 5,
        total_tokens: 20,
        ..Default::default()
    };
    let (swarm, mut events, scripts) = swarm_with(
        vec![(
            "a1",
            vec![
                MockResponse::with_usage(Message::assistant("hello"), usage),
                say("continued"),
            ],
        )],
        dir.path(),
    );
    swarm.set_budget(Some(10));
    let id = swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "go".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "hello");
    assert_eq!(swarm.tokens_used(), 20);

    swarm.send(USER, &id, "more").unwrap();
    let refused = async {
        loop {
            if let Some(SwarmEvent::Mail { to, mail }) = events.recv().await {
                if to == USER && mail.kind == "error" {
                    return mail.text;
                }
            }
        }
    };
    let refused = tokio::time::timeout(Duration::from_secs(10), refused)
        .await
        .unwrap();
    assert!(refused.contains("Token budget exhausted"), "{refused}");
    let root = scripts.lock()["a1"].clone();
    assert_eq!(root.call_count(), 1, "the refused call reached the model");

    swarm.set_budget(Some(1_000));
    assert_eq!(reply_to(&mut events, USER).await, "continued");
    assert!(sent_text(&root, 1).contains("The token budget was raised to 1.0k"));
    assert_eq!(swarm.snapshot().budget_tokens, Some(1_000));
}

async fn eventually(what: &str, check: impl Fn() -> bool) {
    for _ in 0..200 {
        if check() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("timed out waiting until {what}");
}

fn branch_exists(repo: &Path, branch: &str) -> bool {
    git_in(repo)
        .args(["rev-parse", "--verify", "--quiet", branch])
        .stdout(std::process::Stdio::null())
        .status()
        .unwrap()
        .success()
}

#[tokio::test]
async fn cancelling_a_worktree_agent_reclaims_its_target_dir_and_clean_worktree() {
    let dir = tempfile::tempdir().unwrap();
    let repo = init_repo(dir.path(), &[(".gitignore", "target/\n")]);
    let scripts = scripts_of(vec![("a1", vec![say("ok")]), ("a2", vec![say("ok")])]);
    let mut config = SwarmConfig::new(mock_factory(&scripts), &repo);
    config.cargo_target_dir = Some(dir.path().join("target"));
    config.state_dir = Some(dir.path().join("run"));
    let (swarm, mut events) = Swarm::new(config);
    let worktree = SpawnRequest {
        task: "t".into(),
        worktree: true,
        ..Default::default()
    };
    let owner = swarm.spawn(USER, worktree).await.unwrap();
    let child = SpawnRequest {
        task: "t".into(),
        ..Default::default()
    };
    let child = swarm.spawn(&owner, child).await.unwrap();
    reply_to(&mut events, USER).await;
    reply_to(&mut events, &owner).await;
    let checkout = swarm.agent(&owner).unwrap().cwd;
    let target = dir
        .path()
        .join(format!("target-worktrees/{}/{owner}", swarm.run_id()));
    std::fs::create_dir_all(target.join("debug")).unwrap();
    std::fs::write(target.join("debug/lib.rlib"), "x").unwrap();

    // The child shares the owner's directories and owns none: cancelling it removes nothing.
    swarm.cancel(&child).unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(target.exists() && checkout.exists());

    swarm.cancel(&owner).unwrap();
    eventually("the target dir and worktree are gone", || {
        !target.exists() && !checkout.exists()
    })
    .await;
    let branch = format!("swarm/{}/{owner}", swarm.run_id());
    eventually("the branch, with nothing main lacks, is deleted", || {
        !branch_exists(&repo, &branch)
    })
    .await;
    // `/gc` runs in the background and reports through an event, even when nothing is left.
    swarm.gc_in_background();
    let report = async {
        loop {
            if let SwarmEvent::Reclaimed {
                collected,
                requested: true,
            } = events.recv().await.expect("event stream closed")
            {
                return collected;
            }
        }
    };
    let collected = tokio::time::timeout(Duration::from_secs(10), report)
        .await
        .unwrap();
    assert!(collected.iter().all(|agent| agent.is_empty()), "{collected:?}");
    swarm.shutdown().await;
    let events = std::fs::read_to_string(dir.path().join("run/events.jsonl")).unwrap();
    let gc: Vec<&str> = events
        .lines()
        .filter(|l| l.contains("\"event\":\"gc\""))
        .collect();
    assert_eq!(gc.len(), 1, "{events}");
    assert!(gc[0].contains(&target.display().to_string()), "{}", gc[0]);
    let root = checkout.display().to_string();
    assert!(gc[0].contains(root.trim_end_matches('/')), "{}", gc[0]);
}

#[tokio::test]
async fn a_dirty_worktree_is_kept_until_gc_finds_it_clean_and_resume_collects_cancelled_agents() {
    let dir = tempfile::tempdir().unwrap();
    let repo = init_repo(dir.path(), &[(".gitignore", "target/\n")]);
    let scripts = scripts_of(vec![("a1", vec![say("ok")])]);
    let mut config = SwarmConfig::new(mock_factory(&scripts), &repo);
    config.cargo_target_dir = Some(dir.path().join("target"));
    config.state_dir = Some(dir.path().join("run"));
    let (swarm, mut events) = Swarm::new(config.clone());
    let worktree = SpawnRequest {
        task: "t".into(),
        worktree: true,
        ..Default::default()
    };
    let owner = swarm.spawn(USER, worktree).await.unwrap();
    reply_to(&mut events, USER).await;
    let checkout = swarm.agent(&owner).unwrap().cwd;
    std::fs::write(checkout.join("wip.txt"), "unsaved work").unwrap();

    swarm.cancel(&owner).unwrap();
    eventually("the background pass has run", || {
        std::fs::read_to_string(dir.path().join("run/events.jsonl"))
            .is_ok_and(|events| events.contains("uncommitted changes"))
    })
    .await;
    assert!(checkout.join("wip.txt").exists(), "dirty worktrees are kept");
    let collected = swarm.gc().await;
    assert_eq!(collected.len(), 1);
    assert_eq!(collected[0].kept[0].1, "uncommitted changes");
    swarm.shutdown().await;

    // Committed later: resuming the run reclaims the worktree of the already-cancelled agent.
    git(&checkout, &["add", "."]);
    git(&checkout, &["commit", "-qm", "wip"]);
    let saved = strands_swarm::state::load(&dir.path().join("run"))
        .await
        .unwrap()
        .unwrap();
    let (resumed, _events) = Swarm::resume(config, saved).await.unwrap();
    eventually("the worktree is removed on resume", || !checkout.exists()).await;
    let branch = format!("swarm/{}/{owner}", resumed.run_id());
    assert!(branch_exists(&repo, &branch), "its commit is not on main");
    let left = resumed.gc().await;
    assert!(
        left.iter().all(|c| c.removed.is_empty() && c.kept.is_empty()),
        "{left:?}"
    );
    let why = left[0].kept_branch.as_ref().map(|(_, why)| why.as_str());
    assert_eq!(why, Some("1 commit not on main"), "{left:?}");
    resumed.shutdown().await;
}

#[tokio::test]
async fn cancelling_says_what_it_kept_and_a_run_without_state_cleans_up_on_exit() {
    let dir = tempfile::tempdir().unwrap();
    let repo = init_repo(dir.path(), &[(".gitignore", "target/\n")]);
    let cancel = vec![
        tool_call("cancel_agent", "c1", json!({"id": "a1"})),
        say("cancelled"),
    ];
    let scripts = scripts_of(vec![
        ("a1", vec![say("ok")]),
        ("a2", vec![say("ok")]),
        ("a3", cancel),
    ]);
    // No state dir: nothing can resume this run once it exits.
    let (swarm, mut events) = Swarm::new(SwarmConfig::new(mock_factory(&scripts), &repo));
    let worktree = || SpawnRequest {
        task: "t".into(),
        worktree: true,
        ..Default::default()
    };
    let committed = swarm.spawn(USER, worktree()).await.unwrap();
    let untouched = swarm.spawn(USER, worktree()).await.unwrap();
    reply_to(&mut events, USER).await;
    reply_to(&mut events, USER).await;
    let committed_dir = swarm.agent(&committed).unwrap().cwd;
    let untouched_dir = swarm.agent(&untouched).unwrap().cwd;
    std::fs::write(committed_dir.join("work.txt"), "done").unwrap();
    git(&committed_dir, &["add", "."]);
    git(&committed_dir, &["commit", "-qm", "work"]);

    spawn_root(&swarm, "cancel a1").await;
    while reply_to(&mut events, USER).await != "cancelled" {}
    let calls = scripts.lock()["a3"].calls();
    let result = format!("{:?}", calls.last().unwrap().0.last().unwrap());
    let run = swarm.run_id().to_owned();
    let kept = format!("a1: kept branch swarm/{run}/a1 (1 commit not on main)");
    assert!(result.contains(&kept), "{result}");
    let root = committed_dir.display().to_string();
    let removed = format!("a1: removed {}", root.trim_end_matches('/'));
    assert!(result.contains(&removed), "{result}");
    assert!(!committed_dir.exists() && branch_exists(&repo, &format!("swarm/{run}/a1")));

    // On exit the live agent's clean worktree and its branch, which main contains, go too.
    assert!(untouched_dir.exists());
    swarm.shutdown().await;
    assert!(!untouched_dir.exists());
    assert!(!branch_exists(&repo, &format!("swarm/{run}/a2")));
    assert!(
        branch_exists(&repo, &format!("swarm/{run}/a1")),
        "unmerged work stays"
    );
}

fn system_prompt(model: &MockMessageModel, call: usize) -> String {
    let (_, options) = &model.calls()[call];
    options
        .system_prompt
        .as_ref()
        .map(|prompt| prompt.text())
        .unwrap_or_default()
}

#[tokio::test]
async fn memory_reaches_every_agent_and_updates_on_the_next_turn() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.md");
    std::fs::write(&path, "- 2026-01-01: the user likes short replies\n").unwrap();
    let scripts = scripts_of(vec![(
        "a1",
        vec![
            tool_call("remember", "t1", json!({"text": "deploys happen on Fridays"})),
            say("noted"),
            say("again"),
        ],
    )]);
    let mut config = SwarmConfig::new(mock_factory(&scripts), dir.path());
    config.memory = Some(Arc::new(strands_swarm::Memory::new(&path)));
    let (swarm, mut events) = Swarm::new(config);
    let id = spawn_root(&swarm, "hello").await;
    assert_eq!(reply_to(&mut events, USER).await, "noted");

    let a1 = scripts.lock()["a1"].clone();
    let first = system_prompt(&a1, 0);
    assert!(
        first.contains("# Memory (things the user asked to remember)"),
        "{first}"
    );
    assert!(
        first.contains("likes short replies") && !first.contains("Fridays"),
        "{first}"
    );
    assert!(std::fs::read_to_string(&path)
        .unwrap()
        .contains(": deploys happen on Fridays\n"));

    swarm.remember("tabs, not spaces").await.unwrap();
    swarm.send(USER, &id, "next").unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "again");
    let next = system_prompt(&a1, 2);
    assert!(
        next.contains("Fridays") && next.contains("tabs, not spaces"),
        "{next}"
    );
    assert_eq!(next.matches("# Memory").count(), 1, "{next}");
}

#[tokio::test]
async fn without_memory_there_are_no_memory_tools() {
    let dir = tempfile::tempdir().unwrap();
    let (swarm, mut events, scripts) = swarm_with(vec![("a1", vec![say("hi")])], dir.path());
    spawn_root(&swarm, "hello").await;
    assert_eq!(reply_to(&mut events, USER).await, "hi");
    let a1 = scripts.lock()["a1"].clone();
    let (_, options) = &a1.calls()[0];
    assert!(!options.tool_specs.iter().any(|spec| spec.name == "remember"));
    assert!(swarm.remember("x").await.is_err());
}

/// [`windowed_factory`] plus a [`GoalLoop`] judge for agents spawned with a goal.
fn goal_factory(scripts: &Scripts) -> AgentFactory {
    let scripts = scripts.clone();
    Arc::new(move |spec: AgentSpec| {
        let model = Windowed(scripts.lock()[&spec.id].clone());
        Box::pin(async move {
            let mut builder = Agent::builder()
                .id(spec.id)
                .model(model)
                .tools(spec.tools)
                .no_sandbox();
            if let Some(goal) = spec.goal {
                builder = builder.plugin(strands_agents::plugins::vended::GoalLoop::goal(goal));
            }
            builder.build()
        })
    })
}

fn verdict(passed: bool, usage: strands_agents::types::Usage) -> MockResponse {
    let input = json!({ "passed": passed, "feedback": "not yet" });
    let call = ToolUseBlock::new("strands_structured_output", "verdict", input);
    MockResponse::with_usage(
        Message::new(Role::Assistant, vec![ContentBlock::ToolUse(call)]),
        usage,
    )
}

async fn spawn_with_goal(swarm: &Swarm) -> String {
    let request = SpawnRequest {
        task: "go".into(),
        goal: Some("say done".into()),
        ..Default::default()
    };
    swarm.spawn(USER, request).await.unwrap()
}

#[tokio::test]
async fn the_goal_judge_runs_on_the_side_model_and_counts_toward_the_agents_tokens() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![(
        "a1",
        vec![MockResponse::with_usage(
            Message::assistant("done"),
            usage(100, 0, 10),
        )],
    )]);
    // 30 of the judge's 50 input tokens come from the prompt cache.
    let judge = MockMessageModel::shared(vec![verdict(true, usage(20, 30, 5))]);
    let mut config = SwarmConfig::new(goal_factory(&scripts), dir.path());
    config.side_model = Some(judge.clone());
    config.state_dir = Some(dir.path().join("run"));
    let (swarm, mut events) = Swarm::new(config);
    let id = spawn_with_goal(&swarm).await;

    let seen = usage_until_reply(&mut events, &id).await;
    assert_eq!(judge.call_count(), 1);
    assert_eq!(
        scripts.lock()["a1"].call_count(),
        1,
        "the judge is not the agent's model"
    );
    // The judge's call adds to the totals without touching the agent's context fill (100 tokens
    // of the 3400-token compaction point: 85% of the 4000-token window).
    let fill = Some(100.0 / 3400.0);
    assert_eq!(seen, [(100, 10, fill), (150, 15, fill)]);
    assert_eq!(swarm.tokens_used(), 165);
    assert_eq!(swarm.snapshot().agents[0].cache_read_tokens, 30);
    // `/cost` and the event log tell the judge's share apart.
    let side = swarm.cost().side_calls();
    assert_eq!(side.keys().collect::<Vec<_>>(), ["judge"]);
    let judged = &side["judge"].0;
    assert_eq!((judged.calls, judged.input, judged.output), (1, 50, 5));
    assert_eq!(judged.cache_read, 30);
    swarm.shutdown().await;
    let log = std::fs::read_to_string(dir.path().join("run/events.jsonl")).unwrap();
    let calls: Vec<serde_json::Value> = log
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .filter(|event: &serde_json::Value| event["event"] == "side_call")
        .collect();
    assert_eq!(calls.len(), 1, "{log}");
    assert_eq!(calls[0]["id"], id.as_str());
    assert_eq!(calls[0]["purpose"], "judge");
    assert_eq!(
        (calls[0]["input"].as_u64(), calls[0]["output"].as_u64()),
        (Some(50), Some(5))
    );
}

#[tokio::test]
async fn a_spent_budget_also_stops_the_goal_judge() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![(
        "a1",
        vec![MockResponse::with_usage(
            Message::assistant("done"),
            usage(100, 0, 10),
        )],
    )]);
    let judge = MockMessageModel::shared(vec![verdict(true, usage(50, 0, 5))]);
    let mut config = SwarmConfig::new(goal_factory(&scripts), dir.path());
    config.side_model = Some(judge.clone());
    config.budget_tokens = Some(50);
    let (swarm, mut events) = Swarm::new(config);
    spawn_with_goal(&swarm).await;

    let failed = async {
        loop {
            if let Some(SwarmEvent::Mail { to, mail }) = events.recv().await {
                if to == USER && mail.kind == "error" {
                    return mail.text;
                }
            }
        }
    };
    let failed = tokio::time::timeout(Duration::from_secs(10), failed)
        .await
        .expect("no error after the budget ran out");
    assert!(failed.contains("Token budget exhausted"), "{failed}");
    assert_eq!(judge.call_count(), 0);
    assert_eq!(swarm.tokens_used(), 110);
    swarm.shutdown().await;
}

/// Like [`mock_factory`], plus a `slow` tool that runs until it is cancelled.
fn slow_tool_factory(scripts: &Scripts) -> AgentFactory {
    with_slow_tool(mock_factory(scripts))
}

/// `inner`, with agents also given a `slow` tool that runs until cancelled.
fn with_slow_tool(inner: AgentFactory) -> AgentFactory {
    Arc::new(move |mut spec: AgentSpec| {
        let slow = strands_agents::tools::FunctionTool::builder("slow")
            .description("Takes forever.")
            .handler(
                |_input: serde_json::Value, ctx: strands_agents::tools::ToolContext| async move {
                    ctx.cancel.cancelled().await;
                    // What a real tool (shell) returns when its call is cancelled.
                    strands_agents::Result::<String>::Err(strands_agents::Error::Cancelled)
                },
            );
        spec.tools.push(Arc::new(slow));
        inner(spec)
    })
}

async fn tool_started(events: &mut UnboundedReceiver<SwarmEvent>, tool: &str) {
    let wait = async {
        loop {
            if let Some(SwarmEvent::ToolStarted { name, .. }) = events.recv().await {
                if name == tool {
                    return;
                }
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(10), wait).await.unwrap();
}

async fn mail_to(events: &mut UnboundedReceiver<SwarmEvent>, to: &str, kind: &str) -> String {
    let wait = async {
        loop {
            if let Some(SwarmEvent::Mail { to: receiver, mail }) = events.recv().await {
                if receiver == to && mail.kind == kind {
                    return mail.text;
                }
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(10), wait).await.unwrap()
}

#[tokio::test]
async fn interrupting_mid_tool_idles_the_agent_and_the_next_message_continues_the_conversation() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![
        (
            "a1",
            vec![
                tool_call("spawn_agent", "s1", json!({ "task": "help", "name": "helper" })),
                tool_call("slow", "t1", json!({})),
                say("picked up again"),
            ],
        ),
        ("a2", vec![tool_call("slow", "t2", json!({})), say("helper done")]),
    ]);
    let (swarm, mut events) = Swarm::new(SwarmConfig::new(slow_tool_factory(&scripts), dir.path()));
    let root = spawn_root(&swarm, "work").await;
    tool_started(&mut events, "slow").await;
    tool_started(&mut events, "slow").await;

    swarm.interrupt(&root).unwrap();
    let told = mail_to(&mut events, USER, "interrupted").await;
    assert!(told.contains("Interrupted by the user"), "{told}");
    assert_eq!(swarm.agent(&root).unwrap().status, AgentStatus::Idle);
    assert_eq!(
        swarm.agent("a2").unwrap().status,
        AgentStatus::Running,
        "children keep going"
    );
    assert!(swarm.interrupt(&root).is_err(), "nothing left to interrupt");

    swarm.send(USER, &root, "carry on").unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "picked up again");
    let model = scripts.lock()[&root].clone();
    let (messages, _) = model.calls().pop().unwrap();
    let uses: Vec<String> = messages
        .iter()
        .flat_map(|m| m.tool_uses().map(|u| u.tool_use_id.clone()).collect::<Vec<_>>())
        .collect();
    let results: Vec<String> = messages
        .iter()
        .flat_map(|m| {
            m.tool_results()
                .map(|r| r.tool_use_id.clone())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(uses, results, "every tool call is answered");
    assert!(
        messages.iter().any(|m| m.text().contains("work")),
        "same conversation"
    );
    assert!(messages.last().unwrap().text().contains("carry on"));
    swarm.cancel("a2").unwrap();
}

#[tokio::test]
async fn a_restored_agent_that_had_finished_rests_until_its_next_message() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![("a1", vec![say("done")])]);
    let (swarm, mut events) = Swarm::new(SwarmConfig::new(mock_factory(&scripts), dir.path()));
    spawn_root(&swarm, "work").await;
    assert_eq!(reply_to(&mut events, USER).await, "done");
    let mut saved = swarm.snapshot();
    swarm.shutdown().await;
    // Even with a goal: its turn ended, so the goal was met or given up on.
    saved.agents[0].goal = Some("shipped".into());

    let resumed_scripts = scripts_of(vec![("a1", vec![say("again")])]);
    let config = SwarmConfig::new(mock_factory(&resumed_scripts), dir.path());
    let (resumed, mut events) = Swarm::resume(config, saved).await.unwrap();
    assert_eq!(
        resumed.agent("a1").unwrap().status,
        AgentStatus::Idle,
        "not starting"
    );
    assert!(!resumed.is_busy());
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(resumed_scripts.lock()["a1"].call_count(), 0, "no turn of its own");
    resumed.send(USER, "a1", "more").unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "again");
    resumed.shutdown().await;
}

#[tokio::test]
async fn an_interrupted_goal_agent_stays_idle_until_its_next_message_even_across_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![(
        "a1",
        vec![tool_call("slow", "t1", json!({})), say("back to the goal")],
    )]);
    let (swarm, mut events) = Swarm::new(SwarmConfig::new(slow_tool_factory(&scripts), dir.path()));
    let request = SpawnRequest {
        task: "ship it".into(),
        goal: Some("shipped".into()),
        ..Default::default()
    };
    let id = swarm.spawn(USER, request).await.unwrap();
    tool_started(&mut events, "slow").await;
    swarm.interrupt(&id).unwrap();
    mail_to(&mut events, USER, "interrupted").await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let model = scripts.lock()[&id].clone();
    assert_eq!(model.call_count(), 1, "no retry of the goal");
    assert_eq!(swarm.agent(&id).unwrap().status, AgentStatus::Idle);
    let saved = swarm.snapshot();
    assert!(saved.agents[0].paused);
    swarm.shutdown().await;

    let resumed_scripts = scripts_of(vec![("a1", vec![say("resumed")])]);
    let config = SwarmConfig::new(slow_tool_factory(&resumed_scripts), dir.path());
    let (resumed, mut events) = Swarm::resume(config, saved).await.unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    let model = resumed_scripts.lock()[&id].clone();
    assert_eq!(
        model.call_count(),
        0,
        "a paused goal agent does not restart on its own"
    );
    resumed.send(USER, &id, "go on").unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "resumed");
    assert!(!resumed.snapshot().agents[0].paused);
    resumed.shutdown().await;
}

/// A side model whose calls never finish, counting how many were started.
struct Hanging(Arc<std::sync::atomic::AtomicUsize>);

impl Model for Hanging {
    fn model_id(&self) -> Option<String> {
        Some("hanging".into())
    }

    fn stream<'a>(
        &'a self,
        _messages: &'a [Message],
        _options: &'a strands_agents::models::StreamOptions,
    ) -> strands_agents::models::ModelEventStream<'a> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::pin(futures::stream::pending())
    }
}

#[tokio::test]
async fn an_interrupt_also_stops_the_goal_judge() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![("a1", vec![say("done")])]);
    let started = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut config = SwarmConfig::new(goal_factory(&scripts), dir.path());
    config.side_model = Some(Arc::new(Hanging(started.clone())));
    let (swarm, mut events) = Swarm::new(config);
    let id = spawn_with_goal(&swarm).await;
    eventually("the judge is running", || {
        started.load(std::sync::atomic::Ordering::SeqCst) == 1
    })
    .await;
    swarm.interrupt(&id).unwrap();
    mail_to(&mut events, USER, "interrupted").await;
    assert_eq!(swarm.agent(&id).unwrap().status, AgentStatus::Idle);
    swarm.shutdown().await;
}

#[tokio::test]
async fn a_message_with_an_image_reaches_the_model_as_an_image_block() {
    let dir = tempfile::tempdir().unwrap();
    let shot = dir.path().join("shot.png");
    std::fs::write(&shot, b"\x89PNG fake").unwrap();
    let (swarm, mut events, scripts) = swarm_with(vec![("a1", vec![say("first"), say("a cat")])], dir.path());
    let request = SpawnRequest {
        task: "hello".into(),
        ..Default::default()
    };
    let id = swarm.spawn(USER, request).await.unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "first");

    let unsupported = swarm.send_with_images(USER, &id, "this?", vec![dir.path().join("notes.txt")]);
    assert!(unsupported.unwrap_err().to_string().contains("not a png"));
    swarm
        .send_with_images(USER, &id, "what is this?", vec![shot.clone()])
        .unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "a cat");

    let root = scripts.lock()["a1"].clone();
    let (messages, _) = &root.calls()[1];
    let last = messages.last().unwrap();
    assert!(last.text().contains("what is this?") && last.text().contains("[attached image: "));
    let image = last.content.iter().find_map(|block| match block {
        ContentBlock::Image(image) => Some(image),
        _ => None,
    });
    let image = image.expect("no image block sent to the model");
    assert!(matches!(&image.source, strands_agents::types::ImageSource::Bytes(b) if b == b"\x89PNG fake"));
}

#[tokio::test]
async fn an_image_in_a_tool_result_is_published_for_the_ui() {
    let dir = tempfile::tempdir().unwrap();
    let model = MockMessageModel::shared(vec![
        tool_call("screenshot", "t1", json!({"path": "/w/shot.png"})),
        say("done"),
    ]);
    let factory: AgentFactory = Arc::new(move |spec: AgentSpec| {
        let model = model.clone();
        Box::pin(async move {
            let screenshot = strands_agents::tools::FunctionTool::builder("screenshot")
                .description("Take a screenshot.")
                .handler(|_input: serde_json::Value, _ctx| async {
                    let image =
                        strands_agents::types::ToolResultContent::Image(strands_agents::types::ImageBlock {
                            format: strands_agents::types::ImageFormat::Png,
                            source: strands_agents::types::ImageSource::Bytes(b"\x89PNG pixels".to_vec()),
                        });
                    strands_agents::Result::Ok(serde_json::to_value(image)?)
                });
            let mut tools = spec.tools;
            tools.push(Arc::new(screenshot));
            Agent::builder()
                .id(spec.id)
                .model(model)
                .system_prompt(spec.instructions)
                .tools(tools)
                .no_sandbox()
                .build()
        })
    });
    let (swarm, mut events) = Swarm::new(SwarmConfig::new(factory, dir.path()));
    let request = SpawnRequest {
        task: "take one".into(),
        ..Default::default()
    };
    swarm.spawn(USER, request).await.unwrap();
    let image = async {
        while let Some(event) = events.recv().await {
            if let SwarmEvent::Image { id, source, bytes } = event {
                return (id, source, bytes);
            }
        }
        panic!("event stream closed");
    };
    let (id, source, bytes) = tokio::time::timeout(Duration::from_secs(10), image)
        .await
        .unwrap();
    assert_eq!((id.as_str(), source.as_str()), ("a1", "/w/shot.png"));
    assert_eq!(bytes.as_slice(), b"\x89PNG pixels");
}

#[test]
fn saved_mail_keeps_attached_image_paths_and_old_mail_still_loads() {
    let old: strands_swarm::state::MailState = serde_json::from_value(
        json!({"from": "user", "from_label": "user", "kind": "message", "text": "hi"}),
    )
    .unwrap();
    assert!(old.images.is_empty());
    let with_image = strands_swarm::state::MailState {
        images: vec!["/w/shot.png".into()],
        ..old
    };
    let text = serde_json::to_string(&with_image).unwrap();
    assert!(text.contains("/w/shot.png") && !text.contains("PNG"));
    let back: strands_swarm::state::MailState = serde_json::from_str(&text).unwrap();
    assert_eq!(back, with_image);
}

/// A persistent swarm with the `slow` tool whose idle-mode reloads wait at most `timeout`.
fn reloadable_swarm(
    scripts: &Scripts,
    dir: &Path,
    timeout: Duration,
) -> (Swarm, UnboundedReceiver<SwarmEvent>) {
    let mut config = SwarmConfig::new(slow_tool_factory(scripts), dir);
    config.state_dir = Some(dir.join("run"));
    config.reload_idle_timeout = timeout;
    Swarm::new(config)
}

async fn reload_pending(events: &mut UnboundedReceiver<SwarmEvent>) -> usize {
    let wait = async {
        loop {
            if let Some(SwarmEvent::ReloadPending { tool_calls }) = events.recv().await {
                return tool_calls;
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(10), wait).await.unwrap()
}

async fn reloads_within(swarm: &Swarm, within: Duration) -> bool {
    tokio::time::timeout(within, swarm.reload_requested())
        .await
        .is_ok()
}

#[tokio::test]
async fn an_idle_reload_waits_for_tool_calls_in_flight_but_not_for_wait() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![
        (
            "a1",
            vec![
                tool_call("spawn_agent", "s1", json!({ "task": "help", "name": "helper" })),
                tool_call("wait", "w1", json!({ "from": ["a2"] })),
                say("done"),
            ],
        ),
        ("a2", vec![tool_call("slow", "t1", json!({})), say("helper done")]),
    ]);
    let (swarm, mut events) = reloadable_swarm(&scripts, dir.path(), Duration::from_secs(600));
    spawn_root(&swarm, "work").await;
    tool_started(&mut events, "slow").await;

    swarm
        .reload(Path::new("/bin/true"), ReloadWhen::Idle)
        .await
        .unwrap();
    assert_eq!(
        reload_pending(&mut events).await,
        1,
        "the root's wait does not count"
    );
    assert!(!reloads_within(&swarm, Duration::from_millis(500)).await);
    assert!(
        dir.path().join("run/bin/current").exists(),
        "installed right away"
    );

    swarm.interrupt("a2").unwrap();
    assert!(reloads_within(&swarm, Duration::from_secs(5)).await);
    swarm.shutdown().await;
}

#[tokio::test]
async fn an_idle_reload_gives_up_waiting_after_its_timeout() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![(
        "a1",
        vec![tool_call("slow", "t1", json!({})), say("done")],
    )]);
    let (swarm, mut events) = reloadable_swarm(&scripts, dir.path(), Duration::from_millis(300));
    spawn_root(&swarm, "work").await;
    tool_started(&mut events, "slow").await;

    swarm
        .reload(Path::new("/bin/true"), ReloadWhen::Idle)
        .await
        .unwrap();
    assert!(reloads_within(&swarm, Duration::from_secs(5)).await);
    assert_eq!(
        swarm.tool_calls_in_flight(),
        1,
        "restarted with the call still running"
    );
    swarm.shutdown().await;
}

#[tokio::test]
async fn a_call_cut_by_a_restart_is_listed_with_how_long_it_ran() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![(
        "a1",
        vec![tool_call("slow", "t1", json!({})), say("done")],
    )]);
    let run = dir.path().join("run");
    let factory = || with_slow_tool(session_factory(&scripts, run.join("sessions")));
    let mut config = SwarmConfig::new(factory(), dir.path());
    config.state_dir = Some(run.clone());
    let (swarm, mut events) = Swarm::new(config);
    spawn_root(&swarm, "work").await;
    tool_started(&mut events, "slow").await;
    tokio::time::sleep(Duration::from_millis(1100)).await;
    swarm.shutdown().await;

    let saved = strands_swarm::state::load(&run)
        .await
        .unwrap()
        .expect("state saved");
    assert_eq!(
        saved.agents[0].running_calls,
        std::collections::BTreeMap::from([("t1".to_owned(), 1)])
    );
    scripts
        .lock()
        .insert("a1".into(), MockMessageModel::shared(vec![say("resumed")]));
    let mut config = SwarmConfig::new(factory(), dir.path());
    config.state_dir = Some(run.clone());
    let (resumed, mut events) = Swarm::resume(config, saved).await.unwrap();
    assert_eq!(reply_to(&mut events, USER).await, "resumed");
    let sent = sent_text(&scripts.lock()["a1"].clone(), 0);
    assert!(sent.contains("- slow (cut after 1s) {}"), "{sent}");
    resumed.shutdown().await;
}

#[tokio::test]
async fn a_newer_reload_supersedes_a_pending_one() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![(
        "a1",
        vec![tool_call("slow", "t1", json!({})), say("done")],
    )]);
    let (swarm, mut events) = reloadable_swarm(&scripts, dir.path(), Duration::from_secs(1));
    spawn_root(&swarm, "work").await;
    tool_started(&mut events, "slow").await;

    swarm
        .reload(Path::new("/bin/true"), ReloadWhen::Idle)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(600)).await;
    swarm
        .reload(Path::new("/bin/true"), ReloadWhen::Idle)
        .await
        .unwrap();
    // The first request's deadline passes; only the second one's counts.
    assert!(!reloads_within(&swarm, Duration::from_millis(600)).await);
    assert!(reloads_within(&swarm, Duration::from_secs(5)).await);

    swarm.shutdown().await;
}

#[tokio::test]
async fn a_reload_now_does_not_wait() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![(
        "a1",
        vec![tool_call("slow", "t1", json!({})), say("done")],
    )]);
    let (swarm, mut events) = reloadable_swarm(&scripts, dir.path(), Duration::from_secs(600));
    spawn_root(&swarm, "work").await;
    tool_started(&mut events, "slow").await;

    swarm
        .reload(Path::new("/bin/true"), ReloadWhen::Now)
        .await
        .unwrap();
    assert!(reloads_within(&swarm, Duration::from_millis(100)).await);
    swarm.shutdown().await;
}

#[tokio::test]
async fn resuming_under_another_commit_reports_what_changed() {
    let dir = tempfile::tempdir().unwrap();
    let repo = init_repo(dir.path(), &[(".gitignore", "target/\n")]);
    let commit = |subject: &str| {
        git(&repo, &["commit", "-q", "--allow-empty", "-m", subject]);
        git(&repo, &["rev-parse", "HEAD"])
    };
    let old = commit("feat: before");
    let new = commit("fix: after");
    let map: Scripts = Arc::new(Mutex::new(HashMap::new()));
    let mut config = SwarmConfig::new(mock_factory(&map), dir.path());
    config.state_dir = Some(dir.path().join("run"));
    config.source = repo.clone();
    config.commit = Some(old.clone());
    let (swarm, _events) = Swarm::new(config.clone());
    swarm.shutdown().await;
    let saved = strands_swarm::state::load(&dir.path().join("run"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        saved.commit.as_deref(),
        Some(old.as_str()),
        "saved with its commit"
    );

    config.commit = Some(new.clone());
    let (resumed, mut events) = Swarm::resume(config.clone(), saved.clone()).await.unwrap();
    let Ok(SwarmEvent::Reloaded { from, to, changes }) = events.try_recv() else {
        panic!("no Reloaded event");
    };
    assert_eq!((from, to), (old.clone(), new.clone()));
    assert_eq!(changes, [format!("{} fix: after", &new[..7])]);
    resumed.shutdown().await;
    let journal = std::fs::read_to_string(dir.path().join("run/events.jsonl")).unwrap();
    assert!(
        journal.contains("\"event\":\"changes\"") && journal.contains("fix: after"),
        "{journal}"
    );

    // Same commit, or a run saved before commits were recorded: nothing to report.
    for commit in [Some(old.clone()), None] {
        let mut saved = saved.clone();
        saved.commit = commit;
        config.commit = Some(old.clone());
        let (resumed, mut events) = Swarm::resume(config.clone(), saved).await.unwrap();
        while let Ok(event) = events.try_recv() {
            assert!(!matches!(event, SwarmEvent::Reloaded { .. }), "{event:?}");
        }
        resumed.shutdown().await;
    }
}

/// [`windowed_factory`] with a context manager that summarizes everything but the latest message
/// before every model call.
fn summarizing_factory(scripts: &Scripts) -> AgentFactory {
    use strands_agents::context_manager::strategies::{Offload, OffloadConditions, OffloadTarget};
    use strands_agents::context_manager::{ContextManagerConfig, ContextManagerOption, StashOption};
    let scripts = scripts.clone();
    Arc::new(move |spec: AgentSpec| {
        let model = Windowed(scripts.lock()[&spec.id].clone());
        let summarize = Offload::summarize(OffloadTarget::All).when(OffloadConditions {
            utilization: Some(0.0),
            preserve_recent: Some(1.0),
            ..Default::default()
        });
        let manager = ContextManagerOption::Config(ContextManagerConfig {
            strategies: Some(vec![Arc::new(summarize)]),
            stash: StashOption::Disabled,
        });
        Box::pin(async move {
            Agent::builder()
                .id(spec.id)
                .model(model)
                .tools(spec.tools)
                .context_manager(manager)
                .no_sandbox()
                .build()
        })
    })
}

#[tokio::test]
async fn context_summarization_runs_on_the_side_model_and_counts_toward_the_agents_tokens() {
    let dir = tempfile::tempdir().unwrap();
    let call = Message::new(
        Role::Assistant,
        vec![ContentBlock::ToolUse(ToolUseBlock::new(
            "list_agents",
            "t1",
            json!({}),
        ))],
    );
    let scripts = scripts_of(vec![(
        "a1",
        vec![
            MockResponse::with_usage(call, usage(100, 0, 10)),
            MockResponse::with_usage(Message::assistant("done"), usage(40, 0, 4)),
        ],
    )]);
    let summarizer = MockMessageModel::shared(vec![MockResponse::with_usage(
        Message::assistant("they listed agents"),
        usage(30, 0, 3),
    )]);
    let mut config = SwarmConfig::new(summarizing_factory(&scripts), dir.path());
    config.side_model = Some(summarizer.clone());
    config.state_dir = Some(dir.path().join("run"));
    let (swarm, mut events) = Swarm::new(config);
    let id = spawn_root(&swarm, "go").await;

    let seen = usage_until_reply(&mut events, &id).await;
    let summaries = summarizer.call_count() as u64;
    assert!(summaries >= 1, "nothing was summarized");
    let (input, output, _) = *seen.last().unwrap();
    assert_eq!((input, output), (140 + 30 * summaries, 14 + 3 * summaries));
    assert_eq!(swarm.tokens_used(), input + output);
    let side = swarm.cost().side_calls();
    assert_eq!(side.keys().collect::<Vec<_>>(), ["summarizer"]);
    assert_eq!(side["summarizer"].0.calls, summaries);
    assert_eq!(side["summarizer"].0.input, 30 * summaries);
    swarm.shutdown().await;
    let log = std::fs::read_to_string(dir.path().join("run/events.jsonl")).unwrap();
    let purposes: Vec<serde_json::Value> = log
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .filter(|event| event["event"] == "side_call")
        .map(|event| event["purpose"].clone())
        .collect();
    assert_eq!(purposes, vec![json!("summarizer"); summaries as usize]);
}

fn every(minutes: u64, to: &str, message: &str) -> strands_swarm::ScheduleRequest {
    strands_swarm::ScheduleRequest {
        to: to.into(),
        message: message.into(),
        every: Some(Duration::from_secs(minutes * 60)),
        ..Default::default()
    }
}

#[tokio::test(start_paused = true)]
async fn a_recurring_schedule_wakes_its_agent_until_it_is_cancelled() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![("a1", vec![say("started"), say("tick 1"), say("tick 2")])]);
    let (swarm, mut events) = Swarm::new(SwarmConfig::new(mock_factory(&scripts), dir.path()));
    let root = spawn_root(&swarm, "monitor").await;
    assert_eq!(reply_to(&mut events, USER).await, "started");

    let schedule = swarm
        .schedule(&root, every(30, &root, "check the queue"))
        .unwrap();
    assert_eq!(schedule.id, "s1");
    tokio::time::sleep(Duration::from_secs(29 * 60)).await;
    assert_eq!(scripts.lock()["a1"].call_count(), 1, "not before its time");
    tokio::time::sleep(Duration::from_secs(2 * 60)).await;
    assert_eq!(reply_to(&mut events, USER).await, "tick 1");
    let woken = sent_text(&scripts.lock()["a1"].clone(), 1);
    assert!(
        woken.contains("kind=\\\"schedule\\\"") && woken.contains("[schedule s1] check the queue"),
        "{woken}"
    );

    tokio::time::sleep(Duration::from_secs(30 * 60)).await;
    assert_eq!(reply_to(&mut events, USER).await, "tick 2");
    swarm.unschedule("s1").unwrap();
    assert!(swarm.schedules().is_empty());
    tokio::time::sleep(Duration::from_secs(3 * 3600)).await;
    assert_eq!(scripts.lock()["a1"].call_count(), 3);
    swarm.shutdown().await;
}

#[tokio::test]
async fn an_idle_agent_waiting_on_a_schedule_holds_no_agent_slot() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![("a1", vec![say("idle now")]), ("a2", vec![say("hi")])]);
    let mut config = SwarmConfig::new(mock_factory(&scripts), dir.path());
    config.max_agents = 1;
    let (swarm, mut events) = Swarm::new(config);
    let root = spawn_root(&swarm, "monitor").await;
    reply_to(&mut events, USER).await;
    eventually("the root is idle", || {
        swarm.agent(&root).unwrap().status == AgentStatus::Idle
    })
    .await;

    let queued = spawn_root(&swarm, "more").await;
    assert_eq!(swarm.agent(&queued).unwrap().status, AgentStatus::Queued);
    swarm.schedule(&root, every(60, &root, "poll")).unwrap();
    assert_eq!(
        reply_to(&mut events, USER).await,
        "hi",
        "the queued agent started"
    );

    // Cancelling the receiver drops its schedules.
    swarm.cancel(&root).unwrap();
    assert!(swarm.schedules().is_empty());
    swarm.shutdown().await;
}

#[tokio::test]
async fn schedules_persist_and_missed_fires_fire_once_on_resume() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![("a1", vec![say("started")])]);
    let mut config = SwarmConfig::new(mock_factory(&scripts), dir.path());
    config.state_dir = Some(dir.path().join("run"));
    let (swarm, mut events) = Swarm::new(config.clone());
    let root = spawn_root(&swarm, "monitor").await;
    reply_to(&mut events, USER).await;
    swarm.schedule(&root, every(30, &root, "repeating")).unwrap();
    let once = strands_swarm::ScheduleRequest {
        to: root.clone(),
        message: "one-shot".into(),
        at: Some(swarm.now() + chrono::Duration::hours(1)),
        ..Default::default()
    };
    swarm.schedule(&root, once).unwrap();
    swarm.shutdown().await;

    let mut saved = strands_swarm::state::load(&dir.path().join("run"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(saved.schedules.len(), 2, "saved with the run");
    // As if the swarm had been down for five hours: ten repeats and the one-shot were missed.
    let now = chrono::Utc::now();
    for schedule in &mut saved.schedules {
        schedule.next = now - chrono::Duration::hours(5);
    }
    let scripts = scripts_of(vec![("a1", vec![say("caught up"), say("again")])]);
    let config = SwarmConfig {
        factory: mock_factory(&scripts),
        ..config
    };
    let (resumed, mut events) = Swarm::resume(config, saved).await.unwrap();
    reply_to(&mut events, USER).await;
    let fired: Vec<String> = scripts.lock()["a1"]
        .calls()
        .iter()
        .flat_map(|(messages, _)| messages.last().map(|m| m.text()))
        .collect();
    let fired = fired.join("\n");
    assert_eq!(fired.matches("[schedule s1] repeating").count(), 1, "{fired}");
    assert_eq!(fired.matches("[schedule s2] one-shot").count(), 1, "{fired}");

    let left = resumed.schedules();
    assert_eq!(left.len(), 1, "the one-shot is done");
    assert!(left[0].next > now && left[0].next <= now + chrono::Duration::minutes(30));
    resumed.shutdown().await;
}

#[tokio::test]
async fn agents_schedule_list_and_cancel_through_their_tools() {
    let dir = tempfile::tempdir().unwrap();
    let (swarm, mut events, scripts) = swarm_with(
        vec![(
            "a1",
            vec![
                tool_call("schedule", "t1", json!({"message": "poll slack", "every": "30m"})),
                tool_call("schedule", "t2", json!({"message": "x", "every": "5 minutes"})),
                tool_call("list_agents", "t3", json!({})),
                tool_call("schedule", "t4", json!({})),
                tool_call("schedule", "t5", json!({"cancel": "s1"})),
                say("done"),
            ],
        )],
        dir.path(),
    );
    spawn_root(&swarm, "monitor slack").await;
    assert_eq!(reply_to(&mut events, USER).await, "done");
    let root = scripts.lock()["a1"].clone();
    let result = |call: usize| {
        let (messages, _) = &root.calls()[call];
        format!("{:?}", messages.last().unwrap())
    };
    assert!(
        result(1).contains("Scheduled s1 every 30m, next in 30m"),
        "{}",
        result(1)
    );
    assert!(
        result(1).contains("→ monitor-slack (a1): poll slack"),
        "{}",
        result(1)
    );
    assert!(result(2).contains("Not a duration"), "{}", result(2));
    assert!(
        result(3).contains("  scheduled: s1 every 30m, next in "),
        "{}",
        result(3)
    );
    assert!(result(4).contains("s1 every 30m"), "{}", result(4));
    assert!(result(5).contains("Removed s1"), "{}", result(5));
    assert!(swarm.schedules().is_empty());
}

#[tokio::test]
async fn status_reports_todos_and_branch_commits_without_a_model_call() {
    let dir = tempfile::tempdir().unwrap();
    let repo = init_repo(dir.path(), &[("init.txt", "init")]);
    let commit = |dir: &Path, message: &str| {
        std::fs::write(dir.join(format!("{message}.txt")), message).unwrap();
        git(dir, &["add", "."]);
        git(dir, &["commit", "-qm", message]);
    };

    let map: Scripts = Arc::new(Mutex::new(HashMap::new()));
    // The root answers its task, then the builder's report, then lists the agents.
    let list = vec![
        say("ok"),
        say("ok"),
        tool_call("list_agents", "l1", json!({})),
        say("listed"),
    ];
    map.lock().insert("a1".to_owned(), MockMessageModel::shared(list));
    map.lock()
        .insert("a2".to_owned(), MockMessageModel::shared(vec![say("ok")]));
    let inner = mock_factory(&map);
    let factory: AgentFactory = Arc::new(move |spec: AgentSpec| {
        let build = inner(spec);
        Box::pin(async move {
            let agent = build.await?;
            let todos = serde_json::json!([
                {"content": "Write it", "activeForm": "Writing it", "status": "in_progress"},
                {"content": "Test it", "activeForm": "Testing it", "status": "pending"},
            ]);
            agent.state().set("todos", todos);
            Ok(agent)
        })
    });
    let (swarm, mut events) = Swarm::new(SwarmConfig::new(factory, &repo));
    let root = spawn_root(&swarm, "lead").await;
    let child = swarm
        .spawn(
            &root,
            SpawnRequest {
                task: "build".into(),
                name: Some("builder".into()),
                worktree: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    wait_for_status(&swarm, &child, AgentStatus::Idle).await;
    commit(&swarm.agent(&child).unwrap().cwd, "feature");

    let report = loop {
        let report = swarm.status();
        if report
            .agents
            .iter()
            .all(|a| a.branch.is_none() || a.git.is_some())
        {
            break report;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert_eq!(report.base.as_deref(), Some("main"));
    let builder = report.agents.iter().find(|a| a.id == child).unwrap();
    assert_eq!(builder.label, format!("builder ({child})"));
    assert_eq!(builder.todos.len(), 2);
    let state = builder.git.as_ref().unwrap();
    assert_eq!(state.ahead.len(), 1, "{state:?}");
    assert!(state.ahead[0].ends_with(" feature"), "{state:?}");
    let text = report.lines(&|_| None).join("\n");
    assert!(text.contains("  [>] Writing it\n  [ ] Test it"), "{text}");
    assert!(text.contains("1 not on main, fast-forward:"), "{text}");

    // Main moves on and touches the same file: the integrator sees the conflict up front.
    std::fs::write(repo.join("feature.txt"), "theirs").unwrap();
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-qm", "theirs"]);
    let (base, states) = swarm.branch_states().await;
    assert_eq!(base.as_deref(), Some("main"));
    let state = &states[&child];
    assert_eq!((state.ahead.len(), state.behind), (1, 1));
    assert_eq!(state.merge, Merge::Conflicts(vec!["feature.txt".into()]));
    assert!(swarm
        .status()
        .lines(&|_| None)
        .join("\n")
        .contains("conflicts in feature.txt:"));
    // `list_agents` says so too.
    while map.lock()["a1"].call_count() < 2 {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    wait_for_status(&swarm, &root, AgentStatus::Idle).await;
    swarm.send(USER, &root, "who is there?").unwrap();
    while reply_to(&mut events, USER).await != "listed" {}
    let calls = map.lock()["a1"].calls();
    let listing = format!("{:?}", calls.last().unwrap().0.last().unwrap());
    let branch = format!(
        "{}/{child} (1 not on main, 1 behind, conflicts in feature.txt)",
        swarm.run_id()
    );
    assert!(listing.contains(&branch), "{listing}");

    swarm.cancel(&child).unwrap();
    let report = swarm.status();
    assert!(
        report.agents.iter().all(|a| a.id != child),
        "cancelled agents are left out"
    );
}

fn persistent_swarm(scripts: &Scripts, dir: &Path) -> SwarmConfig {
    let mut config = SwarmConfig::new(mock_factory(scripts), dir);
    config.state_dir = Some(dir.join("run"));
    config
}

#[tokio::test]
async fn a_background_job_mails_its_exit_code_and_output_tail_to_the_idle_agent_that_started_it() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![(
        "a1",
        vec![
            tool_call(
                "run_background",
                "r1",
                json!({ "command": "echo built; echo $PWD; exit 2", "name": "gates" }),
            ),
            say("started"),
            say("saw it"),
        ],
    )]);
    let (swarm, mut events) = Swarm::new(persistent_swarm(&scripts, dir.path()));
    spawn_root(&swarm, "build it").await;
    assert_eq!(reply_to(&mut events, USER).await, "started");
    let started = sent_text(&scripts.lock()["a1"].clone(), 1);
    assert!(started.contains("Started job j1 (gates)"), "{started}");

    let report = mail_to(&mut events, "a1", "job").await;
    assert!(report.contains("Job j1 (gates) exited 2 after"), "{report}");
    assert!(report.contains("built"), "{report}");
    assert!(
        report.contains(&dir.path().display().to_string()),
        "runs in the agent's cwd: {report}"
    );
    assert_eq!(reply_to(&mut events, USER).await, "saw it");
    let woken = sent_text(&scripts.lock()["a1"].clone(), 2);
    assert!(woken.contains("kind=\\\"job\\\""), "{woken}");
    let jobs = swarm.jobs().await;
    assert!(jobs[0].reported, "reported once");
}

#[tokio::test]
async fn running_jobs_show_in_list_agents_and_a_killed_job_does_not_report() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![(
        "a1",
        vec![
            tool_call(
                "run_background",
                "r1",
                json!({ "command": "sleep 30", "name": "long" }),
            ),
            tool_call("list_agents", "l1", json!({})),
            tool_call("jobs", "j1", json!({})),
            tool_call("jobs", "j2", json!({ "kill": "j1" })),
            say("done"),
        ],
    )]);
    let (swarm, mut events) = Swarm::new(persistent_swarm(&scripts, dir.path()));
    spawn_root(&swarm, "go").await;
    assert_eq!(reply_to(&mut events, USER).await, "done");
    let model = scripts.lock()["a1"].clone();
    let listed = sent_text(&model, 2);
    assert!(listed.contains(", jobs j1 (long)"), "{listed}");
    let jobs = sent_text(&model, 3);
    assert!(jobs.contains("j1 (long): running"), "{jobs}");
    let killed = sent_text(&model, 4);
    assert!(killed.contains("Job j1 (long): killed after"), "{killed}");

    tokio::time::sleep(Duration::from_millis(1500)).await;
    while let Ok(event) = events.try_recv() {
        if let SwarmEvent::Mail { mail, .. } = event {
            assert_ne!(mail.kind, "job", "a killed job is not reported: {}", mail.text);
        }
    }
}

#[tokio::test]
async fn a_job_started_before_a_restart_reports_to_the_resumed_agent() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![(
        "a1",
        vec![
            tool_call(
                "run_background",
                "r1",
                json!({ "command": "sleep 1; echo finished" }),
            ),
            say("started"),
        ],
    )]);
    let config = persistent_swarm(&scripts, dir.path());
    let (swarm, mut events) = Swarm::new(config.clone());
    spawn_root(&swarm, "go").await;
    assert_eq!(reply_to(&mut events, USER).await, "started");
    swarm.shutdown().await;
    let jobs = swarm.jobs().await;
    assert_eq!(
        jobs[0].state,
        strands_swarm::jobs::JobState::Running,
        "outlives the swarm"
    );

    let saved = strands_swarm::state::load(&dir.path().join("run"))
        .await
        .unwrap()
        .expect("state saved");
    scripts
        .lock()
        .insert("a1".into(), MockMessageModel::shared(vec![say("noted")]));
    let (_resumed, mut events) = Swarm::resume(config, saved).await.unwrap();
    let report = mail_to(&mut events, "a1", "job").await;
    assert!(report.contains("Job j1 exited 0"), "{report}");
    assert!(report.contains("finished"), "{report}");
    assert_eq!(reply_to(&mut events, USER).await, "noted");
}

#[tokio::test]
async fn cancelling_an_agent_kills_its_background_jobs() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = scripts_of(vec![(
        "a1",
        vec![
            tool_call("run_background", "r1", json!({ "command": "sleep 30" })),
            say("started"),
        ],
    )]);
    let (swarm, mut events) = Swarm::new(persistent_swarm(&scripts, dir.path()));
    spawn_root(&swarm, "go").await;
    assert_eq!(reply_to(&mut events, USER).await, "started");
    swarm.cancel("a1").unwrap();
    for _ in 0..100 {
        if swarm.jobs().await[0].state == strands_swarm::jobs::JobState::Killed {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("the job outlived its agent");
}

#[tokio::test]
async fn land_rebases_checks_and_fast_forwards_the_roots_branch_as_a_job() {
    let dir = tempfile::tempdir().unwrap();
    let repo = init_repo(dir.path(), &[("a.txt", "a")]);

    let scripts = scripts_of(vec![
        ("a1", vec![say("lead")]),
        ("a2", vec![say("built"), say("landed, thanks")]),
    ]);
    let mut config = SwarmConfig::new(mock_factory(&scripts), &repo);
    config.state_dir = Some(dir.path().join("run"));
    config.exe = Some(env!("CARGO_BIN_EXE_strands-swarm").into());
    config.gates = vec!["test -f feature.txt".into()];
    let (off, _) = Swarm::new(SwarmConfig::new(mock_factory(&scripts), &repo));
    assert!(!off.allows_land(), "opt-in");
    let root = spawn_root(&off, "lead").await;
    let error = off.land(&root, LandRequest::default()).await.unwrap_err();
    assert!(error.to_string().contains("--allow-land"), "{error}");
    off.shutdown().await;

    config.allow_land = true;
    let (swarm, mut events) = Swarm::new(config);
    let root = spawn_root(&swarm, "lead").await;
    let tools: Vec<String> = strands_swarm::tools::swarm_tools(swarm.clone(), &root)
        .iter()
        .map(|tool| tool.name().to_owned())
        .collect();
    assert!(tools.contains(&"land".to_owned()), "{tools:?}");
    let child = swarm
        .spawn(
            &root,
            SpawnRequest {
                task: "build".into(),
                name: Some("builder".into()),
                worktree: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    wait_for_status(&swarm, &child, AgentStatus::Idle).await;
    let cwd = swarm.agent(&child).unwrap().cwd;
    std::fs::write(cwd.join("feature.txt"), "f").unwrap();
    git(&cwd, &["add", "."]);
    git(&cwd, &["commit", "-qm", "feature"]);
    // `main` moves on meanwhile; landing rebases onto it.
    std::fs::write(repo.join("other.txt"), "o").unwrap();
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-qm", "other"]);

    let job = swarm.land(&child, LandRequest::default()).await.unwrap();
    assert!(job.label().contains("→ main"), "{}", job.label());
    let report = mail_to(&mut events, &child, "job").await;
    assert!(report.contains("Landed 1 commit(s)"), "{report}");
    assert!(report.contains("ok `test -f feature.txt`"), "{report}");
    assert!(
        repo.join("feature.txt").is_file(),
        "the root's checkout was fast-forwarded"
    );
    assert_eq!(
        git(&repo, &["log", "--format=%s", "main"]),
        "feature\nother\ninit"
    );
    swarm.shutdown().await;
}

#[tokio::test]
async fn land_refuses_other_targets_and_branches_that_are_not_the_callers() {
    let dir = tempfile::tempdir().unwrap();
    let repo = init_repo(dir.path(), &[("a.txt", "a")]);
    git(&repo, &["branch", "release-1"]);
    git(&repo, &["branch", "precious"]);

    let scripts = scripts_of(vec![
        ("a1", vec![say("lead")]),
        ("a2", vec![say("one")]),
        ("a3", vec![say("two")]),
        ("a4", vec![say("grandchild")]),
    ]);
    let make = |land_into: Vec<String>| {
        let mut config = SwarmConfig::new(mock_factory(&scripts), &repo);
        config.state_dir = Some(dir.path().join("run"));
        config.exe = Some(env!("CARGO_BIN_EXE_strands-swarm").into());
        config.allow_land = true;
        config.land_into = land_into;
        config
    };
    let (swarm, mut events) = Swarm::new(make(vec!["release-*".into()]));
    let root = spawn_root(&swarm, "lead").await;
    let worker = |task: &str| SpawnRequest {
        task: task.into(),
        worktree: true,
        ..Default::default()
    };
    let one = swarm.spawn(&root, worker("one")).await.unwrap();
    let two = swarm.spawn(&root, worker("two")).await.unwrap();
    let grandchild = swarm.spawn(&one, worker("grandchild")).await.unwrap();
    for id in [&one, &two, &grandchild] {
        wait_for_status(&swarm, id, AgentStatus::Idle).await;
    }
    let branch_of = |id: &str| swarm.agent(id).unwrap().branch.unwrap();
    let request = |branch: Option<String>, into: Option<&str>| LandRequest {
        branch,
        into: into.map(str::to_owned),
        check: Some(vec![]),
    };
    let refused = |result: strands_agents::Result<_>| result.map(|_| ()).unwrap_err().to_string();

    // Targets: the root's checked-out branch, or one matching --land-into; nothing else.
    let error = refused(swarm.land(&one, request(None, Some("precious"))).await);
    assert!(error.contains("landing on precious is not allowed"), "{error}");
    assert!(error.contains("only main"), "{error}");
    assert!(error.contains("release-*"), "{error}");
    // Sources: one's own branch or a descendant's, never a sibling's or an arbitrary one.
    let error = refused(swarm.land(&two, request(Some(branch_of(&one)), None)).await);
    assert!(error.contains("is not yours to land"), "{error}");
    assert!(
        error.contains(&branch_of(&two)),
        "lists the caller's own: {error}"
    );
    let error = refused(
        swarm
            .land(&grandchild, request(Some(branch_of(&one)), None))
            .await,
    );
    assert!(
        error.contains("is not yours to land"),
        "not a parent's either: {error}"
    );
    let error = refused(swarm.land(&one, request(Some("precious".into()), None)).await);
    assert!(error.contains("is not yours to land"), "{error}");

    // Allowed: a descendant's branch on an allow-listed target, and the root's own default.
    let cwd = swarm.agent(&grandchild).unwrap().cwd;
    std::fs::write(cwd.join("g.txt"), "g").unwrap();
    git(&cwd, &["add", "."]);
    git(&cwd, &["commit", "-qm", "grandchild work"]);
    swarm
        .land(&one, request(Some(branch_of(&grandchild)), Some("release-1")))
        .await
        .unwrap();
    let report = mail_to(&mut events, &one, "job").await;
    assert!(report.contains("Landed 1 commit(s)"), "{report}");
    let error = refused(swarm.land(&root, request(None, None)).await);
    assert!(
        error.contains("cannot be landed on itself"),
        "root's own is main: {error}"
    );
    swarm.shutdown().await;

    // Without --land-into only the default target is open.
    let (strict, _events) = Swarm::new(make(vec![]));
    let root = spawn_root(&strict, "lead").await;
    let error = refused(strict.land(&root, request(None, Some("release-1"))).await);
    assert!(error.contains("--land-into GLOB"), "{error}");
    strict.shutdown().await;
}

#[tokio::test]
async fn list_agents_shows_each_agents_estimated_spend_and_its_subtrees() {
    let dir = tempfile::tempdir().unwrap();
    // Small enough to stay under the compaction point.
    let spawn = Message::new(
        Role::Assistant,
        vec![ContentBlock::ToolUse(ToolUseBlock::new(
            "spawn_agent",
            "s1",
            json!({"task": "sub", "name": "worker"}),
        ))],
    );
    let scripts = scripts_of(vec![
        (
            "a1",
            vec![
                MockResponse::with_usage(spawn, usage(50_000, 0, 0)),
                tool_call("wait", "w1", json!({})),
                tool_call("list_agents", "l1", json!({})),
                say("done"),
            ],
        ),
        (
            "a2",
            vec![MockResponse::with_usage(
                Message::assistant("hi"),
                usage(100_000, 0, 0),
            )],
        ),
    ]);
    let mut config = persistent_swarm(&scripts, dir.path());
    let prices = r#"{"test-model": {"input": 100, "output": 500}}"#;
    config.prices = Arc::new(strands_swarm::cost::Prices::with_overrides(prices).unwrap());
    let (swarm, mut events) = Swarm::new(config);
    spawn_root(&swarm, "go").await;
    assert_eq!(reply_to(&mut events, USER).await, "done");
    let listed = sent_text(&scripts.lock()["a1"].clone(), 3);
    assert!(
        listed.contains("model test-model, est. $5.00 / 50.0k tokens (with 1 descendant $15.00 / 150.0k)"),
        "{listed}"
    );
    assert!(listed.contains("est. $10.00 / 100.0k tokens"), "{listed}");
    assert!(
        listed.contains("Swarm total: est. $15.00 / 150.0k tokens"),
        "{listed}"
    );
}

#[tokio::test]
async fn compactions_reported_by_the_factorys_pipeline_become_swarm_events() {
    let dir = tempfile::tempdir().unwrap();
    let factory: AgentFactory = Arc::new(|spec: AgentSpec| {
        // Stands in for the context pipeline summarizing before a model call.
        let sink = spec.on_compacted.as_ref().expect("a compaction sink");
        sink(Compaction::Failed {
            tokens: 300_000,
            error: "throttled".into(),
        });
        sink(Compaction::Done {
            before: 300_000,
            after: 90_000,
        });
        Box::pin(async move {
            Agent::builder()
                .id(spec.id)
                .model(MockMessageModel::shared(vec![say("done")]))
                .tools(spec.tools)
                .no_sandbox()
                .build()
        })
    });
    let (swarm, mut events) = Swarm::new(SwarmConfig::new(factory, dir.path()));
    swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "go".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let compacted = async {
        let mut failed = None;
        loop {
            match events.recv().await {
                Some(SwarmEvent::CompactFailed { id, tokens, error }) => failed = Some((id, tokens, error)),
                Some(SwarmEvent::Compacted { id, before, after }) => return (failed, (id, before, after)),
                _ => {}
            }
        }
    };
    let compacted = tokio::time::timeout(Duration::from_secs(10), compacted)
        .await
        .expect("timed out waiting for the compaction");
    let failed = Some(("a1".to_owned(), 300_000, "throttled".to_owned()));
    assert_eq!(compacted, (failed, ("a1".to_owned(), 300_000, 90_000)));
}

/// Asks for the `pause` tool until it sees a tool result, then says "done"; every response
/// reports a cache hit on a 1,000-token prompt. Counts calls by whether they repeat the first.
#[derive(Default)]
struct CachingModel {
    calls: std::sync::atomic::AtomicUsize,
    repeats: std::sync::atomic::AtomicUsize,
}

impl Model for CachingModel {
    fn model_id(&self) -> Option<String> {
        Some("caching".into())
    }

    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        _options: &'a strands_agents::models::StreamOptions,
    ) -> strands_agents::models::ModelEventStream<'a> {
        use std::sync::atomic::Ordering;
        let answered = messages
            .last()
            .is_some_and(|m| m.content.iter().any(|b| matches!(b, ContentBlock::ToolResult(_))));
        self.calls.fetch_add(1, Ordering::SeqCst);
        if !answered && self.calls.load(Ordering::SeqCst) > 1 {
            self.repeats.fetch_add(1, Ordering::SeqCst);
        }
        let (message, stop) = match answered {
            false => (
                Message::new(
                    Role::Assistant,
                    vec![ContentBlock::ToolUse(ToolUseBlock::new("pause", "p1", json!({})))],
                ),
                strands_agents::types::StopReason::ToolUse,
            ),
            true => (
                Message::assistant("done"),
                strands_agents::types::StopReason::EndTurn,
            ),
        };
        let events =
            strands_agents::testing::events_for_message(&message, Some(stop), Some(usage(10, 990, 5)));
        Box::pin(futures::stream::iter(events.into_iter().map(Ok)))
    }
}

#[tokio::test]
async fn a_turn_waiting_on_a_tool_keeps_its_prompt_cache_warm_until_the_next_call() {
    use std::sync::atomic::Ordering;
    let dir = tempfile::tempdir().unwrap();
    let model = Arc::new(CachingModel::default());
    let shared = model.clone();
    let built: Arc<Mutex<Option<Agent>>> = Arc::default();
    let keep = built.clone();
    let factory: AgentFactory = Arc::new(move |spec: AgentSpec| {
        let model = shared.clone();
        let keep = keep.clone();
        let pause = strands_agents::tools::FunctionTool::builder("pause")
            .description("Takes a while.")
            .handler(|_input: serde_json::Value, _ctx| async move {
                tokio::time::sleep(Duration::from_millis(400)).await;
                strands_agents::Result::Ok("back".to_owned())
            });
        let mut tools = spec.tools;
        tools.push(Arc::new(pause));
        Box::pin(async move {
            let agent = Agent::builder()
                .id(spec.id)
                .model(model)
                .tools(tools)
                .no_sandbox()
                .build()?;
            *keep.lock() = Some(agent.clone());
            Ok(agent)
        })
    });
    let mut config = SwarmConfig::new(factory, dir.path());
    config.keep_warm = Some(Duration::from_millis(50));
    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let (swarm, mut events) = Swarm::new(config);
    spawn_root(&swarm, "go").await;
    assert_eq!(reply_to(&mut events, USER).await, "done");
    let refreshes = model.repeats.load(Ordering::SeqCst);
    assert!(
        (3..=8).contains(&refreshes),
        "{refreshes} refreshes in 400ms at 50ms"
    );
    // Each refresh counts as a full cache read of the 1,000-token prompt (plus one output token).
    assert_eq!(swarm.tokens_used(), 2 * 1005 + refreshes as u64 * 1001);
    // A refresh counts as a call for cache pruning: the call after the wait finds the cache warm.
    let agent = built.lock().clone().unwrap();
    let noted = agent.state().get(strands_swarm::pruning::LAST_CALL_KEY);
    let noted = noted.and_then(|at| at.as_u64()).expect("refreshes note the call");
    assert!(
        noted >= started + 100,
        "noted at a refresh during the pause: {noted} vs {started}"
    );

    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        model.repeats.load(Ordering::SeqCst),
        refreshes,
        "the turn is over: no more refreshes"
    );
    assert_eq!(model.calls.load(Ordering::SeqCst), 2 + refreshes);
    swarm.shutdown().await;
}
