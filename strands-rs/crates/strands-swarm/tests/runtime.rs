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
use strands_swarm::{
    AgentFactory, AgentSpec, AgentStatus, ModelResolver, SpawnRequest, Swarm, SwarmConfig, SwarmEvent, USER,
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

#[tokio::test]
async fn spawning_beyond_the_live_agent_limit_fails() {
    let dir = tempfile::tempdir().unwrap();
    let map: Scripts = Arc::new(Mutex::new(HashMap::new()));
    map.lock()
        .insert("a1".into(), MockMessageModel::shared(vec![say("hi")]));
    let mut config = SwarmConfig::new(mock_factory(&map), dir.path());
    config.max_agents = 1;
    let (swarm, _events) = Swarm::new(config);

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
    let error = swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "two".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("Agent limit reached"), "{error}");
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

fn git(dir: &Path, args: &[&str]) {
    let status = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .status()
        .unwrap();
    assert!(status.success(), "git {args:?}");
}

#[tokio::test]
async fn a_worktree_agent_gets_its_own_checkout_and_branch() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(repo.join("sub")).unwrap();
    std::fs::write(repo.join("sub/file.txt"), "x").unwrap();
    git(&repo, &["init", "-q"]);
    git(&repo, &["-c", "user.name=t", "-c", "user.email=t@t", "add", "."]);
    git(
        &repo,
        &[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-qm",
            "init",
        ],
    );

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
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    git(&repo, &["init", "-q"]);
    git(
        &repo,
        &[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "init",
        ],
    );
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
    let a1_call = sent_text(&scripts.lock()["a1"].clone(), 0);
    assert!(
        a1_call.contains("spawned") && a1_call.contains("resumed"),
        "{a1_call}"
    );
    assert_eq!(resumed.agents().len(), 2);
}

#[tokio::test]
async fn reload_installs_the_binary_keeps_the_previous_one_and_signals_the_host() {
    let dir = tempfile::tempdir().unwrap();
    let map: Scripts = Arc::new(Mutex::new(HashMap::new()));
    let mut config = SwarmConfig::new(mock_factory(&map), dir.path());
    config.state_dir = Some(dir.path().join("run"));
    let (swarm, _events) = Swarm::new(config);
    let bin = dir.path().join("run/bin");

    swarm.reload(Path::new("/bin/true")).await.unwrap();
    tokio::time::timeout(Duration::from_secs(1), swarm.reload_requested())
        .await
        .expect("reload not signalled");
    assert!(bin.join("current").exists());

    swarm.reload(Path::new("/bin/true")).await.unwrap();
    assert!(bin.join("previous").exists());

    let error = swarm.reload(Path::new("/bin/false")).await.unwrap_err();
    assert!(error.to_string().contains("--version failed"), "{error}");
}

#[tokio::test]
async fn reload_needs_a_persistent_run() {
    let dir = tempfile::tempdir().unwrap();
    let (swarm, _events, _scripts) = swarm_with(vec![], dir.path());
    let error = swarm.reload(Path::new("/bin/true")).await.unwrap_err();
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
    let (swarm, mut events) = Swarm::new(config.clone());
    let id = spawn_root(&swarm, "count").await;

    let seen = usage_until_reply(&mut events, &id).await;
    // Context is the last call's input over the 4000-token window.
    assert_eq!(
        seen.first(),
        Some(&(1000, 10, Some(0.25))),
        "mid-turn report: {seen:?}"
    );
    assert_eq!(seen.last(), Some(&(1005, 11, Some(5.0 / 4000.0))), "{seen:?}");
    swarm.shutdown().await;

    let saved = strands_swarm::state::load(&state_dir)
        .await
        .unwrap()
        .expect("state saved");
    assert_eq!(
        (saved.agents[0].input_tokens, saved.agents[0].output_tokens),
        (1005, 11)
    );
    assert_eq!(saved.agents[0].context, Some(5.0 / 4000.0));
    let (_resumed, mut events) = Swarm::resume(config, saved).await.unwrap();
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
    assert_eq!(restored, (1005, 11, Some(5.0 / 4000.0)));
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

/// A repo with one commit at `<dir>/repo`, ignoring `target/`.
fn init_repo(dir: &Path) -> std::path::PathBuf {
    let repo = dir.join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(repo.join(".gitignore"), "target/\n").unwrap();
    git(&repo, &["init", "-q"]);
    git(&repo, &["add", "."]);
    git(
        &repo,
        &[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-qm",
            "init",
        ],
    );
    repo
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
    std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "--verify", "--quiet", branch])
        .stdout(std::process::Stdio::null())
        .status()
        .unwrap()
        .success()
}

#[tokio::test]
async fn cancelling_a_worktree_agent_reclaims_its_target_dir_and_clean_worktree() {
    let dir = tempfile::tempdir().unwrap();
    let repo = init_repo(dir.path());
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
    assert!(branch_exists(&repo, &branch), "the branch stays");
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
    let repo = init_repo(dir.path());
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
    git(
        &checkout,
        &[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-qm",
            "wip",
        ],
    );
    let saved = strands_swarm::state::load(&dir.path().join("run"))
        .await
        .unwrap()
        .unwrap();
    let (resumed, _events) = Swarm::resume(config, saved).await.unwrap();
    eventually("the worktree is removed on resume", || !checkout.exists()).await;
    assert!(branch_exists(
        &repo,
        &format!("swarm/{}/{owner}", resumed.run_id())
    ));
    assert!(resumed.gc().await.iter().all(|c| c.is_empty()), "nothing left");
    resumed.shutdown().await;
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
