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
use strands_agents::Agent;
use strands_swarm::{
    AgentFactory, AgentSpec, AgentStatus, SpawnRequest, Swarm, SwarmConfig, SwarmEvent, USER,
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
    assert!(third.contains("a2 (counter)"), "{third}");
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
    assert_eq!(reply_to(&mut events, USER).await, "second");
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
                    tool_call("spawn_agent", "t1", json!({"task": "sub"})),
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
            tool_call("spawn_agent", "t1", json!({"task": "sub"})),
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

    assert_eq!(reply_to(&mut events, USER).await, "got it");
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
