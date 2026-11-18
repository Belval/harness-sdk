//! Reload self-test: a new binary resumes a copy of the run on a stub model before it is
//! installed; the real run is never touched, and a failing binary is refused.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::json;
use strands_agents::testing::{MockMessageModel, MockResponse};
use strands_agents::types::{ContentBlock, Message, Role, ToolUseBlock};
use strands_agents::Agent;
use strands_swarm::{
    selftest, AgentFactory, AgentSpec, ReloadWhen, SpawnRequest, Swarm, SwarmConfig, SwarmEvent, USER,
};

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

fn session_factory(scripts: &Scripts, sessions: PathBuf) -> AgentFactory {
    let scripts = scripts.clone();
    Arc::new(move |spec: AgentSpec| {
        let model = scripts.lock()[&spec.id].clone();
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

/// A saved run with a root and a child that both have conversations, plus a `bin/` directory.
/// Returns the run directory and a config that resumes it on (idle) mocks.
async fn saved_run(root: &Path) -> (PathBuf, SwarmConfig) {
    let state_dir = root.join("run");
    let scripts: Scripts = Arc::new(Mutex::new(HashMap::new()));
    let script = vec![
        tool_call("spawn_agent", "t1", json!({"task": "sub", "name": "sub"})),
        tool_call("wait", "t2", json!({"from": ["sub"], "timeout_seconds": 10})),
        say("done"),
    ];
    scripts
        .lock()
        .insert("a1".into(), MockMessageModel::shared(script));
    scripts
        .lock()
        .insert("a2".into(), MockMessageModel::shared(vec![say("sub done")]));
    let mut config = SwarmConfig::new(session_factory(&scripts, state_dir.join("sessions")), root);
    config.state_dir = Some(state_dir.clone());
    let (swarm, mut events) = Swarm::new(config.clone());
    let request = SpawnRequest {
        task: "go".into(),
        ..Default::default()
    };
    swarm.spawn(USER, request).await.unwrap();
    let done = async {
        while let Some(event) = events.recv().await {
            if let SwarmEvent::Mail { to, mail } = event {
                if to == USER && mail.kind == "reply" {
                    return mail.text;
                }
            }
        }
        panic!("event stream closed");
    };
    let reply = tokio::time::timeout(Duration::from_secs(10), done).await.unwrap();
    assert_eq!(reply, "done");
    swarm.shutdown().await;
    std::fs::create_dir_all(state_dir.join("bin")).unwrap();
    std::fs::write(state_dir.join("bin/current"), "old binary").unwrap();
    (state_dir, config)
}

/// Every file under `dir` with its contents.
fn tree(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(path) = stack.pop() {
        if path.is_dir() {
            stack.extend(std::fs::read_dir(&path).unwrap().map(|e| e.unwrap().path()));
        } else {
            files.insert(path.clone(), std::fs::read(&path).unwrap());
        }
    }
    files
}

/// An executable shell script. Written by a child process: a write descriptor held by this
/// (multi-threaded) process would leak into children other tests fork concurrently, and executing
/// the script would then fail with "Text file busy".
fn script(dir: &Path, body: &str) -> PathBuf {
    let path = dir.join("fake-strands-swarm");
    let status = std::process::Command::new("sh")
        .args(["-c", "printf '%s\\n' \"$1\" > \"$2\" && chmod 755 \"$2\"", "sh"])
        .arg(format!("#!/bin/sh\n{body}"))
        .arg(&path)
        .status()
        .unwrap();
    assert!(status.success());
    path
}

#[tokio::test]
async fn self_test_resumes_a_copy_and_leaves_the_run_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let (state_dir, _config) = saved_run(dir.path()).await;
    let before = tree(&state_dir);

    let copy = dir.path().join("copy");
    selftest::copy_run(&state_dir, &copy).unwrap();
    assert!(copy.join("swarm.json").exists() && copy.join("sessions/a1").is_dir());
    assert!(!copy.join("bin").exists());
    let summary = selftest::run(&copy).await.unwrap();
    assert!(summary.contains("2 live agents replied"), "{summary}");

    assert_eq!(tree(&state_dir), before, "the real run changed");
}

#[tokio::test]
async fn self_test_passes_a_run_whose_agent_had_errored_before_the_restart() {
    let dir = tempfile::tempdir().unwrap();
    let (state_dir, _config) = saved_run(dir.path()).await;
    let copy = dir.path().join("copy");
    selftest::copy_run(&state_dir, &copy).unwrap();
    let path = copy.join("swarm.json");
    let mut saved: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    for agent in saved["agents"].as_array_mut().unwrap() {
        if agent["id"] == "a2" {
            agent["status"] = json!("errored");
        }
    }
    std::fs::write(&path, saved.to_string()).unwrap();
    // Its restored "last turn failed" status is old news; it still answers the probe.
    let summary = selftest::run(&copy).await.unwrap();
    assert!(summary.contains("2 live agents replied"), "{summary}");
}

#[tokio::test]
async fn self_test_fails_when_a_conversation_cannot_be_restored() {
    let dir = tempfile::tempdir().unwrap();
    let (state_dir, _config) = saved_run(dir.path()).await;
    let copy = dir.path().join("copy");
    selftest::copy_run(&state_dir, &copy).unwrap();
    for path in tree(&copy.join("sessions/a2")).keys() {
        std::fs::write(path, "not json").unwrap();
    }
    let error = selftest::run(&copy).await.unwrap_err().to_string();
    assert!(error.contains("sub (a2)"), "{error}");
}

#[tokio::test]
async fn reload_installs_a_binary_that_passes_its_self_test() {
    let dir = tempfile::tempdir().unwrap();
    let (state_dir, config) = saved_run(dir.path()).await;
    let saved = strands_swarm::state::load(&state_dir).await.unwrap().unwrap();
    let (swarm, _events) = Swarm::resume(config, saved).await.unwrap();

    // The real binary, which resumes the copy on its stub model.
    let binary = Path::new(env!("CARGO_BIN_EXE_strands-swarm"));
    swarm.reload(binary, ReloadWhen::Now).await.unwrap();
    let installed = std::fs::read(state_dir.join("bin/current")).unwrap();
    assert_eq!(installed, std::fs::read(binary).unwrap());
    assert_eq!(
        std::fs::read(state_dir.join("bin/previous")).unwrap(),
        b"old binary"
    );

    // A fake that checks what it was given: the copy has the state but no binaries.
    let checks = script(
        dir.path(),
        "[ \"$1\" = --version ] && exit 0\n[ -f \"$2/swarm.json\" ] && [ -d \"$2/sessions\" ] && [ ! -e \"$2/bin\" ]",
    );
    swarm.reload(&checks, ReloadWhen::Now).await.unwrap();
    swarm.shutdown().await;
}

#[tokio::test]
async fn reload_refuses_a_binary_that_fails_its_self_test() {
    let dir = tempfile::tempdir().unwrap();
    let (state_dir, config) = saved_run(dir.path()).await;
    let saved = strands_swarm::state::load(&state_dir).await.unwrap().unwrap();
    let (swarm, _events) = Swarm::resume(config, saved).await.unwrap();
    let marker = dir.path().join("copy-path");
    let failing = script(
        dir.path(),
        &format!(
            "[ \"$1\" = --version ] && exit 0\necho \"$2\" > {}\necho 'boom: cannot restore a1' >&2\nexit 3",
            marker.display()
        ),
    );

    let error = swarm
        .reload(&failing, ReloadWhen::Now)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("boom: cannot restore a1"), "{error}");
    assert!(error.contains("keeps its current binary"), "{error}");
    assert_eq!(
        std::fs::read(state_dir.join("bin/current")).unwrap(),
        b"old binary"
    );
    let copy = std::fs::read_to_string(&marker).unwrap();
    assert!(
        !Path::new(copy.trim()).exists(),
        "the temporary copy was left behind"
    );
    swarm.shutdown().await;
}

/// A saved run whose cancelled worktree agent `a2` owns a private target dir that exists on disk,
/// i.e. one that disk reclamation would delete. Returns the run dir and that target dir.
fn run_with_a_reclaimable_agent(root: &Path) -> (PathBuf, PathBuf) {
    let state_dir = root.join("run");
    let target = root.join("target-worktrees/live/a2");
    std::fs::create_dir_all(&target).unwrap();
    std::fs::write(target.join("build-output"), "live build").unwrap();
    let agent = |id: &str, parent: &str, status: &str| {
        json!({"id": id, "name": id, "parent": parent, "cwd": root, "branch": format!("swarm/live/{id}"),
               "task": "t", "model": null, "goal": null, "status": status, "mailbox": [],
               "env": {"CARGO_TARGET_DIR": root.join("target-worktrees/live").join(id)}})
    };
    let saved = json!({"run_id": "live", "cwd": root, "next_id": 3,
                       "agents": [agent("a1", "user", "idle"), agent("a2", "a1", "cancelled")]});
    std::fs::create_dir_all(&state_dir).unwrap();
    std::fs::write(state_dir.join("swarm.json"), saved.to_string()).unwrap();
    (state_dir, target)
}

#[tokio::test]
async fn self_test_never_reclaims_the_live_runs_directories() {
    let dir = tempfile::tempdir().unwrap();
    let (state_dir, target) = run_with_a_reclaimable_agent(dir.path());
    let scripts: Scripts = Arc::new(Mutex::new(HashMap::new()));
    scripts
        .lock()
        .insert("a1".into(), MockMessageModel::shared(vec![]));
    let config = |reclaim_disk: bool| {
        let mut config = SwarmConfig::new(session_factory(&scripts, dir.path().join("s")), dir.path());
        config.run_id = "live".into();
        config.reclaim_disk = reclaim_disk;
        config
    };
    let saved = || {
        let text = std::fs::read_to_string(state_dir.join("swarm.json")).unwrap();
        serde_json::from_str(&text).unwrap()
    };

    // With reclamation off, neither resume nor an explicit gc touches the directory.
    let (swarm, _events) = Swarm::resume(config(false), saved()).await.unwrap();
    assert!(swarm.gc().await.iter().all(|c| c.removed.is_empty()));
    swarm.shutdown().await;
    assert!(target.join("build-output").exists());

    // The self-test of a copy leaves it alone too.
    let copy = dir.path().join("copy");
    selftest::copy_run(&state_dir, &copy).unwrap();
    selftest::run(&copy).await.unwrap();
    assert!(
        target.join("build-output").exists(),
        "the self-test reclaimed a live directory"
    );

    // Control: a normal swarm does reclaim it, so the checks above are meaningful.
    let (swarm, _events) = Swarm::resume(config(true), saved()).await.unwrap();
    let removed: Vec<PathBuf> = swarm
        .gc()
        .await
        .into_iter()
        .flat_map(|c| c.removed.into_iter().map(|(path, _)| path))
        .collect();
    swarm.shutdown().await;
    assert!(removed.contains(&target) || !target.exists(), "{removed:?}");
    assert!(!target.exists());
}
