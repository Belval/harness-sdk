//! `--keep-cancelled`: a cancelled agent's saved session is removed once it is old enough, and
//! nothing else is.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde_json::json;
use strands_agents::testing::MockMessageModel;
use strands_agents::Agent;
use strands_swarm::state::{self, SwarmState};
use strands_swarm::{
    AgentFactory, AgentSpec, AgentStatus, SpawnRequest, Swarm, SwarmConfig, SwarmEvent, USER,
};
use tokio::sync::mpsc::UnboundedReceiver;

const DAY: i64 = 86_400;

fn factory() -> AgentFactory {
    Arc::new(|spec: AgentSpec| {
        Box::pin(async move {
            Agent::builder()
                .id(spec.id)
                .model(MockMessageModel::shared(vec![]))
                .system_prompt(spec.instructions)
                .tools(spec.tools)
                .no_sandbox()
                .build()
        })
    })
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

fn agent(
    id: &str,
    status: &str,
    cancelled_at: Option<i64>,
    cwd: &Path,
    branch: Option<&str>,
) -> serde_json::Value {
    json!({"id": id, "name": id, "parent": "user", "cwd": cwd, "branch": branch, "task": "t",
           "model": null, "goal": null, "status": status, "mailbox": [], "cancelled_at": cancelled_at})
}

/// A session directory with a conversation snapshots, last written `age` ago.
fn session(dir: &Path, id: &str, age: i64) {
    let session = dir.join("sessions").join(id);
    std::fs::create_dir_all(session.join("snapshots")).unwrap();
    let files = [session.join("snapshot.json"), session.join("snapshots/entry")];
    let when = SystemTime::now() - Duration::from_secs(age as u64);
    for file in files {
        std::fs::write(&file, "x").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&file)
            .unwrap()
            .set_modified(when)
            .unwrap();
    }
    let handle = std::fs::File::open(session.join("snapshots")).unwrap();
    handle.set_modified(when).unwrap();
    std::fs::File::open(&session).unwrap().set_modified(when).unwrap();
}

/// a1 is live; a2 was cancelled 10 days ago, a3 an hour ago, a4 at an unrecorded time (its
/// session was last written 20 days ago), a5 10 days ago with its worktree still there, and a6
/// 10 days ago with no session at all.
fn run(dir: &Path, worktree: &Path) -> SwarmState {
    for (id, age) in [
        ("a1", 30 * DAY),
        ("a2", 10 * DAY),
        ("a3", 3600),
        ("a4", 20 * DAY),
        ("a5", 10 * DAY),
    ] {
        session(dir, id, age);
    }
    serde_json::from_value(json!({
        "run_id": "old", "cwd": dir, "next_id": 7,
        "agents": [
            agent("a1", "idle", None, dir, None),
            agent("a2", "cancelled", Some(now() - 10 * DAY), dir, None),
            agent("a3", "cancelled", Some(now() - 3600), dir, None),
            agent("a4", "cancelled", None, dir, None),
            agent("a5", "cancelled", Some(now() - 10 * DAY), worktree, Some("swarm/old/a5")),
            agent("a6", "cancelled", Some(now() - 10 * DAY), dir, None),
        ],
    }))
    .unwrap()
}

fn kept(dir: &Path) -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(dir.join("sessions"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    ids.sort();
    ids
}

async fn resume(
    dir: &Path,
    saved: SwarmState,
    keep: Option<Duration>,
) -> (Swarm, UnboundedReceiver<SwarmEvent>) {
    let mut config = SwarmConfig::new(factory(), dir);
    config.state_dir = Some(dir.to_path_buf());
    config.keep_cancelled = keep;
    config.reclaim_disk = false;
    Swarm::resume(config, saved).await.unwrap()
}

fn removed(swarm: &Swarm, id: &str) -> Option<u64> {
    swarm
        .snapshot()
        .agents
        .into_iter()
        .find(|a| a.id == id)
        .unwrap()
        .conversation_removed
}

#[tokio::test]
async fn by_default_every_session_is_kept() {
    let dir = tempfile::tempdir().unwrap();
    let worktree = dir.path().join("wt");
    std::fs::create_dir_all(&worktree).unwrap();
    let saved = run(dir.path(), &worktree);
    let (swarm, _events) = resume(dir.path(), saved, None).await;
    assert!(swarm.prune_cancelled_sessions().await.is_empty());
    swarm.shutdown().await;
    assert_eq!(kept(dir.path()), ["a1", "a2", "a3", "a4", "a5"]);
}

#[tokio::test]
async fn a_short_duration_removes_only_old_cancelled_sessions_and_survives_a_resume() {
    let dir = tempfile::tempdir().unwrap();
    let worktree = dir.path().join("wt");
    std::fs::create_dir_all(&worktree).unwrap();
    let saved = run(dir.path(), &worktree);
    let week = Duration::from_secs(7 * DAY as u64);
    let (swarm, mut events) = resume(dir.path(), saved, Some(week)).await;

    // a2 and a4 (by its session's age) go; a1 is live, a3 recent, a5 has its worktree, a6 no session.
    assert_eq!(kept(dir.path()), ["a1", "a3", "a5"]);
    assert_eq!(removed(&swarm, "a2"), Some(7 * DAY as u64));
    assert_eq!(removed(&swarm, "a4"), Some(7 * DAY as u64));
    for id in ["a1", "a3", "a5", "a6"] {
        assert_eq!(removed(&swarm, id), None, "{id}");
    }
    // The entries stay, a4's cancel time now recorded, and the live agent is untouched.
    let agents = swarm.snapshot().agents;
    assert_eq!(agents.len(), 6);
    assert!(agents
        .iter()
        .find(|a| a.id == "a4")
        .unwrap()
        .cancelled_at
        .is_some());
    assert_eq!(swarm.agent("a1").unwrap().status, AgentStatus::Idle);
    let mut announced = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), async {
        while announced.len() < 2 {
            if let Some(SwarmEvent::ConversationRemoved { id, after_secs }) = events.recv().await {
                announced.push((id, after_secs));
            }
        }
    })
    .await
    .expect("viewers are told");
    announced.sort();
    assert_eq!(
        announced,
        [
            ("a2".to_owned(), 7 * DAY as u64),
            ("a4".to_owned(), 7 * DAY as u64)
        ]
    );
    swarm.shutdown().await;
    let events = std::fs::read_to_string(dir.path().join("events.jsonl")).unwrap();
    let line = events
        .lines()
        .find(|line| line.contains("sessions_pruned"))
        .expect(&events);
    let line: serde_json::Value = serde_json::from_str(line).unwrap();
    assert_eq!(
        (line["agents"].as_u64(), line["ids"].clone()),
        (Some(2), json!(["a2", "a4"]))
    );
    assert!(line["bytes"].as_u64().unwrap() > 0);

    // A restart (even without the flag) keeps the mark and tells the viewers again.
    let saved = state::load(dir.path()).await.unwrap().unwrap();
    let (swarm, mut events) = resume(dir.path(), saved, None).await;
    assert_eq!(removed(&swarm, "a2"), Some(7 * DAY as u64));
    let again = tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(event) = events.recv().await {
            if let SwarmEvent::ConversationRemoved { id, after_secs } = event {
                return (id, after_secs);
            }
        }
        panic!("closed")
    })
    .await
    .unwrap();
    assert!(again.0 == "a2" || again.0 == "a4", "{again:?}");
    assert_eq!(again.1, 7 * DAY as u64);
    swarm.shutdown().await;
    assert_eq!(kept(dir.path()), ["a1", "a3", "a5"]);
}

#[tokio::test]
async fn a_live_agent_is_never_removed_however_short_the_duration() {
    let dir = tempfile::tempdir().unwrap();
    let worktree = dir.path().join("wt");
    std::fs::create_dir_all(&worktree).unwrap();
    let saved = run(dir.path(), &worktree);
    let (swarm, _events) = resume(dir.path(), saved, Some(Duration::from_secs(1))).await;
    swarm.prune_cancelled_sessions().await;
    swarm.shutdown().await;
    assert_eq!(kept(dir.path()), ["a1", "a5"]);
}

#[tokio::test]
async fn cancelling_records_when() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = SwarmConfig::new(factory(), dir.path());
    config.state_dir = Some(dir.path().to_path_buf());
    let (swarm, _events) = Swarm::new(config);
    let id = swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "t".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(swarm.snapshot().agents[0].cancelled_at, None);
    swarm.cancel(&id).unwrap();
    let at = swarm.snapshot().agents[0].cancelled_at.expect("recorded");
    assert!((now() - at).abs() < 10, "{at}");
    swarm.shutdown().await;
}
