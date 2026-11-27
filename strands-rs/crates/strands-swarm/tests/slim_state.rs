//! `swarm.json` stays small as cancelled agents pile up: a cancelled agent keeps a prefix of its
//! task (the rest is in its session directory) and a ledger folded to totals, and nothing a view
//! shows changes.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use strands_agents::testing::MockMessageModel;
use strands_agents::Agent;
use strands_swarm::state::{self, SwarmState, TASK_FILE, TASK_PREFIX_CHARS};
use strands_swarm::{AgentFactory, AgentSpec, SpawnRequest, Swarm, SwarmConfig, SwarmEvent, USER};
use tokio::sync::mpsc::UnboundedReceiver;

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

fn config(dir: &Path) -> SwarmConfig {
    let mut config = SwarmConfig::new(factory(), dir);
    config.state_dir = Some(dir.to_path_buf());
    config.reclaim_disk = false;
    config
}

fn long_task() -> String {
    format!("{} ünïcode end", "write a very long brief. ".repeat(80))
}

/// A ledger shaped like the live run's: totals by model and 50 dated calls, hours old.
fn ledger() -> serde_json::Value {
    let now = chrono::Utc::now().timestamp();
    let recent: Vec<_> = (0..50)
        .map(|n| {
            json!({"input": 200_000 + n * 400, "cache_read": 199_000 + n * 400,
                   "at": now - 7200 + n * 30})
        })
        .collect();
    json!({"by_model": {"global.anthropic.claude-opus-5-5":
        {"calls": 171, "input": 23_983_367, "output": 96_996, "cache_read": 23_499_290,
         "cache_write": 479_879}},
        "recent": recent})
}

fn old_run(dir: &Path, task: &str) -> SwarmState {
    let agent = |id: &str, status: &str, task: &str| {
        json!({"id": id, "name": id, "parent": "user", "cwd": dir, "branch": null, "task": task,
               "model": null, "goal": null, "status": status, "mailbox": [],
               "input_tokens": 23_983_367, "output_tokens": 96_996, "ledger": ledger()})
    };
    serde_json::from_value(json!({
        "run_id": "old", "cwd": dir, "next_id": 4,
        "agents": [agent("a1", "idle", task), agent("a2", "cancelled", task),
                   agent("a3", "cancelled", "short")],
    }))
    .unwrap()
}

async fn spawned_task(events: &mut UnboundedReceiver<SwarmEvent>, wanted: &str) -> String {
    tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(event) = events.recv().await {
            if let SwarmEvent::Spawned { id, task, .. } = event {
                if id == wanted {
                    return task;
                }
            }
        }
        panic!("closed")
    })
    .await
    .expect("Spawned")
}

/// The saved run state as one file: `swarm.json` with the agents of `cancelled.json` added.
fn saved(dir: &Path) -> serde_json::Value {
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap()).unwrap()
    };
    let mut file = read(state::STATE_FILE);
    let cancelled = read(state::CANCELLED_FILE)["agents"].as_array().unwrap().clone();
    file["agents"].as_array_mut().unwrap().extend(cancelled);
    file
}

fn agent<'a>(file: &'a serde_json::Value, id: &str) -> &'a serde_json::Value {
    file["agents"]
        .as_array()
        .unwrap()
        .iter()
        .find(|agent| agent["id"] == id)
        .unwrap()
}

#[tokio::test]
async fn cancelling_keeps_the_whole_task_in_the_session_and_a_prefix_in_swarm_json() {
    let dir = tempfile::tempdir().unwrap();
    let (swarm, _events) = Swarm::new(config(dir.path()));
    let task = long_task();
    let long = swarm
        .spawn(
            USER,
            SpawnRequest {
                task: task.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let short = swarm
        .spawn(
            USER,
            SpawnRequest {
                task: "short".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    swarm.cancel(&long).unwrap();
    swarm.cancel(&short).unwrap();
    swarm.shutdown().await;

    let file = saved(dir.path());
    let prefix = agent(&file, &long)["task"].as_str().unwrap();
    assert_eq!(prefix.chars().count(), TASK_PREFIX_CHARS + 1);
    assert!(task.starts_with(prefix.trim_end_matches('…')));
    assert_eq!(agent(&file, &long)["task_archived"], true);
    let kept = dir.path().join("sessions").join(&long).join(TASK_FILE);
    assert_eq!(std::fs::read_to_string(kept).unwrap(), task);
    // A task that is all prefix is saved whole, with no file.
    assert_eq!(agent(&file, &short)["task"], "short");
    assert!(agent(&file, &short).get("task_archived").is_none());
    assert!(!dir.path().join("sessions").join(&short).join(TASK_FILE).exists());

    // A restart shows the whole task again, and saves the same file.
    let before = saved(dir.path())["agents"].clone();
    let loaded = state::load(dir.path()).await.unwrap().unwrap();
    let (swarm, mut events) = Swarm::resume(config(dir.path()), loaded).await.unwrap();
    assert_eq!(spawned_task(&mut events, &long).await, task);
    swarm.shutdown().await;
    assert_eq!(saved(dir.path())["agents"], before);
}

#[tokio::test]
async fn a_resume_compacts_the_cancelled_agents_of_an_old_file_and_only_those() {
    let dir = tempfile::tempdir().unwrap();
    let task = long_task();
    let (swarm, mut events) = Swarm::resume(config(dir.path()), old_run(dir.path(), &task))
        .await
        .unwrap();
    let before = swarm.cost();
    assert_eq!(spawned_task(&mut events, "a2").await, task);
    swarm.shutdown().await;

    let file = saved(dir.path());
    assert_eq!(file["agents"].as_array().unwrap().len(), 3);
    // The live agent is untouched.
    assert_eq!(agent(&file, "a1")["task"], task.as_str());
    assert_eq!(agent(&file, "a1")["ledger"].get("folded"), None);
    assert_eq!(
        agent(&file, "a1")["ledger"]["recent"].as_array().unwrap().len(),
        50
    );
    // The cancelled one is folded.
    let a2 = agent(&file, "a2");
    assert_eq!(a2["task_archived"], true);
    assert!(a2["task"].as_str().unwrap().len() < 600);
    assert_eq!(a2["ledger"].get("recent"), None);
    assert_eq!(a2["ledger"]["folded"]["calls"], 50);
    assert_eq!(a2["ledger"]["by_model"].as_object().unwrap().len(), 1);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("sessions/a2").join(TASK_FILE)).unwrap(),
        task
    );

    // Again, from the compacted file: nothing moves, and the numbers stay.
    let agents = file["agents"].clone();
    let loaded = state::load(dir.path()).await.unwrap().unwrap();
    let (swarm, mut events) = Swarm::resume(config(dir.path()), loaded).await.unwrap();
    assert_eq!(spawned_task(&mut events, "a2").await, task);
    let after = swarm.cost();
    assert_eq!(before.lines(), after.lines());
    assert_eq!(before.total(), after.total());
    assert_eq!(before.spend("a2"), after.spend("a2"));
    swarm.shutdown().await;
    assert_eq!(saved(dir.path())["agents"], agents);
}

#[tokio::test]
async fn a_removed_session_falls_back_to_the_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let task = long_task();
    let mut old = old_run(dir.path(), &task);
    old.agents[1].conversation_removed = Some(86_400);
    let (swarm, mut events) = Swarm::resume(config(dir.path()), old).await.unwrap();
    assert_eq!(spawned_task(&mut events, "a2").await, task);
    swarm.shutdown().await;
    assert!(!dir.path().join("sessions/a2").join(TASK_FILE).exists());
    assert_eq!(agent(&saved(dir.path()), "a2")["task_archived"], true);

    let loaded = state::load(dir.path()).await.unwrap().unwrap();
    let (swarm, mut events) = Swarm::resume(config(dir.path()), loaded).await.unwrap();
    assert_eq!(spawned_task(&mut events, "a2").await, state::task_prefix(&task));
    swarm.shutdown().await;
}

#[tokio::test]
async fn an_unwritable_task_file_leaves_the_task_in_swarm_json() {
    let dir = tempfile::tempdir().unwrap();
    // A file where the session directory should be: nothing can be written under it.
    std::fs::create_dir_all(dir.path().join("sessions")).unwrap();
    std::fs::write(dir.path().join("sessions/a2"), "x").unwrap();
    let task = long_task();
    let (swarm, _events) = Swarm::resume(config(dir.path()), old_run(dir.path(), &task))
        .await
        .unwrap();
    swarm.shutdown().await;
    let file = saved(dir.path());
    assert_eq!(agent(&file, "a2")["task"], task.as_str());
    assert!(agent(&file, "a2").get("task_archived").is_none());
}
