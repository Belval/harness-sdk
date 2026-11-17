//! A lone `cd` through the whole swarm: real harness agents driven by the runtime move, their
//! children start where they moved to, and the move survives a restart ([`crate::workdir`]).

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

async fn calls_reach(model: &MockMessageModel, count: usize) {
    let waiting = async {
        while model.call_count() < count {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(20), waiting)
        .await
        .expect("the calls");
}

type Call = (Vec<Message>, StreamOptions);

/// The last message of a call as the provider sees it.
fn last(call: &Call) -> String {
    serde_json::to_string(&call.0.last().unwrap().content).unwrap()
}

fn prompt(call: &Call) -> String {
    call.1.system_prompt.as_ref().unwrap().text()
}

#[tokio::test]
async fn a_lone_cd_moves_the_agent_and_its_children_and_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let other = root.join("other");
    std::fs::create_dir(&other).unwrap();
    std::fs::write(other.join("AGENTS.md"), "Other rules.").unwrap();
    let state_dir = root.join("run");
    let scripts: Scripts = Arc::default();
    let script = |id: &str, responses| {
        scripts
            .lock()
            .insert(id.to_owned(), MockMessageModel::shared(responses))
    };
    script(
        "a1",
        vec![
            call("shell", "t1", json!({"command": "cd other"})),
            call("shell", "t2", json!({"command": "pwd -P"})),
            call("spawn_agent", "t3", json!({"task": "look", "name": "helper"})),
            say("moved"),
            say("thanks"),
        ],
    );
    script(
        "a2",
        vec![call("shell", "t1", json!({"command": "pwd -P"})), say("seen")],
    );
    let mut config = SwarmConfig::new(factory(&scripts), &root);
    config.state_dir = Some(state_dir.clone());
    let (swarm, mut events) = Swarm::new(config.clone());
    let request = SpawnRequest {
        task: "go".into(),
        ..Default::default()
    };
    swarm.spawn(USER, request).await.unwrap();
    assert_eq!(reply(&mut events).await, "moved");
    let (first, helper) = {
        let scripts = scripts.lock();
        (scripts["a1"].clone(), scripts["a2"].clone())
    };
    calls_reach(&helper, 2).await;
    calls_reach(&first, 5).await;
    let info = swarm.agent("a1").unwrap();
    assert_eq!((info.cwd, info.workdir), (root.clone(), other.clone()));
    swarm.shutdown().await;

    let calls = first.calls();
    let shown = other.display().to_string();
    assert!(
        last(&calls[1]).contains("Your working directory is now"),
        "{}",
        last(&calls[1])
    );
    assert!(
        last(&calls[2]).contains(&format!(r#""text":"{shown}""#)),
        "pwd: {}",
        last(&calls[2])
    );
    let sentence = |dir: &str| format!("Your working directory is `{dir}`");
    assert!(prompt(&calls[0]).contains(&sentence(&root.display().to_string())));
    assert!(
        prompt(&calls[2]).contains(&sentence(&shown)),
        "{}",
        prompt(&calls[2])
    );
    let first_turn = serde_json::to_string(&calls[0].0).unwrap();
    assert!(!first_turn.contains("Other rules."));
    assert_eq!(
        first_turn.matches("<environment>").count(),
        1,
        "only the agent's own environment, not also the process's: {first_turn}"
    );
    let moved_turn = last(&calls[4]);
    assert!(moved_turn.contains("Other rules."), "the environment follows");
    let helper_calls = helper.calls();
    assert!(prompt(&helper_calls[0]).contains(&sentence(&shown)));
    assert!(last(&helper_calls[1]).contains(&format!(r#""text":"{shown}""#)));

    let saved = crate::state::load(&state_dir)
        .await
        .unwrap()
        .expect("state saved");
    let a1 = saved.agents.iter().find(|agent| agent.id == "a1").unwrap();
    assert_eq!((&a1.cwd, a1.workdir.as_ref()), (&root, Some(&other)));
    let a2 = saved.agents.iter().find(|agent| agent.id == "a2").unwrap();
    assert_eq!((&a2.cwd, a2.workdir.as_ref()), (&other, None));

    script("a1", vec![say("back")]);
    let (swarm, mut events) = Swarm::resume(config, saved).await.unwrap();
    assert_eq!(swarm.agent("a1").unwrap().workdir, other);
    swarm.send(USER, "a1", "again").unwrap();
    assert_eq!(reply(&mut events).await, "back");
    let resumed = scripts.lock()["a1"].clone();
    swarm.shutdown().await;
    assert!(prompt(&resumed.calls()[0]).contains(&sentence(&shown)));
}
