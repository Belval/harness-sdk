//! Model failures in a long run: throttling, server errors and streams that drop mid-response.
//! Each test checks that the turn retries or fails cleanly, that the parent hears about a
//! failure, and that the agent answers its next message.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use parking_lot::Mutex;
use serde_json::json;
use strands_agents::models::{ModelStreamEvent, StreamOptions};
use strands_agents::retry::DefaultModelRetryStrategy;
use strands_agents::testing::events_for_message;
use strands_agents::types::{ContentBlock, Message, Role, ToolUseBlock};
use strands_agents::{Agent, Error, Model, Result};
use strands_swarm::{AgentFactory, AgentSpec, SpawnRequest, Swarm, SwarmConfig, SwarmEvent, USER};
use tokio::sync::mpsc::UnboundedReceiver;

/// One scripted model call.
#[derive(Clone)]
enum Step {
    Reply(Message),
    /// Bedrock's `ThrottlingException`.
    Throttle,
    /// A 5xx such as `InternalServerException` or `ServiceUnavailableException`.
    ServerError,
    /// Streams the start of `text`, then the connection drops.
    Drop(String),
    /// Bedrock's `ValidationException`: the request itself is wrong.
    Rejected,
}

fn say(text: &str) -> Step {
    Step::Reply(Message::assistant(text))
}

fn tool_call(name: &str, id: &str, input: serde_json::Value) -> Step {
    Step::Reply(Message::new(
        Role::Assistant,
        vec![ContentBlock::ToolUse(ToolUseBlock::new(name, id, input))],
    ))
}

/// A model that plays `steps` in order and records what each call was sent.
struct Flaky {
    steps: Vec<Step>,
    calls: Mutex<Vec<Vec<Message>>>,
}

impl Flaky {
    fn new(steps: Vec<Step>) -> Arc<Self> {
        Arc::new(Self {
            steps,
            calls: Mutex::new(Vec::new()),
        })
    }

    fn calls(&self) -> Vec<Vec<Message>> {
        self.calls.lock().clone()
    }
}

impl Model for Flaky {
    fn model_id(&self) -> Option<String> {
        Some("flaky".into())
    }

    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        _: &'a StreamOptions,
    ) -> BoxStream<'a, Result<ModelStreamEvent>> {
        let index = {
            let mut calls = self.calls.lock();
            calls.push(messages.to_vec());
            calls.len() - 1
        };
        let events: Vec<Result<ModelStreamEvent>> = match self.steps.get(index).cloned() {
            None => vec![Err(Error::model(format!("script exhausted at call {index}")))],
            Some(Step::Reply(message)) => events_for_message(&message, None, None)
                .into_iter()
                .map(Ok)
                .collect(),
            Some(Step::Throttle) => vec![Err(Error::throttled("Too many requests, please wait."))],
            Some(Step::ServerError) => vec![Err(Error::model(
                "The system encountered an unexpected error during processing. Try your request again.",
            ))],
            Some(Step::Rejected) => vec![Err(Error::model(
                "A conversation must alternate between user and assistant roles.",
            ))],
            Some(Step::Drop(text)) => {
                let mut events: Vec<Result<ModelStreamEvent>> =
                    events_for_message(&Message::assistant(text), None, None)
                        .into_iter()
                        .take(3)
                        .map(Ok)
                        .collect();
                events.push(Err(Error::model("dispatch failure: connection reset by peer")));
                events
            }
        };
        Box::pin(futures::stream::iter(events))
    }
}

fn factory(model: Arc<Flaky>) -> AgentFactory {
    Arc::new(move |spec: AgentSpec| {
        let model = model.clone();
        Box::pin(async move {
            // The swarm's own retry policy, as the harness factory installs it.
            let retry = DefaultModelRetryStrategy::new(strands_swarm::retry::policy())?;
            Agent::builder()
                .id(spec.id)
                .model(model)
                .retry_strategy(retry)
                .system_prompt(spec.instructions)
                .tools(spec.tools)
                .no_sandbox()
                .build()
        })
    })
}

async fn start(model: &Arc<Flaky>, cwd: &Path) -> (Swarm, UnboundedReceiver<SwarmEvent>) {
    let (swarm, events) = Swarm::new(SwarmConfig::new(factory(model.clone()), cwd));
    let request = SpawnRequest {
        task: "go".into(),
        ..Default::default()
    };
    swarm.spawn(USER, request).await.unwrap();
    (swarm, events)
}

/// The next mail to the user: `(kind, text)`.
async fn mail(events: &mut UnboundedReceiver<SwarmEvent>) -> (String, String) {
    loop {
        if let Some(SwarmEvent::Mail { to, mail }) = events.recv().await {
            if to == USER {
                return (mail.kind.to_owned(), mail.text);
            }
        }
    }
}

/// Every call alternated user and assistant messages, as Bedrock Converse requires.
fn assert_alternating(model: &Flaky) {
    for (index, messages) in model.calls().iter().enumerate() {
        let roles: Vec<Role> = messages.iter().map(|m| m.role).collect();
        assert!(
            roles.windows(2).all(|pair| pair[0] != pair[1]),
            "call {index} repeats a role: {roles:?}"
        );
    }
}

fn texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

const ATTEMPTS: usize = strands_swarm::retry::MAX_ATTEMPTS as usize;

#[tokio::test(start_paused = true)]
async fn throttling_is_retried_with_backoff_inside_the_turn() {
    let dir = tempfile::tempdir().unwrap();
    let model = Flaky::new(vec![Step::Throttle, Step::Throttle, Step::Throttle, say("done")]);
    let started = tokio::time::Instant::now();
    let (swarm, mut events) = start(&model, dir.path()).await;
    assert_eq!(mail(&mut events).await, ("reply".into(), "done".into()));
    assert_eq!(model.calls().len(), 4);
    assert!(started.elapsed() > Duration::ZERO, "retries wait");
    swarm.shutdown().await;
}

#[tokio::test(start_paused = true)]
async fn server_errors_and_dropped_streams_are_retried_mid_tool_loop() {
    let dir = tempfile::tempdir().unwrap();
    let model = Flaky::new(vec![
        tool_call("list_agents", "t1", json!({})),
        Step::ServerError,
        Step::Drop("half an answ".into()),
        say("whole answer"),
    ]);
    let (swarm, mut events) = start(&model, dir.path()).await;
    assert_eq!(mail(&mut events).await, ("reply".into(), "whole answer".into()));
    let calls = model.calls();
    assert_eq!(calls.len(), 4);
    assert_eq!(calls[1], calls[3], "each retry resends the same conversation");
    assert!(
        !texts(&calls[3]).iter().any(|t| t.contains("half an answ")),
        "no partial reply is kept"
    );
    swarm.shutdown().await;
}

#[tokio::test(start_paused = true)]
async fn a_rejected_request_fails_at_once() {
    let dir = tempfile::tempdir().unwrap();
    let model = Flaky::new(vec![Step::Rejected, say("unused")]);
    let (swarm, mut events) = start(&model, dir.path()).await;
    let (kind, text) = mail(&mut events).await;
    assert_eq!(kind, "error");
    assert!(text.contains("alternate"), "{text}");
    assert_eq!(model.calls().len(), 1, "not retried");
    swarm.shutdown().await;
}

#[tokio::test(start_paused = true)]
async fn an_outage_that_outlasts_the_retries_fails_the_turn_once_and_the_next_message_works() {
    let dir = tempfile::tempdir().unwrap();
    let mut steps = vec![tool_call("list_agents", "t1", json!({}))];
    steps.extend((0..ATTEMPTS).map(|i| {
        if i % 2 == 0 {
            Step::Throttle
        } else {
            Step::ServerError
        }
    }));
    steps.push(say("back"));
    let model = Flaky::new(steps);
    let (swarm, mut events) = start(&model, dir.path()).await;
    let (kind, text) = mail(&mut events).await;
    assert_eq!(kind, "error", "{text}");
    assert_eq!(model.calls().len(), 1 + ATTEMPTS);

    // The failed call left the turn open on the tool results; the next message joins it.
    swarm.send(USER, "a1", "try again").unwrap();
    assert_eq!(mail(&mut events).await, ("reply".into(), "back".into()));
    assert_alternating(&model);
    let last = model.calls().last().unwrap().clone();
    assert!(texts(&last).iter().any(|t| t.contains("try again")), "{last:?}");
    swarm.shutdown().await;
}

#[tokio::test(start_paused = true)]
async fn a_goal_agent_retries_failed_turns_without_paying_for_a_judge_call() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use strands_agents::plugins::vended::{GoalLoop, ValidationOutcome};
    let dir = tempfile::tempdir().unwrap();
    let model = Flaky::new(vec![Step::Rejected, Step::Rejected, say("done")]);
    let judged = Arc::new(AtomicUsize::new(0));
    let (counter, flaky) = (judged.clone(), model.clone());
    let factory: AgentFactory = Arc::new(move |spec: AgentSpec| {
        let (counter, model) = (counter.clone(), flaky.clone());
        Box::pin(async move {
            let judge = GoalLoop::sync_validator(move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(ValidationOutcome::passed())
            });
            Agent::builder()
                .id(spec.id)
                .model(model)
                .system_prompt(spec.instructions)
                .tools(spec.tools)
                .no_sandbox()
                .plugin(strands_swarm::bounded_goal(judge))
                .build()
        })
    });
    let (swarm, mut events) = Swarm::new(SwarmConfig::new(factory, dir.path()));
    let request = SpawnRequest {
        task: "go".into(),
        goal: Some("say done".into()),
        ..Default::default()
    };
    swarm.spawn(USER, request).await.unwrap();
    assert_eq!(
        mail(&mut events).await.0,
        "error",
        "the first failure is reported"
    );
    assert_eq!(
        mail(&mut events).await,
        ("reply".into(), "done".into()),
        "later ones are not"
    );
    assert_eq!(model.calls().len(), 3);
    assert_eq!(judged.load(Ordering::SeqCst), 1, "only the reply was judged");
    assert_alternating(&model);
    swarm.shutdown().await;
}
