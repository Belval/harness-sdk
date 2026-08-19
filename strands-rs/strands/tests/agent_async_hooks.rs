//! Integration test for asynchronous hook callbacks.
//!
//! Demonstrates the Python `ContextManager` pattern is now expressible: a
//! `BeforeModelCallEvent` hook does async work (awaiting) and then reads/writes
//! `event.agent.state`.

use async_trait::async_trait;
use futures::stream;
use serde_json::json;

use strands_agents::hooks::BeforeModelCallEvent;
use strands_agents::models::{Model, ModelEventStream, StreamOptions};
use strands_agents::types::messages::Role;
use strands_agents::types::streaming::{ContentBlockDelta, ModelStreamEvent};
use strands_agents::{Agent, Message, StopReason};

struct TextModel;

#[async_trait]
impl Model for TextModel {
    fn model_id(&self) -> Option<&str> {
        Some("text-model")
    }
    fn stream<'a>(
        &'a self,
        _messages: &'a [Message],
        _options: &'a StreamOptions,
    ) -> ModelEventStream<'a> {
        let events = vec![
            ModelStreamEvent::MessageStart {
                role: Role::Assistant,
            },
            ModelStreamEvent::ContentBlockStart { start: None },
            ModelStreamEvent::ContentBlockDelta {
                delta: ContentBlockDelta::Text("ok".to_string()),
            },
            ModelStreamEvent::ContentBlockStop,
            ModelStreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
            },
        ];
        Box::pin(stream::iter(events.into_iter().map(Ok)))
    }
}

// An async BeforeModelCall hook awaits, then updates agent state — the shape of
// a hook that calls an async conversation manager. State persists across turns.
#[tokio::test]
async fn async_hook_awaits_then_mutates_agent_state() {
    let mut agent = Agent::builder()
        .model_boxed(Box::new(TextModel))
        .hook_async::<BeforeModelCallEvent, _>(|event| {
            Box::pin(async move {
                // Simulate awaiting async work (e.g. conversation management).
                tokio::task::yield_now().await;
                let count = event
                    .agent
                    .state()
                    .get("cycles")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                event.agent.state().set("cycles", json!(count + 1));
                Ok(())
            })
        })
        .build();

    agent.invoke("first").await.unwrap();
    assert_eq!(agent.state().get("cycles"), Some(json!(1)));

    agent.invoke("second").await.unwrap();
    assert_eq!(agent.state().get("cycles"), Some(json!(2)));
}

// A sync and an async hook on the same event both run, in registration order.
#[tokio::test]
async fn sync_and_async_hooks_coexist() {
    use std::sync::{Arc, Mutex};
    let log: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let sync_log = log.clone();
    let async_log = log.clone();

    let mut agent = Agent::builder()
        .model_boxed(Box::new(TextModel))
        .hook::<BeforeModelCallEvent, _>(move |_| {
            sync_log.lock().unwrap().push("sync");
            Ok(())
        })
        .hook_async::<BeforeModelCallEvent, _>(move |_| {
            let async_log = async_log.clone();
            Box::pin(async move {
                tokio::task::yield_now().await;
                async_log.lock().unwrap().push("async");
                Ok(())
            })
        })
        .build();

    agent.invoke("hi").await.unwrap();
    assert_eq!(*log.lock().unwrap(), vec!["sync", "async"]);
}
