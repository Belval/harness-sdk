//! Integration tests for caller-supplied `InvocationState`.
//!
//! Mirrors the Python `invoke_async(prompt, invocation_state=...)` pattern:
//! request-scoped context seeded by the caller is visible to hooks via
//! `event.invocation_state`.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream;
use serde_json::json;

use strands_agents::hooks::BeforeModelCallEvent;
use strands_agents::models::{Model, ModelEventStream, StreamOptions};
use strands_agents::types::messages::Role;
use strands_agents::types::streaming::{ContentBlockDelta, ModelStreamEvent};
use strands_agents::{Agent, InvocationState, Message, StopReason};

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

// A caller-seeded InvocationState is visible to hooks via event.invocation_state.
#[tokio::test]
async fn invoke_with_state_seeds_invocation_state() {
    let seen: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
    let seen_hook = seen.clone();
    let mut agent = Agent::builder()
        .model_boxed(Box::new(TextModel))
        .hook::<BeforeModelCallEvent, _>(move |event| {
            *seen_hook.lock().unwrap() = event.invocation_state.get("user_id");
            Ok(())
        })
        .build();

    let state = InvocationState::new();
    state.set("user_id", json!("u-1"));

    agent.invoke_with_state("hi", state).await.unwrap();
    assert_eq!(*seen.lock().unwrap(), Some(json!("u-1")));
}

// The plain invoke path still works with a fresh (empty) state.
#[tokio::test]
async fn plain_invoke_uses_fresh_state() {
    let empty: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None));
    let empty_hook = empty.clone();
    let mut agent = Agent::builder()
        .model_boxed(Box::new(TextModel))
        .hook::<BeforeModelCallEvent, _>(move |event| {
            *empty_hook.lock().unwrap() = Some(event.invocation_state.get("user_id").is_none());
            Ok(())
        })
        .build();

    agent.invoke("hi").await.unwrap();
    assert_eq!(*empty.lock().unwrap(), Some(true));
}
