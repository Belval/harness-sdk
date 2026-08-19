//! Integration tests for agent-scoped state exposed to hooks via `event.agent`.
//!
//! Mirrors the Python `ContextManager` pattern of reading/writing
//! `event.agent.state` across cycles and invocations (cycle counters, etc.).

use std::sync::Mutex;

use async_trait::async_trait;
use futures::stream;
use serde_json::json;

use strands_agents::hooks::BeforeModelCallEvent;
use strands_agents::models::{Model, ModelEventStream, StreamOptions};
use strands_agents::types::messages::Role;
use strands_agents::types::streaming::{ContentBlockDelta, ModelStreamEvent};
use strands_agents::{Agent, AgentState, Message, StopReason};

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

// A BeforeModelCall hook increments a counter in agent.state; state persists
// across invocations, matching ContextManager's `_cycle_count` usage.
#[tokio::test]
async fn hook_reads_and_writes_agent_state_across_invocations() {
    let mut agent = Agent::builder()
        .model_boxed(Box::new(TextModel))
        .hook::<BeforeModelCallEvent, _>(|event| {
            let count = event
                .agent
                .state()
                .get("count")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            event.agent.state().set("count", json!(count + 1));
            Ok(())
        })
        .build();

    agent.invoke("first").await.unwrap();
    assert_eq!(agent.state().get("count"), Some(json!(1)));

    agent.invoke("second").await.unwrap();
    // Persisted across invocations: the second turn sees the first turn's value.
    assert_eq!(agent.state().get("count"), Some(json!(2)));
}

// State seeded on the builder is visible to hooks and to the agent.
#[tokio::test]
async fn builder_seeds_agent_state() {
    let seen: std::sync::Arc<Mutex<Option<serde_json::Value>>> =
        std::sync::Arc::new(Mutex::new(None));
    let seen_hook = seen.clone();
    let mut agent = Agent::builder()
        .model_boxed(Box::new(TextModel))
        .state(AgentState::from_value(json!({ "user_id": "u-123" })))
        .hook::<BeforeModelCallEvent, _>(move |event| {
            *seen_hook.lock().unwrap() = event.agent.state().get("user_id");
            Ok(())
        })
        .build();

    assert_eq!(agent.state().get("user_id"), Some(json!("u-123")));
    agent.invoke("hi").await.unwrap();
    assert_eq!(*seen.lock().unwrap(), Some(json!("u-123")));
}
