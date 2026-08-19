//! Integration tests for conversation management on context-window overflow.
//!
//! Mirrors the strands core behavior: when a model call overflows the context
//! window, the agent asks its `ConversationManager` to reduce the history and
//! retries.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;

use strands_agents::agent::AgentHandle;
use strands_agents::conversation_manager::ConversationManager;
use strands_agents::models::{Model, ModelEventStream, StreamOptions};
use strands_agents::types::messages::Role;
use strands_agents::types::streaming::{ContentBlockDelta, ModelStreamEvent};
use strands_agents::{Agent, InvocationState, Message, StopReason, StrandsError};

/// A model that overflows on its first call, then returns text on the next.
struct FlakyModel {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Model for FlakyModel {
    fn model_id(&self) -> Option<&str> {
        Some("flaky")
    }
    fn stream<'a>(
        &'a self,
        _messages: &'a [Message],
        _options: &'a StreamOptions,
    ) -> ModelEventStream<'a> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let events: Vec<Result<ModelStreamEvent, StrandsError>> = if call == 0 {
            vec![Err(StrandsError::ContextWindowOverflow(
                "Input is too long".to_string(),
            ))]
        } else {
            vec![
                Ok(ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                }),
                Ok(ModelStreamEvent::ContentBlockStart { start: None }),
                Ok(ModelStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::Text("recovered".to_string()),
                }),
                Ok(ModelStreamEvent::ContentBlockStop),
                Ok(ModelStreamEvent::MessageStop {
                    stop_reason: StopReason::EndTurn,
                }),
            ]
        };
        Box::pin(stream::iter(events))
    }
}

/// A manager that shrinks the history to its last message on reduce.
struct ReducingManager {
    reductions: Arc<AtomicUsize>,
}

#[async_trait]
impl ConversationManager for ReducingManager {
    async fn apply_management(
        &self,
        _agent: &AgentHandle,
        _current_tokens: Option<u64>,
        _invocation_state: &InvocationState,
    ) -> Result<(), StrandsError> {
        Ok(())
    }
    async fn reduce_context(
        &self,
        agent: &AgentHandle,
        _error: Option<&str>,
    ) -> Result<(), StrandsError> {
        self.reductions.fetch_add(1, Ordering::SeqCst);
        // Shrink the shared history — the loop retries with the reduced context.
        agent.messages().update(|history| {
            if history.len() > 1 {
                let last = history.pop().unwrap();
                history.clear();
                history.push(last);
            }
        });
        Ok(())
    }
}

// "reduces context and retries the model on context-window overflow"
#[tokio::test]
async fn reduces_and_retries_on_overflow() {
    let calls = Arc::new(AtomicUsize::new(0));
    let reductions = Arc::new(AtomicUsize::new(0));
    let mut agent = Agent::builder()
        .model_boxed(Box::new(FlakyModel {
            calls: calls.clone(),
        }))
        .conversation_manager(ReducingManager {
            reductions: reductions.clone(),
        })
        .build();

    let result = agent.invoke("hi").await.unwrap();

    assert_eq!(result.text(), "recovered");
    assert_eq!(reductions.load(Ordering::SeqCst), 1);
    // The model was called twice: the overflow, then the successful retry.
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

// Without a conversation manager, a context-window overflow propagates.
#[tokio::test]
async fn overflow_propagates_without_manager() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut agent = Agent::builder()
        .model_boxed(Box::new(FlakyModel { calls }))
        .build();

    let error = agent.invoke("hi").await.unwrap_err();
    assert!(matches!(error, StrandsError::ContextWindowOverflow(_)));
}
