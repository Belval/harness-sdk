//! Integration test: an async hook persists the session via the manager exposed
//! on `event.agent`, mirroring the Python `ContextManager` calling
//! `agent._session_manager.sync_agent(agent)` after a conversation change.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream;

use strands_agents::agent::AgentHandle;
use strands_agents::hooks::BeforeModelCallEvent;
use strands_agents::models::{Model, ModelEventStream, StreamOptions};
use strands_agents::session::SessionManager;
use strands_agents::types::messages::Role;
use strands_agents::types::streaming::{ContentBlockDelta, ModelStreamEvent};
use strands_agents::{Agent, Message, StopReason, StrandsError};

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

/// Records how many messages the agent had at each `sync_agent` call.
struct RecordingSession {
    synced_lengths: Arc<Mutex<Vec<usize>>>,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl SessionManager for RecordingSession {
    async fn sync_agent(&self, agent: &AgentHandle) -> Result<(), StrandsError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.synced_lengths
            .lock()
            .unwrap()
            .push(agent.messages().len());
        Ok(())
    }
}

// An async BeforeModelCall hook reaches the session manager via event.agent and
// calls sync_agent, which observes the current conversation.
#[tokio::test]
async fn async_hook_syncs_via_session_manager() {
    let calls = Arc::new(AtomicUsize::new(0));
    let synced_lengths = Arc::new(Mutex::new(Vec::new()));

    let mut agent = Agent::builder()
        .model_boxed(Box::new(TextModel))
        .session_manager(RecordingSession {
            synced_lengths: synced_lengths.clone(),
            calls: calls.clone(),
        })
        .hook_async::<BeforeModelCallEvent, _>(|event| {
            Box::pin(async move {
                if let Some(manager) = event.agent.session_manager() {
                    manager.sync_agent(&event.agent).await?;
                }
                Ok(())
            })
        })
        .build();

    agent.invoke("hi").await.unwrap();

    // Synced once, and it saw the user message already in history at model-call time.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(*synced_lengths.lock().unwrap(), vec![1]);
    // The manager is also reachable directly off the agent.
    assert!(agent.session_manager().is_some());
}
