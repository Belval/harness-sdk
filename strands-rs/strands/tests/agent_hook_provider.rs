//! Integration test for `HookProvider` — bundling several hook registrations
//! (sync and async) into one object registered as a unit.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream;

use strands_agents::hooks::{
    AfterModelCallEvent, BeforeModelCallEvent, HookProvider, HookRegistry,
};
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

/// Bundles a sync before-model hook and an async after-model hook.
struct MeteringProvider {
    fired: Arc<Mutex<Vec<&'static str>>>,
}

impl HookProvider for MeteringProvider {
    fn register_hooks(&self, registry: &HookRegistry) {
        let before = self.fired.clone();
        registry.add_callback::<BeforeModelCallEvent, _>(move |_event| {
            before.lock().unwrap().push("before");
            Ok(())
        });

        let after = self.fired.clone();
        registry.add_callback_async::<AfterModelCallEvent, _>(move |_event| {
            let after = after.clone();
            Box::pin(async move {
                tokio::task::yield_now().await;
                after.lock().unwrap().push("after");
                Ok(())
            })
        });
    }
}

// "a HookProvider registers a bundle of sync and async callbacks as one unit"
#[tokio::test]
async fn hook_provider_registers_bundled_callbacks() {
    let fired = Arc::new(Mutex::new(Vec::new()));
    let mut agent = Agent::builder()
        .model_boxed(Box::new(TextModel))
        .hook_provider(MeteringProvider {
            fired: fired.clone(),
        })
        .build();

    agent.invoke("hi").await.unwrap();

    // Both the sync before-model and the async after-model callback fired.
    assert_eq!(*fired.lock().unwrap(), vec!["before", "after"]);
}

// Post-build registration via `Agent::add_hook_provider`.
#[tokio::test]
async fn add_hook_provider_after_build() {
    let fired = Arc::new(Mutex::new(Vec::new()));
    let mut agent = Agent::builder().model_boxed(Box::new(TextModel)).build();
    agent.add_hook_provider(&MeteringProvider {
        fired: fired.clone(),
    });

    agent.invoke("hi").await.unwrap();
    assert_eq!(*fired.lock().unwrap(), vec!["before", "after"]);
}
