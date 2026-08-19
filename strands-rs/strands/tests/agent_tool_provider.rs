//! Integration tests for runtime tool construction and tool providers.
//!
//! Covers a `ToolProvider` whose `load_tools` builds a tool at runtime via
//! `FunctionTool::from_spec`, and the `#[tool(name = "...")]` override.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream;
use serde_json::json;

use strands_agents::models::{Model, ModelEventStream, StreamOptions};
use strands_agents::tools::{FunctionTool, ToolProvider};
use strands_agents::types::messages::Role;
use strands_agents::types::streaming::{ContentBlockDelta, ModelStreamEvent, ToolUseStart};
use strands_agents::{
    tool, Agent, ContentBlock, Message, StopReason, StrandsError, Tool, ToolSpec,
};

// --- Scripted model ---------------------------------------------------------

struct Turn {
    events: Vec<ModelStreamEvent>,
}

impl Turn {
    fn text(text: &str) -> Self {
        Turn {
            events: vec![
                ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                },
                ModelStreamEvent::ContentBlockStart { start: None },
                ModelStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::Text(text.to_string()),
                },
                ModelStreamEvent::ContentBlockStop,
                ModelStreamEvent::MessageStop {
                    stop_reason: StopReason::EndTurn,
                },
            ],
        }
    }

    fn tool_use(name: &str, id: &str) -> Self {
        Turn {
            events: vec![
                ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                },
                ModelStreamEvent::ContentBlockStart {
                    start: Some(ToolUseStart {
                        name: name.to_string(),
                        tool_use_id: id.to_string(),
                        reasoning_signature: None,
                    }),
                },
                ModelStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::ToolUseInput("{}".to_string()),
                },
                ModelStreamEvent::ContentBlockStop,
                ModelStreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ],
        }
    }
}

struct ScriptedModel {
    turns: Mutex<std::collections::VecDeque<Turn>>,
}

#[async_trait]
impl Model for ScriptedModel {
    fn model_id(&self) -> Option<&str> {
        Some("scripted")
    }
    fn stream<'a>(
        &'a self,
        _messages: &'a [Message],
        _options: &'a StreamOptions,
    ) -> ModelEventStream<'a> {
        let turn = self
            .turns
            .lock()
            .unwrap()
            .pop_front()
            .expect("model over-called");
        Box::pin(stream::iter(turn.events.into_iter().map(Ok)))
    }
}

// --- Tool provider ----------------------------------------------------------

/// Builds a runtime tool that flips a flag when the agent calls it.
struct FlagProvider {
    ran: Arc<AtomicBool>,
}

#[async_trait]
impl ToolProvider for FlagProvider {
    async fn load_tools(&self) -> Result<Vec<Arc<dyn Tool>>, StrandsError> {
        let ran = self.ran.clone();
        let spec = ToolSpec {
            name: "provided_tool".to_string(),
            description: "a runtime-built tool".to_string(),
            input_schema: Some(json!({ "type": "object", "properties": {} })),
            output_schema: None,
        };
        let tool = FunctionTool::from_spec(spec, move |_context| {
            let ran = ran.clone();
            async move {
                ran.store(true, Ordering::SeqCst);
                Ok(json!("provider-tool-ran"))
            }
        });
        Ok(vec![Arc::new(tool)])
    }
}

// A tool with a name override.
/// Sends a message somewhere.
#[tool(name = "send_message")]
async fn my_sender(text: String) -> String {
    format!("sent: {text}")
}

// --- Tests ------------------------------------------------------------------

// "a ToolProvider's runtime-built tool is loaded lazily and callable"
#[tokio::test]
async fn tool_provider_loads_runtime_tool() {
    let ran = Arc::new(AtomicBool::new(false));
    let model = ScriptedModel {
        turns: Mutex::new(
            vec![Turn::tool_use("provided_tool", "p1"), Turn::text("done")]
                .into_iter()
                .collect(),
        ),
    };
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool_provider(FlagProvider { ran: ran.clone() })
        .build();

    // Not loaded until the first invocation.
    assert!(agent
        .tools()
        .iter()
        .all(|tool| tool.name() != "provided_tool"));

    let result = agent.invoke("go").await.unwrap();

    assert_eq!(result.text(), "done");
    assert!(
        ran.load(Ordering::SeqCst),
        "the provider tool should have executed"
    );
    // The provider's tool is now registered, and its result is in history.
    assert!(agent
        .tools()
        .iter()
        .any(|tool| tool.name() == "provided_tool"));
    let has_tool_result = agent.messages().iter().any(|message| {
        message
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolResult(_)))
    });
    assert!(has_tool_result);
}

// "#[tool(name = ...)] overrides the derived tool name"
#[test]
fn tool_macro_name_override() {
    let tool = MySenderTool::new();
    assert_eq!(tool.name(), "send_message");
    assert_eq!(tool.tool_spec().name, "send_message");
    // The description still comes from the doc comment.
    assert_eq!(tool.description(), "Sends a message somewhere.");
}
