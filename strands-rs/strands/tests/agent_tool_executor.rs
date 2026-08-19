//! Integration tests for the pluggable `ToolExecutor` seam.
//!
//! Confirms the default `SequentialToolExecutor` runs tools end-to-end and that
//! a custom executor set via the builder replaces the tool phase.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream;
use serde_json::json;

use strands_agents::models::{Model, ModelEventStream, StreamOptions};
use strands_agents::tools::executor::{ToolExecutionContext, ToolExecutor, ToolsExecutionResult};
use strands_agents::tools::ToolContext;
use strands_agents::types::messages::{
    ContentBlock, Role, ToolResultBlock, ToolResultContent, ToolResultStatus,
};
use strands_agents::types::streaming::{ContentBlockDelta, ModelStreamEvent, ToolUseStart};
use strands_agents::{Agent, InvocationState, Message, StopReason, StrandsError, Tool, ToolSpec};

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

impl ScriptedModel {
    fn new(turns: Vec<Turn>) -> Self {
        ScriptedModel {
            turns: Mutex::new(turns.into_iter().collect()),
        }
    }
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

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "echoes"
    }
    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "echo".to_string(),
            description: "echoes".to_string(),
            input_schema: Some(json!({ "type": "object" })),
            output_schema: None,
        }
    }
    async fn invoke(&self, _context: ToolContext) -> Result<serde_json::Value, StrandsError> {
        Ok(json!("echoed"))
    }
}

fn tool_result_text(message: &Message) -> String {
    let ContentBlock::ToolResult(block) = &message.content[0] else {
        panic!("expected a tool result block");
    };
    match &block.content[0] {
        ToolResultContent::Text(text) => text.clone(),
        other => panic!("expected text, got {other:?}"),
    }
}

// The default SequentialToolExecutor runs a tool end-to-end.
#[tokio::test]
async fn default_executor_runs_tool() {
    let model = ScriptedModel::new(vec![Turn::tool_use("echo", "t1"), Turn::text("done")]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(EchoTool)
        .build();

    let result = agent.invoke("go").await.unwrap();
    assert_eq!(result.text(), "done");
    assert_eq!(tool_result_text(&agent.messages()[2]), "echoed");
}

/// A custom executor that records it ran and returns a canned tool-result,
/// bypassing real tool execution.
struct CannedExecutor {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ToolExecutor for CannedExecutor {
    async fn execute_tools(
        &self,
        _ctx: &ToolExecutionContext,
        assistant_message: &Message,
        _state: &InvocationState,
        _completed: Option<HashMap<String, ToolResultBlock>>,
    ) -> Result<ToolsExecutionResult, StrandsError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let result_blocks: Vec<ContentBlock> = assistant_message
            .content
            .iter()
            .filter_map(ContentBlock::as_tool_use)
            .map(|tool_use| {
                ContentBlock::ToolResult(ToolResultBlock {
                    tool_use_id: tool_use.tool_use_id.clone(),
                    status: ToolResultStatus::Success,
                    content: vec![ToolResultContent::Text("canned".to_string())],
                })
            })
            .collect();
        Ok(ToolsExecutionResult {
            message: Message::new(Role::User, result_blocks),
            end_turn: None,
        })
    }
}

// A custom ToolExecutor set via the builder replaces the tool phase.
#[tokio::test]
async fn custom_executor_replaces_tool_phase() {
    let calls = Arc::new(AtomicUsize::new(0));
    // The real tool would return "echoed"; the custom executor returns "canned".
    let model = ScriptedModel::new(vec![Turn::tool_use("echo", "t1"), Turn::text("done")]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(EchoTool)
        .tool_executor(CannedExecutor {
            calls: calls.clone(),
        })
        .build();

    agent.invoke("go").await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(tool_result_text(&agent.messages()[2]), "canned");
}
