//! Integration tests for agent middleware.
//!
//! Ports the loop-integration specs from `middleware/__tests__/agent-middleware.test.ts`:
//! middleware registered on the `InvokeModelStage` and `ExecuteToolStage` wraps,
//! transforms, and short-circuits the model call and tool execution.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use futures::stream;
use serde_json::json;

use strands_agents::models::{Model, ModelEventStream, StreamAggregatedResult, StreamOptions};
use strands_agents::tools::ToolContext;
use strands_agents::types::messages::{Role, ToolResultBlock, ToolResultContent, ToolResultStatus};
use strands_agents::types::streaming::{ContentBlockDelta, ModelStreamEvent, ToolUseStart};
use strands_agents::{
    Agent, ContentBlock, Message, MiddlewareNext, StopReason, StrandsError, Tool, ToolSpec,
};

// --- Models -----------------------------------------------------------------

/// Replays scripted single-turn responses; panics if called more than scripted.
struct ScriptedModel {
    turns: Mutex<std::collections::VecDeque<Vec<ModelStreamEvent>>>,
    calls: std::sync::Arc<AtomicUsize>,
}

impl ScriptedModel {
    fn new(turns: Vec<Vec<ModelStreamEvent>>) -> (Self, std::sync::Arc<AtomicUsize>) {
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        (
            ScriptedModel {
                turns: Mutex::new(turns.into_iter().collect()),
                calls: calls.clone(),
            },
            calls,
        )
    }
}

fn text_turn(text: &str) -> Vec<ModelStreamEvent> {
    vec![
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
    ]
}

fn tool_turn(name: &str, id: &str) -> Vec<ModelStreamEvent> {
    vec![
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
    ]
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        let turn = self
            .turns
            .lock()
            .unwrap()
            .pop_front()
            .expect("model over-called");
        Box::pin(stream::iter(turn.into_iter().map(Ok)))
    }
}

/// Returns the number of messages it was given as its text response, so tests
/// can observe input-middleware transformations of the conversation.
struct MessageCountModel;

#[async_trait]
impl Model for MessageCountModel {
    fn model_id(&self) -> Option<&str> {
        Some("counter")
    }
    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        _options: &'a StreamOptions,
    ) -> ModelEventStream<'a> {
        let turn = text_turn(&messages.len().to_string());
        Box::pin(stream::iter(turn.into_iter().map(Ok)))
    }
}

struct ConstTool;

#[async_trait]
impl Tool for ConstTool {
    fn name(&self) -> &str {
        "constant"
    }
    fn description(&self) -> &str {
        "returns 42"
    }
    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "constant".to_string(),
            description: "returns 42".to_string(),
            input_schema: Some(json!({ "type": "object" })),
            output_schema: None,
        }
    }
    async fn invoke(&self, _context: ToolContext) -> Result<serde_json::Value, StrandsError> {
        Ok(json!(42))
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

// --- Tests ------------------------------------------------------------------

// "a wrap handler that skips next short-circuits the model call"
#[tokio::test]
async fn invoke_model_wrap_short_circuits_the_model() {
    // No turns scripted: if the model were called it would panic.
    let (model, calls) = ScriptedModel::new(vec![]);
    let mut agent = Agent::builder().model_boxed(Box::new(model)).build();

    agent
        .invoke_model_middleware()
        .add(|_context, _next: MiddlewareNext<_, _>| async move {
            Ok(StreamAggregatedResult {
                message: Message::assistant("short-circuited"),
                stop_reason: StopReason::EndTurn,
                usage: None,
                metrics: None,
            })
        });

    let result = agent.invoke("hi").await.unwrap();
    assert_eq!(result.text(), "short-circuited");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

// "an output handler transforms the model result"
#[tokio::test]
async fn invoke_model_output_transforms_result() {
    let (model, _calls) = ScriptedModel::new(vec![text_turn("original")]);
    let mut agent = Agent::builder().model_boxed(Box::new(model)).build();

    agent
        .invoke_model_middleware()
        .add_output(|mut result: StreamAggregatedResult| async move {
            result.message = Message::assistant("wrapped");
            Ok(result)
        });

    let result = agent.invoke("hi").await.unwrap();
    assert_eq!(result.text(), "wrapped");
}

// "an input handler transforms the context before the model runs"
#[tokio::test]
async fn invoke_model_input_transforms_context() {
    let mut agent = Agent::builder()
        .model_boxed(Box::new(MessageCountModel))
        .build();

    // Inject an extra message so the model sees two instead of one.
    agent.invoke_model_middleware().add_input(
        |mut context: strands_agents::InvokeModelContext| async move {
            context.messages.push(Message::user("injected"));
            Ok(context)
        },
    );

    let result = agent.invoke("hi").await.unwrap();
    // The user prompt plus the injected message.
    assert_eq!(result.text(), "2");
}

// "middleware on ExecuteToolStage transforms the tool result"
#[tokio::test]
async fn execute_tool_output_transforms_result() {
    let (model, _calls) = ScriptedModel::new(vec![tool_turn("constant", "t1"), text_turn("done")]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(ConstTool)
        .build();

    agent.execute_tool_middleware().add_output(
        |mut execution: strands_agents::ToolExecutionResult| async move {
            execution.result = ToolResultBlock {
                tool_use_id: execution.result.tool_use_id.clone(),
                status: ToolResultStatus::Success,
                content: vec![ToolResultContent::Text("intercepted".to_string())],
            };
            Ok(execution)
        },
    );

    agent.invoke("go").await.unwrap();
    // The tool-result message (index 2) carries the intercepted content.
    assert_eq!(tool_result_text(&agent.messages[2]), "intercepted");
}
