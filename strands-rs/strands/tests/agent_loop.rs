//! Integration tests for the agent event loop.
//!
//! Ports the loop behavior specs from `agent/__tests__/agent.test.ts`
//! (`invoke > basic invocation` and `invoke > with tool use`). A scripted mock
//! model stands in for `MockMessageModel`: each configured turn emits one
//! content block and a stop reason.

use std::sync::Mutex;

use async_trait::async_trait;
use futures::stream;
use strands_agents::models::{Model, ModelEventStream, StreamOptions};
use strands_agents::tool;
use strands_agents::types::messages::Role;
use strands_agents::types::streaming::{ContentBlockDelta, ModelStreamEvent, ToolUseStart};
use strands_agents::{Agent, ContentBlock, Message, StopReason};

/// One scripted model turn: the events it emits when the loop calls the model.
struct Turn {
    events: Vec<ModelStreamEvent>,
}

impl Turn {
    /// A turn that returns a single text block and ends the turn.
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

    /// A turn that requests a tool call and stops with `toolUse`.
    fn tool_use(name: &str, tool_use_id: &str, input: serde_json::Value) -> Self {
        Turn {
            events: vec![
                ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                },
                ModelStreamEvent::ContentBlockStart {
                    start: Some(ToolUseStart {
                        name: name.to_string(),
                        tool_use_id: tool_use_id.to_string(),
                        reasoning_signature: None,
                    }),
                },
                ModelStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::ToolUseInput(input.to_string()),
                },
                ModelStreamEvent::ContentBlockStop,
                ModelStreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ],
        }
    }
}

/// A model that replays scripted turns in order, one per loop cycle.
struct MockMessageModel {
    turns: Mutex<std::collections::VecDeque<Turn>>,
    seen_tool_result: Mutex<bool>,
}

impl MockMessageModel {
    fn new(turns: Vec<Turn>) -> Self {
        MockMessageModel {
            turns: Mutex::new(turns.into_iter().collect()),
            seen_tool_result: Mutex::new(false),
        }
    }
}

#[async_trait]
impl Model for MockMessageModel {
    fn model_id(&self) -> Option<&str> {
        Some("mock-model")
    }

    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        _options: &'a StreamOptions,
    ) -> ModelEventStream<'a> {
        // Record whether the conversation carried a tool result back to the model,
        // which is what the loop must do after executing a tool.
        if messages.iter().any(|message| {
            message
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolResult(_)))
        }) {
            *self.seen_tool_result.lock().unwrap() = true;
        }
        let turn = self
            .turns
            .lock()
            .unwrap()
            .pop_front()
            .expect("model called more times than scripted");
        Box::pin(stream::iter(turn.events.into_iter().map(Ok)))
    }
}

/// Adds two integers.
#[tool]
async fn calc(a: i64, b: i64) -> i64 {
    a + b
}

// invoke > basic invocation: "returns correct stopReason and lastMessage"
#[tokio::test]
async fn basic_invocation_returns_stop_reason_and_last_message() {
    let model = MockMessageModel::new(vec![Turn::text("Hello there")]);
    let mut agent = Agent::builder().model_boxed(Box::new(model)).build();

    let result = agent.invoke("Hi").await.unwrap();

    assert_eq!(result.stop_reason, StopReason::EndTurn);
    assert_eq!(result.text(), "Hello there");
    // User prompt + assistant reply are both in history.
    assert_eq!(agent.messages().len(), 2);
    assert_eq!(agent.messages()[0].role, Role::User);
    assert_eq!(agent.messages()[1].role, Role::Assistant);
}

// invoke > with tool use: "executes tools and returns final result"
#[tokio::test]
async fn tool_use_executes_tool_and_continues() {
    let model = MockMessageModel::new(vec![
        Turn::tool_use("calc", "tool-1", serde_json::json!({ "a": 1, "b": 2 })),
        Turn::text("The answer is 3"),
    ]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(CalcTool::new())
        .build();

    let result = agent.invoke("What is 1 + 2?").await.unwrap();

    assert_eq!(result.stop_reason, StopReason::EndTurn);
    assert_eq!(result.text(), "The answer is 3");

    // History: user prompt, assistant tool-use, user tool-result, assistant reply.
    assert_eq!(agent.messages().len(), 4);
    assert_eq!(agent.messages()[1].role, Role::Assistant);
    assert!(matches!(
        agent.messages()[1].content[0],
        ContentBlock::ToolUse(_)
    ));
    assert_eq!(agent.messages()[2].role, Role::User);
    let ContentBlock::ToolResult(result_block) = &agent.messages()[2].content[0] else {
        panic!("expected a tool result block");
    };
    assert_eq!(result_block.tool_use_id, "tool-1");
    assert_eq!(
        result_block.content,
        vec![strands_agents::types::messages::ToolResultContent::Text(
            "3".to_string()
        )]
    );
}
