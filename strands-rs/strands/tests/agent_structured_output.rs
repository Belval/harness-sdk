//! Integration tests for structured output.
//!
//! A configured schema offers a `strands_structured_output` tool; the model
//! calling it captures the structured result, plain text forces the tool on the
//! next cycle, and a persistent refusal errors.

use std::sync::Mutex;

use async_trait::async_trait;
use futures::stream;
use serde_json::json;

use strands_agents::models::{Model, ModelEventStream, StreamOptions};
use strands_agents::types::messages::Role;
use strands_agents::types::streaming::{ContentBlockDelta, ModelStreamEvent, ToolUseStart};
use strands_agents::{Agent, Message, StopReason, StrandsError};

const STRUCTURED_TOOL: &str = "strands_structured_output";

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

    fn structured(id: &str, input: serde_json::Value) -> Self {
        Turn {
            events: vec![
                ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                },
                ModelStreamEvent::ContentBlockStart {
                    start: Some(ToolUseStart {
                        name: STRUCTURED_TOOL.to_string(),
                        tool_use_id: id.to_string(),
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

fn schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": { "answer": { "type": "integer" } },
        "required": ["answer"]
    })
}

// Model calls the structured tool directly -> captured structured output.
#[tokio::test]
async fn captures_structured_output_when_tool_called() {
    let model = ScriptedModel::new(vec![Turn::structured("s1", json!({ "answer": 42 }))]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .structured_output_schema(schema())
        .build();

    let result = agent.invoke("what is the answer?").await.unwrap();
    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert_eq!(result.structured_output, Some(json!({ "answer": 42 })));
}

// Plain text first forces the tool; the forced cycle's tool call is captured.
#[tokio::test]
async fn forces_structured_tool_after_plain_text() {
    let model = ScriptedModel::new(vec![
        Turn::text("here is my answer in prose"),
        Turn::structured("s1", json!({ "answer": 7 })),
    ]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .structured_output_schema(schema())
        .build();

    let result = agent.invoke("q").await.unwrap();
    assert_eq!(result.structured_output, Some(json!({ "answer": 7 })));
    // The plain-text turn was dropped; only the structured tool-use and its
    // success result are in history.
    let messages = agent.messages();
    assert!(messages.iter().all(|m| !m.text().contains("prose")));
}

// The model refuses the tool even when forced -> StructuredOutput error.
#[tokio::test]
async fn errors_when_model_refuses_forced_tool() {
    let model = ScriptedModel::new(vec![
        Turn::text("nope, prose again"),
        Turn::text("still prose"),
    ]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .structured_output_schema(schema())
        .build();

    let error = agent.invoke("q").await.unwrap_err();
    assert!(matches!(error, StrandsError::StructuredOutput(_)));
}
