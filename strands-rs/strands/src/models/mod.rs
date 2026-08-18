//! Model provider trait and implementations. Ports `models/model.ts`.

#[cfg(feature = "bedrock")]
pub mod bedrock;

use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};

use crate::errors::StrandsError;
use crate::types::messages::{
    ContentBlock, Message, MessageMetadata, ReasoningBlock, Role, StopReason, SystemPrompt,
    ToolUseBlock,
};
use crate::types::streaming::{ContentBlockDelta, Metrics, ModelStreamEvent, Usage};
use crate::types::tools::{ToolChoice, ToolSpec};

/// A boxed stream of model events, the Rust counterpart to the TypeScript
/// `AsyncIterable<ModelStreamEvent>` returned by `stream()`.
pub type ModelEventStream<'a> =
    Pin<Box<dyn Stream<Item = Result<ModelStreamEvent, StrandsError>> + Send + 'a>>;

/// How a provider should apply prompt caching. Ports `CacheConfig.strategy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    /// Enable caching only when the configured model is known to support it.
    Auto,
    /// Force Anthropic-style caching on (e.g. for inference-profile model ids
    /// whose provider cannot be auto-detected).
    Anthropic,
}

/// Options for a streaming model invocation. Ports `StreamOptions`.
#[derive(Debug, Clone, Default)]
pub struct StreamOptions {
    /// System prompt guiding the model's behavior.
    pub system_prompt: Option<SystemPrompt>,
    /// Tool specifications the model may use.
    pub tool_specs: Vec<ToolSpec>,
    /// Controls how the model selects tools.
    pub tool_choice: Option<ToolChoice>,
}

/// The aggregated result of a model stream. Ports `StreamAggregatedResult`.
#[derive(Debug, Clone)]
pub struct StreamAggregatedResult {
    /// The complete assistant message.
    pub message: Message,
    /// Why the model stopped generating.
    pub stop_reason: StopReason,
    /// Token usage, when the provider reported it.
    pub usage: Option<Usage>,
    /// Performance metrics, when the provider reported them.
    pub metrics: Option<Metrics>,
}

/// Contract that all model provider implementations must follow.
///
/// Ports the abstract `Model` class. `stream` is the single required method a
/// provider implements; [`Model::stream_aggregated`] is a provided method that
/// accumulates raw events into a complete message, mirroring the base class's
/// `streamAggregated`.
#[async_trait]
pub trait Model: Send + Sync {
    /// The configured model identifier, if any.
    fn model_id(&self) -> Option<&str>;

    /// Streams a conversation with the model, yielding events as they occur.
    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        options: &'a StreamOptions,
    ) -> ModelEventStream<'a>;

    /// Streams a conversation and accumulates the raw events into a complete
    /// message with a stop reason and optional metadata.
    ///
    /// Ports `streamAggregated`: it consumes [`Model::stream`], folds text /
    /// tool-input / reasoning deltas into content blocks on each
    /// `ContentBlockStop`, and assembles the final message on `MessageStop`.
    async fn stream_aggregated(
        &self,
        messages: &[Message],
        options: &StreamOptions,
    ) -> Result<StreamAggregatedResult, StrandsError> {
        let mut stream = self.stream(messages, options);

        let mut message_role: Option<Role> = None;
        let mut content_blocks: Vec<ContentBlock> = Vec::new();
        let mut accumulated_text = String::new();
        let mut accumulated_tool_input = String::new();
        let mut tool_name = String::new();
        let mut tool_use_id = String::new();
        let mut tool_reasoning_signature = String::new();
        let mut accumulated_reasoning = ReasoningBlock::default();
        let mut has_reasoning = false;
        let mut final_stop_reason: Option<StopReason> = None;
        let mut usage: Option<Usage> = None;
        let mut metrics: Option<Metrics> = None;
        // A malformed tool-input JSON parse is deferred rather than thrown
        // immediately: the TypeScript `streamAggregated` stores the error, lets
        // the loop finish, and gives the `maxTokens` check precedence over it.
        let mut tool_input_parse_error: Option<serde_json::Error> = None;

        while let Some(event) = stream.next().await {
            let event = event?;
            match event {
                ModelStreamEvent::MessageStart { role } => {
                    message_role = Some(role);
                    content_blocks.clear();
                }
                ModelStreamEvent::ContentBlockStart { start } => {
                    if let Some(start) = start {
                        tool_name = start.name;
                        tool_use_id = start.tool_use_id;
                        tool_reasoning_signature = start.reasoning_signature.unwrap_or_default();
                    }
                    accumulated_tool_input.clear();
                    accumulated_text.clear();
                    accumulated_reasoning = ReasoningBlock::default();
                    has_reasoning = false;
                }
                ModelStreamEvent::ContentBlockDelta { delta } => match delta {
                    ContentBlockDelta::Text(text) => accumulated_text.push_str(&text),
                    ContentBlockDelta::ToolUseInput(input) => {
                        accumulated_tool_input.push_str(&input)
                    }
                    ContentBlockDelta::Reasoning {
                        text,
                        signature,
                        redacted_content,
                    } => {
                        has_reasoning = true;
                        if let Some(text) = text {
                            accumulated_reasoning.text =
                                Some(accumulated_reasoning.text.take().unwrap_or_default() + &text);
                        }
                        if let Some(signature) = signature {
                            accumulated_reasoning.signature = Some(signature);
                        }
                        if let Some(redacted) = redacted_content {
                            accumulated_reasoning.redacted_content = Some(redacted);
                        }
                    }
                },
                ModelStreamEvent::ContentBlockStop => {
                    let block = if !tool_use_id.is_empty() {
                        let input = if accumulated_tool_input.is_empty() {
                            serde_json::json!({})
                        } else {
                            match serde_json::from_str(&accumulated_tool_input) {
                                Ok(input) => input,
                                Err(error) => {
                                    // Defer: store the error, skip this block, and let the
                                    // maxTokens check below take precedence, as TS does.
                                    tracing::error!(error = %error, "unable to parse tool input JSON");
                                    tool_input_parse_error = Some(error);
                                    continue;
                                }
                            }
                        };
                        let reasoning_signature = if tool_reasoning_signature.is_empty() {
                            None
                        } else {
                            Some(std::mem::take(&mut tool_reasoning_signature))
                        };
                        ContentBlock::ToolUse(ToolUseBlock {
                            name: std::mem::take(&mut tool_name),
                            tool_use_id: std::mem::take(&mut tool_use_id),
                            input,
                            reasoning_signature,
                        })
                    } else if has_reasoning {
                        ContentBlock::Reasoning(std::mem::take(&mut accumulated_reasoning))
                    } else {
                        ContentBlock::Text(std::mem::take(&mut accumulated_text))
                    };
                    content_blocks.push(block);
                }
                ModelStreamEvent::MessageStop { stop_reason } => {
                    final_stop_reason = Some(stop_reason);
                }
                ModelStreamEvent::Metadata {
                    usage: event_usage,
                    metrics: event_metrics,
                } => {
                    usage = event_usage;
                    metrics = event_metrics;
                }
            }
        }

        // Stream ended without a completed message/stop reason. A deferred tool-input
        // parse error becomes the cause, matching the TypeScript SDK.
        let (Some(role), Some(stop_reason)) = (message_role, final_stop_reason) else {
            return Err(match tool_input_parse_error {
                Some(error) => StrandsError::model_with_source(
                    "Stream ended without completing a message",
                    error,
                ),
                None => StrandsError::model("Stream ended without completing a message"),
            });
        };

        // Drop empty text blocks, matching the TypeScript filter that strips
        // whitespace-only text blocks before building the message.
        content_blocks
            .retain(|block| !matches!(block, ContentBlock::Text(text) if text.trim().is_empty()));

        let mut message = Message::new(role, content_blocks);
        if usage.is_some() || metrics.is_some() {
            message.metadata = Some(MessageMetadata {
                usage: usage.clone(),
                metrics: metrics.clone(),
            });
        }

        // maxTokens takes precedence over a deferred parse error, as in TypeScript.
        if stop_reason == StopReason::MaxTokens {
            return Err(StrandsError::MaxTokens {
                message: "Model reached maximum token limit. This is an unrecoverable state that requires intervention."
                    .to_string(),
                partial_message: Box::new(message),
            });
        }

        if let Some(error) = tool_input_parse_error {
            return Err(StrandsError::model_with_source(
                "unable to parse tool input JSON",
                error,
            ));
        }

        Ok(StreamAggregatedResult {
            message,
            stop_reason,
            usage,
            metrics,
        })
    }
}

#[cfg(feature = "bedrock")]
pub use bedrock::{BedrockCacheConfig, BedrockModel};

#[cfg(test)]
mod tests {
    //! Ports the `streamAggregated` behavior specs from
    //! `models/__tests__/model.test.ts`. Rust folds the raw events into a final
    //! message rather than yielding intermediate blocks, so these assert on the
    //! aggregated result (the TypeScript tests also assert the returned result).

    use super::*;
    use crate::types::messages::{ContentBlock, Role};
    use crate::types::streaming::ToolUseStart;
    use futures::stream;

    /// A model provider that replays a fixed script of events, the Rust
    /// counterpart to the TypeScript `TestModelProvider`.
    struct TestModelProvider {
        events: Vec<ModelStreamEvent>,
    }

    #[async_trait]
    impl Model for TestModelProvider {
        fn model_id(&self) -> Option<&str> {
            Some("test-model")
        }

        fn stream<'a>(
            &'a self,
            _messages: &'a [Message],
            _options: &'a StreamOptions,
        ) -> ModelEventStream<'a> {
            let events = self.events.clone();
            Box::pin(stream::iter(events.into_iter().map(Ok)))
        }
    }

    fn text_delta(text: &str) -> ModelStreamEvent {
        ModelStreamEvent::ContentBlockDelta {
            delta: ContentBlockDelta::Text(text.to_string()),
        }
    }

    // streamAggregated > when streaming a simple text message:
    // "yields ... aggregated content block and returns final message"
    #[tokio::test]
    async fn aggregates_simple_text_message() {
        let provider = TestModelProvider {
            events: vec![
                ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                },
                ModelStreamEvent::ContentBlockStart { start: None },
                text_delta("Hello"),
                ModelStreamEvent::ContentBlockStop,
                ModelStreamEvent::MessageStop {
                    stop_reason: StopReason::EndTurn,
                },
                ModelStreamEvent::Metadata {
                    usage: Some(Usage {
                        input_tokens: 10,
                        output_tokens: 5,
                        total_tokens: 15,
                        ..Usage::default()
                    }),
                    metrics: None,
                },
            ],
        };

        let result = provider
            .stream_aggregated(&[], &StreamOptions::default())
            .await
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::EndTurn);
        assert_eq!(result.message.role, Role::Assistant);
        assert_eq!(
            result.message.content,
            vec![ContentBlock::Text("Hello".to_string())]
        );
        assert_eq!(result.usage.as_ref().unwrap().total_tokens, 15);
        assert_eq!(
            result
                .message
                .metadata
                .as_ref()
                .unwrap()
                .usage
                .as_ref()
                .unwrap()
                .input_tokens,
            10
        );
    }

    // streamAggregated > when streaming a simple text message:
    // "throws MaxTokenError when stopReason is MaxTokenError"
    #[tokio::test]
    async fn throws_max_tokens_error() {
        let provider = TestModelProvider {
            events: vec![
                ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                },
                ModelStreamEvent::ContentBlockStart { start: None },
                text_delta("Hello"),
                ModelStreamEvent::ContentBlockStop,
                ModelStreamEvent::MessageStop {
                    stop_reason: StopReason::MaxTokens,
                },
            ],
        };

        let error = provider
            .stream_aggregated(&[], &StreamOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(error, StrandsError::MaxTokens { .. }));
        assert!(error
            .to_string()
            .contains("Model reached maximum token limit. This is an unrecoverable state that requires intervention."));
    }

    // streamAggregated > when streaming multiple text blocks: "yields all blocks in order"
    #[tokio::test]
    async fn aggregates_multiple_text_blocks_in_order() {
        let provider = TestModelProvider {
            events: vec![
                ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                },
                ModelStreamEvent::ContentBlockStart { start: None },
                text_delta("First"),
                ModelStreamEvent::ContentBlockStop,
                ModelStreamEvent::ContentBlockStart { start: None },
                text_delta("Second"),
                ModelStreamEvent::ContentBlockStop,
                ModelStreamEvent::MessageStop {
                    stop_reason: StopReason::EndTurn,
                },
            ],
        };

        let result = provider
            .stream_aggregated(&[], &StreamOptions::default())
            .await
            .unwrap();
        assert_eq!(
            result.message.content,
            vec![
                ContentBlock::Text("First".to_string()),
                ContentBlock::Text("Second".to_string())
            ]
        );
    }

    // streamAggregated > when streaming tool use: "yields complete tool use block"
    #[tokio::test]
    async fn aggregates_tool_use_block() {
        let provider = TestModelProvider {
            events: vec![
                ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                },
                ModelStreamEvent::ContentBlockStart {
                    start: Some(ToolUseStart {
                        name: "get_weather".to_string(),
                        tool_use_id: "tool1".to_string(),
                        reasoning_signature: None,
                    }),
                },
                ModelStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::ToolUseInput("{\"location\"".to_string()),
                },
                ModelStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::ToolUseInput(": \"Paris\"}".to_string()),
                },
                ModelStreamEvent::ContentBlockStop,
                ModelStreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ],
        };

        let result = provider
            .stream_aggregated(&[], &StreamOptions::default())
            .await
            .unwrap();
        assert_eq!(result.stop_reason, StopReason::ToolUse);
        let ContentBlock::ToolUse(tool_use) = &result.message.content[0] else {
            panic!("expected a tool use block");
        };
        assert_eq!(tool_use.name, "get_weather");
        assert_eq!(tool_use.tool_use_id, "tool1");
        assert_eq!(tool_use.input, serde_json::json!({ "location": "Paris" }));
    }

    // streamAggregated > when streaming tool use: "yields complete tool use block with empty input"
    #[tokio::test]
    async fn aggregates_tool_use_block_with_empty_input() {
        let provider = TestModelProvider {
            events: vec![
                ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                },
                ModelStreamEvent::ContentBlockStart {
                    start: Some(ToolUseStart {
                        name: "get_time".to_string(),
                        tool_use_id: "tool1".to_string(),
                        reasoning_signature: None,
                    }),
                },
                ModelStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::ToolUseInput(String::new()),
                },
                ModelStreamEvent::ContentBlockStop,
                ModelStreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ],
        };

        let result = provider
            .stream_aggregated(&[], &StreamOptions::default())
            .await
            .unwrap();
        let ContentBlock::ToolUse(tool_use) = &result.message.content[0] else {
            panic!("expected a tool use block");
        };
        assert_eq!(tool_use.input, serde_json::json!({}));
    }

    // streamAggregated > when a content block emits no text deltas ...:
    // "drops the resulting empty TextBlock from the aggregated message"
    #[tokio::test]
    async fn drops_empty_trailing_text_block() {
        let provider = TestModelProvider {
            events: vec![
                ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                },
                ModelStreamEvent::ContentBlockStart {
                    start: Some(ToolUseStart {
                        name: "get_time".to_string(),
                        tool_use_id: "tool1".to_string(),
                        reasoning_signature: None,
                    }),
                },
                ModelStreamEvent::ContentBlockStop,
                // A second block with no deltas would aggregate to an empty text block.
                ModelStreamEvent::ContentBlockStart { start: None },
                ModelStreamEvent::ContentBlockStop,
                ModelStreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ],
        };

        let result = provider
            .stream_aggregated(&[], &StreamOptions::default())
            .await
            .unwrap();
        // Only the tool-use block survives; the empty text block is dropped.
        assert_eq!(result.message.content.len(), 1);
        assert!(matches!(
            result.message.content[0],
            ContentBlock::ToolUse(_)
        ));
    }

    // streamAggregated > when streaming tool use:
    // "throws MaxTokensError when contentBlockStop arrives with truncated tool input JSON
    //  and stopReason is maxTokens" — maxTokens takes precedence over the parse error.
    #[tokio::test]
    async fn max_tokens_takes_precedence_over_truncated_tool_input() {
        let provider = TestModelProvider {
            events: vec![
                ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                },
                ModelStreamEvent::ContentBlockStart {
                    start: Some(ToolUseStart {
                        name: "get_weather".to_string(),
                        tool_use_id: "tool1".to_string(),
                        reasoning_signature: None,
                    }),
                },
                ModelStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::ToolUseInput("{\"location\"".to_string()),
                },
                ModelStreamEvent::ContentBlockStop,
                ModelStreamEvent::MessageStop {
                    stop_reason: StopReason::MaxTokens,
                },
            ],
        };

        let error = provider
            .stream_aggregated(&[], &StreamOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(error, StrandsError::MaxTokens { .. }));
    }

    // streamAggregated > when streaming tool use:
    // "surfaces the parse error as ModelError when tool input JSON is malformed and
    //  stopReason is not maxTokens"
    #[tokio::test]
    async fn malformed_tool_input_without_max_tokens_surfaces_model_error() {
        let provider = TestModelProvider {
            events: vec![
                ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                },
                ModelStreamEvent::ContentBlockStart {
                    start: Some(ToolUseStart {
                        name: "get_weather".to_string(),
                        tool_use_id: "tool1".to_string(),
                        reasoning_signature: None,
                    }),
                },
                ModelStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::ToolUseInput("{\"location\"".to_string()),
                },
                ModelStreamEvent::ContentBlockStop,
                ModelStreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ],
        };

        let error = provider
            .stream_aggregated(&[], &StreamOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(error, StrandsError::Model { .. }));
        assert!(error
            .to_string()
            .contains("unable to parse tool input JSON"));
    }

    // Model.modelId: "returns modelId from model config"
    #[tokio::test]
    async fn returns_model_id() {
        let provider = TestModelProvider { events: vec![] };
        assert_eq!(provider.model_id(), Some("test-model"));
    }
}
