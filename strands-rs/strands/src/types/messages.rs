//! Message types and content blocks for conversational AI interactions.
//!
//! Ports `types/messages.ts`. The TypeScript module pairs a `Data` interface
//! (wire shape) with a class (behavior + discriminator). Rust collapses each
//! pair into a single type: serde derives supply the wire shape, and the
//! discriminated union `ContentBlockData` becomes a Rust `enum`. The object-key
//! discriminator (`{ toolUse: ... }`) maps to serde's internally-untagged
//! external tagging via `#[serde(rename_all = "camelCase")]` on the enum, so the
//! JSON key selects the variant exactly as it does in TypeScript.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::media::{DocumentBlock, ImageBlock, VideoBlock};
use crate::types::streaming::{Metrics, Usage};

/// Role of a message in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Human input.
    User,
    /// Model response.
    Assistant,
}

impl Role {
    /// Returns the lowercase string representation (`"user"` / `"assistant"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Status of a tool execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolResultStatus {
    /// The tool executed successfully.
    Success,
    /// The tool encountered an error.
    Error,
}

/// A tool-use request emitted by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolUseBlock {
    /// The name of the tool to execute.
    pub name: String,
    /// Unique identifier for this tool-use instance.
    pub tool_use_id: String,
    /// The input parameters for the tool.
    pub input: serde_json::Value,
    /// Reasoning signature from thinking models, preserved for multi-turn tool use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_signature: Option<String>,
}

/// Content within a tool result: text, structured JSON, or a media block.
///
/// Mirrors the `ToolResultContentData` discriminated union. The `Text` variant
/// carries the bare `{ "text": ... }` shape rather than a `TextBlock` wrapper to
/// match the TypeScript wire form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ToolResultContent {
    /// Plain text.
    Text(String),
    /// Structured JSON data.
    Json(serde_json::Value),
    /// An image.
    Image(ImageBlock),
    /// A video.
    Video(VideoBlock),
    /// A document.
    Document(DocumentBlock),
}

/// The result of a tool execution, returned to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultBlock {
    /// The ID of the tool-use request this result corresponds to.
    pub tool_use_id: String,
    /// Whether the tool succeeded or errored.
    pub status: ToolResultStatus,
    /// The content returned by the tool.
    pub content: Vec<ToolResultContent>,
}

/// Reasoning (thinking) content produced by the model.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningBlock {
    /// The reasoning text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// A cryptographic signature for verification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Redacted reasoning content, base64-encoded on the wire.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "opt_base64_bytes"
    )]
    pub redacted_content: Option<Vec<u8>>,
}

/// Marks a position where prompt caching should occur.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CachePointBlock {
    /// The cache type. Currently only `"default"` is supported.
    pub cache_type: String,
    /// Optional provider-specific TTL for the cache entry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

/// A block of content within a message.
///
/// Discriminated union mirroring `ContentBlockData`. Serde external tagging
/// selects the variant by JSON key (`text`, `toolUse`, `toolResult`, …), which
/// is exactly how the TypeScript union discriminates via `'toolUse' in block`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ContentBlock {
    /// Plain text content.
    Text(String),
    /// A tool-use request from the model.
    ToolUse(ToolUseBlock),
    /// A tool execution result.
    ToolResult(ToolResultBlock),
    /// Reasoning/thinking content.
    Reasoning(ReasoningBlock),
    /// A prompt-cache point.
    CachePoint(CachePointBlock),
    /// An image.
    Image(ImageBlock),
    /// A video.
    Video(VideoBlock),
    /// A document.
    Document(DocumentBlock),
}

impl ContentBlock {
    /// Creates a text content block.
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock::Text(text.into())
    }

    /// Returns the text of a `Text` block, or `None` for any other variant.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text(text) => Some(text),
            _ => None,
        }
    }

    /// Returns the tool-use request of a `ToolUse` block, or `None` otherwise.
    pub fn as_tool_use(&self) -> Option<&ToolUseBlock> {
        match self {
            ContentBlock::ToolUse(tool_use) => Some(tool_use),
            _ => None,
        }
    }
}

/// Optional metadata attached to a message. Not sent to model providers.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageMetadata {
    /// Token usage from the model response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Performance metrics from the model response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Metrics>,
}

/// A message in a conversation between user and assistant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    /// The role of the message sender.
    pub role: Role,
    /// The content blocks that make up this message.
    pub content: Vec<ContentBlock>,
    /// Durable, stable UUID for the message, minted at construction when absent.
    pub tracking_id: String,
    /// Optional metadata, not sent to model providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<MessageMetadata>,
}

impl Message {
    /// Creates a message with the given role and content, minting a fresh
    /// tracking id. Mirrors the TypeScript constructor, which backfills a UUID
    /// when the caller does not supply one.
    pub fn new(role: Role, content: Vec<ContentBlock>) -> Self {
        Message {
            role,
            content,
            tracking_id: generate_tracking_id(),
            metadata: None,
        }
    }

    /// Creates a user message containing a single text block.
    pub fn user(text: impl Into<String>) -> Self {
        Message::new(Role::User, vec![ContentBlock::text(text)])
    }

    /// Creates an assistant message containing a single text block.
    pub fn assistant(text: impl Into<String>) -> Self {
        Message::new(Role::Assistant, vec![ContentBlock::text(text)])
    }

    /// Returns the concatenated text of all `Text` content blocks.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(ContentBlock::as_text)
            .collect::<Vec<_>>()
            .join("")
    }
}

/// Generates a durable tracking identifier for a message.
///
/// Uses UUID v4, matching the TypeScript SDK's `crypto.randomUUID()`.
pub fn generate_tracking_id() -> String {
    Uuid::new_v4().to_string()
}

/// System prompt for guiding model behavior.
///
/// The TypeScript SDK accepts either a string or an array of content blocks; the
/// vertical slice supports the string form, which covers the common case.
pub type SystemPrompt = String;

/// Reason why the model stopped generating content.
///
/// Mirrors the TypeScript open string union `StopReason`: known values are
/// enumerated, and `Other` preserves any unrecognized value the provider emits
/// (the counterpart to the union's `(string & {})` escape hatch).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Agent invocation was cancelled.
    Cancelled,
    /// Content was filtered by safety mechanisms.
    ContentFiltered,
    /// Natural end of the model's turn.
    EndTurn,
    /// A guardrail policy stopped generation.
    GuardrailIntervened,
    /// The model provider's per-call token cap was reached.
    MaxTokens,
    /// A stop sequence was encountered.
    StopSequence,
    /// The model wants to use a tool.
    ToolUse,
    /// Input exceeded the model's context window.
    ModelContextWindowExceeded,
    /// Agent halted to wait for human input on a raised interrupt.
    Interrupt,
    /// Any value not covered above, preserved verbatim.
    Other(String),
}

impl StopReason {
    /// Returns the camelCase wire string for this stop reason.
    pub fn as_str(&self) -> &str {
        match self {
            StopReason::Cancelled => "cancelled",
            StopReason::ContentFiltered => "contentFiltered",
            StopReason::EndTurn => "endTurn",
            StopReason::GuardrailIntervened => "guardrailIntervened",
            StopReason::MaxTokens => "maxTokens",
            StopReason::StopSequence => "stopSequence",
            StopReason::ToolUse => "toolUse",
            StopReason::ModelContextWindowExceeded => "modelContextWindowExceeded",
            StopReason::Interrupt => "interrupt",
            StopReason::Other(value) => value,
        }
    }

    /// Parses a wire string into a `StopReason`, preserving unknown values in `Other`.
    pub fn from_wire(value: &str) -> Self {
        match value {
            "cancelled" => StopReason::Cancelled,
            "contentFiltered" => StopReason::ContentFiltered,
            "endTurn" => StopReason::EndTurn,
            "guardrailIntervened" => StopReason::GuardrailIntervened,
            "maxTokens" => StopReason::MaxTokens,
            "stopSequence" => StopReason::StopSequence,
            "toolUse" => StopReason::ToolUse,
            "modelContextWindowExceeded" => StopReason::ModelContextWindowExceeded,
            "interrupt" => StopReason::Interrupt,
            other => StopReason::Other(other.to_string()),
        }
    }
}

impl Serialize for StopReason {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for StopReason {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Ok(StopReason::from_wire(&value))
    }
}

#[cfg(test)]
mod tests {
    //! Ports the message/content-block round-trip and construction specs from
    //! `types/__tests__/messages.test.ts`. The TypeScript tests assert on the
    //! `Data`/class serialization pair; here they assert on serde round-trips,
    //! which cover the same wire shapes.

    use super::*;

    // Message id: "every Message has a trackingId" (constructor backfills a UUID)
    #[test]
    fn message_gets_a_tracking_id() {
        let message = Message::user("Hi");
        assert!(!message.tracking_id.is_empty());
        assert!(uuid::Uuid::parse_str(&message.tracking_id).is_ok());
    }

    // TextBlock: serializes to `{ "text": ... }`
    #[test]
    fn text_block_serializes_with_text_key() {
        let block = ContentBlock::text("Hello");
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json, serde_json::json!({ "text": "Hello" }));
    }

    // ToolUseBlock: serializes under the `toolUse` key with camelCase fields
    #[test]
    fn tool_use_block_serializes_under_tool_use_key() {
        let block = ContentBlock::ToolUse(ToolUseBlock {
            name: "get_weather".to_string(),
            tool_use_id: "tool-1".to_string(),
            input: serde_json::json!({ "location": "Paris" }),
            reasoning_signature: None,
        });
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "toolUse": { "name": "get_weather", "toolUseId": "tool-1", "input": { "location": "Paris" } }
            })
        );
    }

    // ToolResultBlock: serializes under `toolResult` with status and content
    #[test]
    fn tool_result_block_serializes_under_tool_result_key() {
        let block = ContentBlock::ToolResult(ToolResultBlock {
            tool_use_id: "tool-1".to_string(),
            status: ToolResultStatus::Success,
            content: vec![ToolResultContent::Text("3".to_string())],
        });
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "toolResult": { "toolUseId": "tool-1", "status": "success", "content": [{ "text": "3" }] }
            })
        );
    }

    // Message.fromMessageData: round-trips content blocks through the wire form
    #[test]
    fn message_round_trips_through_json() {
        let message = Message::new(
            Role::Assistant,
            vec![
                ContentBlock::text("Hello"),
                ContentBlock::ToolUse(ToolUseBlock {
                    name: "calc".to_string(),
                    tool_use_id: "tool-1".to_string(),
                    input: serde_json::json!({ "a": 1 }),
                    reasoning_signature: None,
                }),
            ],
        );
        let json = serde_json::to_string(&message).unwrap();
        let restored: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, message);
    }

    // Role: serializes to lowercase single-word literals ("user"/"assistant")
    #[test]
    fn role_serializes_lowercase() {
        assert_eq!(
            serde_json::to_value(Role::User).unwrap(),
            serde_json::json!("user")
        );
        assert_eq!(
            serde_json::to_value(Role::Assistant).unwrap(),
            serde_json::json!("assistant")
        );
    }

    // StopReason: known values map to camelCase; unknown values pass through
    #[test]
    fn stop_reason_wire_form() {
        assert_eq!(StopReason::ToolUse.as_str(), "toolUse");
        assert_eq!(StopReason::EndTurn.as_str(), "endTurn");
        assert_eq!(StopReason::from_wire("toolUse"), StopReason::ToolUse);
        assert_eq!(
            StopReason::from_wire("somethingNew"),
            StopReason::Other("somethingNew".to_string())
        );
        assert_eq!(
            StopReason::Other("somethingNew".to_string()).as_str(),
            "somethingNew"
        );
    }

    // Message.text(): concatenates text blocks, ignoring non-text content
    #[test]
    fn message_text_concatenates_text_blocks() {
        let message = Message::new(
            Role::Assistant,
            vec![
                ContentBlock::text("Hello "),
                ContentBlock::ToolUse(ToolUseBlock {
                    name: "noop".to_string(),
                    tool_use_id: "id".to_string(),
                    input: serde_json::json!({}),
                    reasoning_signature: None,
                }),
                ContentBlock::text("world"),
            ],
        );
        assert_eq!(message.text(), "Hello world");
    }
}

/// Serde helper for `Option<Vec<u8>>` fields that are base64 on the wire.
mod opt_base64_bytes {
    use base64::Engine;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(
        bytes: &Option<Vec<u8>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match bytes {
            Some(bytes) => {
                let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
                serializer.serialize_str(&encoded)
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Vec<u8>>, D::Error> {
        let encoded = Option::<String>::deserialize(deserializer)?;
        match encoded {
            Some(encoded) => base64::engine::general_purpose::STANDARD
                .decode(encoded.as_bytes())
                .map(Some)
                .map_err(serde::de::Error::custom),
            None => Ok(None),
        }
    }
}
