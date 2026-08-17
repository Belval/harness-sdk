//! Tool definition, registration, and execution.
//!
//! Ports `tools/tool.ts`, `tools/function-tool.ts`, `registry/tool-registry.ts`,
//! and the sequential executor from `tools/executors/`. The TypeScript SDK's
//! streaming tool model (a tool yields progress events then returns a result) is
//! reduced here to a single async result: the vertical slice does not port the
//! tool progress-streaming surface.

pub mod function_tool;
pub mod registry;

use async_trait::async_trait;

use crate::errors::StrandsError;
use crate::types::messages::{
    ContentBlock, Message, ToolResultBlock, ToolResultContent, ToolResultStatus, ToolUseBlock,
};
use crate::types::tools::ToolSpec;

pub use function_tool::FunctionTool;
pub use registry::ToolRegistry;

/// Context provided to a tool during execution. Ports `ToolContext`.
///
/// The vertical slice exposes the triggering tool-use request; agent-handle and
/// invocation-state fields from the TypeScript `ToolContext` are deferred.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// The tool-use request that triggered this execution.
    pub tool_use: ToolUseBlock,
}

/// A tool an agent can invoke. Ports the abstract `Tool` class.
///
/// Implementors provide identity (`name`, `description`, `tool_spec`) and an
/// async `invoke`. The framework wraps `invoke`'s `Result` into a
/// [`ToolResultBlock`] via [`execute_tool`], turning `Err` into an error result
/// the model can react to, matching `FunctionTool.stream`'s error handling.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The unique name of the tool. MUST match `tool_spec().name`.
    fn name(&self) -> &str;

    /// Human-readable description. MUST match `tool_spec().description`.
    fn description(&self) -> &str;

    /// The tool's specification (name, description, input schema).
    fn tool_spec(&self) -> ToolSpec;

    /// Executes the tool, returning JSON output or an error.
    async fn invoke(&self, context: ToolContext) -> Result<serde_json::Value, StrandsError>;
}

/// Executes a tool and wraps the outcome in a [`ToolResultBlock`].
///
/// Ports the result-wrapping behavior of `FunctionTool.stream` + `createErrorResult`:
/// a successful JSON value becomes a success result (text for strings, JSON
/// otherwise, matching Bedrock's content rules), and an error becomes an error
/// result carrying the message.
pub async fn execute_tool(tool: &dyn Tool, tool_use: ToolUseBlock) -> ToolResultBlock {
    execute_tool_reporting_error(tool, tool_use).await.0
}

/// Executes a tool like [`execute_tool`], additionally reporting the error
/// message when the tool fails.
///
/// The tuple's second element is `Some(message)` only when the tool returned
/// `Err` — the counterpart to the `error` field the TypeScript executor sets on
/// `AfterToolCallEvent` (distinct from a tool that deliberately returns an
/// error-status result).
pub async fn execute_tool_reporting_error(
    tool: &dyn Tool,
    tool_use: ToolUseBlock,
) -> (ToolResultBlock, Option<String>) {
    let tool_use_id = tool_use.tool_use_id.clone();
    match tool.invoke(ToolContext { tool_use }).await {
        Ok(value) => (
            ToolResultBlock {
                tool_use_id,
                status: ToolResultStatus::Success,
                content: vec![wrap_value(value)],
            },
            None,
        ),
        Err(error) => (
            ToolResultBlock {
                tool_use_id,
                status: ToolResultStatus::Error,
                content: vec![ToolResultContent::Text(format!("Error: {error}"))],
            },
            Some(error.to_string()),
        ),
    }
}

/// Wraps a tool's JSON return value in tool-result content.
///
/// Mirrors `FunctionTool._wrapInToolResult`: strings, numbers, and booleans
/// become text (Bedrock rejects bare primitives as JSON content); `null`
/// becomes the literal text `"null"`; objects and arrays become JSON.
fn wrap_value(value: serde_json::Value) -> ToolResultContent {
    match value {
        serde_json::Value::String(text) => ToolResultContent::Text(text),
        serde_json::Value::Null => ToolResultContent::Text("null".to_string()),
        number @ serde_json::Value::Number(_) => ToolResultContent::Text(number.to_string()),
        serde_json::Value::Bool(boolean) => ToolResultContent::Text(boolean.to_string()),
        object @ serde_json::Value::Object(_) => ToolResultContent::Json(object),
        array @ serde_json::Value::Array(_) => {
            ToolResultContent::Json(serde_json::json!({ "$value": array }))
        }
    }
}

/// Runs every tool-use block in an assistant message in source order, producing
/// the user message of tool results. Ports the sequential executor's core:
/// unknown tools become error results (via [`ToolResultBlock`]) rather than
/// aborting the turn.
pub async fn execute_tools(registry: &ToolRegistry, assistant_message: &Message) -> Message {
    let mut result_blocks = Vec::new();
    for block in &assistant_message.content {
        let ContentBlock::ToolUse(tool_use) = block else {
            continue;
        };
        match registry.resolve(&tool_use.name) {
            Ok(tool) => result_blocks.push(execute_tool(tool.as_ref(), tool_use.clone()).await),
            Err(error) => result_blocks.push(ToolResultBlock {
                tool_use_id: tool_use.tool_use_id.clone(),
                status: ToolResultStatus::Error,
                content: vec![ToolResultContent::Text(format!("Error: {error}"))],
            }),
        }
    }
    Message::new(
        crate::types::messages::Role::User,
        result_blocks
            .into_iter()
            .map(ContentBlock::ToolResult)
            .collect(),
    )
}
