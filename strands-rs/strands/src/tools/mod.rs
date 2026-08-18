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
use crate::interrupt::{interrupt_from_state, InterruptSource, InterruptState};
use crate::types::interrupt::InterruptParams;
use crate::types::messages::{ToolResultBlock, ToolResultContent, ToolResultStatus, ToolUseBlock};
use crate::types::tools::ToolSpec;

pub use function_tool::FunctionTool;
pub use registry::ToolRegistry;

/// Context provided to a tool during execution. Ports `ToolContext`.
///
/// Exposes the triggering tool-use request and the interrupt surface. The
/// agent-handle field from the TypeScript `ToolContext` is deferred.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// The tool-use request that triggered this execution.
    pub tool_use: ToolUseBlock,
    interrupt_state: InterruptState,
}

impl ToolContext {
    /// Creates a context for `tool_use` with a fresh, unattached interrupt state.
    ///
    /// The framework constructs contexts wired to the agent's interrupt state;
    /// this constructor is for exercising a tool in isolation, where a raised
    /// interrupt simply has no responder.
    pub fn new(tool_use: ToolUseBlock) -> Self {
        ToolContext {
            tool_use,
            interrupt_state: InterruptState::new(),
        }
    }

    /// Raises an interrupt for human-in-the-loop workflows. Returns the response
    /// immediately when resuming; otherwise returns `Err(`[`StrandsError::Interrupt`]`)`
    /// to halt the agent. Ports `ToolContext.interrupt`.
    pub fn interrupt(&self, params: InterruptParams) -> Result<serde_json::Value, StrandsError> {
        let id = format!("tool:{}:{}", self.tool_use.tool_use_id, params.name);
        interrupt_from_state(&self.interrupt_state, id, params, InterruptSource::Tool)
    }
}

/// A tool an agent can invoke. Ports the abstract `Tool` class.
///
/// Implementors provide identity (`name`, `description`, `tool_spec`) and an
/// async `invoke`. The framework wraps `invoke`'s `Result` into a
/// [`ToolResultBlock`] via [`execute_tool`], turning an ordinary `Err` into an
/// error result the model can react to (matching `FunctionTool.stream`'s error
/// handling) while propagating a [`StrandsError::Interrupt`] up to the loop.
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
pub async fn execute_tool(
    tool: &dyn Tool,
    tool_use: ToolUseBlock,
    interrupt_state: InterruptState,
) -> Result<ToolResultBlock, StrandsError> {
    execute_tool_reporting_error(tool, tool_use, interrupt_state)
        .await
        .map(|(result, _error)| result)
}

/// Executes a tool like [`execute_tool`], additionally reporting the error
/// message when the tool fails.
///
/// The `Ok` tuple's second element is `Some(message)` only when the tool
/// returned an ordinary `Err` — the counterpart to the `error` field the
/// TypeScript executor sets on `AfterToolCallEvent` (distinct from a tool that
/// deliberately returns an error-status result). A [`StrandsError::Interrupt`]
/// is *not* turned into an error result; it propagates as `Err` so the agent
/// loop can halt, matching the TypeScript `_executeToolCore` re-throw.
pub async fn execute_tool_reporting_error(
    tool: &dyn Tool,
    tool_use: ToolUseBlock,
    interrupt_state: InterruptState,
) -> Result<(ToolResultBlock, Option<String>), StrandsError> {
    let tool_use_id = tool_use.tool_use_id.clone();
    match tool
        .invoke(ToolContext {
            tool_use,
            interrupt_state,
        })
        .await
    {
        Ok(value) => Ok((
            ToolResultBlock {
                tool_use_id,
                status: ToolResultStatus::Success,
                content: vec![wrap_value(value)],
            },
            None,
        )),
        Err(error @ StrandsError::Interrupt(_)) => Err(error),
        Err(error) => Ok((
            ToolResultBlock {
                tool_use_id,
                status: ToolResultStatus::Error,
                content: vec![ToolResultContent::Text(format!("Error: {error}"))],
            },
            Some(error.to_string()),
        )),
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
