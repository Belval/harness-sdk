//! Agent hook events. Ports the event taxonomy from `hooks/events.ts`.
//!
//! Each event is a plain struct carrying its data plus any mutable control
//! fields a callback may set to steer the loop. `Before*`/`After*` pairs bracket
//! agent operations; `After*` events reverse callback order for cleanup
//! semantics. The TypeScript `agent` back-reference is omitted (see the module
//! docs), and the streaming-update events are deferred with the streaming
//! feature.

use std::sync::Arc;

use super::HookEvent;
use crate::agent::{AgentResult, InvocationState};
use crate::errors::StrandsError;
use crate::interrupt::{interrupt_from_state, Interrupt, InterruptSource, InterruptState};
use crate::tools::Tool;
use crate::types::interrupt::InterruptParams;
use crate::types::messages::{ContentBlock, Message, StopReason, ToolResultBlock};

/// A hook-requested cancellation. Ports the `cancel: boolean | string` field:
/// the string is a cancellation *reason* used as the response/error message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookCancel {
    /// Cancel using the event's default message.
    Default,
    /// Cancel using this string as the message.
    WithMessage(String),
}

impl HookCancel {
    /// Resolves the cancellation message, falling back to `default` for
    /// [`HookCancel::Default`].
    pub fn message<'a>(&'a self, default: &'a str) -> &'a str {
        match self {
            HookCancel::Default => default,
            HookCancel::WithMessage(message) => message,
        }
    }
}

/// A hook-requested early end of turn. Ports `AfterToolsEvent.endTurn`: unlike
/// [`HookCancel`], a string here is *literal assistant content*, not a reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookEndTurn {
    /// End the turn with the event's default assistant message.
    Default,
    /// End the turn using this string as the literal assistant content.
    WithContent(String),
}

impl HookEndTurn {
    /// Resolves the assistant content, falling back to `default` for
    /// [`HookEndTurn::Default`].
    pub fn content<'a>(&'a self, default: &'a str) -> &'a str {
        match self {
            HookEndTurn::Default => default,
            HookEndTurn::WithContent(content) => content,
        }
    }
}

/// Mutable tool-use descriptor carried on tool-call hook events. Ports `ToolUseData`.
///
/// Hooks on [`BeforeToolCallEvent`] may mutate its fields (or replace it) to
/// rewrite the tool input or name before execution. The model-issued
/// `tool_use_id` remains the authoritative provider correlation key.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolUseData {
    /// The name of the tool to execute.
    pub name: String,
    /// Unique identifier for this tool-use instance.
    pub tool_use_id: String,
    /// The input parameters for the tool.
    pub input: serde_json::Value,
}

/// The outcome of a model invocation. Ports `ModelStopData`.
///
/// Guardrail redaction from the TypeScript type is deferred with the guardrails
/// surface.
#[derive(Debug, Clone)]
pub struct ModelStopData {
    /// The message returned by the model.
    pub message: Message,
    /// The reason the model stopped generating.
    pub stop_reason: StopReason,
}

/// Fired once after the agent has been fully constructed. Ports `InitializedEvent`.
#[derive(Debug, Default)]
pub struct InitializedEvent {}
impl HookEvent for InitializedEvent {}

/// Fired at the beginning of a new agent request. Ports `BeforeInvocationEvent`.
#[derive(Debug)]
pub struct BeforeInvocationEvent {
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
    /// Set by a callback to cancel the invocation. The assistant response is the
    /// resolved [`HookCancel`] message (default: `"invocation denied by hook"`).
    pub cancel: Option<HookCancel>,
}
impl HookEvent for BeforeInvocationEvent {}

/// Fired at the end of an agent request, whether it succeeded or failed. Ports
/// `AfterInvocationEvent`. Uses reverse callback ordering.
#[derive(Debug)]
pub struct AfterInvocationEvent {
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
    /// Set by a callback to re-enter the loop with new input after this event's
    /// callbacks complete. Ignored when the invocation ended with an error. If
    /// multiple callbacks set it, the last to run wins.
    pub resume: Option<Message>,
}
impl HookEvent for AfterInvocationEvent {
    fn should_reverse_callbacks(&self) -> bool {
        true
    }
}

/// Fired when the framework adds a message to the conversation. Ports
/// `MessageAddedEvent`. Does not fire for preloaded or manually pushed messages.
#[derive(Debug)]
pub struct MessageAddedEvent {
    /// The message that was added.
    pub message: Message,
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
}
impl HookEvent for MessageAddedEvent {}

/// Fired just before the model is invoked. Ports `BeforeModelCallEvent`.
#[derive(Debug)]
pub struct BeforeModelCallEvent {
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
    /// Set by a callback to skip the model call. The assistant response is the
    /// resolved [`HookCancel`] message (default: `"model call denied by hook"`).
    pub cancel: Option<HookCancel>,
}
impl HookEvent for BeforeModelCallEvent {}

/// Fired after a model invocation completes, whether it succeeded or failed.
/// Ports `AfterModelCallEvent`. Uses reverse callback ordering.
#[derive(Debug)]
pub struct AfterModelCallEvent {
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
    /// 1-indexed count of model attempts for this turn, including the attempt
    /// that just completed or failed.
    pub attempt_count: u32,
    /// The model outcome on success; `None` when the call errored.
    pub stop_data: Option<ModelStopData>,
    /// The error message when the call failed; `None` on success.
    pub error: Option<String>,
    /// Set by a callback to retry the model invocation.
    pub retry: bool,
}
impl HookEvent for AfterModelCallEvent {
    fn should_reverse_callbacks(&self) -> bool {
        true
    }
}

/// Fired when the model completes a full message. Ports `ModelMessageEvent`.
#[derive(Debug)]
pub struct ModelMessageEvent {
    /// The assembled assistant message.
    pub message: Message,
    /// Why the model stopped generating.
    pub stop_reason: StopReason,
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
}
impl HookEvent for ModelMessageEvent {}

/// Fired for each completed content block during model inference. Ports
/// `ContentBlockEvent`.
#[derive(Debug)]
pub struct ContentBlockEvent {
    /// The completed content block.
    pub content_block: ContentBlock,
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
}
impl HookEvent for ContentBlockEvent {}

/// Fired before executing the tools from one model turn. Ports `BeforeToolsEvent`.
///
/// Implements the interrupt surface: a callback may call [`Self::interrupt`] to
/// pause the agent for human input before any tool runs.
#[derive(Debug)]
pub struct BeforeToolsEvent {
    /// The assistant message containing the tool-use blocks.
    pub message: Message,
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
    /// Set by a callback to cancel all tools in the batch. Each tool then yields
    /// the resolved [`HookCancel`] message (default: `"Tool cancelled by hook"`).
    pub cancel: Option<HookCancel>,
    /// Interrupt state handle, backing [`Self::interrupt`].
    pub(crate) interrupt_state: InterruptState,
}
impl HookEvent for BeforeToolsEvent {}

impl BeforeToolsEvent {
    /// Raises an interrupt for human-in-the-loop workflows. Returns the response
    /// immediately when resuming; otherwise returns `Err(`[`StrandsError::Interrupt`]`)`
    /// to halt the agent. Ports `BeforeToolsEvent.interrupt`.
    pub fn interrupt(&self, params: InterruptParams) -> Result<serde_json::Value, StrandsError> {
        let id = format!("hook:beforeTools:{}", params.name);
        interrupt_from_state(&self.interrupt_state, id, params, InterruptSource::Hook)
    }
}

/// Fired after all tools in a turn complete. Ports `AfterToolsEvent`. Uses
/// reverse callback ordering.
#[derive(Debug)]
pub struct AfterToolsEvent {
    /// The tool-result message assembled from all tool outputs.
    pub message: Message,
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
    /// Set by a callback to halt the loop without another model call. The
    /// resolved [`HookEndTurn`] content becomes the final assistant message
    /// (default: `"Turn ended early by hook after tool execution"`).
    pub end_turn: Option<HookEndTurn>,
}
impl HookEvent for AfterToolsEvent {
    fn should_reverse_callbacks(&self) -> bool {
        true
    }
}

/// Fired just before a single tool executes. Ports `BeforeToolCallEvent`.
///
/// A callback may mutate [`Self::tool_use`] to rewrite the input or name, set
/// [`Self::selected_tool`] to run a replacement tool, or set [`Self::cancel`].
pub struct BeforeToolCallEvent {
    /// The tool-use request; mutable so callbacks can rewrite it before execution.
    pub tool_use: ToolUseData,
    /// The tool resolved from the registry, if the name matched one.
    pub tool: Option<Arc<dyn Tool>>,
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
    /// Set by a callback to cancel this tool call. The tool yields the resolved
    /// [`HookCancel`] error message (default: `"Tool cancelled by hook"`).
    pub cancel: Option<HookCancel>,
    /// Set by a callback to execute a replacement tool. Takes precedence over
    /// re-resolving a renamed `tool_use`. If several callbacks set it, the last
    /// to run wins.
    pub selected_tool: Option<Arc<dyn Tool>>,
    /// Interrupt state handle, backing [`Self::interrupt`].
    pub(crate) interrupt_state: InterruptState,
}
impl HookEvent for BeforeToolCallEvent {}

impl BeforeToolCallEvent {
    /// Raises an interrupt for human-in-the-loop workflows. Returns the response
    /// immediately when resuming; otherwise returns `Err(`[`StrandsError::Interrupt`]`)`
    /// to halt the agent. Ports `BeforeToolCallEvent.interrupt`.
    pub fn interrupt(&self, params: InterruptParams) -> Result<serde_json::Value, StrandsError> {
        let id = format!(
            "hook:beforeToolCall:{}:{}",
            self.tool_use.tool_use_id, params.name
        );
        interrupt_from_state(&self.interrupt_state, id, params, InterruptSource::Hook)
    }
}

impl std::fmt::Debug for BeforeToolCallEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BeforeToolCallEvent")
            .field("tool_use", &self.tool_use)
            .field("tool", &self.tool.as_ref().map(|tool| tool.name()))
            .field("cancel", &self.cancel)
            .field(
                "selected_tool",
                &self.selected_tool.as_ref().map(|tool| tool.name()),
            )
            .finish_non_exhaustive()
    }
}

/// Fired after a single tool execution completes. Ports `AfterToolCallEvent`.
/// Uses reverse callback ordering.
///
/// A callback may mutate [`Self::result`] to rewrite the tool result before it
/// enters history, or set [`Self::retry`] to re-run the tool.
pub struct AfterToolCallEvent {
    /// The tool-use request that was executed.
    pub tool_use: ToolUseData,
    /// The tool that ran, if one was resolved.
    pub tool: Option<Arc<dyn Tool>>,
    /// The tool result; mutable so callbacks can redact or transform it.
    pub result: ToolResultBlock,
    /// The error message when the tool failed; `None` on success.
    pub error: Option<String>,
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
    /// Set by a callback to re-execute the tool.
    pub retry: bool,
}
impl HookEvent for AfterToolCallEvent {
    fn should_reverse_callbacks(&self) -> bool {
        true
    }
}

impl std::fmt::Debug for AfterToolCallEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AfterToolCallEvent")
            .field("tool_use", &self.tool_use)
            .field("tool", &self.tool.as_ref().map(|tool| tool.name()))
            .field("result", &self.result)
            .field("error", &self.error)
            .field("retry", &self.retry)
            .finish_non_exhaustive()
    }
}

/// Fired when a tool execution completes. Ports `ToolResultEvent`.
#[derive(Debug)]
pub struct ToolResultEvent {
    /// The tool result block.
    pub result: ToolResultBlock,
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
}
impl HookEvent for ToolResultEvent {}

/// Fired as the final event of an invocation. Ports `AgentResultEvent`.
#[derive(Debug)]
pub struct AgentResultEvent {
    /// The agent result carrying the stop reason and last message.
    pub result: AgentResult,
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
}
impl HookEvent for AgentResultEvent {}

/// Fired once per unanswered interrupt when the agent stops to wait for human
/// input. Ports `InterruptEvent`. The `interrupt.source` field discriminates
/// tool-callback from hook-callback origins.
#[derive(Debug)]
pub struct InterruptEvent {
    /// The interrupt the agent is waiting on.
    pub interrupt: Interrupt,
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
}
impl HookEvent for InterruptEvent {}
