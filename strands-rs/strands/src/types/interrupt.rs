//! Interrupt-related data types for human-in-the-loop workflows. Ports
//! `types/interrupt.ts`.

use serde::{Deserialize, Serialize};

/// Parameters for raising an interrupt. Ports `InterruptParams`.
#[derive(Debug, Clone, Default)]
pub struct InterruptParams {
    /// User-defined name for the interrupt. Must be unique within a single hook
    /// callback or tool execution.
    pub name: String,
    /// User-provided reason for the interrupt.
    pub reason: Option<serde_json::Value>,
    /// Preemptive response. When provided, the interrupt returns this value
    /// immediately without halting execution — useful for reusing a prior
    /// session's trust response.
    pub response: Option<serde_json::Value>,
}

impl InterruptParams {
    /// Creates parameters with only a name.
    pub fn new(name: impl Into<String>) -> Self {
        InterruptParams {
            name: name.into(),
            reason: None,
            response: None,
        }
    }

    /// Sets the reason.
    pub fn with_reason(mut self, reason: serde_json::Value) -> Self {
        self.reason = Some(reason);
        self
    }

    /// Sets a preemptive response.
    pub fn with_response(mut self, response: serde_json::Value) -> Self {
        self.response = Some(response);
        self
    }
}

/// A user's response to an interrupt. Ports `InterruptResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InterruptResponse {
    /// Unique identifier of the interrupt being responded to.
    pub interrupt_id: String,
    /// The user's response to the interrupt.
    pub response: serde_json::Value,
}

impl InterruptResponse {
    /// Creates a response for the interrupt with `interrupt_id`.
    pub fn new(interrupt_id: impl Into<String>, response: serde_json::Value) -> Self {
        InterruptResponse {
            interrupt_id: interrupt_id.into(),
            response,
        }
    }
}

/// A content block carrying a user response to an interrupt. Ports
/// `InterruptResponseContent`; serializes as `{ "interruptResponse": { … } }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InterruptResponseContent {
    /// The interrupt response data.
    pub interrupt_response: InterruptResponse,
}

impl InterruptResponseContent {
    /// Wraps an [`InterruptResponse`] in a content block.
    pub fn new(interrupt_response: InterruptResponse) -> Self {
        InterruptResponseContent { interrupt_response }
    }
}
