//! The result of an agent invocation. Ports `AgentResult` from `types/agent.ts`.

use crate::interrupt::Interrupt;
use crate::types::messages::{Message, StopReason};

/// The outcome of an agent invocation.
///
/// Carries the final stop reason, the last message, and — when the turn halted
/// on an interrupt — the unanswered interrupts to respond to. Metrics, traces,
/// structured output, and checkpoints from the TypeScript `AgentResult` are out
/// of the vertical slice's scope.
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// The stop reason from the final model response.
    pub stop_reason: StopReason,
    /// The last message added to the conversation.
    pub last_message: Message,
    /// Unanswered interrupts when `stop_reason` is [`StopReason::Interrupt`];
    /// empty otherwise. Respond to these via [`crate::Agent::resume`].
    pub interrupts: Vec<Interrupt>,
}

impl AgentResult {
    pub(crate) fn new(stop_reason: StopReason, last_message: Message) -> Self {
        AgentResult {
            stop_reason,
            last_message,
            interrupts: Vec::new(),
        }
    }

    pub(crate) fn with_interrupts(
        stop_reason: StopReason,
        last_message: Message,
        interrupts: Vec<Interrupt>,
    ) -> Self {
        AgentResult {
            stop_reason,
            last_message,
            interrupts,
        }
    }

    /// The concatenated text of the last message's text blocks.
    ///
    /// Mirrors the priority of the TypeScript `toString()` for the text case
    /// (structured output and interrupts are out of scope here).
    pub fn text(&self) -> String {
        self.last_message.text()
    }
}

impl std::fmt::Display for AgentResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text())
    }
}
