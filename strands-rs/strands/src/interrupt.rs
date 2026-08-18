//! Human-in-the-loop interrupt system. Ports `interrupt.ts`.
//!
//! A tool or hook calls `interrupt(...)`; if a response is already available
//! (from a resume) it is returned, otherwise the call raises
//! [`StrandsError::Interrupt`], which the agent loop catches to halt with
//! [`crate::StopReason::Interrupt`]. The caller resumes with
//! [`crate::Agent::resume`].
//!
//! # Deviations from the TypeScript port
//!
//! - **`InterruptError` is an `Err`, not a thrown exception.** Rust has no
//!   exceptions, so `interrupt()` returns `Result` and the loop matches on the
//!   [`StrandsError::Interrupt`] variant.
//! - **`InterruptState` is carried as an `Arc`-backed handle** on the events and
//!   tool context that can raise interrupts, rather than reached through an
//!   `agent` back-reference (which the events do not carry).
//! - **Resume takes `Vec<InterruptResponse>` via [`crate::Agent::resume`]** rather
//!   than `interruptResponseContent` blocks passed to `invoke`. The
//!   [`crate::types::interrupt::InterruptResponseContent`] block exists for
//!   session serialization parity.
//! - **The concurrent executor's deferred-interrupt semantics are not ported**
//!   (only sequential tool execution exists in the slice).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::errors::StrandsError;
use crate::types::interrupt::{InterruptParams, InterruptResponse};
use crate::types::messages::{Message, ToolResultBlock};

/// Origin of an interrupt. Ports `InterruptSource`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InterruptSource {
    /// Raised by a tool callback via `ToolContext::interrupt`.
    Tool,
    /// Raised by an agent-level hook (e.g. `BeforeToolCallEvent::interrupt`).
    Hook,
    /// Raised by middleware (not yet ported).
    Middleware,
    /// Raised by a multi-agent hook (not yet ported).
    #[serde(rename = "multiagent-hook")]
    MultiagentHook,
}

fn default_source() -> InterruptSource {
    // Legacy snapshots that predate the `source` field default to `hook`.
    InterruptSource::Hook
}

/// An interrupt that pauses agent execution for human input. Ports `Interrupt`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interrupt {
    /// Unique identifier for this interrupt.
    pub id: String,
    /// User-defined name for the interrupt.
    pub name: String,
    /// User-provided reason for raising the interrupt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<serde_json::Value>,
    /// Human response provided when resuming after an interrupt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<serde_json::Value>,
    /// Where this interrupt was raised from.
    #[serde(default = "default_source")]
    pub source: InterruptSource,
}

/// The error raised when human input is required to continue. Ports
/// `InterruptError`; surfaced as [`StrandsError::Interrupt`].
#[derive(Debug, Clone)]
pub struct InterruptError {
    /// The interrupts that caused this error.
    pub interrupts: Vec<Interrupt>,
}

impl InterruptError {
    /// Creates an error carrying `interrupts`.
    pub fn new(interrupts: Vec<Interrupt>) -> Self {
        InterruptError { interrupts }
    }
}

impl std::fmt::Display for InterruptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.interrupts.as_slice() {
            [interrupt] => write!(f, "Interrupt raised: {}", interrupt.name),
            interrupts => write!(
                f,
                "{} interrupts raised: {}",
                interrupts.len(),
                interrupts
                    .iter()
                    .map(|interrupt| interrupt.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

impl std::error::Error for InterruptError {}

/// Tool-execution state stored when an interrupt occurs mid-turn, so the model
/// need not be re-invoked on resume. Ports `PendingToolExecution`.
#[derive(Debug, Clone)]
pub struct PendingToolExecution {
    /// The assistant message containing the tool-use blocks.
    pub assistant_message: Message,
    /// Tool results completed before the interrupt, keyed by tool-use id.
    pub completed_tool_results: HashMap<String, ToolResultBlock>,
}

/// Serializable form of [`InterruptState`]. Ports `InterruptStateData`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InterruptStateData {
    /// Interrupts keyed by id.
    pub interrupts: HashMap<String, Interrupt>,
    /// Resume responses provided when resuming.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resume_responses: Vec<InterruptResponse>,
    /// Whether the agent is in an interrupted state.
    pub activated: bool,
}

struct InterruptStateInner {
    // Insertion order preserved so unanswered-interrupt order matches TypeScript's
    // object-insertion order.
    interrupts: Vec<Interrupt>,
    resume_responses: Vec<InterruptResponse>,
    activated: bool,
    pending_tool_execution: Option<PendingToolExecution>,
}

/// Tracks interrupts raised during agent execution. Ports `InterruptState`.
///
/// A cheap-clone `Arc`-backed handle: the agent owns the canonical state and
/// hands clones to the events and tool contexts that can raise interrupts, so
/// they all mutate the same underlying state.
#[derive(Clone)]
pub struct InterruptState {
    inner: Arc<Mutex<InterruptStateInner>>,
}

impl Default for InterruptState {
    fn default() -> Self {
        InterruptState {
            inner: Arc::new(Mutex::new(InterruptStateInner {
                interrupts: Vec::new(),
                resume_responses: Vec::new(),
                activated: false,
                pending_tool_execution: None,
            })),
        }
    }
}

impl InterruptState {
    /// Creates an empty interrupt state.
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, InterruptStateInner> {
        self.inner.lock().expect("interrupt state mutex poisoned")
    }

    /// Returns the interrupt with `id`, or creates one if absent. A new
    /// interrupt with a preemptive `response` returns already-answered. Ports
    /// `getOrCreateInterrupt`.
    pub fn get_or_create_interrupt(
        &self,
        id: String,
        name: String,
        reason: Option<serde_json::Value>,
        response: Option<serde_json::Value>,
        source: InterruptSource,
    ) -> Interrupt {
        let mut inner = self.lock();
        if let Some(existing) = inner.interrupts.iter().find(|interrupt| interrupt.id == id) {
            return existing.clone();
        }
        let interrupt = Interrupt {
            id,
            name,
            reason,
            response,
            source,
        };
        inner.interrupts.push(interrupt.clone());
        interrupt
    }

    /// Registers an existing interrupt (without its response), or returns the
    /// already-registered one. Ports `registerInterrupt`.
    pub fn register_interrupt(&self, interrupt: &Interrupt) -> Interrupt {
        self.get_or_create_interrupt(
            interrupt.id.clone(),
            interrupt.name.clone(),
            interrupt.reason.clone(),
            None,
            interrupt.source,
        )
    }

    /// Whether the agent is in an interrupted state.
    pub fn is_activated(&self) -> bool {
        self.lock().activated
    }

    /// Marks the state activated. Ports `activate`.
    pub fn activate(&self) {
        self.lock().activated = true;
    }

    /// Clears all interrupts and resume/pending state. Ports `deactivate`.
    pub fn deactivate(&self) {
        let mut inner = self.lock();
        inner.interrupts.clear();
        inner.resume_responses.clear();
        inner.activated = false;
        inner.pending_tool_execution = None;
    }

    /// Applies resume responses to matching interrupts. No-op when not
    /// activated. Ports `resume`.
    ///
    /// # Errors
    /// Returns a model error if a response references an unknown interrupt id.
    pub fn resume(&self, responses: Vec<InterruptResponse>) -> Result<(), StrandsError> {
        let mut inner = self.lock();
        if !inner.activated {
            return Ok(());
        }
        for response in &responses {
            let interrupt = inner
                .interrupts
                .iter_mut()
                .find(|interrupt| interrupt.id == response.interrupt_id)
                .ok_or_else(|| {
                    StrandsError::model(format!(
                        "interrupt_id=<{}> | no interrupt found",
                        response.interrupt_id
                    ))
                })?;
            interrupt.response = Some(response.response.clone());
        }
        inner.resume_responses = responses;
        Ok(())
    }

    /// Returns the interrupts that have no response yet. Ports
    /// `getUnansweredInterrupts`.
    pub fn get_unanswered_interrupts(&self) -> Vec<Interrupt> {
        self.lock()
            .interrupts
            .iter()
            .filter(|interrupt| interrupt.response.is_none())
            .cloned()
            .collect()
    }

    /// Stores pending tool-execution state. Ports `setPendingToolExecution`.
    pub fn set_pending_tool_execution(&self, pending: PendingToolExecution) {
        self.lock().pending_tool_execution = Some(pending);
    }

    /// Returns and reconstructs the pending tool-execution state, if any. Ports
    /// `getPendingExecution`.
    pub fn get_pending_execution(&self) -> Option<PendingToolExecution> {
        self.lock().pending_tool_execution.clone()
    }

    /// Clears the pending tool-execution state. Ports `clearPendingToolExecution`.
    pub fn clear_pending_tool_execution(&self) {
        self.lock().pending_tool_execution = None;
    }

    /// Serializes to [`InterruptStateData`]. Ports `toJSON`.
    pub fn to_data(&self) -> InterruptStateData {
        let inner = self.lock();
        let interrupts = inner
            .interrupts
            .iter()
            .map(|interrupt| (interrupt.id.clone(), interrupt.clone()))
            .collect();
        InterruptStateData {
            interrupts,
            resume_responses: inner.resume_responses.clone(),
            activated: inner.activated,
        }
    }

    /// Reconstructs from [`InterruptStateData`]. Ports `fromJSON`. Pending
    /// tool-execution state is not serialized in the slice (it holds live
    /// messages) and is left empty.
    pub fn from_data(data: InterruptStateData) -> Self {
        let interrupts = data.interrupts.into_values().collect();
        InterruptState {
            inner: Arc::new(Mutex::new(InterruptStateInner {
                interrupts,
                resume_responses: data.resume_responses,
                activated: data.activated,
                pending_tool_execution: None,
            })),
        }
    }
}

impl std::fmt::Debug for InterruptState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.lock();
        f.debug_struct("InterruptState")
            .field("interrupts", &inner.interrupts.len())
            .field("activated", &inner.activated)
            .finish_non_exhaustive()
    }
}

/// Raises or resumes an interrupt against `state`. Ports `interruptFromAgent`.
///
/// Returns the response when one is already available (resume or preemptive),
/// otherwise `Err(`[`StrandsError::Interrupt`]`)`.
pub fn interrupt_from_state(
    state: &InterruptState,
    interrupt_id: String,
    params: InterruptParams,
    source: InterruptSource,
) -> Result<serde_json::Value, StrandsError> {
    let interrupt = state.get_or_create_interrupt(
        interrupt_id,
        params.name,
        params.reason,
        params.response,
        source,
    );
    if let Some(response) = interrupt.response.clone() {
        return Ok(response);
    }
    Err(StrandsError::Interrupt(InterruptError::new(vec![
        interrupt,
    ])))
}

#[cfg(test)]
mod tests {
    //! Ports the core `interrupt.test.ts` specs: raise-then-resume, preemptive
    //! response, unknown-id resume error, and unanswered-interrupt tracking.

    use super::*;
    use serde_json::json;

    fn params(name: &str) -> InterruptParams {
        InterruptParams::new(name)
    }

    // "raising an interrupt returns Err with the interrupt; resume returns the response"
    #[test]
    fn raise_then_resume() {
        let state = InterruptState::new();
        let id = "tool:t1:confirm".to_string();

        // First call: no response yet → interrupt raised.
        let error =
            interrupt_from_state(&state, id.clone(), params("confirm"), InterruptSource::Tool)
                .unwrap_err();
        let StrandsError::Interrupt(interrupt_error) = error else {
            panic!("expected an interrupt error");
        };
        assert_eq!(interrupt_error.interrupts.len(), 1);
        assert_eq!(interrupt_error.interrupts[0].id, id);

        // Resume with a response, then the same call returns it.
        state.activate();
        state
            .resume(vec![InterruptResponse::new(id.clone(), json!("approved"))])
            .unwrap();
        let response =
            interrupt_from_state(&state, id, params("confirm"), InterruptSource::Tool).unwrap();
        assert_eq!(response, json!("approved"));
    }

    // "a preemptive response skips the interrupt"
    #[test]
    fn preemptive_response_skips() {
        let state = InterruptState::new();
        let params = InterruptParams::new("confirm").with_response(json!("yes"));
        let response = interrupt_from_state(
            &state,
            "tool:t1:confirm".to_string(),
            params,
            InterruptSource::Tool,
        )
        .unwrap();
        assert_eq!(response, json!("yes"));
    }

    // "resume with an unknown interrupt id errors"
    #[test]
    fn resume_unknown_id_errors() {
        let state = InterruptState::new();
        state.activate();
        let error = state
            .resume(vec![InterruptResponse::new("missing", json!(1))])
            .unwrap_err();
        assert!(error.to_string().contains("no interrupt found"));
    }

    // "unanswered interrupts exclude answered ones"
    #[test]
    fn tracks_unanswered() {
        let state = InterruptState::new();
        let _ = interrupt_from_state(&state, "a".to_string(), params("a"), InterruptSource::Hook);
        let _ = interrupt_from_state(&state, "b".to_string(), params("b"), InterruptSource::Hook);
        state.activate();
        state
            .resume(vec![InterruptResponse::new("a", json!(true))])
            .unwrap();
        let unanswered = state.get_unanswered_interrupts();
        assert_eq!(unanswered.len(), 1);
        assert_eq!(unanswered[0].id, "b");
    }

    // "deactivate clears all state"
    #[test]
    fn deactivate_clears() {
        let state = InterruptState::new();
        let _ = interrupt_from_state(&state, "a".to_string(), params("a"), InterruptSource::Hook);
        state.activate();
        state.deactivate();
        assert!(!state.is_activated());
        assert!(state.get_unanswered_interrupts().is_empty());
    }

    // "source round-trips and defaults to hook when absent"
    #[test]
    fn source_serialization() {
        let interrupt = Interrupt {
            id: "x".to_string(),
            name: "x".to_string(),
            reason: None,
            response: None,
            source: InterruptSource::Tool,
        };
        let value = serde_json::to_value(&interrupt).unwrap();
        assert_eq!(value["source"], json!("tool"));

        // A snapshot without `source` deserializes to hook.
        let restored: Interrupt =
            serde_json::from_value(json!({ "id": "y", "name": "y" })).unwrap();
        assert_eq!(restored.source, InterruptSource::Hook);
    }
}
