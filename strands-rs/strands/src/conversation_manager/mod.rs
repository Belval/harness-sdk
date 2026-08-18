//! Conversation-history management. Ports `agent/conversation-manager`.
//!
//! A [`ConversationManager`] keeps the conversation within the model's context
//! window: [`ConversationManager::apply_management`] proactively trims or
//! summarizes history, and [`ConversationManager::reduce_context`] reactively
//! shrinks it when the model reports an overflow. The agent holds one and the
//! loop calls `reduce_context` when a model call fails with
//! [`StrandsError::ContextWindowOverflow`], then retries.
//!
//! # Deviations from the TypeScript/Python port
//!
//! - **`apply_management` is not yet driven from hooks.** In Python a
//!   `ContextManager` hook calls `agent.conversation_manager.apply_management`;
//!   strands-rs hook callbacks are synchronous and cannot await, so proactive
//!   management is exposed on the agent/handle but its loop/hook trigger lands
//!   with async hooks. The reactive `reduce_context` path is loop-driven and
//!   fully integrated.

use async_trait::async_trait;

use crate::agent::{AgentHandle, InvocationState};
use crate::errors::StrandsError;

/// Manages the conversation history to keep it within the context window. Ports
/// the `ConversationManager` base class.
#[async_trait]
pub trait ConversationManager: Send + Sync {
    /// Proactively manages the history (trim/summarize). `current_tokens` is the
    /// caller's token estimate, or `None` to force full management (the Python
    /// `float('inf')` case). Ports `apply_management`.
    async fn apply_management(
        &self,
        agent: &AgentHandle,
        current_tokens: Option<u64>,
        invocation_state: &InvocationState,
    ) -> Result<(), StrandsError>;

    /// Reactively reduces the history after the model reports a context-window
    /// overflow, then the loop retries. `error` is the overflow message. Returns
    /// `Err` if the history cannot be reduced further (the overflow propagates).
    /// Ports `reduce_context`.
    async fn reduce_context(
        &self,
        agent: &AgentHandle,
        error: Option<&str>,
    ) -> Result<(), StrandsError>;
}

/// A conversation manager that does nothing. Ports `NullConversationManager`.
///
/// `apply_management` is a no-op; `reduce_context` cannot reduce, so it
/// re-raises the overflow.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullConversationManager;

#[async_trait]
impl ConversationManager for NullConversationManager {
    async fn apply_management(
        &self,
        _agent: &AgentHandle,
        _current_tokens: Option<u64>,
        _invocation_state: &InvocationState,
    ) -> Result<(), StrandsError> {
        Ok(())
    }

    async fn reduce_context(
        &self,
        _agent: &AgentHandle,
        error: Option<&str>,
    ) -> Result<(), StrandsError> {
        Err(StrandsError::ContextWindowOverflow(
            error
                .unwrap_or("context window overflow and no conversation manager to reduce it")
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentState, Messages};

    fn handle() -> AgentHandle {
        AgentHandle::new(AgentState::new(), Messages::default(), None)
    }

    // NullConversationManager: apply is a no-op, reduce re-raises the overflow
    #[tokio::test]
    async fn null_manager_reraises_on_reduce() {
        let manager = NullConversationManager;
        let state = InvocationState::new();
        manager
            .apply_management(&handle(), None, &state)
            .await
            .unwrap();

        let error = manager
            .reduce_context(&handle(), Some("too long"))
            .await
            .unwrap_err();
        assert!(matches!(error, StrandsError::ContextWindowOverflow(_)));
        assert!(error.to_string().contains("too long"));
    }
}
