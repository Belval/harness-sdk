//! Session persistence. Ports the `SessionManager` surface of `strands.session`.
//!
//! A [`SessionManager`] persists an agent's evolving conversation and state so a
//! session can be restored later. This slice ports the minimal surface the
//! ported consumer needs — [`SessionManager::sync_agent`], which snapshots the
//! agent's current messages/state — reached through the hook-facing
//! [`crate::agent::AgentHandle`] so an (async) hook can persist after it changes
//! the conversation.
//!
//! # Deviations from the port
//!
//! - **Minimal surface.** Only `sync_agent` is ported; full session persistence
//!   (initialization, per-message append, restore) and the SessionManager's own
//!   hook registration for auto-sync-on-`MessageAddedEvent` are deferred — the
//!   consumer drives `sync_agent` explicitly from its own hook for now.

use async_trait::async_trait;

use crate::agent::AgentHandle;
use crate::errors::StrandsError;

/// Persists an agent's session state. Ports `SessionManager`.
#[async_trait]
pub trait SessionManager: Send + Sync {
    /// Synchronizes the agent's current state (messages, agent state) to the
    /// backing store. Ports `session_manager.sync_agent(agent)`; the readable
    /// agent surfaces are reached through `agent` ([`AgentHandle`]).
    async fn sync_agent(&self, agent: &AgentHandle) -> Result<(), StrandsError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentState, Messages};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct CountingManager {
        syncs: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl SessionManager for CountingManager {
        async fn sync_agent(&self, _agent: &AgentHandle) -> Result<(), StrandsError> {
            self.syncs.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    // A SessionManager's sync_agent runs against the hook handle.
    #[tokio::test]
    async fn sync_agent_is_callable() {
        let syncs = Arc::new(AtomicUsize::new(0));
        let manager = CountingManager {
            syncs: syncs.clone(),
        };
        let handle = AgentHandle::new(AgentState::new(), Messages::default(), None, None);
        manager.sync_agent(&handle).await.unwrap();
        assert_eq!(syncs.load(Ordering::SeqCst), 1);
    }
}
