//! Per-invocation shared state. Ports `InvocationState` from `types/agent.ts`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A per-invocation mutable bag shared across hooks (and, once ported, tools and
/// middleware) within a single agent invocation.
///
/// Ports the TypeScript `InvocationState`: it lets any callback correlate back to
/// the caller's request context (`user_id`, `trace_id`, …) without closure
/// workarounds. The handle is cheap to clone — every clone shares the same
/// underlying map — so each hook event can carry one by value.
#[derive(Clone, Default)]
pub struct InvocationState {
    inner: Arc<Mutex<HashMap<String, serde_json::Value>>>,
}

impl InvocationState {
    /// Creates an empty invocation state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the value stored under `key`, if any.
    pub fn get(&self, key: &str) -> Option<serde_json::Value> {
        self.inner
            .lock()
            .expect("invocation state mutex poisoned")
            .get(key)
            .cloned()
    }

    /// Stores `value` under `key`, replacing any existing value.
    pub fn set(&self, key: impl Into<String>, value: serde_json::Value) {
        self.inner
            .lock()
            .expect("invocation state mutex poisoned")
            .insert(key.into(), value);
    }

    /// Returns `true` if no values are stored.
    pub fn is_empty(&self) -> bool {
        self.inner
            .lock()
            .expect("invocation state mutex poisoned")
            .is_empty()
    }
}

impl std::fmt::Debug for InvocationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Avoid locking in Debug output; the map contents are not part of the
        // stable representation.
        f.debug_struct("InvocationState").finish_non_exhaustive()
    }
}
