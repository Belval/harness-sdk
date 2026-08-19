//! Agent-scoped persisted state and the hook-facing agent handle.
//!
//! Ports `strands.agent.state.AgentState` and the `event.agent` reference that
//! hook callbacks read and mutate.

use std::sync::{Arc, Mutex};

use crate::types::messages::Message;

/// A persisted, agent-scoped key/value store. Ports `AgentState`.
///
/// Unlike [`crate::agent::InvocationState`] (which lives for one invocation),
/// this state persists across invocations for the life of the agent. It is a
/// cheap-clone handle over a shared JSON object, so hook callbacks that receive
/// it through [`AgentHandle`] mutate the same underlying state the agent owns.
///
/// Values are [`serde_json::Value`]s, matching the JSON-serializable contract of
/// the TypeScript/Python `AgentState`.
#[derive(Clone, Default)]
pub struct AgentState {
    inner: Arc<Mutex<serde_json::Map<String, serde_json::Value>>>,
}

impl AgentState {
    /// Creates an empty state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a state seeded from a JSON object.
    ///
    /// # Panics
    /// Panics if `value` is not a JSON object, mirroring `AgentState`'s
    /// requirement that the backing document is an object.
    pub fn from_value(value: serde_json::Value) -> Self {
        let map = match value {
            serde_json::Value::Object(map) => map,
            other => panic!("AgentState must be a JSON object, got {other}"),
        };
        AgentState {
            inner: Arc::new(Mutex::new(map)),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, serde_json::Map<String, serde_json::Value>> {
        self.inner.lock().expect("agent state mutex poisoned")
    }

    /// Returns the value stored under `key`, if any. Ports `state.get(key)`.
    pub fn get(&self, key: &str) -> Option<serde_json::Value> {
        self.lock().get(key).cloned()
    }

    /// Stores `value` under `key`, replacing any existing value. Ports
    /// `state.set(key, value)`.
    pub fn set(&self, key: impl Into<String>, value: serde_json::Value) {
        self.lock().insert(key.into(), value);
    }

    /// Removes `key`, returning the previous value if present. Ports
    /// `state.delete(key)`.
    pub fn delete(&self, key: &str) -> Option<serde_json::Value> {
        self.lock().remove(key)
    }

    /// The keys currently stored, in arbitrary order.
    pub fn keys(&self) -> Vec<String> {
        self.lock().keys().cloned().collect()
    }

    /// Returns `true` if no values are stored.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// Snapshots the whole state as a JSON object. Ports the JSON round-trip.
    pub fn to_value(&self) -> serde_json::Value {
        serde_json::Value::Object(self.lock().clone())
    }
}

impl std::fmt::Debug for AgentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentState")
            .field("keys", &self.lock().len())
            .finish_non_exhaustive()
    }
}

/// The agent's conversation history as a shared, interior-mutable handle. Ports
/// the mutable `agent.messages` list.
///
/// A cheap-clone handle over the message vec, so hook callbacks and conversation
/// managers reached through [`AgentHandle`] read and rewrite the same history the
/// loop drives. Hook dispatch is synchronous, so the loop never holds the lock
/// across a callback.
#[derive(Clone, Default)]
pub struct Messages {
    inner: Arc<Mutex<Vec<Message>>>,
}

impl Messages {
    /// Creates a store seeded with `initial`.
    pub fn new(initial: Vec<Message>) -> Self {
        Messages {
            inner: Arc::new(Mutex::new(initial)),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Message>> {
        self.inner.lock().expect("messages mutex poisoned")
    }

    /// Appends a message to the history.
    pub fn push(&self, message: Message) {
        self.lock().push(message);
    }

    /// The number of messages.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// The message at `index`, cloned.
    pub fn get(&self, index: usize) -> Option<Message> {
        self.lock().get(index).cloned()
    }

    /// The last message, cloned.
    pub fn last(&self) -> Option<Message> {
        self.lock().last().cloned()
    }

    /// A cloned snapshot of the whole history.
    pub fn snapshot(&self) -> Vec<Message> {
        self.lock().clone()
    }

    /// Replaces the entire history — the primitive a conversation manager uses to
    /// reduce or rewrite context.
    pub fn replace(&self, messages: Vec<Message>) {
        *self.lock() = messages;
    }

    /// Mutates the history in place, e.g. to append a cache point to the last
    /// message or drop old ones.
    pub fn update<F: FnOnce(&mut Vec<Message>)>(&self, edit: F) {
        edit(&mut self.lock());
    }
}

impl std::fmt::Debug for Messages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Messages")
            .field("len", &self.lock().len())
            .finish_non_exhaustive()
    }
}

/// The hook-facing view of the agent. Ports the `event.agent` reference.
///
/// Hook callbacks receive this on every lifecycle event to read and mutate
/// agent-scoped surfaces: the persisted [`AgentState`], the conversation
/// [`Messages`], and the model id. It is the seam through which further shared
/// agent surfaces (metrics, conversation manager) are exposed as they are ported.
#[derive(Clone, Debug)]
pub struct AgentHandle {
    state: AgentState,
    messages: Messages,
    model_id: Option<String>,
}

impl AgentHandle {
    pub(crate) fn new(state: AgentState, messages: Messages, model_id: Option<String>) -> Self {
        AgentHandle {
            state,
            messages,
            model_id,
        }
    }

    /// The agent's persisted state. Ports `agent.state`.
    pub fn state(&self) -> &AgentState {
        &self.state
    }

    /// The agent's conversation history. Ports `agent.messages`.
    pub fn messages(&self) -> &Messages {
        &self.messages
    }

    /// The configured model id. Ports `agent.model.get_config()["model_id"]`.
    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // AgentState: get/set/delete round-trip; clones share the same store
    #[test]
    fn get_set_delete_and_sharing() {
        let state = AgentState::new();
        assert!(state.is_empty());
        state.set("_cycle_count", json!(0));
        assert_eq!(state.get("_cycle_count"), Some(json!(0)));

        // A clone is a shared handle — mutations are visible through both.
        let shared = state.clone();
        shared.set("_cycle_count", json!(1));
        assert_eq!(state.get("_cycle_count"), Some(json!(1)));

        assert_eq!(state.delete("_cycle_count"), Some(json!(1)));
        assert_eq!(state.get("_cycle_count"), None);
    }

    // AgentState: seeds from and snapshots to a JSON object
    #[test]
    fn from_and_to_value() {
        let state = AgentState::from_value(json!({ "a": 1, "b": "two" }));
        assert_eq!(state.get("a"), Some(json!(1)));
        assert_eq!(state.to_value(), json!({ "a": 1, "b": "two" }));
    }

    // AgentHandle exposes the shared state, messages, and model id
    #[test]
    fn handle_exposes_shared_surfaces() {
        let state = AgentState::new();
        let messages = Messages::default();
        let handle = AgentHandle::new(state.clone(), messages.clone(), Some("m-1".to_string()));

        handle.state().set("k", json!(true));
        assert_eq!(state.get("k"), Some(json!(true)));

        handle
            .messages()
            .push(crate::types::messages::Message::user("hi"));
        assert_eq!(messages.len(), 1);

        assert_eq!(handle.model_id(), Some("m-1"));
    }

    // Messages: shared push/replace/update reflect across clones
    #[test]
    fn messages_shared_and_mutable() {
        use crate::types::messages::Message;
        let messages = Messages::new(vec![Message::user("first")]);
        let shared = messages.clone();
        shared.push(Message::assistant("second"));
        assert_eq!(messages.len(), 2);
        assert_eq!(messages.last().unwrap().text(), "second");

        // In-place edit and full replace, the conversation-manager primitives.
        messages.update(|history| history.truncate(1));
        assert_eq!(messages.len(), 1);
        messages.replace(vec![Message::user("x"), Message::user("y")]);
        assert_eq!(shared.len(), 2);
    }
}
