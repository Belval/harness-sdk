//! The `SessionManager` plugin (`strands-ts/src/session/session-manager.ts`).

use std::collections::HashMap;
use std::hash::Hasher;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use super::agent_snapshot::{load_agent_snapshot, take_agent_snapshot_json};
use super::storage::{SnapshotLocation, SnapshotStorage};
use super::validation::validate_identifier;
use crate::agent::Agent;
use crate::error::Result;
use crate::hooks::{AfterInvocationEvent, AfterModelCallEvent, InitializedEvent, MessageAddedEvent};
use crate::plugins::Plugin;

/// Session id used when none is configured.
pub const DEFAULT_SESSION_ID: &str = "default-session";

/// Plugin name registered on the agent.
pub(crate) const PLUGIN_NAME: &str = "strands:session-manager";

/// Controls when `snapshot_latest` is saved automatically.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SaveLatestStrategy {
    /// Save after every message is added to the conversation.
    Message,
    /// Save once per agent invocation (default).
    #[default]
    Invocation,
    /// Never save automatically; only `snapshot_trigger` drives persistence.
    Trigger,
}

/// Decides after an invocation whether to write an immutable snapshot.
pub type SnapshotTrigger = Arc<dyn Fn(&Agent) -> bool + Send + Sync>;

/// Optional [`SessionManager`] settings (`SessionManagerConfig` in TS, minus `storage`/`sessionId`).
#[derive(Clone, Default)]
pub struct SessionManagerOptions {
    /// When `snapshot_latest` is written automatically.
    pub save_latest_on: SaveLatestStrategy,
    /// Callback deciding when to add an immutable snapshot.
    pub snapshot_trigger: Option<SnapshotTrigger>,
    /// Write snapshots as compact JSON instead of TS's 2-space pretty form: about a quarter
    /// smaller and faster. Loading accepts both.
    pub compact: bool,
}

impl std::fmt::Debug for SessionManagerOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionManagerOptions")
            .field("save_latest_on", &self.save_latest_on)
            .field("snapshot_trigger", &self.snapshot_trigger.is_some())
            .field("compact", &self.compact)
            .finish()
    }
}

struct Inner {
    session_id: String,
    storage: Arc<dyn SnapshotStorage>,
    options: SessionManagerOptions,
    /// Length of the last snapshot serialized, to size the next one's buffer.
    last_len: AtomicUsize,
    /// Per agent id, the [`fingerprint`] of the latest snapshot this manager wrote. Held across
    /// each save of the latest snapshot, so saves land in the order they were taken.
    saved: tokio::sync::Mutex<HashMap<String, (usize, u64)>>,
}

/// Persists agent state to a [`SnapshotStorage`] so conversations survive across invocations.
///
/// Restores the latest snapshot on `InitializedEvent` and saves it again according to
/// [`SaveLatestStrategy`]. Cheap to clone; clones share one session.
#[derive(Clone)]
pub struct SessionManager {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for SessionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionManager")
            .field("session_id", &self.inner.session_id)
            .field("options", &self.inner.options)
            .finish_non_exhaustive()
    }
}

impl SessionManager {
    /// Manager for `session_id` with default options. Fails if the id is not a valid identifier.
    pub fn new(session_id: &str, storage: Arc<dyn SnapshotStorage>) -> Result<Self> {
        Self::with_options(session_id, storage, SessionManagerOptions::default())
    }

    /// Manager for `session_id` with explicit options.
    pub fn with_options(
        session_id: &str,
        storage: Arc<dyn SnapshotStorage>,
        options: SessionManagerOptions,
    ) -> Result<Self> {
        validate_identifier(session_id)?;
        Ok(Self {
            inner: Arc::new(Inner {
                session_id: session_id.to_owned(),
                storage,
                options,
                last_len: AtomicUsize::new(0),
                saved: Default::default(),
            }),
        })
    }

    /// The session identifier for this session manager.
    pub fn session_id(&self) -> &str {
        &self.inner.session_id
    }

    fn location(&self, agent: &Agent) -> SnapshotLocation {
        SnapshotLocation::agent(&self.inner.session_id, agent.id())
    }

    /// The agent's snapshot as JSON, serialized straight from the agent into one buffer sized
    /// after the last one (measured for the first).
    fn snapshot_json(&self, agent: &Agent) -> Result<Vec<u8>> {
        let last = self.inner.last_len.load(Ordering::Relaxed);
        let capacity = (last > 0).then_some(last + last / 16);
        let json = take_agent_snapshot_json(agent, !self.inner.options.compact, capacity)?;
        self.inner.last_len.store(json.len(), Ordering::Relaxed);
        Ok(json)
    }

    /// Saves a snapshot of the agent's current state. The latest snapshot is not rewritten when
    /// all but its `createdAt` is what this manager last wrote there.
    pub async fn save_snapshot(&self, agent: &Agent, is_latest: bool) -> Result<()> {
        let location = self.location(agent);
        if is_latest {
            let mut saved = self.inner.saved.lock().await;
            let json = self.snapshot_json(agent)?;
            return self.save_latest(&mut saved, &location, &json).await;
        }
        let json = self.snapshot_json(agent)?;
        let snapshot_id = Uuid::now_v7().to_string();
        self.inner
            .storage
            .save_snapshot_json(&location, &snapshot_id, false, &json)
            .await
    }

    /// Writes `json` as `location`'s latest snapshot unless `saved` says it is already there.
    async fn save_latest(
        &self,
        saved: &mut HashMap<String, (usize, u64)>,
        location: &SnapshotLocation,
        json: &[u8],
    ) -> Result<()> {
        let print = fingerprint(json);
        if saved.get(&location.scope_id) == Some(&print) {
            return Ok(());
        }
        saved.remove(&location.scope_id);
        self.inner
            .storage
            .save_snapshot_json(location, "latest", true, json)
            .await?;
        saved.insert(location.scope_id.clone(), print);
        Ok(())
    }

    /// Loads a snapshot (the latest when `snapshot_id` is `None`) into the agent.
    /// Returns `false` when no snapshot exists.
    pub async fn restore_snapshot(&self, agent: &Agent, snapshot_id: Option<&str>) -> Result<bool> {
        let snapshot = self
            .inner
            .storage
            .load_snapshot(&self.location(agent), snapshot_id)
            .await?;
        let Some(snapshot) = snapshot else {
            return Ok(false);
        };
        load_agent_snapshot(agent, &snapshot)?;
        // The agent may now differ from what this manager last saved.
        self.inner.saved.lock().await.remove(agent.id());
        Ok(true)
    }

    /// Immutable snapshot ids for this agent, oldest first.
    pub async fn list_snapshot_ids(
        &self,
        agent: &Agent,
        limit: Option<usize>,
        start_after: Option<&str>,
    ) -> Result<Vec<String>> {
        self.inner
            .storage
            .list_snapshot_ids(&self.location(agent), limit, start_after)
            .await
    }

    /// Deletes all snapshots and manifests for this session from storage.
    pub async fn delete_session(&self) -> Result<()> {
        let mut saved = self.inner.saved.lock().await;
        saved.clear();
        self.inner.storage.delete_session(&self.inner.session_id).await
    }

    /// Restores session state into the agent; called from the `InitializedEvent` hook.
    pub async fn initialize(&self, agent: &Agent) -> Result<()> {
        let had_messages = !agent.messages().is_empty();
        let restored = self.restore_snapshot(agent, None).await?;

        if restored && had_messages {
            tracing::warn!(
                "agent_id=<{}>, session_id=<{}> | agent had existing messages that were overwritten by session restore",
                agent.id(),
                self.inner.session_id
            );
        }

        // Stateful models keep history server-side, so restored messages would drift from the
        // provider's view on the next invocation.
        if restored && agent.model().stateful() && !agent.messages().is_empty() {
            tracing::debug!(
                "agent_id=<{}>, message_count=<{}> | discarding restored messages for stateful model",
                agent.id(),
                agent.messages().len()
            );
            agent.with_messages(Vec::clear);
        }
        Ok(())
    }

    /// Captures one snapshot and writes it to both immutable history and `snapshot_latest`.
    async fn save_immutable_and_latest(&self, agent: &Agent) -> Result<()> {
        let mut saved = self.inner.saved.lock().await;
        let json = self.snapshot_json(agent)?;
        let location = self.location(agent);
        let snapshot_id = Uuid::now_v7().to_string();
        tokio::try_join!(
            self.inner
                .storage
                .save_snapshot_json(&location, &snapshot_id, false, &json),
            self.save_latest(&mut saved, &location, &json),
        )?;
        Ok(())
    }

    async fn on_after_invocation(&self, agent: &Agent) -> Result<()> {
        if self.inner.options.save_latest_on != SaveLatestStrategy::Trigger {
            self.save_snapshot(agent, true).await?;
        }
        let fires = self
            .inner
            .options
            .snapshot_trigger
            .as_ref()
            .is_some_and(|trigger| trigger(agent));
        if fires {
            self.save_immutable_and_latest(agent).await?;
        }
        Ok(())
    }

    /// Saves a snapshot when a message was redacted after a model call, so guardrail redactions
    /// are persisted immediately.
    async fn on_after_model_call(&self, event: &AfterModelCallEvent) -> Result<()> {
        if event
            .stop_data
            .as_ref()
            .is_some_and(|data| data.redaction.is_some())
        {
            self.save_snapshot(&event.agent, true).await?;
        }
        Ok(())
    }
}

/// What decides whether a snapshot changed: its JSON's length and hash, both without the
/// `createdAt` value every save renews.
fn fingerprint(json: &[u8]) -> (usize, u64) {
    const KEY: &[u8] = b"\"createdAt\":";
    // It follows only `scope` and `schemaVersion`, near the start.
    let head = &json[..json.len().min(256)];
    let value = head.windows(KEY.len()).position(|w| w == KEY).and_then(|at| {
        let start = at + KEY.len();
        let open = start + head[start..].iter().position(|&b| b == b'"')?;
        let close = open + 1 + head[open + 1..].iter().position(|&b| b == b'"')?;
        Some((start, close + 1))
    });
    let (start, end) = value.unwrap_or((json.len(), json.len()));
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hasher.write(&json[..start]);
    hasher.write(&json[end..]);
    (json.len() - (end - start), hasher.finish())
}

#[async_trait]
impl Plugin for SessionManager {
    fn name(&self) -> &str {
        PLUGIN_NAME
    }

    async fn init_agent(&self, agent: &Agent) -> Result<()> {
        let hooks = agent.hooks();

        let manager = self.clone();
        hooks.add_callback_boxed::<InitializedEvent, _>(move |event| {
            let manager = manager.clone();
            Box::pin(async move { manager.initialize(&event.agent).await })
        });

        if self.inner.options.save_latest_on == SaveLatestStrategy::Message {
            let manager = self.clone();
            hooks.add_callback_boxed::<MessageAddedEvent, _>(move |event| {
                let manager = manager.clone();
                Box::pin(async move { manager.save_snapshot(&event.agent, true).await })
            });
        }

        // 'trigger' is an explicit opt-out from auto-saves, so the caller's snapshot_trigger
        // stays in control there.
        if self.inner.options.save_latest_on != SaveLatestStrategy::Trigger {
            let manager = self.clone();
            hooks.add_callback_boxed::<AfterModelCallEvent, _>(move |event| {
                let manager = manager.clone();
                Box::pin(async move { manager.on_after_model_call(event).await })
            });
        }

        let manager = self.clone();
        hooks.add_callback_boxed::<AfterInvocationEvent, _>(move |event| {
            let manager = manager.clone();
            Box::pin(async move { manager.on_after_invocation(&event.agent).await })
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::test_agent_for_session;
    use crate::hooks::{ModelStopData, Redaction};
    use crate::models::{Model, ModelStreamEvent, StreamOptions};
    use crate::session::test_support::MockSnapshotStorage;
    use crate::session::types::{Scope, SNAPSHOT_SCHEMA_VERSION};
    use crate::types::{Message, StopReason};
    use futures::stream::BoxStream;
    use serde_json::{json, Map};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn mock_storage() -> Arc<MockSnapshotStorage> {
        Arc::new(MockSnapshotStorage::new())
    }

    fn new_manager(
        session_id: &str,
        storage: &Arc<MockSnapshotStorage>,
        options: SessionManagerOptions,
    ) -> SessionManager {
        SessionManager::with_options(session_id, storage.clone(), options).unwrap()
    }

    fn options(save_latest_on: SaveLatestStrategy) -> SessionManagerOptions {
        SessionManagerOptions {
            save_latest_on,
            ..Default::default()
        }
    }

    fn trigger_options(
        save_latest_on: SaveLatestStrategy,
        trigger: impl Fn(&Agent) -> bool + Send + Sync + 'static,
    ) -> SessionManagerOptions {
        SessionManagerOptions {
            save_latest_on,
            snapshot_trigger: Some(Arc::new(trigger)),
            ..Default::default()
        }
    }

    fn test_snapshot() -> crate::session::Snapshot {
        crate::session::Snapshot {
            scope: Scope::Agent,
            schema_version: SNAPSHOT_SCHEMA_VERSION.to_owned(),
            created_at: "2024-01-01T00:00:00.000Z".to_owned(),
            data: json!({
                "messages": [],
                "state": { "testKey": "testValue" },
                "systemPrompt": "You are a test assistant",
            })
            .as_object()
            .unwrap()
            .clone(),
            app_data: Map::new(),
        }
    }

    async fn load_latest(
        storage: &Arc<MockSnapshotStorage>,
        session_id: &str,
        agent_id: &str,
    ) -> Option<crate::session::Snapshot> {
        storage
            .load_snapshot(&SnapshotLocation::agent(session_id, agent_id), None)
            .await
            .unwrap()
    }

    async fn history_ids(
        storage: &Arc<MockSnapshotStorage>,
        session_id: &str,
        agent_id: &str,
    ) -> Vec<String> {
        storage
            .list_snapshot_ids(&SnapshotLocation::agent(session_id, agent_id), None, None)
            .await
            .unwrap()
    }

    async fn invoke_initialized(agent: &Agent) {
        let mut event = InitializedEvent { agent: agent.clone() };
        agent.hooks().invoke_callbacks(&mut event).await.unwrap();
    }

    async fn invoke_after_invocation(agent: &Agent) {
        let mut event = AfterInvocationEvent {
            agent: agent.clone(),
            invocation_state: Default::default(),
            resume: None,
            stop_reason: Some(crate::types::StopReason::EndTurn),
        };
        agent.hooks().invoke_callbacks(&mut event).await.unwrap();
    }

    async fn invoke_message_added(agent: &Agent) {
        let mut event = MessageAddedEvent {
            agent: agent.clone(),
            message: Message::user("test"),
            invocation_state: Default::default(),
        };
        agent.hooks().invoke_callbacks(&mut event).await.unwrap();
    }

    async fn invoke_after_model_call(agent: &Agent, redacted: bool) {
        let mut event = AfterModelCallEvent {
            agent: agent.clone(),
            model: agent.model(),
            stop_data: Some(ModelStopData {
                message: Message::assistant("Response"),
                stop_reason: StopReason::EndTurn,
                usage: None,
                redaction: redacted.then(|| Redaction {
                    user_message: "[User input redacted.]".to_owned(),
                }),
            }),
            error: None,
            invocation_state: Default::default(),
            attempt_count: 1,
            retry: false,
        };
        agent.hooks().invoke_callbacks(&mut event).await.unwrap();
    }

    #[tokio::test]
    async fn defaults_save_latest_on_to_invocation() {
        let storage = mock_storage();
        let manager = SessionManager::new("test-default", storage.clone()).unwrap();
        let agent = test_agent_for_session("agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_after_invocation(&agent).await;

        assert!(load_latest(&storage, "test-default", "agent").await.is_some());
    }

    #[tokio::test]
    async fn exposes_name_and_session_id() {
        let manager = SessionManager::new("test-session", mock_storage()).unwrap();
        assert_eq!(manager.name(), "strands:session-manager");
        assert_eq!(manager.session_id(), "test-session");
    }

    #[tokio::test]
    async fn rejects_an_invalid_session_id() {
        let err = SessionManager::new("Invalid/Session", mock_storage()).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Identifier 'Invalid/Session' can only contain lowercase letters, numbers, hyphens, and underscores"
        );
    }

    #[tokio::test]
    async fn saves_snapshot_latest_when_is_latest_is_true() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");

        manager.save_snapshot(&agent, true).await.unwrap();

        let snapshot = load_latest(&storage, "test-session", "test-agent").await.unwrap();
        assert_eq!(snapshot.scope, Scope::Agent);
    }

    #[tokio::test]
    async fn saves_immutable_snapshot_when_is_latest_is_false() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");

        manager.save_snapshot(&agent, false).await.unwrap();

        assert!(!history_ids(&storage, "test-session", "test-agent")
            .await
            .is_empty());
    }

    #[tokio::test]
    async fn allocates_unique_snapshot_ids() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");

        for _ in 0..3 {
            manager.save_snapshot(&agent, false).await.unwrap();
        }

        assert_eq!(history_ids(&storage, "test-session", "test-agent").await.len(), 3);
    }

    #[tokio::test]
    async fn returns_empty_array_when_no_snapshots_exist() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");

        assert!(manager
            .list_snapshot_ids(&agent, None, None)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn returns_snapshot_ids_for_the_target_agent() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");
        let other = test_agent_for_session("other-agent");

        manager.save_snapshot(&agent, false).await.unwrap();
        manager.save_snapshot(&agent, false).await.unwrap();
        manager.save_snapshot(&other, false).await.unwrap();

        assert_eq!(
            manager.list_snapshot_ids(&agent, None, None).await.unwrap().len(),
            2
        );
    }

    #[tokio::test]
    async fn does_not_return_latest_snapshot_id() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");

        manager.save_snapshot(&agent, true).await.unwrap();

        assert!(manager
            .list_snapshot_ids(&agent, None, None)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn forwards_limit_parameter() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");
        for _ in 0..3 {
            manager.save_snapshot(&agent, false).await.unwrap();
        }

        assert_eq!(
            manager
                .list_snapshot_ids(&agent, Some(2), None)
                .await
                .unwrap()
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn forwards_start_after_parameter() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");
        manager.save_snapshot(&agent, false).await.unwrap();
        manager.save_snapshot(&agent, false).await.unwrap();

        let all = manager.list_snapshot_ids(&agent, None, None).await.unwrap();
        let page2 = manager
            .list_snapshot_ids(&agent, None, Some(&all[0]))
            .await
            .unwrap();

        assert_eq!(page2, [all[1].clone()]);
    }

    #[tokio::test]
    async fn restores_snapshot_latest_when_no_snapshot_id_provided() {
        let storage = mock_storage();
        storage
            .save_snapshot(
                &SnapshotLocation::agent("test-session", "test-agent"),
                "latest",
                true,
                &test_snapshot(),
            )
            .await
            .unwrap();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");

        assert!(manager.restore_snapshot(&agent, None).await.unwrap());
    }

    #[tokio::test]
    async fn restores_specific_snapshot_by_id() {
        let storage = mock_storage();
        storage
            .save_snapshot(
                &SnapshotLocation::agent("test-session", "test-agent"),
                "5",
                false,
                &test_snapshot(),
            )
            .await
            .unwrap();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");

        assert!(manager.restore_snapshot(&agent, Some("5")).await.unwrap());
    }

    #[tokio::test]
    async fn returns_false_when_snapshot_not_found() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");

        assert!(!manager.restore_snapshot(&agent, Some("999")).await.unwrap());
    }

    #[tokio::test]
    async fn loads_snapshot_latest_on_initialization() {
        let storage = mock_storage();
        let restored = Message::user("restored");
        let mut snapshot = test_snapshot();
        snapshot.data.insert(
            "messages".to_owned(),
            serde_json::to_value(vec![restored.clone()]).unwrap(),
        );
        storage
            .save_snapshot(
                &SnapshotLocation::agent("test-session", "test-agent"),
                "latest",
                true,
                &snapshot,
            )
            .await
            .unwrap();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_initialized(&agent).await;

        assert_eq!(agent.messages(), vec![restored]);
        assert_eq!(agent.state().get("testKey"), Some(json!("testValue")));
    }

    #[tokio::test]
    async fn handles_missing_snapshot_gracefully() {
        let storage = mock_storage();
        let manager = new_manager("new-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_initialized(&agent).await;

        assert!(agent.messages().is_empty());
    }

    #[tokio::test]
    async fn warns_when_snapshot_restore_overwrites_existing_messages() {
        let storage = mock_storage();
        storage
            .save_snapshot(
                &SnapshotLocation::agent("test-session", "test-agent"),
                "latest",
                true,
                &test_snapshot(),
            )
            .await
            .unwrap();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");
        agent.set_messages(vec![Message::user("existing")]);
        manager.init_agent(&agent).await.unwrap();

        invoke_initialized(&agent).await;

        // The snapshot's empty message list replaced the agent's own messages (TS logs a warning).
        assert!(agent.messages().is_empty());
    }

    #[tokio::test]
    async fn does_not_warn_when_restoring_into_agent_with_no_messages() {
        let storage = mock_storage();
        storage
            .save_snapshot(
                &SnapshotLocation::agent("test-session", "test-agent"),
                "latest",
                true,
                &test_snapshot(),
            )
            .await
            .unwrap();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_initialized(&agent).await;

        assert!(agent.messages().is_empty());
    }

    #[tokio::test]
    async fn discards_restored_messages_for_a_stateful_model() {
        struct StatefulModel;
        #[async_trait]
        impl Model for StatefulModel {
            fn model_id(&self) -> Option<String> {
                None
            }
            fn stateful(&self) -> bool {
                true
            }
            fn stream<'a>(
                &'a self,
                _messages: &'a [Message],
                _options: &'a StreamOptions,
            ) -> BoxStream<'a, Result<ModelStreamEvent>> {
                Box::pin(futures::stream::empty())
            }
        }

        let storage = mock_storage();
        let mut snapshot = test_snapshot();
        snapshot.data.insert(
            "messages".to_owned(),
            serde_json::to_value(vec![Message::user("restored")]).unwrap(),
        );
        storage
            .save_snapshot(
                &SnapshotLocation::agent("test-session", "test-agent"),
                "latest",
                true,
                &snapshot,
            )
            .await
            .unwrap();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");
        agent.set_model(StatefulModel);
        manager.init_agent(&agent).await.unwrap();

        invoke_initialized(&agent).await;

        assert!(agent.messages().is_empty());
        assert_eq!(agent.state().get("testKey"), Some(json!("testValue")));
    }

    #[tokio::test]
    async fn saves_snapshot_latest_when_save_latest_on_is_message() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, options(SaveLatestStrategy::Message));
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_message_added(&agent).await;

        assert!(load_latest(&storage, "test-session", "test-agent")
            .await
            .is_some());
    }

    #[tokio::test]
    async fn does_not_save_on_message_added_when_save_latest_on_is_invocation() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, options(SaveLatestStrategy::Invocation));
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        assert_eq!(agent.hooks().count::<MessageAddedEvent>(), 0);

        invoke_message_added(&agent).await;

        assert!(load_latest(&storage, "test-session", "test-agent")
            .await
            .is_none());
    }

    #[tokio::test]
    async fn saves_snapshot_latest_on_after_invocation_when_save_latest_on_is_invocation() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, options(SaveLatestStrategy::Invocation));
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_after_invocation(&agent).await;

        assert!(load_latest(&storage, "test-session", "test-agent")
            .await
            .is_some());
    }

    #[tokio::test]
    async fn saves_snapshot_latest_on_after_invocation_when_save_latest_on_is_message() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, options(SaveLatestStrategy::Message));
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_after_invocation(&agent).await;

        assert!(load_latest(&storage, "test-session", "test-agent")
            .await
            .is_some());
    }

    #[tokio::test]
    async fn does_not_save_snapshot_latest_when_save_latest_on_is_trigger() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, options(SaveLatestStrategy::Trigger));
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_after_invocation(&agent).await;

        assert!(load_latest(&storage, "test-session", "test-agent")
            .await
            .is_none());
    }

    #[tokio::test]
    async fn creates_immutable_snapshot_when_trigger_returns_true() {
        let storage = mock_storage();
        let manager = new_manager(
            "test-session",
            &storage,
            trigger_options(SaveLatestStrategy::Trigger, |_| true),
        );
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_after_invocation(&agent).await;

        assert_eq!(history_ids(&storage, "test-session", "test-agent").await.len(), 1);
    }

    #[tokio::test]
    async fn does_not_create_immutable_snapshot_when_trigger_returns_false() {
        let storage = mock_storage();
        let manager = new_manager(
            "test-session",
            &storage,
            trigger_options(SaveLatestStrategy::Trigger, |_| false),
        );
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_after_invocation(&agent).await;

        assert!(history_ids(&storage, "test-session", "test-agent")
            .await
            .is_empty());
    }

    #[tokio::test]
    async fn provides_agent_data_to_trigger() {
        let storage = mock_storage();
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        let manager = new_manager(
            "test-session",
            &storage,
            trigger_options(SaveLatestStrategy::Trigger, |agent| {
                CALLS.fetch_add(1, Ordering::SeqCst);
                assert_eq!(agent.id(), "test-agent");
                assert_eq!(agent.messages().len(), 1);
                false
            }),
        );
        let agent = test_agent_for_session("test-agent");
        agent.set_messages(vec![Message::user("one")]);
        manager.init_agent(&agent).await.unwrap();

        invoke_after_invocation(&agent).await;

        assert_eq!(CALLS.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn saves_both_immutable_and_latest_when_trigger_fires() {
        let storage = mock_storage();
        let manager = new_manager(
            "test-session",
            &storage,
            trigger_options(SaveLatestStrategy::Trigger, |_| true),
        );
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_after_invocation(&agent).await;

        assert_eq!(history_ids(&storage, "test-session", "test-agent").await.len(), 1);
        assert!(load_latest(&storage, "test-session", "test-agent")
            .await
            .is_some());
        assert_eq!(storage.saves(), 2);
    }

    #[tokio::test]
    async fn trigger_based_on_message_count_via_agent_data() {
        let storage = mock_storage();
        let manager = new_manager(
            "test-session",
            &storage,
            trigger_options(SaveLatestStrategy::Trigger, |agent| agent.messages().len() >= 2),
        );
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_after_invocation(&agent).await;
        assert!(history_ids(&storage, "test-session", "test-agent")
            .await
            .is_empty());

        agent.set_messages(vec![Message::user("a"), Message::assistant("b")]);
        invoke_after_invocation(&agent).await;
        assert_eq!(history_ids(&storage, "test-session", "test-agent").await.len(), 1);
    }

    #[tokio::test]
    async fn trigger_based_on_agent_state_via_agent_data() {
        let storage = mock_storage();
        let manager = new_manager(
            "test-session",
            &storage,
            trigger_options(SaveLatestStrategy::Trigger, |agent| {
                agent.state().get("checkpoint") == Some(json!(true))
            }),
        );
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_after_invocation(&agent).await;
        assert!(history_ids(&storage, "test-session", "test-agent")
            .await
            .is_empty());

        agent.state().set("checkpoint", json!(true));
        invoke_after_invocation(&agent).await;
        assert_eq!(history_ids(&storage, "test-session", "test-agent").await.len(), 1);
    }

    #[tokio::test]
    async fn handles_complete_session_lifecycle() {
        let storage = mock_storage();
        let manager = new_manager(
            "lifecycle-test",
            &storage,
            trigger_options(SaveLatestStrategy::Invocation, |_| true),
        );
        let agent = test_agent_for_session("agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_initialized(&agent).await;
        for _ in 0..3 {
            invoke_after_invocation(&agent).await;
        }

        assert!(load_latest(&storage, "lifecycle-test", "agent").await.is_some());
        assert_eq!(history_ids(&storage, "lifecycle-test", "agent").await.len(), 3);
    }

    #[tokio::test]
    async fn supports_resuming_from_immutable_snapshot() {
        let storage = mock_storage();
        let manager = new_manager(
            "resume-test",
            &storage,
            trigger_options(SaveLatestStrategy::Trigger, |agent| agent.messages().len() == 2),
        );
        let agent = test_agent_for_session("agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_initialized(&agent).await;
        agent.set_messages(vec![Message::user("a"), Message::assistant("b")]);
        invoke_after_invocation(&agent).await;

        let ids = history_ids(&storage, "resume-test", "agent").await;
        assert_eq!(ids.len(), 1);

        let new_agent = test_agent_for_session("agent");
        let resumed = new_manager("resume-test", &storage, SessionManagerOptions::default());
        resumed.init_agent(&new_agent).await.unwrap();
        invoke_initialized(&new_agent).await;
        resumed.restore_snapshot(&new_agent, Some(&ids[0])).await.unwrap();

        assert_eq!(new_agent.messages(), agent.messages());
    }

    #[tokio::test]
    async fn saves_snapshot_latest_when_save_latest_on_is_message_and_redaction_occurred() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, options(SaveLatestStrategy::Message));
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_after_model_call(&agent, true).await;

        assert!(load_latest(&storage, "test-session", "test-agent")
            .await
            .is_some());
    }

    #[tokio::test]
    async fn does_not_save_when_save_latest_on_is_message_but_no_redaction_occurred() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, options(SaveLatestStrategy::Message));
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_after_model_call(&agent, false).await;

        assert!(load_latest(&storage, "test-session", "test-agent")
            .await
            .is_none());
    }

    #[tokio::test]
    async fn saves_snapshot_latest_on_redaction_for_each_auto_save_strategy() {
        for strategy in [SaveLatestStrategy::Invocation, SaveLatestStrategy::Message] {
            let storage = mock_storage();
            let manager = new_manager("test-session", &storage, options(strategy));
            let agent = test_agent_for_session("test-agent");
            manager.init_agent(&agent).await.unwrap();

            invoke_after_model_call(&agent, true).await;

            assert!(
                load_latest(&storage, "test-session", "test-agent")
                    .await
                    .is_some(),
                "{strategy:?}"
            );
        }
    }

    #[tokio::test]
    async fn does_not_register_after_model_call_hook_when_save_latest_on_is_trigger() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, options(SaveLatestStrategy::Trigger));
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        assert_eq!(agent.hooks().count::<AfterModelCallEvent>(), 0);
        assert_eq!(agent.hooks().count::<InitializedEvent>(), 1);
        assert_eq!(agent.hooks().count::<AfterInvocationEvent>(), 1);
    }

    #[tokio::test]
    async fn does_not_save_on_after_model_call_without_redaction_under_invocation() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, options(SaveLatestStrategy::Invocation));
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();

        invoke_after_model_call(&agent, false).await;

        assert!(load_latest(&storage, "test-session", "test-agent")
            .await
            .is_none());
    }

    #[tokio::test]
    async fn delete_session_removes_this_session_only() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let other = new_manager("other-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");
        manager.save_snapshot(&agent, true).await.unwrap();
        other.save_snapshot(&agent, true).await.unwrap();

        manager.delete_session().await.unwrap();

        assert!(load_latest(&storage, "test-session", "test-agent")
            .await
            .is_none());
        assert!(load_latest(&storage, "other-session", "test-agent")
            .await
            .is_some());
    }

    #[tokio::test]
    async fn storage_errors_propagate_from_hooks() {
        let storage = mock_storage();
        let manager = new_manager("test-session", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");
        manager.init_agent(&agent).await.unwrap();
        *storage.should_throw_errors.lock() = true;

        let mut event = AfterInvocationEvent {
            agent: agent.clone(),
            invocation_state: Default::default(),
            resume: None,
            stop_reason: Some(crate::types::StopReason::EndTurn),
        };
        let err = agent.hooks().invoke_callbacks(&mut event).await.unwrap_err();

        assert_eq!(err.name(), "SessionError");
    }

    #[tokio::test]
    async fn rewrites_the_latest_snapshot_only_when_more_than_its_time_changed() {
        let storage = mock_storage();
        let manager = new_manager("s", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");
        agent.set_messages(vec![Message::user("one")]);

        manager.save_snapshot(&agent, true).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        manager.save_snapshot(&agent, true).await.unwrap();
        assert_eq!(storage.saves(), 1, "unchanged but for createdAt");

        agent.with_messages(|m| m.push(Message::assistant("two")));
        manager.save_snapshot(&agent, true).await.unwrap();
        agent.state().set("k", json!(1));
        manager.save_snapshot(&agent, true).await.unwrap();
        assert_eq!(storage.saves(), 3, "messages, then state");
        let latest = load_latest(&storage, "s", "test-agent").await.unwrap();
        assert_eq!(latest.data["state"], json!({ "k": 1 }));
        assert_eq!(latest.data["messages"].as_array().unwrap().len(), 2);

        // Immutable snapshots are always written; another agent has its own latest.
        manager.save_snapshot(&agent, false).await.unwrap();
        manager
            .save_snapshot(&test_agent_for_session("other"), true)
            .await
            .unwrap();
        assert_eq!(storage.saves(), 5);
    }

    #[tokio::test]
    async fn a_turn_saves_once_per_message_and_not_again_after_the_invocation() {
        use crate::testing::{MockMessageModel, MockResponse};
        let storage = mock_storage();
        let manager = new_manager("s", &storage, options(SaveLatestStrategy::Message));
        let model = MockMessageModel::new(vec![MockResponse::message(Message::assistant("hi"))]);
        let agent = Agent::builder()
            .model(model)
            .session_manager(manager)
            .build()
            .unwrap();

        agent.invoke("hello").await.unwrap();

        assert_eq!(storage.saves(), 2, "the user message and the reply");
        let latest = load_latest(&storage, "s", agent.id()).await.unwrap();
        assert_eq!(latest.data["messages"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn rewrites_an_unchanged_snapshot_after_a_failed_save_a_restore_or_a_delete() {
        let storage = mock_storage();
        let manager = new_manager("s", &storage, SessionManagerOptions::default());
        let agent = test_agent_for_session("test-agent");
        manager.save_snapshot(&agent, true).await.unwrap();

        agent.state().set("k", json!(1));
        *storage.should_throw_errors.lock() = true;
        assert!(manager.save_snapshot(&agent, true).await.is_err());
        *storage.should_throw_errors.lock() = false;
        manager.save_snapshot(&agent, true).await.unwrap();
        assert_eq!(storage.saves(), 2, "the failed save is retried");

        assert!(manager.restore_snapshot(&agent, None).await.unwrap());
        manager.save_snapshot(&agent, true).await.unwrap();
        assert_eq!(storage.saves(), 3, "after a restore");

        manager.delete_session().await.unwrap();
        manager.save_snapshot(&agent, true).await.unwrap();
        assert_eq!(storage.saves(), 4, "after a delete");
        assert!(load_latest(&storage, "s", "test-agent").await.is_some());
    }

    #[test]
    fn a_fingerprint_leaves_out_only_the_created_at_value() {
        let mut snapshot = test_snapshot();
        let print = |snapshot: &crate::session::Snapshot, pretty: bool| {
            fingerprint(&if pretty {
                serde_json::to_vec_pretty(snapshot).unwrap()
            } else {
                serde_json::to_vec(snapshot).unwrap()
            })
        };
        for pretty in [true, false] {
            let before = print(&snapshot, pretty);
            snapshot.created_at = "2030-12-31T23:59:59.999Z".to_owned();
            assert_eq!(print(&snapshot, pretty), before);
            snapshot.schema_version = "1.1".to_owned();
            assert_ne!(print(&snapshot, pretty), before);
            snapshot = test_snapshot();
            snapshot.data.insert("state".to_owned(), json!({}));
            assert_ne!(print(&snapshot, pretty), before);
            snapshot = test_snapshot();
        }
    }

    #[tokio::test]
    async fn compact_and_pretty_snapshots_both_load() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn SnapshotStorage> = Arc::new(crate::session::FileStorage::new(dir.path()));
        let pretty = SessionManager::new("s", Arc::clone(&storage)).unwrap();
        let compact = SessionManagerOptions {
            compact: true,
            ..Default::default()
        };
        let compact = SessionManager::with_options("s", storage, compact).unwrap();
        let agent = test_agent_for_session("test-agent");
        agent.set_messages(vec![Message::user("hello\nthere"), Message::assistant("hi")]);
        agent.state().set("k", json!([1, "two"]));
        let path = dir
            .path()
            .join("s/scopes/agent/test-agent/snapshots/snapshot_latest.json");

        for (saver, is_pretty) in [(&pretty, true), (&compact, false)] {
            saver.save_snapshot(&agent, true).await.unwrap();
            let text = std::fs::read_to_string(&path).unwrap();
            assert_eq!(text.contains('\n'), is_pretty, "{text}");
            for loader in [&pretty, &compact] {
                let target = test_agent_for_session("test-agent");
                assert!(loader.restore_snapshot(&target, None).await.unwrap());
                assert_eq!(target.messages(), agent.messages());
                assert_eq!(target.state().to_json(), json!({ "k": [1, "two"] }));
            }
        }
    }
}
