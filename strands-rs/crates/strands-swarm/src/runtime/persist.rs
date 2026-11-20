//! Saving the swarm to `swarm.json` and restoring it on resume.

use super::*;

impl Swarm {
    /// Reopen a saved run: restore every agent with its conversation and unread mail, wake those
    /// that were mid-turn, and keep cancelled ones visible but inert.
    pub async fn resume(
        mut config: SwarmConfig,
        state: SwarmState,
    ) -> Result<(Self, mpsc::UnboundedReceiver<SwarmEvent>)> {
        // Without `--budget-tokens` the last one given stays the reference, so a later start that
        // replays it (a supervisor's saved arguments) does not undo a `/budget` change either.
        config.budget_tokens = config.budget_tokens.or(state.budget_flag);
        let (swarm, events) = Self::build(config);
        swarm.inner.next_id.store(state.next_id, Ordering::SeqCst);
        *swarm.inner.max_agents.lock() = state.max_agents;
        *swarm.inner.max_calls.lock() = state.max_calls;
        *swarm.inner.host.max_load.lock() = state.max_load;
        *swarm.inner.host.min_free_mem.lock() = state.min_free_mem;
        *swarm.inner.priority.lock() = state.priority.clone();
        let budget = resumed_budget(swarm.inner.config.budget_tokens, &state);
        *swarm.inner.budget.lock() = budget;
        if budget.is_some() {
            swarm.emit(SwarmEvent::Budget { tokens: budget });
        }
        swarm.journal_event("resume", serde_json::json!({ "agents": state.agents.len() }));
        if let Some(dir) = swarm.inner.config.state_dir.clone() {
            swarm.kill_cut_calls(&dir, &state).await;
        }
        if let (Some(from), Some(to)) = (state.commit.clone(), swarm.inner.config.commit.clone()) {
            if from != to {
                let changes = crate::version::changes(&swarm.inner.config.source, &from, &to).await;
                swarm.emit(SwarmEvent::Reloaded { from, to, changes });
            }
        }
        swarm.prune_stashes(&state.agents).await;
        for agent in state.agents {
            swarm.restore_agent(agent);
        }
        swarm.prune_cancelled_sessions().await;
        swarm.archive_old_tasks().await;
        swarm.publish_cancelled_history();
        // Fires missed while the run was down are due now: each fires once.
        if !state.schedules.is_empty() {
            *swarm.inner.schedules.lock() = state.schedules;
            swarm.start_scheduler();
        }
        swarm.start_queued();
        swarm.deliver_supervisor_notice().await;
        swarm.persist();
        swarm.collect_in_background();
        swarm.tidy_landings_in_background();
        swarm.watch_jobs();
        swarm.watch_dormant();
        Ok((swarm, events))
    }

    /// Keep the whole task of cancelled `node` in its session directory, so that `swarm.json`
    /// can hold only its prefix ([`AgentState::task_archived`]). Left as it was if that fails.
    pub(super) fn archive_task(&self, node: &Node) {
        let Some(dir) = &self.inner.config.state_dir else {
            return;
        };
        if node.task_archived.load(Ordering::SeqCst) {
            return;
        }
        let sessions = dir.join(SESSIONS_DIR);
        match crate::state::archive_task(&sessions, &node.id, &node.task) {
            Ok(archived) => node.task_archived.store(archived, Ordering::SeqCst),
            Err(error) => tracing::warn!(agent = %node.id, %error, "cannot save a cancelled agent's task"),
        }
    }

    /// On resume, [`Swarm::archive_task`] for the cancelled agents of runs saved before. One
    /// whose session was removed has nothing to keep it in, so it keeps its prefix. A cancel time
    /// never recorded is taken from the session directory first, as the file would change it.
    async fn archive_old_tasks(&self) {
        let nodes: Vec<Arc<Node>> = self
            .nodes()
            .into_iter()
            .filter(|node| {
                *node.status.lock() == AgentStatus::Cancelled && !node.task_archived.load(Ordering::SeqCst)
            })
            .collect();
        let Some(dir) = self.inner.config.state_dir.clone() else {
            return;
        };
        if nodes.is_empty() {
            return;
        }
        let swarm = self.clone();
        let now = self.now().timestamp();
        let _ = tokio::task::spawn_blocking(move || {
            for node in nodes {
                if node.conversation_removed.load(Ordering::Relaxed) > 0 {
                    node.task_archived.store(true, Ordering::SeqCst);
                    continue;
                }
                if node.cancelled_at.load(Ordering::Relaxed) == 0 {
                    let at = super::retention::newest(&dir.join(SESSIONS_DIR).join(&node.id));
                    let at = Some(at).filter(|at| *at > 0).unwrap_or(now);
                    node.cancelled_at.store(at, Ordering::Relaxed);
                }
                swarm.archive_task(&node);
            }
        })
        .await;
    }

    /// Cancelled agents get no actor on resume, so nothing restores their session. Read each
    /// one's saved conversation in the background and publish its tail ([`SwarmEvent::History`]),
    /// so a viewer still shows what they did. An agent whose session was removed
    /// ([`Swarm::prune_cancelled_sessions`]) publishes [`SwarmEvent::ConversationRemoved`] instead.
    fn publish_cancelled_history(&self) {
        if !self.inner.config.cancelled_history {
            return;
        }
        let Some(dir) = self
            .inner
            .config
            .state_dir
            .as_ref()
            .map(|dir| dir.join(SESSIONS_DIR))
        else {
            return;
        };
        let cancelled: Vec<(String, u64)> = self
            .inner
            .agents
            .lock()
            .values()
            .filter(|node| *node.status.lock() == AgentStatus::Cancelled)
            .map(|node| (node.id.clone(), node.conversation_removed.load(Ordering::Relaxed)))
            .collect();
        if cancelled.is_empty() {
            return;
        }
        let swarm = self.clone();
        self.track(tokio::spawn(async move {
            let storage = strands_agents::FileStorage::new(dir);
            for (id, removed) in cancelled {
                if removed > 0 {
                    swarm.emit(SwarmEvent::ConversationRemoved {
                        id,
                        after_secs: removed,
                    });
                    continue;
                }
                match saved_messages(&storage, &id).await {
                    Ok(messages) => {
                        if let Some(history) = history_event(&id, &messages) {
                            swarm.emit(history);
                        }
                    }
                    Err(error) => tracing::warn!(agent = %id, %error, "saved conversation unreadable"),
                }
            }
        }));
    }

    pub(super) fn restore_agent(&self, saved: AgentState) {
        let parent = self.node(&saved.parent);
        let cancel = parent
            .as_ref()
            .map_or(&self.inner.root_cancel, |node| &node.cancel)
            .child_token();
        // An agent that was resting stays so until its next message; only one that was mid-turn
        // starts again on its own.
        let status = match saved.status.as_str() {
            "cancelled" => AgentStatus::Cancelled,
            "queued" => AgentStatus::Queued,
            "idle" => AgentStatus::Idle,
            "errored" => AgentStatus::Errored(RESTORED_ERROR.to_owned()),
            _ => AgentStatus::Starting,
        };
        let interrupted = matches!(saved.status.as_str(), "running" | "starting");
        let ran = saved.running_calls;
        let mailbox = saved.mailbox.into_iter().map(mail_from_state).collect();
        // Runs saved before private target dirs existed get one on resume.
        let env = match saved.branch.is_some() && saved.env.is_empty() {
            true => self.worktree_env(&saved.id),
            false => saved.env,
        };
        let sessions = self
            .inner
            .config
            .state_dir
            .as_ref()
            .map(|dir| dir.join(SESSIONS_DIR));
        let task = match (&sessions, saved.task_archived) {
            (Some(sessions), true) => crate::state::load_task(sessions, &saved.id, saved.task),
            _ => saved.task,
        };
        let mut agents = self.inner.agents.lock();
        let taken = agents.values().map(|node| node.name.as_str());
        let node = Arc::new(Node {
            id: saved.id.clone(),
            name: names::choose(saved.name.as_deref(), &task, taken),
            parent: saved.parent,
            workdir: crate::workdir::Workdir::new(saved.workdir.as_ref().unwrap_or(&saved.cwd)),
            cwd: saved.cwd,
            branch: saved.branch,
            status: Mutex::new(status.clone()),
            mailbox: Mutex::new(mailbox),
            cancel,
            task,
            task_archived: AtomicBool::new(saved.task_archived),
            model: Mutex::new(saved.model),
            goal: saved.goal,
            env,
            compact_at: saved.compact_at,
            budget: Mutex::new(saved.budget_tokens),
            usage: Mutex::new(Tokens {
                input: saved.input_tokens,
                output: saved.output_tokens,
                cache_read: saved.cache_read_tokens,
                cache_write: saved.cache_write_tokens,
            }),
            ledger: Mutex::new(saved.ledger),
            finished: AtomicBool::new(matches!(status, AgentStatus::Cancelled | AgentStatus::Queued)),
            context: Mutex::new(saved.context),
            paused: AtomicBool::new(saved.paused),
            reclaim_logged: Mutex::new(saved.reclaim_logged),
            progress: AtomicI64::new(saved.progress.unwrap_or_else(|| self.now().timestamp())),
            cancelled_at: AtomicI64::new(saved.cancelled_at.unwrap_or(0)),
            conversation_removed: AtomicU64::new(saved.conversation_removed.unwrap_or(0)),
            dormancy: Mutex::new(saved.dormancy),
            ..Node::default()
        });
        agents.insert(saved.id.clone(), node.clone());
        drop(agents);
        // Replayed for the UI only: the journal recorded them when they happened.
        let _ = self.inner.events.send(SwarmEvent::Spawned {
            id: node.id.clone(),
            parent: node.parent.clone(),
            name: node.name.clone(),
            task: node.task.clone(),
            cwd: node.workdir.get(),
            branch: node.branch.clone(),
        });
        let _ = self.inner.events.send(SwarmEvent::Status {
            id: node.id.clone(),
            status: status.clone(),
        });
        self.emit_usage(&node);
        if status.is_live() && status != AgentStatus::Queued {
            self.start_actor(node, Start::Resume { interrupted, ran });
        }
    }

    /// The persistable view of the swarm.
    pub fn snapshot(&self) -> SwarmState {
        let agents = self.nodes().iter().map(|node| node_state(node)).collect();
        SwarmState {
            run_id: self.inner.config.run_id.clone(),
            cwd: self.inner.config.cwd.clone(),
            next_id: self.inner.next_id.load(Ordering::SeqCst),
            agents,
            budget_tokens: *self.inner.budget.lock(),
            budget_flag: self.inner.config.budget_tokens,
            task_flag: self.inner.config.task_flag.clone(),
            max_agents: *self.inner.max_agents.lock(),
            max_calls: *self.inner.max_calls.lock(),
            max_load: *self.inner.host.max_load.lock(),
            min_free_mem: *self.inner.host.min_free_mem.lock(),
            priority: self.inner.priority.lock().clone(),
            commit: self.inner.config.commit.clone(),
            schedules: self.inner.schedules.lock().clone(),
        }
    }

    /// After a hard crash (no clean-stop mark), kill the shell commands its cut calls left running
    /// ([`crate::orphans`]). A clean stop has cut them already.
    async fn kill_cut_calls(&self, dir: &Path, state: &SwarmState) {
        if crate::orphans::take_clean_stop(dir) {
            return;
        }
        let mut agents = Vec::new();
        for agent in &state.agents {
            if let Some((_, Ok(batch))) = self.saved_batch(&agent.id).await {
                if batch.cut_shell() {
                    agents.push(agent.id.clone());
                }
            }
        }
        let killed = crate::orphans::kill_cut_calls(dir, &agents);
        if killed > 0 {
            tracing::info!(
                killed,
                "killed the shell commands of calls a hard crash cut short"
            );
            self.journal_event("orphans_killed", serde_json::json!({ "groups": killed }));
        }
    }

    /// `<run>/sessions/<id>/open-batch.json`; `None` for a run kept in memory.
    fn batch_file(&self, id: &str) -> Option<PathBuf> {
        let dir = self.inner.config.state_dir.as_ref()?;
        Some(dir.join(SESSIONS_DIR).join(id).join(resume::OPEN_BATCH_FILE))
    }

    /// Before any agent runs, bound the saved agents' stashes ([`crate::stash_gc::prune_run`]).
    async fn prune_stashes(&self, agents: &[AgentState]) {
        let Some(sessions) = self
            .inner
            .config
            .state_dir
            .as_ref()
            .map(|dir| dir.join(SESSIONS_DIR))
        else {
            return;
        };
        let agents: Vec<(String, bool)> = agents
            .iter()
            .map(|agent| (agent.id.clone(), agent.status == "cancelled"))
            .collect();
        let pruned = tokio::task::spawn_blocking(move || {
            crate::stash_gc::prune_run(
                &sessions,
                agents.iter().map(|(id, cancelled)| (id.as_str(), *cancelled)),
            )
        })
        .await
        .unwrap_or_default();
        self.journal_stash(None, pruned);
    }

    /// Delete the stash of `id`, cancelled and done ([`crate::stash_gc::drop_stash`]).
    pub(super) async fn drop_stash(&self, id: &str) {
        let Some(dir) = self
            .inner
            .config
            .state_dir
            .as_ref()
            .map(|dir| dir.join(SESSIONS_DIR).join(id))
        else {
            return;
        };
        let pruned = tokio::task::spawn_blocking(move || crate::stash_gc::drop_stash(&dir))
            .await
            .unwrap_or_default();
        self.journal_stash(Some(id), pruned);
    }

    /// Between two of `node`'s turns, once [`SwarmConfig::stash_prune_every`] has passed since
    /// `last`, save its conversation and then delete the stash entries nothing in its session
    /// names any more ([`crate::stash_gc::prune_unnamed`]), as a resume does.
    ///
    /// This cannot race a write to the stash. The SDK writes entries only while it builds the
    /// agent and while the agent runs a turn (each added message, the context pipeline, the
    /// `context` tool); only the agent's own actor calls this, after one turn returned and before
    /// it starts the next, so none of its turns runs meanwhile and no one else writes its stash.
    /// The save puts the conversation as it is in memory on disk first (or finds it there), so
    /// every name the agent still holds is in the files read; when it fails, nothing is pruned.
    pub(super) async fn prune_stash_between_turns(
        &self,
        node: &Node,
        agent: &Agent,
        last: &mut tokio::time::Instant,
    ) {
        let (Some(every), Some(state_dir)) = (
            self.inner.config.stash_prune_every,
            self.inner.config.state_dir.as_ref(),
        ) else {
            return;
        };
        if last.elapsed() < every
            || node.cancel.is_cancelled()
            || self.inner.shutting_down.load(Ordering::SeqCst)
        {
            return;
        }
        *last = tokio::time::Instant::now();
        let dir = state_dir.join(SESSIONS_DIR).join(&node.id);
        if !self.save_session(node, agent).await {
            return;
        }
        let pruned = tokio::task::spawn_blocking(move || crate::stash_gc::prune_unnamed(&dir))
            .await
            .unwrap_or_default();
        self.journal_stash(Some(&node.id), pruned);
    }

    fn journal_stash(&self, id: Option<&str>, pruned: crate::stash_gc::Pruned) {
        if pruned.entries > 0 {
            let mut fields = serde_json::json!({ "entries": pruned.entries, "bytes": pruned.bytes });
            if let Some(id) = id {
                fields["id"] = id.into();
            }
            self.journal_event("stash_pruned", fields);
        }
    }

    /// `id`'s [`Swarm::batch_file`] and what it holds, if there is one.
    async fn saved_batch(&self, id: &str) -> Option<(PathBuf, serde_json::Result<resume::OpenBatch>)> {
        let path = self.batch_file(id)?;
        let bytes = tokio::fs::read(&path).await.ok()?;
        Some((path, serde_json::from_slice(&bytes)))
    }

    /// Keep `node`'s tool batch on disk as its stream reports it ([`resume::OpenBatch`]): opened
    /// before the calls run, a result added as each finishes, dropped once the SDK has added the
    /// batch (or anything else from the user side) to the conversation.
    pub(super) fn track_batch(&self, node: &Node, event: &AgentStreamEvent) {
        let mut batch = node.batch.lock();
        match event {
            AgentStreamEvent::BeforeTools(tools) => {
                *batch = Some(resume::OpenBatch {
                    message: tools.message.clone(),
                    results: Vec::new(),
                })
            }
            AgentStreamEvent::AfterToolCall(call) => match batch.as_mut() {
                Some(open) => open.results.push(call.result.clone()),
                None => return,
            },
            AgentStreamEvent::MessageAdded(added)
                if added.message.role == strands_agents::types::Role::User =>
            {
                if batch.take().is_none() {
                    return;
                }
            }
            _ => return,
        }
        let Some(path) = self.batch_file(&node.id) else {
            return;
        };
        if let Err(error) = resume::OpenBatch::save(&path, batch.as_ref()) {
            tracing::warn!(agent = %node.id, %error, "could not save the tool batch in progress");
        }
    }

    /// Put back the tool batch a hard crash cut short ([`resume::restore_batch`]), save the
    /// conversation with it, then drop the file.
    pub(super) async fn restore_open_batch(&self, node: &Node, agent: &Agent) {
        let Some((path, batch)) = self.saved_batch(&node.id).await else {
            return;
        };
        match batch {
            Ok(batch) => {
                if agent.with_messages(|messages| resume::restore_batch(messages, batch)) {
                    self.save_session(node, agent).await;
                }
            }
            Err(error) => tracing::warn!(agent = %node.id, %error, "unreadable tool batch dropped"),
        }
        let _ = tokio::fs::remove_file(&path).await;
    }

    /// Save `agent`'s conversation now, after a change the SDK does not save itself (it saves
    /// when it adds a message). Through the agent's own session manager, which knows what it last
    /// wrote and so skips a save that would change nothing. Whether the files on disk now hold
    /// the conversation as it is in memory.
    pub(super) async fn save_session(&self, node: &Node, agent: &Agent) -> bool {
        let Some(dir) = self.inner.config.state_dir.as_ref() else {
            return false;
        };
        let manager = match agent.session_manager() {
            Some(manager) => Ok(manager),
            None => crate::agent_session_manager(&dir.join(SESSIONS_DIR), &node.id).map(Arc::new),
        };
        let saved = match manager {
            Ok(manager) => manager.save_snapshot(agent, true).await,
            Err(error) => Err(error),
        };
        if let Err(error) = &saved {
            tracing::warn!(agent = %node.id, %error, "could not save the conversation");
        }
        saved.is_ok()
    }

    /// Save the state in the background.
    pub(super) fn persist(&self) {
        if let Some(save) = self.state_save() {
            tokio::spawn(save);
        }
    }

    /// [`Swarm::persist`], returning once the state is on disk.
    pub(super) async fn persist_now(&self) {
        if let Some(save) = self.state_save() {
            save.await;
        }
    }

    /// The save of the state as it is now, for a persistent run that is not shutting down.
    fn state_save(&self) -> Option<impl std::future::Future<Output = ()> + Send + 'static> {
        let persister = self.inner.persister.clone()?;
        if self.inner.shutting_down.load(Ordering::SeqCst) {
            return None;
        }
        let ticket = persister.request();
        let swarm = self.clone();
        Some(async move {
            if let Err(error) = persister.save(ticket, || swarm.snapshot()).await {
                tracing::warn!(%error, "state save failed");
            }
        })
    }
}

/// What a restored errored agent shows: the error itself is not persisted.
pub(crate) const RESTORED_ERROR: &str = "its last turn failed before the restart";

pub(super) fn status_name(status: &AgentStatus) -> &'static str {
    match status {
        AgentStatus::Queued => "queued",
        AgentStatus::Starting => "starting",
        AgentStatus::Running => "running",
        AgentStatus::Idle => "idle",
        AgentStatus::Errored(_) => "errored",
        AgentStatus::Cancelled => "cancelled",
    }
}

/// The budget a resumed run starts with: the saved one, unless `--budget-tokens` was given with a
/// value other than the last one given ([`SwarmState::budget_flag`]).
fn resumed_budget(flag: Option<u64>, state: &SwarmState) -> Option<u64> {
    match flag {
        Some(tokens) if state.budget_flag != Some(tokens) => Some(tokens),
        _ => state.budget_tokens,
    }
}

/// The persistable view of one agent. Each lock is read on its own, as in `Node::info`.
pub(super) fn node_state(node: &Node) -> AgentState {
    let usage = *node.usage.lock();
    let model = node.model.lock().clone();
    let status = status_name(&node.status.lock()).to_owned();
    let mailbox = node.mailbox.lock().iter().map(mail_state).collect();
    let cancelled = status == "cancelled";
    let ledger = match cancelled {
        true => node.ledger.lock().compacted(chrono::Utc::now().timestamp()),
        false => node.ledger.lock().clone(),
    };
    let task_archived = cancelled && node.task_archived.load(Ordering::SeqCst);
    let context = *node.context.lock();
    let running_calls = node
        .tool_calls
        .lock()
        .iter()
        .map(|(id, started)| (id.clone(), started.elapsed().as_secs()))
        .collect();
    AgentState {
        id: node.id.clone(),
        name: Some(node.name.clone()),
        parent: node.parent.clone(),
        cwd: node.cwd.clone(),
        workdir: Some(node.workdir.get()).filter(|dir| *dir != node.cwd),
        branch: node.branch.clone(),
        task: match task_archived {
            true => crate::state::task_prefix(&node.task),
            false => node.task.clone(),
        },
        task_archived,
        model,
        goal: node.goal.clone(),
        status,
        mailbox,
        env: node.env.clone(),
        compact_at: node.compact_at,
        budget_tokens: *node.budget.lock(),
        input_tokens: usage.input,
        output_tokens: usage.output,
        cache_read_tokens: usage.cache_read,
        cache_write_tokens: usage.cache_write,
        ledger,
        context,
        paused: node.paused.load(Ordering::SeqCst),
        running_calls,
        reclaim_logged: node.reclaim_logged.lock().clone(),
        progress: Some(node.progress.load(Ordering::Relaxed)).filter(|at| *at > 0),
        cancelled_at: Some(node.cancelled_at.load(Ordering::Relaxed)).filter(|at| *at > 0),
        conversation_removed: Some(node.conversation_removed.load(Ordering::Relaxed))
            .filter(|secs| *secs > 0),
        dormancy: *node.dormancy.lock(),
    }
}

pub(super) fn mail_state(mail: &Mail) -> MailState {
    MailState {
        from: mail.from.clone(),
        from_label: mail.from_label.clone(),
        kind: mail.kind.to_owned(),
        text: mail.text.clone(),
        images: mail.images.clone(),
    }
}

pub(super) fn mail_from_state(mail: MailState) -> Mail {
    let kind = match mail.kind.as_str() {
        "reply" => "reply",
        "error" => "error",
        "interrupted" => "interrupted",
        "schedule" => "schedule",
        "fyi" => super::FYI,
        _ => "message",
    };
    Mail {
        from: mail.from,
        from_label: mail.from_label,
        kind,
        text: mail.text,
        images: mail.images,
    }
}

/// The messages of agent `id`'s latest saved snapshot, as [`crate::agent_session_manager`] keeps
/// it; empty when it has none.
async fn saved_messages(storage: &strands_agents::FileStorage, id: &str) -> Result<Vec<Message>> {
    use strands_agents::session::{SnapshotLocation, SnapshotStorage};
    let snapshot = storage
        .load_snapshot(&SnapshotLocation::agent(id, id), None)
        .await?;
    match snapshot.and_then(|mut snapshot| snapshot.data.remove("messages")) {
        Some(messages) => Ok(serde_json::from_value(messages)?),
        None => Ok(Vec::new()),
    }
}

/// [`SwarmEvent::History`] with the last [`HISTORY_MESSAGES`] of `messages`, or `None` for none.
pub(super) fn history_event(id: &str, messages: &[Message]) -> Option<SwarmEvent> {
    let omitted = messages.len().saturating_sub(HISTORY_MESSAGES);
    (!messages.is_empty()).then(|| SwarmEvent::History {
        id: id.to_owned(),
        messages: messages[omitted..].to_vec(),
        omitted,
    })
}
