//! Saving the swarm to `swarm.json` and restoring it on resume.

use super::*;

impl Swarm {
    /// Reopen a saved run: restore every agent with its conversation and unread mail, wake those
    /// that were mid-turn, and keep cancelled ones visible but inert.
    pub async fn resume(
        config: SwarmConfig,
        state: SwarmState,
    ) -> Result<(Self, mpsc::UnboundedReceiver<SwarmEvent>)> {
        let (swarm, events) = Self::new(config);
        swarm.inner.next_id.store(state.next_id, Ordering::SeqCst);
        *swarm.inner.max_agents.lock() = state.max_agents;
        if let Some(tokens) = state.budget_tokens {
            *swarm.inner.budget.lock() = Some(tokens);
            swarm.emit(SwarmEvent::Budget { tokens: Some(tokens) });
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
        for agent in state.agents {
            swarm.restore_agent(agent);
        }
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
        swarm.watch_jobs();
        Ok((swarm, events))
    }

    /// Cancelled agents get no actor on resume, so nothing restores their session. Read each
    /// one's saved conversation in the background and publish its tail ([`SwarmEvent::History`]),
    /// so a viewer still shows what they did.
    fn publish_cancelled_history(&self) {
        let Some(dir) = self
            .inner
            .config
            .state_dir
            .as_ref()
            .map(|dir| dir.join(SESSIONS_DIR))
        else {
            return;
        };
        let cancelled: Vec<String> = self
            .inner
            .agents
            .lock()
            .values()
            .filter(|node| *node.status.lock() == AgentStatus::Cancelled)
            .map(|node| node.id.clone())
            .collect();
        if cancelled.is_empty() {
            return;
        }
        let swarm = self.clone();
        self.track(tokio::spawn(async move {
            let storage = strands_agents::FileStorage::new(dir);
            for id in cancelled {
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
        let mut agents = self.inner.agents.lock();
        let taken = agents.values().map(|node| node.name.as_str());
        let node = Arc::new(Node {
            id: saved.id.clone(),
            name: names::choose(saved.name.as_deref(), &saved.task, taken),
            parent: saved.parent,
            cwd: saved.cwd,
            branch: saved.branch,
            status: Mutex::new(status.clone()),
            mailbox: Mutex::new(mailbox),
            cancel,
            task: saved.task,
            model: Mutex::new(saved.model),
            goal: saved.goal,
            env,
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
            cwd: node.cwd.clone(),
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
            max_agents: *self.inner.max_agents.lock(),
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
            let Some(path) = self.batch_file(&agent.id) else {
                continue;
            };
            let Ok(bytes) = tokio::fs::read(&path).await else {
                continue;
            };
            let batch = serde_json::from_slice::<resume::OpenBatch>(&bytes);
            if batch.is_ok_and(|batch| batch.cut_shell()) {
                agents.push(agent.id.clone());
            }
        }
        let killed = crate::orphans::kill_cut_calls(dir, &agents).await;
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
        let written = match batch.as_ref() {
            Some(open) => write_atomically(&path, open),
            None => std::fs::remove_file(&path).or_else(|e| match e.kind() {
                std::io::ErrorKind::NotFound => Ok(()),
                _ => Err(e),
            }),
        };
        if let Err(error) = written {
            tracing::warn!(agent = %node.id, %error, "could not save the tool batch in progress");
        }
    }

    /// Put back the tool batch a hard crash cut short ([`resume::restore_batch`]), save the
    /// conversation with it, then drop the file.
    pub(super) async fn restore_open_batch(&self, node: &Node, agent: &Agent) {
        let Some(path) = self.batch_file(&node.id) else {
            return;
        };
        let Ok(bytes) = tokio::fs::read(&path).await else {
            return;
        };
        match serde_json::from_slice::<resume::OpenBatch>(&bytes) {
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
    /// when it adds a message).
    pub(super) async fn save_session(&self, node: &Node, agent: &Agent) {
        let Some(dir) = self.inner.config.state_dir.as_ref() else {
            return;
        };
        let saved = match crate::agent_session_manager(&dir.join(SESSIONS_DIR), &node.id) {
            Ok(manager) => manager.save_snapshot(agent, true).await,
            Err(error) => Err(error),
        };
        if let Err(error) = saved {
            tracing::warn!(agent = %node.id, %error, "could not save the conversation");
        }
    }

    pub(super) fn persist(&self) {
        let Some(persister) = self.inner.persister.clone() else {
            return;
        };
        if self.inner.shutting_down.load(Ordering::SeqCst) {
            return;
        }
        let generation = self.inner.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let state = self.snapshot();
        tokio::spawn(async move {
            if let Err(error) = persister.write(generation, &state).await {
                tracing::warn!(%error, "state save failed");
            }
        });
    }

    /// [`Swarm::persist`], returning once the state is on disk.
    pub(super) async fn persist_now(&self) {
        let Some(persister) = self.inner.persister.clone() else {
            return;
        };
        if self.inner.shutting_down.load(Ordering::SeqCst) {
            return;
        }
        let generation = self.inner.generation.fetch_add(1, Ordering::SeqCst) + 1;
        if let Err(error) = persister.write(generation, &self.snapshot()).await {
            tracing::warn!(%error, "state save failed");
        }
    }
}

/// Write `value` as JSON to `path` through a temporary file and a rename, so a crash leaves the
/// old file or the new one, never half of one.
fn write_atomically(path: &Path, value: &impl serde::Serialize) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    std::fs::write(&tmp, serde_json::to_vec(value)?)?;
    std::fs::rename(&tmp, path)
}

/// What a restored errored agent shows: the error itself is not persisted.
const RESTORED_ERROR: &str = "its last turn failed before the restart";

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

/// The persistable view of one agent. Each lock is read on its own (see `Node::info`): guards
/// taken inside the struct literal are all held until it is built, and holding `model` while
/// waiting for `status` deadlocked against `Node::info`.
pub(super) fn node_state(node: &Node) -> AgentState {
    let usage = *node.usage.lock();
    let model = node.model.lock().clone();
    let status = status_name(&node.status.lock()).to_owned();
    let mailbox = node.mailbox.lock().iter().map(mail_state).collect();
    let ledger = node.ledger.lock().clone();
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
        branch: node.branch.clone(),
        task: node.task.clone(),
        model,
        goal: node.goal.clone(),
        status,
        mailbox,
        env: node.env.clone(),
        input_tokens: usage.input,
        output_tokens: usage.output,
        cache_read_tokens: usage.cache_read,
        cache_write_tokens: usage.cache_write,
        ledger,
        context,
        paused: node.paused.load(Ordering::SeqCst),
        running_calls,
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
