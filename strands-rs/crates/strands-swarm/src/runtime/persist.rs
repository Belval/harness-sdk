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
        if let Some(journal) = &swarm.inner.journal {
            journal.event("resume", serde_json::json!({ "agents": state.agents.len() }));
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
        // Fires missed while the run was down are due now: each fires once.
        if !state.schedules.is_empty() {
            *swarm.inner.schedules.lock() = state.schedules;
            swarm.start_scheduler();
        }
        swarm.start_queued();
        swarm.persist();
        swarm.collect_in_background();
        swarm.watch_jobs();
        Ok((swarm, events))
    }

    pub(super) fn restore_agent(&self, saved: AgentState) {
        let parent = self.node(&saved.parent);
        let cancel = parent
            .as_ref()
            .map_or(&self.inner.root_cancel, |node| &node.cancel)
            .child_token();
        let status = match saved.status.as_str() {
            "cancelled" => AgentStatus::Cancelled,
            "queued" => AgentStatus::Queued,
            _ => AgentStatus::Starting,
        };
        let interrupted = matches!(saved.status.as_str(), "running" | "starting");
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
            notify: Notify::new(),
            cancel,
            agent: OnceLock::new(),
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
            collected: AtomicBool::new(false),
            context: Mutex::new(saved.context),
            turn: Mutex::new(None),
            paused: AtomicBool::new(saved.paused),
            tool_calls: Mutex::new(0),
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
            self.start_actor(node, Start::Resume { interrupted });
        }
    }

    /// The persistable view of the swarm.
    pub fn snapshot(&self) -> SwarmState {
        // Release the agents lock before taking per-agent locks.
        let nodes: Vec<Arc<Node>> = self.inner.agents.lock().values().cloned().collect();
        let agents = nodes.iter().map(|node| node_state(node)).collect();
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
}

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

pub(super) fn node_state(node: &Node) -> AgentState {
    let usage = *node.usage.lock();
    AgentState {
        id: node.id.clone(),
        name: Some(node.name.clone()),
        parent: node.parent.clone(),
        cwd: node.cwd.clone(),
        branch: node.branch.clone(),
        task: node.task.clone(),
        model: node.model.lock().clone(),
        goal: node.goal.clone(),
        status: status_name(&node.status.lock()).to_owned(),
        mailbox: node.mailbox.lock().iter().map(mail_state).collect(),
        env: node.env.clone(),
        input_tokens: usage.input,
        output_tokens: usage.output,
        cache_read_tokens: usage.cache_read,
        cache_write_tokens: usage.cache_write,
        ledger: node.ledger.lock().clone(),
        context: *node.context.lock(),
        paused: node.paused.load(Ordering::SeqCst),
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
