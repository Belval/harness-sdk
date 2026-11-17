//! Limits: the live-agent cap with its spawn queue, the token budget, and the model-call gate.

use super::*;

impl Swarm {
    /// Start queued agents, oldest first, while there are free slots.
    pub(super) fn start_queued(&self) {
        if self.inner.shutting_down.load(Ordering::SeqCst) {
            return;
        }
        let scheduled = self.scheduled_targets();
        let ready: Vec<Arc<Node>> = {
            let agents = self.inner.agents.lock();
            let free = self.max_agents().saturating_sub(started(&agents, &scheduled));
            let mut ready = Vec::new();
            for node in agents.values() {
                if ready.len() == free {
                    break;
                }
                // Checked and claimed under the status lock, so a concurrent cancel wins.
                let mut status = node.status.lock();
                if *status == AgentStatus::Queued && !node.cancel.is_cancelled() {
                    *status = AgentStatus::Starting;
                    node.finished.store(false, Ordering::SeqCst);
                    ready.push(node.clone());
                }
            }
            ready
        };
        if ready.is_empty() {
            return;
        }
        self.persist();
        for node in ready {
            self.emit(SwarmEvent::Status {
                id: node.id.clone(),
                status: AgentStatus::Starting,
            });
            let task = node.task.clone();
            self.start_actor(node, Start::Fresh(task));
        }
    }

    /// Tokens (input + output) used by every agent so far, saved runs included.
    pub fn tokens_used(&self) -> u64 {
        self.nodes()
            .iter()
            .map(|node| {
                let usage = *node.usage.lock();
                usage.input + usage.output
            })
            .sum()
    }

    /// The swarm-wide token budget; `None` for no limit.
    pub fn budget(&self) -> Option<u64> {
        *self.inner.budget.lock()
    }

    /// Set (or, with `None`, lift) the token budget. If that leaves budget to spend, every agent
    /// that was stopped for lack of it is told to continue.
    pub fn set_budget(&self, tokens: Option<u64>) {
        *self.inner.budget.lock() = tokens;
        self.persist();
        self.emit(SwarmEvent::Budget { tokens });
        if self.budget_exhausted().is_some() {
            return;
        }
        let stopped: Vec<String> = std::mem::take(&mut *self.inner.over_budget.lock());
        let limit = tokens.map_or_else(
            || "lifted".to_owned(),
            |t| format!("raised to {}", crate::text::compact(t)),
        );
        for id in stopped {
            let note = format!("[The token budget was {limit}. Continue where you left off.]");
            let _ = self.send(USER, &id, note);
        }
    }

    /// The live-agent limit in force: the last [`Swarm::set_max_agents`], else the config's.
    pub fn max_agents(&self) -> usize {
        self.inner
            .max_agents
            .lock()
            .unwrap_or(self.inner.config.max_agents)
    }

    /// Change the live-agent limit for this run (kept across restarts, ahead of `--max-agents`);
    /// `None` returns to the configured one. Raising it starts queued agents at once.
    pub fn set_max_agents(&self, limit: Option<usize>) -> Result<()> {
        if limit == Some(0) {
            return Err(Error::other("The agent limit must be at least 1."));
        }
        *self.inner.max_agents.lock() = limit;
        self.persist();
        if let Some(journal) = &self.inner.journal {
            journal.event("max_agents", serde_json::json!({ "limit": self.max_agents() }));
        }
        self.start_queued();
        Ok(())
    }

    /// The refusal message when the budget is spent.
    pub(super) fn budget_exhausted(&self) -> Option<String> {
        let budget = self.budget()?;
        let used = self.tokens_used();
        (used >= budget).then(|| {
            format!(
                "Token budget exhausted: the swarm has used {} of {} tokens, so no further model \
                 calls run. Report this to whoever is waiting on you; you will be told when the user \
                 raises the budget (TUI `/budget N`).",
                crate::text::compact(used),
                crate::text::compact(budget)
            )
        })
    }

    /// Gate every model call: refuse it once the token budget is spent, and hold a swarm-wide
    /// permit for its duration, so a large swarm queues instead of tripping provider throttling.
    /// Waiting is published as [`SwarmEvent::ModelQueued`].
    pub(super) fn install_model_gate(&self, agent: &Agent, node: &Node) {
        let swarm = self.clone();
        let id = node.id.clone();
        agent.add_middleware::<InvokeModelStage>(wrap_fn(
            move |context: InvokeModelContext, events, next| {
                let (swarm, id) = (swarm.clone(), id.clone());
                // The invocation's token: cancelled by an interrupt as well as by a cancel.
                let cancel = context.agent.cancel_token();
                async move {
                    swarm.hold_for_reload(&id, &cancel).await?;
                    let _permit = swarm.admit_model_call(&id, &cancel).await?;
                    next.run(context, &events).await
                }
            },
        ));
    }

    /// Refuse a model call of agent `id` once the budget is spent, else wait for a swarm-wide
    /// permit (if calls are capped), to be held for the call's duration.
    pub(super) async fn admit_model_call(
        &self,
        id: &str,
        cancel: &CancellationToken,
    ) -> Result<Option<OwnedSemaphorePermit>> {
        if let Some(message) = self.budget_exhausted() {
            let mut stopped = self.inner.over_budget.lock();
            if !stopped.iter().any(|stopped| stopped == id) {
                stopped.push(id.to_owned());
            }
            return Err(Error::other(message));
        }
        let Some(permits) = self.inner.model_calls.clone() else {
            return Ok(None);
        };
        if let Ok(permit) = permits.clone().try_acquire_owned() {
            return Ok(Some(permit));
        }
        self.emit(SwarmEvent::ModelQueued {
            id: id.to_owned(),
            queued: true,
        });
        let acquired = tokio::select! {
            permit = permits.acquire_owned() => permit.ok(),
            _ = cancel.cancelled() => None,
        };
        self.emit(SwarmEvent::ModelQueued {
            id: id.to_owned(),
            queued: false,
        });
        acquired.map(Some).ok_or(Error::Cancelled)
    }
}

/// Agents holding a slot under [`SwarmConfig::max_agents`]: live and not queued. An idle agent
/// that a schedule will wake (`scheduled` lists their ids) holds none while it sleeps.
pub(super) fn started(agents: &IndexMap<String, Arc<Node>>, scheduled: &[String]) -> usize {
    agents
        .values()
        .filter(|node| match *node.status.lock() {
            AgentStatus::Cancelled | AgentStatus::Queued => false,
            AgentStatus::Idle => !scheduled.contains(&node.id),
            _ => true,
        })
        .count()
}
