//! Limits: the live-agent cap with its spawn queue, the token budget, and the model-call gate.

use super::*;

/// The outcome of a token-budget change: what was asked for and what is already used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetChange {
    /// The budget asked for; `None` lifts it.
    pub requested: Option<u64>,
    /// Tokens already used by the agent and its descendants (the swarm, for the swarm budget).
    pub used: u64,
}

impl BudgetChange {
    /// Whether the budget was changed. One at or below the tokens already used would stop the
    /// agents it covers at once, which is what cancelling is for, so it is refused (and a
    /// caller that wanted to raise a budget, but not knowing how much was spent, got it wrong,
    /// finds out).
    pub fn applied(&self) -> bool {
        self.requested.is_none_or(|tokens| tokens > self.used)
    }

    /// The outcome in a sentence; `who` is `a849`, or `the swarm`.
    pub fn report(&self, who: &str) -> String {
        let used = crate::text::compact(self.used);
        match self.requested {
            None => format!("Budget of {who} lifted ({used} used so far)"),
            Some(tokens) if self.applied() => format!(
                "Budget of {who} set to {} ({used} used so far, {} left)",
                crate::text::compact(tokens),
                crate::text::compact(tokens - self.used)
            ),
            Some(tokens) => format!(
                "Not set: {who} has already used {used}, so a {} budget would stop it at once; \
                 give a budget above that (e.g. {used} + what {who} may still spend)",
                crate::text::compact(tokens)
            ),
        }
    }
}

impl Swarm {
    /// Start queued agents, oldest first, while there are free slots; queued agents in the
    /// priority lane start whatever the cap, and count toward it. Returns how many started.
    pub(super) fn start_queued(&self) -> usize {
        if self.inner.shutting_down.load(Ordering::SeqCst) {
            return 0;
        }
        let ready: Vec<Arc<Node>> = {
            let agents = self.inner.agents.lock();
            let mut free = self.max_agents().saturating_sub(working(&agents));
            let mut ready = Vec::new();
            for node in agents.values() {
                let priority = self.priority_of(&node.id, &node.name, &node.parent);
                if free == 0 && !priority {
                    continue;
                }
                // Checked and claimed under the status lock, so a concurrent cancel wins.
                let mut status = node.status.lock();
                if *status == AgentStatus::Queued && !node.cancel.is_cancelled() {
                    *status = AgentStatus::Starting;
                    node.finished.store(false, Ordering::SeqCst);
                    ready.push(node.clone());
                    // It holds a slot like any other, even past the cap.
                    free = free.saturating_sub(1);
                }
            }
            ready
        };
        if ready.is_empty() {
            return 0;
        }
        let started = ready.len();
        self.persist();
        for node in ready {
            self.emit(SwarmEvent::Status {
                id: node.id.clone(),
                status: AgentStatus::Starting,
            });
            let task = node.task.clone();
            self.start_actor(node, Start::Fresh(task));
        }
        started
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
    /// that was stopped for lack of it is told to continue. A budget at or below what the swarm
    /// has used would stop everyone at once, so it is refused ([`BudgetChange::applied`]).
    pub fn set_budget(&self, tokens: Option<u64>) -> BudgetChange {
        let change = BudgetChange {
            requested: tokens,
            used: self.tokens_used(),
        };
        if !change.applied() {
            return change;
        }
        *self.inner.budget.lock() = tokens;
        self.persist();
        self.emit(SwarmEvent::Budget { tokens });
        self.resume_stopped("The token budget", tokens);
        change
    }

    /// Set (or lift) agent `id`'s own token budget, which `by` (an ancestor of `id`, or the
    /// user) is allowed to do; the budget covers `id` and its descendants. Agents it had
    /// stopped that now have budget left are told to continue. A budget at or below what `id` and
    /// its descendants have used would stop it at once, so it is refused and nothing changes
    /// ([`BudgetChange::applied`]); `cancel_agent` is the way to stop an agent.
    pub fn set_agent_budget(&self, by: &str, id: &str, tokens: Option<u64>) -> Result<BudgetChange> {
        let node = self.known(id)?;
        let mut parent = node.parent.clone();
        while by != USER && parent != by {
            parent = match self.node(&parent) {
                Some(above) => above.parent.clone(),
                None => {
                    return Err(Error::other(format!(
                        "Only an ancestor of {} can set its budget.",
                        node.label()
                    )))
                }
            };
        }
        let change = BudgetChange {
            requested: tokens,
            used: self.subtree_tokens(id),
        };
        if !change.applied() {
            return Ok(change);
        }
        *node.budget.lock() = tokens;
        self.persist();
        self.journal_event("agent_budget", serde_json::json!({ "id": id, "tokens": tokens }));
        self.resume_stopped(&format!("The token budget of {id}"), tokens);
        Ok(change)
    }

    /// Tell each agent stopped for lack of budget that can run again to continue.
    fn resume_stopped(&self, what: &str, tokens: Option<u64>) {
        let limit = tokens.map_or_else(
            || "lifted".to_owned(),
            |t| format!("raised to {}", crate::text::compact(t)),
        );
        let stopped: Vec<String> = self.inner.over_budget.lock().clone();
        for id in stopped
            .into_iter()
            .filter(|id| self.budget_exhausted(id).is_none())
        {
            self.inner.over_budget.lock().retain(|stopped| *stopped != id);
            let note = format!("[{what} was {limit}. Continue where you left off.]");
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
    /// `None` returns to the configured one. Raising it starts queued agents at once; returns how
    /// many it started.
    pub fn set_max_agents(&self, limit: Option<usize>) -> Result<usize> {
        if limit == Some(0) {
            return Err(Error::other("The agent limit must be at least 1."));
        }
        *self.inner.max_agents.lock() = limit;
        self.persist();
        self.journal_event("max_agents", serde_json::json!({ "limit": self.max_agents() }));
        Ok(self.start_queued())
    }

    /// Agent slots in use (agents working: starting, mid-turn, or idle with waking mail), the
    /// limit, and the mid-turn agents that hold none because they only wait. Idle agents hold none
    /// either.
    pub fn agent_slots(&self) -> crate::status::AgentSlots {
        let agents = self.inner.agents.lock();
        crate::status::AgentSlots {
            in_use: working(&agents),
            limit: self.max_agents(),
            waiting: agents.values().filter(|node| only_waits(node)).count(),
        }
    }

    /// Agents waiting for a slot.
    pub fn queued_agents(&self) -> usize {
        let agents = self.inner.agents.lock();
        agents
            .values()
            .filter(|node| *node.status.lock() == AgentStatus::Queued)
            .count()
    }

    /// Tokens (input + output) used by `id` and its descendants.
    pub fn subtree_tokens(&self, id: &str) -> u64 {
        self.subtree(id)
            .iter()
            .map(|node| {
                let usage = *node.usage.lock();
                usage.input + usage.output
            })
            .sum()
    }

    /// The refusal message when the swarm's budget, or that of `id` or of an ancestor of it, is
    /// spent.
    pub(super) fn budget_exhausted(&self, id: &str) -> Option<String> {
        if let Some(budget) = self.budget() {
            let used = self.tokens_used();
            if used >= budget {
                return Some(format!(
                    "Token budget exhausted ({}), so no further model calls run. Report this to \
                     whoever is waiting on you; you will be told when the user raises the budget (TUI \
                     `/budget N`).",
                    crate::text::usage(used, budget)
                ));
            }
        }
        let mut next = self.node(id);
        while let Some(node) = next {
            let budget = *node.budget.lock();
            if let Some(budget) = budget {
                let used = self.subtree_tokens(&node.id);
                if used >= budget {
                    return Some(format!(
                        "{} token budget exhausted ({}), so no model calls run for it or its \
                         descendants. Report this to whoever is waiting on you; its parent can \
                         raise it (send_message to {} with budget_tokens).",
                        node.label(),
                        crate::text::usage(used, budget),
                        node.id
                    ));
                }
            }
            next = self.node(&node.parent);
        }
        None
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
                    swarm.wait_for_host(&id, false, &cancel).await?;
                    let _permit = swarm.admit_model_call(&id, &cancel).await?;
                    next.run(context, &events).await
                }
            },
        ));
    }
}

/// Agents holding a slot under [`SwarmConfig::max_agents`]: the working ones, starting, mid-turn,
/// or idle with waking mail they are about to start a turn on. An idle agent holds none, nor does
/// one mid-turn whose only calls in flight are waits ([`only_waits`]), so the cap bounds agents
/// doing work at once, not how many exist.
pub(super) fn working(agents: &IndexMap<String, Arc<Node>>) -> usize {
    agents.values().filter(|node| holds_slot(node)).count()
}

/// Whether `node` holds a slot ([`working`]).
pub(super) fn holds_slot(node: &Node) -> bool {
    // The status lock is let go before the mailbox is read.
    let status = node.status.lock().clone();
    match status {
        AgentStatus::Starting => true,
        AgentStatus::Running => !only_waits(node),
        AgentStatus::Idle | AgentStatus::Errored(_) => node.mailbox.lock().iter().any(Mail::wakes),
        AgentStatus::Queued | AgentStatus::Cancelled => false,
    }
}

/// Whether `node` is mid-turn with only waiting tools ([`WAITING_TOOLS`]) in flight: it holds no
/// slot until one returns, so a parent waiting for its queued children lets them start.
pub(super) fn only_waits(node: &Node) -> bool {
    // One lock at a time.
    let running = *node.status.lock() == AgentStatus::Running;
    let other_calls = !node.tool_calls.lock().is_empty();
    running && !other_calls && *node.waits.lock() > 0
}
