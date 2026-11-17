//! Reclaiming the disk of cancelled agents (worktrees, branches, sessions).

use super::*;

impl Swarm {
    /// Cancelled worktree agents whose own actor and every descendant's have exited, so nothing
    /// still runs in their worktree or target dir. With `claim`, only those not claimed before.
    pub(super) fn collectable(&self, claim: bool) -> Vec<Arc<Node>> {
        let owners: Vec<Arc<Node>> = {
            let agents = self.inner.agents.lock();
            agents
                .values()
                .filter(|node| node.branch.is_some() && *node.status.lock() == AgentStatus::Cancelled)
                .cloned()
                .collect()
        };
        owners
            .into_iter()
            .filter(|node| {
                self.subtree(&node.id)
                    .iter()
                    .all(|n| n.finished.load(Ordering::SeqCst))
            })
            .filter(|node| !claim || !node.collected.swap(true, Ordering::SeqCst))
            .collect()
    }

    /// Reclaim the disk of newly collectable agents without waiting (see [`Swarm::gc`]).
    pub(super) fn collect_in_background(&self) {
        if self.inner.shutting_down.load(Ordering::SeqCst) {
            return;
        }
        for node in self.collectable(true) {
            let swarm = self.clone();
            tokio::spawn(async move {
                let collected = swarm.collect_node(&node).await;
                if !collected.is_empty() {
                    swarm.emit(SwarmEvent::Reclaimed {
                        collected: vec![collected],
                        requested: false,
                    });
                }
            });
        }
    }

    /// Reclaim the disk of every cancelled worktree agent that has stopped: delete its private
    /// `CARGO_TARGET_DIR`, remove its worktree if it is clean, then delete its branch if the base
    /// ([`SwarmConfig::status_base`], default the main checkout's branch) has all of it, merged or
    /// cherry-picked.
    /// Runs on its own whenever an agent stops and on resume; call it to retry (e.g. after
    /// committing in a kept worktree). The outcome is also published as
    /// [`SwarmEvent::Reclaimed`], which the journal records.
    pub async fn gc(&self) -> Vec<Collected> {
        let mut collected = Vec::new();
        for node in self.collectable(false) {
            collected.push(self.collect_node(&node).await);
        }
        self.emit(SwarmEvent::Reclaimed {
            collected: collected.clone(),
            requested: true,
        });
        collected
    }

    /// [`Swarm::gc`] without waiting; the outcome arrives as [`SwarmEvent::Reclaimed`].
    pub fn gc_in_background(&self) {
        let swarm = self.clone();
        tokio::spawn(async move { swarm.gc().await });
    }

    pub(super) async fn collect_node(&self, node: &Node) -> Collected {
        if !self.inner.config.reclaim_disk {
            return Collected {
                id: node.id.clone(),
                ..Default::default()
            };
        }
        let config = &self.inner.config;
        let base = config.status_base.as_deref();
        let branch = node.branch.as_deref();
        let collected = gc::collect(&config.run_id, &node.id, branch, &node.cwd, &node.env, base).await;
        *node.reclaimed.lock() = Some(collected.clone());
        collected
    }

    /// After [`Swarm::cancel`]: what reclaiming the cancelled worktree agents of `id`'s subtree
    /// did, once all of them are done or `timeout` passes (`None` for those still stopping).
    pub async fn reclaimed_after_cancel(
        &self,
        id: &str,
        timeout: Duration,
    ) -> Vec<(String, Option<Collected>)> {
        let owners: Vec<Arc<Node>> = self
            .subtree(id)
            .into_iter()
            .filter(|node| node.branch.is_some())
            .collect();
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let done = owners.iter().all(|node| node.reclaimed.lock().is_some());
            if done || tokio::time::Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        owners
            .iter()
            .map(|node| (node.id.clone(), node.reclaimed.lock().clone()))
            .collect()
    }

    /// On shutdown of a run that cannot be resumed (no `--run`), reclaim every worktree agent's
    /// disk as if it were cancelled; a persistent run only reclaims cancelled agents not yet
    /// reclaimed, the rest resume into their worktrees.
    pub(super) async fn collect_on_exit(&self) {
        let ephemeral = self.inner.config.state_dir.is_none();
        let nodes: Vec<Arc<Node>> = self.inner.agents.lock().values().cloned().collect();
        let mut collected = Vec::new();
        for node in nodes.iter().filter(|node| node.branch.is_some()) {
            let cancelled = *node.status.lock() == AgentStatus::Cancelled;
            if !(ephemeral || cancelled) || node.collected.swap(true, Ordering::SeqCst) {
                continue;
            }
            let done = self.collect_node(node).await;
            if !done.is_empty() {
                collected.push(done);
            }
        }
        if !collected.is_empty() {
            self.emit(SwarmEvent::Reclaimed {
                collected,
                requested: false,
            });
        }
    }
}
