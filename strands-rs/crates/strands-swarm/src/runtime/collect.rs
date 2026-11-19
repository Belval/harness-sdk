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

    /// On resume: [`Swarm::tidy_landings`] in the background, published if it did anything.
    pub(super) fn tidy_landings_in_background(&self) {
        let swarm = self.clone();
        tokio::spawn(async move {
            if let Some(tidied) = swarm.tidy_landings(false).await {
                swarm.emit(SwarmEvent::Reclaimed {
                    collected: vec![tidied],
                    requested: false,
                });
            }
        });
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
                if swarm.newly_reclaimed(&node, &collected) {
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
    /// (the one `/status` compares against: [`SwarmConfig::status_base`], default the branch at
    /// the root agent's directory) has all of it, merged or cherry-picked.
    /// Runs on its own whenever an agent stops and on resume; call it to retry (e.g. after
    /// committing in a kept worktree). The outcome is also published as
    /// [`SwarmEvent::Reclaimed`], which the journal records.
    pub async fn gc(&self) -> Vec<Collected> {
        let mut collected = Vec::new();
        for node in self.collectable(false) {
            collected.push(self.collect_node(&node).await);
        }
        collected.extend(self.tidy_landings(true).await);
        self.prune_cancelled_sessions_announced().await;
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
        let base = self.base_branch().await;
        let branch = node.branch.as_deref();
        let collected = gc::collect(
            &config.run_id,
            &node.id,
            branch,
            &node.cwd,
            &node.env,
            base.as_deref(),
        )
        .await;
        *node.reclaimed.lock() = Some(collected.clone());
        collected
    }

    /// Whether `collected` is worth publishing on its own: something was done or kept, and not
    /// the same as last time (a resume retrying a worktree that is still kept).
    fn newly_reclaimed(&self, node: &Node, collected: &Collected) -> bool {
        if collected.is_empty() {
            return false;
        }
        let lines = Some(collected.lines().join("\n"));
        if *node.reclaim_logged.lock() == lines {
            return false;
        }
        *node.reclaim_logged.lock() = lines;
        self.persist();
        true
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
        let mut collected = Vec::new();
        for node in self.nodes().iter().filter(|node| node.branch.is_some()) {
            let cancelled = *node.status.lock() == AgentStatus::Cancelled;
            if !(ephemeral || cancelled) || node.collected.swap(true, Ordering::SeqCst) {
                continue;
            }
            let done = self.collect_node(node).await;
            if self.newly_reclaimed(node, &done) {
                collected.push(done);
            }
        }
        // Nothing reuses a finished ephemeral run's land checks' target dir.
        if ephemeral && self.inner.config.reclaim_disk {
            collected.extend(self.tidy_landings(true).await);
        }
        if !collected.is_empty() {
            self.emit(SwarmEvent::Reclaimed {
                collected,
                requested: false,
            });
        }
    }
}
