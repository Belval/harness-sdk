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
    /// `CARGO_TARGET_DIR`, and remove its worktree if it is clean. Branches are never deleted.
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
        let run_id = &self.inner.config.run_id;
        gc::collect(run_id, &node.id, node.branch.as_deref(), &node.cwd, &node.env).await
    }
}
