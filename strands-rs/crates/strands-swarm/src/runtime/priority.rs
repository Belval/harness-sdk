//! The priority lane: agents that never wait on the host or the caps, such as the one relaying the
//! user's messages.

use super::*;

impl Swarm {
    /// Who is in the priority lane besides the root agents: the last [`Swarm::set_priority`],
    /// else `--priority` ([`SwarmConfig::priority`]); ids or names.
    pub fn priority(&self) -> Vec<String> {
        let set = self.inner.priority.lock().clone();
        set.unwrap_or_else(|| self.inner.config.priority.clone())
    }

    /// Replace the priority lane for this run (kept across restarts, ahead of `--priority`); every
    /// agent named must exist, and is kept by id. `Some(vec![])` leaves only the root agents,
    /// `None` returns to the flag. Held calls and queued agents it names go ahead at once.
    pub fn set_priority(&self, agents: Option<Vec<String>>) -> Result<()> {
        let ids = match agents {
            Some(agents) => {
                let mut ids = Vec::new();
                for who in agents {
                    let id = self.resolve(&who)?;
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
                Some(ids)
            }
            None => None,
        };
        *self.inner.priority.lock() = ids;
        self.persist();
        self.journal_event("priority", serde_json::json!({ "agents": self.priority() }));
        self.inner.host.wake.notify_waiters();
        self.inner.calls.wake();
        self.start_queued();
        Ok(())
    }

    /// Whether agent `id` is in the priority lane: a root agent, or one [`Swarm::priority`]
    /// names. Its model calls and jobs skip the host's load and memory holds (not the disk
    /// hold), its calls never wait for a slot under the call cap and it is never queued at the
    /// agent cap; it still counts toward both caps.
    pub fn is_priority(&self, id: &str) -> bool {
        self.node(id)
            .is_some_and(|node| self.priority_of(&node.id, &node.name, &node.parent))
    }

    /// [`Swarm::is_priority`] for an agent not looked up yet (or whose lock is held).
    pub(super) fn priority_of(&self, id: &str, name: &str, parent: &str) -> bool {
        parent == USER || self.priority().iter().any(|who| names::addresses(who, id, name))
    }
}
