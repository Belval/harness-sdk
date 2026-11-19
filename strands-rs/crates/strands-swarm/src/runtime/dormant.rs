//! Dormant agents: idle with nothing that could wake them.
//!
//! An agent wakes on unread mail (a schedule, a job's end report and a child's reply are all
//! mail), so an idle one is dormant when none is unread, none can still come from a pending
//! schedule aimed at it, a running background job it owns, or a child that is itself going to act,
//! and no agent is `wait`ing on it. A periodic sweep stamps when it first saw each agent in that
//! state; `list_agents` and `/status` show it after [`crate::status::DORMANT_AFTER`], and the
//! agent's parent gets one [`FYI`] note per spell, unless the agent's last reply reached the parent
//! and nothing has reached it since: it is done and its parent has the report.

use std::collections::{HashMap, HashSet};

use super::*;
use crate::state::Dormancy;

/// How often the sweep runs.
const SWEEP_EVERY: Duration = Duration::from_secs(30);

/// Removes a `wait`'s targets from the waiter's list when the wait ends.
pub(super) struct Waiting {
    node: Arc<Node>,
    from: Vec<String>,
}

impl Waiting {
    pub(super) fn new(node: &Arc<Node>, from: &[String]) -> Self {
        node.waiting.lock().push(from.to_vec());
        Self {
            node: node.clone(),
            from: from.to_vec(),
        }
    }
}

impl Drop for Waiting {
    fn drop(&mut self) {
        let mut waits = self.node.waiting.lock();
        if let Some(at) = waits.iter().position(|from| *from == self.from) {
            waits.remove(at);
        }
    }
}

impl Swarm {
    /// Run [`Swarm::sweep_dormant`] every [`SWEEP_EVERY`] until shutdown; started once.
    pub(super) fn watch_dormant(&self) {
        if self.inner.watching_dormant.swap(true, Ordering::SeqCst) {
            return;
        }
        let swarm = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(SWEEP_EVERY) => {}
                    _ = swarm.inner.root_cancel.cancelled() => return,
                }
                if swarm.inner.shutting_down.load(Ordering::SeqCst) {
                    return;
                }
                swarm.sweep_dormant().await;
            }
        });
    }

    /// A schedule or job now points at `id`: it is no longer dormant.
    pub(super) fn clear_dormancy(&self, id: &str) {
        if let Some(node) = self.node(id) {
            let mut dormancy = node.dormancy.lock();
            *dormancy = Dormancy {
                replied: dormancy.replied,
                ..Dormancy::default()
            };
        }
    }

    /// Stamp the idle agents that nothing can wake and clear the others; tell the parent of each
    /// that has been so for [`crate::status::DORMANT_AFTER`], once per spell.
    pub async fn sweep_dormant(&self) {
        let nodes = self.nodes();
        let running: HashSet<String> = self
            .jobs()
            .await
            .into_iter()
            .filter(|job| job.state == jobs::JobState::Running)
            .map(|job| job.meta.owner)
            .collect();
        let scheduled: HashSet<String> = self.scheduled_targets().into_iter().collect();
        let status: HashMap<&str, AgentStatus> = nodes
            .iter()
            .map(|n| (n.id.as_str(), n.status.lock().clone()))
            .collect();
        let live = |id: &str| status.get(id).is_some_and(AgentStatus::is_live);
        let mut awaited: HashSet<&str> = HashSet::new();
        for node in nodes.iter().filter(|node| live(&node.id)) {
            for from in node.waiting.lock().iter() {
                match from.is_empty() {
                    true => awaited.extend(
                        nodes
                            .iter()
                            .filter(|n| n.parent == node.id)
                            .map(|n| n.id.as_str()),
                    ),
                    false => awaited.extend(
                        nodes
                            .iter()
                            .map(|n| n.id.as_str())
                            .filter(|id| from.iter().any(|f| f == id)),
                    ),
                }
            }
        }
        // Who will act: working agents, idle ones with a way to be woken, and (below) whoever
        // has a child among them, since that child will report.
        let mut acts: HashSet<&str> = nodes
            .iter()
            .filter(|node| match &status[node.id.as_str()] {
                AgentStatus::Running | AgentStatus::Starting | AgentStatus::Queued => true,
                AgentStatus::Idle | AgentStatus::Errored(_) => {
                    node.mailbox.lock().iter().any(Mail::wakes)
                        || scheduled.contains(&node.id)
                        || running.contains(&node.id)
                        || awaited.contains(node.id.as_str())
                }
                AgentStatus::Cancelled => false,
            })
            .map(|node| node.id.as_str())
            .collect();
        while let Some(parent) = nodes
            .iter()
            .filter(|node| acts.contains(node.id.as_str()) && live(&node.parent))
            .map(|node| node.parent.as_str())
            .find(|parent| !acts.contains(parent))
        {
            acts.insert(parent);
        }
        let now = self.now().timestamp();
        let mut changed = false;
        for node in &nodes {
            let dormant = status[node.id.as_str()] == AgentStatus::Idle
                && !acts.contains(node.id.as_str())
                && !node.dormancy.lock().replied;
            let tell = {
                let mut state = node.dormancy.lock();
                let before = *state;
                match dormant {
                    false => {
                        state.since = 0;
                        state.told = false;
                    }
                    true if state.since == 0 => state.since = now,
                    true => {}
                }
                let due = dormant
                    && !state.told
                    && node.parent != USER
                    && Duration::from_secs((now - state.since).max(0) as u64) >= crate::status::DORMANT_AFTER;
                state.told |= due;
                changed |= *state != before;
                due.then_some(now - state.since)
            };
            if let Some(idle) = tell.filter(|_| live(&node.parent)) {
                let text = format!(
                    "{} ({}) has been idle {} with nothing to wake it (no mail, schedule, job or \
                     child). Message it to resume, or cancel it if it is done.",
                    node.name,
                    node.id,
                    schedules::human(Duration::from_secs(idle.max(0) as u64))
                );
                let _ = self.post(&node.id, &node.parent, FYI, text);
            }
        }
        if changed {
            self.persist();
        }
    }

    /// How long `info`'s agent has been dormant, once that is worth saying
    /// ([`crate::status::dormant_note`]).
    pub fn dormant_for(&self, info: &AgentInfo) -> Option<Duration> {
        let since = info.dormant_since.filter(|_| info.status == AgentStatus::Idle)?;
        Some(Duration::from_secs((self.now().timestamp() - since).max(0) as u64))
    }
}
