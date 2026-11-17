//! The model-free status report behind `/status`.

use super::*;

impl Swarm {
    /// A snapshot of every live agent without a model call: state, todos, branch commits not on
    /// the base, running background jobs and schedules. The git part comes from a cache that is
    /// refreshed in the background when older than [`crate::status::TTL`], so the first call
    /// may report branches as unknown.
    pub fn status(&self) -> crate::status::StatusReport {
        self.refresh_branches();
        let cache = self.inner.branches.lock().clone();
        let schedules = self.schedules();
        let now = self.now();
        let nodes: Vec<Arc<Node>> = self.inner.agents.lock().values().cloned().collect();
        let agents = nodes
            .iter()
            .filter(|node| node.status.lock().is_live())
            .map(|node| {
                let info = node.info();
                let agent = node.agent();
                let agent = agent.as_ref();
                let todos = agent.and_then(|agent| {
                    agent
                        .state()
                        .get_as::<Vec<strands_harness::plugins::TodoItem>>(crate::status::TODOS_KEY)
                });
                let jobs = agent.and_then(Agent::background_tasks);
                let background = jobs.map_or_else(Vec::new, |jobs| {
                    jobs.list()
                        .into_iter()
                        .filter(|job| !job.status.is_terminal())
                        .map(|job| {
                            let now = Utc::now().timestamp_millis().max(0) as u64;
                            let age = Duration::from_millis(now.saturating_sub(job.created_at));
                            format!("{} ({})", job.tool_name, crate::schedule::human(age))
                        })
                        .collect()
                });
                crate::status::AgentReport {
                    label: node.label(),
                    status: info.status,
                    unread: info.unread,
                    branch: info.branch,
                    unmerged: cache.commits.get(&node.id).cloned(),
                    todos: todos.unwrap_or_default(),
                    background,
                    schedules: schedules
                        .iter()
                        .filter(|s| s.to == node.id)
                        .map(|s| s.describe(now))
                        .collect(),
                    id: info.id,
                }
            })
            .collect();
        crate::status::StatusReport {
            base: cache.base,
            agents,
        }
    }

    /// Start a background refresh of the branch cache unless one runs or it is fresh.
    pub(super) fn refresh_branches(&self) {
        {
            let mut cache = self.inner.branches.lock();
            let fresh = cache.at.is_some_and(|at| at.elapsed() < crate::status::TTL);
            if cache.refreshing || fresh {
                return;
            }
            cache.refreshing = true;
        }
        let nodes: Vec<Arc<Node>> = self.inner.agents.lock().values().cloned().collect();
        let root = nodes
            .iter()
            .find(|node| node.parent == USER)
            .map(|node| node.cwd.clone());
        let branches = nodes
            .iter()
            .filter(|node| node.status.lock().is_live())
            .filter_map(|node| Some((node.id.clone(), node.cwd.clone(), node.branch.clone()?)))
            .collect();
        let base = self.inner.config.status_base.clone();
        let swarm = self.clone();
        tokio::spawn(async move {
            let (base, commits) = crate::status::refresh(base, root, branches).await;
            let mut cache = swarm.inner.branches.lock();
            *cache = crate::status::BranchCache {
                at: Some(tokio::time::Instant::now()),
                refreshing: false,
                base,
                commits,
            };
        });
    }
}
