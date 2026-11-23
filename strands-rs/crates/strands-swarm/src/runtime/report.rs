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
        let agents = self
            .nodes()
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
                let quiet = self.quiet_for(&info, cache.branches.get(&node.id).and_then(|b| b.tip));
                let dormant = self.dormant_for(&info);
                crate::status::AgentReport {
                    label: node.label(),
                    status: info.status,
                    unread: info.unread,
                    priority: self.priority_of(&node.id, &node.name, &node.parent),
                    branch: info.branch,
                    git: cache.branches.get(&node.id).cloned(),
                    quiet,
                    dormant,
                    waking: self.waking_in(&node.id),
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
        let started = working(&self.inner.agents.lock());
        crate::status::StatusReport {
            limits: Some(crate::status::Limits {
                agents: (started, self.max_agents()),
                calls: (self.calls_in_flight(), self.max_calls()),
                load: Some(self.max_load())
                    .filter(|max| *max > 0.0)
                    .and_then(|max| Some((self.host()?.load, max))),
                disk: *self.inner.host.disk_free.lock(),
            }),
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
        let swarm = self.clone();
        tokio::spawn(async move { swarm.branch_states().await });
    }

    /// The base branch and, per live agent with a branch, how it stands against it (commits
    /// ahead, behind, trial merge), freshly from git; also refreshes the `/status` cache.
    pub async fn branch_states(
        &self,
    ) -> (
        Option<String>,
        std::collections::HashMap<String, crate::status::BranchState>,
    ) {
        let nodes = self.nodes();
        let root = Some(self.base_root());
        let branches = nodes
            .iter()
            .filter(|node| node.status.lock().is_live())
            .filter_map(|node| Some((node.id.clone(), node.cwd.clone(), node.branch.clone()?)))
            .collect();
        let base = self.inner.config.status_base.clone();
        let mut pairs = std::mem::take(&mut self.inner.branches.lock().pairs);
        let (base, branches) = crate::status::refresh(base, root, branches, &mut pairs).await;
        *self.inner.branches.lock() = crate::status::BranchCache {
            at: Some(tokio::time::Instant::now()),
            refreshing: false,
            base: base.clone(),
            branches: branches.clone(),
            pairs,
        };
        (base, branches)
    }

    /// Where the base branch is checked out: a root agent without a worktree of its own (a
    /// cancelled worktree root's checkout is gone, a live one has its own branch checked out),
    /// else the swarm's directory. Not necessarily the repository's main worktree.
    fn base_root(&self) -> PathBuf {
        let nodes = self.inner.agents.lock();
        let root = nodes
            .values()
            .find(|node| node.parent == USER && node.branch.is_none());
        root.map_or_else(|| self.inner.config.cwd.clone(), |node| node.cwd.clone())
    }

    /// The branch `/status` and reclaiming compare agents' branches against:
    /// [`SwarmConfig::status_base`], else the branch checked out at [`Self::base_root`].
    pub(super) async fn base_branch(&self) -> Option<String> {
        match &self.inner.config.status_base {
            Some(base) => Some(base.clone()),
            None => crate::status::current_branch(&self.base_root()).await,
        }
    }
}
