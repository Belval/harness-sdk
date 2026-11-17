//! A model-free status report of the swarm ([`crate::Swarm::status`], the TUI's `/status`): per
//! live agent its state, todo list, branch commits not yet on the base, background jobs and
//! schedules. The git part runs in the background and is cached for [`TTL`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use strands_harness::plugins::{TodoItem, TodoStatus};

/// How long cached branch commits are served before a background refresh.
pub const TTL: Duration = Duration::from_secs(5);

/// Most commits listed per branch.
pub const MAX_COMMITS: usize = 20;

/// Agent-state key of the harness todo list ([`strands_harness::plugins::Todos`] default).
pub const TODOS_KEY: &str = "todos";

/// Cached result of the last git refresh.
#[derive(Debug, Clone, Default)]
pub struct BranchCache {
    /// When the last refresh finished.
    pub at: Option<tokio::time::Instant>,
    /// A refresh is running.
    pub refreshing: bool,
    /// The branch the commits are compared against.
    pub base: Option<String>,
    /// Agent id → `<short hash> <subject>` of its branch's commits not on the base, newest first.
    pub commits: HashMap<String, Vec<String>>,
}

/// One live agent in a [`StatusReport`].
#[derive(Debug, Clone, PartialEq)]
pub struct AgentReport {
    /// Agent id.
    pub id: String,
    /// `name (id)`.
    pub label: String,
    /// Lifecycle state.
    pub status: crate::AgentStatus,
    /// Unread messages.
    pub unread: usize,
    /// Its worktree branch, if any.
    pub branch: Option<String>,
    /// Commits on `branch` not on the base; `None` while unknown.
    pub unmerged: Option<Vec<String>>,
    /// Its todo list.
    pub todos: Vec<TodoItem>,
    /// Running background jobs: `tool (age)`.
    pub background: Vec<String>,
    /// Schedules posting to it, described.
    pub schedules: Vec<String>,
}

/// What [`crate::Swarm::status`] returns.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StatusReport {
    /// The branch commits are compared against; `None` until known.
    pub base: Option<String>,
    /// Live agents in spawn order.
    pub agents: Vec<AgentReport>,
}

impl StatusReport {
    /// The report as text lines; `doing` adds an agent's current activity after its status.
    pub fn lines(&self, doing: &dyn Fn(&str) -> Option<String>) -> Vec<String> {
        let count = self.agents.len();
        let mut head = format!("{count} live agent{}", if count == 1 { "" } else { "s" });
        if let Some(base) = &self.base {
            head.push_str(&format!(" · base {base}"));
        }
        let base = self.base.as_deref().unwrap_or("the base");
        let mut lines = vec![head];
        for agent in &self.agents {
            let mut head = format!("{} · {}", agent.label, agent.status);
            if agent.unread > 0 {
                head.push_str(&format!(" · {} unread", agent.unread));
            }
            if let Some(doing) = doing(&agent.id).filter(|d| !d.is_empty()) {
                head.push_str(&format!(" · {doing}"));
            }
            lines.push(String::new());
            lines.push(head);
            lines.extend(agent.todos.iter().map(todo_line));
            if let Some(branch) = &agent.branch {
                lines.push(match agent.unmerged.as_deref() {
                    None => format!("  branch {branch} · checking…"),
                    Some([]) => format!("  branch {branch} · nothing beyond {base}"),
                    Some(commits) => {
                        let more = if commits.len() >= MAX_COMMITS { "+" } else { "" };
                        format!("  branch {branch} · {}{more} not on {base}:", commits.len())
                    }
                });
                let commits = agent.unmerged.iter().flatten();
                lines.extend(commits.map(|commit| format!("    {commit}")));
            }
            lines.extend(agent.background.iter().map(|job| format!("  job {job}")));
            lines.extend(agent.schedules.iter().map(|s| format!("  scheduled {s}")));
        }
        lines
    }
}

fn todo_line(item: &TodoItem) -> String {
    match item.status {
        TodoStatus::Pending => format!("  [ ] {}", item.content),
        TodoStatus::InProgress => format!("  [>] {}", item.active_form),
        TodoStatus::Completed => format!("  [x] {}", item.content),
    }
}

/// Look up the base (`base`, else the branch checked out at `root`) and, for each
/// `(agent id, cwd, branch)`, the branch's commits not on it.
pub async fn refresh(
    base: Option<String>,
    root: Option<PathBuf>,
    branches: Vec<(String, PathBuf, String)>,
) -> (Option<String>, HashMap<String, Vec<String>>) {
    let base = match (base, root) {
        (Some(base), _) => Some(base),
        (None, Some(root)) => current_branch(&root).await,
        (None, None) => None,
    };
    let mut commits = HashMap::new();
    if let Some(base) = &base {
        for (id, cwd, branch) in branches {
            if let Some(list) = unmerged(&cwd, base, &branch).await {
                commits.insert(id, list);
            }
        }
    }
    (base, commits)
}

/// The branch checked out in `dir`; `None` when detached or not a repository.
pub async fn current_branch(dir: &Path) -> Option<String> {
    let out = git(dir, &["rev-parse", "--abbrev-ref", "HEAD"]).await?;
    let branch = out.trim();
    (!branch.is_empty() && branch != "HEAD").then(|| branch.to_owned())
}

/// `<short hash> <subject>` of the commits on `branch` not on `base`, newest first, at most
/// [`MAX_COMMITS`]; `None` when git fails.
pub async fn unmerged(dir: &Path, base: &str, branch: &str) -> Option<Vec<String>> {
    let range = format!("{base}..{branch}");
    let max = format!("--max-count={MAX_COMMITS}");
    let out = git(dir, &["log", "--format=%h %s", &max, &range, "--"]).await?;
    Some(out.lines().map(str::to_owned).collect())
}

async fn git(dir: &Path, args: &[&str]) -> Option<String> {
    crate::worktree::git(dir, args).await.ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn todo(content: &str, status: TodoStatus) -> TodoItem {
        TodoItem {
            content: content.into(),
            active_form: format!("{content}ing"),
            status,
        }
    }

    #[test]
    fn the_report_reads_as_one_block_per_agent() {
        let report = StatusReport {
            base: Some("main".into()),
            agents: vec![
                AgentReport {
                    id: "a2".into(),
                    label: "tui (a2)".into(),
                    status: crate::AgentStatus::Running,
                    unread: 1,
                    branch: Some("swarm/r/a2".into()),
                    unmerged: Some(vec!["abc1234 feat: x".into()]),
                    todos: vec![
                        todo("Test", TodoStatus::Completed),
                        todo("Commit", TodoStatus::InProgress),
                        todo("Report", TodoStatus::Pending),
                    ],
                    background: vec!["shell (2m)".into()],
                    schedules: vec!["s1 every 30m, next in 12m (…): poll".into()],
                },
                AgentReport {
                    id: "a3".into(),
                    label: "a3".into(),
                    status: crate::AgentStatus::Idle,
                    unread: 0,
                    branch: Some("swarm/r/a3".into()),
                    unmerged: Some(vec![]),
                    todos: vec![],
                    background: vec![],
                    schedules: vec![],
                },
            ],
        };
        let doing = |id: &str| (id == "a2").then(|| "shell cargo test".to_owned());
        assert_eq!(
            report.lines(&doing),
            [
                "2 live agents · base main",
                "",
                "tui (a2) · running · 1 unread · shell cargo test",
                "  [x] Test",
                "  [>] Commiting",
                "  [ ] Report",
                "  branch swarm/r/a2 · 1 not on main:",
                "    abc1234 feat: x",
                "  job shell (2m)",
                "  scheduled s1 every 30m, next in 12m (…): poll",
                "",
                "a3 · idle",
                "  branch swarm/r/a3 · nothing beyond main",
            ]
        );
    }

    #[test]
    fn an_unknown_branch_state_shows_as_checking() {
        let report = StatusReport {
            base: None,
            agents: vec![AgentReport {
                id: "a1".into(),
                label: "a1".into(),
                status: crate::AgentStatus::Queued,
                unread: 0,
                branch: Some("b".into()),
                unmerged: None,
                todos: vec![],
                background: vec![],
                schedules: vec![],
            }],
        };
        let lines = report.lines(&|_| None);
        assert_eq!(lines[0], "1 live agent", "no base outside a repository");
        assert_eq!(lines[3], "  branch b · checking…");
    }
}
