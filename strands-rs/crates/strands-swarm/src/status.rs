//! A model-free status report of the swarm ([`crate::Swarm::status`], the TUI's `/status`): per
//! live agent its state, todo list, branch commits not yet on the base (and whether the branch
//! would merge into it cleanly), background jobs and schedules. The git part runs in the
//! background and is cached for [`TTL`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use strands_harness::plugins::{TodoItem, TodoStatus};

/// How long cached branch commits are served before a background refresh.
pub const TTL: Duration = Duration::from_secs(5);

/// Most commits listed per branch.
pub const MAX_COMMITS: usize = 20;

/// Most conflicted files named per branch.
pub const MAX_CONFLICTS: usize = 10;

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
    /// Agent id → how its branch stands against the base.
    pub branches: HashMap<String, BranchState>,
}

/// How an agent's branch stands against the base branch.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BranchState {
    /// `<short hash> <subject>` of its commits not on the base, newest first, at most
    /// [`MAX_COMMITS`].
    pub ahead: Vec<String>,
    /// Commits on the base not on the branch.
    pub behind: usize,
    /// How many of its commits not on the base have a patch-equivalent commit there.
    pub applied: usize,
    /// How many have none, merge commits included; 0 with `applied > 0` means nothing is left
    /// to integrate.
    pub pending: usize,
    /// Whether merging the branch into the base would conflict.
    pub merge: Merge,
    /// The branch checked out in the agent's cwd, which the rest describes, when it is not the one
    /// the agent was given.
    pub head: Option<String>,
    /// The agent's other branches (its own and `<its own>-*`) with commits the base lacks.
    pub others: Vec<String>,
}

/// The outcome of a trial merge (`git merge-tree`) of a branch into the base.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Merge {
    /// Nothing to merge, or the base can fast-forward to the branch.
    #[default]
    FastForward,
    /// A real merge without conflicts.
    Clean,
    /// The files that would conflict (at most [`MAX_CONFLICTS`]).
    Conflicts(Vec<String>),
    /// The trial merge failed (e.g. git older than 2.38).
    Unknown,
}

impl BranchState {
    /// `3 not on main, 5 behind, conflicts in README.md; also unmerged: swarm/r/a2-old`: the
    /// summary after the branch name.
    pub fn describe(&self, base: &str) -> String {
        let mut text = self.compared(base);
        if !self.others.is_empty() {
            text.push_str(&format!("; also unmerged: {}", self.others.join(", ")));
        }
        text
    }

    /// [`Self::describe`] of the compared branch alone.
    fn compared(&self, base: &str) -> String {
        let more = if self.ahead.len() >= MAX_COMMITS { "+" } else { "" };
        let mut text = match self.ahead.len() {
            0 => format!("nothing beyond {base}"),
            n => format!("{n}{more} not on {base}"),
        };
        let all_applied = self.applied > 0 && self.pending == 0;
        match (all_applied, self.applied) {
            (true, _) => text.push_str(" (all applied)"),
            (false, 0) => {}
            (false, n) => text.push_str(&format!(" ({n} applied)")),
        }
        if self.behind > 0 {
            text.push_str(&format!(", {} behind", self.behind));
        }
        if self.ahead.is_empty() || all_applied {
            return text;
        }
        match &self.merge {
            Merge::FastForward => text.push_str(", fast-forward"),
            Merge::Clean => text.push_str(", merges cleanly"),
            Merge::Conflicts(files) => text.push_str(&format!(", conflicts in {}", files.join(", "))),
            Merge::Unknown => {}
        }
        text
    }
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
    /// How `branch` stands against the base; `None` while unknown.
    pub git: Option<BranchState>,
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
            let head = agent.git.as_ref().and_then(|state| state.head.as_ref());
            if let Some(branch) = head.or(agent.branch.as_ref()) {
                lines.push(match &agent.git {
                    None => format!("  branch {branch} · checking…"),
                    Some(state) if state.ahead.is_empty() => {
                        format!("  branch {branch} · {}", state.describe(base))
                    }
                    Some(state) => format!("  branch {branch} · {}:", state.describe(base)),
                });
                let commits = agent.git.iter().flat_map(|state| &state.ahead);
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
/// `(agent id, cwd, branch)`, how what is checked out in `cwd` stands against it (`branch` when
/// detached), and which of the agent's other branches the base still lacks.
pub async fn refresh(
    base: Option<String>,
    root: Option<PathBuf>,
    branches: Vec<(String, PathBuf, String)>,
) -> (Option<String>, HashMap<String, BranchState>) {
    let base = match (base, root) {
        (Some(base), _) => Some(base),
        (None, Some(root)) => current_branch(&root).await,
        (None, None) => None,
    };
    let mut states = HashMap::new();
    if let Some(base) = &base {
        for (id, cwd, branch) in branches {
            let head = current_branch(&cwd).await.unwrap_or_else(|| branch.clone());
            if let Some(mut state) = branch_state(&cwd, base, &head).await {
                state.others = other_unmerged(&cwd, base, &branch, &head).await;
                state.head = (head != branch).then_some(head);
                states.insert(id, state);
            }
        }
    }
    (base, states)
}

/// How `branch` stands against `base`, seen from the repository at `dir`; `None` when git fails.
pub async fn branch_state(dir: &Path, base: &str, branch: &str) -> Option<BranchState> {
    let ahead = unmerged(dir, base, branch).await?;
    let behind = git(dir, &["rev-list", "--count", &format!("{branch}..{base}"), "--"]).await?;
    let behind = behind.trim().parse().ok()?;
    let mut state = BranchState {
        ahead,
        behind,
        ..Default::default()
    };
    if state.ahead.is_empty() || behind == 0 {
        return Some(state);
    }
    // Only a base that moved can hold copies (cherry-picks, rebases) of the branch's commits.
    if let Ok(found) = applied(dir, base, branch).await {
        (state.applied, state.pending) = (found.applied, found.pending());
    }
    if state.applied == 0 || state.pending > 0 {
        state.merge = trial_merge(dir, base, branch).await;
    }
    Some(state)
}

/// `branch` and the `<branch>-*` branches, `head` aside, with commits whose changes `base` lacks
/// (neither merged nor cherry-picked there).
async fn other_unmerged(dir: &Path, base: &str, branch: &str, head: &str) -> Vec<String> {
    let no_merged = format!("--no-merged={base}");
    let (own, derived) = (format!("refs/heads/{branch}"), format!("refs/heads/{branch}-*"));
    let args = [
        "for-each-ref",
        "--format=%(refname:lstrip=2)",
        &no_merged,
        &own,
        &derived,
    ];
    let Some(out) = git(dir, &args).await else {
        return Vec::new();
    };
    let mut others = Vec::new();
    for name in out.lines().filter(|name| !name.is_empty() && *name != head) {
        if !matches!(applied(dir, base, name).await, Ok(found) if found.pending() == 0) {
            others.push(name.to_owned());
        }
    }
    others
}

/// Which of the commits on `rev` not on `base` the base already has a copy of: by patch id
/// (`git cherry`), e.g. after a cherry-pick or rebase, or because `land` landed them
/// ([`crate::land::landed_sources`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Applied {
    /// Commits with a patch-equivalent commit on the base.
    pub applied: usize,
    /// Commits without one.
    pub unapplied: usize,
    /// Merge commits, which `git cherry` skips; their changes (e.g. conflict resolutions) are not
    /// compared, so they count as not applied.
    pub merges: usize,
}

impl Applied {
    /// Commits whose changes the base may lack.
    pub fn pending(&self) -> usize {
        self.unapplied + self.merges
    }
}

/// [`Applied`] for `rev` against `base`, in repository `dir`.
pub async fn applied(dir: &Path, base: &str, rev: &str) -> Result<Applied, String> {
    use crate::worktree::git;
    // The commits of `base..rev` no landing covered: not reachable from a landed source.
    let not_base = format!("^{base}");
    let landed: Vec<String> = crate::land::landed_sources(dir, base)
        .await
        .iter()
        .map(|source| format!("^{source}"))
        .collect();
    let mut uncovered = vec![rev, &not_base, "--ignore-missing"];
    uncovered.extend(landed.iter().map(String::as_str));
    let cherry = git(dir, &["cherry", base, rev]).await?;
    let merges = git(
        dir,
        &[&["rev-list", "--merges", "--count"][..], &uncovered].concat(),
    )
    .await?;
    let mut out = Applied {
        merges: merges
            .trim()
            .parse()
            .map_err(|_| format!("bad count {merges:?}"))?,
        ..Default::default()
    };
    let landed_too = match landed.is_empty() {
        true => None,
        false => Some(git(dir, &[&["rev-list"][..], &uncovered].concat()).await?),
    };
    for line in cherry.lines() {
        let sha = line.get(2..).unwrap_or_default();
        let covered = landed_too
            .as_ref()
            .is_some_and(|uncovered| !uncovered.lines().any(|line| line == sha));
        match line.chars().next() {
            Some('-') => out.applied += 1,
            Some('+') if covered => out.applied += 1,
            Some('+') => out.unapplied += 1,
            _ => {}
        }
    }
    Ok(out)
}

/// Merge `branch` into `base` without touching any checkout or ref (`git merge-tree`).
async fn trial_merge(dir: &Path, base: &str, branch: &str) -> Merge {
    let args = [
        "merge-tree",
        "--write-tree",
        "--name-only",
        "--no-messages",
        base,
        branch,
    ];
    let Ok(output) = crate::worktree::git_in(dir).args(args).output().await else {
        return Merge::Unknown;
    };
    match output.status.code() {
        Some(0) => Merge::Clean,
        // Exit 1: conflicts; stdout is the tree id, then one conflicted file per line.
        Some(1) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut files: Vec<String> = Vec::new();
            for file in stdout.lines().skip(1).filter(|line| !line.is_empty()) {
                if !files.iter().any(|seen| seen == file) {
                    files.push(file.to_owned());
                }
            }
            files.truncate(MAX_CONFLICTS);
            Merge::Conflicts(files)
        }
        _ => Merge::Unknown,
    }
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
                    git: Some(BranchState {
                        ahead: vec!["abc1234 feat: x".into()],
                        behind: 2,
                        merge: Merge::Conflicts(vec!["README.md".into(), "src/a.rs".into()]),
                        ..Default::default()
                    }),
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
                    git: Some(BranchState::default()),
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
                "  branch swarm/r/a2 · 1 not on main, 2 behind, conflicts in README.md, src/a.rs:",
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
                git: None,
                todos: vec![],
                background: vec![],
                schedules: vec![],
            }],
        };
        let lines = report.lines(&|_| None);
        assert_eq!(lines[0], "1 live agent", "no base outside a repository");
        assert_eq!(lines[3], "  branch b · checking…");
    }

    #[test]
    fn a_branch_summary_says_how_it_would_merge() {
        let state = |ahead: usize, behind, merge| BranchState {
            ahead: vec!["c".into(); ahead],
            behind,
            merge,
            ..Default::default()
        };
        let applied = |ahead: usize, applied, merge| BranchState {
            applied,
            pending: ahead - applied,
            ..state(ahead, 4, merge)
        };
        let cases = [
            (state(0, 0, Merge::FastForward), "nothing beyond main"),
            (state(0, 3, Merge::FastForward), "nothing beyond main, 3 behind"),
            (state(2, 0, Merge::FastForward), "2 not on main, fast-forward"),
            (
                state(2, 3, Merge::Clean),
                "2 not on main, 3 behind, merges cleanly",
            ),
            (state(2, 3, Merge::Unknown), "2 not on main, 3 behind"),
            (
                state(MAX_COMMITS, 1, Merge::Clean),
                "20+ not on main, 1 behind, merges cleanly",
            ),
            (
                applied(3, 3, Merge::FastForward),
                "3 not on main (all applied), 4 behind",
            ),
            (
                applied(3, 1, Merge::Clean),
                "3 not on main (1 applied), 4 behind, merges cleanly",
            ),
        ];
        for (state, text) in cases {
            assert_eq!(state.describe("main"), text);
        }
    }

    #[tokio::test]
    async fn the_branch_state_comes_from_a_trial_merge_that_changes_nothing() {
        use crate::worktree::test_git as git;
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path();
        let write = |file: &str, text: &str| std::fs::write(repo.join(file), text).unwrap();
        let commit = |message: &str| {
            git(repo, &["add", "-A"]);
            git(repo, &["commit", "-q", "-m", message]);
        };
        git(repo, &["init", "-q", "-b", "main"]);
        write("a.txt", "a\n");
        write("b.txt", "b\n");
        commit("init");
        git(repo, &["checkout", "-q", "-b", "work"]);
        write("a.txt", "mine\n");
        commit("edit a");
        let state = branch_state(repo, "main", "work").await.unwrap();
        assert_eq!((state.ahead.len(), state.behind), (1, 0));
        assert!(state.ahead[0].ends_with(" edit a"), "{state:?}");
        assert_eq!(state.merge, Merge::FastForward);

        git(repo, &["checkout", "-q", "main"]);
        write("b.txt", "theirs\n");
        commit("edit b");
        let state = branch_state(repo, "main", "work").await.unwrap();
        assert_eq!((state.ahead.len(), state.behind), (1, 1));
        assert_eq!(state.merge, Merge::Clean);

        write("a.txt", "theirs\n");
        commit("edit a too");
        let main = git(repo, &["rev-parse", "main"]);
        let state = branch_state(repo, "main", "work").await.unwrap();
        assert_eq!(state.behind, 2);
        assert_eq!(state.merge, Merge::Conflicts(vec!["a.txt".into()]));
        assert_eq!(git(repo, &["rev-parse", "main"]), main, "no ref moved");
        assert_eq!(
            git(repo, &["status", "--porcelain"]),
            "",
            "the checkout is untouched"
        );
        assert_eq!(branch_state(repo, "main", "nope").await, None);

        // Cherry-picked onto main, the branch's commit is there under another hash.
        git(repo, &["checkout", "-q", "-b", "picked", "main~2"]);
        write("c.txt", "c\n");
        commit("add c");
        git(repo, &["checkout", "-q", "main"]);
        git(repo, &["cherry-pick", "picked"]);
        let state = branch_state(repo, "main", "picked").await.unwrap();
        assert_eq!((state.ahead.len(), state.applied, state.pending), (1, 1, 0));
        assert_eq!(state.describe("main"), "1 not on main (all applied), 3 behind");
    }

    #[tokio::test]
    async fn the_branch_state_is_of_what_the_agent_has_checked_out() {
        use crate::worktree::test_git as git;
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path();
        let commit = |file: &str| {
            std::fs::write(repo.join(file), file).unwrap();
            git(repo, &["add", "-A"]);
            git(repo, &["commit", "-q", "-m", file]);
        };
        git(repo, &["init", "-q", "-b", "main"]);
        commit("init");
        // Given `swarm/r/a7`, the agent commits there, then works on branches of its own.
        git(repo, &["checkout", "-q", "-b", "swarm/r/a7"]);
        commit("given");
        git(repo, &["checkout", "-q", "-b", "swarm/r/a7-merged", "main"]);
        commit("merged");
        git(repo, &["checkout", "-q", "-b", "swarm/r/a7-picked", "main"]);
        commit("picked");
        git(repo, &["checkout", "-q", "-b", "swarm/r/a70", "main"]);
        commit("someone else's");
        git(repo, &["checkout", "-q", "main"]);
        git(
            repo,
            &["merge", "-q", "--no-ff", "-m", "merge", "swarm/r/a7-merged"],
        );
        git(repo, &["cherry-pick", "swarm/r/a7-picked"]);
        git(repo, &["checkout", "-q", "-b", "swarm/r/a7-fix"]);
        commit("fix");

        let agents = vec![("a7".to_owned(), repo.to_path_buf(), "swarm/r/a7".to_owned())];
        let (base, states) = refresh(Some("main".into()), None, agents.clone()).await;
        assert_eq!(base.as_deref(), Some("main"));
        let state = &states["a7"];
        assert_eq!(state.head.as_deref(), Some("swarm/r/a7-fix"));
        assert_eq!((state.ahead.len(), state.behind), (1, 0), "{state:?}");
        assert!(state.ahead[0].ends_with(" fix"), "{state:?}");
        assert_eq!(state.others, ["swarm/r/a7"]);
        assert_eq!(
            state.describe("main"),
            "1 not on main, fast-forward; also unmerged: swarm/r/a7"
        );
        let report = StatusReport {
            base,
            agents: vec![AgentReport {
                id: "a7".into(),
                label: "a7".into(),
                status: crate::AgentStatus::Running,
                unread: 0,
                branch: Some("swarm/r/a7".into()),
                git: Some(state.clone()),
                todos: vec![],
                background: vec![],
                schedules: vec![],
            }],
        };
        assert_eq!(
            report.lines(&|_| None)[3],
            "  branch swarm/r/a7-fix · 1 not on main, fast-forward; also unmerged: swarm/r/a7:"
        );

        // Back on its own branch, that is what is compared.
        git(repo, &["checkout", "-q", "swarm/r/a7"]);
        let (_, states) = refresh(Some("main".into()), None, agents).await;
        let state = &states["a7"];
        assert_eq!(state.head, None);
        assert!(state.ahead[0].ends_with(" given"), "{state:?}");
        assert_eq!(state.others, ["swarm/r/a7-fix"]);
    }
}
