//! Disk reclamation for cancelled worktree agents.
//!
//! A worktree agent owns two directories: its git worktree at `<repo>.swarm/<run>/<agent>` and,
//! when the swarm has a shared `CARGO_TARGET_DIR`, a private target dir at
//! `<target>-worktrees/<run>/<agent>`, and its branch `swarm/<run>/<agent>`. Once the agent is
//! cancelled, its target dir is deleted, its worktree is removed if it is clean, and its branch is
//! deleted once no worktree has it checked out and every commit on it is reachable from the base
//! branch. Nothing else is ever touched: each path must have exactly that shape before anything
//! is deleted, the worktree goes through `git worktree remove` (which refuses dirty trees as a
//! second guard), and the branch ref is only deleted if it still points at the commit checked.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use serde_json::{json, Value};

/// What collecting one agent did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Collected {
    /// Agent id.
    pub id: String,
    /// Deleted directories and the bytes they held.
    pub removed: Vec<(PathBuf, u64)>,
    /// Directories left in place, and why.
    pub kept: Vec<(PathBuf, String)>,
    /// The agent's branch, deleted because the base branch contains all of it.
    pub deleted_branch: Option<String>,
    /// The agent's branch left in place, and why.
    pub kept_branch: Option<(String, String)>,
}

impl Collected {
    /// Nothing removed and nothing worth reporting kept.
    pub fn is_empty(&self) -> bool {
        self.removed.is_empty()
            && self.kept.is_empty()
            && self.deleted_branch.is_none()
            && self.kept_branch.is_none()
    }

    /// Bytes reclaimed.
    pub fn bytes(&self) -> u64 {
        self.removed.iter().map(|(_, bytes)| bytes).sum()
    }

    /// One line per directory, e.g. `a3: removed /x/target-worktrees/run/a3 (4.2 GB)`.
    pub fn lines(&self) -> Vec<String> {
        let removed = self
            .removed
            .iter()
            .map(|(path, bytes)| format!("{}: removed {} ({})", self.id, path.display(), human(*bytes)));
        let kept = self
            .kept
            .iter()
            .map(|(path, why)| format!("{}: kept {} ({why})", self.id, path.display()));
        let deleted = self
            .deleted_branch
            .iter()
            .map(|branch| format!("{}: deleted branch {branch} (merged)", self.id));
        let kept_branch = self
            .kept_branch
            .iter()
            .map(|(branch, why)| format!("{}: kept branch {branch} ({why})", self.id));
        removed.chain(kept).chain(deleted).chain(kept_branch).collect()
    }

    /// Fields of the `gc` journal event.
    pub fn journal_fields(&self) -> Value {
        let removed: Vec<Value> = self
            .removed
            .iter()
            .map(|(path, bytes)| json!({ "path": path, "bytes": bytes }))
            .collect();
        let kept: Vec<Value> = self
            .kept
            .iter()
            .map(|(path, why)| json!({ "path": path, "reason": why }))
            .collect();
        let kept_branch = self
            .kept_branch
            .as_ref()
            .map(|(branch, why)| json!({ "branch": branch, "reason": why }));
        json!({
            "id": self.id,
            "removed": removed,
            "kept": kept,
            "deleted_branch": self.deleted_branch,
            "kept_branch": kept_branch,
        })
    }
}

/// A summary line for several agents, e.g. `reclaimed 8.1 GB from 2 agents; 1 worktree kept`.
pub fn summary(collected: &[Collected]) -> String {
    let bytes: u64 = collected.iter().map(Collected::bytes).sum();
    let agents = collected.iter().filter(|c| !c.removed.is_empty()).count();
    let kept: usize = collected
        .iter()
        .map(|c| c.kept.len() + usize::from(c.kept_branch.is_some()))
        .sum();
    let deleted = collected.iter().filter(|c| c.deleted_branch.is_some()).count();
    let mut text = format!("reclaimed {} from {agents} cancelled agents", human(bytes));
    if deleted > 0 {
        let plural = if deleted == 1 { "" } else { "es" };
        text.push_str(&format!("; deleted {deleted} merged branch{plural}"));
    }
    if kept > 0 {
        text.push_str(&format!("; kept {kept} (see the events journal)"));
    }
    text
}

/// The TUI notice for a reclamation pass, e.g. `gc: reclaimed 8.1 GB from 2 cancelled agents`.
pub fn notice(collected: &[Collected]) -> String {
    match collected.iter().all(Collected::is_empty) {
        true => "gc: nothing to reclaim".to_owned(),
        false => format!("gc: {}", summary(collected)),
    }
}

/// Reclaim the directories of every cancelled agent of a saved run (`--gc`, offline).
pub async fn collect_run(state: &crate::state::SwarmState) -> Vec<Collected> {
    let mut out = Vec::new();
    for agent in state.agents.iter().filter(|agent| agent.status == "cancelled") {
        let branch = agent.branch.as_deref();
        let collected = collect(&state.run_id, &agent.id, branch, &agent.cwd, &agent.env, None);
        out.push(collected.await);
    }
    out
}

/// Reclaim the directories and branch cancelled agent `id` of run `run_id` owns. Only worktree
/// agents (`branch` set) own any; `cwd` is where the agent worked, `env` its command environment.
/// The branch must be merged into `base` (default: the branch the main checkout has out).
pub async fn collect(
    run_id: &str,
    id: &str,
    branch: Option<&str>,
    cwd: &Path,
    env: &BTreeMap<String, String>,
    base: Option<&str>,
) -> Collected {
    let mut out = Collected {
        id: id.to_owned(),
        ..Default::default()
    };
    let Some(branch) = branch else {
        return out;
    };
    if let Some(dir) = private_target_dir(run_id, id, env) {
        remove_dir(&dir, &mut out).await;
    }
    let Some((root, repo)) = layout(run_id, id, cwd) else {
        return out;
    };
    if root.exists() {
        if worktree_root(run_id, id, &root).await.as_ref() != Some(&root) {
            return out;
        }
        remove_worktree(&root, branch, &mut out).await;
    }
    if !root.exists() && branch == format!("swarm/{run_id}/{id}") {
        delete_merged_branch(&repo, branch, base, &mut out).await;
    }
    out
}

/// The agent's worktree root `/…/<repo>.swarm/<run_id>/<id>` among `cwd` and its ancestors, and
/// the main checkout `/…/<repo>` it belongs to. Path shape only: either may be gone.
fn layout(run_id: &str, id: &str, cwd: &Path) -> Option<(PathBuf, PathBuf)> {
    let swarm_dir = |owner: &str| owner.len() > ".swarm".len() && owner.ends_with(".swarm");
    let root = cwd
        .ancestors()
        .find(|dir| owned_path(dir, run_id, id, swarm_dir))?;
    let owner = root.parent()?.parent()?;
    let name = owner.file_name()?.to_str()?.strip_suffix(".swarm")?;
    Some((root.to_path_buf(), owner.parent()?.join(name)))
}

/// Delete `branch` in `repo` if no worktree has it checked out and `base` contains every commit
/// on it; otherwise record why it stays.
async fn delete_merged_branch(repo: &Path, branch: &str, base: Option<&str>, out: &mut Collected) {
    let reference = format!("refs/heads/{branch}");
    if !repo.is_dir() {
        return;
    }
    let Ok(commit) = git(repo, &["rev-parse", "--verify", "--quiet", &reference]).await else {
        return; // Already gone.
    };
    let mut keep = |why: String| out.kept_branch = Some((branch.to_owned(), why));
    let checked_out = match git(repo, &["worktree", "list", "--porcelain"]).await {
        Ok(list) => list.lines().any(|line| line == format!("branch {reference}")),
        Err(error) => return keep(error),
    };
    if checked_out {
        return keep("checked out in a worktree".to_owned());
    }
    let base = match base {
        Some(base) => base.to_owned(),
        None => match git(repo, &["symbolic-ref", "--quiet", "--short", "HEAD"]).await {
            Ok(base) => base,
            Err(_) => return keep("no base branch to compare with".to_owned()),
        },
    };
    if base == branch {
        return keep("it is the base".to_owned());
    }
    let unmerged = git(repo, &["rev-list", "--count", &format!("{base}..{commit}"), "--"]).await;
    match unmerged.as_deref().map(str::trim) {
        Ok("0") => {}
        Ok("1") => return keep(format!("1 commit not on {base}")),
        Ok(count) => return keep(format!("{count} commits not on {base}")),
        Err(error) => return keep(error.clone()),
    }
    // Deleted only if it still points at the commit checked above.
    match git(repo, &["update-ref", "-d", &reference, &commit]).await {
        Ok(_) => out.deleted_branch = Some(branch.to_owned()),
        // A concurrent pass deleted it first.
        Err(_)
            if git(repo, &["rev-parse", "--verify", "--quiet", &reference])
                .await
                .is_err() => {}
        Err(error) => keep(error),
    }
}

/// The agent's private `CARGO_TARGET_DIR`, if it has one of the shape
/// `/…/<name>-worktrees/<run_id>/<id>`.
pub fn private_target_dir(run_id: &str, id: &str, env: &BTreeMap<String, String>) -> Option<PathBuf> {
    let dir = PathBuf::from(env.get("CARGO_TARGET_DIR")?);
    owned_path(&dir, run_id, id, |owner| {
        owner.len() > "-worktrees".len() && owner.ends_with("-worktrees")
    })
    .then_some(dir)
}

/// Whether `path` is absolute, plain (no `.`/`..`) and ends in `<owner>/<run_id>/<id>` with an
/// `owner` that `owner_ok` accepts.
fn owned_path(path: &Path, run_id: &str, id: &str, owner_ok: impl Fn(&str) -> bool) -> bool {
    let plain = path.is_absolute()
        && path
            .components()
            .all(|c| matches!(c, Component::RootDir | Component::Normal(_)));
    let names: Vec<&str> = path
        .components()
        .rev()
        .take(3)
        .filter_map(|c| c.as_os_str().to_str())
        .collect();
    plain && names.len() == 3 && names[0] == id && names[1] == run_id && owner_ok(names[2])
}

async fn remove_dir(dir: &Path, out: &mut Collected) {
    // A symlink is never followed: only a real directory at the expected path is deleted.
    match tokio::fs::symlink_metadata(dir).await {
        Ok(meta) if meta.is_dir() => {}
        _ => return,
    }
    let path = dir.to_path_buf();
    let bytes = tokio::task::spawn_blocking(move || dir_size(&path))
        .await
        .unwrap_or(0);
    match tokio::fs::remove_dir_all(dir).await {
        Ok(()) => out.removed.push((dir.to_path_buf(), bytes)),
        Err(error) => out
            .kept
            .push((dir.to_path_buf(), format!("delete failed: {error}"))),
    }
}

/// The root of the agent's worktree if `cwd` is inside one at `/…/<repo>.swarm/<run_id>/<id>`.
async fn worktree_root(run_id: &str, id: &str, cwd: &Path) -> Option<PathBuf> {
    if !cwd.is_dir() {
        return None;
    }
    let root = PathBuf::from(git(cwd, &["rev-parse", "--show-toplevel"]).await.ok()?);
    let owned = owned_path(&root, run_id, id, |owner| {
        owner.len() > ".swarm".len() && owner.ends_with(".swarm")
    });
    owned.then_some(root)
}

async fn remove_worktree(root: &Path, branch: &str, out: &mut Collected) {
    let keep = |out: &mut Collected, why: String| out.kept.push((root.to_path_buf(), why));
    let head = git(root, &["symbolic-ref", "--quiet", "--short", "HEAD"]).await;
    if head.as_deref() != Ok(branch) {
        return keep(out, format!("not on branch {branch}"));
    }
    match git(root, &["status", "--porcelain"]).await {
        Ok(status) if status.is_empty() => {}
        Ok(_) => return keep(out, "uncommitted changes".to_owned()),
        Err(error) => return keep(out, error),
    }
    let Ok(common) = git(root, &["rev-parse", "--path-format=absolute", "--git-common-dir"]).await else {
        return keep(out, "no main checkout".to_owned());
    };
    let Some(main) = Path::new(&common).parent().map(Path::to_path_buf) else {
        return keep(out, "no main checkout".to_owned());
    };
    let path = root.to_path_buf();
    let bytes = tokio::task::spawn_blocking(move || dir_size(&path))
        .await
        .unwrap_or(0);
    let root_arg = root.to_string_lossy().into_owned();
    // No `--force`: git refuses a worktree with modified or untracked files.
    match git(&main, &["worktree", "remove", &root_arg]).await {
        Ok(_) => out.removed.push((root.to_path_buf(), bytes)),
        Err(error) => keep(out, error),
    }
}

/// Total size of the files under `dir`, without following symlinks.
fn dir_size(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| match entry.metadata() {
            Ok(meta) if meta.is_dir() => dir_size(&entry.path()),
            Ok(meta) => meta.len(),
            Err(_) => 0,
        })
        .sum()
}

/// `4.2 GB`, `310 MB`, `12 KB`, `9 B`.
pub fn human(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KB", "MB", "GB", "TB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    match value < 10.0 {
        true => format!("{value:.1} {}", UNITS[unit]),
        false => format!("{value:.0} {}", UNITS[unit]),
    }
}

async fn git(cwd: &Path, args: &[&str]) -> Result<String, String> {
    crate::worktree::git(cwd, args)
        .await
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn run(dir: &Path, args: &[&str]) {
        crate::worktree::test_git(dir, args);
    }

    /// A repo at `<tmp>/repo` with a worktree for agent `a2` of run `r1`, on its own branch.
    fn repo_with_worktree(tmp: &Path) -> (PathBuf, PathBuf) {
        let repo = tmp.join("repo");
        fs::create_dir_all(repo.join("sub")).unwrap();
        fs::write(repo.join("sub/file.txt"), "x").unwrap();
        fs::write(repo.join(".gitignore"), "target/\n").unwrap();
        run(&repo, &["init", "-q"]);
        run(&repo, &["add", "."]);
        run(&repo, &["commit", "-qm", "init"]);
        let worktree = tmp.join("repo.swarm/r1/a2");
        run(
            &repo,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "swarm/r1/a2",
                worktree.to_str().unwrap(),
                "HEAD",
            ],
        );
        (repo, worktree)
    }

    fn env(dir: &Path) -> BTreeMap<String, String> {
        BTreeMap::from([("CARGO_TARGET_DIR".to_owned(), dir.display().to_string())])
    }

    fn branch_exists(repo: &Path, branch: &str) -> bool {
        let mut git = std::process::Command::new("git");
        git.arg("-C")
            .arg(repo)
            .args(["rev-parse", "--verify", "--quiet", branch]);
        for var in crate::worktree::GIT_LOCATION_VARS {
            git.env_remove(var);
        }
        git.output().unwrap().status.success()
    }

    #[test]
    fn only_paths_of_the_exact_shape_count_as_owned() {
        let ok = |p: &str| private_target_dir("r1", "a2", &env(Path::new(p))).is_some();
        assert!(ok("/c/target-worktrees/r1/a2"));
        assert!(!ok("/c/target/r1/a2"), "not a -worktrees dir");
        assert!(!ok("/c/-worktrees/r1/a2"), "empty owner name");
        assert!(!ok("/c/target-worktrees/r1/a3"), "another agent");
        assert!(!ok("/c/target-worktrees/r2/a2"), "another run");
        assert!(!ok("/c/target-worktrees/r1/a2/.."), "not plain");
        assert!(!ok("c/target-worktrees/r1/a2"), "relative");
        assert!(private_target_dir("r1", "a2", &BTreeMap::new()).is_none());
    }

    #[tokio::test]
    async fn removes_the_target_dir_a_clean_worktree_and_its_merged_branch() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, worktree) = repo_with_worktree(tmp.path());
        let target = tmp.path().join("target-worktrees/r1/a2");
        fs::create_dir_all(target.join("debug")).unwrap();
        fs::write(target.join("debug/big"), vec![0u8; 5000]).unwrap();
        // Ignored build output does not make the worktree dirty.
        fs::create_dir_all(worktree.join("target")).unwrap();
        fs::write(worktree.join("target/out"), "o").unwrap();

        let collected = collect(
            "r1",
            "a2",
            Some("swarm/r1/a2"),
            &worktree.join("sub"),
            &env(&target),
            None,
        )
        .await;

        assert!(
            collected.kept.is_empty() && collected.kept_branch.is_none(),
            "{collected:?}"
        );
        assert_eq!(collected.removed.len(), 2, "{collected:?}");
        assert_eq!(collected.deleted_branch.as_deref(), Some("swarm/r1/a2"));
        assert_eq!(collected.removed[0], (target.clone(), 5000));
        assert!(!target.exists() && !worktree.exists());
        assert!(
            tmp.path().join("target-worktrees/r1").exists(),
            "only the agent's own dir goes"
        );
        assert!(
            !branch_exists(&repo, "swarm/r1/a2"),
            "nothing on it the base lacks"
        );
        let lines = collected.lines();
        assert_eq!(lines[2], "a2: deleted branch swarm/r1/a2 (merged)");
        assert!(
            lines[0].starts_with("a2: removed ") && lines[0].ends_with("(4.9 KB)"),
            "{lines:?}"
        );

        let again = collect("r1", "a2", Some("swarm/r1/a2"), &worktree, &env(&target), None).await;
        assert!(again.is_empty(), "idempotent: {again:?}");
    }

    #[tokio::test]
    async fn keeps_a_branch_with_commits_the_base_lacks_until_they_are_merged() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, worktree) = repo_with_worktree(tmp.path());
        fs::write(worktree.join("work.txt"), "done").unwrap();
        run(&worktree, &["add", "."]);
        run(&worktree, &["commit", "-qm", "work"]);
        let base = crate::worktree::test_git(&repo, &["symbolic-ref", "--short", "HEAD"]);
        let none = BTreeMap::new();

        let collected = collect("r1", "a2", Some("swarm/r1/a2"), &worktree, &none, None).await;
        assert!(!worktree.exists(), "the clean worktree goes: {collected:?}");
        let why = format!("1 commit not on {base}");
        assert_eq!(
            collected.kept_branch,
            Some(("swarm/r1/a2".to_owned(), why.clone()))
        );
        assert!(collected
            .lines()
            .contains(&format!("a2: kept branch swarm/r1/a2 ({why})")));
        assert!(branch_exists(&repo, "swarm/r1/a2"));

        // A branch someone has checked out stays too.
        let elsewhere = tmp.path().join("elsewhere");
        run(
            &repo,
            &[
                "worktree",
                "add",
                "-q",
                elsewhere.to_str().unwrap(),
                "swarm/r1/a2",
            ],
        );
        let busy = collect("r1", "a2", Some("swarm/r1/a2"), &worktree, &none, Some(&base)).await;
        let why = ("swarm/r1/a2".to_owned(), "checked out in a worktree".to_owned());
        assert_eq!(busy.kept_branch, Some(why));
        run(&repo, &["worktree", "remove", elsewhere.to_str().unwrap()]);

        // Once merged, a later pass (e.g. on resume) deletes it.
        run(&repo, &["merge", "-q", "--ff-only", "swarm/r1/a2"]);
        let merged = collect("r1", "a2", Some("swarm/r1/a2"), &worktree, &none, None).await;
        assert_eq!(
            merged.deleted_branch.as_deref(),
            Some("swarm/r1/a2"),
            "{merged:?}"
        );
        assert!(!branch_exists(&repo, "swarm/r1/a2"));
        assert!(summary(&[merged]).contains("deleted 1 merged branch"));
    }

    #[tokio::test]
    async fn keeps_a_dirty_worktree_and_ignores_agents_without_a_branch() {
        let tmp = tempfile::tempdir().unwrap();
        let (_repo, worktree) = repo_with_worktree(tmp.path());
        fs::write(worktree.join("new.txt"), "untracked").unwrap();
        let target = tmp.path().join("target-worktrees/r1/a2");
        fs::create_dir_all(&target).unwrap();

        let shared = collect("r1", "a2", None, &worktree, &env(&target), None).await;
        assert!(
            shared.is_empty() && target.exists(),
            "a non-worktree agent owns nothing"
        );

        let collected = collect("r1", "a2", Some("swarm/r1/a2"), &worktree, &env(&target), None).await;
        assert_eq!(
            collected.kept,
            [(worktree.clone(), "uncommitted changes".to_owned())]
        );
        assert!(worktree.join("new.txt").exists());
        assert!(!target.exists(), "the target dir goes regardless");
        assert!(collected.deleted_branch.is_none() && collected.kept_branch.is_none());
        assert!(
            branch_exists(&_repo, "swarm/r1/a2"),
            "a kept worktree keeps its branch"
        );
    }

    #[tokio::test]
    async fn never_touches_a_checkout_outside_the_swarm_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, worktree) = repo_with_worktree(tmp.path());
        // The main checkout, or a worktree of another agent, is never removed.
        let main = collect("r1", "a2", Some("swarm/r1/a2"), &repo, &BTreeMap::new(), None).await;
        assert!(main.is_empty() && repo.exists(), "{main:?}");
        let other = collect("r1", "a3", Some("swarm/r1/a2"), &worktree, &BTreeMap::new(), None).await;
        assert!(other.is_empty() && worktree.exists(), "{other:?}");
        // A worktree switched to another branch is reported, not removed.
        run(&worktree, &["checkout", "-q", "-b", "elsewhere"]);
        let switched = collect("r1", "a2", Some("swarm/r1/a2"), &worktree, &BTreeMap::new(), None).await;
        assert_eq!(switched.kept.len(), 1, "{switched:?}");
        assert!(worktree.exists());
    }

    #[tokio::test]
    async fn a_saved_run_is_collected_for_its_cancelled_agents_only() {
        let tmp = tempfile::tempdir().unwrap();
        let (_repo, worktree) = repo_with_worktree(tmp.path());
        let agent = |id: &str, status: &str| crate::state::AgentState {
            id: id.into(),
            name: None,
            parent: "user".into(),
            cwd: worktree.clone(),
            branch: Some(format!("swarm/r1/{id}")),
            task: String::new(),
            model: None,
            goal: None,
            status: status.into(),
            mailbox: Vec::new(),
            env: env(&tmp.path().join(format!("target-worktrees/r1/{id}"))),
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            ledger: Default::default(),
            context: None,
            paused: false,
        };
        fs::create_dir_all(tmp.path().join("target-worktrees/r1/a1")).unwrap();
        fs::create_dir_all(tmp.path().join("target-worktrees/r1/a2")).unwrap();
        let state = crate::state::SwarmState {
            run_id: "r1".into(),
            cwd: tmp.path().to_path_buf(),
            next_id: 3,
            agents: vec![agent("a1", "idle"), agent("a2", "cancelled")],
            budget_tokens: None,
            max_agents: None,
            commit: None,
            schedules: Vec::new(),
        };
        let collected = collect_run(&state).await;
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].removed.len(), 2, "{collected:?}");
        assert!(
            tmp.path().join("target-worktrees/r1/a1").exists(),
            "live agents keep theirs"
        );
        assert!(summary(&collected).starts_with("reclaimed "));
    }

    #[test]
    fn sizes_read_like_du() {
        assert_eq!(human(9), "9 B");
        assert_eq!(human(5000), "4.9 KB");
        assert_eq!(human(310 << 20), "310 MB");
        assert_eq!(human(4_500_000_000), "4.2 GB");
    }
}
