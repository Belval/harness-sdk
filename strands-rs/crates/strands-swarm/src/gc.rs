//! Disk reclamation for cancelled worktree agents.
//!
//! A worktree agent owns two directories: its git worktree at `<repo>.swarm/<run>/<agent>` and,
//! when the swarm has a shared `CARGO_TARGET_DIR`, a private target dir at
//! `<target>-worktrees/<run>/<agent>`, and its branch `swarm/<run>/<agent>`. Once the agent is
//! cancelled, its target dir is deleted, its worktree is removed if it is clean, and its branch is
//! deleted once no worktree has it checked out and the base has every commit on it (reachable,
//! or a cherry-picked copy by patch id). Nothing else is ever touched: each path must have
//! exactly that shape before anything is deleted, the worktree goes through `git worktree remove`
//! (which refuses dirty trees as a second guard), and the branch ref is only deleted if it still
//! points at the commit checked.

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
    /// The agent's branch, deleted because the base has all of it, and how: `merged` (reachable
    /// from it) or `all applied` (every commit has a patch-equivalent copy there).
    pub deleted_branch: Option<(String, String)>,
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
            .map(|(branch, how)| format!("{}: deleted branch {branch} ({how})", self.id));
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
            "deleted_branch": self.deleted_branch.as_ref().map(|(branch, how)| json!({ "branch": branch, "how": how })),
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
pub async fn collect_run(state: &crate::state::SwarmState, base: Option<&str>) -> Vec<Collected> {
    // The base as the running swarm has it: the branch at the root agent's own checkout.
    let base = match base {
        Some(base) => Some(base.to_owned()),
        None => {
            let root = state
                .agents
                .iter()
                .find(|a| a.parent == crate::USER && a.branch.is_none());
            let root = root.map_or(&state.cwd, |agent| &agent.cwd);
            crate::status::current_branch(root).await
        }
    };
    let mut out = Vec::new();
    for agent in state.agents.iter().filter(|agent| agent.status == "cancelled") {
        let branch = agent.branch.as_deref();
        let collected = collect(
            &state.run_id,
            &agent.id,
            branch,
            &agent.cwd,
            &agent.env,
            base.as_deref(),
        );
        out.push(collected.await);
    }
    out
}

/// Reclaim the directories and branch cancelled agent `id` of run `run_id` owns. Only worktree
/// agents (`branch` set) own any; `cwd` is where the agent worked, `env` its command environment.
/// The branch must be merged into `base` (default: the branch the repository's main worktree
/// has out; the swarm passes the root agent's branch, which may be another worktree's).
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
    let base = match base {
        Some(base) => Some(base.to_owned()),
        None if repo.is_dir() => git(&repo, &["symbolic-ref", "--quiet", "--short", "HEAD"])
            .await
            .ok(),
        None => None,
    };
    if root.exists() {
        if worktree_root(run_id, id, &root).await.as_ref() != Some(&root) {
            return out;
        }
        remove_worktree(&root, branch, base.as_deref(), &mut out).await;
    }
    if !root.exists() && branch == format!("swarm/{run_id}/{id}") {
        delete_merged_branch(&repo, branch, base.as_deref(), &mut out).await;
    }
    out
}

/// How `base` already has every commit of `rev`: `merged`, or `all applied` (cherry-picked or
/// rebased, same patch id); else why not.
async fn integrated(dir: &Path, base: Option<&str>, rev: &str) -> Result<&'static str, String> {
    let base = base.ok_or("no base branch to compare with")?;
    match crate::status::applied(dir, base, rev).await? {
        found if found.pending() == 0 && found.applied == 0 => Ok("merged"),
        found if found.pending() == 0 => Ok("all applied"),
        found => {
            let count = found.pending();
            let commits = if count == 1 { "commit" } else { "commits" };
            Err(format!("{count} {commits} not on {base}"))
        }
    }
}

/// The agent's worktree root `/…/<repo>.swarm/<run_id>/<id>` among `cwd` and its ancestors, and
/// the main checkout `/…/<repo>` it belongs to. Path shape only: either may be gone.
fn layout(run_id: &str, id: &str, cwd: &Path) -> Option<(PathBuf, PathBuf)> {
    let root = cwd
        .ancestors()
        .find(|dir| owned_path(dir, run_id, id, ".swarm"))?;
    let owner = root.parent()?.parent()?;
    let name = owner.file_name()?.to_str()?.strip_suffix(".swarm")?;
    Some((root.to_path_buf(), owner.parent()?.join(name)))
}

/// Delete `branch` in `repo` if no worktree has it checked out and `base` (`None`: unknown)
/// contains every commit on it; otherwise record why it stays.
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
    if base == Some(branch) {
        return keep("it is the base".to_owned());
    }
    let how = match integrated(repo, base, &commit).await {
        Ok(how) => how,
        Err(why) => return keep(why),
    };
    // Deleted only if it still points at the commit checked above.
    match git(repo, &["update-ref", "-d", &reference, &commit]).await {
        Ok(_) => out.deleted_branch = Some((branch.to_owned(), how.to_owned())),
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
    owned_path(&dir, run_id, id, "-worktrees").then_some(dir)
}

/// Whether `path` is absolute, plain (no `.`/`..`) and ends in `<name><suffix>/<run_id>/<id>`
/// with a non-empty `name`.
fn owned_path(path: &Path, run_id: &str, id: &str, suffix: &str) -> bool {
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
    plain
        && names.len() == 3
        && names[0] == id
        && names[1] == run_id
        && names[2].strip_suffix(suffix).is_some_and(|name| !name.is_empty())
}

async fn remove_dir(dir: &Path, out: &mut Collected) {
    // A symlink is never followed: only a real directory at the expected path is deleted.
    match tokio::fs::symlink_metadata(dir).await {
        Ok(meta) if meta.is_dir() => {}
        _ => return,
    }
    let bytes = size(dir).await;
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
    owned_path(&root, run_id, id, ".swarm").then_some(root)
}

/// Why a worktree with `git status --porcelain` output `status` stays, naming the first paths:
/// often only build output (`?? tests/__pycache__/`) that its owner can delete.
fn uncommitted(status: &str) -> String {
    const SHOWN: usize = 3;
    let lines: Vec<&str> = status.lines().collect();
    let mut shown = lines.iter().take(SHOWN).copied().collect::<Vec<_>>().join(", ");
    if lines.len() > SHOWN {
        shown.push_str(&format!(", {} more", lines.len() - SHOWN));
    }
    format!("uncommitted changes: {shown}")
}

/// Remove the worktree at `root` if it is clean and on `branch`, or on another branch `base`
/// already has all of (which nothing then tracks); otherwise record why it stays.
async fn remove_worktree(root: &Path, branch: &str, base: Option<&str>, out: &mut Collected) {
    let keep = |out: &mut Collected, why: String| out.kept.push((root.to_path_buf(), why));
    match git(root, &["symbolic-ref", "--quiet", "--short", "HEAD"]).await {
        Ok(head) if head == branch => {}
        Ok(head) => {
            if let Err(why) = integrated(root, base, &head).await {
                return keep(out, format!("on branch {head}: {why}"));
            }
        }
        Err(_) => return keep(out, format!("not on branch {branch}")),
    }
    match git(root, &["status", "--porcelain"]).await {
        Ok(status) if status.is_empty() => {}
        Ok(status) => return keep(out, uncommitted(&status)),
        Err(error) => return keep(out, error),
    }
    let common = git(root, &["rev-parse", "--path-format=absolute", "--git-common-dir"]).await;
    let Some(main) = common.as_deref().ok().and_then(|dir| Path::new(dir).parent()) else {
        return keep(out, "no main checkout".to_owned());
    };
    let bytes = size(root).await;
    let root_arg = root.to_string_lossy().into_owned();
    // No `--force`: git refuses a worktree with modified or untracked files.
    match git(main, &["worktree", "remove", &root_arg]).await {
        Ok(_) => {
            out.removed.push((root.to_path_buf(), bytes));
            // `<repo>.swarm/<run>`, then `<repo>.swarm`, once nothing else is in them: a
            // non-empty directory is not removed.
            for dir in root.ancestors().skip(1).take(2) {
                if std::fs::remove_dir(dir).is_err() {
                    break;
                }
            }
        }
        Err(error) => keep(out, error),
    }
}

/// [`dir_size`] off the async runtime.
async fn size(dir: &Path) -> u64 {
    let dir = dir.to_path_buf();
    tokio::task::spawn_blocking(move || dir_size(&dir))
        .await
        .unwrap_or(0)
}

/// Total size of the files under `dir`, without following symlinks.
pub(crate) fn dir_size(dir: &Path) -> u64 {
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
        let deleted = ("swarm/r1/a2".to_owned(), "merged".to_owned());
        assert_eq!(collected.deleted_branch, Some(deleted));
        assert_eq!(collected.removed[0], (target.clone(), 5000));
        assert!(!target.exists() && !worktree.exists());
        assert!(
            tmp.path().join("target-worktrees/r1").exists(),
            "only the agent's own dir goes"
        );
        assert!(
            !tmp.path().join("repo.swarm").exists(),
            "the worktree's emptied run dir and .swarm dir go too"
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
        let deleted = ("swarm/r1/a2".to_owned(), "merged".to_owned());
        assert_eq!(merged.deleted_branch, Some(deleted), "{merged:?}");
        assert!(!branch_exists(&repo, "swarm/r1/a2"));
        assert!(summary(&[merged]).contains("deleted 1 merged branch"));
    }

    #[tokio::test]
    async fn a_branch_cherry_picked_onto_the_base_counts_as_merged() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, worktree) = repo_with_worktree(tmp.path());
        for name in ["one", "two"] {
            fs::write(worktree.join(name), name).unwrap();
            run(&worktree, &["add", "."]);
            run(&worktree, &["commit", "-qm", name]);
        }
        fs::write(repo.join("other"), "x").unwrap();
        run(&repo, &["add", "."]);
        run(&repo, &["commit", "-qm", "other work on main"]);
        run(&repo, &["cherry-pick", "swarm/r1/a2~1"]);
        let none = BTreeMap::new();

        let half = collect("r1", "a2", Some("swarm/r1/a2"), &worktree, &none, None).await;
        let base = crate::worktree::test_git(&repo, &["symbolic-ref", "--short", "HEAD"]);
        let why = ("swarm/r1/a2".to_owned(), format!("1 commit not on {base}"));
        assert_eq!(half.kept_branch, Some(why), "only one of two is applied");

        // A branch that merged the base in is never "all applied": the merge commit may carry
        // changes of its own (a conflict resolution), which `git cherry` does not compare.
        let merged_in = tmp.path().join("merged-in");
        let path = merged_in.to_str().unwrap();
        run(
            &repo,
            &["worktree", "add", "-q", "-b", "side", path, "swarm/r1/a2"],
        );
        run(&merged_in, &["merge", "-q", "--no-edit", &base]);

        run(&repo, &["cherry-pick", "swarm/r1/a2"]);
        let all = collect("r1", "a2", Some("swarm/r1/a2"), &worktree, &none, None).await;
        let deleted = ("swarm/r1/a2".to_owned(), "all applied".to_owned());
        assert_eq!(all.deleted_branch, Some(deleted), "{all:?}");
        assert!(!branch_exists(&repo, "swarm/r1/a2"));
        assert!(all
            .lines()
            .contains(&"a2: deleted branch swarm/r1/a2 (all applied)".to_owned()));
        let side = crate::status::applied(&repo, &base, "side").await.unwrap();
        assert!(side.merges == 1 && side.pending() > 0, "{side:?}");
    }

    #[tokio::test]
    async fn a_worktree_on_another_branch_goes_once_the_base_has_all_of_it() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, worktree) = repo_with_worktree(tmp.path());
        let base = crate::worktree::test_git(&repo, &["symbolic-ref", "--short", "HEAD"]);
        run(&worktree, &["checkout", "-q", "-b", "swarm/r1/a2-fix"]);
        fs::write(worktree.join("fix.txt"), "fix").unwrap();
        run(&worktree, &["add", "."]);
        run(&worktree, &["commit", "-qm", "fix"]);
        let none = BTreeMap::new();
        let reclaim = || collect("r1", "a2", Some("swarm/r1/a2"), &worktree, &none, None);

        let unmerged = reclaim().await;
        let why = format!("on branch swarm/r1/a2-fix: 1 commit not on {base}");
        assert_eq!(unmerged.kept, [(worktree.clone(), why)]);
        assert!(worktree.exists());
        let busy = ("swarm/r1/a2".to_owned(), "checked out in a worktree".to_owned());
        assert_ne!(unmerged.kept_branch, Some(busy), "its given branch is not out");

        // Detached, its commits may be on no branch.
        run(&worktree, &["checkout", "-q", "--detach"]);
        let detached = reclaim().await;
        let why = "not on branch swarm/r1/a2".to_owned();
        assert_eq!(detached.kept, [(worktree.clone(), why)]);
        run(&worktree, &["checkout", "-q", "swarm/r1/a2-fix"]);

        // Cherry-picked onto the base: nothing is lost with the worktree.
        run(&repo, &["cherry-pick", "swarm/r1/a2-fix"]);
        let applied = reclaim().await;
        assert!(applied.kept.is_empty(), "{applied:?}");
        assert!(!worktree.exists());
        let deleted = ("swarm/r1/a2".to_owned(), "merged".to_owned());
        assert_eq!(applied.deleted_branch, Some(deleted), "the given branch goes too");
        assert!(
            branch_exists(&repo, "swarm/r1/a2-fix"),
            "a branch it was not given stays"
        );
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
            [(worktree.clone(), "uncommitted changes: ?? new.txt".to_owned())]
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
    async fn a_run_dir_with_another_worktree_in_it_stays() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, worktree) = repo_with_worktree(tmp.path());
        let sibling = tmp.path().join("repo.swarm/r1/a3");
        let sibling_arg = sibling.to_str().unwrap();
        run(
            &repo,
            &["worktree", "add", "-q", "-b", "swarm/r1/a3", sibling_arg, "HEAD"],
        );
        let target = tmp.path().join("target-worktrees/r1/a2");

        let collected = collect("r1", "a2", Some("swarm/r1/a2"), &worktree, &env(&target), None).await;

        assert_eq!(collected.removed.len(), 1, "{collected:?}");
        assert!(!worktree.exists() && sibling.exists());
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
    }

    fn saved_agent(
        id: &str,
        cwd: &Path,
        branch: Option<&str>,
        status: &str,
        env: BTreeMap<String, String>,
    ) -> crate::state::AgentState {
        crate::state::AgentState {
            id: id.into(),
            name: None,
            parent: crate::USER.into(),
            cwd: cwd.to_path_buf(),
            branch: branch.map(str::to_owned),
            task: String::new(),
            model: None,
            goal: None,
            status: status.into(),
            mailbox: Vec::new(),
            env,
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            ledger: Default::default(),
            context: None,
            paused: false,
            running_calls: Default::default(),
        }
    }

    fn saved_run(cwd: &Path, agents: Vec<crate::state::AgentState>) -> crate::state::SwarmState {
        crate::state::SwarmState {
            run_id: "r1".into(),
            cwd: cwd.to_path_buf(),
            next_id: agents.len() as u64 + 1,
            agents,
            budget_tokens: None,
            budget_flag: None,
            max_agents: None,
            commit: None,
            schedules: Vec::new(),
        }
    }

    #[tokio::test]
    async fn a_saved_run_is_collected_for_its_cancelled_agents_only() {
        let tmp = tempfile::tempdir().unwrap();
        let (_repo, worktree) = repo_with_worktree(tmp.path());
        let agent = |id: &str, status: &str| {
            let target = tmp.path().join(format!("target-worktrees/r1/{id}"));
            fs::create_dir_all(&target).unwrap();
            let branch = format!("swarm/r1/{id}");
            saved_agent(id, &worktree, Some(&branch), status, env(&target))
        };
        let agents = vec![agent("a1", "idle"), agent("a2", "cancelled")];
        let state = saved_run(tmp.path(), agents);
        let collected = collect_run(&state, None).await;
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].removed.len(), 2, "{collected:?}");
        assert!(
            tmp.path().join("target-worktrees/r1/a1").exists(),
            "live agents keep theirs"
        );
        assert!(summary(&collected).starts_with("reclaimed "));
    }

    #[tokio::test]
    async fn a_saved_run_compares_with_the_root_agents_branch() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, worktree) = repo_with_worktree(tmp.path());
        fs::write(worktree.join("work.txt"), "done").unwrap();
        run(&worktree, &["add", "."]);
        run(&worktree, &["commit", "-qm", "work"]);
        // The root agent ran in a linked worktree whose branch has the work; the main one has not.
        let root = tmp.path().join("integration");
        let root_arg = root.to_str().unwrap();
        run(
            &repo,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "integration",
                root_arg,
                "swarm/r1/a2",
            ],
        );
        let agents = vec![
            saved_agent("a1", &root, None, "idle", BTreeMap::new()),
            saved_agent("a2", &worktree, Some("swarm/r1/a2"), "cancelled", BTreeMap::new()),
        ];
        let state = saved_run(&repo, agents);
        let collected = collect_run(&state, None).await;
        let deleted = ("swarm/r1/a2".to_owned(), "merged".to_owned());
        assert_eq!(collected[0].deleted_branch, Some(deleted), "{collected:?}");
        assert!(!branch_exists(&repo, "swarm/r1/a2"));
    }

    #[test]
    fn sizes_read_like_du() {
        assert_eq!(human(9), "9 B");
        assert_eq!(human(5000), "4.9 KB");
        assert_eq!(human(310 << 20), "310 MB");
        assert_eq!(human(4_500_000_000), "4.2 GB");
    }
}
