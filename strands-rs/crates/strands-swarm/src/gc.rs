//! Disk reclamation for cancelled worktree agents.
//!
//! A worktree agent owns two directories: its git worktree at `<repo>.swarm/<run>/<agent>` and,
//! when the swarm has a shared `CARGO_TARGET_DIR`, a private target dir at
//! `<target>-worktrees/<run>/<agent>`. Once the agent is cancelled, its target dir is deleted and
//! its worktree is removed if it is clean. Nothing else is ever touched: each path must have
//! exactly that shape before anything is deleted, the worktree goes through `git worktree remove`
//! (which refuses dirty trees as a second guard), and the branch always stays.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use serde_json::{json, Value};
use tokio::process::Command;

/// What collecting one agent did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Collected {
    /// Agent id.
    pub id: String,
    /// Deleted directories and the bytes they held.
    pub removed: Vec<(PathBuf, u64)>,
    /// Directories left in place, and why.
    pub kept: Vec<(PathBuf, String)>,
}

impl Collected {
    /// Nothing removed and nothing worth reporting kept.
    pub fn is_empty(&self) -> bool {
        self.removed.is_empty() && self.kept.is_empty()
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
        removed.chain(kept).collect()
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
        json!({ "id": self.id, "removed": removed, "kept": kept })
    }
}

/// A summary line for several agents, e.g. `reclaimed 8.1 GB from 2 agents; 1 worktree kept`.
pub fn summary(collected: &[Collected]) -> String {
    let bytes: u64 = collected.iter().map(Collected::bytes).sum();
    let agents = collected.iter().filter(|c| !c.removed.is_empty()).count();
    let kept: usize = collected.iter().map(|c| c.kept.len()).sum();
    let mut text = format!("reclaimed {} from {agents} cancelled agents", human(bytes));
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
        out.push(collect(&state.run_id, &agent.id, branch, &agent.cwd, &agent.env).await);
    }
    out
}

/// Reclaim the directories cancelled agent `id` of run `run_id` owns. Only worktree agents
/// (`branch` set) own any; `cwd` is where the agent worked, `env` its command environment.
pub async fn collect(
    run_id: &str,
    id: &str,
    branch: Option<&str>,
    cwd: &Path,
    env: &BTreeMap<String, String>,
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
    if let Some(root) = worktree_root(run_id, id, cwd).await {
        remove_worktree(&root, branch, &mut out).await;
    }
    out
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
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .output()
        .await
        .map_err(|error| format!("git: {error}"))?;
    match output.status.success() {
        true => Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned()),
        false => Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn run(dir: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(["-c", "user.name=t", "-c", "user.email=t@t"])
            .args(args)
            .status()
            .unwrap();
        assert!(status.success(), "git {args:?}");
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
        std::process::Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(["rev-parse", "--verify", "--quiet", branch])
            .stdout(std::process::Stdio::null())
            .status()
            .unwrap()
            .success()
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
    async fn removes_the_target_dir_and_a_clean_worktree_but_keeps_the_branch() {
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
        )
        .await;

        assert!(collected.kept.is_empty(), "{collected:?}");
        assert_eq!(collected.removed.len(), 2, "{collected:?}");
        assert_eq!(collected.removed[0], (target.clone(), 5000));
        assert!(!target.exists() && !worktree.exists());
        assert!(
            tmp.path().join("target-worktrees/r1").exists(),
            "only the agent's own dir goes"
        );
        assert!(branch_exists(&repo, "swarm/r1/a2"));
        let lines = collected.lines();
        assert!(
            lines[0].starts_with("a2: removed ") && lines[0].ends_with("(4.9 KB)"),
            "{lines:?}"
        );

        let again = collect("r1", "a2", Some("swarm/r1/a2"), &worktree, &env(&target)).await;
        assert!(again.is_empty(), "idempotent: {again:?}");
    }

    #[tokio::test]
    async fn keeps_a_dirty_worktree_and_ignores_agents_without_a_branch() {
        let tmp = tempfile::tempdir().unwrap();
        let (_repo, worktree) = repo_with_worktree(tmp.path());
        fs::write(worktree.join("new.txt"), "untracked").unwrap();
        let target = tmp.path().join("target-worktrees/r1/a2");
        fs::create_dir_all(&target).unwrap();

        let shared = collect("r1", "a2", None, &worktree, &env(&target)).await;
        assert!(
            shared.is_empty() && target.exists(),
            "a non-worktree agent owns nothing"
        );

        let collected = collect("r1", "a2", Some("swarm/r1/a2"), &worktree, &env(&target)).await;
        assert_eq!(
            collected.kept,
            [(worktree.clone(), "uncommitted changes".to_owned())]
        );
        assert!(worktree.join("new.txt").exists());
        assert!(!target.exists(), "the target dir goes regardless");
    }

    #[tokio::test]
    async fn never_touches_a_checkout_outside_the_swarm_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, worktree) = repo_with_worktree(tmp.path());
        // The main checkout, or a worktree of another agent, is never removed.
        let main = collect("r1", "a2", Some("swarm/r1/a2"), &repo, &BTreeMap::new()).await;
        assert!(main.is_empty() && repo.exists(), "{main:?}");
        let other = collect("r1", "a3", Some("swarm/r1/a2"), &worktree, &BTreeMap::new()).await;
        assert!(other.is_empty() && worktree.exists(), "{other:?}");
        // A worktree switched to another branch is reported, not removed.
        run(&worktree, &["checkout", "-q", "-b", "elsewhere"]);
        let switched = collect("r1", "a2", Some("swarm/r1/a2"), &worktree, &BTreeMap::new()).await;
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
