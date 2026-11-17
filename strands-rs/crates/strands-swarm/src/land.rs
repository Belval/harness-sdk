//! Landing an agent's commits on a branch it does not own ([`crate::Swarm::land`], the `land`
//! tool): rebase them onto the target in a scratch worktree, run the checks there, and
//! fast-forward the target only if everything passed. A conflict or a failing check leaves the
//! target, its checkout and the agent's own branch exactly as they were.
//!
//! [`run`] does the work. The swarm runs it as a background job (`strands-swarm --land PLAN`) so
//! that slow checks survive a restart and the outcome reaches the agent as a job message.

use std::path::{Path, PathBuf};
use std::time::Instant;

use nix::errno::Errno;
use nix::fcntl::{Flock, FlockArg};
use serde::{Deserialize, Serialize};

use strands_agents::sandbox::detach_from_terminal;

use crate::worktree::{git, git_in, same_subdir};

/// Lines of a failing check's output that the report keeps.
pub const TAIL_LINES: usize = 30;

/// File in the repository's common git dir recording what was landed, one `<source> <landed>`
/// commit pair per line: the rebase gives the landed commits new ids and, where the lines around
/// them changed, new patch ids, so `git cherry` alone cannot tell the source's commits are in.
pub const LANDED_FILE: &str = "swarm-landed";

/// File in the repository's common git dir that land jobs hold an exclusive lock on from checking
/// that the target has not moved to recording the landing, so two of them never fast-forward at
/// once (two `git merge`s race for the checkout's `index.lock`).
pub const LOCK_FILE: &str = "swarm-land.lock";

/// What sets a land job's command apart: `<exe> --land <plan>`.
pub const LAND_FLAG: &str = " --land ";

/// Most landings [`LANDED_FILE`] keeps (the latest).
const LANDED_KEPT: usize = 500;

/// Everything one landing needs; written to disk for the job that carries it out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LandPlan {
    /// The agent's working directory: a checkout of the repository; checks run in the same
    /// subdirectory of the scratch worktree.
    pub repo: PathBuf,
    /// Commit to land (the tip of the agent's branch when it asked).
    pub source: String,
    /// What `source` was called, for the report.
    pub label: String,
    /// Branch to fast-forward.
    pub into: String,
    /// Shell commands that must all succeed; none means no checks.
    pub checks: Vec<String>,
    /// Scratch worktree to create (and remove afterwards).
    pub scratch: PathBuf,
    /// `CARGO_TARGET_DIR` for the checks.
    pub target_dir: PathBuf,
}

/// Land `plan`: `Ok` with a report when `into` was fast-forwarded (or had nothing to take),
/// `Err` with the reason (conflicted files, a failing check's output tail, a dirty or moved
/// target) when nothing was changed.
pub async fn run(plan: &LandPlan) -> Result<String, String> {
    run_until(plan, std::future::pending()).await
}

/// [`run`], given up once `stop` resolves (to why): the check in flight is killed with its
/// children, the scratch worktree is removed, and the report says it was stopped.
pub async fn run_until(
    plan: &LandPlan,
    stop: impl std::future::Future<Output = String>,
) -> Result<String, String> {
    let repo = &plan.repo;
    let into = &plan.into;
    let into_ref = format!("refs/heads/{into}");
    let old = git(
        repo,
        &["rev-parse", "--verify", &format!("{into_ref}^{{commit}}")],
    )
    .await
    .map_err(|_| format!("Not landed: there is no branch {into}."))?;
    let checkout = checked_out(repo, &into_ref).await;
    if let Some(dir) = &checkout {
        ensure_clean(dir, into).await?;
    }
    let pending = log(repo, &old, &plan.source).await?;
    if pending.is_empty() {
        return Ok(format!(
            "Nothing to land: {} has no commits that are not on {into}.",
            plan.label
        ));
    }
    let _ = tokio::fs::remove_dir_all(&plan.scratch).await;
    let scratch = plan.scratch.to_string_lossy().into_owned();
    git(repo, &["worktree", "add", "--detach", &scratch, &plan.source])
        .await
        .map_err(|e| format!("Not landed: could not create the scratch worktree: {e}"))?;
    let outcome = tokio::select! {
        outcome = land_in_scratch(plan, &old, checkout.as_deref()) => outcome,
        why = stop => {
            let now = git(repo, &["rev-parse", &into_ref]).await.unwrap_or_default();
            Err(match now == old {
                true => format!("Not landed: {why}. Nothing was changed."),
                false => format!(
                    "Stopped: {why} after {into} moved from {} to {}, which may hold the landing.",
                    short(&old),
                    short(&now)
                ),
            })
        }
    };
    let _ = git(repo, &["worktree", "remove", "--force", &scratch]).await;
    let _ = tokio::fs::remove_dir_all(&plan.scratch).await;
    let _ = git(repo, &["worktree", "prune"]).await;
    outcome
}

async fn land_in_scratch(plan: &LandPlan, old: &str, checkout: Option<&Path>) -> Result<String, String> {
    let (scratch, into, label) = (&plan.scratch, &plan.into, &plan.label);
    if let Err(error) = git(scratch, &["rebase", "--quiet", old]).await {
        let conflicted = git(scratch, &["diff", "--name-only", "--diff-filter=U"])
            .await
            .unwrap_or_default();
        let _ = git(scratch, &["rebase", "--abort"]).await;
        if conflicted.is_empty() {
            return Err(format!(
                "Not landed: rebasing {label} onto {into} failed: {error}"
            ));
        }
        return Err(format!(
            "Not landed: {label} conflicts with {into} ({}) in:\n{conflicted}\nRebase onto {into}, \
             resolve, commit, and land again. Nothing was changed.",
            short(old)
        ));
    }
    let new = git(scratch, &["rev-parse", "HEAD"])
        .await
        .map_err(|e| format!("Not landed: {e}"))?;
    let commits = log(scratch, old, &new).await?;
    if commits.is_empty() {
        return Ok(format!("Nothing to land: {into} already has {label}'s changes."));
    }
    let toplevel = git(&plan.repo, &["rev-parse", "--show-toplevel"])
        .await
        .map_err(|e| format!("Not landed: {e}"))?;
    let dir = same_subdir(scratch, Path::new(&toplevel), &plan.repo);
    let mut passed = Vec::new();
    for check in &plan.checks {
        let started = Instant::now();
        check_passes(&dir, check, &plan.target_dir).await?;
        passed.push(format!("  ok `{check}` ({}s)", started.elapsed().as_secs()));
    }

    let _lock = land_lock(&plan.repo).await?;
    let into_ref = format!("refs/heads/{into}");
    let now = git(scratch, &["rev-parse", &into_ref]).await.unwrap_or_default();
    if now != old {
        return Err(format!(
            "Not landed: {into} moved from {} to {} meanwhile. Land again.",
            short(old),
            short(&now)
        ));
    }
    let moved = match checkout {
        Some(dir) => {
            ensure_clean(dir, into).await?;
            git(dir, &["merge", "--ff-only", "--quiet", &new]).await
        }
        None => git(scratch, &["update-ref", &into_ref, &new, old]).await,
    };
    moved.map_err(|e| format!("Not landed: could not fast-forward {into}: {e}"))?;
    record_landed(&plan.repo, &plan.source, &new).await;

    let checks = match passed.is_empty() {
        true => "No checks ran.".to_owned(),
        false => format!("Checks passed:\n{}", passed.join("\n")),
    };
    Ok(format!(
        "Landed {} commit(s) of {label} on {into} ({}..{}):\n{commits}\n{checks}\n{label} itself \
         is unchanged; `git rebase {into}` drops the landed commits from it.",
        commits.lines().count(),
        short(old),
        short(&new)
    ))
}

/// What a land job killed outright (`kill -9`, a reboot) left in `land_dir` (the swarm's
/// `jobs/land`): scratch worktrees, unregistered from their repository and deleted, as
/// [`crate::gc::Collected`] under the id `land`. Call it only while no land job runs; the
/// shared target dir (`target`, without `CARGO_TARGET_DIR`) is left alone.
pub async fn tidy(land_dir: &Path) -> crate::gc::Collected {
    let mut out = crate::gc::Collected {
        id: "land".into(),
        ..Default::default()
    };
    let Ok(mut entries) = tokio::fs::read_dir(land_dir).await else {
        return out;
    };
    let mut scratches = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let is_dir = entry.file_type().await.is_ok_and(|kind| kind.is_dir());
        if is_dir && entry.file_name() != "target" {
            scratches.push(entry.path());
        }
    }
    scratches.sort();
    for scratch in scratches {
        let plan = tokio::fs::read(scratch.with_extension("json")).await.ok();
        let plan = plan.and_then(|bytes| serde_json::from_slice::<LandPlan>(&bytes).ok());
        let bytes = crate::gc::dir_size(&scratch);
        if let Some(plan) = &plan {
            let path = scratch.to_string_lossy();
            let _ = git(&plan.repo, &["worktree", "remove", "--force", "--force", &path]).await;
        }
        if scratch.exists() {
            if let Err(error) = tokio::fs::remove_dir_all(&scratch).await {
                out.kept.push((scratch, format!("delete failed: {error}")));
                continue;
            }
        }
        if let Some(plan) = &plan {
            let _ = git(&plan.repo, &["worktree", "prune"]).await;
        }
        out.removed.push((scratch, bytes));
    }
    out
}

/// The landed-commit record of the repository at `dir`.
async fn landed_file(dir: &Path) -> Option<PathBuf> {
    common_file(dir, LANDED_FILE).await
}

/// File `name` in the common git dir of the repository at `dir`.
async fn common_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let common = git(dir, &["rev-parse", "--path-format=absolute", "--git-common-dir"]).await;
    Some(PathBuf::from(common.ok()?).join(name))
}

/// Wait for the exclusive lock on [`LOCK_FILE`]; it is held until dropped (or the process dies).
/// Polled, so a stopped landing gives up waiting at once.
async fn land_lock(repo: &Path) -> Result<Flock<std::fs::File>, String> {
    let path = common_file(repo, LOCK_FILE)
        .await
        .ok_or("Not landed: could not find the repository's git dir")?;
    let failed = |e: &dyn std::fmt::Display| format!("Not landed: could not lock {}: {e}", path.display());
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .map_err(|e| failed(&e))?;
    loop {
        match Flock::lock(file, FlockArg::LockExclusiveNonblock) {
            Ok(lock) => return Ok(lock),
            Err((unlocked, Errno::EWOULDBLOCK)) => {
                file = unlocked;
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            Err((_, e)) => return Err(failed(&e)),
        }
    }
}

/// Note in [`LANDED_FILE`] that `source`'s commits landed as `landed`. Best effort: a missing
/// record only means a landed branch keeps looking unmerged.
async fn record_landed(repo: &Path, source: &str, landed: &str) {
    let (Some(path), Ok(source)) = (
        landed_file(repo).await,
        git(repo, &["rev-parse", &format!("{source}^{{commit}}")]).await,
    ) else {
        return;
    };
    let old = tokio::fs::read_to_string(&path).await.unwrap_or_default();
    let mut lines: Vec<&str> = old.lines().collect();
    let line = format!("{source} {landed}");
    lines.push(&line);
    let start = lines.len().saturating_sub(LANDED_KEPT);
    let text = lines[start..].join("\n") + "\n";
    let tmp = path.with_extension("tmp");
    let written = match tokio::fs::write(&tmp, text).await {
        Ok(()) => tokio::fs::rename(&tmp, &path).await,
        Err(error) => Err(error),
    };
    if let Err(error) = written {
        tracing::warn!(%error, "could not record the landing");
    }
}

/// The landed sources (see [`LANDED_FILE`]) whose landed commits `base` still has: every commit
/// reachable from them is on `base` in substance.
pub async fn landed_sources(dir: &Path, base: &str) -> Vec<String> {
    let Some(path) = landed_file(dir).await else {
        return Vec::new();
    };
    let text = tokio::fs::read_to_string(&path).await.unwrap_or_default();
    let pairs: Vec<(&str, &str)> = text.lines().filter_map(|line| line.split_once(' ')).collect();
    if pairs.is_empty() {
        return Vec::new();
    }
    // One walk for all: the landed commits `base` lacks (e.g. after a reset). It fails if one
    // is gone altogether (so not on `base` either); then each is asked about on its own.
    let mut args = vec!["rev-list"];
    args.extend(pairs.iter().map(|(_, landed)| *landed));
    let not_base = format!("^{base}");
    args.extend([not_base.as_str(), "--"]);
    let lacked = match git(dir, &args).await {
        Ok(lacked) => lacked,
        Err(_) => {
            let mut sources = Vec::new();
            for (source, landed) in &pairs {
                let is_in = ["merge-base", "--is-ancestor", landed, base];
                if git(dir, &is_in).await.is_ok() {
                    sources.push(source.to_string());
                }
            }
            return sources;
        }
    };
    let lacked: std::collections::HashSet<&str> = lacked.lines().collect();
    pairs
        .iter()
        .filter(|(_, landed)| !lacked.contains(landed))
        .map(|(source, _)| source.to_string())
        .collect()
}

/// Run `check` with `sh` in `dir`; on failure the error carries the tail of its output.
/// SIGKILLs the process group led by its pid when dropped, unless that was cleared: a check given
/// up on (its landing stopped) does not run on, cargo and test binaries included.
struct KillGroupOnDrop(Option<u32>);

impl Drop for KillGroupOnDrop {
    fn drop(&mut self) {
        if let Some(pid) = self.0 {
            let _ = std::process::Command::new("kill")
                .args(["-KILL", "--", &format!("-{pid}")])
                .stderr(std::process::Stdio::null())
                .status();
        }
    }
}

/// The first SIGTERM, SIGINT or SIGHUP after this call, named: what stops a land job (`jobs`
/// kill, its agent cancelled) so that it cleans up ([`run_until`]).
pub fn stop_signal() -> impl std::future::Future<Output = String> {
    use tokio::signal::unix::{signal, SignalKind};
    let streams = (
        signal(SignalKind::terminate()),
        signal(SignalKind::interrupt()),
        signal(SignalKind::hangup()),
    );
    async move {
        let (Ok(mut term), Ok(mut int), Ok(mut hup)) = streams else {
            return std::future::pending().await;
        };
        let name = tokio::select! {
            _ = term.recv() => "SIGTERM",
            _ = int.recv() => "SIGINT",
            _ = hup.recv() => "SIGHUP",
        };
        format!("the land job was stopped ({name})")
    }
}

async fn check_passes(dir: &Path, check: &str, target_dir: &Path) -> Result<(), String> {
    let mut command = tokio::process::Command::new("sh");
    detach_from_terminal(&mut command)
        .arg("-c")
        .arg(format!("exec 2>&1\n{check}"))
        .current_dir(dir)
        .env("CARGO_TARGET_DIR", target_dir)
        .stdin(std::process::Stdio::null());
    for var in crate::worktree::GIT_LOCATION_VARS {
        command.env_remove(var);
    }
    let child = command
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Not landed: could not run `{check}`: {e}"))?;
    // The check leads a session of its own, out of reach of a signal to the land job's group.
    let mut group = KillGroupOnDrop(child.id());
    let output = child
        .wait_with_output()
        .await
        .map_err(|e| format!("Not landed: could not run `{check}`: {e}"))?;
    group.0 = None;
    if output.status.success() {
        return Ok(());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = text.lines().collect();
    let tail = lines[lines.len().saturating_sub(TAIL_LINES)..].join("\n");
    let code = output
        .status
        .code()
        .map_or_else(|| "a signal".to_owned(), |code| code.to_string());
    Err(format!(
        "Not landed: `{check}` failed (exit {code}) on the rebased commits. Nothing was changed. \
         Last lines:\n{tail}"
    ))
}

/// The worktree that has `branch_ref` checked out, if any.
async fn checked_out(repo: &Path, branch_ref: &str) -> Option<PathBuf> {
    let list = git(repo, &["worktree", "list", "--porcelain"]).await.ok()?;
    let mut path = None;
    for line in list.lines() {
        if let Some(dir) = line.strip_prefix("worktree ") {
            path = Some(PathBuf::from(dir));
        } else if line.strip_prefix("branch ") == Some(branch_ref) {
            return path;
        }
    }
    None
}

/// Refuse when `dir` (where `into` is checked out) has uncommitted changes to tracked files.
async fn ensure_clean(dir: &Path, into: &str) -> Result<(), String> {
    let output = git_in(dir)
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .await
        .map_err(|e| format!("Not landed: git status failed: {e}"))?;
    let status = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() || !status.trim().is_empty() {
        return Err(format!(
            "Not landed: {into} is checked out at {} with uncommitted changes; they must be \
             committed or stashed first. Nothing was changed.\n{}",
            dir.display(),
            status.trim_end()
        ));
    }
    Ok(())
}

/// `<short hash> <subject>` of the commits in `from..to`, newest first.
async fn log(dir: &Path, from: &str, to: &str) -> Result<String, String> {
    git(dir, &["log", "--format=%h %s", &format!("{from}..{to}"), "--"])
        .await
        .map_err(|e| format!("Not landed: {e}"))
}

fn short(sha: &str) -> &str {
    &sha[..sha.len().min(9)]
}

/// Whether `text` matches `pattern`, where `*` stands for any run of characters (`/` included)
/// and `?` for one character.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let (pattern, text): (Vec<char>, Vec<char>) = (pattern.chars().collect(), text.chars().collect());
    let (mut p, mut t) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while t < text.len() {
        if p < pattern.len() && (pattern[p] == '?' || pattern[p] == text[t]) {
            p += 1;
            t += 1;
        } else if p < pattern.len() && pattern[p] == '*' {
            star = Some((p, t));
            p += 1;
        } else if let Some((star_p, star_t)) = star {
            p = star_p + 1;
            t = star_t + 1;
            star = Some((star_p, star_t + 1));
        } else {
            return false;
        }
    }
    pattern[p..].iter().all(|&c| c == '*')
}

/// `s` as one `sh` word.
pub fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        _dir: tempfile::TempDir,
        /// Main checkout, on `main`.
        main: PathBuf,
        /// Worktree on `feature`, one commit ahead of `main`.
        feature: PathBuf,
        land: PathBuf,
    }

    async fn sh(dir: &Path, script: &str) {
        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(script)
            .current_dir(dir)
            .output()
            .await
            .unwrap();
        assert!(output.status.success(), "{script}: {output:?}");
    }

    async fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let main = dir.path().join("repo");
        let feature = dir.path().join("feature");
        std::fs::create_dir_all(main.join("sub")).unwrap();
        sh(
            &main,
            "git init -q -b main && git config user.name t && git config user.email t@t && \
             echo base > a.txt && git add . && git commit -qm init",
        )
        .await;
        sh(
            &main,
            &format!("git worktree add -q -b feature {}", feature.display()),
        )
        .await;
        sh(
            &feature,
            "echo new > b.txt && git add . && git commit -qm 'add b'",
        )
        .await;
        let land = dir.path().join("land");
        Fixture {
            main,
            feature,
            land,
            _dir: dir,
        }
    }

    fn plan(fx: &Fixture, checks: &[&str]) -> LandPlan {
        LandPlan {
            repo: fx.feature.clone(),
            source: "feature".into(),
            label: "feature".into(),
            into: "main".into(),
            checks: checks.iter().map(|c| c.to_string()).collect(),
            scratch: fx.land.join("scratch"),
            target_dir: fx.land.join("target"),
        }
    }

    async fn head(dir: &Path, rev: &str) -> String {
        git(dir, &["rev-parse", rev]).await.unwrap()
    }

    async fn worktrees(dir: &Path) -> usize {
        let list = git(dir, &["worktree", "list", "--porcelain"]).await.unwrap();
        list.lines().filter(|l| l.starts_with("worktree ")).count()
    }

    #[tokio::test]
    async fn a_clean_branch_with_passing_checks_fast_forwards_the_checked_out_target() {
        let fx = fixture().await;
        // `main` moves on independently: landing rebases `feature` onto it.
        sh(
            &fx.main,
            "echo more > c.txt && git add . && git commit -qm 'add c'",
        )
        .await;
        let feature_before = head(&fx.main, "feature").await;
        let checks = [
            "test -f b.txt && test -f c.txt",
            r#"test "$CARGO_TARGET_DIR" = "$EXPECTED""#,
        ];
        let mut plan = plan(&fx, &checks);
        plan.checks[1] = plan.checks[1].replace("$EXPECTED", &plan.target_dir.display().to_string());
        let report = run(&plan).await.unwrap();
        assert!(
            report.starts_with("Landed 1 commit(s) of feature on main"),
            "{report}"
        );
        assert!(report.contains("add b"), "{report}");
        assert!(report.contains("ok `test -f b.txt"), "{report}");
        assert!(fx.main.join("b.txt").is_file(), "the main checkout follows");
        assert_eq!(
            git(&fx.main, &["status", "--porcelain"]).await.unwrap(),
            "",
            "and is clean"
        );
        let log = git(&fx.main, &["log", "--format=%s", "main"]).await.unwrap();
        assert_eq!(log, "add b\nadd c\ninit", "linear, no merge commit");
        assert_eq!(
            head(&fx.main, "feature").await,
            feature_before,
            "own branch untouched"
        );
        assert_eq!(worktrees(&fx.main).await, 2, "scratch worktree removed");
        assert!(!plan.scratch.exists());
    }

    #[tokio::test]
    async fn checks_run_in_the_same_subdirectory_and_an_unchecked_out_target_moves_by_ref() {
        let fx = fixture().await;
        sh(&fx.main, "git checkout -q -b other").await;
        sh(
            &fx.feature,
            "mkdir -p sub && echo x > sub/x.txt && git add . && git commit -qm sub",
        )
        .await;
        let mut plan = plan(&fx, &["test -f x.txt"]);
        plan.repo = fx.feature.join("sub");
        let report = run(&plan).await.unwrap();
        assert!(report.contains("Landed 2 commit(s)"), "{report}");
        assert_eq!(head(&fx.main, "main").await, head(&fx.main, "feature").await);
        assert_eq!(head(&fx.main, "HEAD").await, head(&fx.main, "other").await);
    }

    /// A rebase that changes a landed commit's context gives it a new patch id, so `git cherry`
    /// sees it as missing from the target; the landing record says it is there. Commits made
    /// after the landing, or landings the target lost, still count as pending.
    #[tokio::test]
    async fn landed_commits_count_as_applied_even_when_their_patch_id_changed() {
        let fx = fixture().await;
        sh(
            &fx.main,
            "printf 'l1\\nl2\\nl3\\nl4\\n' > c.txt && git add . && git commit -qm c",
        )
        .await;
        sh(&fx.feature, "git rebase -q main").await;
        sh(&fx.main, "sed -i s/l1/L1/ c.txt && git commit -qam 'main l1'").await;
        sh(
            &fx.feature,
            "sed -i s/l3/L3/ c.txt && git commit -qam 'feature l3'",
        )
        .await;
        let applied = || crate::status::applied(&fx.feature, "main", "feature");
        assert_eq!(applied().await.unwrap().pending(), 2);

        let report = run(&plan(&fx, &[])).await.unwrap();
        assert!(report.contains("Landed 2 commit(s)"), "{report}");
        let cherry = git(&fx.feature, &["cherry", "main", "feature"]).await.unwrap();
        assert!(cherry.contains('+'), "the patch id changed: {cherry}");
        let found = applied().await.unwrap();
        assert_eq!((found.pending(), found.applied), (0, 2), "{found:?}");

        sh(
            &fx.feature,
            "echo later > d.txt && git add . && git commit -qm later",
        )
        .await;
        assert_eq!(applied().await.unwrap().pending(), 1, "a later commit is pending");
        sh(&fx.main, "git reset -q --hard HEAD~2").await;
        assert_eq!(
            applied().await.unwrap().pending(),
            3,
            "the target lost the landing"
        );
        // A landing whose commits are gone altogether (garbage-collected) is ignored too.
        let record = fx.main.join(".git").join(LANDED_FILE);
        let gone = format!("{} {}\n", "1".repeat(40), "2".repeat(40));
        let text = std::fs::read_to_string(&record).unwrap() + &gone;
        std::fs::write(&record, text).unwrap();
        sh(&fx.main, "git reset -q --hard ORIG_HEAD").await;
        let found = applied().await.unwrap();
        assert_eq!((found.pending(), found.applied), (1, 2), "{found:?}");
    }

    #[tokio::test]
    async fn a_conflict_names_the_files_and_changes_nothing() {
        let fx = fixture().await;
        sh(
            &fx.main,
            "echo mine > b.txt && git add . && git commit -qm 'main b'",
        )
        .await;
        let main_before = head(&fx.main, "main").await;
        let error = run(&plan(&fx, &["true"])).await.unwrap_err();
        assert!(error.contains("conflicts with main"), "{error}");
        assert!(error.contains("\nb.txt\n"), "{error}");
        assert_eq!(head(&fx.main, "main").await, main_before);
        assert_eq!(worktrees(&fx.main).await, 2, "scratch worktree removed");
        assert_eq!(git(&fx.feature, &["status", "--porcelain"]).await.unwrap(), "");
    }

    #[tokio::test]
    async fn a_failing_check_reports_its_output_tail_and_changes_nothing() {
        let fx = fixture().await;
        let main_before = head(&fx.main, "main").await;
        let error = run(&plan(
            &fx,
            &["true", "echo boom; echo detail >&2; exit 3", "touch never"],
        ))
        .await
        .unwrap_err();
        assert!(
            error.contains("`echo boom; echo detail >&2; exit 3` failed (exit 3)"),
            "{error}"
        );
        assert!(error.ends_with("boom\ndetail"), "{error}");
        assert_eq!(head(&fx.main, "main").await, main_before);
        assert!(!fx.main.join("b.txt").exists());
        assert_eq!(worktrees(&fx.main).await, 2);
    }

    #[tokio::test]
    async fn a_dirty_target_checkout_is_refused() {
        let fx = fixture().await;
        sh(&fx.main, "echo edited > a.txt").await;
        let main_before = head(&fx.main, "main").await;
        let error = run(&plan(&fx, &[])).await.unwrap_err();
        assert!(error.contains("uncommitted changes"), "{error}");
        assert!(error.contains("a.txt"), "{error}");
        assert_eq!(head(&fx.main, "main").await, main_before);
        assert_eq!(
            std::fs::read_to_string(fx.main.join("a.txt")).unwrap(),
            "edited\n"
        );
        assert!(
            !plan(&fx, &[]).scratch.exists(),
            "refused before any scratch work"
        );
    }

    #[tokio::test]
    async fn nothing_to_land_is_a_success_without_changes() {
        let fx = fixture().await;
        let mut plan = plan(&fx, &["false"]);
        plan.source = "main".into();
        let report = run(&plan).await.unwrap();
        assert!(report.starts_with("Nothing to land"), "{report}");
        let error = run(&LandPlan {
            into: "nope".into(),
            ..plan
        })
        .await
        .unwrap_err();
        assert!(error.contains("no branch nope"), "{error}");
    }

    #[tokio::test]
    async fn a_landing_waits_for_another_s_fast_forward_and_then_sees_the_target_moved() {
        let fx = fixture().await;
        let before = head(&fx.main, "main").await;
        let held = std::fs::File::create(fx.main.join(".git").join(LOCK_FILE)).unwrap();
        let held = Flock::lock(held, FlockArg::LockExclusive).unwrap();
        let checked = fx.land.join("checked");
        let plan = plan(&fx, &[&format!("touch '{}'", checked.display())]);
        let landing = tokio::spawn(async move { run(&plan).await });
        while !checked.exists() {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        assert_eq!(
            head(&fx.main, "main").await,
            before,
            "fast-forwarded under another's lock"
        );

        // The other land job fast-forwards `main`, then lets go.
        sh(&fx.main, "echo c > c.txt && git add . && git commit -qm 'add c'").await;
        drop(held);
        let error = landing.await.unwrap().unwrap_err();

        assert!(
            error.contains("moved from") && error.contains("meanwhile"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn tidying_removes_the_scratch_worktree_of_a_killed_land_job_and_keeps_the_target() {
        let fx = fixture().await;
        let plan = plan(&fx, &[]);
        std::fs::create_dir_all(&fx.land).unwrap();
        std::fs::write(fx.land.join("scratch.json"), serde_json::to_vec(&plan).unwrap()).unwrap();
        // What `kill -9` leaves: the job's scratch worktree, mid-rebase or mid-check.
        let path = plan.scratch.to_string_lossy().into_owned();
        git(&fx.main, &["worktree", "add", "-q", "--detach", &path, "feature"])
            .await
            .unwrap();
        std::fs::write(plan.scratch.join("built.o"), "x").unwrap();
        std::fs::create_dir_all(fx.land.join("target/debug")).unwrap();

        let tidied = tidy(&fx.land).await;

        assert_eq!(tidied.id, "land");
        assert_eq!(tidied.removed.len(), 1, "{tidied:?}");
        assert_eq!(tidied.removed[0].0, plan.scratch);
        assert!(tidied.kept.is_empty(), "{tidied:?}");
        assert!(!plan.scratch.exists());
        assert_eq!(worktrees(&fx.main).await, 2, "main and feature only");
        assert!(
            fx.land.join("target/debug").exists(),
            "the shared target dir stays"
        );
        assert!(tidy(&fx.land).await.is_empty());
    }

    #[tokio::test]
    async fn a_stopped_landing_kills_its_check_and_removes_its_scratch_worktree() {
        let fx = fixture().await;
        let before = head(&fx.main, "main").await;
        let started = fx.land.join("started");
        let late = fx.land.join("late");
        // A check whose child would outlive it: a test binary under `cargo test`, say.
        let check = format!(
            "touch '{}'; (sleep 2; touch '{}') & wait",
            started.display(),
            late.display()
        );
        let stop = async {
            while !started.exists() {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            "the land job was stopped (SIGTERM)".to_owned()
        };

        let error = run_until(&plan(&fx, &[check.as_str()]), stop).await.unwrap_err();

        assert_eq!(
            error,
            "Not landed: the land job was stopped (SIGTERM). Nothing was changed."
        );
        assert_eq!(head(&fx.main, "main").await, before);
        assert!(!fx.land.join("scratch").exists());
        assert_eq!(worktrees(&fx.main).await, 2, "main and feature only");
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        assert!(!late.exists(), "the check's child ran on");
    }

    #[test]
    fn globs_match_whole_branch_names() {
        assert!(glob_match("swarm-self", "swarm-self"));
        assert!(!glob_match("swarm-self", "swarm-self2"));
        assert!(!glob_match("swarm-self", "main"));
        assert!(glob_match("swarm/*", "swarm/run/a2"));
        assert!(glob_match("*-self", "swarm-self"));
        assert!(glob_match("release-?", "release-1"));
        assert!(!glob_match("release-?", "release-10"));
        assert!(glob_match("a*b*c", "aXbYbZc"));
        assert!(!glob_match("a*b*c", "aXbYbZ"));
        assert!(glob_match("*", ""));
    }

    #[test]
    fn quoting_survives_single_quotes() {
        assert_eq!(sh_quote("/a b/it's"), r"'/a b/it'\''s'");
    }
}
