//! Per-agent git worktrees.
//!
//! Worktrees live beside the main checkout, at `<repo>.swarm/<run>/<agent>`, so they never show up
//! inside the repository the agents are searching. Each gets the branch `swarm/<run>/<agent>`,
//! cut from the spawner's `HEAD`.

use std::path::{Path, PathBuf};

use strands_agents::sandbox::detach_from_terminal;
use strands_agents::{Error, Result};
use tokio::process::Command;

/// A freshly created worktree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Worktree {
    /// Directory the agent should work in: the worktree root, or the same subdirectory of it the
    /// spawner was in.
    pub cwd: PathBuf,
    /// Branch checked out in the worktree.
    pub branch: String,
}

/// Create a worktree for `agent_id` from the repository containing `base`.
pub async fn create(base: &Path, run_id: &str, agent_id: &str) -> Result<Worktree> {
    let git = |args| async move { git(base, args).await.map_err(Error::other) };
    let toplevel = PathBuf::from(git(&["rev-parse", "--show-toplevel"]).await?);
    let common = PathBuf::from(git(&["rev-parse", "--path-format=absolute", "--git-common-dir"]).await?);
    let main_root = common
        .parent()
        .ok_or_else(|| Error::other("Cannot locate the main checkout of this repository."))?;
    let repo_name = main_root
        .file_name()
        .map_or_else(|| "repo".into(), |n| n.to_string_lossy());
    let dir = main_root
        .with_file_name(format!("{repo_name}.swarm"))
        .join(run_id)
        .join(agent_id);
    let branch = format!("swarm/{run_id}/{agent_id}");

    let dir_arg = dir.to_string_lossy().into_owned();
    git(&["worktree", "add", "-b", &branch, &dir_arg, "HEAD"]).await?;

    Ok(Worktree {
        cwd: same_subdir(&dir, &toplevel, base),
        branch,
    })
}

/// `dir` plus the path of `base` below `toplevel`. Joining an empty subdirectory would append a
/// trailing `/`, so the root case returns `dir` as is.
pub(crate) fn same_subdir(dir: &Path, toplevel: &Path, base: &Path) -> PathBuf {
    match base.strip_prefix(toplevel) {
        Ok(subdir) if !subdir.as_os_str().is_empty() => dir.join(subdir),
        _ => dir.to_path_buf(),
    }
}

/// Variables that point git at a repository. Inherited (from a `git rebase --exec` or a hook,
/// say), they would make `git -C <dir>` act on that repository instead of `<dir>`'s.
pub const GIT_LOCATION_VARS: &[&str] = &[
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
];

/// `git -C <dir>` with no stdin and off the terminal (a credential prompt fails instead of
/// drawing over the TUI and hanging), blind to [`GIT_LOCATION_VARS`].
pub(crate) fn git_in(dir: &Path) -> Command {
    let mut command = Command::new("git");
    detach_from_terminal(&mut command)
        .arg("-C")
        .arg(dir)
        .stdin(std::process::Stdio::null());
    for var in GIT_LOCATION_VARS {
        command.env_remove(var);
    }
    command
}

/// `git <args>` in `cwd`: its trimmed stdout, or an error carrying its stderr.
pub(crate) async fn git(cwd: &Path, args: &[&str]) -> std::result::Result<String, String> {
    let output = git_in(cwd)
        .args(args)
        .output()
        .await
        .map_err(|error| format!("Failed to run git: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git {} failed: {}", args.join(" "), stderr.trim()));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

/// Test fixture: `git <args>` in `dir`, blocking, with a throwaway identity and no signing; its
/// trimmed stdout. Panics when git fails.
#[cfg(test)]
pub(crate) fn test_git(dir: &Path, args: &[&str]) -> String {
    let mut command = std::process::Command::new("git");
    command.arg("-C").arg(dir);
    for var in GIT_LOCATION_VARS {
        command.env_remove(var);
    }
    let ident = [
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "-c",
        "commit.gpgsign=false",
    ];
    let output = command.args(ident).args(args).output().unwrap();
    assert!(output.status.success(), "git {args:?}: {output:?}");
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_ignores_inherited_repository_variables() {
        let command = git_in(Path::new("/repo"));
        let removed: Vec<_> = command
            .as_std()
            .get_envs()
            .filter(|(_, value)| value.is_none())
            .map(|(key, _)| key.to_string_lossy().into_owned())
            .collect();
        assert_eq!(removed.len(), GIT_LOCATION_VARS.len());
        assert!(GIT_LOCATION_VARS
            .iter()
            .all(|var| removed.iter().any(|r| r == var)));
    }

    #[test]
    fn the_cwd_mirrors_the_subdirectory_without_a_trailing_slash() {
        let dir = Path::new("/w/repo.swarm/run/a2");
        let top = Path::new("/w/repo");
        let root = same_subdir(dir, top, top);
        assert_eq!(root, dir);
        assert!(!root.to_string_lossy().ends_with('/'), "{}", root.display());
        assert_eq!(same_subdir(dir, top, Path::new("/w/repo/a/b")), dir.join("a/b"));
        assert_eq!(same_subdir(dir, top, Path::new("/elsewhere")), dir);
    }

    #[tokio::test]
    async fn a_worktree_from_the_repo_root_has_no_trailing_slash() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        test_git(&repo, &["init", "-q"]);
        test_git(&repo, &["commit", "-q", "--allow-empty", "-m", "init"]);
        let worktree = create(&repo, "run", "a2").await.unwrap();
        let cwd = worktree.cwd.to_string_lossy().into_owned();
        assert!(cwd.ends_with("repo.swarm/run/a2"), "{cwd}");
        assert!(worktree.cwd.is_dir(), "{cwd}");
    }
}
