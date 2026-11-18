//! The commit this binary was built from, and what changed between two builds.

use std::path::Path;

/// The git commit this binary was built from (`build.rs`); `None` when built outside git.
pub const COMMIT: Option<&str> = option_env!("STRANDS_SWARM_COMMIT");

/// The checkout this binary was built from, where [`changes`] looks commits up.
pub const SOURCE: &str = env!("CARGO_MANIFEST_DIR");

/// Most commits [`changes`] lists.
pub const MAX_CHANGES: usize = 30;

/// Why this binary is stale, if it is: it embeds `commit` but the checkout at `source` it was
/// built from is at another one. Cargo reuses a build script's output while the files it watched
/// look unchanged, and a target dir shared by worktrees shares that verdict (the package hashes
/// the same in each), so a build can embed an older commit than the checkout's HEAD. `None` when
/// the binary is current, has no commit, or `source` is gone or no git checkout.
pub async fn stale(commit: Option<&str>, source: &Path) -> Option<String> {
    let commit = commit?;
    let head = crate::worktree::git(source, &["rev-parse", "HEAD"]).await.ok()?;
    (head != commit).then(|| {
        format!(
            "this binary embeds commit {commit} but its checkout {} is at {head}: the build script \
             did not rerun (cargo reused its output). Rebuild with \
             `STRANDS_SWARM_BUILD_HEAD=$(git rev-parse HEAD) cargo build --release -p strands-swarm`, \
             which always reruns it when the head moved, then retry.",
            source.display()
        )
    })
}

/// `<short hash> <subject>` of the commits in `to` but not in `from`, newest first, at most
/// [`MAX_CHANGES`]; empty when `repo` is gone or does not know either commit.
pub async fn changes(repo: &Path, from: &str, to: &str) -> Vec<String> {
    let range = format!("{from}..{to}");
    let max = format!("--max-count={MAX_CHANGES}");
    let log = ["log", "--format=%h %s", &max, &range, "--"];
    let output = crate::worktree::git(repo, &log).await.unwrap_or_default();
    output.lines().map(str::to_owned).collect()
}

/// One line for the TUI: `reloaded 1a2b3c4 → 5d6e7f8: 2 commits: feat: … · fix: …`.
pub fn notice(from: &str, to: &str, changes: &[String]) -> String {
    let short = |commit: &str| commit.chars().take(7).collect::<String>();
    let head = format!("reloaded {} → {}", short(from), short(to));
    if changes.is_empty() {
        return head;
    }
    let subjects: Vec<&str> = changes
        .iter()
        .map(|line| line.split_once(' ').map_or(line.as_str(), |(_, subject)| subject))
        .collect();
    let count = match changes.len() {
        1 => "1 commit".to_owned(),
        n if n >= MAX_CHANGES => format!("{n}+ commits"),
        n => format!("{n} commits"),
    };
    format!("{head}: {count}: {}", subjects.join(" · "))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::worktree::test_git as git;

    #[tokio::test]
    async fn changes_lists_the_commits_between_two_builds_newest_first() {
        let dir = tempfile::tempdir().unwrap();
        git(dir.path(), &["init", "-q"]);
        let commit = |subject: &str| {
            git(dir.path(), &["commit", "-q", "--allow-empty", "-m", subject]);
            git(dir.path(), &["rev-parse", "HEAD"])
        };
        let old = commit("feat: base");
        commit("feat: one");
        let new = commit("fix: two");

        let changes = changes(dir.path(), &old, &new).await;
        let subjects: Vec<_> = changes.iter().map(|l| l.split_once(' ').unwrap().1).collect();
        assert_eq!(subjects, ["fix: two", "feat: one"]);
        assert_eq!(
            notice(&old, &new, &changes),
            format!(
                "reloaded {} → {}: 2 commits: fix: two · feat: one",
                &old[..7],
                &new[..7]
            )
        );
        assert!(super::changes(dir.path(), "nope", &new).await.is_empty());
        assert!(super::changes(&dir.path().join("gone"), &old, &new)
            .await
            .is_empty());
        assert_eq!(
            notice("abcdef0123", "0123456789", &[]),
            "reloaded abcdef0 → 0123456"
        );
    }

    #[tokio::test]
    async fn a_binary_older_than_its_checkout_is_stale() {
        let dir = tempfile::tempdir().unwrap();
        git(dir.path(), &["init", "-q"]);
        git(dir.path(), &["commit", "-q", "--allow-empty", "-m", "one"]);
        let old = git(dir.path(), &["rev-parse", "HEAD"]);
        assert_eq!(stale(Some(&old), dir.path()).await, None);
        git(dir.path(), &["commit", "-q", "--allow-empty", "-m", "two"]);
        let new = git(dir.path(), &["rev-parse", "HEAD"]);

        let message = stale(Some(&old), dir.path()).await.unwrap();
        assert!(message.contains(&old) && message.contains(&new), "{message}");
        assert_eq!(stale(Some(&new), dir.path()).await, None);
        // Nothing to compare: built outside git, or the checkout is gone or not a repository.
        assert_eq!(stale(None, dir.path()).await, None);
        assert_eq!(stale(Some(&old), &dir.path().join("gone")).await, None);
        let plain = tempfile::tempdir_in("/tmp").unwrap();
        assert_eq!(stale(Some(&old), plain.path()).await, None);
    }

    #[test]
    fn the_build_embeds_its_commit() {
        let head = std::process::Command::new("git")
            .args(["-C", SOURCE, "rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|output| output.status.success());
        let head = head.map(|output| String::from_utf8(output.stdout).unwrap().trim().to_owned());
        assert_eq!(COMMIT.map(str::to_owned), head);
    }
}
