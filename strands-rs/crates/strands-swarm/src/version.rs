//! The commit this binary was built from, and what changed between two builds.

use std::path::Path;

/// The git commit this binary was built from (`build.rs`); `None` when built outside git.
pub const COMMIT: Option<&str> = option_env!("STRANDS_SWARM_COMMIT");

/// The checkout this binary was built from, where [`changes`] looks commits up.
pub const SOURCE: &str = env!("CARGO_MANIFEST_DIR");

/// Most commits [`changes`] lists.
pub const MAX_CHANGES: usize = 30;

/// `<short hash> <subject>` of the commits in `to` but not in `from`, newest first, at most
/// [`MAX_CHANGES`]; empty when `repo` is gone or does not know either commit.
pub async fn changes(repo: &Path, from: &str, to: &str) -> Vec<String> {
    let range = format!("{from}..{to}");
    let max = format!("--max-count={MAX_CHANGES}");
    let output = tokio::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["log", "--format=%h %s", &max, &range, "--"])
        .stdin(std::process::Stdio::null())
        .output()
        .await;
    match output {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
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

    fn git(repo: &Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(repo)
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    }

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
