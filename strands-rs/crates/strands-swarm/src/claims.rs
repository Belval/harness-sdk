//! Advisory path claims: which files each agent is changing, so agents working in one repo (other
//! worktrees of it, or one checkout) learn of each other before they conflict. A claim is derived,
//! never declared: a file the agent changed through `edit`/`write` in its current turn, or, when
//! it has its checkout to itself, any file uncommitted there. Nothing blocks; the swarm only says
//! who else is on a file (the note on `edit`/`write` results, `list_agents`).

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// How long a checkout's `git status` is reused.
pub const TTL: Duration = Duration::from_secs(5);

/// Shared files named per other agent in `list_agents`.
const SHOWN: usize = 4;

/// A file of a repo: the repo's common git dir, so worktrees of one repo agree, and the path
/// relative to the checkout's top.
pub type Key = (PathBuf, String);

/// A checkout: its top directory and the git dir it shares with the repo's other worktrees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Repo {
    /// Top directory.
    pub top: PathBuf,
    /// The common git dir.
    pub common: PathBuf,
}

/// The uncommitted files of a checkout, as of when its status was taken.
#[derive(Debug, Clone)]
pub struct Dirty {
    /// When the status was started (an edit after it may be missing from it).
    pub at: Instant,
    /// Repo-relative paths.
    pub paths: BTreeSet<String>,
}

/// What one agent changed in its current turn through `edit`/`write`.
#[derive(Debug, Default)]
pub struct Mine {
    /// The files, with when and in which checkout (by top).
    pub edited: HashMap<Key, (Instant, PathBuf)>,
    /// Notes given this turn: (file, the other agent).
    pub noted: BTreeSet<(Key, String)>,
}

/// Every agent's [`Mine`], and each checkout's last status.
#[derive(Debug, Default)]
pub struct Claims {
    /// By agent id.
    pub agents: HashMap<String, Mine>,
    /// By checkout top.
    pub status: HashMap<PathBuf, Dirty>,
}

impl Claims {
    /// Keep `dirty` for the checkout at `top`, and drop every edit made there that is older than
    /// it and not uncommitted: committed, landed, reverted or ignored.
    pub fn settle(&mut self, top: PathBuf, dirty: Dirty) {
        for mine in self.agents.values_mut() {
            mine.edited.retain(|(_, path), (at, from)| {
                *from != top || dirty.at <= *at || dirty.paths.contains(path)
            });
        }
        self.status.insert(top, dirty);
    }
}

/// The checkout containing `dir`, found without running git.
pub fn repo_at(dir: &Path) -> Option<Repo> {
    let dir = dir.canonicalize().ok()?;
    let top = dir.ancestors().find(|d| d.join(".git").exists())?.to_path_buf();
    let dot = top.join(".git");
    if dot.is_dir() {
        return Some(Repo { top, common: dot });
    }
    // A linked worktree: `.git` is a file naming its private git dir, which names the common one.
    let text = std::fs::read_to_string(&dot).ok()?;
    let private = top.join(text.trim().strip_prefix("gitdir:")?.trim());
    let common = match std::fs::read_to_string(private.join("commondir")) {
        Ok(rel) => private.join(rel.trim()),
        Err(_) => private,
    };
    Some(Repo {
        top,
        common: common.canonicalize().ok()?,
    })
}

/// The claim key of `path` (which must lie in an existing directory of a checkout), and that
/// checkout's top.
pub fn key_of(path: &Path) -> Option<(Key, PathBuf)> {
    let dir = path.parent()?.canonicalize().ok()?;
    let repo = repo_at(&dir)?;
    let rel = dir
        .join(path.file_name()?)
        .strip_prefix(&repo.top)
        .ok()?
        .to_str()?
        .to_owned();
    Some(((repo.common, rel), repo.top))
}

/// The uncommitted files of `repo`, untracked ones included; `None` when git cannot tell.
pub async fn dirty(repo: &Repo) -> Option<Dirty> {
    let at = Instant::now();
    let args = ["status", "--porcelain", "-z", "--no-renames", "-uall"];
    let output = crate::worktree::git_in(&repo.top)
        .args(args)
        .output()
        .await
        .ok()?;
    output.status.success().then(|| Dirty {
        at,
        paths: parse_porcelain(&String::from_utf8_lossy(&output.stdout)),
    })
}

/// `XY path` entries separated by NUL.
fn parse_porcelain(text: &str) -> BTreeSet<String> {
    let entries = text.split('\0').filter_map(|entry| entry.get(3..));
    entries.filter(|p| !p.is_empty()).map(str::to_owned).collect()
}

/// `shares a.rs, b.rs with b (a12)` for each other agent (label, files) having files in common
/// with `mine`.
pub fn shared_with(mine: &BTreeSet<Key>, others: &[(String, BTreeSet<Key>)]) -> Vec<String> {
    let mut notes = Vec::new();
    for (label, theirs) in others {
        let files: Vec<&str> = mine.intersection(theirs).map(|(_, path)| path.as_str()).collect();
        if files.is_empty() {
            continue;
        }
        let more = match files.len().saturating_sub(SHOWN) {
            0 => String::new(),
            n => format!(", +{n} more"),
        };
        let shown = files[..files.len().min(SHOWN)].join(", ");
        notes.push(format!("shares {shown}{more} with {label}"));
    }
    notes
}

/// `editing: 3 files`.
pub fn editing(files: usize) -> String {
    format!("editing: {files} file{}", if files == 1 { "" } else { "s" })
}

/// The note appended to an `edit`/`write` result.
pub fn note(path: &str, label: &str) -> String {
    format!("note: {path} is also being changed by {label}; coordinate or message them.")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(path: &str) -> Key {
        (PathBuf::from("/r/.git"), path.to_owned())
    }

    #[test]
    fn porcelain_entries_lose_their_status_columns() {
        let text = " M src/a.rs\0?? new dir/b.rs\0A  c.rs\0";
        let paths: Vec<_> = parse_porcelain(text).into_iter().collect();
        assert_eq!(paths, ["c.rs", "new dir/b.rs", "src/a.rs"]);
    }

    #[test]
    fn shared_files_are_named_per_agent_and_capped() {
        let mine: BTreeSet<Key> = ["a", "b", "c", "d", "e", "f"].map(key).into();
        let others = vec![
            ("x (a2)".to_owned(), mine.clone()),
            ("y (a3)".to_owned(), BTreeSet::from([key("zzz")])),
            ("z (a4)".to_owned(), BTreeSet::from([key("b")])),
        ];
        assert_eq!(
            shared_with(&mine, &others),
            ["shares a, b, c, d, +2 more with x (a2)", "shares b with z (a4)"]
        );
    }

    #[test]
    fn a_later_status_drops_edits_that_are_no_longer_uncommitted() {
        let before = Instant::now();
        let mut claims = Claims::default();
        let mine = claims.agents.entry("a1".into()).or_default();
        let edit = |at, top: &str| (at, PathBuf::from(top));
        mine.edited.insert(key("kept.rs"), edit(before, "/r"));
        mine.edited.insert(key("landed.rs"), edit(before, "/r"));
        mine.edited.insert(
            (PathBuf::from("/other/.git"), "landed.rs".into()),
            edit(before, "/other"),
        );
        mine.edited
            .insert(key("newer.rs"), edit(before + Duration::from_secs(2), "/r"));
        let dirty = Dirty {
            at: before + Duration::from_secs(1),
            paths: BTreeSet::from(["kept.rs".to_owned()]),
        };
        claims.settle("/r".into(), dirty);
        let mut left: Vec<_> = claims.agents["a1"]
            .edited
            .keys()
            .map(|(c, p)| format!("{}:{p}", c.display()))
            .collect();
        left.sort();
        assert_eq!(
            left,
            ["/other/.git:landed.rs", "/r/.git:kept.rs", "/r/.git:newer.rs"]
        );
    }

    #[test]
    fn a_path_is_keyed_by_the_common_git_dir_of_any_worktree() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("repo");
        let tree = dir.path().join("tree");
        std::fs::create_dir_all(repo.join("src")).unwrap();
        crate::worktree::test_git(&repo, &["init", "-q", "-b", "main"]);
        crate::worktree::test_git(&repo, &["commit", "-q", "--allow-empty", "-m", "init"]);
        crate::worktree::test_git(&repo, &["worktree", "add", "-q", tree.to_str().unwrap()]);
        std::fs::create_dir_all(tree.join("src")).unwrap();
        let (a, b) = (
            key_of(&repo.join("src/a.rs")).unwrap().0,
            key_of(&tree.join("src/a.rs")).unwrap().0,
        );
        assert_eq!(a, b);
        assert_eq!(a.1, "src/a.rs");
        assert_eq!(key_of(&dir.path().join("loose.rs")), None);
    }
}
