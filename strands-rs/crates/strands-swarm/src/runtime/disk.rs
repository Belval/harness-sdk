//! Disk-space guard: builds fail in confusing ways once a disk is full, and per-agent cargo
//! target dirs grow fast, so the swarm tells its root agents (and the TUI) when space runs low.

use super::*;

/// Sender of the disk guard's warnings.
pub const DISK_SENDER: &str = "disk";

/// In the run dir while the free space is low and the roots have been told
/// ([`Swarm::guard_disk`]); holds the free bytes then.
pub const DISK_WARNED_FILE: &str = "disk-warned";

/// How often [`Swarm::guard_disk`] checks.
pub const DISK_CHECK_EVERY: Duration = Duration::from_secs(60);

/// Free bytes on the filesystem holding a path; `None` when unknown.
pub type FreeSpace = Arc<dyn Fn(&Path) -> Option<u64> + Send + Sync>;

/// How many of the biggest target dirs a warning lists.
const BIGGEST: usize = 5;

impl Swarm {
    /// Every `every`, check the free space (`free`) on the filesystems the swarm writes to: the
    /// run dir, the agents' working directories and the swarm's cargo target dirs. Once it drops
    /// below `min_free` bytes, tell the live root agents once, with the biggest per-agent target
    /// dirs; a new warning needs the space to recover first (by a tenth of `min_free`). A
    /// persistent run keeps that state on disk ([`DISK_WARNED_FILE`]), so a restart does not
    /// warn again. Runs until the swarm shuts down.
    pub async fn guard_disk(self, min_free: u64, every: Duration, free: FreeSpace) {
        let marker = self
            .inner
            .config
            .state_dir
            .as_ref()
            .map(|dir| dir.join(DISK_WARNED_FILE));
        let mut warned = marker.as_ref().is_some_and(|marker| marker.exists());
        loop {
            tokio::select! {
                () = self.inner.root_cancel.cancelled() => return,
                () = tokio::time::sleep(every) => {}
            }
            let paths = self.disk_paths();
            let lowest = {
                let free = free.clone();
                tokio::task::spawn_blocking(move || lowest_free(&paths, &*free))
                    .await
                    .ok()
                    .flatten()
            };
            let Some((path, bytes)) = lowest else { continue };
            if bytes >= min_free.saturating_add(min_free / 10) {
                if std::mem::take(&mut warned) {
                    if let Some(marker) = &marker {
                        let _ = std::fs::remove_file(marker);
                    }
                }
            } else if bytes < min_free && !warned {
                warned = true;
                if let Some(marker) = &marker {
                    let _ = std::fs::write(marker, format!("{bytes}\n"));
                }
                let targets = self.target_dirs();
                let sizes = tokio::task::spawn_blocking(move || biggest(targets))
                    .await
                    .unwrap_or_default();
                self.tell_roots(DISK_SENDER, "disk", disk_warning(&path, bytes, min_free, &sizes));
            }
        }
    }

    /// Where the swarm writes: run dir, agents' working directories, shared and private target
    /// dirs.
    fn disk_paths(&self) -> Vec<PathBuf> {
        let config = &self.inner.config;
        let mut paths: Vec<PathBuf> = config.state_dir.iter().cloned().collect();
        paths.push(config.cwd.clone());
        paths.extend(config.cargo_target_dir.clone());
        paths.extend(self.private_targets_root());
        paths.extend(self.nodes().iter().map(|node| node.cwd.clone()));
        paths.extend(self.target_dirs().into_iter().map(|(_, dir)| dir));
        paths.sort();
        paths.dedup();
        paths
    }

    /// Every agent's private target dir that exists, with the agent's label and state.
    fn target_dirs(&self) -> Vec<(String, PathBuf)> {
        let run = &self.inner.config.run_id;
        self.nodes()
            .iter()
            .filter_map(|node| {
                let dir = crate::gc::private_target_dir(run, &node.id, &node.env)?;
                let status = match &*node.status.lock() {
                    AgentStatus::Cancelled => " (cancelled)",
                    _ => "",
                };
                dir.exists().then(|| (format!("{}{status}", node.label()), dir))
            })
            .collect()
    }

    /// Mail every live root agent, or the user when there is none.
    pub(super) fn tell_roots(&self, from: &str, kind: &'static str, text: String) {
        let roots: Vec<String> = self
            .inner
            .agents
            .lock()
            .values()
            .filter(|node| node.parent == USER && node.status.lock().is_live())
            .map(|node| node.id.clone())
            .collect();
        if roots.is_empty() {
            let _ = self.post(from, USER, kind, text);
            return;
        }
        for root in roots {
            let _ = self.post(from, &root, kind, text.clone());
        }
    }
}

/// The path (its nearest existing ancestor) with the least free space, and that space.
fn lowest_free(paths: &[PathBuf], free: &dyn Fn(&Path) -> Option<u64>) -> Option<(PathBuf, u64)> {
    paths
        .iter()
        .filter_map(|path| {
            let existing = path.ancestors().find(|dir| dir.exists())?;
            Some((path.clone(), free(existing)?))
        })
        .min_by_key(|(_, bytes)| *bytes)
}

/// `dirs` with their sizes, biggest first, at most [`BIGGEST`].
fn biggest(dirs: Vec<(String, PathBuf)>) -> Vec<(String, PathBuf, u64)> {
    let mut sized: Vec<_> = dirs
        .into_iter()
        .map(|(label, dir)| {
            let bytes = crate::gc::dir_size(&dir);
            (label, dir, bytes)
        })
        .collect();
    sized.sort_by_key(|(_, _, bytes)| std::cmp::Reverse(*bytes));
    sized.truncate(BIGGEST);
    sized
}

/// The warning's text; its first line is the TUI's footer notice.
fn disk_warning(path: &Path, free: u64, min_free: u64, sizes: &[(String, PathBuf, u64)]) -> String {
    use crate::gc::human;
    let mut text = format!(
        "disk space low: {} free on the filesystem of {} (warning below {})\n",
        human(free),
        path.display(),
        human(min_free)
    );
    if !sizes.is_empty() {
        text.push_str("Biggest swarm-owned cargo target dirs:\n");
        for (label, dir, bytes) in sizes {
            text.push_str(&format!("  {} {label}: {}\n", human(*bytes), dir.display()));
        }
    }
    text.push_str(
        "Builds fail once the disk is full. Free space: cancel worktree agents you no longer \
         need (cancelling deletes their target dir), or `cargo clean` a target dir you own.",
    );
    text
}

/// Free bytes on the filesystem holding `path`, from `df -Pk` (POSIX output, 1024-byte blocks).
pub fn free_space(path: &Path) -> Option<u64> {
    let output = std::process::Command::new("df")
        .arg("-Pk")
        .arg(path)
        .output()
        .ok()?;
    let stdout = String::from_utf8(output.stdout).ok()?;
    let available = stdout.lines().nth(1)?.split_whitespace().nth(3)?;
    available.parse::<u64>().ok()?.checked_mul(1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn df_reports_free_space_for_an_existing_dir() {
        let dir = tempfile::tempdir().unwrap();
        assert!(free_space(dir.path()).is_some_and(|bytes| bytes > 0));
    }

    #[test]
    fn the_lowest_filesystem_wins_and_missing_paths_use_their_parent() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("a"), dir.path().join("b/not/yet"));
        std::fs::create_dir(&a).unwrap();
        let free = |path: &Path| Some(if path.ends_with("a") { 5 } else { 9 });
        assert_eq!(lowest_free(std::slice::from_ref(&b), &free), Some((b.clone(), 9)));
        assert_eq!(lowest_free(&[a.clone(), b], &free), Some((a, 5)));
    }

    #[test]
    fn a_warning_lists_the_biggest_dirs_first() {
        let dir = tempfile::tempdir().unwrap();
        let (small, big) = (dir.path().join("a2"), dir.path().join("a3"));
        for (path, len) in [(&small, 10), (&big, 3000)] {
            std::fs::create_dir(path).unwrap();
            std::fs::write(path.join("f"), vec![0; len]).unwrap();
        }
        let sizes = biggest(vec![("x (a2)".into(), small), ("y (a3) (cancelled)".into(), big)]);
        let text = disk_warning(Path::new("/w"), 3 << 30, 20 << 30, &sizes);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[0],
            "disk space low: 3.0 GB free on the filesystem of /w (warning below 20 GB)"
        );
        assert!(lines[2].starts_with("  2.9 KB y (a3) (cancelled): /"), "{text}");
        assert!(lines[3].starts_with("  10 B x (a2): /"), "{text}");
    }
}
