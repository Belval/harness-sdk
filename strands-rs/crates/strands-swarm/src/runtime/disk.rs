//! Disk-space guard: builds fail in confusing ways once a disk is full, and per-agent cargo
//! target dirs grow fast, so the swarm tells its root agents (and the TUI) when space runs low.
//! Free inodes count like free bytes: a filesystem out of inodes fails every file create with
//! `ENOSPC` while terabytes are free, and a cargo target dir is tens of thousands of small files.

use super::*;
use crate::text::count;

/// Sender of the disk guard's warnings.
pub const DISK_SENDER: &str = "disk";

/// In the run dir while the free space is low and the roots have been told
/// ([`Swarm::guard_disk`]); holds the free bytes then.
pub const DISK_WARNED_FILE: &str = "disk-warned";

/// How often [`Swarm::guard_disk`] checks.
pub const DISK_CHECK_EVERY: Duration = Duration::from_secs(60);

/// How often it checks at most while space is low, so held spawns go on soon after it returns.
pub const LOW_DISK_CHECK_EVERY: Duration = Duration::from_secs(5);

/// The most free inodes the default floor asks for (see [`inode_floor`]).
pub const MIN_FREE_INODES: u64 = 1_000_000;

/// What a filesystem has left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Space {
    /// Free bytes for an unprivileged writer.
    pub bytes: u64,
    /// `(free, total)` inodes; `None` where the filesystem has no fixed inode count (btrfs, xfs
    /// with dynamic inodes report none).
    pub inodes: Option<(u64, u64)>,
}

impl From<u64> for Space {
    /// `bytes` free, inodes unknown.
    fn from(bytes: u64) -> Self {
        Space { bytes, inodes: None }
    }
}

/// The free space on the filesystem holding a path; `None` when unknown.
pub type FreeSpace = Arc<dyn Fn(&Path) -> Option<Space> + Send + Sync>;

/// The free-inode floor of a filesystem with `total` inodes: `configured` if set, else 5% of
/// `total` or [`MIN_FREE_INODES`], whichever is smaller. A million inodes is some twenty cargo
/// target dirs (50k files each) of headroom, where the guard can still reclaim before saves
/// fail; 5% keeps a small filesystem from being held on its first build.
pub fn inode_floor(configured: Option<u64>, total: u64) -> u64 {
    configured.unwrap_or_else(|| (total / 20).min(MIN_FREE_INODES))
}

/// How a warning names the target dir of the land jobs' checks.
const LAND_CHECKS: &str = "land checks (any land job's; safe to delete while none runs)";

/// How many of the biggest target dirs a warning lists.
const BIGGEST: usize = 5;

impl Swarm {
    /// Check the free space (`free`) on the filesystems the swarm writes to (the run dir, the
    /// agents' working directories and the swarm's cargo target dirs) now and every `every`
    /// ([`LOW_DISK_CHECK_EVERY`] at most while it is low). Below `min_free` bytes, or below the
    /// free-inode floor ([`SwarmConfig::min_free_inodes`], [`inode_floor`]) of a filesystem that
    /// counts inodes:
    /// - delete the target dirs of [`Self::finished_agents`], again every `every` while low;
    /// - hold agents' worktree spawns and land jobs ([`Swarm::wait_for_host`]) until it recovers;
    /// - tell the live root agents once, with what was deleted and the biggest per-agent target
    ///   dirs; a new warning needs the space to recover first (by a tenth of the limit). A
    ///   persistent run keeps that state on disk ([`DISK_WARNED_FILE`]), so a restart does not
    ///   warn again.
    ///
    /// Once recovered, it recreates the run's disk reserve ([`SwarmConfig::disk_reserve`]) if a
    /// full disk spent it. Runs until the swarm shuts down.
    pub async fn guard_disk(self, min_free: u64, every: Duration, free: FreeSpace) {
        let marker = self
            .inner
            .config
            .state_dir
            .as_ref()
            .map(|dir| dir.join(DISK_WARNED_FILE));
        let mut warned = marker.as_ref().is_some_and(|marker| marker.exists());
        let inodes = self.inner.config.min_free_inodes;
        // The last pass over finished agents' target dirs, while space is low.
        let mut reclaimed_at: Option<tokio::time::Instant> = None;
        loop {
            self.prune_cancelled_sessions_announced().await;
            let paths = self.disk_paths();
            let readings = {
                let free = free.clone();
                tokio::task::spawn_blocking(move || read_space(&paths, &*free))
                    .await
                    .unwrap_or_default()
            };
            *self.inner.host.disk_free.lock() = least_free(&readings);
            let low = match judge(&readings, min_free, inodes) {
                Verdict::Recovered => {
                    *self.inner.host.disk_low.lock() = None;
                    if std::mem::take(&mut warned) {
                        if let Some(marker) = &marker {
                            let _ = std::fs::remove_file(marker);
                        }
                    }
                    self.keep_reserve().await;
                    false
                }
                Verdict::Low(low) => {
                    *self.inner.host.disk_low.lock() = Some(low.hold());
                    let due = reclaimed_at.is_none_or(|at| at.elapsed() >= every);
                    let mut reclaimed = Vec::new();
                    if due {
                        reclaimed_at = Some(tokio::time::Instant::now());
                        reclaimed = self.reclaim_finished_targets().await;
                    }
                    if !warned {
                        warned = true;
                        if let Some(marker) = &marker {
                            let _ = std::fs::write(marker, low.marker());
                        }
                        let targets = self.target_dirs();
                        let by_files = low.inodes();
                        let sizes = tokio::task::spawn_blocking(move || biggest(targets, by_files))
                            .await
                            .unwrap_or_default();
                        let text = disk_warning(&low, &reclaimed, &sizes);
                        self.tell_roots(DISK_SENDER, "disk", text);
                    }
                    true
                }
                // Between the threshold and recovery: as it was.
                Verdict::Between => self.inner.host.disk_low.lock().is_some(),
            };
            if !low {
                reclaimed_at = None;
            }
            // While low, held spawns and land jobs should go on soon after space returns.
            let pause = if low {
                every.min(LOW_DISK_CHECK_EVERY)
            } else {
                every
            };
            tokio::select! {
                () = self.inner.root_cancel.cancelled() => return,
                () = tokio::time::sleep(pause) => {}
            }
        }
    }

    /// Recreate the run's disk reserve ([`crate::reserve`]) if a full disk spent it.
    async fn keep_reserve(&self) {
        let (Some(dir), bytes) = (&self.inner.config.state_dir, self.inner.config.disk_reserve) else {
            return;
        };
        if bytes == 0 || crate::reserve::path(dir).exists() {
            return;
        }
        let dir = dir.clone();
        // Allocating can mean writing it all, where the filesystem cannot preallocate.
        match tokio::task::spawn_blocking(move || crate::reserve::ensure(&dir, bytes)).await {
            Ok(Ok(true)) => tracing::info!(bytes, "disk reserve created"),
            Ok(Ok(false)) | Err(_) => {}
            Ok(Err(error)) => tracing::warn!(%error, "could not create the disk reserve"),
        }
    }

    /// Why a worktree spawn or a land job waits now: the disk guard found space low.
    pub(super) fn disk_low(&self) -> Option<String> {
        self.inner.host.disk_low.lock().clone()
    }

    /// Live worktree agents that look finished: idle, without a running background job, with a
    /// clean worktree and nothing on their branches that the base lacks (`branches`, from
    /// [`Self::branch_states`]), so cancelling one loses only its conversation. Each comes as
    /// `(id, label, its private target dir if that exists)`.
    pub async fn finished_agents(
        &self,
        branches: &std::collections::HashMap<String, crate::status::BranchState>,
        jobs: &[Job],
    ) -> Vec<(String, String, Option<PathBuf>)> {
        let mut finished = Vec::new();
        for node in self.nodes() {
            let done = branches
                .get(&node.id)
                .is_some_and(|state| state.integrated() && state.others.is_empty());
            let busy = jobs
                .iter()
                .any(|job| job.meta.owner == node.id && job.state == jobs::JobState::Running);
            if node.branch.is_none() || !done || busy || !self.is_idle(&node.id) {
                continue;
            }
            let clean = crate::worktree::git(&node.cwd, &["status", "--porcelain"]).await;
            if !matches!(clean.as_deref(), Ok("")) {
                continue;
            }
            let target = gc::private_target_dir(&self.inner.config.run_id, &node.id, &node.env);
            let target = target.filter(|dir| dir.exists());
            finished.push((node.id.clone(), node.label(), target));
        }
        finished
    }

    /// Whether agent `id` is idle (not working, waiting, stopped or cancelled).
    fn is_idle(&self, id: &str) -> bool {
        let nodes = self.inner.agents.lock();
        nodes
            .get(id)
            .is_some_and(|node| matches!(*node.status.lock(), AgentStatus::Idle))
    }

    /// Delete the private target dirs of [`Self::finished_agents`], keeping their worktrees and
    /// conversations: a rebuild is all it costs them. `(label, bytes, inodes freed)` per deleted
    /// dir.
    async fn reclaim_finished_targets(&self) -> Vec<(String, u64, u64)> {
        if !self.inner.config.reclaim_disk {
            return Vec::new();
        }
        let (_, branches) = self.branch_states().await;
        let jobs = self.jobs().await;
        let mut reclaimed = Vec::new();
        for (id, label, target) in self.finished_agents(&branches, &jobs).await {
            // A turn started since would be building into it.
            let Some(target) = target.filter(|_| self.is_idle(&id)) else {
                continue;
            };
            let dir = target.clone();
            let freed = tokio::task::spawn_blocking(move || {
                let (bytes, files) = gc::dir_usage(&dir);
                std::fs::remove_dir_all(&dir).map(|()| (bytes, files))
            })
            .await;
            match freed {
                Ok(Ok((bytes, files))) => {
                    tracing::info!(agent = %id, dir = %target.display(), bytes, files, "deleted a finished agent's target dir");
                    reclaimed.push((label, bytes, files));
                }
                Ok(Err(error)) => {
                    tracing::warn!(agent = %id, dir = %target.display(), %error, "could not delete a finished agent's target dir");
                }
                Err(_) => {}
            }
        }
        reclaimed
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

    /// Every agent's private target dir that exists, with the agent's label and state, and the
    /// land jobs' shared one.
    fn target_dirs(&self) -> Vec<(String, PathBuf)> {
        let run = &self.inner.config.run_id;
        let land = super::landing::land_target_dir(&self.inner.config);
        let land = land.exists().then(|| (LAND_CHECKS.to_owned(), land));
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
            .chain(land)
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

/// Each path (as given) with the space on the filesystem of its nearest existing ancestor.
fn read_space(paths: &[PathBuf], free: &dyn Fn(&Path) -> Option<Space>) -> Vec<(PathBuf, Space)> {
    paths
        .iter()
        .filter_map(|path| {
            let existing = path.ancestors().find(|dir| dir.exists())?;
            Some((path.clone(), free(existing)?))
        })
        .collect()
}

/// The least free bytes and the least free inodes among `readings`, for `/status`.
fn least_free(readings: &[(PathBuf, Space)]) -> Option<(u64, Option<u64>)> {
    let bytes = readings.iter().map(|(_, space)| space.bytes).min()?;
    let inodes = readings
        .iter()
        .filter_map(|(_, space)| space.inodes)
        .map(|(free, _)| free)
        .min();
    Some((bytes, inodes))
}

/// What is short.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Low {
    /// Free bytes on the filesystem of a path, and the floor.
    Bytes(PathBuf, u64, u64),
    /// Free inodes there, and the floor.
    Inodes(PathBuf, u64, u64),
}

/// Where the readings stand against the limits.
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    /// Some filesystem is below a limit (bytes before inodes, the lowest first).
    Low(Low),
    /// Every one is a tenth above its limit.
    Recovered,
    /// Neither: whatever it was stays.
    Between,
}

/// Judge `readings` against `min_free` bytes and the inode floor ([`inode_floor`] of
/// `min_inodes`; `Some(0)` turns the inode check off).
fn judge(readings: &[(PathBuf, Space)], min_free: u64, min_inodes: Option<u64>) -> Verdict {
    let floor = |total| {
        if min_inodes == Some(0) {
            0
        } else {
            inode_floor(min_inodes, total)
        }
    };
    let bytes = readings
        .iter()
        .filter(|(_, space)| space.bytes < min_free)
        .min_by_key(|(_, space)| space.bytes);
    if let Some((path, space)) = bytes {
        return Verdict::Low(Low::Bytes(path.clone(), space.bytes, min_free));
    }
    let inodes = readings
        .iter()
        .filter_map(|(path, space)| {
            let (free, total) = space.inodes?;
            (free < floor(total)).then(|| (path, free, floor(total)))
        })
        .min_by_key(|(_, free, _)| *free);
    if let Some((path, free, floor)) = inodes {
        return Verdict::Low(Low::Inodes(path.clone(), free, floor));
    }
    let recovered = readings.iter().all(|(_, space)| {
        let enough = |free: u64, min: u64| free >= min.saturating_add(min / 10);
        enough(space.bytes, min_free)
            && space
                .inodes
                .is_none_or(|(free, total)| enough(free, floor(total)))
    });
    if recovered {
        Verdict::Recovered
    } else {
        Verdict::Between
    }
}

impl Low {
    fn path(&self) -> &Path {
        match self {
            Low::Bytes(path, ..) | Low::Inodes(path, ..) => path,
        }
    }

    fn inodes(&self) -> bool {
        matches!(self, Low::Inodes(..))
    }

    /// `disk space low: 3.0 GB free` / `disk inodes low: 40k free`.
    fn what(&self) -> String {
        match self {
            Low::Bytes(_, free, _) => format!("disk space low: {} free", crate::gc::human(*free)),
            Low::Inodes(_, free, _) => format!("disk inodes low: {} free", count(*free)),
        }
    }

    /// `20 GB` / `1M`.
    fn floor(&self) -> String {
        match self {
            Low::Bytes(_, _, min) => crate::gc::human(*min),
            Low::Inodes(_, _, min) => count(*min),
        }
    }

    /// What the warned marker holds.
    fn marker(&self) -> String {
        match self {
            Low::Bytes(_, free, _) => format!("{free}\n"),
            Low::Inodes(_, free, _) => format!("{free} inodes\n"),
        }
    }

    /// Why a worktree spawn or land job waits while space is low.
    fn hold(&self) -> String {
        format!(
            "{} on the filesystem of {} (below {}; cancel finished worktree agents to free some)",
            self.what(),
            self.path().display(),
            self.floor()
        )
    }
}

/// `dirs` with their sizes (bytes, entries), biggest first (by entries when `by_files`), at most
/// [`BIGGEST`].
fn biggest(dirs: Vec<(String, PathBuf)>, by_files: bool) -> Vec<(String, PathBuf, u64, u64)> {
    let mut sized: Vec<_> = dirs
        .into_iter()
        .map(|(label, dir)| {
            let (bytes, files) = crate::gc::dir_usage(&dir);
            (label, dir, bytes, files)
        })
        .collect();
    sized.sort_by_key(|(_, _, bytes, files)| std::cmp::Reverse(if by_files { *files } else { *bytes }));
    sized.truncate(BIGGEST);
    sized
}

/// The warning's text; its first line is the TUI's footer notice. `reclaimed`: the finished
/// agents' target dirs just deleted.
fn disk_warning(
    low: &Low,
    reclaimed: &[(String, u64, u64)],
    sizes: &[(String, PathBuf, u64, u64)],
) -> String {
    use crate::gc::human;
    let size = |bytes: u64, files: u64| {
        if low.inodes() {
            format!("{} files", count(files))
        } else {
            human(bytes)
        }
    };
    let mut text = format!(
        "{} on the filesystem of {} (warning below {})\n",
        low.what(),
        low.path().display(),
        low.floor()
    );
    if !reclaimed.is_empty() {
        let each: Vec<String> = reclaimed
            .iter()
            .map(|(label, bytes, files)| format!("{} {label}", size(*bytes, *files)))
            .collect();
        text.push_str(&format!(
            "Deleted the target dirs of idle agents with nothing left to merge (their next build \
             rebuilds it): {}\n",
            each.join(", ")
        ));
    }
    if !sizes.is_empty() {
        text.push_str("Biggest swarm-owned cargo target dirs:\n");
        for (label, dir, bytes, files) in sizes {
            text.push_str(&format!(
                "  {} {label}: {}\n",
                size(*bytes, *files),
                dir.display()
            ));
        }
    }
    text.push_str(if low.inodes() {
        "Every file create fails once the filesystem is out of inodes, even with space free (saves \
         and logs too), and the disk reserve cannot help. Free inodes: cancel worktree agents you \
         no longer need (cancelling deletes their target dir), or `cargo clean` a target dir you own."
    } else {
        "Builds fail once the disk is full. Free space: cancel worktree agents you no longer \
         need (cancelling deletes their target dir), or `cargo clean` a target dir you own."
    });
    text
}

/// Free bytes and inodes on the filesystem holding `path` for an unprivileged writer (`statvfs`'s
/// available blocks and file slots, as `df` shows them).
#[allow(clippy::useless_conversion)] // Not `u64` on every platform.
pub fn free_space(path: &Path) -> Option<Space> {
    let stats = nix::sys::statvfs::statvfs(path).ok()?;
    let bytes = u64::from(stats.blocks_available()).checked_mul(u64::from(stats.fragment_size()))?;
    let total = u64::from(stats.files());
    let inodes = (total > 0).then(|| (u64::from(stats.files_available()), total));
    Some(Space { bytes, inodes })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(path: &str, bytes: u64, inodes: Option<(u64, u64)>) -> (PathBuf, Space) {
        (PathBuf::from(path), Space { bytes, inodes })
    }

    #[test]
    fn statvfs_reports_free_space_and_inodes_for_an_existing_dir() {
        let dir = tempfile::tempdir().unwrap();
        let space = free_space(dir.path()).unwrap();
        assert!(space.bytes > 0);
        if let Some((free, total)) = space.inodes {
            assert!(free <= total);
        }
    }

    #[test]
    fn the_inode_floor_is_five_percent_of_the_inodes_at_most_a_million() {
        assert_eq!(inode_floor(None, 100_000_000), 1_000_000);
        assert_eq!(inode_floor(None, 4_000_000), 200_000);
        assert_eq!(inode_floor(Some(7), 4_000_000), 7);
    }

    #[test]
    fn the_least_free_filesystem_wins_and_missing_paths_use_their_parent() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("a"), dir.path().join("b/not/yet"));
        std::fs::create_dir(&a).unwrap();
        let free = |path: &Path| {
            Some(Space {
                bytes: if path.ends_with("a") { 5 } else { 9 },
                inodes: Some((if path.ends_with("a") { 90 } else { 40 }, 1000)),
            })
        };
        let readings = read_space(&[a, b], &free);
        assert_eq!(readings.len(), 2);
        assert_eq!(least_free(&readings), Some((5, Some(40))));
        assert_eq!(least_free(&[]), None);
    }

    #[test]
    fn inodes_are_judged_like_bytes_with_the_lowest_filesystem_named() {
        let big = Some((50_000_000, 100_000_000));
        let plenty = 100 << 30;
        let ok = [reading("/a", plenty, big), reading("/b", plenty, None)];
        assert_eq!(judge(&ok, 20 << 30, None), Verdict::Recovered);

        // 40k left of 100M: below the 1M cap.
        let low = [
            reading("/a", plenty, big),
            reading("/b", plenty, Some((40_000, 100_000_000))),
        ];
        assert_eq!(
            judge(&low, 20 << 30, None),
            Verdict::Low(Low::Inodes("/b".into(), 40_000, 1_000_000))
        );
        // Bytes win over inodes.
        let both = [reading("/c", 1 << 30, Some((10, 100)))];
        assert!(matches!(
            judge(&both, 20 << 30, None),
            Verdict::Low(Low::Bytes(..))
        ));
        // The floor scales down: 5% of 10k inodes is 500.
        let small = [reading("/s", plenty, Some((520, 10_000)))];
        assert_eq!(judge(&small, 20 << 30, None), Verdict::Between);
        let small = [reading("/s", plenty, Some((400, 10_000)))];
        assert!(matches!(
            judge(&small, 20 << 30, None),
            Verdict::Low(Low::Inodes(..))
        ));
        // An explicit floor, and off.
        assert!(matches!(judge(&ok, 20 << 30, Some(60_000_000)), Verdict::Low(_)));
        assert_eq!(judge(&low, 20 << 30, Some(0)), Verdict::Recovered);
    }

    #[test]
    fn the_hold_and_the_warning_name_inodes() {
        let low = Low::Inodes("/w".into(), 40_000, 1_000_000);
        assert_eq!(
            low.hold(),
            "disk inodes low: 40k free on the filesystem of /w (below 1M; cancel finished \
             worktree agents to free some)"
        );
        let reclaimed = [("done (a4)".to_owned(), 12 << 30, 300_000)];
        let text = disk_warning(&low, &reclaimed, &[("x (a2)".into(), "/t".into(), 5, 70_000)]);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[0],
            "disk inodes low: 40k free on the filesystem of /w (warning below 1M)"
        );
        assert!(lines[1].ends_with("300k files done (a4)"), "{text}");
        assert_eq!(lines[3], "  70k files x (a2): /t");
        assert!(text.contains("reserve cannot help"), "{text}");
    }

    #[test]
    fn a_warning_lists_the_biggest_dirs_first() {
        let dir = tempfile::tempdir().unwrap();
        let (small, big) = (dir.path().join("a2"), dir.path().join("a3"));
        for (path, len) in [(&small, 10), (&big, 3000)] {
            std::fs::create_dir(path).unwrap();
            std::fs::write(path.join("f"), vec![0; len]).unwrap();
        }
        let dirs = vec![("x (a2)".into(), small), ("y (a3) (cancelled)".into(), big)];
        let sizes = biggest(dirs, false);
        let low = Low::Bytes("/w".into(), 3 << 30, 20 << 30);
        let text = disk_warning(&low, &[], &sizes);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[0],
            "disk space low: 3.0 GB free on the filesystem of /w (warning below 20 GB)"
        );
        assert!(lines[2].starts_with("  2.9 KB y (a3) (cancelled): /"), "{text}");
        assert!(lines[3].starts_with("  10 B x (a2): /"), "{text}");

        let reclaimed = [("done (a4)".to_owned(), 12 << 30, 9)];
        let text = disk_warning(&low, &reclaimed, &sizes);
        assert_eq!(
            text.lines().nth(1),
            Some(
                "Deleted the target dirs of idle agents with nothing left to merge (their next \
                 build rebuilds it): 12 GB done (a4)"
            )
        );
    }
}
