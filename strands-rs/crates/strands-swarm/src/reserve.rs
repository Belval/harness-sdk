//! A reserve of disk space in the run dir: deleted by the first save that finds the disk full, so
//! the state and the conversations can still be written, and recreated once space returns
//! ([`crate::Swarm::guard_disk`]).

use std::future::Future;
use std::path::{Path, PathBuf};

/// The reserve file in the run dir.
pub const RESERVE_FILE: &str = "disk-reserve";

/// The reserve's size the CLI keeps.
pub const RESERVE_BYTES: u64 = 1 << 30;

/// The reserve file of run dir `dir`.
pub fn path(dir: &Path) -> PathBuf {
    dir.join(RESERVE_FILE)
}

/// Create the reserve of `bytes` in `dir` unless it is there; whether it was created. The space
/// is allocated, not sparse, so deleting the file frees it.
pub fn ensure(dir: &Path, bytes: u64) -> std::io::Result<bool> {
    let path = path(dir);
    if path.exists() {
        return Ok(false);
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let file = std::fs::File::create(&tmp)?;
    let allocated = allocate(&file, bytes);
    drop(file);
    match allocated.and_then(|()| std::fs::rename(&tmp, &path)) {
        Ok(()) => Ok(true),
        Err(error) => {
            let _ = std::fs::remove_file(&tmp);
            Err(error)
        }
    }
}

#[cfg(target_os = "linux")]
fn allocate(file: &std::fs::File, bytes: u64) -> std::io::Result<()> {
    let len = i64::try_from(bytes).map_err(std::io::Error::other)?;
    nix::fcntl::posix_fallocate(file, 0, len).map_err(std::io::Error::from)
}

#[cfg(not(target_os = "linux"))]
fn allocate(file: &std::fs::File, bytes: u64) -> std::io::Result<()> {
    use std::io::Write;
    let chunk = vec![0u8; 1 << 20];
    let mut file = file;
    let mut left = bytes;
    while left > 0 {
        let n = left.min(chunk.len() as u64) as usize;
        file.write_all(&chunk[..n])?;
        left -= n as u64;
    }
    file.sync_all()
}

/// Delete the reserve of run dir `dir`; whether there was one to free.
pub fn release(dir: &Path) -> bool {
    let path = path(dir);
    let bytes = std::fs::metadata(&path).map_or(0, |meta| meta.len());
    let released = std::fs::remove_file(&path).is_ok();
    if released {
        tracing::warn!(
            bytes,
            path = %path.display(),
            "disk full: released the disk reserve so the run can still be saved"
        );
    }
    released
}

/// Whether `error`, or an error it wraps, is a full disk (`ENOSPC`, or a quota).
pub fn is_disk_full(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut next = Some(error);
    while let Some(error) = next {
        if let Some(io) = error.downcast_ref::<std::io::Error>() {
            if matches!(
                io.kind(),
                std::io::ErrorKind::StorageFull | std::io::ErrorKind::QuotaExceeded
            ) {
                return true;
            }
        }
        next = error.source();
    }
    let text = error.to_string();
    text.contains("No space left on device") || text.contains("os error 28")
}

/// Run `save`; when it fails because the disk is full and run dir `dir` still has its reserve,
/// delete the reserve and run it once more.
pub async fn save_with<T, E, F, Fut>(dir: &Path, mut save: F) -> Result<T, E>
where
    E: std::error::Error + 'static,
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    match save().await {
        Err(error) if is_disk_full(&error) && release(dir) => save().await,
        outcome => outcome,
    }
}

/// [`SnapshotStorage`] whose saves fall back on the run's reserve ([`save_with`]).
///
/// [`SnapshotStorage`]: strands_agents::session::SnapshotStorage
pub struct ReserveStorage<S> {
    inner: S,
    run_dir: PathBuf,
}

impl<S> ReserveStorage<S> {
    /// `inner`, saving into the run dir `run_dir` (where the reserve is).
    pub fn new(inner: S, run_dir: impl Into<PathBuf>) -> Self {
        Self {
            inner,
            run_dir: run_dir.into(),
        }
    }
}

#[async_trait::async_trait]
impl<S: strands_agents::session::SnapshotStorage> strands_agents::session::SnapshotStorage
    for ReserveStorage<S>
{
    async fn save_snapshot(
        &self,
        location: &strands_agents::session::SnapshotLocation,
        snapshot_id: &str,
        is_latest: bool,
        snapshot: &strands_agents::session::Snapshot,
    ) -> strands_agents::Result<()> {
        save_with(&self.run_dir, || {
            self.inner
                .save_snapshot(location, snapshot_id, is_latest, snapshot)
        })
        .await
    }

    async fn load_snapshot(
        &self,
        location: &strands_agents::session::SnapshotLocation,
        snapshot_id: Option<&str>,
    ) -> strands_agents::Result<Option<strands_agents::session::Snapshot>> {
        self.inner.load_snapshot(location, snapshot_id).await
    }

    async fn list_snapshot_ids(
        &self,
        location: &strands_agents::session::SnapshotLocation,
        limit: Option<usize>,
        start_after: Option<&str>,
    ) -> strands_agents::Result<Vec<String>> {
        self.inner.list_snapshot_ids(location, limit, start_after).await
    }

    async fn delete_session(&self, session_id: &str) -> strands_agents::Result<()> {
        self.inner.delete_session(session_id).await
    }

    async fn load_manifest(
        &self,
        location: &strands_agents::session::SnapshotLocation,
    ) -> strands_agents::Result<strands_agents::session::SnapshotManifest> {
        self.inner.load_manifest(location).await
    }

    async fn save_manifest(
        &self,
        location: &strands_agents::session::SnapshotLocation,
        manifest: &strands_agents::session::SnapshotManifest,
    ) -> strands_agents::Result<()> {
        save_with(&self.run_dir, || self.inner.save_manifest(location, manifest)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full() -> std::io::Error {
        std::io::Error::from_raw_os_error(28)
    }

    #[test]
    fn a_reserve_is_allocated_once_and_released_once() {
        let dir = tempfile::tempdir().unwrap();
        assert!(ensure(dir.path(), 1 << 20).unwrap());
        assert!(!ensure(dir.path(), 1 << 20).unwrap(), "already there");
        let meta = std::fs::metadata(path(dir.path())).unwrap();
        assert_eq!(meta.len(), 1 << 20);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert!(meta.blocks() * 512 >= 1 << 20, "allocated, not sparse");
        }
        assert!(release(dir.path()));
        assert!(!release(dir.path()));
        assert!(!path(dir.path()).exists());
    }

    #[test]
    fn a_full_disk_is_recognized_through_wrapping_errors() {
        assert!(is_disk_full(&full()));
        let wrapped: strands_agents::Error = full().into();
        assert!(is_disk_full(&wrapped));
        let other: strands_agents::Error = std::io::Error::from(std::io::ErrorKind::NotFound).into();
        assert!(!is_disk_full(&other));
        let text =
            strands_agents::Error::other("Failed to write file x: No space left on device (os error 28)");
        assert!(is_disk_full(&text));
    }

    #[tokio::test]
    async fn the_first_full_disk_save_frees_the_reserve_and_is_tried_again() {
        let dir = tempfile::tempdir().unwrap();
        ensure(dir.path(), 4096).unwrap();
        let tries = std::sync::atomic::AtomicUsize::new(0);
        let attempt = || {
            let n = tries.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async move {
                match n {
                    0 => Err(full()),
                    _ => Ok(n),
                }
            }
        };
        assert_eq!(save_with(dir.path(), attempt).await.unwrap(), 1);
        assert!(!path(dir.path()).exists(), "the reserve went");

        // Without a reserve left, a full disk is the save's error.
        let failing = || async { Err::<(), _>(full()) };
        assert!(save_with(dir.path(), failing).await.is_err());
        // Other errors leave the reserve alone.
        ensure(dir.path(), 4096).unwrap();
        let denied = || async { Err::<(), _>(std::io::Error::from(std::io::ErrorKind::PermissionDenied)) };
        assert!(save_with(dir.path(), denied).await.is_err());
        assert!(path(dir.path()).exists());
    }
}
