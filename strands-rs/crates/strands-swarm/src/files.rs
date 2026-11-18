//! Replacing a file whole, so that a crash never leaves half of one.

use std::io::Write;
use std::path::Path;

#[cfg(test)]
thread_local! {
    /// Files [`replace`] synced on this thread, for tests of what is written durably.
    pub static SYNCED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Write `bytes` to `path` (creating its directory) through a temporary file beside it and a
/// rename, so a crash leaves the old file or the new one, never half of one. `durable` also syncs
/// the file before the rename and the directory after it, so that holds after a power loss too.
pub fn replace(path: &Path, bytes: &[u8], durable: bool) -> std::io::Result<()> {
    let dir = path.parent().filter(|dir| !dir.as_os_str().is_empty());
    if let Some(dir) = dir {
        std::fs::create_dir_all(dir)?;
    }
    // Per process: two swarms may share a file (the memory file).
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!("{TMP_INFIX}{}", std::process::id()));
    let written = (|| {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(bytes)?;
        if durable {
            file.sync_all()?;
            #[cfg(test)]
            SYNCED.with(|synced| synced.set(synced.get() + 1));
        }
        drop(file);
        std::fs::rename(&tmp, path)
    })();
    if written.is_err() {
        // A full disk would otherwise keep the half-written copy, and fill up the more for it.
        let _ = std::fs::remove_file(&tmp);
    }
    written?;
    if let (true, Some(dir)) = (durable, dir) {
        let _ = std::fs::File::open(dir).and_then(|dir| dir.sync_all());
    }
    Ok(())
}

/// What [`replace`] puts between a file's name and its process id to name the temporary copy.
const TMP_INFIX: &str = ".tmp-";

/// Delete the temporary copies in `dir` that [`replace`] left in processes no longer running
/// (killed mid-write); returns how many. A live process's copy may be mid-write and is kept.
pub fn remove_stale_temps(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name
            .to_str()
            .and_then(|name| name.rsplit_once(TMP_INFIX))
            .and_then(|(_, pid)| pid.parse::<u32>().ok())
        else {
            continue;
        };
        let alive = pid == std::process::id() || process_exists(pid);
        if !alive && std::fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// Whether a process `pid` exists: signal 0 reaches it, or it belongs to another user (EPERM).
/// Only "no such process" counts as gone, so a copy is never deleted from under its writer.
fn process_exists(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    if pid <= 0 {
        return true;
    }
    !matches!(
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None),
        Err(nix::errno::Errno::ESRCH)
    )
}

/// [`replace`] on a blocking thread. `bytes` may be shared (an `Arc<Vec<u8>>`), so that a retry
/// needs no copy.
pub async fn replace_async<B>(path: &Path, bytes: B, durable: bool) -> std::io::Result<()>
where
    B: std::ops::Deref + Send + 'static,
    B::Target: AsRef<[u8]>,
{
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || replace(&path, (*bytes).as_ref(), durable))
        .await
        .map_err(std::io::Error::other)?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_left_by_dead_processes_are_removed_and_a_live_ones_kept() {
        let dir = tempfile::tempdir().unwrap();
        // Pids above the kernel's limit (4194304) are never alive.
        let dead = dir.path().join("swarm.json.tmp-4194399");
        let mine = dir.path().join(format!("swarm.json.tmp-{}", std::process::id()));
        let other = dir.path().join("notes.tmp-old");
        for path in [&dead, &mine, &other] {
            std::fs::write(path, "x").unwrap();
        }
        // pid 1 is alive and another user's: EPERM must not count as gone.
        let foreign = dir.path().join("swarm.json.tmp-1");
        std::fs::write(&foreign, "x").unwrap();
        assert_eq!(remove_stale_temps(dir.path()), 1);
        assert!(!dead.exists() && mine.exists() && other.exists() && foreign.exists());
    }

    #[test]
    fn a_failed_write_leaves_no_temporary_copy() {
        let dir = tempfile::tempdir().unwrap();
        // Renaming a file over a non-empty directory fails after the copy is written.
        let path = dir.path().join("state");
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("inside"), "x").unwrap();
        assert!(replace(&path, b"one", false).is_err());
        let names: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert_eq!(names, ["state"]);
    }

    #[tokio::test]
    async fn a_file_is_replaced_whole_and_no_temporary_file_stays() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new/dir/state.json");
        replace_async(&path, b"one".to_vec(), true).await.unwrap();
        replace(&path, b"two", false).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "two");
        let names: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(names, ["state.json"]);
    }
}
