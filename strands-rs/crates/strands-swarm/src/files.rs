//! Replacing a file whole, so that a crash never leaves half of one.

use std::io::Write;
use std::path::Path;

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
    tmp.push(format!(".tmp-{}", std::process::id()));
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(bytes)?;
    if durable {
        file.sync_all()?;
    }
    drop(file);
    std::fs::rename(&tmp, path)?;
    if let (true, Some(dir)) = (durable, dir) {
        let _ = std::fs::File::open(dir).and_then(|dir| dir.sync_all());
    }
    Ok(())
}

/// [`replace`] on a blocking thread.
pub async fn replace_async(path: &Path, bytes: Vec<u8>, durable: bool) -> std::io::Result<()> {
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || replace(&path, &bytes, durable))
        .await
        .map_err(std::io::Error::other)?
}

#[cfg(test)]
mod tests {
    use super::*;

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
