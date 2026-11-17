//! Where `tracing` output goes. The TUI owns the terminal, so without a log file every warning
//! (a failed summarization, an unreadable memory file, a skipped MCP server) would be lost.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use std::sync::Mutex;

use tracing_subscriber::fmt::writer::BoxMakeWriter;
use tracing_subscriber::EnvFilter;

/// The log file of a run, next to its journal.
pub const LOG_FILE: &str = "swarm.log";
/// A log file over this size is moved to `<file>.1` when the next process opens it.
pub const MAX_LOG_BYTES: u64 = 8 << 20;

/// Opens `path` for appending, creating its directory, and first moves a file that outgrew
/// `max_bytes` to `<path>.1` (replacing the previous one) so the log stays bounded.
pub fn open_log(path: &Path, max_bytes: u64) -> io::Result<File> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    if std::fs::metadata(path).is_ok_and(|meta| meta.len() > max_bytes) {
        let mut rotated = path.as_os_str().to_owned();
        rotated.push(".1");
        std::fs::rename(path, rotated)?;
    }
    OpenOptions::new().create(true).append(true).open(path)
}

/// The filter: `RUST_LOG` when set and valid, else warnings and errors.
pub fn filter() -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"))
}

/// Installs the global subscriber, writing to `file` or, without one, to stderr.
pub fn init(file: Option<File>) {
    let writer = match file {
        Some(file) => BoxMakeWriter::new(Mutex::new(file)),
        None => BoxMakeWriter::new(io::stderr),
    };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter())
        .with_ansi(false)
        .with_writer(writer)
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn appends_and_rotates_an_oversized_log() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run").join(LOG_FILE);
        open_log(&path, 10).unwrap().write_all(b"first\n").unwrap();
        open_log(&path, 10).unwrap().write_all(b"second\n").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "first\nsecond\n");

        open_log(&path, 10).unwrap().write_all(b"third\n").unwrap();
        let rotated = dir.path().join("run").join(format!("{LOG_FILE}.1"));
        assert_eq!(std::fs::read_to_string(rotated).unwrap(), "first\nsecond\n");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "third\n");
    }
}
