//! Append-only audit trail of a persistent run, next to its state:
//!
//! - `journal.jsonl`: every message to or from the user, so their requests and the swarm's
//!   answers are on record.
//! - `events.jsonl`: compact lifecycle events (spawns, status changes, tool calls, model switches,
//!   resumes, reloads, shutdowns), so a run can be audited after the fact.
//!
//! Lines are JSON objects with an RFC 3339 `ts`. A dedicated thread does the file I/O, so
//! recording never blocks the swarm. Each file is rotated to `<file>.1` once it would exceed
//! [`MAX_FILE_BYTES`], which bounds a run to twice that per file.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::runtime::SwarmEvent;

/// Messages to and from the user.
pub const JOURNAL_FILE: &str = "journal.jsonl";
/// Lifecycle events.
pub const EVENTS_FILE: &str = "events.jsonl";
/// Size at which a file is rotated to `<file>.1`.
pub const MAX_FILE_BYTES: u64 = 8 << 20;

/// One line of `journal.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalEntry {
    /// RFC 3339 UTC timestamp.
    pub ts: String,
    /// Sender id (`user` for the human).
    pub from: String,
    /// Sender label, e.g. `parser (a2)`.
    pub from_label: String,
    /// Receiver id.
    pub to: String,
    /// Receiver label.
    pub to_label: String,
    /// `message`, `reply` or `error`.
    pub kind: String,
    /// Body.
    pub text: String,
}

impl JournalEntry {
    /// `2025-01-02T03:04:05.678Z user → parser (a2) [message]` followed by the indented text.
    pub fn render(&self) -> String {
        let body: Vec<String> = self.text.lines().map(|line| format!("  {line}")).collect();
        format!(
            "{} {} → {} [{}]\n{}\n",
            self.ts,
            self.from_label,
            self.to_label,
            self.kind,
            body.join("\n")
        )
    }
}

enum Op {
    Line(&'static str, String),
    Flush(tokio::sync::oneshot::Sender<()>),
}

/// Handle to the background writer of one run directory.
pub struct Journal {
    tx: mpsc::Sender<Op>,
}

impl Journal {
    /// Start a writer appending to `dir` (created on first write).
    pub fn open(dir: impl Into<PathBuf>) -> Self {
        Self::with_limit(dir, MAX_FILE_BYTES)
    }

    fn with_limit(dir: impl Into<PathBuf>, max_bytes: u64) -> Self {
        let (tx, rx) = mpsc::channel();
        let mut writer = Writer {
            dir: dir.into(),
            max_bytes,
            files: Vec::new(),
        };
        std::thread::Builder::new()
            .name("swarm-journal".into())
            .spawn(move || writer.run(rx))
            .expect("spawn journal thread");
        Self { tx }
    }

    /// Record a message to or from the user.
    pub fn mail(&self, from: &str, from_label: &str, to: &str, to_label: &str, kind: &str, text: &str) {
        let entry = JournalEntry {
            ts: now(),
            from: from.to_owned(),
            from_label: from_label.to_owned(),
            to: to.to_owned(),
            to_label: to_label.to_owned(),
            kind: kind.to_owned(),
            text: text.to_owned(),
        };
        if let Ok(line) = serde_json::to_string(&entry) {
            self.send(Op::Line(JOURNAL_FILE, line));
        }
    }

    /// Record a lifecycle event: `fields` (a JSON object) gains `ts` and `event`.
    pub fn event(&self, event: &str, fields: Value) {
        let mut line = json!({ "ts": now(), "event": event });
        if let (Some(line), Value::Object(fields)) = (line.as_object_mut(), fields) {
            line.extend(fields);
        }
        self.send(Op::Line(EVENTS_FILE, line.to_string()));
    }

    /// Record the lifecycle part of a swarm event; streaming text, usage and mail are skipped.
    pub fn record(&self, event: &SwarmEvent) {
        if let SwarmEvent::Reclaimed { collected, .. } = event {
            for agent in collected.iter().filter(|agent| !agent.is_empty()) {
                self.event("gc", agent.journal_fields());
            }
            return;
        }
        let (name, fields) = match event {
            SwarmEvent::Spawned {
                id,
                parent,
                name,
                task,
                cwd,
                branch,
            } => (
                "spawned",
                json!({"id": id, "parent": parent, "name": name, "task": task, "cwd": cwd, "branch": branch}),
            ),
            SwarmEvent::Status { id, status } => ("status", json!({"id": id, "status": status.to_string()})),
            SwarmEvent::ToolStarted { id, name, summary } => {
                ("tool", json!({"id": id, "tool": name, "input": summary}))
            }
            SwarmEvent::ToolFinished { id, name, ok } => {
                ("tool_done", json!({"id": id, "tool": name, "ok": ok}))
            }
            SwarmEvent::Model { id, model } => ("model", json!({"id": id, "model": model})),
            SwarmEvent::Budget { tokens } => ("budget", json!({"tokens": tokens})),
            SwarmEvent::Text { .. }
            | SwarmEvent::Usage { .. }
            | SwarmEvent::Mail { .. }
            | SwarmEvent::MailRead { .. }
            | SwarmEvent::ModelQueued { .. }
            | SwarmEvent::Reclaimed { .. }
            | SwarmEvent::History { .. } => return,
        };
        self.event(name, fields);
    }

    /// Wait until everything recorded so far is on disk.
    pub async fn flush(&self) {
        let (done, wait) = tokio::sync::oneshot::channel();
        self.send(Op::Flush(done));
        let _ = wait.await;
    }

    fn send(&self, op: Op) {
        let _ = self.tx.send(op);
    }
}

struct Writer {
    dir: PathBuf,
    max_bytes: u64,
    files: Vec<(&'static str, File, u64)>,
}

impl Writer {
    fn run(&mut self, rx: mpsc::Receiver<Op>) {
        while let Ok(op) = rx.recv() {
            let mut pending = vec![op];
            pending.extend(rx.try_iter());
            for op in pending {
                match op {
                    Op::Line(file, line) => {
                        if let Err(error) = self.append(file, &line) {
                            tracing::warn!(%error, file, "journal write failed");
                        }
                    }
                    Op::Flush(done) => {
                        let _ = done.send(());
                    }
                }
            }
        }
    }

    fn append(&mut self, name: &'static str, line: &str) -> std::io::Result<()> {
        let len = line.len() as u64 + 1;
        let index = match self.files.iter().position(|(n, ..)| *n == name) {
            Some(index) => index,
            None => {
                fs::create_dir_all(&self.dir)?;
                let file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(self.dir.join(name))?;
                let size = file.metadata()?.len();
                self.files.push((name, file, size));
                self.files.len() - 1
            }
        };
        let (_, file, size) = &mut self.files[index];
        if *size > 0 && *size + len > self.max_bytes {
            let path = self.dir.join(name);
            fs::rename(&path, rotated(&path))?;
            *file = OpenOptions::new().create(true).append(true).open(&path)?;
            *size = 0;
        }
        file.write_all(format!("{line}\n").as_bytes())?;
        *size += len;
        Ok(())
    }
}

fn rotated(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".1");
    PathBuf::from(name)
}

/// Every journal entry of the run in `dir`, oldest first (the rotated file, then the current
/// one). Unreadable lines are skipped.
pub fn read(dir: &Path) -> std::io::Result<Vec<JournalEntry>> {
    let path = dir.join(JOURNAL_FILE);
    let mut entries = Vec::new();
    for path in [rotated(&path), path] {
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        for line in BufReader::new(file).lines() {
            if let Ok(entry) = serde_json::from_str(&line?) {
                entries.push(entry);
            }
        }
    }
    Ok(entries)
}

/// Now as RFC 3339 UTC with milliseconds.
fn now() -> String {
    let elapsed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    rfc3339(elapsed.as_millis() as i64)
}

fn rfc3339(millis: i64) -> String {
    let (days, ms_of_day) = (millis.div_euclid(86_400_000), millis.rem_euclid(86_400_000));
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let secs = ms_of_day / 1000;
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        secs / 3600,
        secs / 60 % 60,
        secs % 60,
        ms_of_day % 1000
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::AgentStatus;

    #[test]
    fn formats_timestamps_as_rfc3339() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(rfc3339(951_782_400_123), "2000-02-29T00:00:00.123Z");
        assert_eq!(rfc3339(1_735_787_045_678), "2025-01-02T03:04:05.678Z");
    }

    #[tokio::test]
    async fn writes_mail_and_lifecycle_events_and_reads_the_journal_back() {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open(dir.path());
        journal.mail("user", "user", "a1", "root (a1)", "message", "fix it\nplease");
        journal.record(&SwarmEvent::Status {
            id: "a1".into(),
            status: AgentStatus::Running,
        });
        journal.record(&SwarmEvent::Text {
            id: "a1".into(),
            delta: "skipped".into(),
        });
        journal.flush().await;

        let entries = read(dir.path()).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "fix it\nplease");
        assert!(entries[0]
            .render()
            .contains("user → root (a1) [message]\n  fix it\n  please"));

        let events = fs::read_to_string(dir.path().join(EVENTS_FILE)).unwrap();
        let lines: Vec<Value> = events.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["event"], "status");
        assert_eq!(lines[0]["status"], "running");
        assert!(lines[0]["ts"].as_str().unwrap().ends_with('Z'));
    }

    #[tokio::test]
    async fn rotates_a_full_file_and_reads_both_halves() {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::with_limit(dir.path(), 300);
        for n in 0..6 {
            journal.mail("user", "user", "a1", "a1", "message", &format!("request {n}"));
        }
        journal.flush().await;
        assert!(dir.path().join("journal.jsonl.1").exists());
        let texts: Vec<String> = read(dir.path()).unwrap().into_iter().map(|e| e.text).collect();
        // At most two files are kept; whatever survives is in order and ends with the newest.
        assert_eq!(texts.last().unwrap(), "request 5");
        assert!(texts.windows(2).all(|w| w[0] < w[1]), "{texts:?}");
        assert!(fs::metadata(dir.path().join(JOURNAL_FILE)).unwrap().len() <= 300);
    }

    #[test]
    fn reading_a_run_without_a_journal_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read(dir.path()).unwrap().is_empty());
    }
}
