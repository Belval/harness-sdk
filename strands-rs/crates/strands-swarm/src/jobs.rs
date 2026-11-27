//! Background shell jobs: commands that run detached from the swarm process, so a restart does
//! not kill them, and report back to the agent that started them when they end.
//!
//! Everything lives on disk under one directory (`<run>/jobs`): `<id>.json` describes the job,
//! `<id>.log` takes its output, a small `sh` wrapper writes the exit code to `<id>.exit`, and
//! `<id>.reported` marks a job whose end was told to its owner. `<id>.exit` stays after that, until
//! the job is pruned ([`Jobs::tidy`]), so a script can wait on it. A restarted swarm reads the same
//! files, so jobs started before a reload still report.

use std::collections::BTreeMap;
use std::io::SeekFrom;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use nix::sys::signal::Signal;
use nix::unistd::Pid;
use serde::{Deserialize, Serialize};
use strands_agents::sandbox::session_command;
use strands_agents::{Error, Result};
use tokio::io::{AsyncReadExt, AsyncSeekExt};

/// Lines of output a job's end report carries.
pub const TAIL_LINES: usize = 40;

/// Bytes read from the end of a log for its tail.
const TAIL_BYTES: u64 = 16 * 1024;

/// Runs `$1` with output appended to `$2` (append, so [`Jobs::tidy`] can cut the log while it
/// runs), then writes its exit code to `$3` (renamed into place, so a reader never sees it
/// half-written).
const WRAPPER: &str = r#"sh -c "$1" >>"$2" 2>&1 </dev/null; echo $? >"$3.tmp"; mv "$3.tmp" "$3""#;

/// A running job's log is cut to its last [`LOG_KEEP`] bytes once it exceeds this.
pub const LOG_MAX: u64 = 32 * 1024 * 1024;

/// What is kept of a log that exceeded [`LOG_MAX`].
pub const LOG_KEEP: u64 = 1024 * 1024;

/// Reported jobs kept on disk; older ones' files are deleted.
pub const KEEP_JOBS: usize = 200;

/// How a job runs and reports: [`JobMeta::restart`] and [`JobMeta::quiet`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mode {
    /// [`JobMeta::restart`].
    pub restart: bool,
    /// [`JobMeta::quiet`].
    pub quiet: bool,
}

impl Mode {
    /// Rerun until killed.
    pub const RESTART: Mode = Mode {
        restart: true,
        quiet: false,
    };
}

/// What `<id>.json` holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobMeta {
    /// `j1`, `j2`, … unique within the run.
    pub id: String,
    /// Agent that started it and is told when it ends.
    pub owner: String,
    /// Optional short label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The shell command.
    pub command: String,
    /// Working directory.
    pub cwd: PathBuf,
    /// Process id of the wrapper, which leads the job's process group.
    pub pid: u32,
    /// Start time (of the latest run), seconds since the Unix epoch.
    pub started: u64,
    /// Run it again each time it ends, under this id, until it is killed (the watcher pattern).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub restart: bool,
    /// A run that exits 0 reports without waking the owner (an FYI); any other end still wakes it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub quiet: bool,
    /// How many times it has been started again.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub restarts: u32,
    /// Runs in a row that ended within [`QUICK_EXIT`], up to the latest restart.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub quick: u32,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

/// A restarting job that ends within this many seconds counts as a quick exit.
pub const QUICK_EXIT: u64 = 10;

/// Quick exits in a row after which restarts back off.
const QUICK_EXITS_FREE: u32 = 3;

/// The wait before the next restart after `quick` quick exits in a row: none for the first
/// [`QUICK_EXITS_FREE`] - 1, then a minute, doubling up to 30 minutes.
pub fn restart_delay(quick: u32) -> std::time::Duration {
    let seconds = match quick.checked_sub(QUICK_EXITS_FREE) {
        None => 0,
        Some(n) => (60u64 << n.min(5)).min(30 * 60),
    };
    std::time::Duration::from_secs(seconds)
}

/// Where a job is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobState {
    /// Still running.
    Running,
    /// Described but not started: waiting for a slot under `--max-jobs` (`pid` is 0).
    Queued,
    /// Exited with this code.
    Exited(i32),
    /// Killed through [`Jobs::kill`].
    Killed,
    /// Its process is gone without an exit code (e.g. the machine rebooted).
    Lost,
}

/// A job as read from disk.
#[derive(Debug, Clone)]
pub struct Job {
    /// Its description.
    pub meta: JobMeta,
    /// Its state.
    pub state: JobState,
    /// Its owner has been told it ended.
    pub reported: bool,
    /// Seconds it ran (so far, while running).
    pub seconds: u64,
}

impl Job {
    /// Ended, to be started again ([`JobMeta::restart`]); a killed job never is.
    pub fn restarting(&self) -> bool {
        self.meta.restart && matches!(self.state, JobState::Exited(_) | JobState::Lost)
    }

    /// Running, or waiting to be started again.
    pub fn live(&self) -> bool {
        matches!(self.state, JobState::Running | JobState::Queued) || self.restarting()
    }

    /// Quick exits in a row once this run is over: the count the next restart is delayed by.
    pub fn quick_after(&self) -> u32 {
        match self.seconds < QUICK_EXIT {
            true => self.meta.quick.saturating_add(1),
            false => 0,
        }
    }

    /// `j3 (gates)`.
    pub fn label(&self) -> String {
        match &self.meta.name {
            Some(name) => format!("{} ({name})", self.meta.id),
            None => self.meta.id.clone(),
        }
    }

    /// `running 2m05s`, `exited 0 after 41s`, `killed after 3s`.
    pub fn status(&self) -> String {
        let time = duration(self.seconds);
        let status = match self.state {
            JobState::Running => format!("running {time}"),
            JobState::Queued => format!("queued {time}: waiting for a slot (--max-jobs)"),
            JobState::Exited(code) => format!("exited {code} after {time}"),
            JobState::Killed => format!("killed after {time}"),
            JobState::Lost => "ended without an exit code".to_owned(),
        };
        match self.meta.restarts {
            0 => status,
            n => format!("{status}, restarts {n}"),
        }
    }
}

/// `41s`, `2m05s`, `1h03m`.
fn duration(seconds: u64) -> String {
    match seconds {
        s if s < 60 => format!("{s}s"),
        s if s < 3600 => format!("{}m{:02}s", s / 60, s % 60),
        s => format!("{}h{:02}m", s / 3600, s % 3600 / 60),
    }
}

/// The number of job id `j<n>`.
fn number(id: &str) -> Option<u64> {
    id.strip_prefix('j')?.parse().ok()
}

/// `command` with no stdin and its output discarded.
fn quiet(command: &mut tokio::process::Command) -> &mut tokio::process::Command {
    command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The jobs of one run, kept in a directory.
#[derive(Debug)]
pub struct Jobs {
    dir: PathBuf,
    /// Serializes id allocation within this process.
    start: tokio::sync::Mutex<()>,
}

impl Jobs {
    /// Jobs kept in `dir` (created on the first start).
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            start: tokio::sync::Mutex::new(()),
        }
    }

    /// The directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Where job `id` writes its output.
    pub fn log_path(&self, id: &str) -> PathBuf {
        self.file(id, "log")
    }

    fn file(&self, id: &str, ext: &str) -> PathBuf {
        self.dir.join(format!("{id}.{ext}"))
    }

    /// Start `command` with `sh` in `cwd`, with `env` on top of this process's environment, in
    /// a session of its own so that neither the swarm exiting nor a Ctrl-C in its terminal stops
    /// it, and nothing it runs can write to that terminal.
    pub async fn start(
        &self,
        owner: &str,
        name: Option<String>,
        command: &str,
        cwd: &Path,
        env: &BTreeMap<String, String>,
        mode: Mode,
    ) -> Result<Job> {
        self.create(owner, name, command, cwd, mode, Some(env)).await
    }

    /// Describe a job that is not started yet ([`JobState::Queued`]); [`Self::launch`] starts it.
    pub async fn queue(
        &self,
        owner: &str,
        name: Option<String>,
        command: &str,
        cwd: &Path,
        mode: Mode,
    ) -> Result<Job> {
        self.create(owner, name, command, cwd, mode, None).await
    }

    /// A new job, started with `env`, or only described when there is none.
    async fn create(
        &self,
        owner: &str,
        name: Option<String>,
        command: &str,
        cwd: &Path,
        mode: Mode,
        env: Option<&BTreeMap<String, String>>,
    ) -> Result<Job> {
        let _guard = self.start.lock().await;
        tokio::fs::create_dir_all(&self.dir).await?;
        let id = self.next_id().await?;
        let pid = match env {
            Some(env) => self.spawn(&id, command, cwd, env)?,
            None => 0,
        };
        let meta = JobMeta {
            id: id.clone(),
            owner: owner.to_owned(),
            name: name.filter(|n| !n.trim().is_empty()),
            command: command.to_owned(),
            cwd: cwd.to_path_buf(),
            pid,
            started: now(),
            restart: mode.restart,
            quiet: mode.quiet,
            restarts: 0,
            quick: 0,
        };
        self.describe(&meta).await?;
        Ok(Job {
            meta,
            state: match env {
                Some(_) => JobState::Running,
                None => JobState::Queued,
            },
            reported: false,
            seconds: 0,
        })
    }

    /// Start queued job `id` now, with `env`; `None` when it is not queued any more (killed).
    pub async fn launch(&self, id: &str, env: &BTreeMap<String, String>) -> Result<Option<Job>> {
        let _guard = self.start.lock().await;
        let job = self.get(id).await?;
        if job.state != JobState::Queued {
            return Ok(None);
        }
        let mut meta = job.meta;
        meta.pid = self.spawn(id, &meta.command, &meta.cwd, env)?;
        meta.started = now();
        self.describe(&meta).await?;
        Ok(Some(self.read(meta).await))
    }

    /// Spawn the wrapper for job `id`; its process id.
    fn spawn(&self, id: &str, command: &str, cwd: &Path, env: &BTreeMap<String, String>) -> Result<u32> {
        let mut child = quiet(&mut session_command("sh"))
            .arg("-c")
            .arg(WRAPPER)
            .arg("job")
            .arg(command)
            .arg(self.log_path(id))
            .arg(self.file(id, "exit"))
            .current_dir(cwd)
            .envs(env)
            .spawn()
            .map_err(|e| Error::other(format!("Could not start the job: {e}")))?;
        let pid = child.id().unwrap_or_default();
        // Reap the wrapper once it ends; the job reports through its files, not this handle.
        tokio::spawn(async move { child.wait().await });
        Ok(pid)
    }

    /// Write `<id>.json`. A crash mid-write must not leave a half one, which no listing could read.
    async fn describe(&self, meta: &JobMeta) -> Result<()> {
        let json = serde_json::to_vec_pretty(meta)?;
        crate::files::replace_async(&self.file(&meta.id, "json"), json, false).await?;
        Ok(())
    }

    /// Start restarting job `id` again, under the same id and with a fresh log, if it is still
    /// waiting for it: reported ended (or its process gone after a crash between the steps
    /// here) and not killed. `None` when it is not. The report mark goes last, so the watcher
    /// never reports the old run again nor misses the new one.
    pub async fn restart(&self, id: &str, env: &BTreeMap<String, String>) -> Result<Option<Job>> {
        let _guard = self.start.lock().await;
        let job = self.get(id).await?;
        let gone = job.state == JobState::Running && !alive(job.meta.pid);
        if !job.meta.restart || !job.reported || !(job.restarting() || gone) {
            return Ok(None);
        }
        for ext in ["exit", "stalled"] {
            let _ = tokio::fs::remove_file(self.file(id, ext)).await;
        }
        tokio::fs::write(self.log_path(id), b"").await?;
        let mut meta = job.meta.clone();
        meta.pid = self.spawn(id, &meta.command, &meta.cwd, env)?;
        meta.started = now();
        meta.restarts += 1;
        meta.quick = match gone {
            true => 0,
            false => job.quick_after(),
        };
        self.describe(&meta).await?;
        let _ = tokio::fs::remove_file(self.file(id, "reported")).await;
        Ok(Some(self.read(meta).await))
    }

    /// The restarting jobs whose owner was told they ended but which have not started again
    /// (the swarm went down meanwhile), with how long each still waits.
    pub async fn pending_restarts(&self) -> Vec<(String, std::time::Duration)> {
        let mut pending = Vec::new();
        for job in self.list().await {
            let gone = job.state == JobState::Running && !alive(job.meta.pid);
            if !job.meta.restart || !job.reported || !(job.restarting() || gone) {
                continue;
            }
            let ended = modified(&self.file(&job.meta.id, "exit"))
                .await
                .unwrap_or_else(now);
            let waited = std::time::Duration::from_secs(now().saturating_sub(ended));
            let wait = restart_delay(job.quick_after()).saturating_sub(waited);
            pending.push((job.meta.id, wait));
        }
        pending
    }

    /// One more than the highest job number on disk. Any file counts, not just a job's `.json`:
    /// a swarm killed between starting a job and describing it leaves only its `.log` (and later
    /// `.exit`), and a new job must not share them.
    async fn next_id(&self) -> Result<String> {
        let mut highest = 0;
        if let Ok(mut entries) = tokio::fs::read_dir(&self.dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let name = entry.file_name();
                let id = name.to_str().and_then(|name| name.split('.').next());
                highest = highest.max(id.and_then(number).unwrap_or(0));
            }
        }
        Ok(format!("j{}", highest + 1))
    }

    async fn metas(&self) -> Vec<JobMeta> {
        let Ok(mut entries) = tokio::fs::read_dir(&self.dir).await else {
            return Vec::new();
        };
        let mut metas = Vec::new();
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                let parsed = tokio::fs::read(&path)
                    .await
                    .ok()
                    .and_then(|bytes| serde_json::from_slice::<JobMeta>(&bytes).ok());
                metas.extend(parsed);
            }
        }
        metas.sort_by_key(|meta| (number(&meta.id), meta.id.clone()));
        metas
    }

    /// Every job, oldest first.
    pub async fn list(&self) -> Vec<Job> {
        let mut jobs = Vec::new();
        for meta in self.metas().await {
            jobs.push(self.read(meta).await);
        }
        jobs
    }

    /// One job.
    pub async fn get(&self, id: &str) -> Result<Job> {
        let bytes = tokio::fs::read(self.file(id, "json"))
            .await
            .map_err(|_| Error::other(format!("No job `{id}`; call jobs to list them.")))?;
        Ok(self.read(serde_json::from_slice(&bytes)?).await)
    }

    async fn read(&self, meta: JobMeta) -> Job {
        let exit = self.file(&meta.id, "exit");
        let queued = meta.pid == 0;
        let state = match tokio::fs::read_to_string(&exit).await {
            Ok(text) => match text.trim() {
                "killed" => JobState::Killed,
                "lost" => JobState::Lost,
                code => code.parse().map_or(JobState::Lost, JobState::Exited),
            },
            Err(_) if queued => JobState::Queued,
            Err(_) => JobState::Running,
        };
        let end = match state {
            JobState::Running | JobState::Queued => now(),
            _ => modified(&exit).await.unwrap_or_else(now),
        };
        let reported = tokio::fs::try_exists(self.file(&meta.id, "reported"))
            .await
            .unwrap_or(false);
        Job {
            seconds: end.saturating_sub(meta.started),
            meta,
            state,
            reported,
        }
    }

    /// Record that job `id`'s owner was told it ended.
    pub async fn mark_reported(&self, id: &str) -> Result<()> {
        tokio::fs::write(self.file(id, "reported"), b"").await?;
        Ok(())
    }

    /// Running jobs whose log has not grown (nor been touched, see [`crate::land`]) for `after`
    /// and that were not noted before ([`Self::mark_stalled`]); a log that does not exist yet
    /// counts from the job's start.
    pub async fn stalled(&self, after: std::time::Duration) -> Vec<Job> {
        let mut stalled = Vec::new();
        for id in self.unreported_ids().await {
            if tokio::fs::try_exists(self.file(&id, "stalled"))
                .await
                .unwrap_or(false)
            {
                continue;
            }
            let Ok(job) = self.get(&id).await else {
                continue;
            };
            if job.state != JobState::Running || job.reported {
                continue;
            }
            let started = UNIX_EPOCH + std::time::Duration::from_secs(job.meta.started);
            let written = tokio::fs::metadata(self.log_path(&id))
                .await
                .and_then(|m| m.modified());
            let last = written.map_or(started, |written| written.max(started));
            if last.elapsed().is_ok_and(|quiet| quiet >= after) {
                stalled.push(job);
            }
        }
        stalled
    }

    /// Record that job `id`'s owner was told its log stopped growing, so it is not told again
    /// (not after a restart either).
    pub async fn mark_stalled(&self, id: &str) -> Result<()> {
        tokio::fs::write(self.file(id, "stalled"), b"").await?;
        Ok(())
    }

    /// Ended jobs whose owner has not been told yet. Running jobs whose process has disappeared
    /// are recorded as lost first when `check_alive`.
    ///
    /// The watcher asks twice a second, so reported jobs (up to [`KEEP_JOBS`] of them) are told
    /// apart by name in one directory listing, without reading their files.
    pub async fn unreported(&self, check_alive: bool) -> Vec<Job> {
        let mut ended = Vec::new();
        for id in self.unreported_ids().await {
            let Ok(job) = self.get(&id).await else {
                continue;
            };
            if job.reported || job.state == JobState::Queued {
                continue;
            }
            if job.state != JobState::Running {
                ended.push(job);
            } else if check_alive && !alive(job.meta.pid) {
                // The wrapper may have written its exit code just before it went.
                let exit = self.file(&job.meta.id, "exit");
                if !tokio::fs::try_exists(&exit).await.unwrap_or(false) {
                    let _ = tokio::fs::write(&exit, "lost").await;
                }
                // Re-read: it may also have been killed meanwhile, which needs no report.
                let job = self.read(job.meta).await;
                if !job.reported {
                    ended.push(job);
                }
            }
        }
        ended
    }

    /// The ids of the jobs described on disk and not marked reported, oldest first.
    async fn unreported_ids(&self) -> Vec<String> {
        let Ok(mut entries) = tokio::fs::read_dir(&self.dir).await else {
            return Vec::new();
        };
        let (mut described, mut reported) = (Vec::new(), std::collections::HashSet::new());
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(id) = name.strip_suffix(".json") {
                described.push(id.to_owned());
            } else if let Some(id) = name.strip_suffix(".reported") {
                reported.insert(id.to_owned());
            }
        }
        described.retain(|id| !reported.contains(id));
        described.sort_by_key(|id| (number(id), id.clone()));
        described
    }

    /// Whether a job that `matches` still runs (its process alive, whatever its files say).
    pub async fn any_running(&self, matches: impl Fn(&JobMeta) -> bool) -> bool {
        for job in self.list().await {
            if job.state == JobState::Running && matches(&job.meta) && alive(job.meta.pid) {
                return true;
            }
        }
        false
    }

    /// Kill job `id`'s process group, mark it killed and reported (whoever kills it knows).
    pub async fn kill(&self, id: &str) -> Result<Job> {
        // Not while it is being started again.
        let _guard = self.start.lock().await;
        let job = self.get(id).await?;
        if job.restarting() || job.state == JobState::Queued {
            // Waiting to start (again): never will.
            self.mark_reported(id).await?;
            tokio::fs::write(self.file(id, "exit"), "killed").await?;
            return self.get(id).await;
        }
        if job.state != JobState::Running {
            return Ok(job);
        }
        self.mark_reported(id).await?;
        signal_group(job.meta.pid, Signal::SIGTERM);
        tokio::fs::write(self.file(id, "exit"), "killed").await?;
        self.get(id).await
    }

    /// Keep the directory bounded: cut running jobs' logs over `max` bytes to their last `keep`
    /// bytes, and delete the files of all but the newest `jobs` reported jobs.
    pub async fn tidy(&self, max: u64, keep: u64, jobs: usize) {
        let all = self.list().await;
        // A job waiting to start again is not done with, though its end was reported.
        let done = |job: &Job| job.reported && !job.restarting();
        let reported = all.iter().filter(|job| done(job)).count();
        let mut old = reported.saturating_sub(jobs);
        for job in &all {
            let id = &job.meta.id;
            if done(job) && old > 0 {
                old -= 1;
                // The `.json` last: a job is listed while it exists.
                for ext in ["log", "exit", "reported", "stalled", "json"] {
                    let _ = tokio::fs::remove_file(self.file(id, ext)).await;
                }
            } else if job.state == JobState::Running {
                if let Err(error) = cut(&self.log_path(id), max, keep).await {
                    tracing::warn!(job = %id, %error, "could not cut the job's log");
                }
            }
        }
    }

    /// The last `lines` lines of job `id`'s output, as a terminal would show them
    /// ([`crate::text::plain`]).
    pub async fn tail(&self, id: &str, lines: usize) -> String {
        let Ok(mut file) = tokio::fs::File::open(self.log_path(id)).await else {
            return String::new();
        };
        let len = file.metadata().await.map_or(0, |m| m.len());
        let from = len.saturating_sub(TAIL_BYTES);
        if file.seek(SeekFrom::Start(from)).await.is_err() {
            return String::new();
        }
        let mut bytes = Vec::new();
        let _ = file.read_to_end(&mut bytes).await;
        let text = String::from_utf8_lossy(&bytes);
        let text = crate::text::plain(&text);
        let all: Vec<&str> = text.trim_end().lines().collect();
        // A tail cut mid-file starts mid-line; drop that partial line.
        let partial = usize::from(from > 0 && all.len() > 1);
        let keep = all.len().saturating_sub(lines).max(partial);
        all[keep..].join("\n")
    }
}

/// Cut the log at `path` to its last `keep` bytes, marked, once it exceeds `max`. The job
/// appends, so its next write lands after the kept tail; output written while cutting is lost.
async fn cut(path: &Path, max: u64, keep: u64) -> std::io::Result<()> {
    let mut file = tokio::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .await?;
    let len = file.metadata().await?.len();
    if len <= max {
        return Ok(());
    }
    file.seek(SeekFrom::Start(len - keep)).await?;
    let mut tail = Vec::new();
    file.read_to_end(&mut tail).await?;
    let mut text = format!("[strands-swarm: log over {max} bytes, earlier output cut]\n").into_bytes();
    text.extend(tail);
    file.set_len(0).await?;
    file.seek(SeekFrom::Start(0)).await?;
    tokio::io::AsyncWriteExt::write_all(&mut file, &text).await
}

async fn modified(path: &Path) -> Option<u64> {
    let time = tokio::fs::metadata(path).await.ok()?.modified().ok()?;
    time.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs())
}

/// Whether process `pid` still exists (`kill -0`).
fn alive(pid: u32) -> bool {
    positive(pid).is_some_and(|pid| nix::sys::signal::kill(pid, None).is_ok())
}

/// Send `signal` to the process group led by `leader`; whether it went to any process. A child
/// spawned a moment ago may not lead its group yet, which
/// [`strands_agents::sandbox::signal_session`] handles.
pub(crate) fn signal_group(leader: u32, signal: Signal) -> bool {
    strands_agents::sandbox::signal_session(leader, signal)
}

/// `pid` as a [`Pid`], if it names one process (`0` and the negative ones name groups).
fn positive(pid: u32) -> Option<Pid> {
    i32::try_from(pid).ok().filter(|pid| *pid > 0).map(Pid::from_raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    async fn until_ended(jobs: &Jobs, id: &str) -> Job {
        for _ in 0..200 {
            let job = jobs.get(id).await.unwrap();
            if job.state != JobState::Running {
                return job;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("job {id} did not end");
    }

    #[tokio::test]
    async fn a_tail_is_the_text_a_terminal_shows() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = Jobs::new(dir.path().join("jobs"));
        let command = r"printf '\033[1m\033[32mok\033[0m\n 10%%\r 90%%\r100%%\r\nend\n'";
        jobs.start("a1", None, command, dir.path(), &BTreeMap::new(), Mode::default())
            .await
            .unwrap();
        until_ended(&jobs, "j1").await;
        assert_eq!(jobs.tail("j1", 2).await, "100%\nend");
        assert_eq!(jobs.tail("j1", 3).await, "ok\n100%\nend");
    }

    #[tokio::test]
    async fn a_job_runs_in_its_cwd_and_env_and_records_its_output_and_exit_code() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = Jobs::new(dir.path().join("jobs"));
        let env = BTreeMap::from([("JOB_TEST_VAR".to_owned(), "hello".to_owned())]);
        let job = jobs
            .start(
                "a1",
                Some("probe".into()),
                "pwd; echo $JOB_TEST_VAR; seq 1 100; exit 3",
                dir.path(),
                &env,
                Mode::default(),
            )
            .await
            .unwrap();
        assert_eq!(job.meta.id, "j1");
        assert_eq!(job.label(), "j1 (probe)");
        let ended = until_ended(&jobs, "j1").await;
        assert_eq!(ended.state, JobState::Exited(3));
        assert!(!ended.reported);
        let log = tokio::fs::read_to_string(jobs.log_path("j1")).await.unwrap();
        let cwd = dir.path().canonicalize().unwrap();
        assert!(
            log.starts_with(&format!("{}\nhello\n1\n", cwd.display())),
            "{log}"
        );
        let tail = jobs.tail("j1", 3).await;
        assert_eq!(tail, "98\n99\n100");

        assert_eq!(jobs.unreported(false).await.len(), 1);
        jobs.mark_reported("j1").await.unwrap();
        assert!(jobs.unreported(true).await.is_empty());

        let next = jobs
            .start("a2", None, "true", dir.path(), &BTreeMap::new(), Mode::default())
            .await;
        assert_eq!(next.unwrap().meta.id, "j2", "ids continue after those on disk");
    }

    #[tokio::test]
    async fn a_job_the_swarm_died_before_describing_keeps_its_id_and_files() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = Jobs::new(dir.path().join("jobs"));
        tokio::fs::create_dir_all(dir.path().join("jobs")).await.unwrap();
        // Killed between starting j1 and writing its `.json` (or mid-write, before the rename).
        tokio::fs::write(jobs.log_path("j1"), "orphan output\n")
            .await
            .unwrap();
        tokio::fs::write(jobs.file("j1", "exit"), "0").await.unwrap();
        tokio::fs::write(jobs.file("j2", "json.tmp"), "{\"id\": \"j2\", \"ow")
            .await
            .unwrap();
        assert!(jobs.list().await.is_empty(), "nothing half-written is listed");

        let job = jobs
            .start("a1", None, "true", dir.path(), &BTreeMap::new(), Mode::default())
            .await
            .unwrap();
        assert_eq!(job.meta.id, "j3", "no new job writes into a dead one's files");
        let log = tokio::fs::read_to_string(jobs.log_path("j1")).await.unwrap();
        assert_eq!(log, "orphan output\n");
        assert!(
            !std::fs::read_dir(jobs.dir()).unwrap().any(|entry| entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("j3.json.")),
            "the description is renamed into place"
        );
    }

    #[tokio::test]
    async fn killing_a_job_stops_its_whole_process_group() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = Jobs::new(dir.path());
        let (go, marker) = (dir.path().join("go"), dir.path().join("late"));
        // Its child touches `late` once the test, after the kill, creates `go`.
        let command = format!(
            "(until [ -e go ] || [ ! -e \"$PWD\" ]; do sleep 0.01; done; touch {}) & wait",
            marker.display()
        );
        jobs.start(
            "a1",
            None,
            &command,
            dir.path(),
            &BTreeMap::new(),
            Mode::default(),
        )
        .await
        .unwrap();
        let killed = jobs.kill("j1").await.unwrap();
        assert_eq!(killed.state, JobState::Killed);
        assert!(killed.reported, "the killer already knows");
        std::fs::write(&go, "").unwrap();
        // A child that survived would touch `late` within its next 10ms poll.
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(!marker.exists(), "the command's children were killed too");
        assert!(jobs.unreported(true).await.is_empty());
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn a_job_runs_in_a_session_of_its_own_without_a_terminal() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = Jobs::new(dir.path());
        let command = "read -r pid comm state ppid group session rest < /proc/$$/stat
            echo \"$group $session\"
            if (: > /dev/tty) 2>/dev/null; then echo tty; else echo no tty; fi";
        let job = jobs
            .start("a1", None, command, dir.path(), &BTreeMap::new(), Mode::default())
            .await
            .unwrap();
        until_ended(&jobs, "j1").await;
        let log = tokio::fs::read_to_string(jobs.log_path("j1")).await.unwrap();
        let leader = job.meta.pid;
        assert_eq!(log, format!("{leader} {leader}\nno tty\n"));
    }

    #[tokio::test]
    async fn a_running_job_whose_process_is_gone_is_reported_lost() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = Jobs::new(dir.path());
        let meta = JobMeta {
            id: "j7".into(),
            owner: "a1".into(),
            name: None,
            command: "make".into(),
            cwd: dir.path().into(),
            pid: u32::MAX / 2,
            started: now(),
            restart: false,
            quiet: false,
            restarts: 0,
            quick: 0,
        };
        tokio::fs::write(dir.path().join("j7.json"), serde_json::to_vec(&meta).unwrap())
            .await
            .unwrap();
        assert!(jobs.unreported(false).await.is_empty(), "running, not checked");
        let lost = jobs.unreported(true).await;
        assert_eq!(lost.len(), 1);
        assert_eq!(lost[0].state, JobState::Lost);
        assert_eq!(lost[0].status(), "ended without an exit code");
    }

    #[tokio::test]
    async fn ended_jobs_not_yet_reported_are_listed_oldest_first() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = Jobs::new(dir.path());
        for n in [2, 10, 1, 3] {
            let id = format!("j{n}");
            let meta = JobMeta {
                id: id.clone(),
                owner: "a1".into(),
                name: None,
                command: "true".into(),
                cwd: dir.path().into(),
                pid: u32::MAX / 2,
                started: now(),
                restart: false,
                quiet: false,
                restarts: 0,
                quick: 0,
            };
            std::fs::write(
                dir.path().join(format!("{id}.json")),
                serde_json::to_vec(&meta).unwrap(),
            )
            .unwrap();
            std::fs::write(dir.path().join(format!("{id}.exit")), "0").unwrap();
        }
        jobs.mark_reported("j2").await.unwrap();
        // A report mark without a description is not a job.
        std::fs::write(dir.path().join("j9.reported"), "").unwrap();
        let ids: Vec<String> = jobs
            .unreported(true)
            .await
            .into_iter()
            .map(|job| job.meta.id)
            .collect();
        assert_eq!(ids, ["j1", "j3", "j10"]);
    }

    #[tokio::test]
    async fn tidying_cuts_a_runaway_log_and_drops_old_reported_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = Jobs::new(dir.path());
        let none = BTreeMap::new();
        for _ in 0..3 {
            let job = jobs
                .start("a1", None, "true", dir.path(), &none, Mode::default())
                .await
                .unwrap();
            until_ended(&jobs, &job.meta.id).await;
            jobs.mark_reported(&job.meta.id).await.unwrap();
        }
        // Writes 3000 bytes, waits to be cut (until the test creates `go`), then appends a last
        // line.
        let command = "head -c 3000 /dev/zero | tr '\\0' x; echo; touch written; \
                       until [ -e go ] || [ ! -e \"$PWD\" ]; do sleep 0.01; done; echo last";
        jobs.start("a1", None, command, dir.path(), &none, Mode::default())
            .await
            .unwrap();
        let written = dir.path().join("written");
        tokio::time::timeout(Duration::from_secs(10), async {
            while !written.exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the job never wrote its output");

        jobs.tidy(1000, 100, 1).await;
        let ids: Vec<String> = jobs.list().await.into_iter().map(|job| job.meta.id).collect();
        assert_eq!(ids, ["j3", "j4"], "only the newest reported job is kept");
        assert!(!dir.path().join("j1.log").exists() && !dir.path().join("j2.exit").exists());
        let log = tokio::fs::read_to_string(jobs.log_path("j4")).await.unwrap();
        assert!(log.starts_with("[strands-swarm: log over 1000 bytes"), "{log}");
        assert!(log.len() < 200, "{}", log.len());

        std::fs::write(dir.path().join("go"), "").unwrap();
        until_ended(&jobs, "j4").await;
        let log = tokio::fs::read_to_string(jobs.log_path("j4")).await.unwrap();
        assert!(
            log.ends_with("x\nlast\n"),
            "appends continue after the cut: {log:?}"
        );
        assert_eq!(jobs.tail("j4", 1).await, "last");
        let next = jobs
            .start("a1", None, "true", dir.path(), &none, Mode::default())
            .await
            .unwrap();
        assert_eq!(next.meta.id, "j5");
    }

    #[test]
    fn restarts_back_off_after_three_quick_exits_in_a_row_up_to_half_an_hour() {
        let secs = |quick| restart_delay(quick).as_secs();
        assert_eq!([secs(0), secs(1), secs(2)], [0, 0, 0]);
        assert_eq!([secs(3), secs(4), secs(5)], [60, 120, 240]);
        assert_eq!([secs(8), secs(9), secs(100)], [1800, 1800, 1800]);
    }

    #[tokio::test]
    async fn a_restarting_job_starts_again_under_its_id_with_a_fresh_log_and_only_once_reported() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = Jobs::new(dir.path().join("jobs"));
        let none = BTreeMap::new();
        jobs.start("a1", None, "echo run; exit 4", dir.path(), &none, Mode::RESTART)
            .await
            .unwrap();
        until_ended(&jobs, "j1").await;
        assert!(
            jobs.restart("j1", &none).await.unwrap().is_none(),
            "its end is not reported yet"
        );

        jobs.mark_reported("j1").await.unwrap();
        let pending = jobs.pending_restarts().await;
        assert_eq!(
            pending,
            vec![("j1".to_owned(), Duration::ZERO)],
            "found again after a restart"
        );
        let again = jobs.restart("j1", &none).await.unwrap().expect("restarted");
        assert_eq!(
            (again.meta.id.as_str(), again.meta.restarts, again.meta.quick),
            ("j1", 1, 1)
        );
        assert!(!again.reported, "its next end is reported anew");
        let ended = until_ended(&jobs, "j1").await;
        assert_eq!(ended.state, JobState::Exited(4));
        assert!(ended.status().ends_with("restarts 1"), "{}", ended.status());
        assert_eq!(jobs.tail("j1", 10).await, "run", "a fresh log, not two runs");
        assert_eq!(jobs.list().await.len(), 1);
    }

    #[tokio::test]
    async fn a_killed_restarting_job_never_starts_again_and_one_that_was_not_asked_to_does_not() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = Jobs::new(dir.path().join("jobs"));
        let none = BTreeMap::new();
        jobs.start("a1", None, "true", dir.path(), &none, Mode::RESTART)
            .await
            .unwrap();
        jobs.start("a1", None, "true", dir.path(), &none, Mode::default())
            .await
            .unwrap();
        let (first, second) = (until_ended(&jobs, "j1").await, until_ended(&jobs, "j2").await);
        assert!(first.restarting() && !second.restarting());
        jobs.mark_reported("j1").await.unwrap();
        jobs.mark_reported("j2").await.unwrap();
        assert!(jobs.restart("j2", &none).await.unwrap().is_none());
        assert_eq!(jobs.pending_restarts().await.len(), 1);

        assert_eq!(
            jobs.kill("j1").await.unwrap().state,
            JobState::Killed,
            "while waiting to restart"
        );
        assert!(jobs.restart("j1", &none).await.unwrap().is_none());
        assert!(jobs.pending_restarts().await.is_empty());
        assert!(!jobs.get("j1").await.unwrap().live());
    }

    /// The contract scripts wait on: `<id>.exit` holds the code once the job ends and stays.
    #[tokio::test]
    async fn the_exit_marker_outlives_the_report_and_a_resume_and_goes_with_the_pruned_job() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = Jobs::new(dir.path().join("jobs"));
        let none = BTreeMap::new();
        for _ in 0..2 {
            jobs.start("a1", None, "exit 3", dir.path(), &none, Mode::default())
                .await
                .unwrap();
        }
        until_ended(&jobs, "j1").await;
        until_ended(&jobs, "j2").await;
        let marker = |id: &str| std::fs::read_to_string(dir.path().join(format!("jobs/{id}.exit")));
        assert_eq!(jobs.unreported(true).await.len(), 2);

        jobs.mark_reported("j1").await.unwrap();
        jobs.mark_reported("j2").await.unwrap();
        jobs.tidy(LOG_MAX, LOG_KEEP, KEEP_JOBS).await;
        assert_eq!(
            (marker("j1").unwrap().trim(), marker("j2").unwrap().trim()),
            ("3", "3")
        );

        // A resumed swarm reads the same files: nothing is delivered again.
        let resumed = Jobs::new(dir.path().join("jobs"));
        assert!(resumed.unreported(true).await.is_empty());
        assert_eq!(resumed.get("j1").await.unwrap().state, JobState::Exited(3));

        // Pruning takes the marker with the rest of the job's files.
        resumed.tidy(LOG_MAX, LOG_KEEP, 1).await;
        assert!(marker("j1").is_err() && !dir.path().join("jobs/j1.json").exists());
        assert_eq!(marker("j2").unwrap().trim(), "3");
    }

    #[tokio::test]
    async fn a_restart_clears_the_previous_runs_exit_marker() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = Jobs::new(dir.path().join("jobs"));
        let none = BTreeMap::new();
        let go = dir.path().join("go");
        let command = format!("if [ -e {} ]; then sleep 30; else exit 4; fi", go.display());
        jobs.start("a1", None, &command, dir.path(), &none, Mode::RESTART)
            .await
            .unwrap();
        until_ended(&jobs, "j1").await;
        jobs.mark_reported("j1").await.unwrap();
        let marker = dir.path().join("jobs/j1.exit");
        assert_eq!(std::fs::read_to_string(&marker).unwrap().trim(), "4");

        std::fs::write(&go, "").unwrap();
        let again = jobs.restart("j1", &none).await.unwrap().expect("restarted");
        assert_eq!(again.state, JobState::Running, "the old marker is not this run's");
        assert!(!marker.exists());
        assert_eq!(jobs.get("j1").await.unwrap().state, JobState::Running);
        jobs.kill("j1").await.unwrap();
    }

    #[tokio::test]
    async fn a_job_waiting_to_restart_is_kept_by_tidy() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = Jobs::new(dir.path().join("jobs"));
        let none = BTreeMap::new();
        jobs.start("a1", None, "true", dir.path(), &none, Mode::RESTART)
            .await
            .unwrap();
        until_ended(&jobs, "j1").await;
        jobs.mark_reported("j1").await.unwrap();
        jobs.tidy(LOG_MAX, LOG_KEEP, 0).await;
        assert_eq!(jobs.list().await.len(), 1);
    }

    #[test]
    fn durations_are_short() {
        assert_eq!(duration(41), "41s");
        assert_eq!(duration(125), "2m05s");
        assert_eq!(duration(3780), "1h03m");
    }
}
