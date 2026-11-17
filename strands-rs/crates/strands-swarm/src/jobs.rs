//! Background shell jobs: commands that run detached from the swarm process, so a restart does
//! not kill them, and report back to the agent that started them when they end.
//!
//! Everything lives on disk under one directory (`<run>/jobs`): `<id>.json` describes the job,
//! `<id>.log` takes its output, a small `sh` wrapper writes the exit code to `<id>.exit`, and
//! `<id>.reported` marks a job whose end was told to its owner. A restarted swarm reads the same
//! files, so jobs started before a reload still report.

use std::collections::BTreeMap;
use std::io::SeekFrom;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use nix::sys::signal::Signal;
use nix::unistd::Pid;
use serde::{Deserialize, Serialize};
use strands_agents::sandbox::detach_from_terminal;
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
    /// Start time, seconds since the Unix epoch.
    pub started: u64,
}

/// Where a job is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobState {
    /// Still running.
    Running,
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
        match self.state {
            JobState::Running => format!("running {time}"),
            JobState::Exited(code) => format!("exited {code} after {time}"),
            JobState::Killed => format!("killed after {time}"),
            JobState::Lost => "ended without an exit code".to_owned(),
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
    ) -> Result<Job> {
        let _guard = self.start.lock().await;
        tokio::fs::create_dir_all(&self.dir).await?;
        let id = self.next_id().await?;
        let mut child = quiet(detach_from_terminal(&mut tokio::process::Command::new("sh")))
            .arg("-c")
            .arg(WRAPPER)
            .arg("job")
            .arg(command)
            .arg(self.log_path(&id))
            .arg(self.file(&id, "exit"))
            .current_dir(cwd)
            .envs(env)
            .spawn()
            .map_err(|e| Error::other(format!("Could not start the job: {e}")))?;
        let meta = JobMeta {
            id: id.clone(),
            owner: owner.to_owned(),
            name: name.filter(|n| !n.trim().is_empty()),
            command: command.to_owned(),
            cwd: cwd.to_path_buf(),
            pid: child.id().unwrap_or_default(),
            started: now(),
        };
        // Reap the wrapper once it ends; the job reports through its files, not this handle.
        tokio::spawn(async move { child.wait().await });
        // A crash mid-write must not leave a half `.json`, which no listing could read.
        let json = serde_json::to_vec_pretty(&meta)?;
        crate::files::replace_async(&self.file(&id, "json"), json, false).await?;
        Ok(Job {
            meta,
            state: JobState::Running,
            reported: false,
            seconds: 0,
        })
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
        let state = match tokio::fs::read_to_string(&exit).await {
            Ok(text) => match text.trim() {
                "killed" => JobState::Killed,
                "lost" => JobState::Lost,
                code => code.parse().map_or(JobState::Lost, JobState::Exited),
            },
            Err(_) => JobState::Running,
        };
        let end = match state {
            JobState::Running => now(),
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
            if job.reported {
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
        let job = self.get(id).await?;
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
        let reported = all.iter().filter(|job| job.reported).count();
        let mut old = reported.saturating_sub(jobs);
        for job in &all {
            let id = &job.meta.id;
            if job.reported && old > 0 {
                old -= 1;
                // The `.json` last: a job is listed while it exists.
                for ext in ["log", "exit", "reported", "json"] {
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

/// Send `signal` to the process group led by `leader`; whether it went to any process.
pub(crate) fn signal_group(leader: u32, signal: Signal) -> bool {
    positive(leader).is_some_and(|group| nix::sys::signal::killpg(group, signal).is_ok())
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
        jobs.start("a1", None, command, dir.path(), &BTreeMap::new())
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

        let next = jobs.start("a2", None, "true", dir.path(), &BTreeMap::new()).await;
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
            .start("a1", None, "true", dir.path(), &BTreeMap::new())
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
        let marker = dir.path().join("late");
        let command = format!("sleep 2 && touch {}", marker.display());
        jobs.start("a1", None, &command, dir.path(), &BTreeMap::new())
            .await
            .unwrap();
        let killed = jobs.kill("j1").await.unwrap();
        assert_eq!(killed.state, JobState::Killed);
        assert!(killed.reported, "the killer already knows");
        tokio::time::sleep(Duration::from_millis(2500)).await;
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
            .start("a1", None, command, dir.path(), &BTreeMap::new())
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
            let job = jobs.start("a1", None, "true", dir.path(), &none).await.unwrap();
            until_ended(&jobs, &job.meta.id).await;
            jobs.mark_reported(&job.meta.id).await.unwrap();
        }
        // Writes 3000 bytes, waits to be cut, then appends a last line.
        let command = "head -c 3000 /dev/zero | tr '\\0' x; echo; sleep 1; echo last";
        jobs.start("a1", None, command, dir.path(), &none).await.unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;

        jobs.tidy(1000, 100, 1).await;
        let ids: Vec<String> = jobs.list().await.into_iter().map(|job| job.meta.id).collect();
        assert_eq!(ids, ["j3", "j4"], "only the newest reported job is kept");
        assert!(!dir.path().join("j1.log").exists() && !dir.path().join("j2.exit").exists());
        let log = tokio::fs::read_to_string(jobs.log_path("j4")).await.unwrap();
        assert!(log.starts_with("[strands-swarm: log over 1000 bytes"), "{log}");
        assert!(log.len() < 200, "{}", log.len());

        until_ended(&jobs, "j4").await;
        let log = tokio::fs::read_to_string(jobs.log_path("j4")).await.unwrap();
        assert!(
            log.ends_with("x\nlast\n"),
            "appends continue after the cut: {log:?}"
        );
        assert_eq!(jobs.tail("j4", 1).await, "last");
        let next = jobs.start("a1", None, "true", dir.path(), &none).await.unwrap();
        assert_eq!(next.meta.id, "j5");
    }

    #[test]
    fn durations_are_short() {
        assert_eq!(duration(41), "41s");
        assert_eq!(duration(125), "2m05s");
        assert_eq!(duration(3780), "1h03m");
    }
}
