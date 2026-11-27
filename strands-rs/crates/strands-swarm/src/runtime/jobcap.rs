//! A cap on the background jobs one root subtree runs at once (`--max-jobs`, `/max-jobs`).
//!
//! Host load protection ([`super::host`]) is all or nothing: past the limit every agent's model
//! calls wait, the unrelated ones too. This bounds the one source instead: a job started while
//! its root's subtree already runs `N` counted jobs is only described on disk
//! ([`crate::jobs::JobState::Queued`], so a restart keeps it queued) and started by the job
//! watcher, oldest first, once the subtree is below `N`.
//!
//! A job counts while it runs, unless it is *light*: it has been running a full
//! [`SwarmConfig::job_light_window`] and used at most 5% of one core over it, measured from
//! `/proc` for the job's whole session (so a `sleep 600; curl …` watcher, or a build blocked on
//! a lock, stops taking a slot while a compile does not). Waiting out a restart back-off is not
//! running. Priority-lane agents' jobs never wait, but still count for the others.

use super::*;
use std::collections::{HashMap, VecDeque};
use tokio::time::Instant;

/// How long a job is watched before it can be called light ([`SwarmConfig::job_light_window`]).
pub const JOB_LIGHT_WINDOW: Duration = Duration::from_secs(60);

/// The share of one core a job may use over the window and still be light.
const LIGHT_SHARE: f64 = 0.05;

/// CPU seconds used so far by everything in each of the given sessions (a job's session id is
/// its process id); sessions it cannot read are left out.
pub type JobCpuProbe = Arc<dyn Fn(&[u32]) -> HashMap<u32, f64> + Send + Sync>;

/// [`JobCpuProbe`] reading `/proc/<pid>/stat` (user, system and reaped children's time, at the
/// usual 100 ticks a second); empty where there is no `/proc`.
pub fn proc_job_cpu(sessions: &[u32]) -> HashMap<u32, f64> {
    let mut used = HashMap::new();
    let Ok(procs) = std::fs::read_dir("/proc") else {
        return used;
    };
    for entry in procs.flatten() {
        if !entry
            .file_name()
            .as_encoded_bytes()
            .iter()
            .all(u8::is_ascii_digit)
        {
            continue;
        }
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        if let Some((session, secs)) = parse_stat(&stat).filter(|(session, _)| sessions.contains(session)) {
            *used.entry(session).or_default() += secs;
        }
    }
    used
}

/// The session id and CPU seconds in a `/proc/<pid>/stat` line.
fn parse_stat(line: &str) -> Option<(u32, f64)> {
    // The command in parentheses may hold spaces; the fields after it start with the state.
    let fields: Vec<&str> = line.rsplit_once(')')?.1.split_whitespace().collect();
    let session = fields.get(3)?.parse().ok()?;
    let mut ticks = 0u64;
    for at in 11..=14 {
        ticks += fields.get(at)?.parse::<u64>().ok()?;
    }
    Some((session, ticks as f64 / 100.0))
}

/// CPU samples per running job, as many as the window needs.
#[derive(Default)]
pub(super) struct CpuTracker {
    samples: HashMap<String, VecDeque<(Instant, f64)>>,
    last: Option<Instant>,
}

impl CpuTracker {
    fn record(&mut self, id: &str, at: Instant, cpu: f64, window: Duration) {
        let samples = self.samples.entry(id.to_owned()).or_default();
        samples.push_back((at, cpu));
        // One sample at least a window old is the baseline; older ones are of no use.
        while samples.len() > 1 && at.duration_since(samples[1].0) >= window {
            samples.pop_front();
        }
    }

    /// Whether `id` used little CPU over a full window; a job not watched that long is not.
    fn is_light(&self, id: &str, window: Duration) -> bool {
        let Some(samples) = self.samples.get(id) else {
            return false;
        };
        let (Some(&(then, before)), Some(&(now, after))) = (samples.front(), samples.back()) else {
            return false;
        };
        let elapsed = now.duration_since(then);
        elapsed >= window && after - before <= LIGHT_SHARE * elapsed.as_secs_f64()
    }
}

/// The cap, who is waiting and the CPU history.
#[derive(Default)]
pub(super) struct JobGate {
    /// Set with [`Swarm::set_max_jobs`]; overrides the config.
    pub(super) max_jobs: Mutex<Option<usize>>,
    /// Held while a job is admitted or queued jobs are started: one decision at a time.
    gate: tokio::sync::Mutex<()>,
    cpu: Mutex<CpuTracker>,
    /// Running and queued jobs as the watcher last counted them, for `/status`.
    seen: Mutex<(usize, usize)>,
}

impl Swarm {
    /// The jobs a root subtree may run at once (`0`: no cap): the last [`Swarm::set_max_jobs`],
    /// else the config's.
    pub fn max_jobs(&self) -> usize {
        let set = *self.inner.jobcap.max_jobs.lock();
        set.unwrap_or(self.inner.config.max_jobs)
    }

    /// Change the cap for this run (kept across restarts, ahead of `--max-jobs`); `Some(0)`
    /// lifts it and `None` returns to the configured one. Queued jobs start as it allows.
    pub fn set_max_jobs(&self, limit: Option<usize>) {
        *self.inner.jobcap.max_jobs.lock() = limit;
        self.persist();
        self.journal_event("max_jobs", serde_json::json!({ "limit": self.max_jobs() }));
    }

    /// Running and queued jobs and the cap, for `/status`; `None` without a cap.
    pub(super) fn job_totals(&self) -> Option<(usize, usize, usize)> {
        let cap = self.max_jobs();
        let (running, queued) = *self.inner.jobcap.seen.lock();
        (cap > 0).then_some((running, queued, cap))
    }

    /// The top-level agent whose subtree `owner`'s jobs belong to; `None` for an agent that is
    /// gone.
    fn job_root(&self, owner: &str) -> Option<String> {
        let node = self.node(owner)?;
        Some(self.root_of(&node).id.clone())
    }

    /// Jobs counted against each root's cap, and queued, from `jobs`.
    fn job_counts(&self, jobs: &[Job]) -> (HashMap<String, usize>, HashMap<String, usize>) {
        let window = self.inner.config.job_light_window;
        let tracker = self.inner.jobcap.cpu.lock();
        let (mut running, mut queued) = (HashMap::new(), HashMap::new());
        for job in jobs {
            let counts = match job.state {
                jobs::JobState::Running if !tracker.is_light(&job.meta.id, window) => &mut running,
                jobs::JobState::Queued => &mut queued,
                _ => continue,
            };
            let root = self
                .job_root(&job.meta.owner)
                .unwrap_or_else(|| job.meta.owner.clone());
            *counts.entry(root).or_default() += 1;
        }
        (running, queued)
    }

    /// How many counted jobs `owner`'s subtree runs when a new job of `owner` must wait for a
    /// slot; `None` when it need not.
    pub(super) async fn job_over_cap(&self, owner: &str) -> Option<usize> {
        let cap = self.max_jobs();
        let root = self
            .job_root(owner)
            .filter(|_| cap > 0 && !self.is_priority(owner))?;
        let (running, queued) = self.job_counts(&self.inner.jobs.list().await);
        let running = running.get(&root).copied().unwrap_or(0);
        // Behind the jobs already waiting too: oldest first.
        (running >= cap || queued.contains_key(&root)).then_some(running)
    }

    /// Hold the admission gate while deciding about, and starting or describing, one job.
    pub(super) async fn job_gate(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.inner.jobcap.gate.lock().await
    }

    /// What the caller of `run_background` is told about a job that waits.
    pub async fn job_queue_note(&self, owner: &str) -> String {
        let (running, _) = self.job_counts(&self.inner.jobs.list().await);
        let root = self.job_root(owner).unwrap_or_default();
        let running = running.get(&root).copied().unwrap_or(0);
        format!("queued: subtree has {running} jobs running (--max-jobs)")
    }

    /// One pass of the job watcher: sample the running jobs' CPU and start the queued jobs the
    /// cap allows, oldest first. Skipped while there is no cap and nothing waits, apart from
    /// the `first` pass after a restart, which finds jobs queued under an earlier cap.
    pub(super) async fn tick_job_cap(&self, first: bool) {
        let cap = self.max_jobs();
        if !(first || cap > 0 || self.inner.jobcap.seen.lock().1 > 0) {
            return;
        }
        let _gate = self.job_gate().await;
        let jobs = self.inner.jobs.list().await;
        if cap > 0 {
            self.sample_job_cpu(&jobs).await;
        }
        let (mut running, _) = self.job_counts(&jobs);
        let mut started = 0;
        let mut queued = 0;
        for job in jobs.iter().filter(|job| job.state == jobs::JobState::Queued) {
            let id = &job.meta.id;
            let owner = &job.meta.owner;
            let Some(node) = self.node(owner).filter(|node| node.status.lock().is_live()) else {
                // Nobody to tell: its owner was cancelled while it waited.
                let _ = self.inner.jobs.kill(id).await;
                continue;
            };
            let root = self.job_root(owner).unwrap_or_default();
            let free = cap == 0 || self.is_priority(owner) || running.get(&root).copied().unwrap_or(0) < cap;
            if !free {
                queued += 1;
                continue;
            }
            let env = self.with_build_jobs(node.env.clone());
            match self.inner.jobs.launch(id, &env).await {
                Ok(_) => {
                    *running.entry(root).or_default() += 1;
                    started += 1;
                    self.journal_event(
                        "job",
                        serde_json::json!({ "id": id, "agent": owner, "queued": true }),
                    );
                }
                Err(error) => {
                    tracing::warn!(job = %id, %error, "could not start the queued job");
                    queued += 1;
                }
            }
        }
        let total = jobs
            .iter()
            .filter(|job| job.state == jobs::JobState::Running)
            .count();
        *self.inner.jobcap.seen.lock() = (total + started, queued);
    }

    /// Add a CPU sample for each running job, at most one every twelfth of the window.
    async fn sample_job_cpu(&self, jobs: &[Job]) {
        let window = self.inner.config.job_light_window;
        let now = Instant::now();
        let running: Vec<(String, u32)> = jobs
            .iter()
            .filter(|job| job.state == jobs::JobState::Running)
            .map(|job| (job.meta.id.clone(), job.meta.pid))
            .collect();
        {
            let mut tracker = self.inner.jobcap.cpu.lock();
            if tracker
                .last
                .is_some_and(|last| now.duration_since(last) < window / 12)
            {
                return;
            }
            tracker.last = Some(now);
            tracker
                .samples
                .retain(|id, _| running.iter().any(|(running, _)| running == id));
        }
        let probe = self.inner.config.job_cpu_probe.clone();
        let sessions: Vec<u32> = running.iter().map(|(_, pid)| *pid).collect();
        let Ok(used) = tokio::task::spawn_blocking(move || probe(&sessions)).await else {
            return;
        };
        let mut tracker = self.inner.jobcap.cpu.lock();
        for (id, pid) in &running {
            if let Some(cpu) = used.get(pid) {
                tracker.record(id, now, *cpu, window);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stat_line_gives_the_session_and_all_cpu_time() {
        let line = "123 (my (cmd) x) S 1 123 456 0 -1 4194560 100 0 0 0 250 50 30 20 20 0 1 0 1000 0 0";
        assert_eq!(parse_stat(line), Some((456, 3.5)));
        assert_eq!(parse_stat("garbage"), None);
        let own = std::fs::read_to_string("/proc/self/stat").unwrap();
        assert!(parse_stat(&own).is_some());
    }

    #[test]
    fn a_job_is_light_only_after_a_full_window_of_little_cpu() {
        let window = Duration::from_secs(60);
        let start = Instant::now();
        let at = |secs| start + Duration::from_secs(secs);
        let mut tracker = CpuTracker::default();
        for (secs, cpu) in [(0, 0.0), (30, 0.5), (60, 1.0), (90, 1.5)] {
            tracker.record("j1", at(secs), cpu, window);
            // Busy: 30 CPU seconds every 30.
            tracker.record("j2", at(secs), cpu * 60.0, window);
        }
        assert!(tracker.is_light("j1", window), "0.5 CPU s a minute");
        assert!(!tracker.is_light("j2", window));
        assert!(!tracker.is_light("j3", window), "unknown");
        assert_eq!(tracker.samples["j1"].len(), 3, "older samples are dropped");
        let mut young = CpuTracker::default();
        young.record("j1", at(0), 0.0, window);
        young.record("j1", at(30), 0.0, window);
        assert!(!young.is_light("j1", window), "not watched a whole window yet");
    }
}
