//! Background shell jobs: starting, listing and killing them, and reporting ended ones.

use super::*;

impl Swarm {
    /// Kill the running background jobs of these agents, without waiting.
    pub(super) fn kill_jobs_of(&self, owners: Vec<String>) {
        let swarm = self.clone();
        tokio::spawn(async move {
            for job in swarm.jobs().await {
                if job.state == jobs::JobState::Running && owners.contains(&job.meta.owner) {
                    let _ = swarm.kill_job(&job.meta.id).await;
                }
            }
        });
    }

    /// Start `command` as a background job of agent `owner`: in its working directory and
    /// environment, detached from this process so a restart does not stop it. When it ends, the
    /// owner gets a `job` message with its exit code and the tail of its output.
    pub async fn start_job(&self, owner: &str, command: &str, name: Option<String>) -> Result<Job> {
        let node = self
            .node(owner)
            .ok_or_else(|| Error::other(format!("Unknown agent id {owner}.")))?;
        let job = self
            .inner
            .jobs
            .start(owner, name, command, &node.cwd, &node.env)
            .await?;
        if let Some(journal) = &self.inner.journal {
            journal.event(
                "job",
                serde_json::json!({ "id": job.meta.id, "agent": owner, "command": command }),
            );
        }
        self.watch_jobs();
        Ok(job)
    }

    /// Every background job of the run, oldest first.
    pub async fn jobs(&self) -> Vec<Job> {
        self.inner.jobs.list().await
    }

    /// Kill background job `id` (and everything it started); its owner gets no end report.
    pub async fn kill_job(&self, id: &str) -> Result<Job> {
        self.inner.jobs.kill(id).await
    }

    /// Where job `id` writes its output.
    pub fn job_log(&self, id: &str) -> PathBuf {
        self.inner.jobs.log_path(id)
    }

    /// Report ended jobs to their owners until shutdown; started once, on the first job or on
    /// resume (which picks up jobs started before the restart).
    pub(super) fn watch_jobs(&self) {
        if self.inner.watching_jobs.swap(true, Ordering::SeqCst) {
            return;
        }
        let swarm = self.clone();
        tokio::spawn(async move {
            let mut ticks: u64 = 0;
            loop {
                if swarm.inner.shutting_down.load(Ordering::SeqCst) {
                    return;
                }
                let check_alive = ticks % JOB_ALIVE_EVERY == 0;
                for job in swarm.inner.jobs.unreported(check_alive).await {
                    swarm.report_job(&job).await;
                }
                ticks += 1;
                tokio::select! {
                    _ = tokio::time::sleep(JOB_POLL) => {}
                    _ = swarm.inner.root_cancel.cancelled() => return,
                }
            }
        });
    }

    pub(super) async fn report_job(&self, job: &Job) {
        if self.inner.shutting_down.load(Ordering::SeqCst) {
            return;
        }
        let id = &job.meta.id;
        let tail = self.inner.jobs.tail(id, jobs::TAIL_LINES).await;
        let output = match tail.is_empty() {
            true => "(no output)".to_owned(),
            false => format!("last lines:\n{tail}"),
        };
        let text = format!(
            "Job {} {}: `{}`\nlog: {}\n{output}",
            job.label(),
            job.status(),
            job.meta.command,
            self.job_log(id).display()
        );
        // Posted before it is marked: a crash in between repeats the report rather than lose it.
        // An owner that is gone (cancelled) cannot be told; the job counts as reported.
        let _ = self.post(JOBS_SENDER, &job.meta.owner, "job", text);
        if let Some(journal) = &self.inner.journal {
            journal.event("job_end", serde_json::json!({ "id": id, "status": job.status() }));
        }
        if let Err(error) = self.inner.jobs.mark_reported(id).await {
            tracing::warn!(job = %id, %error, "could not mark the job reported");
        }
    }
}

/// How often the job watcher looks for ended jobs.
pub(super) const JOB_POLL: Duration = Duration::from_millis(500);

/// Every this many polls the watcher also checks that running jobs' processes still exist.
pub(super) const JOB_ALIVE_EVERY: u64 = 60;

/// `<run>/jobs`, or for a run kept in memory a directory under the system's temp dir.
pub(super) fn jobs_dir(config: &SwarmConfig) -> PathBuf {
    match &config.state_dir {
        Some(dir) => dir.join("jobs"),
        None => std::env::temp_dir()
            .join("strands-swarm-jobs")
            .join(&config.run_id),
    }
}
