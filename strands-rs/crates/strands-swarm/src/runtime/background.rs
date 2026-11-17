//! Background shell jobs: starting, listing and killing them, and reporting ended ones.

use super::*;

/// A `shell` command sleeping at least this long while its agent has a job running is taken
/// for polling the job, and its result says how to wait instead.
pub const POLL_SLEEP_SECONDS: u64 = 30;

impl Swarm {
    /// Append [`poll_hint`] to a `shell` result that slept [`POLL_SLEEP_SECONDS`] or more while
    /// the agent had a background job running (`sleep 240; tail job.log`). The fact sits where
    /// the habit shows, rather than as one more prompt rule.
    pub(super) fn install_poll_hint(&self, agent: &Agent, id: &str) {
        let swarm = self.clone();
        let id = id.to_owned();
        agent
            .hooks()
            .add_callback_boxed(move |event: &mut AfterToolCallEvent| {
                let (swarm, id) = (swarm.clone(), id.clone());
                Box::pin(async move {
                    let command = event.tool_use.input.get("command").and_then(|c| c.as_str());
                    let polls = event.tool_use.name == "shell"
                        && command
                            .and_then(longest_sleep)
                            .is_some_and(|s| s >= POLL_SLEEP_SECONDS);
                    if !polls {
                        return Ok(());
                    }
                    let running: Vec<String> = swarm
                        .jobs()
                        .await
                        .into_iter()
                        .filter(|job| job.state == jobs::JobState::Running && job.meta.owner == id)
                        .map(|job| job.meta.id)
                        .collect();
                    if !running.is_empty() {
                        let hint = ToolResultContent::Text(poll_hint(&running));
                        event.result.content.push(hint);
                    }
                    Ok(())
                })
            });
    }

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
                if check_alive {
                    let jobs = &swarm.inner.jobs;
                    jobs.tidy(jobs::LOG_MAX, jobs::LOG_KEEP, jobs::KEEP_JOBS).await;
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

/// Seconds of the longest `sleep N` in a shell command (`N`, `Ns`, `Nm` or `Nh`), if any.
pub fn longest_sleep(command: &str) -> Option<u64> {
    let words: Vec<&str> = command
        .split(|c: char| c.is_whitespace() || ";&|()".contains(c))
        .filter(|word| !word.is_empty())
        .collect();
    let seconds = |arg: &str| {
        let (digits, unit) = arg.split_at(arg.find(|c: char| !c.is_ascii_digit()).unwrap_or(arg.len()));
        let scale = match unit {
            "" | "s" => 1,
            "m" => 60,
            "h" => 3600,
            _ => return None,
        };
        digits.parse::<u64>().ok().map(|n| n * scale)
    };
    words
        .windows(2)
        .filter(|pair| pair[0] == "sleep")
        .filter_map(|pair| seconds(pair[1]))
        .max()
}

/// What a polling `shell` result is told: the agent's running jobs and how to wait for them.
pub fn poll_hint(running: &[String]) -> String {
    let jobs = running.join(", ");
    let verb = if running.len() == 1 { "is" } else { "are" };
    format!(
        "\n\n[Job {jobs} {verb} still running. `wait` with from: [\"jobs\"] returns as soon as a job \
         ends; a sleep can't be interrupted by mail or a reload.]"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_longest_sleep_of_a_command_is_found() {
        assert_eq!(
            longest_sleep("sleep 240; cat /tmp/qw-m3d.log | tail -30"),
            Some(240)
        );
        assert_eq!(longest_sleep("cd x && sleep 4m && tail log"), Some(240));
        assert_eq!(longest_sleep("(sleep 5; sleep 1h)"), Some(3600));
        assert_eq!(longest_sleep("for i in 1 2; do sleep 10; done"), Some(10));
        assert_eq!(longest_sleep("sleep $N; grep sleep log"), None);
        assert_eq!(longest_sleep("cargo test sleep_tests"), None);
    }

    #[test]
    fn the_hint_names_the_running_jobs() {
        let one = poll_hint(&["j109".into()]);
        assert!(one.contains("[Job j109 is still running."), "{one}");
        assert!(one.contains("from: [\"jobs\"]"), "{one}");
        assert!(poll_hint(&["j1".into(), "j2".into()]).contains("Job j1, j2 are still"));
    }
}
