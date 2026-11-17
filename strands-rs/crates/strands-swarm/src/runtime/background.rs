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
    /// owner gets a `job` message with its exit code and the tail of its output. Waits first
    /// while the host is overloaded ([`Swarm::host_busy`]).
    pub async fn start_job(&self, owner: &str, command: &str, name: Option<String>) -> Result<Job> {
        let node = self.known(owner)?;
        self.wait_for_host(owner, false, &node.turn_token()).await?;
        let job = self
            .inner
            .jobs
            .start(
                owner,
                name,
                command,
                &node.workdir.get(),
                &self.with_build_jobs(node.env.clone()),
            )
            .await?;
        self.journal_event(
            "job",
            serde_json::json!({ "id": job.meta.id, "agent": owner, "command": command }),
        );
        self.watch_jobs();
        Ok(job)
    }

    /// Every background job of the run, oldest first.
    pub async fn jobs(&self) -> Vec<Job> {
        // A job whose report is posted counts as reported while the mark waits for the save. The
        // set is read before the listing too: a mark that lands after the listing read the job's
        // files would otherwise be missed by both, and its ended job look unreported.
        let before = self.inner.reports_posted.lock().clone();
        let mut jobs = self.inner.jobs.list().await;
        let posted = self.inner.reports_posted.lock();
        for job in &mut jobs {
            job.reported |= before.contains(&job.meta.id) || posted.contains(&job.meta.id);
        }
        jobs
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
                swarm.note_stalled_jobs().await;
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

    /// Tell the owner, once per job and without waking it ([`FYI`]), that a running job's log has
    /// not grown for [`SwarmConfig::job_quiet`]. The job is left alone.
    async fn note_stalled_jobs(&self) {
        let quiet = self.inner.config.job_quiet;
        for job in self.inner.jobs.stalled(quiet).await {
            let text = format!(
                "Job {} has written nothing for {}; still running: `jobs` to check or kill it.",
                job.label(),
                schedules::human(quiet)
            );
            let _ = self.post(JOBS_SENDER, &job.meta.owner, FYI, text);
            if let Err(error) = self.inner.jobs.mark_stalled(&job.meta.id).await {
                tracing::warn!(job = %job.meta.id, %error, "could not mark the job stalled");
            }
        }
    }

    pub(super) async fn report_job(&self, job: &Job) {
        if self.inner.shutting_down.load(Ordering::SeqCst) {
            return;
        }
        let id = &job.meta.id;
        let tail = self.inner.jobs.tail(id, jobs::TAIL_LINES).await;
        let output = match tail.is_empty() {
            true => empty_output(&job.meta.command),
            false => format!("last lines:\n{tail}"),
        };
        let text = format!(
            "Job {} {}: `{}`\nlog: {}\n{output}",
            job.label(),
            job.status(),
            job.meta.command,
            self.job_log(id).display()
        );
        // Posted and saved before it is marked: a crash in between repeats the report rather than
        // lose it. An owner that is gone (cancelled) cannot be told; the job counts as reported.
        self.inner.reports_posted.lock().insert(id.clone());
        let _ = self.post(JOBS_SENDER, &job.meta.owner, "job", text);
        self.journal_event("job_end", serde_json::json!({ "id": id, "status": job.status() }));
        // `post` saves the mailbox in the background; marked first, a crash before that save
        // would lose the report.
        self.persist_now().await;
        if let Err(error) = self.inner.jobs.mark_reported(id).await {
            tracing::warn!(job = %id, %error, "could not mark the job reported");
        }
        self.inner.reports_posted.lock().remove(id);
    }
}

/// How often the job watcher looks for ended jobs.
pub(super) const JOB_POLL: Duration = Duration::from_millis(500);

/// The report's output line for a job whose log is empty: a command that redirected its output
/// is told so, with the files it names, so its agent knows where to look.
fn empty_output(command: &str) -> String {
    let targets = redirect_targets(command);
    let files: Vec<&str> = targets
        .iter()
        .map(String::as_str)
        .filter(|target| *target != "/dev/null")
        .collect();
    match (files.as_slice(), targets.is_empty()) {
        (_, true) => "(no output)".to_owned(),
        ([], false) => "(no output: the command redirected it to /dev/null)".to_owned(),
        (files, false) => format!(
            "(no output in the log: the command redirected it to {}; read that instead)",
            files
                .iter()
                .map(|file| format!("`{file}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// The files a shell command redirects output to (`> f`, `>> f`, `2> f`, `&> f`, `>|f`), in
/// order, without duplicates and quotes. Descriptor copies (`2>&1`) and quoted `>` are skipped.
fn redirect_targets(command: &str) -> Vec<String> {
    let chars: Vec<char> = command.chars().collect();
    let mut targets = Vec::<String>::new();
    let mut quote = None;
    let mut at = 0;
    while at < chars.len() {
        let c = chars[at];
        at += 1;
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some('"'), '\\') => at += 1,
            (Some(_), _) => {}
            (None, '\\') => at += 1,
            (None, '\'' | '"') => quote = Some(c),
            (None, '>') => {
                if matches!(chars.get(at), Some('>' | '|')) {
                    at += 1;
                }
                if chars.get(at) == Some(&'&') {
                    // `>&2`, `>&-`: a descriptor copy, not a file.
                    continue;
                }
                while chars.get(at).is_some_and(|c| c.is_whitespace()) {
                    at += 1;
                }
                let mut word = String::new();
                let mut inner = None;
                while let Some(&c) = chars.get(at) {
                    match (inner, c) {
                        (Some(q), c) if c == q => inner = None,
                        (Some(_), c) => word.push(c),
                        (None, '\'' | '"') => inner = Some(c),
                        (None, c) if c.is_whitespace() || ";&|<>()".contains(c) => break,
                        (None, c) => word.push(c),
                    }
                    at += 1;
                }
                if !word.is_empty() && !targets.contains(&word) {
                    targets.push(word);
                }
            }
            (None, _) => {}
        }
    }
    targets
}

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
    fn redirect_targets_are_the_files_a_command_writes_to() {
        let cases: [(&str, &[&str]); 8] = [
            ("cargo test > test.log 2>&1", &["test.log"]),
            (
                "make >>build.log 2> 'err file.log'",
                &["build.log", "err file.log"],
            ),
            ("cmd &> out.log; tail out.log >| out.log", &["out.log"]),
            ("cd x && ./run.sh >\"$LOG\" 2>&1 &", &["$LOG"]),
            ("echo '>' \"a > b\" \\> c 1>&2", &[]),
            ("quiet >/dev/null 2>&1", &["/dev/null"]),
            ("exec >(tee log)", &[]),
            ("ls", &[]),
        ];
        for (command, expected) in cases {
            assert_eq!(redirect_targets(command), expected, "{command}");
        }
    }

    /// A job whose log is empty because it wrote elsewhere says where, instead of "(no output)".
    #[test]
    fn an_empty_log_names_the_files_the_command_redirected_to() {
        assert_eq!(
            empty_output("cargo test -p x > /tmp/t.log 2>&1"),
            "(no output in the log: the command redirected it to `/tmp/t.log`; read that instead)"
        );
        assert_eq!(
            empty_output("quiet > /dev/null"),
            "(no output: the command redirected it to /dev/null)"
        );
        assert_eq!(empty_output("true"), "(no output)");
    }

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

    /// A report's mark that lands while `jobs` reads the other jobs' files, after it read this
    /// one's: the job still counts as reported, or a `wait` from jobs would wait for mail that
    /// already came. The listing's progress cannot be observed, so the mark lands at a range of
    /// delays; one of them hits that window.
    #[tokio::test]
    async fn a_report_marked_while_the_jobs_are_listed_still_counts() {
        let dir = tempfile::tempdir().unwrap();
        let factory: AgentFactory = Arc::new(|_| Box::pin(async { Err(Error::other("no agents")) }));
        let mut config = SwarmConfig::new(factory, dir.path());
        config.state_dir = Some(dir.path().join("run"));
        let (swarm, _events) = Swarm::new(config);
        let files = dir.path().join("run/jobs");
        std::fs::create_dir_all(&files).unwrap();
        for n in 1..=100 {
            let meta = jobs::JobMeta {
                id: format!("j{n}"),
                owner: "a1".into(),
                name: None,
                command: "true".into(),
                cwd: dir.path().into(),
                pid: 0,
                started: n,
            };
            std::fs::write(
                files.join(format!("j{n}.json")),
                serde_json::to_vec(&meta).unwrap(),
            )
            .unwrap();
            std::fs::write(files.join(format!("j{n}.exit")), "0").unwrap();
        }
        for delay in 0..30 {
            let _ = std::fs::remove_file(files.join("j1.reported"));
            swarm.inner.reports_posted.lock().insert("j1".into());
            let listing = tokio::spawn({
                let swarm = swarm.clone();
                async move { swarm.jobs().await }
            });
            tokio::time::sleep(Duration::from_micros(500 * delay)).await;
            // What report_job does once the report is saved.
            swarm.inner.jobs.mark_reported("j1").await.unwrap();
            swarm.inner.reports_posted.lock().remove("j1");
            let jobs = listing.await.unwrap();
            let j1 = jobs.iter().find(|job| job.meta.id == "j1").unwrap();
            assert!(j1.reported, "j1 looked unreported with the mark {delay}/2 ms in");
        }
    }
}
