//! Hot reload: installing a new binary and restarting into it now or once tool calls finish.

use super::*;

impl Swarm {
    /// Whether the run is persistent, which [`Swarm::reload`] needs (agents get `reload_swarm`).
    pub fn can_reload(&self) -> bool {
        self.inner.config.state_dir.is_some()
    }

    /// Install `binary` as the run's next binary and ask the host process to restart into it:
    /// right away, or with [`ReloadWhen::Idle`] once no agent has a tool call in flight (at most
    /// [`SwarmConfig::reload_idle_timeout`] later). A newer reload supersedes a pending one.
    /// Meanwhile agents start no new tool calls: their next model call waits for the restart
    /// ([`Swarm::hold_for_reload`]).
    ///
    /// The binary must answer `--version` within 30 s and pass its own `--self-test` on a
    /// temporary copy of the run within [`SELF_TEST_TIMEOUT`] (see [`crate::selftest`]); otherwise
    /// the reload fails and the swarm keeps its binary. It is copied to `<run>/bin/current`, the
    /// running one kept as `<run>/bin/previous` for the supervisor's crash rollback. A newer
    /// reload replaces only `current`: the pending binary never ran, so it is no rollback target.
    /// Returns what the reload does to the agents' prompt caches ([`Swarm::reload_check`]).
    pub async fn reload(&self, binary: &Path, when: ReloadWhen) -> Result<String> {
        let dir = self.inner.config.state_dir.as_ref().ok_or_else(|| {
            Error::other("Hot reload needs a persistent run (start the swarm with --run <name>).")
        })?;
        check_binary(binary).await?;
        selftest::check_binary(binary, dir, SELF_TEST_TIMEOUT).await?;
        let note = self.cache_note(binary).await;
        let mut installed = self.inner.reload_installed.lock().await;
        install_binary(binary, &dir.join("bin"), !*installed).await?;
        *installed = true;
        self.journal_event(
            "reload",
            serde_json::json!({ "binary": binary, "when": format!("{when:?}") }),
        );
        let generation = self.inner.reload_generation.fetch_add(1, Ordering::SeqCst) + 1;
        drop(installed);
        match when {
            ReloadWhen::Now => self.inner.reload.cancel(),
            ReloadWhen::Idle => self.reload_when_idle(generation),
        }
        Ok(note)
    }

    /// What reloading into `binary` does to the agents' prompt caches, without reloading: whether
    /// its fixed prefix ([`crate::prefix`]) differs from the running binary's and, if so, what the
    /// agents with a warm cache would pay to rewrite theirs.
    pub async fn reload_check(&self, binary: &Path) -> Result<String> {
        check_binary(binary).await?;
        let note = self.cache_note(binary).await;
        Ok(match note.is_empty() {
            true => format!(
                "{} prints no fixed prefix to compare (it predates --print-prefix).",
                binary.display()
            ),
            false => note,
        })
    }

    /// [`Swarm::reload_check`]'s note; empty when either binary cannot print its prefix.
    async fn cache_note(&self, binary: &Path) -> String {
        // The running binary, even after a newer one was installed over its path.
        let proc = Path::new("/proc/self/exe");
        let running = match (&self.inner.config.exe, proc.exists()) {
            (Some(exe), _) => exe.clone(),
            (None, true) => proc.to_path_buf(),
            (None, false) => match std::env::current_exe() {
                Ok(path) => path,
                Err(_) => return String::new(),
            },
        };
        let allow_land = self.allows_land();
        let (old, new) = tokio::join!(prefix_of(&running, allow_land), prefix_of(binary, allow_land));
        let (Some(old), Some(new)) = (old, new) else {
            return String::new();
        };
        let changes = old.changes(&new);
        if changes.is_empty() {
            return "Its fixed prefix (system prompt and tool specs) is the running binary's: prompt \
                    caches survive the restart."
                .to_owned();
        }
        let (agents, tokens, usd) = self.warm_caches();
        let agents = match agents {
            1 => "1 agent has".to_owned(),
            n => format!("{n} agents have"),
        };
        let shown = changes.iter().take(6).cloned().collect::<Vec<_>>().join(", ");
        let more = match changes.len().saturating_sub(6) {
            0 => String::new(),
            n => format!(" and {n} more"),
        };
        format!(
            "It changes the fixed prefix every call starts with ({shown}{more}), so each agent's next \
             call rewrites its whole prompt cache instead of reading it. {agents} a warm cache now: {} tokens, est. {} more than a reload that keeps the prefix. Prefix changes \
             cost less batched into one reload.",
            crate::text::compact(tokens),
            crate::cost::dollars(usd)
        )
    }

    /// Agents whose prompt cache is warm (their last call or keep-warm refresh within
    /// [`crate::pruning::CACHE_TTL`]), the tokens of their last calls, and the estimated extra USD
    /// of writing those instead of reading them.
    pub fn warm_caches(&self) -> (usize, u64, f64) {
        let prices = &self.inner.config.prices;
        let default = self.inner.config.default_model.clone();
        let ttl = crate::pruning::CACHE_TTL;
        let now = self.now().timestamp();
        let mut warm = (0, 0, 0.0);
        for node in self.nodes() {
            let Some(last) = node.ledger.lock().recent.back().copied() else {
                continue;
            };
            let refreshed = node
                .agent()
                .and_then(|agent| crate::pruning::last_call(&agent))
                .and_then(|at| at.elapsed().ok())
                .is_some_and(|age| age < ttl);
            if !refreshed && now - last.at >= ttl.as_secs() as i64 {
                continue;
            }
            let tokens = last.input;
            let model = node.model_id().or_else(|| default.clone()).unwrap_or_default();
            let extra = prices.price(&model).map_or(0.0, |price| {
                let counts = |read, write| crate::cost::Counts {
                    input: tokens,
                    cache_read: read,
                    cache_write: write,
                    ..Default::default()
                };
                price.cost(&counts(0, tokens)) - price.cost(&counts(tokens, 0))
            });
            warm = (warm.0 + 1, warm.1 + tokens, warm.2 + extra);
        }
        warm
    }

    /// Signal the reload once nothing is in flight, unless a newer reload took over first.
    pub(super) fn reload_when_idle(&self, generation: u64) {
        self.inner.reload_held.store(true, Ordering::SeqCst);
        let swarm = self.clone();
        let deadline = Instant::now() + self.inner.config.reload_idle_timeout;
        tokio::spawn(async move {
            let mut shown = None;
            loop {
                let current = swarm.inner.reload_generation.load(Ordering::SeqCst) == generation;
                if !current || swarm.inner.shutting_down.load(Ordering::SeqCst) {
                    return;
                }
                let busy = swarm.tool_calls_in_flight();
                if busy == 0 || Instant::now() >= deadline {
                    break;
                }
                if shown != Some(busy) {
                    swarm.emit(SwarmEvent::ReloadPending { tool_calls: busy });
                    shown = Some(busy);
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            swarm.inner.reload.cancel();
        });
    }

    /// While an idle-mode reload waits, hold an agent's model call until the swarm shuts down for
    /// it (or `cancel` fires), then fail it as cancelled. Otherwise each finished tool call is
    /// followed by a model call that starts the next, and a busy swarm never gets idle. The tool
    /// results are saved, so the resumed agent makes the held call; nothing is cut. Keep-warm
    /// refreshes are not held: they keep the cache that call reads warm through the restart.
    pub(super) async fn hold_for_reload(&self, id: &str, cancel: &CancellationToken) -> Result<()> {
        if !self.inner.reload_held.load(Ordering::SeqCst) {
            return Ok(());
        }
        let held = |held| SwarmEvent::HeldForReload {
            id: id.to_owned(),
            held,
        };
        self.emit(held(true));
        tokio::select! {
            _ = cancel.cancelled() => {}
            _ = self.inner.root_cancel.cancelled() => {}
        }
        self.emit(held(false));
        Err(Error::Cancelled)
    }

    /// Tool calls in flight across running agents, not counting tools that only wait (`wait`).
    pub fn tool_calls_in_flight(&self) -> usize {
        self.nodes()
            .iter()
            .filter(|node| *node.status.lock() == AgentStatus::Running)
            .map(|node| node.tool_calls.lock().len())
            .sum()
    }

    /// Take what the supervisor left in [`SUPERVISOR_NOTICE`] (a rollback, a crash loop) and mail
    /// it to every live root agent, or to the user when there is none, so that whoever reloaded
    /// learns the new binary did not stick.
    pub(super) async fn deliver_supervisor_notice(&self) {
        let Some(dir) = &self.inner.config.state_dir else {
            return;
        };
        let path = dir.join(SUPERVISOR_NOTICE);
        let Ok(text) = tokio::fs::read_to_string(&path).await else {
            return;
        };
        if let Err(error) = tokio::fs::remove_file(&path).await {
            // Delivered again next start rather than lost.
            tracing::warn!(%error, "supervisor notice not removed");
        }
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        let start = text.len().saturating_sub(SUPERVISOR_NOTICE_MAX);
        let start = (start..text.len())
            .find(|&i| text.is_char_boundary(i))
            .unwrap_or(0);
        let text = format!("[From the swarm's supervisor (swarmctl)]\n{}", &text[start..]);
        self.tell_roots(SUPERVISOR_SENDER, "supervisor", text);
    }

    /// Resolves once an agent has requested a reload; the host should then call
    /// [`Swarm::shutdown`] and exit with [`RELOAD_EXIT_CODE`].
    pub async fn reload_requested(&self) {
        self.inner.reload.cancelled().await;
    }
}

pub(super) async fn check_binary(binary: &Path) -> Result<()> {
    let run = tokio::process::Command::new(binary).arg("--version").output();
    let output = tokio::time::timeout(Duration::from_secs(30), run)
        .await
        .map_err(|_| Error::other(format!("{} --version timed out.", binary.display())))?
        .map_err(|e| Error::other(format!("Cannot run {}: {e}", binary.display())))?;
    if !output.status.success() {
        return Err(Error::other(format!(
            "{} --version failed: {}",
            binary.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(())
}

/// Copy `binary` to `<bin_dir>/current`, first moving the old one to `previous` if `keep_previous`.
pub(super) async fn install_binary(binary: &Path, bin_dir: &Path, keep_previous: bool) -> Result<()> {
    let io = |e: std::io::Error| Error::other(format!("Failed to install the new binary: {e}"));
    tokio::fs::create_dir_all(bin_dir).await.map_err(io)?;
    let current = bin_dir.join("current");
    let staged = bin_dir.join("current.new");
    tokio::fs::copy(binary, &staged).await.map_err(io)?;
    if keep_previous && tokio::fs::try_exists(&current).await.map_err(io)? {
        tokio::fs::rename(&current, bin_dir.join("previous"))
            .await
            .map_err(io)?;
    }
    tokio::fs::rename(&staged, &current).await.map_err(io)
}

/// `binary --print-prefix` ([`crate::prefix::canonical`]); `None` when it cannot say (a binary
/// from before the flag, a failure, a minute without an answer).
async fn prefix_of(binary: &Path, allow_land: bool) -> Option<crate::prefix::Prefix> {
    let mut command = tokio::process::Command::new(binary);
    command.arg("--print-prefix");
    if allow_land {
        command.arg("--allow-land");
    }
    command
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(60), command.output())
        .await
        .ok()?
        .ok()?;
    if !output.status.success() {
        return None;
    }
    serde_json::from_slice(&output.stdout).ok()
}
