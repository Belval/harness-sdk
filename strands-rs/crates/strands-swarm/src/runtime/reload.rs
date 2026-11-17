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
    /// old one kept as `<run>/bin/previous` for the supervisor's crash rollback.
    pub async fn reload(&self, binary: &Path, when: ReloadWhen) -> Result<()> {
        let dir = self.inner.config.state_dir.as_ref().ok_or_else(|| {
            Error::other("Hot reload needs a persistent run (start the swarm with --run <name>).")
        })?;
        check_binary(binary).await?;
        selftest::check_binary(binary, dir, SELF_TEST_TIMEOUT).await?;
        install_binary(binary, &dir.join("bin")).await?;
        self.journal_event(
            "reload",
            serde_json::json!({ "binary": binary, "when": format!("{when:?}") }),
        );
        let generation = self.inner.reload_generation.fetch_add(1, Ordering::SeqCst) + 1;
        match when {
            ReloadWhen::Now => self.inner.reload.cancel(),
            ReloadWhen::Idle => self.reload_when_idle(generation),
        }
        Ok(())
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

pub(super) async fn install_binary(binary: &Path, bin_dir: &Path) -> Result<()> {
    let io = |e: std::io::Error| Error::other(format!("Failed to install the new binary: {e}"));
    tokio::fs::create_dir_all(bin_dir).await.map_err(io)?;
    let current = bin_dir.join("current");
    let staged = bin_dir.join("current.new");
    tokio::fs::copy(binary, &staged).await.map_err(io)?;
    if tokio::fs::try_exists(&current).await.map_err(io)? {
        tokio::fs::rename(&current, bin_dir.join("previous"))
            .await
            .map_err(io)?;
    }
    tokio::fs::rename(&staged, &current).await.map_err(io)
}
