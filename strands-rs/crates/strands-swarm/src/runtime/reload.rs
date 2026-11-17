//! Hot reload: installing a new binary and restarting into it now or once tool calls finish.

use super::*;

impl Swarm {
    /// Install `binary` as the run's next binary and ask the host process to restart into it:
    /// right away, or with [`ReloadWhen::Idle`] once no agent has a tool call in flight (at most
    /// [`SwarmConfig::reload_idle_timeout`] later). A newer reload supersedes a pending one.
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
        if let Some(journal) = &self.inner.journal {
            journal.event(
                "reload",
                serde_json::json!({ "binary": binary, "when": format!("{when:?}") }),
            );
        }
        let generation = self.inner.reload_generation.fetch_add(1, Ordering::SeqCst) + 1;
        match when {
            ReloadWhen::Now => self.inner.reload.cancel(),
            ReloadWhen::Idle => self.reload_when_idle(generation),
        }
        Ok(())
    }

    /// Signal the reload once nothing is in flight, unless a newer reload took over first.
    pub(super) fn reload_when_idle(&self, generation: u64) {
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

    /// Tool calls in flight across running agents, not counting tools that only wait (`wait`).
    pub fn tool_calls_in_flight(&self) -> usize {
        let nodes: Vec<Arc<Node>> = self.inner.agents.lock().values().cloned().collect();
        nodes
            .iter()
            .filter(|node| *node.status.lock() == AgentStatus::Running)
            .map(|node| *node.tool_calls.lock())
            .sum()
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
