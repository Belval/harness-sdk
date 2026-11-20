//! Removing the saved sessions of agents cancelled long ago ([`SwarmConfig::keep_cancelled`]).

use std::time::SystemTime;

use super::*;

impl Swarm {
    /// Remove the session directory (conversation snapshots, stash, open batch) of every agent
    /// cancelled at least [`SwarmConfig::keep_cancelled`] ago, as on resume. Its entry in
    /// `swarm.json` stays, marked ([`AgentState::conversation_removed`]), so the tree, cost and
    /// branch records survive. Live agents are never touched, nor are cancelled ones whose actor
    /// still runs or whose worktree is still there (kept for its owner's changes).
    ///
    /// A cancel time that was never recorded (runs saved before it was) is taken from the newest
    /// modification time in the session directory. Journals `sessions_pruned` and returns the ids
    /// of the agents removed now.
    pub async fn prune_cancelled_sessions(&self) -> Vec<String> {
        let (Some(keep), Some(state_dir)) = (self.inner.config.keep_cancelled, &self.inner.config.state_dir)
        else {
            return Vec::new();
        };
        let sessions = state_dir.join(SESSIONS_DIR);
        let candidates: Vec<(Arc<Node>, i64)> = self
            .nodes()
            .into_iter()
            .filter(|node| {
                *node.status.lock() == AgentStatus::Cancelled
                    && node.finished.load(Ordering::SeqCst)
                    && node.conversation_removed.load(Ordering::Relaxed) == 0
                    && !(node.branch.is_some() && node.cwd.is_dir())
            })
            .map(|node| {
                let at = node.cancelled_at.load(Ordering::Relaxed);
                (node, at)
            })
            .collect();
        if candidates.is_empty() {
            return Vec::new();
        }
        let now = self.now().timestamp();
        let keep_secs = keep.as_secs().max(1);
        let ids: Vec<(String, i64)> = candidates
            .iter()
            .map(|(node, at)| (node.id.clone(), *at))
            .collect();
        let outcome = tokio::task::spawn_blocking(move || {
            ids.into_iter()
                .filter_map(|(id, known)| {
                    let dir = sessions.join(&id);
                    // Not through a symlink: only a real directory of the run is ever deleted.
                    std::fs::symlink_metadata(&dir)
                        .ok()
                        .filter(|meta| meta.is_dir())?;
                    let at = if known > 0 { known } else { newest(&dir) };
                    let old = at > 0 && now.saturating_sub(at) >= keep_secs as i64;
                    let bytes = old.then(|| crate::gc::dir_size(&dir));
                    let removed = bytes.filter(|_| match std::fs::remove_dir_all(&dir) {
                        Ok(()) => true,
                        Err(error) => {
                            tracing::warn!(agent = %id, %error, "cannot remove a cancelled agent's session");
                            false
                        }
                    });
                    Some((id, at, removed))
                })
                .collect::<Vec<_>>()
        })
        .await
        .unwrap_or_default();

        let mut pruned = Vec::new();
        let mut total = 0;
        for (id, at, removed) in outcome {
            let Some((node, _)) = candidates.iter().find(|(node, _)| node.id == id) else {
                continue;
            };
            node.cancelled_at.store(at, Ordering::Relaxed);
            if let Some(bytes) = removed {
                node.conversation_removed.store(keep_secs, Ordering::Relaxed);
                total += bytes;
                pruned.push(id);
            }
        }
        // The cancel times found are worth keeping even when nothing was removed.
        self.persist();
        if !pruned.is_empty() {
            self.journal_event(
                "sessions_pruned",
                serde_json::json!({ "agents": pruned.len(), "bytes": total, "ids": pruned }),
            );
        }
        pruned
    }

    /// [`Swarm::prune_cancelled_sessions`], telling viewers what it removed.
    pub(super) async fn prune_cancelled_sessions_announced(&self) {
        let keep = self
            .inner
            .config
            .keep_cancelled
            .map_or(0, |keep| keep.as_secs().max(1));
        for id in self.prune_cancelled_sessions().await {
            self.emit(SwarmEvent::ConversationRemoved { id, after_secs: keep });
        }
    }
}

/// The newest modification time under `dir` (itself included), unix seconds; 0 when unreadable.
pub(super) fn newest(dir: &Path) -> i64 {
    fn walk(path: &Path, newest: &mut SystemTime) {
        let Ok(meta) = std::fs::symlink_metadata(path) else {
            return;
        };
        if let Ok(modified) = meta.modified() {
            *newest = (*newest).max(modified);
        }
        if meta.is_dir() {
            for entry in std::fs::read_dir(path).into_iter().flatten().flatten() {
                walk(&entry.path(), newest);
            }
        }
    }
    let mut newest = SystemTime::UNIX_EPOCH;
    walk(dir, &mut newest);
    newest
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |since| since.as_secs() as i64)
}
