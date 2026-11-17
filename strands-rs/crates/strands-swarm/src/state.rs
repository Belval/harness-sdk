//! On-disk run state, so a swarm survives restarts and reboots.
//!
//! A run directory holds `swarm.json` (the agent tree, statuses and unread mail) and `sessions/`
//! (each agent's conversation, written by the SDK's `SessionManager`). `swarm.json` is replaced
//! atomically; writes carry a generation number so a slow, older write never lands after a newer
//! one.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use strands_agents::{Error, Result};

/// File name of the swarm state inside a run directory.
pub const STATE_FILE: &str = "swarm.json";
/// Directory of per-agent sessions inside a run directory.
pub const SESSIONS_DIR: &str = "sessions";

/// Persisted swarm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwarmState {
    /// Run id.
    pub run_id: String,
    /// Root working directory.
    pub cwd: PathBuf,
    /// Next agent number.
    pub next_id: u64,
    /// Agents in spawn order.
    pub agents: Vec<AgentState>,
    /// Swarm-wide token budget (input + output), as last set; `None` for no limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u64>,
    /// The last `--budget-tokens` given (a start without it keeps the previous one). A resume
    /// given the same flag again (a supervisor replays its arguments) keeps `budget_tokens`, which
    /// `/budget` may have changed since; a different one overrides it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_flag: Option<u64>,
    /// The TASK the run was last started with. A resume given the same one again (a supervisor
    /// replays its arguments) does not send it to the live root a second time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_flag: Option<String>,
    /// Live-agent limit set at runtime, overriding `--max-agents`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_agents: Option<usize>,
    /// Model-call limit set at runtime (`0`: none), overriding `--max-concurrent-calls`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_calls: Option<usize>,
    /// Commit of the binary that saved it (`None` for older runs or builds outside git).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    /// Pending scheduled messages.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub schedules: Vec<crate::schedule::Schedule>,
}

/// Persisted agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentState {
    /// Id.
    pub id: String,
    /// Name.
    pub name: Option<String>,
    /// Parent id.
    pub parent: String,
    /// Where it was started: its worktree, if it has one.
    pub cwd: PathBuf,
    /// Where a lone `cd` moved it, if it did ([`crate::workdir`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workdir: Option<PathBuf>,
    /// Own branch.
    pub branch: Option<String>,
    /// Initial task.
    pub task: String,
    /// Model override.
    pub model: Option<String>,
    /// Goal.
    pub goal: Option<String>,
    /// `starting`, `running`, `idle`, `errored` or `cancelled`.
    pub status: String,
    /// Unread mail.
    pub mailbox: Vec<MailState>,
    /// Environment variables for the agent's commands.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// Compaction point from `spawn_agent` (tokens); `None` follows the swarm's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compact_at: Option<u64>,
    /// Token budget of the agent and its descendants, set at spawn or raised since.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u64>,
    /// Input tokens used so far (cache reads and writes included), across restarts.
    #[serde(default)]
    pub input_tokens: u64,
    /// Output tokens used so far, across restarts.
    #[serde(default)]
    pub output_tokens: u64,
    /// The part of `input_tokens` read from the prompt cache (0 for runs saved before it was
    /// tracked, whose input then counts as uncached).
    #[serde(default)]
    pub cache_read_tokens: u64,
    /// The part of `input_tokens` written to the prompt cache.
    #[serde(default)]
    pub cache_write_tokens: u64,
    /// Tokens by model and recent calls, for cost estimates (empty for runs saved before).
    #[serde(default, skip_serializing_if = "crate::cost::Ledger::is_empty")]
    pub ledger: crate::cost::Ledger,
    /// Context window utilization of the last model call (`0..1+`), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<f64>,
    /// Interrupted by the user: a goal agent stays idle until its next message, restarts included.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub paused: bool,
    /// Tool calls in flight when saved (waiting tools left out): seconds each had run, by tool use
    /// id, for the restart notice.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub running_calls: BTreeMap<String, u64>,
    /// What reclaiming this agent's disk last did, as journaled ([`crate::gc::Collected::lines`]),
    /// so a restart that finds the same (a worktree still kept) does not journal it again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reclaim_logged: Option<String>,
    /// When it last showed a sign of progress (a successful tool call or a model reply), unix
    /// seconds, so a restart does not make a stuck agent look fresh.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<i64>,
}

/// Persisted mail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MailState {
    /// Sender id.
    pub from: String,
    /// Sender label.
    pub from_label: String,
    /// Kind.
    pub kind: String,
    /// Body.
    pub text: String,
    /// Attached image paths.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<PathBuf>,
}

/// Where a named run lives: `$XDG_STATE_HOME/strands-swarm/<run>`, else
/// `~/.local/state/strands-swarm/<run>`.
pub fn run_dir(run: &str) -> Result<PathBuf> {
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/state")))
        .ok_or_else(|| Error::other("Neither XDG_STATE_HOME nor HOME is set."))?;
    Ok(base.join("strands-swarm").join(run))
}

/// Files a run (or its supervisor) writes as it goes, read for when it was last alive. Not
/// `swarm.log` or `stderr.log`: a resuming process writes those before it loads the state.
const ALIVE_FILES: [&str; 4] = [
    STATE_FILE,
    crate::journal::EVENTS_FILE,
    crate::journal::JOURNAL_FILE,
    "supervisor.log",
];

/// Load `swarm.json` from `dir`, or `None` when the run has never been saved.
///
/// The saved [`AgentState::running_calls`] are as of the last save, which a hard crash can
/// follow by minutes; each is extended to the run's last sign of life: the newest write among
/// [`ALIVE_FILES`] (swarmctl notes the exit in `supervisor.log` the moment it happens).
pub async fn load(dir: &Path) -> Result<Option<SwarmState>> {
    let mut state = match load_saved(dir).await? {
        Some(state) => state,
        None => return Ok(None),
    };
    let gap = ran_on_after_save(dir).as_secs();
    for agent in &mut state.agents {
        agent.running_calls.values_mut().for_each(|ran| *ran += gap);
    }
    Ok(Some(state))
}

/// How long the run went on after `swarm.json` was last written.
fn ran_on_after_save(dir: &Path) -> std::time::Duration {
    let modified = |name: &str| std::fs::metadata(dir.join(name)).and_then(|m| m.modified()).ok();
    let (Some(saved), Some(alive)) = (
        modified(STATE_FILE),
        ALIVE_FILES.iter().filter_map(|name| modified(name)).max(),
    ) else {
        return std::time::Duration::ZERO;
    };
    alive.duration_since(saved).unwrap_or_default()
}

async fn load_saved(dir: &Path) -> Result<Option<SwarmState>> {
    let path = dir.join(STATE_FILE);
    let text = match tokio::fs::read_to_string(&path).await {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(Error::other(format!(
                "Failed to read {}: {error}",
                path.display()
            )))
        }
    };
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|error| Error::other(format!("Failed to parse {}: {error}", path.display())))
}

/// Serializes writes of `swarm.json` and drops any older than the last one written.
#[derive(Debug)]
pub struct Persister {
    dir: PathBuf,
    written: tokio::sync::Mutex<u64>,
}

impl Persister {
    /// Persister for the run directory `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            written: tokio::sync::Mutex::new(0),
        }
    }

    /// The run directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Write `state` as generation `generation`, unless a newer one is already on disk.
    pub async fn write(&self, generation: u64, state: &SwarmState) -> Result<()> {
        let mut written = self.written.lock().await;
        if generation <= *written {
            return Ok(());
        }
        let json = serde_json::to_vec_pretty(state).map_err(|e| Error::other(e.to_string()))?;
        // Durable: after a power loss, the old state or the new one, never an empty file.
        crate::files::replace_async(&self.dir.join(STATE_FILE), json, true)
            .await
            .map_err(|e| Error::other(format!("Failed to save swarm state: {e}")))?;
        *written = generation;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    /// Set `path`'s modification time to `ago` seconds before `now` (one instant for all the
    /// files of a test, so the gaps between them are exact however slowly it runs).
    fn touch(path: &Path, now: SystemTime, ago: u64) {
        let file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
            .unwrap();
        file.set_modified(now - Duration::from_secs(ago)).unwrap();
    }

    #[tokio::test]
    async fn a_cut_call_runs_on_to_the_last_sign_of_life_not_the_last_save() {
        let dir = tempfile::tempdir().unwrap();
        let state = serde_json::json!({
            "run_id": "r",
            "cwd": dir.path(),
            "next_id": 2,
            "agents": [{"id": "a1", "parent": "user", "cwd": dir.path(), "task": "t",
                        "status": "running", "mailbox": [], "running_calls": {"t1": 10}}],
        });
        std::fs::write(dir.path().join(STATE_FILE), state.to_string()).unwrap();
        let now = SystemTime::now();
        touch(&dir.path().join(STATE_FILE), now, 100);
        let ran = |state: Option<SwarmState>| state.unwrap().agents[0].running_calls["t1"];
        // Nothing written since the save: the saved time stands.
        assert_eq!(ran(load(dir.path()).await.unwrap()), 10);
        // Events went on for 30s after it, and swarmctl noted the exit 60s after it.
        touch(&dir.path().join(crate::journal::EVENTS_FILE), now, 70);
        assert_eq!(ran(load(dir.path()).await.unwrap()), 40);
        touch(&dir.path().join("supervisor.log"), now, 40);
        assert_eq!(ran(load(dir.path()).await.unwrap()), 70);
        // The resuming process's own log is no sign of the old one's life.
        touch(&dir.path().join(crate::logging::LOG_FILE), now, 0);
        assert_eq!(ran(load(dir.path()).await.unwrap()), 70);
    }
}
