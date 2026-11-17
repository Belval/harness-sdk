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
use tokio::io::AsyncWriteExt;

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
    /// The `--budget-tokens` the run was last started with. A resume given the same flag again
    /// (a supervisor replays its arguments) keeps `budget_tokens`, which `/budget` may have
    /// changed since; a different one overrides it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_flag: Option<u64>,
    /// The TASK the run was last started with. A resume given the same one again (a supervisor
    /// replays its arguments) does not send it to the live root a second time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_flag: Option<String>,
    /// Live-agent limit set at runtime, overriding `--max-agents`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_agents: Option<usize>,
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
    /// Working directory.
    pub cwd: PathBuf,
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

/// Load `swarm.json` from `dir`, or `None` when the run has never been saved.
pub async fn load(dir: &Path) -> Result<Option<SwarmState>> {
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
        let io = |e: std::io::Error| Error::other(format!("Failed to save swarm state: {e}"));
        tokio::fs::create_dir_all(&self.dir).await.map_err(io)?;
        let tmp = self.dir.join(format!("{STATE_FILE}.tmp"));
        // Synced before the rename, and the directory after it: after a power loss, the old
        // state or the new one, never an empty file.
        let mut file = tokio::fs::File::create(&tmp).await.map_err(io)?;
        file.write_all(&json).await.map_err(io)?;
        file.sync_all().await.map_err(io)?;
        tokio::fs::rename(&tmp, self.dir.join(STATE_FILE))
            .await
            .map_err(io)?;
        if let Ok(dir) = tokio::fs::File::open(&self.dir).await {
            let _ = dir.sync_all().await;
        }
        *written = generation;
        Ok(())
    }
}
