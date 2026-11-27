//! On-disk run state, so a swarm survives restarts and reboots.
//!
//! A run directory holds `swarm.json` (the agent tree, statuses and unread mail) and `sessions/`
//! (each agent's conversation, written by the SDK's `SessionManager`). `swarm.json` is replaced
//! atomically; writes carry a generation number so a slow, older write never lands after a newer
//! one, and a burst of save requests costs one write: a [`Persister::save`] builds its state only
//! once it holds the write lock, and skips it when a write that began after the request has
//! covered it already.
//!
//! Cancelled agents' records, most of a long run's agents and of its state, are kept in
//! `cancelled.json` instead: they hardly change once cancelled, while `swarm.json` is rewritten
//! many times a minute. `swarm.json` lists every agent's id in spawn order (`order`), so a load
//! puts them back in place; `cancelled.json` is written before it, and only when a cancelled
//! record changed.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use strands_agents::{Error, Result};

/// File name of the swarm state inside a run directory.
pub const STATE_FILE: &str = "swarm.json";
/// File name of the cancelled agents' records inside a run directory, kept out of [`STATE_FILE`].
pub const CANCELLED_FILE: &str = "cancelled.json";
/// Directory of per-agent sessions inside a run directory.
pub const SESSIONS_DIR: &str = "sessions";
/// A cancelled agent's full task, in its session directory ([`archive_task`]).
pub const TASK_FILE: &str = "task.md";
/// Characters of a cancelled agent's task that stay in `swarm.json` once the file holds it all.
pub const TASK_PREFIX_CHARS: usize = 500;

/// `task` cut to [`TASK_PREFIX_CHARS`] characters, with `…` after a cut one. Cutting the result
/// again changes nothing.
pub fn task_prefix(task: &str) -> String {
    match task.char_indices().nth(TASK_PREFIX_CHARS) {
        Some((end, _)) => format!("{}…", &task[..end]),
        None => task.to_owned(),
    }
}

/// Keep `task` whole in `sessions/<id>/task.md` (unless [`task_prefix`] is all of it), so that
/// `swarm.json` can hold only the prefix. Whether it did: a task that is all prefix needs no file.
pub fn archive_task(sessions: &Path, id: &str, task: &str) -> std::io::Result<bool> {
    if task_prefix(task) == task {
        return Ok(false);
    }
    crate::files::replace(&sessions.join(id).join(TASK_FILE), task.as_bytes(), false).map(|()| true)
}

/// The task of an agent saved with `task_archived`: the file's, or `prefix` if it is gone (the
/// session was removed, [`AgentState::conversation_removed`]).
pub fn load_task(sessions: &Path, id: &str, prefix: String) -> String {
    std::fs::read_to_string(sessions.join(id).join(TASK_FILE)).unwrap_or(prefix)
}

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
    /// On disk only (empty once loaded): every agent's id in spawn order, cancelled ones too,
    /// whose records are in [`CANCELLED_FILE`] while `agents` holds the rest. Absent in runs
    /// saved before, whose `agents` hold them all.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub order: Vec<String>,
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
    /// Mail settle window in seconds set at runtime (`0`: none), overriding `--mail-settle`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mail_settle: Option<u64>,
    /// Host load limit set at runtime (`0`: none), overriding `--max-load`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_load: Option<f64>,
    /// Per-subtree job cap set at runtime (`0`: none), overriding `--max-jobs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_jobs: Option<usize>,
    /// Free-memory floor set at runtime (`0`: none), overriding `--min-free-mem`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_free_mem: Option<u64>,
    /// The priority lane set at runtime (agent ids; empty: root agents only), overriding
    /// `--priority`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<Vec<String>>,
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
    /// Tokens by model and recent calls, for cost estimates (empty for runs saved before). A
    /// cancelled agent's is [`crate::cost::Ledger::compacted`].
    #[serde(default, skip_serializing_if = "crate::cost::Ledger::is_empty")]
    pub ledger: crate::cost::Ledger,
    /// Whether `task` is only the first [`TASK_PREFIX_CHARS`] of it, the rest being in
    /// [`TASK_FILE`] ([`archive_task`]). Only cancelled agents.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub task_archived: bool,
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
    /// When it was cancelled, unix seconds; runs saved before this was recorded get it from the
    /// newest file of its session when `--keep-cancelled` first looks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cancelled_at: Option<i64>,
    /// Seconds it was kept after being cancelled when its session directory (conversation, stash,
    /// open batch) was removed by `--keep-cancelled`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_removed: Option<u64>,
    /// Whether nothing can wake it ([`Dormancy`]).
    #[serde(default, skip_serializing_if = "Dormancy::is_default")]
    pub dormancy: Dormancy,
}

/// How long an idle agent has had nothing to wake it, and what its parent was told about that.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dormancy {
    /// When the swarm first saw it idle with nothing to wake it, unix seconds; 0 if it is not.
    #[serde(default)]
    pub since: i64,
    /// Its parent was told about this spell.
    #[serde(default)]
    pub told: bool,
    /// Its last turn ended with a reply to its parent, and nothing has reached it since.
    #[serde(default)]
    pub replied: bool,
}

impl Dormancy {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
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
    let stale = crate::files::remove_stale_temps(dir);
    if stale > 0 {
        tracing::info!(stale, "removed temporary copies left by writes a crash cut short");
    }
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
    let Some(mut state) = read_json::<SwarmState>(&dir.join(STATE_FILE)).await? else {
        return Ok(None);
    };
    let cancelled = read_json::<CancelledState>(&dir.join(CANCELLED_FILE))
        .await?
        .map_or_else(Vec::new, |file| file.agents);
    let order = match state.order.is_empty() {
        true => state.agents.iter().map(|agent| agent.id.clone()).collect(),
        false => std::mem::take(&mut state.order),
    };
    // `swarm.json` wins: written second, it is the newer record of an agent in both (a crash
    // between the two writes of a cancel).
    let mut records: HashMap<String, AgentState> = cancelled
        .iter()
        .cloned()
        .chain(std::mem::take(&mut state.agents))
        .map(|agent| (agent.id.clone(), agent))
        .collect();
    let mut seen = HashSet::new();
    state.agents = order
        .into_iter()
        .filter(|id| seen.insert(id.clone()))
        .filter_map(|id| records.remove(&id))
        .collect();
    // Without `order`, `swarm.json` was written by a binary that kept every agent in it: one from
    // before this file, or one rolled back to since, which dropped the cancelled agents it did
    // not know of. Those go back in by number, which is how ids are handed out, as does one
    // spawned and cancelled within a save that a crash cut between the two files.
    for agent in cancelled.iter().filter_map(|agent| records.remove(&agent.id)) {
        let at = state
            .agents
            .iter()
            .position(|other| id_number(&other.id) > id_number(&agent.id));
        state.agents.insert(at.unwrap_or(state.agents.len()), agent);
    }
    // Never hand out an id again.
    let last = state
        .agents
        .iter()
        .map(|agent| id_number(&agent.id))
        .filter(|n| *n < u64::MAX);
    state.next_id = state.next_id.max(last.max().map_or(0, |n| n + 1));
    Ok(Some(state))
}

/// The number in agent id `a<n>`, for ordering.
fn id_number(id: &str) -> u64 {
    id.strip_prefix('a')
        .and_then(|n| n.parse().ok())
        .unwrap_or(u64::MAX)
}

/// The JSON file at `path`, or `None` when there is none.
async fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    let text = match tokio::fs::read_to_string(path).await {
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

/// [`CANCELLED_FILE`]: the cancelled agents' records, in spawn order.
#[derive(Debug, Serialize, Deserialize)]
struct CancelledState {
    agents: Vec<AgentState>,
}

/// Serializes writes of `swarm.json` and drops any older than the last one written.
#[derive(Debug)]
pub struct Persister {
    dir: PathBuf,
    written: tokio::sync::Mutex<Written>,
    /// Save requests so far ([`Persister::request`]).
    requested: std::sync::atomic::AtomicU64,
    /// Writes of [`CANCELLED_FILE`] so far.
    cancelled_writes: std::sync::atomic::AtomicU64,
}

/// The state last written: its generation, the length and hash of `swarm.json`, and the
/// cancelled agents' records in [`CANCELLED_FILE`].
#[derive(Debug, Default)]
struct Written {
    generation: u64,
    print: Option<(usize, u64)>,
    cancelled: Option<Vec<AgentState>>,
}

impl Persister {
    /// Persister for the run directory `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            written: Default::default(),
            requested: Default::default(),
            cancelled_writes: Default::default(),
        }
    }

    /// Ask for a save: the ticket to hand to [`Persister::save`], once the change to save is made.
    pub fn request(&self) -> u64 {
        self.requested.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1
    }

    /// How many times [`CANCELLED_FILE`] was written.
    pub fn cancelled_writes(&self) -> u64 {
        self.cancelled_writes.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Save the state `snapshot` returns for `ticket`, or nothing when a write that began after
    /// the ticket was issued has covered it. Waiting saves hold no state: a burst of requests
    /// (every model call asks for one) must not queue a full copy of the swarm each behind the
    /// slow write, as a writer that is behind would otherwise let grow without bound.
    pub async fn save(&self, ticket: u64, snapshot: impl FnOnce() -> SwarmState) -> Result<()> {
        let mut written = self.written.lock().await;
        if ticket <= written.generation {
            return Ok(());
        }
        // Every request made so far precedes the snapshot, so this write covers them all.
        let covered = self.requested.load(std::sync::atomic::Ordering::SeqCst);
        self.write_locked(&mut written, covered, snapshot()).await
    }

    /// The run directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Write `state` as generation `generation`, unless a newer one is already on disk or the file
    /// already holds the same bytes. Compact JSON: nobody reads it by hand, and [`load`] takes
    /// the pretty form older runs wrote too.
    pub async fn write(&self, generation: u64, state: &SwarmState) -> Result<()> {
        let mut written = self.written.lock().await;
        self.write_locked(&mut written, generation, state.clone()).await
    }

    async fn write_locked(
        &self,
        written: &mut Written,
        generation: u64,
        mut state: SwarmState,
    ) -> Result<()> {
        if generation <= written.generation {
            return Ok(());
        }
        let agents = std::mem::take(&mut state.agents);
        if agents.iter().any(is_cancelled) {
            state.order = agents.iter().map(|agent| agent.id.clone()).collect();
        }
        let (cancelled, live): (Vec<_>, Vec<_>) = agents.into_iter().partition(is_cancelled);
        state.agents = live;
        // First, so a crash between the two writes leaves an agent cancelled since in both files
        // (and `swarm.json`'s older record wins), never in neither. Written once per process
        // even when empty, so a file a removed run left behind is not read into a new one.
        if written.cancelled.as_ref() != Some(&cancelled) {
            written.cancelled = None;
            let file = CancelledState { agents: cancelled };
            let json = serde_json::to_vec(&file).map_err(|e| Error::other(e.to_string()))?;
            self.replace(CANCELLED_FILE, json).await?;
            self.cancelled_writes
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            written.cancelled = Some(file.agents);
        }
        let capacity = written.print.map_or(0, |(len, _)| len + len / 16);
        let mut json = Vec::with_capacity(capacity);
        serde_json::to_writer(&mut json, &state).map_err(|e| Error::other(e.to_string()))?;
        let print = {
            use std::hash::Hasher;
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            hasher.write(&json);
            (json.len(), hasher.finish())
        };
        if written.print != Some(print) {
            written.print = None;
            self.replace(STATE_FILE, json).await?;
        }
        written.generation = generation;
        written.print = Some(print);
        Ok(())
    }

    /// Replace `name` in the run directory with `json`. Durable: after a power loss, the old
    /// state or the new one, never an empty file. A full disk spends the run's reserve on it.
    async fn replace(&self, name: &str, json: Vec<u8>) -> Result<()> {
        let path = self.dir.join(name);
        let json = std::sync::Arc::new(json);
        crate::reserve::save_with(&self.dir, || {
            crate::files::replace_async(&path, std::sync::Arc::clone(&json), true)
        })
        .await
        .map_err(|e| Error::other(format!("Failed to save swarm state: {e}")))
    }
}

fn is_cancelled(agent: &AgentState) -> bool {
    agent.status == "cancelled"
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

    #[cfg(unix)]
    #[tokio::test]
    async fn the_state_is_written_compact_in_order_and_not_again_when_unchanged() {
        use std::os::unix::fs::MetadataExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(STATE_FILE);
        let mut state: SwarmState = serde_json::from_value(serde_json::json!({
            "run_id": "r", "cwd": dir.path(), "next_id": 2,
            "agents": [{"id": "a1", "parent": "user", "cwd": dir.path(), "task": "t",
                        "status": "idle", "mailbox": []}],
        }))
        .unwrap();
        let persister = Persister::new(dir.path());
        // Each write renames a new file into place.
        let inode = || std::fs::metadata(&path).unwrap().ino();

        persister.write(1, &state).await.unwrap();
        let first = inode();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains('\n') && text.contains("\"next_id\":2"), "{text}");
        persister.write(2, &state).await.unwrap();
        assert_eq!(inode(), first, "the same state");

        state.next_id = 3;
        persister.write(3, &state).await.unwrap();
        assert_ne!(inode(), first, "a new state");
        state.next_id = 4;
        persister.write(3, &state).await.unwrap();
        assert_eq!(
            load(dir.path()).await.unwrap().unwrap().next_id,
            3,
            "not older than written"
        );
        persister.write(4, &state).await.unwrap();
        assert_eq!(load(dir.path()).await.unwrap().unwrap().next_id, 4);
    }

    #[tokio::test]
    async fn a_burst_of_save_requests_builds_one_state_not_one_each() {
        use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
        let dir = tempfile::tempdir().unwrap();
        let state = |next_id| -> SwarmState {
            serde_json::from_value(serde_json::json!({
                "run_id": "r", "cwd": dir.path(), "next_id": next_id,
                "agents": [{"id": "a1", "parent": "user", "cwd": dir.path(), "task": "t",
                            "status": "idle", "mailbox": []}],
            }))
            .unwrap()
        };
        let persister = Persister::new(dir.path());
        let built = AtomicUsize::new(0);
        let snapshot = |next_id| {
            let built = &built;
            let state = &state;
            move || {
                built.fetch_add(1, SeqCst);
                state(next_id)
            }
        };

        // 200 requests wait for the lock behind one another: the first to run saves for all.
        let tickets: Vec<u64> = (0..200).map(|_| persister.request()).collect();
        let saves = tickets.iter().map(|ticket| persister.save(*ticket, snapshot(5)));
        for result in futures::future::join_all(saves).await {
            result.unwrap();
        }
        assert_eq!(built.load(SeqCst), 1, "one state built for the burst");
        assert_eq!(load(dir.path()).await.unwrap().unwrap().next_id, 5);

        // A request made after that write began is not covered by it.
        let ticket = persister.request();
        persister.save(ticket, snapshot(6)).await.unwrap();
        assert_eq!(built.load(SeqCst), 2);
        assert_eq!(load(dir.path()).await.unwrap().unwrap().next_id, 6);
        // A ticket the last write covered is a no-op.
        persister.save(ticket, snapshot(7)).await.unwrap();
        assert_eq!(built.load(SeqCst), 2);
    }

    #[tokio::test]
    async fn a_pretty_state_from_an_older_run_loads() {
        let dir = tempfile::tempdir().unwrap();
        let state = serde_json::json!({
            "run_id": "r", "cwd": dir.path(), "next_id": 7,
            "agents": [{"id": "a1", "parent": "user", "cwd": dir.path(), "task": "t",
                        "status": "idle", "mailbox": []}],
        });
        let pretty = serde_json::to_vec_pretty(&state).unwrap();
        std::fs::write(dir.path().join(STATE_FILE), pretty).unwrap();
        let loaded = load(dir.path()).await.unwrap().unwrap();
        assert_eq!((loaded.next_id, loaded.agents[0].id.as_str()), (7, "a1"));
    }

    /// A state of agents `(id, status)` in that (spawn) order.
    fn agents_state(dir: &Path, agents: &[(&str, &str)]) -> SwarmState {
        let agents: Vec<_> = agents
            .iter()
            .map(|(id, status)| {
                serde_json::json!({"id": id, "parent": "user", "cwd": dir, "task": "t",
                                   "status": status, "mailbox": [], "input_tokens": 10})
            })
            .collect();
        serde_json::from_value(serde_json::json!({
            "run_id": "r", "cwd": dir, "next_id": 9, "agents": agents,
        }))
        .unwrap()
    }

    fn ids(state: &SwarmState) -> Vec<(&str, &str)> {
        state
            .agents
            .iter()
            .map(|a| (a.id.as_str(), a.status.as_str()))
            .collect()
    }

    fn file(dir: &Path, name: &str) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap()).unwrap()
    }

    // Spawn order that is not id order (a concurrent spawn can insert a later id first).
    const MIXED: [(&str, &str); 5] = [
        ("a1", "idle"),
        ("a3", "cancelled"),
        ("a2", "running"),
        ("a5", "cancelled"),
        ("a4", "idle"),
    ];

    #[tokio::test]
    async fn cancelled_agents_are_kept_apart_and_load_back_in_spawn_order() {
        let dir = tempfile::tempdir().unwrap();
        let state = agents_state(dir.path(), &MIXED);
        Persister::new(dir.path()).write(1, &state).await.unwrap();
        let hot = file(dir.path(), STATE_FILE);
        let live: Vec<_> = hot["agents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a["id"].clone())
            .collect();
        assert_eq!(live, ["a1", "a2", "a4"]);
        assert_eq!(hot["order"], serde_json::json!(["a1", "a3", "a2", "a5", "a4"]));
        assert_eq!(
            file(dir.path(), CANCELLED_FILE)["agents"]
                .as_array()
                .unwrap()
                .len(),
            2
        );

        let loaded = load(dir.path()).await.unwrap().unwrap();
        assert_eq!(ids(&loaded), MIXED);
        assert_eq!((loaded.next_id, loaded.order.len()), (9, 0));
        // Readers of the whole run (`--cost`, `--run NAME`) see the cancelled agents' spend.
        assert_eq!(loaded.agents.iter().map(|a| a.input_tokens).sum::<u64>(), 50);
    }

    #[tokio::test]
    async fn an_unchanged_cancelled_set_is_not_rewritten() {
        let dir = tempfile::tempdir().unwrap();
        let persister = Persister::new(dir.path());
        let mut state = agents_state(dir.path(), &MIXED);
        persister.write(1, &state).await.unwrap();
        state.agents[0].input_tokens = 99;
        persister.write(2, &state).await.unwrap();
        assert_eq!(persister.cancelled_writes(), 1, "only swarm.json changed");
        assert_eq!(
            load(dir.path()).await.unwrap().unwrap().agents[0].input_tokens,
            99
        );

        state.agents[2].status = "cancelled".into();
        persister.write(3, &state).await.unwrap();
        state.agents[3].conversation_removed = Some(5);
        persister.write(4, &state).await.unwrap();
        assert_eq!(
            persister.cancelled_writes(),
            3,
            "a cancel and a cancelled record's change"
        );
        let loaded = load(dir.path()).await.unwrap().unwrap();
        assert_eq!(
            (
                loaded.agents[2].status.as_str(),
                loaded.agents[3].conversation_removed
            ),
            ("cancelled", Some(5))
        );

        // A new process writes it once, empty or not: a stale one is never read into a new run.
        let fresh = tempfile::tempdir().unwrap();
        std::fs::copy(dir.path().join(CANCELLED_FILE), fresh.path().join(CANCELLED_FILE)).unwrap();
        let state = agents_state(fresh.path(), &[("a1", "idle")]);
        Persister::new(fresh.path()).write(1, &state).await.unwrap();
        assert_eq!(ids(&load(fresh.path()).await.unwrap().unwrap()), [("a1", "idle")]);
    }

    #[tokio::test]
    async fn an_old_state_with_every_agent_loads_and_is_split_on_save() {
        let dir = tempfile::tempdir().unwrap();
        let state = agents_state(dir.path(), &MIXED);
        std::fs::write(dir.path().join(STATE_FILE), serde_json::to_vec(&state).unwrap()).unwrap();
        let loaded = load(dir.path()).await.unwrap().unwrap();
        assert_eq!(ids(&loaded), MIXED);
        Persister::new(dir.path()).write(1, &loaded).await.unwrap();
        assert_eq!(
            file(dir.path(), STATE_FILE)["agents"].as_array().unwrap().len(),
            3
        );
        assert_eq!(ids(&load(dir.path()).await.unwrap().unwrap()), MIXED);
    }

    #[tokio::test]
    async fn a_crash_between_the_two_writes_loses_and_duplicates_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let persister = Persister::new(dir.path());
        let before = agents_state(dir.path(), &MIXED);
        persister.write(1, &before).await.unwrap();
        let old_hot = std::fs::read(dir.path().join(STATE_FILE)).unwrap();
        // a2 is cancelled, and a6 spawned and cancelled, within one save that dies after
        // cancelled.json: swarm.json is the one before.
        let mut after = before.clone();
        after.agents[2].status = "cancelled".into();
        after.agents[2].input_tokens = 77;
        after
            .agents
            .extend(agents_state(dir.path(), &[("a6", "cancelled")]).agents);
        after.next_id = 7;
        persister.write(2, &after).await.unwrap();
        std::fs::write(dir.path().join(STATE_FILE), old_hot).unwrap();

        let loaded = load(dir.path()).await.unwrap().unwrap();
        // a2 is as swarm.json last had it; a6 is kept, and its id is not handed out again.
        assert_eq!(ids(&loaded), [MIXED.as_slice(), &[("a6", "cancelled")]].concat());
        assert_eq!((loaded.agents[2].input_tokens, loaded.next_id), (10, 9));
        let mut state = before.clone();
        state.next_id = 2;
        std::fs::write(dir.path().join(STATE_FILE), serde_json::to_vec(&state).unwrap()).unwrap();
        assert_eq!(load(dir.path()).await.unwrap().unwrap().next_id, 7);
    }

    #[tokio::test]
    async fn a_rolled_back_binary_dropping_cancelled_agents_loses_none() {
        let dir = tempfile::tempdir().unwrap();
        Persister::new(dir.path())
            .write(1, &agents_state(dir.path(), &MIXED))
            .await
            .unwrap();
        // An older binary loads only swarm.json's live agents, spawns a6 and saves them all.
        let older = agents_state(
            dir.path(),
            &[("a1", "idle"), ("a2", "running"), ("a4", "idle"), ("a6", "idle")],
        );
        std::fs::write(dir.path().join(STATE_FILE), serde_json::to_vec(&older).unwrap()).unwrap();
        let loaded = load(dir.path()).await.unwrap().unwrap();
        assert_eq!(
            ids(&loaded),
            [
                ("a1", "idle"),
                ("a2", "running"),
                ("a3", "cancelled"),
                ("a4", "idle"),
                ("a5", "cancelled"),
                ("a6", "idle")
            ]
        );
        // A self-test copy of just swarm.json and sessions/ loads its live agents.
        std::fs::remove_file(dir.path().join(CANCELLED_FILE)).unwrap();
        Persister::new(dir.path())
            .write(1, &agents_state(dir.path(), &MIXED))
            .await
            .unwrap();
        std::fs::remove_file(dir.path().join(CANCELLED_FILE)).unwrap();
        let live = load(dir.path()).await.unwrap().unwrap();
        assert_eq!(ids(&live), [("a1", "idle"), ("a2", "running"), ("a4", "idle")]);
    }
}
