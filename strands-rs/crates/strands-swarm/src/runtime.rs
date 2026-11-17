//! The swarm runtime: an agent registry, per-agent mailboxes, and the actor loop that drives each
//! agent turn by turn.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use futures::future::BoxFuture;
use futures::StreamExt;
use indexmap::IndexMap;
use parking_lot::Mutex;
use strands_agents::agent::{AgentStreamEvent, InvokeInput, InvokeOptions};
use strands_agents::hooks::AfterToolCallEvent;
use strands_agents::middleware::{wrap_fn, InvokeModelContext, InvokeModelStage};
use strands_agents::models::streaming::{ContentBlockDelta, ModelStreamEvent};
use strands_agents::models::{ModelEventStream, StreamOptions};
use strands_agents::tools::ToolRef;
use strands_agents::types::{ImageBlock, ImageSource, Message, ToolResultContent, ToolResultStatus, Usage};
use strands_agents::{Agent, Error, Model, Result};
use tokio::sync::{mpsc, Notify, OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::gc::{self, Collected};
use crate::images;
use crate::jobs::{self, Job, Jobs};
use crate::journal::Journal;
use crate::memory::Memory;
use crate::names;
use crate::resume;
use crate::schedule::{self as schedules, Schedule};
use crate::selftest;
use crate::state::{AgentState, MailState, Persister, SwarmState, SESSIONS_DIR};
use crate::tools::swarm_tools;
use crate::worktree;

mod actor;
mod background;
mod capacity;
mod collect;
mod disk;
mod landing;
mod mailbox;
mod persist;
mod reload;
mod report;
mod scheduler;
mod spend;
mod warm;

pub(crate) use actor::summarize_input;
use actor::Start;
pub use actor::SILENT_REPLY;
use background::jobs_dir;
use capacity::started;
pub use disk::{free_space, FreeSpace, DISK_CHECK_EVERY, DISK_SENDER};
use persist::history_event;
pub use warm::{KEEP_WARM_EVERY, KEEP_WARM_MAX};

/// Judge rejections in a row after which a goal agent's turn ends anyway, so a goal that cannot
/// be met does not burn turns forever; its parent hears the last attempt and can step in.
pub const GOAL_MAX_ATTEMPTS: usize = 8;
const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

/// How long a new binary's `--self-test` may take before a reload gives up on it.
pub const SELF_TEST_TIMEOUT: Duration = Duration::from_secs(60);

/// How long an idle-mode reload ([`ReloadWhen::Idle`]) waits for tool calls to finish before it
/// restarts anyway (the default of [`SwarmConfig::reload_idle_timeout`]).
pub const RELOAD_IDLE_TIMEOUT: Duration = Duration::from_secs(600);

/// Tool calls that only wait, so a reload does not wait for them ([`Swarm::tool_calls_in_flight`]).
const WAITING_TOOLS: &[&str] = &["wait"];

/// When [`Swarm::reload`] restarts the swarm into the installed binary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ReloadWhen {
    /// Once no agent has a tool call in flight, or after [`SwarmConfig::reload_idle_timeout`].
    #[default]
    Idle,
    /// Right away, interrupting whatever is in flight.
    Now,
}

/// Exit code that tells the supervisor to restart immediately into `<run>/bin/current`.
pub const RELOAD_EXIT_CODE: u8 = 75;

/// Pseudo agent id for the human at the keyboard: the root agent's parent.
pub const USER: &str = "user";

/// Lifecycle of one agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentStatus {
    /// Spawned while the agent limit was reached; it starts when a slot frees. It can already
    /// receive messages.
    Queued,
    /// The agent is being built.
    Starting,
    /// A turn is in progress.
    Running,
    /// Waiting for a message; the last turn ended normally.
    Idle,
    /// Waiting for a message; the last turn (or construction) failed with this error.
    Errored(String),
    /// Cancelled; terminal.
    Cancelled,
}

impl AgentStatus {
    /// Whether the agent can still receive messages and run turns.
    pub fn is_live(&self) -> bool {
        !matches!(self, Self::Cancelled)
    }
}

impl fmt::Display for AgentStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => f.write_str("queued"),
            Self::Starting => f.write_str("starting"),
            Self::Running => f.write_str("running"),
            Self::Idle => f.write_str("idle"),
            Self::Errored(error) => write!(f, "errored: {error}"),
            Self::Cancelled => f.write_str("cancelled"),
        }
    }
}

/// A message in an agent's mailbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mail {
    /// Sender id ([`USER`] for the human).
    pub from: String,
    /// Sender label as rendered to the receiver, e.g. `parser (a2)`.
    pub from_label: String,
    /// What kind of message this is: `message`, `reply` (a turn's final text), `error`,
    /// `interrupted` or `schedule` (posted by a [`Schedule`]).
    pub kind: &'static str,
    /// Body.
    pub text: String,
    /// Attached images, by path; read when the receiver takes the mail.
    pub images: Vec<PathBuf>,
}

impl Mail {
    /// The `<message>` block the receiving model sees (attached images follow as image blocks).
    pub fn render(&self) -> String {
        let attached: String = self
            .images
            .iter()
            .map(|path| format!("\n[attached image: {}]", path.display()))
            .collect();
        format!(
            "<message from=\"{}\" kind=\"{}\">\n{}{attached}\n</message>",
            self.from_label, self.kind, self.text
        )
    }
}

/// The Claude version a tier alias (`opus`, `sonnet`, `haiku`) stands for.
const TIER_VERSION: &str = "5-5";

/// A model id as given to `spawn_agent` or `/model`: a tier alias (`sonnet`) becomes that tier's
/// id with the provider prefix of `default` (`bedrock/global.anthropic.claude-sonnet-5-5` for a
/// Bedrock default) instead of reaching the provider as an id it rejects at the agent's first
/// call. Everything else passes as given.
pub fn expand_model(model: &str, default: Option<&str>) -> String {
    let tier = model.trim().to_ascii_lowercase();
    if !["opus", "sonnet", "haiku"].contains(&tier.as_str()) {
        return model.to_owned();
    }
    let prefix = default
        .and_then(|default| default.find("claude").map(|at| &default[..at]))
        .unwrap_or("bedrock/global.anthropic.");
    format!("{prefix}claude-{tier}-{TIER_VERSION}")
}

/// Render a batch of mail as one text block.
pub fn render_mail(mail: &[Mail]) -> String {
    mail.iter().map(Mail::render).collect::<Vec<_>>().join("\n\n")
}

/// Everything the factory needs to build one agent.
#[derive(Clone)]
pub struct AgentSpec {
    /// Swarm-unique id (`a1`, `a2`, …); also the agent's `id`.
    pub id: String,
    /// Swarm-unique human-readable name, chosen by the spawner or derived from the task.
    pub name: String,
    /// Working directory for shell commands and the environment context.
    pub cwd: PathBuf,
    /// Model override chosen by the spawner (`provider/name` or a Bedrock id).
    pub model: Option<String>,
    /// Swarm instructions to append to the agent's system prompt.
    pub instructions: String,
    /// The swarm tools bound to this agent; the factory must register them.
    pub tools: Vec<ToolRef>,
    /// Completion criterion: the factory should keep the agent working until a judge accepts it.
    pub goal: Option<String>,
    /// Where the factory should persist this agent's conversation (session id = agent id), so
    /// it survives restarts. `None` for an unsaved run.
    pub session_dir: Option<PathBuf>,
    /// Environment variables for the agent's commands, on top of the swarm's own.
    pub env: BTreeMap<String, String>,
    /// Where the agent's context pipeline reports each summary of older turns
    /// ([`SwarmEvent::Compacted`]); a factory without summarization may ignore it.
    pub on_compacted: Option<crate::compaction::CompactedSink>,
}

/// Builds the [`Agent`] for an [`AgentSpec`].
pub type AgentFactory = Arc<dyn Fn(AgentSpec) -> BoxFuture<'static, Result<Agent>> + Send + Sync>;

/// Resolves a model id (`None`: the swarm default) into a model, configured the way the factory
/// would configure it; used by [`Swarm::set_model`] to switch a live agent.
pub type ModelResolver =
    Arc<dyn Fn(Option<String>) -> BoxFuture<'static, Result<Arc<dyn Model>>> + Send + Sync>;

/// Swarm-wide settings.
#[derive(Clone)]
pub struct SwarmConfig {
    /// Builds each agent.
    pub factory: AgentFactory,
    /// Working directory of the root agent.
    pub cwd: PathBuf,
    /// Upper bound on live (non-cancelled) agents that have started, root included; spawns
    /// beyond it are queued and start, in spawn order, as agents are cancelled.
    pub max_agents: usize,
    /// Upper bound on model calls in flight across the swarm; further calls queue. `0` means no
    /// limit.
    pub max_concurrent_calls: usize,
    /// Swarm-wide token budget (input + output over all agents); once spent, model calls fail
    /// until [`Swarm::set_budget`] raises it. A resumed run keeps the budget it was saved with.
    pub budget_tokens: Option<u64>,
    /// Run id, used in worktree paths and branch names.
    pub run_id: String,
    /// Run directory for state and sessions; `None` keeps everything in memory.
    pub state_dir: Option<PathBuf>,
    /// The swarm's shared `CARGO_TARGET_DIR`, if any. Agents in their own worktree get a private
    /// one beside it: cargo would otherwise take another worktree's build as fresh (artifact
    /// hashes do not depend on the checkout path) and test the wrong code.
    pub cargo_target_dir: Option<PathBuf>,
    /// Resolves model ids for [`Swarm::set_model`]; `None` disables model switching.
    pub models: Option<ModelResolver>,
    /// Reclaim the disk of cancelled worktree agents ([`Swarm::gc`]). Off for a swarm that does
    /// not own its agents' directories, such as a self-test of a copied run.
    pub reclaim_disk: bool,
    /// The user's memory file, shown to every agent and written by `remember`/`forget`; `None`
    /// disables memory.
    pub memory: Option<Arc<Memory>>,
    /// Model for every agent's side calls (the goal judge, context summarization); `None` uses the
    /// agent's own model. Side calls are gated and counted like the agent's own calls either way.
    pub side_model: Option<Arc<dyn Model>>,
    /// Input size (tokens) at which the factory summarizes older turns
    /// ([`crate::compaction::compaction_point`]); the context share in [`SwarmEvent::Usage`] is
    /// relative to it.
    pub compact_at: u64,
    /// Longest an idle-mode reload waits for in-flight tool calls before it restarts anyway.
    pub reload_idle_timeout: Duration,
    /// The commit this binary was built from ([`crate::version::COMMIT`]), saved with the state;
    /// on resume under another commit the swarm reports what changed ([`SwarmEvent::Reloaded`]).
    pub commit: Option<String>,
    /// The git checkout where the commits between two builds are looked up.
    pub source: PathBuf,
    /// Branch [`Swarm::status`] compares agents' branches against; `None` uses the branch checked
    /// out at the root agent's working directory.
    pub status_base: Option<String>,
    /// Offer the `land` tool ([`Swarm::land`]); off by default.
    pub allow_land: bool,
    /// Globs (`*`, `?`) of further branches [`Swarm::land`] may fast-forward besides the root
    /// agent's checked-out branch (`--land-into`).
    pub land_into: Vec<String>,
    /// Commands [`Swarm::land`] runs when the caller names no checks (`--gate`).
    pub gates: Vec<String>,
    /// The `strands-swarm` binary that carries out a landing (`--land PLAN`); `None` is this
    /// process's own executable.
    pub exe: Option<PathBuf>,
    /// Prices for [`Swarm::cost`] estimates (`--prices`).
    pub prices: Arc<crate::cost::Prices>,
    /// Model [`Swarm::cost`] prices an agent's tokens as when it has no model of its own and they
    /// predate per-model tracking.
    pub default_model: Option<String>,
    /// Keep the prompt cache warm while a turn waits on a tool call ([`KEEP_WARM_EVERY`] for the
    /// 5-minute cache); `None` is off.
    pub keep_warm: Option<Duration>,
}

impl SwarmConfig {
    /// Defaults: 16 live agents, 8 concurrent model calls and a random 8-character run id.
    pub fn new(factory: AgentFactory, cwd: impl Into<PathBuf>) -> Self {
        Self {
            factory,
            cwd: cwd.into(),
            max_agents: 16,
            max_concurrent_calls: 8,
            budget_tokens: None,
            run_id: uuid::Uuid::new_v4().simple().to_string()[..8].to_owned(),
            state_dir: None,
            cargo_target_dir: None,
            models: None,
            reclaim_disk: true,
            memory: None,
            side_model: None,
            compact_at: crate::compaction::DEFAULT_COMPACT_AT,
            reload_idle_timeout: RELOAD_IDLE_TIMEOUT,
            commit: crate::version::COMMIT.map(str::to_owned),
            source: PathBuf::from(crate::version::SOURCE),
            status_base: None,
            allow_land: false,
            land_into: Vec::new(),
            gates: Vec::new(),
            exe: None,
            prices: Arc::default(),
            default_model: None,
            keep_warm: None,
        }
    }
}

/// What a `land` call asks for ([`Swarm::land`]).
#[derive(Debug, Clone, Default)]
pub struct LandRequest {
    /// Branch to land: the caller's own (the default) or that of an agent below it.
    pub branch: Option<String>,
    /// Branch to fast-forward; default: the branch the caller's root agent has checked out. Any
    /// other must match [`SwarmConfig::land_into`].
    pub into: Option<String>,
    /// Shell commands that must pass on the rebased commits; `None` uses
    /// [`SwarmConfig::gates`].
    pub check: Option<Vec<String>>,
}

/// What a `spawn_agent` call (or the user) asks for.
#[derive(Debug, Clone, Default)]
pub struct SpawnRequest {
    /// The task, sent as the agent's first user message.
    pub task: String,
    /// Short human-readable name; derived from the task when `None`. Made unique with a suffix.
    pub name: Option<String>,
    /// Give the agent its own git worktree and branch, cut from the spawner's `HEAD`.
    pub worktree: bool,
    /// Optional model override.
    pub model: Option<String>,
    /// Optional completion criterion; the agent keeps working until it is met.
    pub goal: Option<String>,
}

/// What a `schedule` call (or the user) asks for; see [`Swarm::schedule`].
#[derive(Debug, Clone, Default)]
pub struct ScheduleRequest {
    /// Id to create or replace; `None` picks the next free `sN`.
    pub id: Option<String>,
    /// Receiving agent id.
    pub to: String,
    /// Text to post.
    pub message: String,
    /// Repeat interval (at least [`schedules::MIN_EVERY`]).
    pub every: Option<Duration>,
    /// First (or only) fire; default one `every` from now.
    pub at: Option<DateTime<Utc>>,
}

/// A snapshot of one agent, for `list_agents` and the UI.
#[derive(Debug, Clone)]
pub struct AgentInfo {
    /// Id.
    pub id: String,
    /// Swarm-unique name.
    pub name: String,
    /// Parent id ([`USER`] for the root).
    pub parent: String,
    /// Working directory.
    pub cwd: PathBuf,
    /// Branch, when the agent has its own worktree.
    pub branch: Option<String>,
    /// Status.
    pub status: AgentStatus,
    /// Unread messages.
    pub unread: usize,
    /// Model in use: the live agent's model id, else its override (`None`: the default).
    pub model: Option<String>,
}

/// Everything observable that happens in the swarm, in order.
#[derive(Debug, Clone)]
pub enum SwarmEvent {
    /// An agent was registered.
    Spawned {
        /// Id.
        id: String,
        /// Parent id.
        parent: String,
        /// Swarm-unique name.
        name: String,
        /// Task.
        task: String,
        /// Working directory.
        cwd: PathBuf,
        /// Own branch, if any.
        branch: Option<String>,
    },
    /// An agent changed status.
    Status {
        /// Id.
        id: String,
        /// New status.
        status: AgentStatus,
    },
    /// Streamed assistant text.
    Text {
        /// Id.
        id: String,
        /// Text delta.
        delta: String,
    },
    /// A tool call started.
    ToolStarted {
        /// Id.
        id: String,
        /// Tool name.
        name: String,
        /// One-line summary of the input.
        summary: String,
    },
    /// A tool call finished.
    ToolFinished {
        /// Id.
        id: String,
        /// Tool name.
        name: String,
        /// Whether it succeeded.
        ok: bool,
    },
    /// A message was posted to a mailbox (`to` may be [`USER`]).
    Mail {
        /// Receiver id.
        to: String,
        /// The message.
        mail: Mail,
    },
    /// An agent's model was set: at construction and on every [`Swarm::set_model`].
    Model {
        /// Id.
        id: String,
        /// Model id.
        model: String,
    },
    /// Cumulative token usage of an agent.
    Usage {
        /// Id.
        id: String,
        /// Input tokens so far, cache reads and writes included.
        input_tokens: u64,
        /// Output tokens so far.
        output_tokens: u64,
        /// The part of `input_tokens` read from the prompt cache.
        cache_read_tokens: u64,
        /// The part of `input_tokens` written to the prompt cache.
        cache_write_tokens: u64,
        /// The last model call's input as a share of the compaction point (`0..1+`; at 1.0 older
        /// turns are summarized before the next call); `None` before the first model call.
        context: Option<f64>,
    },
    /// An agent's older turns were summarized before a model call.
    Compacted {
        /// Id.
        id: String,
        /// Projected input tokens before.
        before: u64,
        /// Estimated input tokens after.
        after: u64,
    },
    /// Summarizing an agent's older turns failed; its context was left as it was.
    CompactFailed {
        /// Id.
        id: String,
        /// Projected input tokens, sent unsummarized.
        tokens: u64,
        /// The summarizer's error.
        error: String,
    },
    /// The token budget was set or lifted (`None`).
    Budget {
        /// New budget.
        tokens: Option<u64>,
    },
    /// An agent started or stopped waiting for a free model-call slot
    /// ([`SwarmConfig::max_concurrent_calls`]).
    ModelQueued {
        /// Id.
        id: String,
        /// `true` while it waits.
        queued: bool,
    },
    /// A tool result carried an image (e.g. `read` on a png).
    Image {
        /// Id of the agent whose tool returned it.
        id: String,
        /// What it came from: the tool input's summary, usually a path.
        source: String,
        /// Encoded image bytes.
        bytes: Arc<Vec<u8>>,
    },
    /// An agent took all of its unread mail, as a new turn or appended to a tool result. Every
    /// [`SwarmEvent::Mail`] to it that came before has now been read.
    MailRead {
        /// Id.
        id: String,
    },
    /// Disk of cancelled worktree agents was reclaimed (or kept, with the reason): after an agent
    /// stopped, on resume, or for [`Swarm::gc`] (`requested`, sent even when nothing was found).
    Reclaimed {
        /// One entry per agent looked at.
        collected: Vec<Collected>,
        /// From [`Swarm::gc`] rather than the automatic pass.
        requested: bool,
    },
    /// An idle-mode reload is installed and waits for this many tool calls to finish.
    ReloadPending {
        /// Tool calls in flight ([`Swarm::tool_calls_in_flight`]).
        tool_calls: usize,
    },
    /// The run was resumed under a binary built from another commit than the one that saved it.
    Reloaded {
        /// The commit that saved the run.
        from: String,
        /// This binary's commit.
        to: String,
        /// `<short hash> <subject>` of the commits in `to` but not `from`, newest first
        /// ([`crate::version::changes`]); empty when unknown.
        changes: Vec<String>,
    },
    /// The tail of a resumed agent's restored conversation, so a viewer can show its past.
    History {
        /// Id.
        id: String,
        /// The last [`HISTORY_MESSAGES`] messages, oldest first.
        messages: Vec<Message>,
        /// Earlier messages left out.
        omitted: usize,
    },
}

/// What the parent of an interrupted agent is told.
pub const INTERRUPTED_TEXT: &str = "Interrupted by the user before finishing this turn.";

/// Sender of the message that tells an agent its background job ended.
pub const JOBS_SENDER: &str = "jobs";

/// Sender of the supervisor's notes (a rollback, a crash loop) from [`SUPERVISOR_NOTICE`].
pub const SUPERVISOR_SENDER: &str = "supervisor";

/// File in the run directory where `swarmctl` leaves notes for the swarm's next start.
pub const SUPERVISOR_NOTICE: &str = "supervisor.notice";

/// Most bytes of [`SUPERVISOR_NOTICE`] delivered (its tail).
const SUPERVISOR_NOTICE_MAX: usize = 4000;

/// How many restored messages [`SwarmEvent::History`] carries at most.
pub const HISTORY_MESSAGES: usize = 100;

/// How a [`Swarm::wait`] ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaitOutcome {
    /// Mail arrived; all of it was taken.
    Mail(Vec<Mail>),
    /// The wait or the agent was cancelled.
    Cancelled,
    /// No awaited agent is left to write; the notice says why.
    Unreachable(String),
}

struct Node {
    id: String,
    name: String,
    parent: String,
    cwd: PathBuf,
    branch: Option<String>,
    status: Mutex<AgentStatus>,
    mailbox: Mutex<VecDeque<Mail>>,
    notify: Notify,
    cancel: CancellationToken,
    /// The built agent while its actor runs; released once it is cancelled and stopped, so a long
    /// run does not keep every finished agent's conversation in memory.
    agent: Mutex<Option<Agent>>,
    /// The model id the agent last used, kept once the agent is released.
    last_model: Mutex<Option<String>>,
    task: String,
    /// Model override; `None` follows the swarm default.
    model: Mutex<Option<String>>,
    goal: Option<String>,
    env: BTreeMap<String, String>,
    /// Tokens used so far, across restarts.
    usage: Mutex<Tokens>,
    /// Tokens by model and recent calls, for cost estimates; across restarts.
    ledger: Mutex<crate::cost::Ledger>,
    /// The actor has exited (or never ran: restored cancelled).
    finished: AtomicBool,
    /// Background disk reclamation has been started for this agent.
    collected: AtomicBool,
    /// What the last reclamation of this agent did.
    reclaimed: Mutex<Option<Collected>>,
    /// Context window utilization of the last model call, across restarts.
    context: Mutex<Option<f64>>,
    /// Cancels the turn in progress only ([`Swarm::interrupt`]); a child of `cancel`.
    turn: Mutex<Option<CancellationToken>>,
    /// Interrupted by the user and not messaged since: a goal agent does not resume on restart.
    paused: AtomicBool,
    /// Tool calls of the current turn in flight, by tool use id, with when each started; waiting
    /// tools ([`WAITING_TOOLS`]) left out.
    tool_calls: Mutex<BTreeMap<String, Instant>>,
}

/// Cumulative `(input, output)` tokens.
#[derive(Debug, Clone, Copy, Default)]
struct Tokens {
    /// All input, cache reads and writes included.
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
}

impl Node {
    /// The current turn's token, else the agent's own (between turns).
    fn turn_token(&self) -> CancellationToken {
        self.turn.lock().clone().unwrap_or_else(|| self.cancel.clone())
    }

    /// `parser (a2)`.
    fn label(&self) -> String {
        format!("{} ({})", self.name, self.id)
    }

    fn info(&self) -> AgentInfo {
        AgentInfo {
            id: self.id.clone(),
            name: self.name.clone(),
            parent: self.parent.clone(),
            cwd: self.cwd.clone(),
            branch: self.branch.clone(),
            status: self.status.lock().clone(),
            unread: self.mailbox.lock().len(),
            model: self.model_id(),
        }
    }

    /// The live agent's model id, else the one it last used, else the override.
    fn model_id(&self) -> Option<String> {
        self.agent()
            .and_then(|agent| agent.model().model_id())
            .or_else(|| self.last_model.lock().clone())
            .or_else(|| self.model.lock().clone())
    }

    /// The built agent, while it is held.
    fn agent(&self) -> Option<Agent> {
        self.agent.lock().clone()
    }

    /// Drop the agent of a cancelled node whose actor has stopped, keeping its model id.
    fn release_agent(&self) {
        if *self.status.lock() != AgentStatus::Cancelled {
            return;
        }
        if let Some(agent) = self.agent.lock().take() {
            *self.last_model.lock() = agent.model().model_id();
        }
    }
}

struct Inner {
    config: SwarmConfig,
    agents: Mutex<IndexMap<String, Arc<Node>>>,
    next_id: AtomicU64,
    events: mpsc::UnboundedSender<SwarmEvent>,
    root_cancel: CancellationToken,
    persister: Option<Arc<Persister>>,
    journal: Option<Journal>,
    generation: AtomicU64,
    shutting_down: AtomicBool,
    actors: Mutex<Vec<tokio::task::JoinHandle<()>>>,
    reload: CancellationToken,
    /// Bumped by every installed reload; a pending idle-mode reload yields to a newer one.
    reload_generation: AtomicU64,
    /// An idle-mode reload is pending: agents' model calls wait for the restart
    /// ([`Swarm::hold_for_reload`]).
    reload_held: AtomicBool,
    model_calls: Option<Arc<Semaphore>>,
    budget: Mutex<Option<u64>>,
    /// Live-agent limit set at runtime with [`Swarm::set_max_agents`]; overrides the config.
    max_agents: Mutex<Option<usize>>,
    /// Agents whose model call was refused for lack of budget, to wake when it is raised.
    over_budget: Mutex<Vec<String>>,
    /// Pending scheduled messages, in creation order.
    schedules: Mutex<Vec<Schedule>>,
    /// Wakes the scheduler when schedules change.
    schedules_changed: Notify,
    /// The scheduler task has been started (on the first schedule).
    scheduler: AtomicBool,
    /// Wall clock and tokio instant at construction: [`Swarm::now`] follows tokio's clock from
    /// here, so paused-time tests can fast-forward schedules.
    epoch: (DateTime<Utc>, tokio::time::Instant),
    /// Branch commits for [`Swarm::status`], refreshed in the background.
    branches: Mutex<crate::status::BranchCache>,
    /// Background shell jobs ([`Swarm::start_job`]).
    jobs: Jobs,
    /// The job watcher has been started.
    watching_jobs: AtomicBool,
}

/// Handle to a running swarm. Cheap to clone.
#[derive(Clone)]
pub struct Swarm {
    inner: Arc<Inner>,
}

impl Swarm {
    /// A new, empty swarm and the receiver of its event stream.
    pub fn new(config: SwarmConfig) -> (Self, mpsc::UnboundedReceiver<SwarmEvent>) {
        let (events, rx) = mpsc::unbounded_channel();
        let inner = Inner {
            agents: Mutex::new(IndexMap::new()),
            next_id: AtomicU64::new(1),
            events,
            root_cancel: CancellationToken::new(),
            persister: config.state_dir.clone().map(|dir| Arc::new(Persister::new(dir))),
            journal: config.state_dir.clone().map(Journal::open),
            generation: AtomicU64::new(0),
            shutting_down: AtomicBool::new(false),
            actors: Mutex::new(Vec::new()),
            reload: CancellationToken::new(),
            reload_generation: AtomicU64::new(0),
            reload_held: AtomicBool::new(false),
            model_calls: (config.max_concurrent_calls > 0)
                .then(|| Arc::new(Semaphore::new(config.max_concurrent_calls))),
            budget: Mutex::new(config.budget_tokens),
            max_agents: Mutex::new(None),
            over_budget: Mutex::new(Vec::new()),
            schedules: Mutex::new(Vec::new()),
            schedules_changed: Notify::new(),
            scheduler: AtomicBool::new(false),
            epoch: (Utc::now(), tokio::time::Instant::now()),
            branches: Mutex::default(),
            jobs: Jobs::new(jobs_dir(&config)),
            watching_jobs: AtomicBool::new(false),
            config,
        };
        let swarm = Self {
            inner: Arc::new(inner),
        };
        if let Some(tokens) = swarm.budget() {
            swarm.emit(SwarmEvent::Budget { tokens: Some(tokens) });
        }
        (swarm, rx)
    }

    /// The run id.
    pub fn run_id(&self) -> &str {
        &self.inner.config.run_id
    }

    /// Snapshot of every agent, in spawn order.
    pub fn agents(&self) -> Vec<AgentInfo> {
        self.inner
            .agents
            .lock()
            .values()
            .map(|node| node.info())
            .collect()
    }

    /// Snapshot of one agent.
    pub fn agent(&self, id: &str) -> Option<AgentInfo> {
        self.node(id).map(|node| node.info())
    }

    /// Whether any agent is mid-turn, starting, or has unread mail. Queued agents are not busy:
    /// they only start once something else frees a slot.
    pub fn is_busy(&self) -> bool {
        self.agents().iter().any(|info| match info.status {
            AgentStatus::Running | AgentStatus::Starting => true,
            AgentStatus::Queued | AgentStatus::Cancelled => false,
            AgentStatus::Idle | AgentStatus::Errored(_) => info.unread > 0,
        })
    }

    /// Spawn an agent under `parent` ([`USER`] for a root agent). Returns its id. At the agent
    /// limit the agent is [`AgentStatus::Queued`] instead, and starts once a slot frees.
    pub async fn spawn(&self, parent: &str, mut request: SpawnRequest) -> Result<String> {
        let parent_node = if parent == USER {
            None
        } else {
            Some(self.known(parent)?)
        };
        let default = self.inner.config.default_model.as_deref();
        request.model = request.model.map(|model| expand_model(&model, default));

        let id = format!("a{}", self.inner.next_id.fetch_add(1, Ordering::Relaxed));
        let base_cwd = parent_node
            .as_ref()
            .map_or_else(|| self.inner.config.cwd.clone(), |node| node.cwd.clone());
        let (cwd, branch, env) = match request.worktree {
            true => {
                let created = worktree::create(&base_cwd, &self.inner.config.run_id, &id).await?;
                (created.cwd, Some(created.branch), self.worktree_env(&id))
            }
            false => {
                let env = parent_node.as_ref().map(|node| node.env.clone());
                (base_cwd, None, env.unwrap_or_default())
            }
        };
        let cancel = parent_node
            .as_ref()
            .map_or(&self.inner.root_cancel, |node| &node.cancel)
            .child_token();

        let scheduled = self.scheduled_targets();
        let node = {
            let mut agents = self.inner.agents.lock();
            // Decided under the lock, so concurrent spawns cannot overshoot the limit.
            let queued = started(&agents, &scheduled) >= self.max_agents();
            let taken = agents.values().map(|node| node.name.as_str());
            let node = Arc::new(Node {
                id: id.clone(),
                name: names::choose(request.name.as_deref(), &request.task, taken),
                parent: parent.to_owned(),
                cwd,
                branch,
                status: Mutex::new(match queued {
                    true => AgentStatus::Queued,
                    false => AgentStatus::Starting,
                }),
                mailbox: Mutex::new(VecDeque::new()),
                notify: Notify::new(),
                cancel,
                agent: Mutex::new(None),
                last_model: Mutex::new(None),
                task: request.task.clone(),
                model: Mutex::new(request.model.clone()),
                goal: request.goal.clone(),
                env,
                usage: Mutex::new(Tokens::default()),
                ledger: Mutex::default(),
                // No actor runs until a queued agent starts.
                finished: AtomicBool::new(queued),
                collected: AtomicBool::new(false),
                reclaimed: Mutex::new(None),
                context: Mutex::new(None),
                turn: Mutex::new(None),
                paused: AtomicBool::new(false),
                tool_calls: Mutex::new(BTreeMap::new()),
            });
            agents.insert(id.clone(), node.clone());
            node
        };
        self.persist();
        self.emit(SwarmEvent::Spawned {
            id: id.clone(),
            parent: parent.to_owned(),
            name: node.name.clone(),
            task: request.task.clone(),
            cwd: node.cwd.clone(),
            branch: node.branch.clone(),
        });
        if let (USER, Some(journal)) = (parent, &self.inner.journal) {
            journal.mail(USER, USER, &id, &node.label(), "task", &request.task);
        }

        let status = node.status.lock().clone();
        match status {
            AgentStatus::Queued => self.emit(SwarmEvent::Status {
                id: id.clone(),
                status,
            }),
            _ => self.start_actor(node, Start::Fresh(request.task)),
        }
        Ok(id)
    }

    /// Stop `id`'s current turn, like Esc in Claude Code, without cancelling it: the agent
    /// becomes idle with its conversation intact (an open tool call is closed as cancelled), its
    /// parent gets an `interrupted` message, and its children keep running. A goal agent stays
    /// idle until its next message. Errors when the agent is not in a turn.
    pub fn interrupt(&self, id: &str) -> Result<()> {
        let node = self.known(id)?;
        let turn = node.turn.lock().clone();
        match turn {
            Some(turn) if node.status.lock().is_live() => {
                turn.cancel();
                Ok(())
            }
            _ => Err(Error::other(format!(
                "{} is not working on a turn.",
                node.label()
            ))),
        }
    }

    /// Cancel an agent and, through its token, every agent it spawned.
    pub fn cancel(&self, id: &str) -> Result<()> {
        let node = self.known(id)?;
        node.cancel.cancel();
        // Recorded right away for the whole subtree, so capacity, mail and `list_agents` reflect
        // it even while an actor is still starting or finishing a tool call.
        for node in self.subtree(&node.id) {
            if let Some(agent) = node.agent() {
                agent.cancel();
            }
            self.set_status(&node, AgentStatus::Cancelled);
        }
        self.kill_jobs_of(self.subtree(&node.id).iter().map(|n| n.id.clone()).collect());
        Ok(())
    }

    /// `id` and all of its descendants, parents first.
    fn subtree(&self, id: &str) -> Vec<Arc<Node>> {
        let agents = self.inner.agents.lock();
        let mut out: Vec<Arc<Node>> = agents.get(id).cloned().into_iter().collect();
        // Agents are kept in spawn order, so every parent comes before its children.
        for node in agents.values() {
            if out.iter().any(|parent| parent.id == node.parent) {
                out.push(node.clone());
            }
        }
        out
    }

    /// Switch agent `id` to `model` (`None`: the swarm default), effective from its next model
    /// call, mid-turn included. The model is resolved first, so an invalid id is rejected and the
    /// agent keeps its current model. The choice is persisted, so a resumed run keeps it. Returns
    /// the new model id.
    pub async fn set_model(&self, id: &str, model: Option<String>) -> Result<String> {
        let node = self.known(id)?;
        let resolver = self.inner.config.models.clone().ok_or_else(|| {
            Error::other("Model switching is not available: no model resolver is configured.")
        })?;
        if !node.status.lock().is_live() {
            return Err(Error::other(format!("Agent {id} is cancelled.")));
        }
        let agent = node
            .agent()
            .ok_or_else(|| Error::other(format!("Agent {id} is still starting; try again in a moment.")))?;
        let default = self.inner.config.default_model.as_deref();
        let model = model.map(|model| expand_model(&model, default));
        let wanted = model.as_deref().unwrap_or("the default model");
        let resolved = resolver(model.clone())
            .await
            .map_err(|error| Error::other(format!("Cannot switch {id} to {wanted}: {error}")))?;
        agent.set_model(resolved);
        *node.model.lock() = model;
        self.persist();
        Ok(self.emit_model(&node))
    }

    fn emit_model(&self, node: &Node) -> String {
        let model = node.model_id().unwrap_or_else(|| "default".to_owned());
        self.emit(SwarmEvent::Model {
            id: node.id.clone(),
            model: model.clone(),
        });
        model
    }

    /// The user's memory, when enabled.
    pub fn memory(&self) -> Option<&Arc<Memory>> {
        self.inner.config.memory.as_ref()
    }

    /// Add `text` to the user's memory as a bullet dated today. Every agent sees it from its next
    /// turn on. Returns the line written.
    pub async fn remember(&self, text: &str) -> Result<String> {
        self.require_memory()?
            .remember(text, &crate::journal::today())
            .await
    }

    /// Drop every memory line containing `text` (case-insensitive). Returns the removed lines.
    pub async fn forget(&self, text: &str) -> Result<Vec<String>> {
        self.require_memory()?.forget(text).await
    }

    fn require_memory(&self) -> Result<&Arc<Memory>> {
        self.memory()
            .ok_or_else(|| Error::other("Memory is off (the swarm was started with --no-memory)."))
    }

    /// Put the memory file's current contents at the end of `agent`'s system prompt.
    async fn refresh_memory(&self, agent: &Agent) {
        let Some(memory) = self.memory() else {
            return;
        };
        let section = memory.section().await;
        let current = agent.system_prompt();
        let updated = crate::memory::with_section(current.clone(), section.as_deref());
        if updated != current {
            agent.set_system_prompt(updated);
        }
    }

    /// Stop every agent without recording them as cancelled, after saving the final state, so
    /// the run can be resumed.
    pub async fn shutdown(&self) {
        let generation = self.inner.generation.fetch_add(1, Ordering::SeqCst) + 1;
        self.inner.shutting_down.store(true, Ordering::SeqCst);
        if let Some(persister) = &self.inner.persister {
            if let Err(error) = persister.write(generation, &self.snapshot()).await {
                tracing::warn!(%error, "final state save failed");
            }
        }
        self.inner.root_cancel.cancel();
        let actors: Vec<_> = self.inner.actors.lock().drain(..).collect();
        let stopped = futures::future::join_all(actors);
        if tokio::time::timeout(SHUTDOWN_GRACE, stopped).await.is_err() {
            tracing::warn!("agents still running after the shutdown grace period");
        } else {
            self.collect_on_exit().await;
        }
        if let Some(journal) = &self.inner.journal {
            journal.event("shutdown", serde_json::json!({}));
            journal.flush().await;
        }
    }

    fn track(&self, handle: tokio::task::JoinHandle<()>) {
        let mut actors = self.inner.actors.lock();
        actors.retain(|actor| !actor.is_finished());
        actors.push(handle);
    }

    /// Environment for a new worktree agent: a private `CARGO_TARGET_DIR` when the swarm has a
    /// shared one.
    fn worktree_env(&self, id: &str) -> BTreeMap<String, String> {
        let Some(shared) = &self.inner.config.cargo_target_dir else {
            return BTreeMap::new();
        };
        let private = PathBuf::from(format!("{}-worktrees", shared.display()))
            .join(&self.inner.config.run_id)
            .join(id);
        BTreeMap::from([(
            "CARGO_TARGET_DIR".to_owned(),
            private.to_string_lossy().into_owned(),
        )])
    }

    /// The id of the agent `who` names: an id, or else a unique agent name.
    pub fn resolve(&self, who: &str) -> Result<String> {
        let agents = self.inner.agents.lock();
        if agents.contains_key(who) {
            return Ok(who.to_owned());
        }
        let matches: Vec<&str> = agents
            .values()
            .filter(|node| node.name == who)
            .map(|node| node.id.as_str())
            .collect();
        match matches.as_slice() {
            [id] => Ok((*id).to_owned()),
            [] => Err(Error::other(format!(
                "No agent with id or name `{who}`; call list_agents to see who exists."
            ))),
            ids => Err(Error::other(format!(
                "Several agents are named `{who}` ({}); use an id.",
                ids.join(", ")
            ))),
        }
    }

    /// `parser (a2)` for an agent, `id` itself for anyone else (the user).
    pub fn label(&self, id: &str) -> String {
        self.node(id).map_or_else(|| id.to_owned(), |node| node.label())
    }

    fn node(&self, id: &str) -> Option<Arc<Node>> {
        self.inner.agents.lock().get(id).cloned()
    }

    /// [`Swarm::node`], or the error tools report for an unknown id.
    fn known(&self, id: &str) -> Result<Arc<Node>> {
        self.node(id)
            .ok_or_else(|| Error::other(format!("Unknown agent id {id}.")))
    }

    /// Every node in spawn order, copied out so per-agent locks are not taken under the agents
    /// lock.
    fn nodes(&self) -> Vec<Arc<Node>> {
        self.inner.agents.lock().values().cloned().collect()
    }

    fn emit(&self, event: SwarmEvent) {
        if let Some(journal) = &self.inner.journal {
            journal.record(&event);
        }
        let _ = self.inner.events.send(event);
    }

    /// Record `event` in the journal, if the run keeps one.
    fn journal_event(&self, event: &str, fields: serde_json::Value) {
        if let Some(journal) = &self.inner.journal {
            journal.event(event, fields);
        }
    }

    /// Record a status change; `Cancelled` is terminal and never overwritten.
    fn set_status(&self, node: &Node, status: AgentStatus) {
        {
            let mut current = node.status.lock();
            if *current == AgentStatus::Cancelled {
                return;
            }
            *current = status.clone();
        }
        // A turn that ended or was cancelled has nothing in flight, finished events or not.
        if status != AgentStatus::Running {
            node.tool_calls.lock().clear();
        }
        if status == AgentStatus::Cancelled {
            // Waits on this agent may now be pointless; let them re-check.
            self.nodes().iter().for_each(|node| node.notify.notify_waiters());
        }
        self.persist();
        let freed = status == AgentStatus::Cancelled;
        self.emit(SwarmEvent::Status {
            id: node.id.clone(),
            status,
        });
        if freed {
            self.start_queued();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::expand_model;

    #[test]
    fn a_tier_alias_takes_the_defaults_provider_prefix() {
        let bedrock = Some("bedrock/global.anthropic.claude-haiku-5-5");
        assert_eq!(
            expand_model("sonnet", bedrock),
            "bedrock/global.anthropic.claude-sonnet-5-5"
        );
        assert_eq!(
            expand_model(" Opus ", Some("anthropic/claude-sonnet-4-5")),
            "anthropic/claude-opus-5-5"
        );
        let bedrock_haiku = "bedrock/global.anthropic.claude-haiku-5-5";
        assert_eq!(expand_model("haiku", Some("openai/gpt-5")), bedrock_haiku);
        assert_eq!(expand_model("haiku", None), bedrock_haiku);
    }

    #[test]
    fn other_ids_pass_as_given() {
        for id in ["us.anthropic.claude-opus-4-6-v1", "openai/gpt-5", "b"] {
            assert_eq!(expand_model(id, None), id);
        }
    }
}
