//! The swarm runtime: an agent registry, per-agent mailboxes, and the actor loop that drives each
//! agent turn by turn.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

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
use strands_agents::types::{
    ContentBlock, ImageBlock, ImageSource, Message, Role, ToolResultBlock, ToolResultContent,
    ToolResultStatus, Usage,
};
use strands_agents::{Agent, Error, Model, Result};
use tokio::sync::{mpsc, Notify, OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::budget;
use crate::gc::{self, Collected};
use crate::images;
use crate::journal::Journal;
use crate::memory::Memory;
use crate::names;
use crate::selftest;
use crate::state::{AgentState, MailState, Persister, SwarmState, SESSIONS_DIR};
use crate::tools::swarm_tools;
use crate::worktree;

const RETRY_INITIAL: Duration = Duration::from_secs(15);
const RETRY_MAX: Duration = Duration::from_secs(600);
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
    /// What kind of message this is: `message`, `reply` (a turn's final text) or `error`.
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
        }
    }
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

/// How many restored messages [`SwarmEvent::History`] carries at most.
pub const HISTORY_MESSAGES: usize = 100;

/// How an actor begins.
enum Start {
    /// A new agent: its first turn is the task.
    Fresh(String),
    /// A restored agent; `interrupted` when it was mid-turn (or starting) when the run stopped.
    Resume { interrupted: bool },
}

const RESTART_NOTICE: &str =
    "[The swarm was restarted. Your conversation was restored; any tool call that was \
in flight was interrupted. Check the state of your work and continue.]";

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
    agent: OnceLock<Agent>,
    task: String,
    /// Model override; `None` follows the swarm default.
    model: Mutex<Option<String>>,
    goal: Option<String>,
    env: BTreeMap<String, String>,
    /// Tokens used so far, across restarts.
    usage: Mutex<Tokens>,
    /// The actor has exited (or never ran: restored cancelled).
    finished: AtomicBool,
    /// Background disk reclamation has been started for this agent.
    collected: AtomicBool,
    /// Context window utilization of the last model call, across restarts.
    context: Mutex<Option<f64>>,
    /// Cancels the turn in progress only ([`Swarm::interrupt`]); a child of `cancel`.
    turn: Mutex<Option<CancellationToken>>,
    /// Interrupted by the user and not messaged since: a goal agent does not resume on restart.
    paused: AtomicBool,
    /// Tool calls of the current turn in flight, waiting tools ([`WAITING_TOOLS`]) left out.
    tool_calls: Mutex<usize>,
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

    /// The live agent's model id, else the override.
    fn model_id(&self) -> Option<String> {
        self.agent
            .get()
            .and_then(|agent| agent.model().model_id())
            .or_else(|| self.model.lock().clone())
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
    model_calls: Option<Arc<Semaphore>>,
    budget: Mutex<Option<u64>>,
    /// Agents whose model call was refused for lack of budget, to wake when it is raised.
    over_budget: Mutex<Vec<String>>,
}

/// An agent's auxiliary model (what side calls such as the goal judge and context summarization
/// run on): gated by the budget and the concurrency cap like the agent's own calls, with its usage
/// added to the agent's totals. Without a fixed model it follows the agent's current one, model
/// switches included.
struct SideModel {
    swarm: Swarm,
    node: Weak<Node>,
    model: Option<Arc<dyn Model>>,
}

impl SideModel {
    fn current(&self) -> Option<Arc<dyn Model>> {
        let host = || self.node.upgrade()?.agent.get().map(Agent::model);
        self.model.clone().or_else(host)
    }
}

impl Model for SideModel {
    fn model_id(&self) -> Option<String> {
        self.current()?.model_id()
    }

    fn context_window_limit(&self) -> Option<u64> {
        self.current()?.context_window_limit()
    }

    fn stream<'a>(&'a self, messages: &'a [Message], options: &'a StreamOptions) -> ModelEventStream<'a> {
        Box::pin(async_stream::try_stream! {
            let gone = || Error::other("The agent of this side call is gone.");
            let node = self.node.upgrade().ok_or_else(gone)?;
            let model = self.current().ok_or_else(gone)?;
            // Side calls (the goal judge) belong to the turn: an interrupt stops them too.
            let cancel = node.turn_token();
            let _permit = self.swarm.admit_model_call(&node.id, &cancel).await?;
            let mut events = model.stream(messages, options);
            loop {
                let next = tokio::select! {
                    next = events.next() => next,
                    _ = cancel.cancelled() => Some(Err(Error::Cancelled)),
                };
                let Some(event) = next else { break };
                let event = event?;
                if let ModelStreamEvent::Metadata { usage: Some(usage), .. } = &event {
                    self.swarm.add_usage(&node, usage, false);
                }
                yield event;
            }
        })
    }
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
            model_calls: (config.max_concurrent_calls > 0)
                .then(|| Arc::new(Semaphore::new(config.max_concurrent_calls))),
            budget: Mutex::new(config.budget_tokens),
            over_budget: Mutex::new(Vec::new()),
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
    pub async fn spawn(&self, parent: &str, request: SpawnRequest) -> Result<String> {
        let parent_node = self.node(parent);
        if parent != USER && parent_node.is_none() {
            return Err(Error::other(format!("Unknown agent id {parent}.")));
        }

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

        let node = {
            let mut agents = self.inner.agents.lock();
            // Decided under the lock, so concurrent spawns cannot overshoot the limit.
            let queued = started(&agents) >= self.inner.config.max_agents;
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
                agent: OnceLock::new(),
                task: request.task.clone(),
                model: Mutex::new(request.model.clone()),
                goal: request.goal.clone(),
                env,
                usage: Mutex::new(Tokens::default()),
                // No actor runs until a queued agent starts.
                finished: AtomicBool::new(queued),
                collected: AtomicBool::new(false),
                context: Mutex::new(None),
                turn: Mutex::new(None),
                paused: AtomicBool::new(false),
                tool_calls: Mutex::new(0),
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

    /// Start queued agents, oldest first, while there are free slots.
    fn start_queued(&self) {
        if self.inner.shutting_down.load(Ordering::SeqCst) {
            return;
        }
        let ready: Vec<Arc<Node>> = {
            let agents = self.inner.agents.lock();
            let free = self.inner.config.max_agents.saturating_sub(started(&agents));
            let mut ready = Vec::new();
            for node in agents.values() {
                if ready.len() == free {
                    break;
                }
                // Checked and claimed under the status lock, so a concurrent cancel wins.
                let mut status = node.status.lock();
                if *status == AgentStatus::Queued && !node.cancel.is_cancelled() {
                    *status = AgentStatus::Starting;
                    node.finished.store(false, Ordering::SeqCst);
                    ready.push(node.clone());
                }
            }
            ready
        };
        if ready.is_empty() {
            return;
        }
        self.persist();
        for node in ready {
            self.emit(SwarmEvent::Status {
                id: node.id.clone(),
                status: AgentStatus::Starting,
            });
            let task = node.task.clone();
            self.start_actor(node, Start::Fresh(task));
        }
    }

    /// Post `text` from `from` to `to`'s mailbox. `to` may be [`USER`].
    pub fn send(&self, from: &str, to: &str, text: impl Into<String>) -> Result<()> {
        self.post(from, to, "message", text.into())
    }

    /// Stop `id`'s current turn, like Esc in Claude Code, without cancelling it: the agent
    /// becomes idle with its conversation intact (an open tool call is closed as cancelled), its
    /// parent gets an `interrupted` message, and its children keep running. A goal agent stays
    /// idle until its next message. Errors when the agent is not in a turn.
    pub fn interrupt(&self, id: &str) -> Result<()> {
        let node = self
            .node(id)
            .ok_or_else(|| Error::other(format!("Unknown agent id {id}.")))?;
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

    /// [`Swarm::send`] with images attached. Only their paths are queued (and saved); each must
    /// pass [`images::check`] now, and is read when `to` takes the mail.
    pub fn send_with_images(
        &self,
        from: &str,
        to: &str,
        text: impl Into<String>,
        images: Vec<PathBuf>,
    ) -> Result<()> {
        for path in &images {
            images::check(path).map_err(Error::other)?;
        }
        self.post_mail(from, to, "message", text.into(), images)
    }

    /// Cancel an agent and, through its token, every agent it spawned.
    pub fn cancel(&self, id: &str) -> Result<()> {
        let node = self
            .node(id)
            .ok_or_else(|| Error::other(format!("Unknown agent id {id}.")))?;
        node.cancel.cancel();
        // Recorded right away for the whole subtree, so capacity, mail and `list_agents` reflect
        // it even while an actor is still starting or finishing a tool call.
        for node in self.subtree(&node.id) {
            if let Some(agent) = node.agent.get() {
                agent.cancel();
            }
            self.set_status(&node, AgentStatus::Cancelled);
        }
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
        let node = self
            .node(id)
            .ok_or_else(|| Error::other(format!("Unknown agent id {id}.")))?;
        let resolver = self.inner.config.models.clone().ok_or_else(|| {
            Error::other("Model switching is not available: no model resolver is configured.")
        })?;
        if !node.status.lock().is_live() {
            return Err(Error::other(format!("Agent {id} is cancelled.")));
        }
        let agent =
            node.agent.get().cloned().ok_or_else(|| {
                Error::other(format!("Agent {id} is still starting; try again in a moment."))
            })?;
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
        }
        if let Some(journal) = &self.inner.journal {
            journal.event("shutdown", serde_json::json!({}));
            journal.flush().await;
        }
    }

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
    fn reload_when_idle(&self, generation: u64) {
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

    /// Run `node`'s actor; once it exits, reclaim the disk of agents that are now fully stopped.
    fn start_actor(&self, node: Arc<Node>, start: Start) {
        let swarm = self.clone();
        self.track(tokio::spawn(async move {
            swarm.clone().run_actor(node.clone(), start).await;
            node.finished.store(true, Ordering::SeqCst);
            swarm.collect_in_background();
        }));
    }

    /// Cancelled worktree agents whose own actor and every descendant's have exited, so nothing
    /// still runs in their worktree or target dir. With `claim`, only those not claimed before.
    fn collectable(&self, claim: bool) -> Vec<Arc<Node>> {
        let owners: Vec<Arc<Node>> = {
            let agents = self.inner.agents.lock();
            agents
                .values()
                .filter(|node| node.branch.is_some() && *node.status.lock() == AgentStatus::Cancelled)
                .cloned()
                .collect()
        };
        owners
            .into_iter()
            .filter(|node| {
                self.subtree(&node.id)
                    .iter()
                    .all(|n| n.finished.load(Ordering::SeqCst))
            })
            .filter(|node| !claim || !node.collected.swap(true, Ordering::SeqCst))
            .collect()
    }

    /// Reclaim the disk of newly collectable agents without waiting (see [`Swarm::gc`]).
    fn collect_in_background(&self) {
        if self.inner.shutting_down.load(Ordering::SeqCst) {
            return;
        }
        for node in self.collectable(true) {
            let swarm = self.clone();
            tokio::spawn(async move {
                let collected = swarm.collect_node(&node).await;
                if !collected.is_empty() {
                    swarm.emit(SwarmEvent::Reclaimed {
                        collected: vec![collected],
                        requested: false,
                    });
                }
            });
        }
    }

    /// Reclaim the disk of every cancelled worktree agent that has stopped: delete its private
    /// `CARGO_TARGET_DIR`, and remove its worktree if it is clean. Branches are never deleted.
    /// Runs on its own whenever an agent stops and on resume; call it to retry (e.g. after
    /// committing in a kept worktree). The outcome is also published as
    /// [`SwarmEvent::Reclaimed`], which the journal records.
    pub async fn gc(&self) -> Vec<Collected> {
        let mut collected = Vec::new();
        for node in self.collectable(false) {
            collected.push(self.collect_node(&node).await);
        }
        self.emit(SwarmEvent::Reclaimed {
            collected: collected.clone(),
            requested: true,
        });
        collected
    }

    /// [`Swarm::gc`] without waiting; the outcome arrives as [`SwarmEvent::Reclaimed`].
    pub fn gc_in_background(&self) {
        let swarm = self.clone();
        tokio::spawn(async move { swarm.gc().await });
    }

    async fn collect_node(&self, node: &Node) -> Collected {
        if !self.inner.config.reclaim_disk {
            return Collected {
                id: node.id.clone(),
                ..Default::default()
            };
        }
        let run_id = &self.inner.config.run_id;
        gc::collect(run_id, &node.id, node.branch.as_deref(), &node.cwd, &node.env).await
    }

    fn track(&self, handle: tokio::task::JoinHandle<()>) {
        let mut actors = self.inner.actors.lock();
        actors.retain(|actor| !actor.is_finished());
        actors.push(handle);
    }

    /// Take all mail for `id`.
    pub fn drain(&self, id: &str) -> Vec<Mail> {
        let mail: Vec<Mail> = self
            .node(id)
            .map(|node| node.mailbox.lock().drain(..).collect())
            .unwrap_or_default();
        if !mail.is_empty() {
            self.persist();
            self.emit(SwarmEvent::MailRead { id: id.to_owned() });
        }
        mail
    }

    /// Wait until `id` has mail matching `filter` (or any mail, when `filter` is empty), then
    /// take all of its mail. `None` on cancellation.
    pub async fn wait_for_mail(
        &self,
        id: &str,
        filter: &[String],
        cancel: &CancellationToken,
    ) -> Option<Vec<Mail>> {
        let node = self.node(id)?;
        loop {
            let notified = node.notify.notified();
            if has_matching_mail(&node, filter) {
                return Some(self.drain(id));
            }
            tokio::select! {
                _ = notified => {}
                _ = cancel.cancelled() => return None,
                _ = node.cancel.cancelled() => return None,
            }
        }
    }

    /// Reopen a saved run: restore every agent with its conversation and unread mail, wake those
    /// that were mid-turn, and keep cancelled ones visible but inert.
    pub async fn resume(
        config: SwarmConfig,
        state: SwarmState,
    ) -> Result<(Self, mpsc::UnboundedReceiver<SwarmEvent>)> {
        let (swarm, events) = Self::new(config);
        swarm.inner.next_id.store(state.next_id, Ordering::SeqCst);
        if let Some(tokens) = state.budget_tokens {
            *swarm.inner.budget.lock() = Some(tokens);
            swarm.emit(SwarmEvent::Budget { tokens: Some(tokens) });
        }
        if let Some(journal) = &swarm.inner.journal {
            journal.event("resume", serde_json::json!({ "agents": state.agents.len() }));
        }
        if let (Some(from), Some(to)) = (state.commit.clone(), swarm.inner.config.commit.clone()) {
            if from != to {
                let changes = crate::version::changes(&swarm.inner.config.source, &from, &to).await;
                swarm.emit(SwarmEvent::Reloaded { from, to, changes });
            }
        }
        for agent in state.agents {
            swarm.restore_agent(agent);
        }
        swarm.start_queued();
        swarm.persist();
        swarm.collect_in_background();
        Ok((swarm, events))
    }

    fn restore_agent(&self, saved: AgentState) {
        let parent = self.node(&saved.parent);
        let cancel = parent
            .as_ref()
            .map_or(&self.inner.root_cancel, |node| &node.cancel)
            .child_token();
        let status = match saved.status.as_str() {
            "cancelled" => AgentStatus::Cancelled,
            "queued" => AgentStatus::Queued,
            _ => AgentStatus::Starting,
        };
        let interrupted = matches!(saved.status.as_str(), "running" | "starting");
        let mailbox = saved.mailbox.into_iter().map(mail_from_state).collect();
        // Runs saved before private target dirs existed get one on resume.
        let env = match saved.branch.is_some() && saved.env.is_empty() {
            true => self.worktree_env(&saved.id),
            false => saved.env,
        };
        let mut agents = self.inner.agents.lock();
        let taken = agents.values().map(|node| node.name.as_str());
        let node = Arc::new(Node {
            id: saved.id.clone(),
            name: names::choose(saved.name.as_deref(), &saved.task, taken),
            parent: saved.parent,
            cwd: saved.cwd,
            branch: saved.branch,
            status: Mutex::new(status.clone()),
            mailbox: Mutex::new(mailbox),
            notify: Notify::new(),
            cancel,
            agent: OnceLock::new(),
            task: saved.task,
            model: Mutex::new(saved.model),
            goal: saved.goal,
            env,
            usage: Mutex::new(Tokens {
                input: saved.input_tokens,
                output: saved.output_tokens,
                cache_read: saved.cache_read_tokens,
                cache_write: saved.cache_write_tokens,
            }),
            finished: AtomicBool::new(matches!(status, AgentStatus::Cancelled | AgentStatus::Queued)),
            collected: AtomicBool::new(false),
            context: Mutex::new(saved.context),
            turn: Mutex::new(None),
            paused: AtomicBool::new(saved.paused),
            tool_calls: Mutex::new(0),
        });
        agents.insert(saved.id.clone(), node.clone());
        drop(agents);
        // Replayed for the UI only: the journal recorded them when they happened.
        let _ = self.inner.events.send(SwarmEvent::Spawned {
            id: node.id.clone(),
            parent: node.parent.clone(),
            name: node.name.clone(),
            task: node.task.clone(),
            cwd: node.cwd.clone(),
            branch: node.branch.clone(),
        });
        let _ = self.inner.events.send(SwarmEvent::Status {
            id: node.id.clone(),
            status: status.clone(),
        });
        self.emit_usage(&node);
        if status.is_live() && status != AgentStatus::Queued {
            self.start_actor(node, Start::Resume { interrupted });
        }
    }

    /// First input after a restore: a restart notice for an interrupted agent (or one with a goal,
    /// which must keep going), otherwise nothing, so it waits for mail.
    async fn resume_input(&self, node: &Node, agent: &Agent, interrupted: bool) -> Option<InvokeInput> {
        if let Err(error) = agent.initialize().await {
            tracing::warn!(agent = %node.id, %error, "session restore failed");
        }
        let (messages, omitted) = agent.with_messages(|messages| {
            let omitted = messages.len().saturating_sub(HISTORY_MESSAGES);
            (messages[omitted..].to_vec(), omitted)
        });
        if !messages.is_empty() {
            self.emit(SwarmEvent::History {
                id: node.id.clone(),
                messages,
                omitted,
            });
        }
        let paused = node.paused.load(Ordering::SeqCst);
        if !interrupted && (node.goal.is_none() || paused) {
            return None;
        }
        let folded = agent.with_messages(append_restart_notice);
        Some(match folded {
            true => InvokeInput::None,
            false => InvokeInput::Text(RESTART_NOTICE.to_owned()),
        })
    }

    /// The persistable view of the swarm.
    pub fn snapshot(&self) -> SwarmState {
        // Release the agents lock before taking per-agent locks.
        let nodes: Vec<Arc<Node>> = self.inner.agents.lock().values().cloned().collect();
        let agents = nodes.iter().map(|node| node_state(node)).collect();
        SwarmState {
            run_id: self.inner.config.run_id.clone(),
            cwd: self.inner.config.cwd.clone(),
            next_id: self.inner.next_id.load(Ordering::SeqCst),
            agents,
            budget_tokens: *self.inner.budget.lock(),
            commit: self.inner.config.commit.clone(),
        }
    }

    fn persist(&self) {
        let Some(persister) = self.inner.persister.clone() else {
            return;
        };
        if self.inner.shutting_down.load(Ordering::SeqCst) {
            return;
        }
        let generation = self.inner.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let state = self.snapshot();
        tokio::spawn(async move {
            if let Err(error) = persister.write(generation, &state).await {
                tracing::warn!(%error, "state save failed");
            }
        });
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

    /// Tokens (input + output) used by every agent so far, saved runs included.
    pub fn tokens_used(&self) -> u64 {
        let nodes: Vec<Arc<Node>> = self.inner.agents.lock().values().cloned().collect();
        nodes
            .iter()
            .map(|node| {
                let usage = *node.usage.lock();
                usage.input + usage.output
            })
            .sum()
    }

    /// The swarm-wide token budget; `None` for no limit.
    pub fn budget(&self) -> Option<u64> {
        *self.inner.budget.lock()
    }

    /// Set (or, with `None`, lift) the token budget. If that leaves budget to spend, every agent
    /// that was stopped for lack of it is told to continue.
    pub fn set_budget(&self, tokens: Option<u64>) {
        *self.inner.budget.lock() = tokens;
        self.persist();
        self.emit(SwarmEvent::Budget { tokens });
        if self.budget_exhausted().is_some() {
            return;
        }
        let stopped: Vec<String> = std::mem::take(&mut *self.inner.over_budget.lock());
        let limit = tokens.map_or_else(
            || "lifted".to_owned(),
            |t| format!("raised to {}", budget::compact(t)),
        );
        for id in stopped {
            let note = format!("[The token budget was {limit}. Continue where you left off.]");
            let _ = self.send(USER, &id, note);
        }
    }

    /// The refusal message when the budget is spent.
    fn budget_exhausted(&self) -> Option<String> {
        let budget = self.budget()?;
        let used = self.tokens_used();
        (used >= budget).then(|| budget::exhausted_message(used, budget))
    }

    /// `parser (a2)` for an agent, `id` itself for anyone else (the user).
    pub fn label(&self, id: &str) -> String {
        self.node(id).map_or_else(|| id.to_owned(), |node| node.label())
    }

    fn node(&self, id: &str) -> Option<Arc<Node>> {
        self.inner.agents.lock().get(id).cloned()
    }

    fn emit(&self, event: SwarmEvent) {
        if let Some(journal) = &self.inner.journal {
            journal.record(&event);
        }
        let _ = self.inner.events.send(event);
    }

    /// Put a message to or from the user on record.
    fn journal_mail(&self, mail: &Mail, to: &str) {
        let Some(journal) = &self.inner.journal else {
            return;
        };
        if mail.from == USER || to == USER {
            let to_label = self.label(to);
            let attached: String = mail
                .images
                .iter()
                .map(|path| format!("\n[attached image: {}]", path.display()))
                .collect();
            let text = format!("{}{attached}", mail.text);
            journal.mail(&mail.from, &mail.from_label, to, &to_label, mail.kind, &text);
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
            *node.tool_calls.lock() = 0;
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

    fn post(&self, from: &str, to: &str, kind: &'static str, text: String) -> Result<()> {
        self.post_mail(from, to, kind, text, Vec::new())
    }

    fn post_mail(
        &self,
        from: &str,
        to: &str,
        kind: &'static str,
        text: String,
        images: Vec<PathBuf>,
    ) -> Result<()> {
        let from_label = self.label(from);
        let mail = Mail {
            from: from.to_owned(),
            from_label,
            kind,
            text,
            images,
        };
        let event = SwarmEvent::Mail {
            to: to.to_owned(),
            mail: mail.clone(),
        };
        if to == USER {
            self.journal_mail(&mail, to);
            self.emit(event);
            return Ok(());
        }
        let node = self
            .node(to)
            .ok_or_else(|| Error::other(format!("Unknown agent id {to}.")))?;
        if !node.status.lock().is_live() {
            return Err(Error::other(format!("Agent {to} is cancelled.")));
        }
        self.journal_mail(&mail, to);
        // Emitted under the mailbox lock, so it always precedes the `MailRead` of its drain.
        let mut mailbox = node.mailbox.lock();
        self.emit(event);
        mailbox.push_back(mail);
        drop(mailbox);
        self.persist();
        node.notify.notify_waiters();
        node.notify.notify_one();
        Ok(())
    }

    async fn run_actor(self, node: Arc<Node>, start: Start) {
        // Read under its own statement: `instructions` takes the agents lock, and `snapshot` takes
        // the agents lock before per-agent locks.
        let model = node.model.lock().clone();
        let spec = AgentSpec {
            id: node.id.clone(),
            name: node.name.clone(),
            cwd: node.cwd.clone(),
            model,
            instructions: self.instructions(&node, node.goal.as_deref()),
            tools: swarm_tools(self.clone(), &node.id),
            goal: node.goal.clone(),
            session_dir: self
                .inner
                .config
                .state_dir
                .as_ref()
                .map(|dir| dir.join(SESSIONS_DIR)),
            env: node.env.clone(),
        };
        let agent = match (self.inner.config.factory)(spec).await {
            Ok(agent) => agent,
            Err(error) => return self.fail_construction(&node, error),
        };
        self.install_mail_hook(&agent, &node.id);
        self.install_model_gate(&agent, &node);
        agent.set_aux_model(Some(Arc::new(SideModel {
            swarm: self.clone(),
            node: Arc::downgrade(&node),
            model: self.inner.config.side_model.clone(),
        })));
        let _ = node.agent.set(agent.clone());
        self.emit_model(&node);

        let mut input = match start {
            Start::Fresh(task) => Some(InvokeInput::Text(task)),
            Start::Resume { interrupted } => self.resume_input(&node, &agent, interrupted).await,
        };
        let mut backoff = RETRY_INITIAL;
        while let Some(text) = self.next_input(&node, input.take()).await {
            let Some(error) = self.run_turn(&node, &agent, text).await else {
                backoff = RETRY_INITIAL;
                continue;
            };
            // A goal-driven agent must survive transient failures (throttling, outages) unattended.
            // Out of budget, it waits for the mail that `set_budget` sends instead of retrying.
            if node.goal.is_some()
                && self.budget_exhausted().is_none()
                && self.sleep_unless_cancelled(&node, backoff).await
            {
                input = Some(InvokeInput::Text(format!(
                    "Your previous turn failed with: {error}\nContinue working toward your goal."
                )));
                backoff = (backoff * 2).min(RETRY_MAX);
            }
        }
        if !self.inner.shutting_down.load(Ordering::SeqCst) {
            self.set_status(&node, AgentStatus::Cancelled);
        }
    }

    fn fail_construction(&self, node: &Node, error: Error) {
        let message = format!("Failed to start: {error}");
        self.set_status(node, AgentStatus::Errored(message.clone()));
        let _ = self.post(&node.id, &node.parent, "error", message);
        self.set_status(node, AgentStatus::Cancelled);
    }

    /// The next turn's input: the initial task, else the agent's mail once some arrives.
    async fn next_input(&self, node: &Node, initial: Option<InvokeInput>) -> Option<InvokeInput> {
        if node.cancel.is_cancelled() {
            return None;
        }
        if let Some(task) = initial {
            return Some(task);
        }
        let mail = self
            .wait_for_mail(&node.id, &[], &CancellationToken::new())
            .await?;
        let text = render_mail(&mail);
        Some(match mail.iter().any(|m| !m.images.is_empty()) {
            true => InvokeInput::Blocks(images::input_blocks(text, &mail)),
            false => InvokeInput::Text(text),
        })
    }

    /// `true` when the full delay elapsed, `false` when the agent was cancelled first.
    async fn sleep_unless_cancelled(&self, node: &Node, delay: Duration) -> bool {
        tokio::select! {
            _ = tokio::time::sleep(delay) => true,
            _ = node.cancel.cancelled() => false,
        }
    }

    /// Run one turn; returns the error message when it failed (not when it was cancelled).
    async fn run_turn(&self, node: &Node, agent: &Agent, input: InvokeInput) -> Option<String> {
        let turn = node.cancel.child_token();
        *node.turn.lock() = Some(turn.clone());
        node.paused.store(false, Ordering::SeqCst);
        self.set_status(node, AgentStatus::Running);
        self.refresh_memory(agent).await;
        let options = InvokeOptions::default().with_cancel(turn.clone());
        let mut stream = agent.stream_with(input, options);
        let mut reply = None;
        let mut failure = None;
        while let Some(event) = stream.next().await {
            match event {
                Ok(event) => reply = self.forward(node, &event).or(reply),
                Err(error) => failure = Some(error),
            }
        }
        drop(stream);
        node.turn.lock().take();
        if node.cancel.is_cancelled() || self.inner.shutting_down.load(Ordering::SeqCst) {
            return None;
        }
        if turn.is_cancelled() {
            // Interrupted: the SDK has already closed any open tool call, so the conversation
            // continues cleanly with the next message. No retry, even with a goal.
            node.paused.store(true, Ordering::SeqCst);
            self.set_status(node, AgentStatus::Idle);
            let _ = self.post(&node.id, &node.parent, "interrupted", INTERRUPTED_TEXT.to_owned());
            return None;
        }
        if let Some(error) = failure {
            let message = error.to_string();
            self.set_status(node, AgentStatus::Errored(message.clone()));
            let _ = self.post(&node.id, &node.parent, "error", message.clone());
            return Some(message);
        }
        self.set_status(node, AgentStatus::Idle);
        let _ = self.post(&node.id, &node.parent, "reply", reply.unwrap_or_default());
        None
    }

    /// Translate one stream event into swarm events; returns the final reply on `AgentResult`.
    fn forward(&self, node: &Node, event: &AgentStreamEvent) -> Option<String> {
        let id = node.id.clone();
        match event {
            AgentStreamEvent::ModelStreamUpdate(update) => match &update.event {
                ModelStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::Text { text },
                } => self.emit(SwarmEvent::Text {
                    id,
                    delta: text.clone(),
                }),
                ModelStreamEvent::Metadata {
                    usage: Some(usage), ..
                } => self.add_usage(node, usage, true),
                _ => {}
            },
            AgentStreamEvent::BeforeToolCall(call) => {
                if !WAITING_TOOLS.contains(&call.tool_use.name.as_str()) {
                    *node.tool_calls.lock() += 1;
                }
                self.emit(SwarmEvent::ToolStarted {
                    id,
                    name: call.tool_use.name.clone(),
                    summary: summarize_input(&call.tool_use.input),
                })
            }
            AgentStreamEvent::AfterToolCall(call) => {
                if !WAITING_TOOLS.contains(&call.tool_use.name.as_str()) {
                    let mut calls = node.tool_calls.lock();
                    *calls = calls.saturating_sub(1);
                }
                self.emit(SwarmEvent::ToolFinished {
                    id: id.clone(),
                    name: call.tool_use.name.clone(),
                    ok: call.result.status == ToolResultStatus::Success,
                });
                for content in &call.result.content {
                    if let ToolResultContent::Image(ImageBlock {
                        source: ImageSource::Bytes(bytes),
                        ..
                    }) = content
                    {
                        self.emit(SwarmEvent::Image {
                            id: id.clone(),
                            source: summarize_input(&call.tool_use.input),
                            bytes: Arc::new(bytes.clone()),
                        });
                    }
                }
            }
            AgentStreamEvent::AgentResult(result) => return Some(result.result.last_message.text()),
            _ => {}
        }
        None
    }

    /// Add one model call's usage (its trailing metadata) to the agent's totals and publish them;
    /// `own_call` says whether the call carried the agent's context (not a side call's).
    fn add_usage(&self, node: &Node, usage: &Usage, own_call: bool) {
        let cache_read = usage.cache_read_input_tokens.unwrap_or(0);
        let cache_write = usage.cache_write_input_tokens.unwrap_or(0);
        let input = usage.input_tokens + cache_read + cache_write;
        {
            let mut total = node.usage.lock();
            total.input += input;
            total.output += usage.output_tokens;
            total.cache_read += cache_read;
            total.cache_write += cache_write;
        }
        // Relative to the compaction point, so 100% means older turns are summarized before the
        // next call; the window is the one the context manager assumes. A side call's input is
        // not the agent's context.
        if let Some(agent) = node.agent.get().filter(|_| own_call) {
            let window = crate::compaction::window_of(agent.model().as_ref());
            let point = crate::compaction::compaction_point(window, self.inner.config.compact_at);
            *node.context.lock() = Some(input as f64 / point as f64);
        }
        self.emit_usage(node);
    }

    fn emit_usage(&self, node: &Node) {
        let usage = *node.usage.lock();
        let context = *node.context.lock();
        self.emit(SwarmEvent::Usage {
            id: node.id.clone(),
            input_tokens: usage.input,
            output_tokens: usage.output,
            cache_read_tokens: usage.cache_read,
            cache_write_tokens: usage.cache_write,
            context,
        });
    }

    /// Gate every model call: refuse it once the token budget is spent, and hold a swarm-wide
    /// permit for its duration, so a large swarm queues instead of tripping provider throttling.
    /// Waiting is published as [`SwarmEvent::ModelQueued`].
    fn install_model_gate(&self, agent: &Agent, node: &Node) {
        let swarm = self.clone();
        let id = node.id.clone();
        agent.add_middleware::<InvokeModelStage>(wrap_fn(
            move |context: InvokeModelContext, events, next| {
                let (swarm, id) = (swarm.clone(), id.clone());
                // The invocation's token: cancelled by an interrupt as well as by a cancel.
                let cancel = context.agent.cancel_token();
                async move {
                    let _permit = swarm.admit_model_call(&id, &cancel).await?;
                    next.run(context, &events).await
                }
            },
        ));
    }

    /// Refuse a model call of agent `id` once the budget is spent, else wait for a swarm-wide
    /// permit (if calls are capped), to be held for the call's duration.
    async fn admit_model_call(
        &self,
        id: &str,
        cancel: &CancellationToken,
    ) -> Result<Option<OwnedSemaphorePermit>> {
        if let Some(message) = self.budget_exhausted() {
            let mut stopped = self.inner.over_budget.lock();
            if !stopped.iter().any(|stopped| stopped == id) {
                stopped.push(id.to_owned());
            }
            return Err(Error::other(message));
        }
        let Some(permits) = self.inner.model_calls.clone() else {
            return Ok(None);
        };
        if let Ok(permit) = permits.clone().try_acquire_owned() {
            return Ok(Some(permit));
        }
        self.emit(SwarmEvent::ModelQueued {
            id: id.to_owned(),
            queued: true,
        });
        let acquired = tokio::select! {
            permit = permits.acquire_owned() => permit.ok(),
            _ = cancel.cancelled() => None,
        };
        self.emit(SwarmEvent::ModelQueued {
            id: id.to_owned(),
            queued: false,
        });
        acquired.map(Some).ok_or(Error::Cancelled)
    }

    /// Deliver mail that arrives mid-turn by appending it to the agent's next tool result.
    fn install_mail_hook(&self, agent: &Agent, id: &str) {
        let swarm = self.clone();
        let id = id.to_owned();
        agent
            .hooks()
            .add_sync_callback(move |event: &mut AfterToolCallEvent| {
                let mail = swarm.drain(&id);
                if !mail.is_empty() {
                    let text = format!(
                        "\n\n[New messages arrived while this tool ran]\n{}",
                        render_mail(&mail)
                    );
                    event.result.content.push(ToolResultContent::Text(text));
                    event.result.content.extend(images::result_content(&mail));
                }
                Ok(())
            });
    }

    fn instructions(&self, node: &Node, goal: Option<&str>) -> String {
        let who = format!("agent {} (\"{}\")", node.id, node.name);
        let reports_to = match node.parent.as_str() {
            USER => "the user".to_owned(),
            parent => format!("agent {}, who spawned you", self.label(parent)),
        };
        let branch = node
            .branch
            .as_ref()
            .map(|b| format!(" It is your own git worktree on branch `{b}`."))
            .unwrap_or_default();
        let env = node
            .env
            .get("CARGO_TARGET_DIR")
            .map(|dir| format!(" Your commands run with `CARGO_TARGET_DIR={dir}`, private to this worktree (the first build starts from scratch)."))
            .unwrap_or_default();
        let swarm = format!(
            "# Swarm\n\
             You are {who}, one of a group of cooperating agents. Your working directory is `{}`; every `shell` call starts there, so there is no need to `cd` into it.{branch}{env}\n\
             You can start other agents to work in parallel (`spawn_agent`), message any agent \
             (`send_message`), wait for messages (`wait`), see who exists (`list_agents`) and stop agents \
             (`cancel_agent`). Whether and how to split the work is your call.\n\
             Messages from other agents reach you as `<message from=\"...\">` blocks, either appended to a \
             tool result or as a new turn. Your final reply at the end of each turn is delivered to {reports_to}.",
            node.cwd.display()
        );
        match goal {
            Some(goal) => format!(
                "{swarm}\n\n# Goal\nA judge checks this goal whenever you end a turn; until it is met you are \
                 sent back to work, and your final reply does not reach {reports_to}. To ask them something, \
                 `send_message` to `parent` and `wait` for the answer.\n{goal}"
            ),
            None => swarm,
        }
    }
}

/// Add the restart notice to the end of a restored conversation without breaking role
/// alternation: answer dangling tool calls (with errors) and put the notice in the same user
/// message, or fold it into a trailing user message. Returns `false` when the conversation ends
/// with a plain assistant message (or is empty), so the notice can be sent as normal input.
fn append_restart_notice(messages: &mut Vec<Message>) -> bool {
    let Some(last) = messages.last_mut() else {
        return false;
    };
    let notice = ContentBlock::Text(RESTART_NOTICE.to_owned());
    if last.role == Role::User {
        last.content.push(notice);
        return true;
    }
    let mut content: Vec<ContentBlock> = last
        .tool_uses()
        .map(|call| {
            ContentBlock::ToolResult(ToolResultBlock::error(
                call.tool_use_id.clone(),
                "Interrupted by a swarm restart.",
            ))
        })
        .collect();
    if content.is_empty() {
        return false;
    }
    content.push(notice);
    messages.push(Message::new(Role::User, content));
    true
}

async fn check_binary(binary: &Path) -> Result<()> {
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

async fn install_binary(binary: &Path, bin_dir: &Path) -> Result<()> {
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

/// Agents holding a slot under [`SwarmConfig::max_agents`]: live and not queued.
fn started(agents: &IndexMap<String, Arc<Node>>) -> usize {
    agents
        .values()
        .filter(|node| !matches!(*node.status.lock(), AgentStatus::Cancelled | AgentStatus::Queued))
        .count()
}

fn status_name(status: &AgentStatus) -> &'static str {
    match status {
        AgentStatus::Queued => "queued",
        AgentStatus::Starting => "starting",
        AgentStatus::Running => "running",
        AgentStatus::Idle => "idle",
        AgentStatus::Errored(_) => "errored",
        AgentStatus::Cancelled => "cancelled",
    }
}

fn node_state(node: &Node) -> AgentState {
    let usage = *node.usage.lock();
    AgentState {
        id: node.id.clone(),
        name: Some(node.name.clone()),
        parent: node.parent.clone(),
        cwd: node.cwd.clone(),
        branch: node.branch.clone(),
        task: node.task.clone(),
        model: node.model.lock().clone(),
        goal: node.goal.clone(),
        status: status_name(&node.status.lock()).to_owned(),
        mailbox: node.mailbox.lock().iter().map(mail_state).collect(),
        env: node.env.clone(),
        input_tokens: usage.input,
        output_tokens: usage.output,
        cache_read_tokens: usage.cache_read,
        cache_write_tokens: usage.cache_write,
        context: *node.context.lock(),
        paused: node.paused.load(Ordering::SeqCst),
    }
}

fn mail_state(mail: &Mail) -> MailState {
    MailState {
        from: mail.from.clone(),
        from_label: mail.from_label.clone(),
        kind: mail.kind.to_owned(),
        text: mail.text.clone(),
        images: mail.images.clone(),
    }
}

fn mail_from_state(mail: MailState) -> Mail {
    let kind = match mail.kind.as_str() {
        "reply" => "reply",
        "error" => "error",
        "interrupted" => "interrupted",
        _ => "message",
    };
    Mail {
        from: mail.from,
        from_label: mail.from_label,
        kind,
        text: mail.text,
        images: mail.images,
    }
}

fn has_matching_mail(node: &Node, filter: &[String]) -> bool {
    let mailbox = node.mailbox.lock();
    match filter.is_empty() {
        true => !mailbox.is_empty(),
        false => mailbox.iter().any(|mail| filter.contains(&mail.from)),
    }
}

/// One-line summary of a tool input: the most telling string field, else compact JSON.
pub(crate) fn summarize_input(input: &serde_json::Value) -> String {
    const KEYS: &[&str] = &["command", "path", "task", "to", "url", "query"];
    let text = KEYS
        .iter()
        .find_map(|key| input.get(*key).and_then(|v| v.as_str()).map(str::to_owned))
        .unwrap_or_else(|| input.to_string());
    let line = text.lines().next().unwrap_or_default();
    match line.chars().count() > 120 {
        true => format!("{}…", line.chars().take(120).collect::<String>()),
        false => line.to_owned(),
    }
}
