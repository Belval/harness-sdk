//! The swarm runtime: an agent registry, per-agent mailboxes, and the actor loop that drives each
//! agent turn by turn.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use futures::future::BoxFuture;
use futures::StreamExt;
use indexmap::IndexMap;
use parking_lot::Mutex;
use strands_agents::agent::{AgentStreamEvent, InvokeInput, InvokeOptions};
use strands_agents::hooks::AfterToolCallEvent;
use strands_agents::models::streaming::{ContentBlockDelta, ModelStreamEvent};
use strands_agents::tools::ToolRef;
use strands_agents::types::{
    ContentBlock, Message, Role, ToolResultBlock, ToolResultContent, ToolResultStatus,
};
use strands_agents::{Agent, Error, Model, Result};
use tokio::sync::{mpsc, Notify};
use tokio_util::sync::CancellationToken;

use crate::names;
use crate::state::{AgentState, MailState, Persister, SwarmState, SESSIONS_DIR};
use crate::tools::swarm_tools;
use crate::worktree;

const RETRY_INITIAL: Duration = Duration::from_secs(15);
const RETRY_MAX: Duration = Duration::from_secs(600);
const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

/// Exit code that tells the supervisor to restart immediately into `<run>/bin/current`.
pub const RELOAD_EXIT_CODE: u8 = 75;

/// Pseudo agent id for the human at the keyboard: the root agent's parent.
pub const USER: &str = "user";

/// Lifecycle of one agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentStatus {
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
}

impl Mail {
    /// The `<message>` block the receiving model sees.
    pub fn render(&self) -> String {
        format!(
            "<message from=\"{}\" kind=\"{}\">\n{}\n</message>",
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
    /// Upper bound on live (non-cancelled) agents, root included.
    pub max_agents: usize,
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
}

impl SwarmConfig {
    /// Defaults: 16 live agents and a random 8-character run id.
    pub fn new(factory: AgentFactory, cwd: impl Into<PathBuf>) -> Self {
        Self {
            factory,
            cwd: cwd.into(),
            max_agents: 16,
            run_id: uuid::Uuid::new_v4().simple().to_string()[..8].to_owned(),
            state_dir: None,
            cargo_target_dir: None,
            models: None,
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
        /// Input tokens so far.
        input_tokens: u64,
        /// Output tokens so far.
        output_tokens: u64,
    },
}

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
}

impl Node {
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
    generation: AtomicU64,
    shutting_down: AtomicBool,
    actors: Mutex<Vec<tokio::task::JoinHandle<()>>>,
    reload: CancellationToken,
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
            generation: AtomicU64::new(0),
            shutting_down: AtomicBool::new(false),
            actors: Mutex::new(Vec::new()),
            reload: CancellationToken::new(),
            config,
        };
        (
            Self {
                inner: Arc::new(inner),
            },
            rx,
        )
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

    /// Whether any agent is mid-turn, starting, or has unread mail.
    pub fn is_busy(&self) -> bool {
        self.agents().iter().any(|info| {
            matches!(info.status, AgentStatus::Running | AgentStatus::Starting)
                || (info.status.is_live() && info.unread > 0)
        })
    }

    /// Spawn an agent under `parent` ([`USER`] for a root agent). Returns its id.
    pub async fn spawn(&self, parent: &str, request: SpawnRequest) -> Result<String> {
        let parent_node = self.node(parent);
        if parent != USER && parent_node.is_none() {
            return Err(Error::other(format!("Unknown agent id {parent}.")));
        }
        self.check_capacity()?;

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
            let taken = agents.values().map(|node| node.name.as_str());
            let node = Arc::new(Node {
                id: id.clone(),
                name: names::choose(request.name.as_deref(), &request.task, taken),
                parent: parent.to_owned(),
                cwd,
                branch,
                status: Mutex::new(AgentStatus::Starting),
                mailbox: Mutex::new(VecDeque::new()),
                notify: Notify::new(),
                cancel,
                agent: OnceLock::new(),
                task: request.task.clone(),
                model: Mutex::new(request.model.clone()),
                goal: request.goal.clone(),
                env,
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

        let swarm = self.clone();
        let start = Start::Fresh(request.task);
        self.track(tokio::spawn(async move { swarm.run_actor(node, start).await }));
        Ok(id)
    }

    /// Post `text` from `from` to `to`'s mailbox. `to` may be [`USER`].
    pub fn send(&self, from: &str, to: &str, text: impl Into<String>) -> Result<()> {
        self.post(from, to, "message", text.into())
    }

    /// Cancel an agent and, through its token, every agent it spawned.
    pub fn cancel(&self, id: &str) -> Result<()> {
        let node = self
            .node(id)
            .ok_or_else(|| Error::other(format!("Unknown agent id {id}.")))?;
        node.cancel.cancel();
        if let Some(agent) = node.agent.get() {
            agent.cancel();
        }
        Ok(())
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
    }

    /// Install `binary` as the run's next binary and ask the host process to restart into it.
    ///
    /// The binary must answer `--version` within 30 s. It is copied to `<run>/bin/current`, the
    /// old one kept as `<run>/bin/previous` for the supervisor's crash rollback.
    pub async fn reload(&self, binary: &Path) -> Result<()> {
        let dir = self.inner.config.state_dir.as_ref().ok_or_else(|| {
            Error::other("Hot reload needs a persistent run (start the swarm with --run <name>).")
        })?;
        check_binary(binary).await?;
        install_binary(binary, &dir.join("bin")).await?;
        self.inner.reload.cancel();
        Ok(())
    }

    /// Resolves once an agent has requested a reload; the host should then call
    /// [`Swarm::shutdown`] and exit with [`RELOAD_EXIT_CODE`].
    pub async fn reload_requested(&self) {
        self.inner.reload.cancelled().await;
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
        for agent in state.agents {
            swarm.restore_agent(agent);
        }
        swarm.persist();
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
        });
        agents.insert(saved.id.clone(), node.clone());
        drop(agents);
        self.emit(SwarmEvent::Spawned {
            id: node.id.clone(),
            parent: node.parent.clone(),
            name: node.name.clone(),
            task: node.task.clone(),
            cwd: node.cwd.clone(),
            branch: node.branch.clone(),
        });
        self.emit(SwarmEvent::Status {
            id: node.id.clone(),
            status: status.clone(),
        });
        if status.is_live() {
            let swarm = self.clone();
            let start = Start::Resume { interrupted };
            self.track(tokio::spawn(async move { swarm.run_actor(node, start).await }));
        }
    }

    /// First input after a restore: a restart notice for an interrupted agent (or one with a goal,
    /// which must keep going), otherwise nothing, so it waits for mail.
    async fn resume_input(&self, node: &Node, agent: &Agent, interrupted: bool) -> Option<InvokeInput> {
        if let Err(error) = agent.initialize().await {
            tracing::warn!(agent = %node.id, %error, "session restore failed");
        }
        if !interrupted && node.goal.is_none() {
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

    /// `parser (a2)` for an agent, `id` itself for anyone else (the user).
    pub fn label(&self, id: &str) -> String {
        self.node(id).map_or_else(|| id.to_owned(), |node| node.label())
    }

    fn node(&self, id: &str) -> Option<Arc<Node>> {
        self.inner.agents.lock().get(id).cloned()
    }

    fn check_capacity(&self) -> Result<()> {
        let live = self
            .inner
            .agents
            .lock()
            .values()
            .filter(|n| n.status.lock().is_live())
            .count();
        if live >= self.inner.config.max_agents {
            return Err(Error::other(format!(
                "Agent limit reached ({live} live agents, max {}). Cancel or reuse an existing agent.",
                self.inner.config.max_agents
            )));
        }
        Ok(())
    }

    fn emit(&self, event: SwarmEvent) {
        let _ = self.inner.events.send(event);
    }

    fn set_status(&self, node: &Node, status: AgentStatus) {
        *node.status.lock() = status.clone();
        self.persist();
        self.emit(SwarmEvent::Status {
            id: node.id.clone(),
            status,
        });
    }

    fn post(&self, from: &str, to: &str, kind: &'static str, text: String) -> Result<()> {
        let from_label = self.label(from);
        let mail = Mail {
            from: from.to_owned(),
            from_label,
            kind,
            text,
        };
        self.emit(SwarmEvent::Mail {
            to: to.to_owned(),
            mail: mail.clone(),
        });
        if to == USER {
            return Ok(());
        }
        let node = self
            .node(to)
            .ok_or_else(|| Error::other(format!("Unknown agent id {to}.")))?;
        if !node.status.lock().is_live() {
            return Err(Error::other(format!("Agent {to} is cancelled.")));
        }
        node.mailbox.lock().push_back(mail);
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
            if node.goal.is_some() && self.sleep_unless_cancelled(&node, backoff).await {
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
        Some(InvokeInput::Text(render_mail(&mail)))
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
        self.set_status(node, AgentStatus::Running);
        let options = InvokeOptions::default().with_cancel(node.cancel.clone());
        let mut stream = agent.stream_with(input, options);
        let mut reply = None;
        let mut failure = None;
        while let Some(event) = stream.next().await {
            match event {
                Ok(event) => reply = self.forward(node, &event).or(reply),
                Err(error) => failure = Some(error),
            }
        }
        self.report_usage(node, agent);
        if node.cancel.is_cancelled() || self.inner.shutting_down.load(Ordering::SeqCst) {
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
            AgentStreamEvent::ModelStreamUpdate(update) => {
                if let ModelStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::Text { text },
                } = &update.event
                {
                    self.emit(SwarmEvent::Text {
                        id,
                        delta: text.clone(),
                    });
                }
            }
            AgentStreamEvent::BeforeToolCall(call) => self.emit(SwarmEvent::ToolStarted {
                id,
                name: call.tool_use.name.clone(),
                summary: summarize_input(&call.tool_use.input),
            }),
            AgentStreamEvent::AfterToolCall(call) => self.emit(SwarmEvent::ToolFinished {
                id,
                name: call.tool_use.name.clone(),
                ok: call.result.status == ToolResultStatus::Success,
            }),
            AgentStreamEvent::AgentResult(result) => return Some(result.result.last_message.text()),
            _ => {}
        }
        None
    }

    fn report_usage(&self, node: &Node, agent: &Agent) {
        let usage = agent.meter().metrics().accumulated_usage;
        self.emit(SwarmEvent::Usage {
            id: node.id.clone(),
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
        });
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
             You are {who}, one of a group of cooperating agents. Your working directory is `{}`.{branch}{env}\n\
             You can start other agents to work in parallel (`spawn_agent`), message any agent \
             (`send_message`), wait for messages (`wait`), see who exists (`list_agents`) and stop agents \
             (`cancel_agent`). Whether and how to split the work is your call.\n\
             Messages from other agents reach you as `<message from=\"...\">` blocks, either appended to a \
             tool result or as a new turn. Your final reply at the end of each turn is delivered to {reports_to}.",
            node.cwd.display()
        );
        match goal {
            Some(goal) => format!("{swarm}\n\n# Goal\nA judge checks this goal whenever you end a turn; until it is met you are sent back to work.\n{goal}"),
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

fn status_name(status: &AgentStatus) -> &'static str {
    match status {
        AgentStatus::Starting => "starting",
        AgentStatus::Running => "running",
        AgentStatus::Idle => "idle",
        AgentStatus::Errored(_) => "errored",
        AgentStatus::Cancelled => "cancelled",
    }
}

fn node_state(node: &Node) -> AgentState {
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
    }
}

fn mail_state(mail: &Mail) -> MailState {
    MailState {
        from: mail.from.clone(),
        from_label: mail.from_label.clone(),
        kind: mail.kind.to_owned(),
        text: mail.text.clone(),
    }
}

fn mail_from_state(mail: MailState) -> Mail {
    let kind = match mail.kind.as_str() {
        "reply" => "reply",
        "error" => "error",
        _ => "message",
    };
    Mail {
        from: mail.from,
        from_label: mail.from_label,
        kind,
        text: mail.text,
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
fn summarize_input(input: &serde_json::Value) -> String {
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
