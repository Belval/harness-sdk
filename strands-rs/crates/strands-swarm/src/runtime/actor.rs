//! The actor loop that drives one agent turn by turn, and forwarding of its stream events.

use super::*;

impl Swarm {
    /// Run `node`'s actor; once it exits, reclaim the disk of agents that are now fully stopped.
    pub(super) fn start_actor(&self, node: Arc<Node>, start: Start) {
        let swarm = self.clone();
        self.track(tokio::spawn(async move {
            swarm.clone().run_actor(node.clone(), start).await;
            node.finished.store(true, Ordering::SeqCst);
            swarm.collect_in_background();
        }));
    }

    /// First input after a restore: a restart notice for an interrupted agent (or one with a goal,
    /// which must keep going), otherwise nothing, so it waits for mail.
    pub(super) async fn resume_input(
        &self,
        node: &Node,
        agent: &Agent,
        interrupted: bool,
    ) -> Option<InvokeInput> {
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
        let in_flight = agent.with_messages(|messages| resume::in_flight(messages));
        let status = resume::git_status(&node.cwd).await;
        let notice = resume::notice(&in_flight, &node.cwd, status.as_deref());
        let folded = agent.with_messages(|messages| resume::append_notice(messages, &notice));
        Some(match folded {
            true => InvokeInput::None,
            false => InvokeInput::Text(notice),
        })
    }

    /// Where agent `id`'s context pipeline reports a summary: a [`SwarmEvent::Compacted`]. It
    /// holds the swarm weakly, as the agent it is handed to is owned by the swarm.
    fn compaction_sink(&self, id: &str) -> crate::compaction::CompactedSink {
        let inner = Arc::downgrade(&self.inner);
        let id = id.to_owned();
        Arc::new(move |before, after| {
            if let Some(inner) = inner.upgrade() {
                Swarm { inner }.emit(SwarmEvent::Compacted {
                    id: id.clone(),
                    before,
                    after,
                });
            }
        })
    }

    pub(super) async fn run_actor(self, node: Arc<Node>, start: Start) {
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
            on_compacted: Some(self.compaction_sink(&node.id)),
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
            // Only the first failure of a retry streak reaches the parent: each one would wake it.
            let retrying = backoff > RETRY_INITIAL;
            let Some(error) = self.run_turn(&node, &agent, text, !retrying).await else {
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

    pub(super) fn fail_construction(&self, node: &Node, error: Error) {
        let message = format!("Failed to start: {error}");
        self.set_status(node, AgentStatus::Errored(message.clone()));
        let _ = self.post(&node.id, &node.parent, "error", message);
        self.set_status(node, AgentStatus::Cancelled);
    }

    /// The next turn's input: the initial task, else the agent's mail once some arrives.
    pub(super) async fn next_input(&self, node: &Node, initial: Option<InvokeInput>) -> Option<InvokeInput> {
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
    pub(super) async fn sleep_unless_cancelled(&self, node: &Node, delay: Duration) -> bool {
        tokio::select! {
            _ = tokio::time::sleep(delay) => true,
            _ = node.cancel.cancelled() => false,
        }
    }

    /// Run one turn; returns the error message when it failed (not when it was cancelled). A
    /// failure is posted to the parent when `report` is set.
    pub(super) async fn run_turn(
        &self,
        node: &Node,
        agent: &Agent,
        input: InvokeInput,
        report: bool,
    ) -> Option<String> {
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
            if report {
                let _ = self.post(&node.id, &node.parent, "error", message.clone());
            }
            return Some(message);
        }
        self.set_status(node, AgentStatus::Idle);
        let _ = self.post(&node.id, &node.parent, "reply", reply.unwrap_or_default());
        None
    }

    /// Translate one stream event into swarm events; returns the final reply on `AgentResult`.
    pub(super) fn forward(&self, node: &Node, event: &AgentStreamEvent) -> Option<String> {
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
                } => {
                    let model = node.agent.get().and_then(|agent| agent.model().model_id());
                    self.add_usage(node, usage, model, true)
                }
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
    /// `own_call` says whether the call carried the agent's context (not a side call's), `model`
    /// which model answered (for cost estimates).
    pub(super) fn add_usage(&self, node: &Node, usage: &Usage, model: Option<String>, own_call: bool) {
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
        let counts = crate::cost::Counts {
            calls: 1,
            input,
            output: usage.output_tokens,
            cache_read,
            cache_write,
        };
        let model = model.unwrap_or_else(|| "unknown".to_owned());
        node.ledger.lock().record(&model, counts, own_call);
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

    pub(super) fn emit_usage(&self, node: &Node) {
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

    pub(super) fn instructions(&self, node: &Node, goal: Option<&str>) -> String {
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
                 `send_message` to `parent` and `wait` for the answer. After {GOAL_MAX_ATTEMPTS} rejected attempts \
                 in a row the turn ends anyway and your reply goes to {reports_to}, so then say plainly what is \
                 still unmet and why; a new message starts a fresh round.\n{goal}"
            ),
            None => swarm,
        }
    }
}

pub(super) const RETRY_INITIAL: Duration = Duration::from_secs(15);

pub(super) const RETRY_MAX: Duration = Duration::from_secs(600);

/// How an actor begins.
pub(super) enum Start {
    /// A new agent: its first turn is the task.
    Fresh(String),
    /// A restored agent; `interrupted` when it was mid-turn (or starting) when the run stopped.
    Resume { interrupted: bool },
}

/// An agent's auxiliary model (what side calls such as the goal judge and context summarization
/// run on): gated by the budget and the concurrency cap like the agent's own calls, with its usage
/// added to the agent's totals. Without a fixed model it follows the agent's current one, model
/// switches included.
pub(super) struct SideModel {
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
                    self.swarm.add_usage(&node, usage, model.model_id(), false);
                }
                yield event;
            }
        })
    }
}

/// One-line summary of a tool input: the most telling string field, else compact JSON.
pub(crate) fn summarize_input(input: &serde_json::Value) -> String {
    const KEYS: &[&str] = &["command", "path", "task", "to", "url", "query"];
    let text = KEYS
        .iter()
        .find_map(|key| input.get(*key).and_then(|v| v.as_str()).map(str::to_owned))
        .unwrap_or_else(|| input.to_string());
    crate::text::clip(text.lines().next().unwrap_or_default(), 120)
}
