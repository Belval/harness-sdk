//! The actor loop that drives one agent turn by turn, and forwarding of its stream events.

use super::*;

impl Swarm {
    /// Run `node`'s actor; once it exits, reclaim the disk of agents that are now fully stopped.
    pub(super) fn start_actor(&self, node: Arc<Node>, start: Start) {
        let swarm = self.clone();
        self.track(tokio::spawn(async move {
            swarm.clone().run_actor(node.clone(), start).await;
            node.finished.store(true, Ordering::SeqCst);
            node.release_agent();
            swarm.collect_in_background();
        }));
    }

    /// First input after a restore: a restart notice for an interrupted agent (or one with a goal,
    /// which must keep going), otherwise nothing, so it waits for mail.
    async fn resume_input(
        &self,
        node: &Node,
        agent: &Agent,
        interrupted: bool,
        ran: &BTreeMap<String, u64>,
    ) -> Option<InvokeInput> {
        if let Err(error) = agent.initialize().await {
            tracing::warn!(agent = %node.id, %error, "session restore failed");
        }
        self.restore_open_batch(node, agent).await;
        if interrupted {
            agent.with_messages(|messages| resume::reword_cancel(messages));
        }
        if let Some(history) = agent.with_messages(|messages| history_event(&node.id, messages)) {
            self.emit(history);
        }
        // A goal agent whose turn failed retries, as it would have without the restart; one that
        // finished its turn (goal met or given up on) waits for mail like any other.
        let errored = matches!(*node.status.lock(), AgentStatus::Errored(_));
        let retry = errored && node.goal.is_some() && !node.paused.load(Ordering::SeqCst);
        if !interrupted && !retry {
            return None;
        }
        let in_flight = agent.with_messages(|messages| resume::in_flight(messages));
        let status = resume::git_status(&node.cwd).await;
        let notice = resume::notice(&in_flight, ran, &node.cwd, status.as_deref());
        let folded = agent.with_messages(|messages| resume::append_notice(messages, &notice));
        Some(match folded {
            true => InvokeInput::None,
            false => InvokeInput::Text(notice),
        })
    }

    /// Where agent `id`'s context pipeline reports a summary ([`SwarmEvent::Compacted`]) or a
    /// failed one ([`SwarmEvent::CompactFailed`]). It holds the swarm weakly, as the agent it is
    /// handed to is owned by the swarm.
    fn compaction_sink(&self, id: &str) -> crate::compaction::CompactedSink {
        let inner = Arc::downgrade(&self.inner);
        let id = id.to_owned();
        Arc::new(move |compaction| {
            let Some(inner) = inner.upgrade() else { return };
            let id = id.clone();
            Swarm { inner }.emit(match compaction {
                crate::compaction::Compaction::Done { before, after } => {
                    SwarmEvent::Compacted { id, before, after }
                }
                crate::compaction::Compaction::Failed { tokens, error } => {
                    SwarmEvent::CompactFailed { id, tokens, error }
                }
            });
        })
    }

    /// The environment of `node`'s shell commands: its own, plus in a persistent run the marker
    /// a resume finds them by if this process dies hard ([`crate::orphans`]).
    fn shell_env(&self, node: &Node) -> BTreeMap<String, String> {
        let mut env = node.env.clone();
        if let Some(dir) = &self.inner.config.state_dir {
            env.insert(
                crate::orphans::SHELL_MARKER.to_owned(),
                crate::orphans::marker(dir, &node.id),
            );
        }
        env
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
            env: self.shell_env(&node),
            on_compacted: Some(self.compaction_sink(&node.id)),
        };
        let agent = match (self.inner.config.factory)(spec).await {
            Ok(agent) => agent,
            Err(error) => return self.fail_construction(&node, error),
        };
        self.install_mail_hook(&agent, &node.id);
        self.install_poll_hint(&agent, &node.id);
        self.install_model_gate(&agent, &node);
        self.install_keep_warm(&agent, &node);
        agent.set_aux_model(Some(Arc::new(SideModel {
            swarm: self.clone(),
            node: Arc::downgrade(&node),
            model: self.inner.config.side_model.clone(),
        })));
        *node.agent.lock() = Some(agent.clone());
        self.emit_model(&node);

        let mut input = match start {
            Start::Fresh(task) => Some(InvokeInput::Text(task)),
            Start::Resume { interrupted, ran } => self.resume_input(&node, &agent, interrupted, &ran).await,
        };
        let mut backoff = RETRY_INITIAL;
        while let Some(text) = self.next_input(&node, input.take()).await {
            // Only the first failure of a retry streak reaches the parent: each one would wake it.
            // So does only the first refusal for want of budget, however often mail (a schedule)
            // wakes the agent until the budget is raised.
            let retrying = backoff > RETRY_INITIAL;
            let refused = self.inner.over_budget.lock().contains(&node.id);
            let report = !retrying && !refused;
            let Some(error) = self.run_turn(&node, &agent, text, report).await else {
                backoff = RETRY_INITIAL;
                continue;
            };
            // A goal-driven agent must survive transient failures (throttling, outages) unattended.
            // Out of budget, it waits for the mail that `set_budget` sends instead of retrying.
            if node.goal.is_some()
                && self.budget_exhausted().is_none()
                && sleep_unless_cancelled(&node, backoff).await
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
        // Mail before status, as at the end of a turn (see `run_turn`).
        let _ = self.post(&node.id, &node.parent, "error", message.clone());
        self.set_status(node, AgentStatus::Errored(message));
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

    /// Run one turn; returns the error message when it failed (not when it was cancelled). A
    /// failure is posted to the parent when `report` is set.
    async fn run_turn(&self, node: &Node, agent: &Agent, input: InvokeInput, report: bool) -> Option<String> {
        let turn = node.cancel.child_token();
        *node.turn.lock() = Some(turn.clone());
        node.paused.store(false, Ordering::SeqCst);
        self.set_status(node, AgentStatus::Running);
        self.refresh_memory(agent).await;
        let options = InvokeOptions::default().with_cancel(turn.clone());
        let given = matches!(input, InvokeInput::Text(_) | InvokeInput::Blocks(_));
        let input = agent.with_messages(|messages| resume::fold_into_open_turn(messages, input));
        if given && matches!(input, InvokeInput::None) {
            // The SDK adds no message for it, so nothing would save the mail it holds, which has
            // already left the mailbox, before the turn's first message: a crash would lose it.
            self.save_session(node, agent).await;
        }
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
        // Each ending posts its mail before the status leaves `Running`: in between, the agent
        // would look done with nothing unread anywhere, which `Swarm::is_busy` reads as a swarm
        // with nothing left to do (a headless run exited there, the root's error unprinted).
        if turn.is_cancelled() {
            // Interrupted: the SDK has already closed any open tool call, so the conversation
            // continues cleanly with the next message. No retry, even with a goal.
            node.paused.store(true, Ordering::SeqCst);
            let _ = self.post(&node.id, &node.parent, "interrupted", INTERRUPTED_TEXT.to_owned());
            self.set_status(node, AgentStatus::Idle);
            return None;
        }
        if let Some(error) = failure {
            let message = error.to_string();
            if report {
                let _ = self.post(&node.id, &node.parent, "error", message.clone());
            }
            self.set_status(node, AgentStatus::Errored(message.clone()));
            return Some(message);
        }
        let reply = reply.unwrap_or_default();
        // The user sees replies as streamed text anyway, and headless runs wait for one.
        if node.parent == USER || !is_silent(&reply) {
            let _ = self.post(&node.id, &node.parent, "reply", reply);
        }
        self.set_status(node, AgentStatus::Idle);
        None
    }

    /// Translate one stream event into swarm events; returns the final reply on `AgentResult`.
    fn forward(&self, node: &Node, event: &AgentStreamEvent) -> Option<String> {
        self.track_batch(node, event);
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
                    let model = node.agent().and_then(|agent| agent.model().model_id());
                    self.add_usage(node, usage, model, true)
                }
                _ => {}
            },
            AgentStreamEvent::BeforeToolCall(call) => {
                if !WAITING_TOOLS.contains(&call.tool_use.name.as_str()) {
                    let id = call.tool_use.tool_use_id.clone();
                    node.tool_calls.lock().insert(id, Instant::now());
                }
                self.emit(SwarmEvent::ToolStarted {
                    id,
                    name: call.tool_use.name.clone(),
                    summary: summarize_input(&call.tool_use.input),
                })
            }
            AgentStreamEvent::AfterToolCall(call) => {
                node.tool_calls.lock().remove(&call.tool_use.tool_use_id);
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
        let counts = usage_counts(usage);
        let model = model.unwrap_or_else(|| "unknown".to_owned());
        node.ledger.lock().record(&model, counts, own_call);
        self.add_totals(node, counts, own_call);
    }

    /// [`Swarm::add_usage`] for a side call, tagged with what it was for. Each one is also an
    /// `side_call` event in `events.jsonl`.
    fn add_side_usage(
        &self,
        node: &Node,
        usage: &Usage,
        model: Option<String>,
        purpose: crate::cost::SidePurpose,
    ) {
        let counts = usage_counts(usage);
        let model = model.unwrap_or_else(|| "unknown".to_owned());
        node.ledger.lock().record_side(purpose, &model, counts);
        self.journal_event(
            "side_call",
            serde_json::json!({
                "id": node.id,
                "purpose": purpose.name(),
                "model": model,
                "input": counts.input,
                "output": counts.output,
                "cache_read": counts.cache_read,
                "cache_write": counts.cache_write,
            }),
        );
        self.add_totals(node, counts, false);
    }

    fn add_totals(&self, node: &Node, counts: crate::cost::Counts, own_call: bool) {
        {
            let mut total = node.usage.lock();
            total.input += counts.input;
            total.output += counts.output;
            total.cache_read += counts.cache_read;
            total.cache_write += counts.cache_write;
        }
        // Relative to the compaction point, so 100% means older turns are summarized before the
        // next call; the window is the one the context manager assumes. A side call's input is
        // not the agent's context.
        if let Some(agent) = node.agent().filter(|_| own_call) {
            let window = crate::compaction::window_of(agent.model().as_ref());
            let point = crate::compaction::compaction_point(window, self.inner.config.compact_at);
            *node.context.lock() = Some(counts.input as f64 / point as f64);
        }
        self.emit_usage(node);
        // Saved now, not at the next status change or mail: a turn can run many calls without
        // either, and a hard crash would lose their tokens from the cost and the budget.
        self.persist();
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
        let silent = match node.parent == USER {
            true => String::new(),
            false => format!(
                " When there is nothing worth telling them (e.g. a poll found no change), end with just `{SILENT_REPLY}` and nothing is sent."
            ),
        };
        let swarm = format!(
            "# Swarm\n\
             You are {who}, one of a group of cooperating agents. Your working directory is `{}`; every `shell` call starts there, so there is no need to `cd` into it.{branch}{env}\n\
             You can start other agents to work in parallel (`spawn_agent`), message any agent \
             (`send_message`), wait for messages (`wait`), see who exists (`list_agents`) and stop agents \
             (`cancel_agent`). Whether and how to split the work is your call.\n\
             Messages from other agents reach you as `<message from=\"...\">` blocks, either appended to a \
             tool result or as a new turn. Your final reply at the end of each turn is delivered to {reports_to}.{silent}",
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

/// A final reply that is only this (or only whitespace) is not delivered to an agent parent.
pub const SILENT_REPLY: &str = "[silent]";

/// Whether `reply` ends the turn without a message to the parent ([`SILENT_REPLY`]).
pub(super) fn is_silent(reply: &str) -> bool {
    let reply = reply.trim();
    reply.is_empty() || reply.eq_ignore_ascii_case(SILENT_REPLY)
}

pub(super) const RETRY_INITIAL: Duration = Duration::from_secs(15);

pub(super) const RETRY_MAX: Duration = Duration::from_secs(600);

/// How an actor begins.
pub(super) enum Start {
    /// A new agent: its first turn is the task.
    Fresh(String),
    /// A restored agent; `interrupted` when it was mid-turn (or starting) when the run stopped,
    /// `ran` how long its tool calls in flight had run by then ([`crate::state::AgentState::running_calls`]).
    Resume {
        interrupted: bool,
        ran: BTreeMap<String, u64>,
    },
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
        let host = || self.node.upgrade()?.agent().map(|agent| agent.model());
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
            let prompt = options.system_prompt.as_ref().map(|prompt| prompt.text());
            let purpose = crate::cost::SidePurpose::of(prompt.as_deref());
            self.swarm.hold_for_reload(&node.id, &cancel).await?;
            let _permit = self.swarm.admit_model_call(&node.id, &cancel).await?;
            // A summary or a verdict is asked for once, of a prompt never sent again: a cache
            // write there costs 25% more than plain input and is never read.
            let once = StreamOptions { no_cache: purpose != crate::cost::SidePurpose::Other, ..options.clone() };
            let mut events = model.stream(messages, &once);
            loop {
                let next = tokio::select! {
                    next = events.next() => next,
                    _ = cancel.cancelled() => Some(Err(Error::Cancelled)),
                };
                let Some(event) = next else { break };
                let event = event?;
                if let ModelStreamEvent::Metadata { usage: Some(usage), .. } = &event {
                    let model = model.model_id();
                    self.swarm.add_side_usage(&node, usage, model, purpose);
                }
                yield event;
            }
        })
    }
}

/// One model call's tokens from its usage metadata; `input` includes cache reads and writes.
/// `true` when the full delay elapsed, `false` when the agent was cancelled first.
async fn sleep_unless_cancelled(node: &Node, delay: Duration) -> bool {
    tokio::select! {
        _ = tokio::time::sleep(delay) => true,
        _ = node.cancel.cancelled() => false,
    }
}

pub(super) fn usage_counts(usage: &Usage) -> crate::cost::Counts {
    let cache_read = usage.cache_read_input_tokens.unwrap_or(0);
    let cache_write = usage.cache_write_input_tokens.unwrap_or(0);
    crate::cost::Counts {
        calls: 1,
        input: usage.input_tokens + cache_read + cache_write,
        output: usage.output_tokens,
        cache_read,
        cache_write,
    }
}

/// One-line summary of a tool input: the most telling string field, else every field as
/// `key value` (`from a2, jobs · timeout_seconds 300`).
pub(crate) fn summarize_input(input: &serde_json::Value) -> String {
    const KEYS: &[&str] = &["command", "path", "task", "to", "url", "query"];
    let text = KEYS
        .iter()
        .find_map(|key| input.get(*key).and_then(|v| v.as_str()).map(str::to_owned))
        .unwrap_or_else(|| match input {
            serde_json::Value::Object(fields) => fields
                .iter()
                .map(|(key, value)| format!("{key} {}", scalar_list(value)))
                .collect::<Vec<_>>()
                .join(" · "),
            other => other.to_string(),
        });
    crate::text::clip(text.lines().next().unwrap_or_default(), 120)
}

/// A string bare, a list of strings or numbers comma-separated, anything else as JSON.
fn scalar_list(value: &serde_json::Value) -> String {
    let scalar = |value: &serde_json::Value| match value {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Number(_) | serde_json::Value::Bool(_) => Some(value.to_string()),
        _ => None,
    };
    let list = || {
        let items = value.as_array()?.iter().map(scalar).collect::<Option<Vec<_>>>()?;
        Some(items.join(", "))
    };
    scalar(value).or_else(list).unwrap_or_else(|| value.to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::summarize_input;

    #[test]
    fn summarizes_a_tool_input_in_words() {
        assert_eq!(summarize_input(&json!({"command": "ls -la\npwd"})), "ls -la");
        assert_eq!(
            summarize_input(&json!({"from": ["a2", "jobs"], "timeout_seconds": 300})),
            "from a2, jobs · timeout_seconds 300"
        );
        assert_eq!(summarize_input(&json!({})), "");
        assert_eq!(
            summarize_input(&json!({"todos": [{"content": "x"}]})),
            r#"todos [{"content":"x"}]"#
        );
    }
}
