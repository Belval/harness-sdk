//! The actor loop that drives one agent turn by turn, and forwarding of its stream events.

use super::*;
use strands_agents::hooks::BeforeToolCallEvent;
use strands_agents::tools::FunctionTool;

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
        self.pair_tool_calls(node, agent);
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
        let workdir = node.workdir.get();
        let status = resume::git_status(&workdir).await;
        let notice = resume::notice(&in_flight, ran, &workdir, status.as_deref());
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

    /// The environment of `node`'s shell commands: its own with `CARGO_BUILD_JOBS`, plus in a
    /// persistent run the marker a resume finds them by if this process dies hard
    /// ([`crate::orphans`]).
    fn shell_env(&self, node: &Node) -> BTreeMap<String, String> {
        let mut env = self.with_build_jobs(node.env.clone());
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
            workdir: node.workdir.clone(),
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
            compact_at: node.compact_at(&self.inner.config),
        };
        let agent = match (self.inner.config.factory)(spec).await {
            Ok(agent) => agent,
            Err(error) => return self.fail_construction(&node, error),
        };
        self.install_mail_hook(&agent, &node.id);
        self.install_cd_hook(&agent, &node);
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
            Start::Fresh(task) => {
                // A root's task is the user's.
                node.user_asked.store(node.parent == USER, Ordering::SeqCst);
                Some(InvokeInput::Text(task))
            }
            Start::Resume { interrupted, ran } => {
                let input = self.resume_input(&node, &agent, interrupted, &ran).await;
                // Whose words the interrupted turn answered is not saved: deliver its reply.
                node.user_asked.fetch_or(interrupted, Ordering::SeqCst);
                input
            }
        };
        let mut backoff = RETRY_INITIAL;
        // Failed turns in a row that the agent retried on its own, and when the first started.
        let mut streak: Option<(u32, tokio::time::Instant)> = None;
        // A resume has just pruned the stash, and a fresh agent has none.
        let mut stash_pruned = tokio::time::Instant::now();
        while let Some(text) = self.next_input(&node, input.take()).await {
            // Of a retry streak only the first failure and then every doubling from the 8th on
            // reach the parent: each would wake it, but a retry loop that went quiet after its
            // first report looks to the parent like one that recovered. Only the first refusal
            // for want of budget is reported, however often mail (a schedule) wakes the agent
            // until the budget is raised.
            let (failures, since) =
                streak.map_or((1, tokio::time::Instant::now()), |(n, since)| (n + 1, since));
            let refused = self.inner.over_budget.lock().contains(&node.id);
            let report = (!refused && reports_failure(failures)).then(|| FailureReport {
                failures,
                since,
                retries: node.goal.is_some(),
            });
            let failed = self.run_turn(&node, &agent, text, report).await;
            self.prune_stash_between_turns(&node, &agent, &mut stash_pruned)
                .await;
            let Some(error) = failed else {
                backoff = RETRY_INITIAL;
                streak = None;
                continue;
            };
            streak = None;
            // A goal-driven agent must survive transient failures (throttling, outages) unattended.
            // Out of budget, it waits for the mail that `set_budget` sends instead of retrying.
            if node.goal.is_some()
                && self.budget_exhausted(&node.id).is_none()
                && sleep_unless_cancelled(&node, backoff).await
            {
                input = Some(InvokeInput::Text(format!(
                    "Your previous turn failed with: {error}\nContinue working toward your goal."
                )));
                backoff = (backoff * 2).min(RETRY_MAX);
                streak = Some((failures, since));
            }
        }
        if !self.inner.shutting_down.load(Ordering::SeqCst) {
            self.set_status(&node, AgentStatus::Cancelled);
            // Its turns are over, so nothing can retrieve from the stash or add to it.
            self.drop_stash(&node.id).await;
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

    /// Repair calls left without results (or results without calls) by a save that failed
    /// mid-turn ([`resume::pair_tool_calls`]), which would otherwise fail every later turn.
    fn pair_tool_calls(&self, node: &Node, agent: &Agent) {
        let changed = agent.with_messages(resume::pair_tool_calls);
        if changed > 0 {
            tracing::warn!(agent = %node.id, changed, "repaired tool calls without results");
        }
    }

    /// Run one turn; returns the error message when it failed (not when it was cancelled). A
    /// failure is posted to the parent when `report` is set.
    async fn run_turn(
        &self,
        node: &Node,
        agent: &Agent,
        input: InvokeInput,
        report: Option<FailureReport>,
    ) -> Option<String> {
        let turn = node.cancel.child_token();
        *node.turn.lock() = Some(turn.clone());
        node.paused.store(false, Ordering::SeqCst);
        *node.dormancy.lock() = Default::default();
        self.set_status(node, AgentStatus::Running);
        self.refresh_memory(agent).await;
        let options = InvokeOptions::default().with_cancel(turn.clone());
        let given = matches!(input, InvokeInput::Text(_) | InvokeInput::Blocks(_));
        self.pair_tool_calls(node, agent);
        let closed = agent.with_messages(resume::close_open_calls);
        if closed > 0 {
            tracing::warn!(agent = %node.id, closed, "answered calls a failed turn left without results");
        }
        let input = agent.with_messages(|messages| resume::fold_into_open_turn(messages, input));
        if given && matches!(input, InvokeInput::None) {
            // The SDK adds no message for it, so nothing would save the mail it holds, which has
            // already left the mailbox, before the turn's first message: a crash would lose it.
            self.save_session(node, agent).await;
        }
        let stream = agent.stream_with(input, options.clone());
        let (mut reply, mut failure) = self.drain_turn(node, stream).await;
        if let Some(error) = failure.as_ref().map(Error::to_string) {
            // One image over the provider's limits fails every call that resends it, so this
            // agent would never recover: replace it by a note and try once more.
            if !turn.is_cancelled() && images::names_image_limit(&error) {
                let removed = agent.with_messages(|messages| images::drop_oversized(messages, &error));
                tracing::warn!(agent = %node.id, removed, %error, "removed images over the model's limit");
                if removed > 0 {
                    let stream = agent.stream_with(InvokeInput::None, options);
                    (reply, failure) = self.drain_turn(node, stream).await;
                }
            }
        }
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
            node.user_asked.store(false, Ordering::SeqCst);
            let _ = self.post(&node.id, &node.parent, "interrupted", INTERRUPTED_TEXT.to_owned());
            self.set_status(node, AgentStatus::Idle);
            return None;
        }
        if let Some(error) = failure {
            let message = error.to_string();
            if let Some(report) = report {
                let _ = self.post(&node.id, &node.parent, "error", report.render(&message));
            }
            self.set_status(node, AgentStatus::Errored(message.clone()));
            return Some(message);
        }
        let reply = reply.unwrap_or_default();
        // The user's own words always get their reply (headless runs wait for it); a root's turn
        // that something else started ends quietly when it says so, like an agent's.
        let asked = node.user_asked.swap(false, Ordering::SeqCst);
        if (node.parent == USER && asked) || !is_silent(&reply) {
            let _ = self.post(&node.id, &node.parent, "reply", reply);
            node.dormancy.lock().replied = true;
        }
        self.set_status(node, AgentStatus::Idle);
        None
    }

    /// Forward a turn's stream to the end; returns its final reply and its error, if any.
    async fn drain_turn(&self, node: &Node, mut stream: AgentStream) -> (Option<String>, Option<Error>) {
        let (mut reply, mut failure) = (None, None);
        while let Some(event) = stream.next().await {
            match event {
                Ok(event) => reply = self.forward(node, &event).or(reply),
                Err(error) => failure = Some(error),
            }
        }
        (reply, failure)
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
                    self.mark_progress(node);
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
                if call.result.status == ToolResultStatus::Success {
                    self.mark_progress(node);
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
    /// Note a sign of progress from `node`: a model reply or a tool call that ended successfully.
    fn mark_progress(&self, node: &Node) {
        node.progress.store(self.now().timestamp(), Ordering::Relaxed);
    }

    /// How long `info`'s agent has shown no sign of progress: since the newest of its last model
    /// reply, successful tool call and (given its branch's `tip` commit time) commit. `None`
    /// unless it is running.
    pub fn quiet_for(&self, info: &AgentInfo, tip: Option<i64>) -> Option<Duration> {
        if info.status != AgentStatus::Running {
            return None;
        }
        let last = info.progress.max(tip.unwrap_or(0));
        let quiet = self.now().timestamp().saturating_sub(last);
        Some(Duration::from_secs(quiet.max(0) as u64))
    }

    pub(super) fn add_usage(&self, node: &Node, usage: &Usage, model: Option<String>, own_call: bool) {
        let counts = usage_counts(usage);
        let model = model.unwrap_or_else(|| "unknown".to_owned());
        let own = own_call.then(|| self.now().timestamp());
        node.ledger.lock().record(&model, counts, own);
        if own_call {
            // One line per call, so cache hits and misses can be read by time and agent later.
            self.journal_event(
                "model_call",
                serde_json::json!({
                    "id": node.id,
                    "model": model,
                    "input": counts.input,
                    "output": counts.output,
                    "cache_read": counts.cache_read,
                    "cache_write": counts.cache_write,
                }),
            );
        }
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
            let point = crate::compaction::compaction_point(window, node.compact_at(&self.inner.config));
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

    /// A `shell` call that is only `cd DIR` moves `node` there ([`crate::workdir`]) instead of
    /// running a command that would change nothing.
    fn install_cd_hook(&self, agent: &Agent, node: &Arc<Node>) {
        let swarm = self.clone();
        let node = Arc::downgrade(node);
        agent
            .hooks()
            .add_sync_callback(move |event: &mut BeforeToolCallEvent| {
                let Some(node) = node.upgrade().filter(|_| event.tool_use.name == "shell") else {
                    return Ok(());
                };
                let command = event.tool_use.input.get("command").and_then(|c| c.as_str());
                let Some(dir) = command.and_then(|c| crate::workdir::lone_cd(c, &node.workdir.get())) else {
                    return Ok(());
                };
                swarm.move_to(&node, &event.agent, dir.clone());
                let text = crate::workdir::moved_text(&dir);
                let tool = FunctionTool::builder("shell")
                    .description("")
                    .handler(move |_input, _ctx| std::future::ready(Ok(text.clone())));
                event.selected_tool = Some(Arc::new(tool));
                Ok(())
            });
    }

    /// Move `node` to `dir`: its commands, the system prompt's working directory, the saved
    /// state and the UI.
    fn move_to(&self, node: &Node, agent: &Agent, dir: PathBuf) {
        let from = node.workdir.get();
        node.workdir.set(dir.clone());
        if let Some(prompt) = agent.system_prompt() {
            agent.set_system_prompt(Some(crate::workdir::moved_prompt(prompt, &from, &dir)));
        }
        self.persist();
        self.emit(SwarmEvent::Moved {
            id: node.id.clone(),
            cwd: dir,
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
        // The user's root has its own wording, so the sentence stays within the pinned prefix.
        let delivery = match node.parent == USER {
            true => format!(
                "Your final reply goes to the user; end a turn they didn't start with just `{SILENT_REPLY}` if nothing is for them."
            ),
            false => format!(
                "Your final reply at the end of each turn is delivered to {reports_to}. When there is nothing worth telling them (e.g. a poll found no change), end with just `{SILENT_REPLY}` and nothing is sent."
            ),
        };
        let swarm = format!(
            "# Swarm\n\
             You are {who}, one of a group of cooperating agents. Your working directory is `{}`; every `shell` call starts there, so there is no need to `cd` into it.{branch}{env}\n\
             Other agents can work in parallel with you; whether and how to split the work is your call. \
             Within one turn, file, shell, job and agent-tool calls run one after another in your order (an \
             `edit`, then the test that needs it); MCP tools, `web_fetch` and `skills` start at once.\n\
             Messages from other agents reach you as `<message from=\"...\">` blocks, either appended to a \
             tool result or as a new turn. {delivery}",
            node.workdir.get().display()
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

/// A final reply that starts with this, ends with it on a line of its own, or is only whitespace,
/// is not delivered to an agent parent. Models often add a line on why before or after the
/// marker; that still means "nothing to tell".
pub const SILENT_REPLY: &str = "[silent]";

/// Whether `reply` ends the turn without a message to the parent ([`SILENT_REPLY`]).
pub(super) fn is_silent(reply: &str) -> bool {
    let reply = reply.trim();
    let marker = |text: &str| text.eq_ignore_ascii_case(SILENT_REPLY);
    reply.is_empty()
        || reply.get(..SILENT_REPLY.len()).is_some_and(marker)
        // A note on why, then the marker on its own last line, is just as silent.
        || reply.lines().last().is_some_and(|last| marker(last.trim()))
}

pub(super) const RETRY_INITIAL: Duration = Duration::from_secs(15);

pub(super) const RETRY_MAX: Duration = Duration::from_secs(600);

/// Whether the `failures`th failed turn in a row is reported: the first, then the 8th, 16th,
/// 32nd… (with the backoff about 26 min, 1h45m, 4h25m… into the streak).
pub(super) fn reports_failure(failures: u32) -> bool {
    failures == 1 || (failures >= 8 && failures.is_power_of_two())
}

/// What the parent is told of a failed turn.
pub(super) struct FailureReport {
    /// Failed turns in a row, this one included.
    pub failures: u32,
    /// When the first of them started.
    pub since: tokio::time::Instant,
    /// The agent retries on its own (it has a goal).
    pub retries: bool,
}

impl FailureReport {
    pub(super) fn render(&self, error: &str) -> String {
        let retry = format!(
            "It retries on its own, backing off up to {} between turns, and reports again if it \
             keeps failing.",
            crate::schedule::human(RETRY_MAX)
        );
        match (self.failures, self.retries) {
            (1, false) => error.to_owned(),
            (1, true) => format!("{error}\n{retry}"),
            (n, _) => format!(
                "Still failing: {n} turns in a row over {}. {retry} Last error: {error}",
                crate::schedule::human(self.since.elapsed())
            ),
        }
    }
}

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

/// `true` when the full delay elapsed, `false` when the agent was cancelled first.
async fn sleep_unless_cancelled(node: &Node, delay: Duration) -> bool {
    tokio::select! {
        _ = tokio::time::sleep(delay) => true,
        _ = node.cancel.cancelled() => false,
    }
}

/// One model call's tokens from its usage metadata; `input` includes cache reads and writes,
/// which Bedrock and Anthropic report beside `input_tokens` but OpenAI's APIs (Bedrock Mantle)
/// count in it ([`total_prompt_tokens`](strands_agents::models::streaming::total_prompt_tokens)).
pub(super) fn usage_counts(usage: &Usage) -> crate::cost::Counts {
    let cache_read = usage.cache_read_input_tokens.unwrap_or(0);
    let cache_write = usage.cache_write_input_tokens.unwrap_or(0);
    crate::cost::Counts {
        calls: 1,
        input: strands_agents::models::streaming::total_prompt_tokens(usage),
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

    use super::{is_silent, summarize_input, usage_counts};
    use strands_agents::types::Usage;

    #[test]
    fn a_reply_that_starts_with_the_silent_marker_is_silent_whatever_follows() {
        for silent in [
            "[silent]",
            " [SILENT]\n",
            "",
            "  \n",
            "[silent]\n\n(poll at 07:05: nothing new)",
            "Nothing new in the inbox or either room.\n\n[silent]\n",
        ] {
            assert!(is_silent(silent), "{silent:?}");
        }
        for loud in ["CI is green", "Nothing new. [silent]", "[sile", "é[silent]"] {
            assert!(!is_silent(loud), "{loud:?}");
        }
    }

    #[test]
    fn counts_cache_tokens_into_the_input_once_whichever_way_the_provider_reports_them() {
        // Bedrock Converse (Claude, and GPT on Bedrock): cache tokens beside `input_tokens`.
        let converse = Usage {
            input_tokens: 0,
            output_tokens: 5,
            total_tokens: 3318,
            cache_read_input_tokens: None,
            cache_write_input_tokens: Some(3313),
        };
        let counts = usage_counts(&converse);
        assert_eq!((counts.input, counts.cache_write), (3313, 3313));
        // OpenAI's Responses API (Bedrock Mantle): `input_tokens` includes the cached ones.
        let responses = Usage {
            input_tokens: 1851,
            output_tokens: 5,
            total_tokens: 1856,
            cache_read_input_tokens: Some(1800),
            cache_write_input_tokens: None,
        };
        let counts = usage_counts(&responses);
        assert_eq!((counts.input, counts.cache_read), (1851, 1800));
    }

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
