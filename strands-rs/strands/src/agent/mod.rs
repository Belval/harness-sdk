//! Core agent: the event loop, builder, result type, and invocation state.
//!
//! Ports the essential agent loop from `agent/agent.ts` (`stream`'s resume loop,
//! `_stream`, `_invokeModel`, `executeTools`) and `AgentResult` from
//! `types/agent.ts`. The loop fires the [`crate::hooks`] lifecycle events at each
//! point and honors their control fields (`cancel`, `retry`, `selected_tool`,
//! `resume`, `end_turn`, and the mutable `tool_use` / `result`), raises and
//! resumes [`crate::interrupt`]s, emits [`crate::telemetry`] spans around the
//! agent, each cycle, model calls, and tool calls, and runs model calls and tool
//! executions through their [`crate::middleware`] stacks.
//!
//! The checkpoint, session, structured-output, and cancellation surfaces of the
//! TypeScript loop are out of scope for the vertical slice; the control flow they
//! wrap is preserved.

mod builder;
mod invocation;
mod result;
mod state;

pub use builder::AgentBuilder;
pub use invocation::InvocationState;
pub use result::AgentResult;
pub use state::{AgentHandle, AgentState, Messages};

use std::collections::HashMap;
use std::sync::Arc;

use crate::conversation_manager::ConversationManager;
use crate::errors::StrandsError;
use crate::hooks::{
    AfterInvocationEvent, AfterModelCallEvent, AfterToolCallEvent, AfterToolsEvent,
    AgentResultEvent, BeforeInvocationEvent, BeforeModelCallEvent, BeforeToolCallEvent,
    BeforeToolsEvent, ContentBlockEvent, HookCleanup, HookEndTurn, HookEvent, HookFuture,
    HookProvider, HookRegistry, InterruptEvent, MessageAddedEvent, ModelMessageEvent,
    ModelStopData, ToolResultEvent, ToolUseData,
};
use crate::interrupt::{InterruptError, InterruptState, PendingToolExecution};
use crate::middleware::{
    ExecuteToolContext, InvokeModelContext, MiddlewareStack, ToolExecutionResult,
};
use crate::models::{Model, StreamAggregatedResult, StreamOptions};
use crate::telemetry::Tracer;
use crate::tools::{Tool, ToolRegistry};
use crate::types::interrupt::InterruptResponse;
use crate::types::messages::{
    ContentBlock, Message, Role, StopReason, SystemPrompt, ToolResultBlock, ToolResultContent,
    ToolResultStatus, ToolUseBlock,
};

/// Upper bound on agent-loop cycles per invocation.
///
/// The TypeScript loop is bounded by hook-driven limits (`InvokeOptions.limits`);
/// the vertical slice omits that surface, so this guard prevents an unbounded
/// tool-call loop. Reaching it stops the turn with [`StopReason::EndTurn`].
const MAX_LOOP_ITERATIONS: usize = 100;

/// Upper bound on conversation-manager context reductions per model call, so a
/// manager that reports success without actually shrinking history cannot loop
/// forever on a persistent context-window overflow.
const MAX_CONTEXT_REDUCTIONS: usize = 10;

/// Whether the loop should stop with a result or run another cycle.
enum CycleOutcome {
    /// The turn is complete; return this result.
    Done(AgentResult),
    /// Run another cycle.
    Continue,
}

/// The outcome of running the tools for one model turn.
struct ToolsExecutionResult {
    /// The user message carrying every tool result.
    message: Message,
    /// Set when a hook requested the turn end early after tools.
    end_turn: Option<HookEndTurn>,
}

/// A model-driven agent.
///
/// Drives the loop: send the conversation to the model, and while the model
/// requests tool use, run the tools and feed their results back until the model
/// stops requesting tools. Lifecycle [`crate::hooks`] events fire throughout.
pub struct Agent {
    /// The conversation history, a shared handle so hooks and conversation
    /// managers can read and rewrite it. Read a snapshot via [`Agent::messages`].
    messages: Messages,
    /// The system prompt, if any.
    pub system_prompt: Option<SystemPrompt>,
    /// Human-readable agent name, used in telemetry (`gen_ai.agent.name`).
    pub name: String,
    /// Stable agent identifier, used in telemetry (`gen_ai.agent.id`).
    pub id: String,
    model: Arc<dyn Model>,
    tool_registry: ToolRegistry,
    hooks: HookRegistry,
    interrupt_state: InterruptState,
    state: AgentState,
    conversation_manager: Option<Arc<dyn ConversationManager>>,
    tracer: Tracer,
    invoke_model_mw: MiddlewareStack<InvokeModelContext, StreamAggregatedResult>,
    execute_tool_mw: MiddlewareStack<ExecuteToolContext, ToolExecutionResult>,
}

impl Agent {
    /// Returns a builder for constructing an [`Agent`].
    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        model: Arc<dyn Model>,
        name: String,
        system_prompt: Option<SystemPrompt>,
        messages: Vec<Message>,
        tool_registry: ToolRegistry,
        hooks: HookRegistry,
        state: AgentState,
        conversation_manager: Option<Arc<dyn ConversationManager>>,
    ) -> Self {
        Agent {
            messages: Messages::new(messages),
            system_prompt,
            name,
            id: crate::types::messages::generate_tracking_id(),
            model,
            tool_registry,
            hooks,
            interrupt_state: InterruptState::new(),
            state,
            conversation_manager,
            tracer: Tracer::new(),
            invoke_model_mw: MiddlewareStack::new(),
            execute_tool_mw: MiddlewareStack::new(),
        }
    }

    /// The agent's persisted state, shared with hook callbacks via
    /// [`AgentHandle`]. Ports `agent.state`.
    pub fn state(&self) -> &AgentState {
        &self.state
    }

    /// A snapshot of the conversation history. Ports reading `agent.messages`.
    pub fn messages(&self) -> Vec<Message> {
        self.messages.snapshot()
    }

    /// The shared conversation-history handle, for advanced in-place access.
    pub fn messages_handle(&self) -> &Messages {
        &self.messages
    }

    /// The conversation manager, if one is configured. Ports
    /// `agent.conversation_manager`.
    pub fn conversation_manager(&self) -> Option<&Arc<dyn ConversationManager>> {
        self.conversation_manager.as_ref()
    }

    /// Builds the hook-facing handle passed to events fired this loop.
    fn agent_handle(&self) -> AgentHandle {
        AgentHandle::new(
            self.state.clone(),
            self.messages.clone(),
            self.model.model_id().map(str::to_string),
        )
    }

    /// The middleware stack wrapping model invocations, for registering handlers.
    /// Ports the `InvokeModelStage`.
    pub fn invoke_model_middleware(
        &self,
    ) -> &MiddlewareStack<InvokeModelContext, StreamAggregatedResult> {
        &self.invoke_model_mw
    }

    /// The middleware stack wrapping tool executions, for registering handlers.
    /// Ports the `ExecuteToolStage`.
    pub fn execute_tool_middleware(
        &self,
    ) -> &MiddlewareStack<ExecuteToolContext, ToolExecutionResult> {
        &self.execute_tool_mw
    }

    /// The tools registered on this agent, in registration order.
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tool_registry.list()
    }

    /// The hook registry backing this agent, for advanced registration.
    pub fn hooks(&self) -> &HookRegistry {
        &self.hooks
    }

    /// Registers a hook callback for event type `E` at the default order.
    ///
    /// Returns a [`HookCleanup`] that removes the callback when invoked. Ports
    /// `agent.addHook`.
    pub fn add_hook<E, F>(&self, callback: F) -> HookCleanup
    where
        E: HookEvent + 'static,
        F: Fn(&mut E) -> Result<(), StrandsError> + Send + Sync + 'static,
    {
        self.hooks.add_callback(callback)
    }

    /// Registers a hook callback for event type `E` with an explicit order.
    pub fn add_hook_with_order<E, F>(&self, callback: F, order: i32) -> HookCleanup
    where
        E: HookEvent + 'static,
        F: Fn(&mut E) -> Result<(), StrandsError> + Send + Sync + 'static,
    {
        self.hooks.add_callback_with_order(callback, order)
    }

    /// Registers an asynchronous hook callback for event type `E`. The callback
    /// returns a boxed future that may borrow the event across `await` points.
    pub fn add_hook_async<E, F>(&self, callback: F) -> HookCleanup
    where
        E: HookEvent + 'static,
        F: for<'a> Fn(&'a mut E) -> HookFuture<'a> + Send + Sync + 'static,
    {
        self.hooks.add_callback_async(callback)
    }

    /// Registers all of a [`HookProvider`]'s callbacks on this agent, for
    /// post-build registration. Ports registering a `HookProvider`.
    pub fn add_hook_provider(&self, provider: &impl HookProvider) {
        provider.register_hooks(&self.hooks);
    }

    /// Runs the agent loop with a text prompt, returning the final result.
    ///
    /// Appends the prompt as a user message, then drives the loop to completion.
    pub async fn invoke(&mut self, prompt: impl Into<String>) -> Result<AgentResult, StrandsError> {
        self.invoke_message(Message::user(prompt.into())).await
    }

    /// Runs the agent loop starting from a caller-constructed user message.
    ///
    /// # Errors
    /// Returns an error if the agent is waiting on an interrupt — resume with
    /// [`Agent::resume`] before starting a new turn.
    pub async fn invoke_message(&mut self, message: Message) -> Result<AgentResult, StrandsError> {
        if self.interrupt_state.is_activated() {
            return Err(StrandsError::model(
                "Agent is in an interrupted state. Resume with `Agent::resume` before invoking.",
            ));
        }
        self.run(Some(message), InvocationState::new()).await
    }

    /// Resumes a turn that halted on an interrupt, supplying the human responses.
    ///
    /// Applies the responses to the matching interrupts, then re-enters the loop:
    /// a pending tool execution is replayed without re-invoking the model, and
    /// the previously interrupted tool/hook `interrupt(...)` calls now return
    /// their responses. Ports resuming via `invoke` with interrupt-response
    /// content blocks.
    ///
    /// # Errors
    /// Returns an error if the agent is not in an interrupted state, or if a
    /// response references an unknown interrupt id.
    pub async fn resume(
        &mut self,
        responses: Vec<InterruptResponse>,
    ) -> Result<AgentResult, StrandsError> {
        if !self.interrupt_state.is_activated() {
            return Err(StrandsError::model(
                "Agent is not in an interrupted state; call `invoke` to start a new turn.",
            ));
        }
        self.interrupt_state.resume(responses)?;
        self.run(None, InvocationState::new()).await
    }

    /// The interrupt state backing this agent, for inspection and serialization.
    pub fn interrupt_state(&self) -> &InterruptState {
        &self.interrupt_state
    }

    /// The resume loop: brackets each pass with [`BeforeInvocationEvent`] /
    /// [`AfterInvocationEvent`] and re-enters when a hook sets `resume`. Ports
    /// the `while (true)` resume loop in `stream()`.
    async fn run(
        &mut self,
        mut new_input: Option<Message>,
        state: InvocationState,
    ) -> Result<AgentResult, StrandsError> {
        let hooks = self.hooks.clone();
        loop {
            let mut before = BeforeInvocationEvent {
                invocation_state: state.clone(),
                agent: self.agent_handle(),
                cancel: None,
            };
            hooks.invoke_callbacks(&mut before).await?;

            if let Some(cancel) = &before.cancel {
                let message =
                    Message::assistant(cancel.message("invocation denied by hook").to_string());
                self.append_message(message.clone(), &state).await?;
                let mut after = AfterInvocationEvent {
                    invocation_state: state.clone(),
                    agent: self.agent_handle(),
                    resume: None,
                };
                hooks.invoke_callbacks(&mut after).await?;
                return Ok(AgentResult::new(StopReason::EndTurn, message));
            }

            let core_result = self.stream_core(new_input.take(), &state).await;

            // AfterInvocationEvent always fires — even on error — before the
            // error propagates or resume is honored.
            let mut after = AfterInvocationEvent {
                invocation_state: state.clone(),
                agent: self.agent_handle(),
                resume: None,
            };
            hooks.invoke_callbacks(&mut after).await?;

            let result = core_result?;

            if let Some(resume) = after.resume.take() {
                new_input = Some(resume);
                continue;
            }

            let mut agent_result = AgentResultEvent {
                result: result.clone(),
                invocation_state: state.clone(),
                agent: self.agent_handle(),
            };
            hooks.invoke_callbacks(&mut agent_result).await?;

            return Ok(result);
        }
    }

    /// A single pass through the main loop, bracketed by the agent span. Ports
    /// `_stream`'s span lifecycle around the cycle loop.
    async fn stream_core(
        &mut self,
        new_input: Option<Message>,
        state: &InvocationState,
    ) -> Result<AgentResult, StrandsError> {
        let tool_names: Vec<String> = self
            .tool_registry
            .list()
            .iter()
            .map(|tool| tool.name().to_string())
            .collect();
        let system_prompt = self.system_prompt.as_ref().map(SystemPrompt::text);
        let agent_span = self.tracer.start_agent_span(
            &self.name,
            &self.id,
            self.model.model_id(),
            &tool_names,
            system_prompt.as_deref(),
        );

        let outcome = self.run_cycles(new_input, state).await;

        match &outcome {
            Ok(_) => self.tracer.end_agent_span(&agent_span, None, None),
            Err(error) => self
                .tracer
                .end_agent_span(&agent_span, None, Some(&error.to_string())),
        }
        outcome
    }

    /// The cycle loop: send the conversation to the model, run any requested
    /// tools, and repeat, one `execute_agent_loop_cycle` span per iteration.
    async fn run_cycles(
        &mut self,
        mut new_input: Option<Message>,
        state: &InvocationState,
    ) -> Result<AgentResult, StrandsError> {
        let mut iterations = 0;

        loop {
            if iterations >= MAX_LOOP_ITERATIONS {
                return Ok(AgentResult::new(StopReason::EndTurn, self.last_message()));
            }
            iterations += 1;

            let cycle_span = self.tracer.start_cycle_span(&format!("cycle-{iterations}"));
            match self.run_one_cycle(&mut new_input, state).await {
                Ok(CycleOutcome::Done(result)) => {
                    self.tracer.end_cycle_span(&cycle_span, None);
                    return Ok(result);
                }
                Ok(CycleOutcome::Continue) => {
                    self.tracer.end_cycle_span(&cycle_span, None);
                }
                Err(error) => {
                    self.tracer
                        .end_cycle_span(&cycle_span, Some(&error.to_string()));
                    return Err(error);
                }
            }
        }
    }

    /// Runs one loop cycle: a model call (or pending-execution replay) followed
    /// by tool execution. Returns whether the loop should stop or continue.
    async fn run_one_cycle(
        &mut self,
        new_input: &mut Option<Message>,
        state: &InvocationState,
    ) -> Result<CycleOutcome, StrandsError> {
        // Resuming from a tool interrupt reuses the stored assistant message and
        // completed results, skipping the model call. Ports the
        // `getPendingExecution` short-circuit.
        let (assistant_message, completed) = match self.interrupt_state.get_pending_execution() {
            Some(pending) => (
                pending.assistant_message,
                Some(pending.completed_tool_results),
            ),
            None => {
                if let Some(message) = new_input.take() {
                    self.append_message(message, state).await?;
                }

                let model_result = self.invoke_model(state).await?;

                if model_result.stop_reason != StopReason::ToolUse {
                    self.append_message(model_result.message.clone(), state)
                        .await?;
                    return Ok(CycleOutcome::Done(AgentResult::new(
                        model_result.stop_reason,
                        model_result.message,
                    )));
                }
                (model_result.message, None)
            }
        };

        let tools_result = match self
            .execute_tools(&assistant_message, state, completed)
            .await
        {
            Ok(tools_result) => tools_result,
            Err(StrandsError::Interrupt(interrupt_error)) => {
                // execute_tools stored the pending execution before propagating;
                // stop the turn to wait for human input.
                return Ok(CycleOutcome::Done(
                    self.stop_for_interrupt(interrupt_error, state).await?,
                ));
            }
            Err(error) => return Err(error),
        };

        // Deferred append: both messages are pushed together after tools run, so
        // history never holds a tool-use without its matching results.
        self.append_message(assistant_message, state).await?;
        self.append_message(tools_result.message.clone(), state)
            .await?;

        // The pair is in history, so any stored pending execution is stale; clear
        // it and leave the interrupted state so fresh interrupts can be raised.
        self.interrupt_state.clear_pending_tool_execution();
        if self.interrupt_state.is_activated() {
            self.interrupt_state.deactivate();
        }

        if let Some(end_turn) = &tools_result.end_turn {
            let text = end_turn
                .content("Turn ended early by hook after tool execution")
                .to_string();
            let message = Message::assistant(text);
            self.append_message(message.clone(), state).await?;
            return Ok(CycleOutcome::Done(AgentResult::new(
                StopReason::EndTurn,
                message,
            )));
        }

        Ok(CycleOutcome::Continue)
    }

    /// Registers the raised interrupts, activates the interrupted state, fires an
    /// [`InterruptEvent`] per unanswered interrupt, and returns an interrupt
    /// result. Ports `_createInterruptResult` plus the interrupt fan-out.
    async fn stop_for_interrupt(
        &self,
        error: InterruptError,
        state: &InvocationState,
    ) -> Result<AgentResult, StrandsError> {
        let hooks = self.hooks.clone();
        for interrupt in &error.interrupts {
            self.interrupt_state.register_interrupt(interrupt);
        }
        self.interrupt_state.activate();

        let unanswered = self.interrupt_state.get_unanswered_interrupts();
        for interrupt in &unanswered {
            let mut event = InterruptEvent {
                interrupt: interrupt.clone(),
                invocation_state: state.clone(),
                agent: self.agent_handle(),
            };
            hooks.invoke_callbacks(&mut event).await?;
        }

        let last_message = self.messages.last().unwrap_or_else(|| {
            Message::new(Role::Assistant, vec![ContentBlock::text("Interrupted")])
        });
        Ok(AgentResult::with_interrupts(
            StopReason::Interrupt,
            last_message,
            unanswered,
        ))
    }

    /// Invokes the model, firing [`BeforeModelCallEvent`], per-block
    /// [`ContentBlockEvent`], [`ModelMessageEvent`], and [`AfterModelCallEvent`],
    /// and honoring `cancel` / `retry`. Ports `_invokeModel`.
    async fn invoke_model(
        &self,
        state: &InvocationState,
    ) -> Result<StreamAggregatedResult, StrandsError> {
        let hooks = self.hooks.clone();
        let mut attempt_count = 1;
        let mut reduce_attempts = 0;

        loop {
            let mut before = BeforeModelCallEvent {
                invocation_state: state.clone(),
                agent: self.agent_handle(),
                cancel: None,
            };
            hooks.invoke_callbacks(&mut before).await?;

            if let Some(cancel) = &before.cancel {
                let message =
                    Message::assistant(cancel.message("model call denied by hook").to_string());
                let mut after = AfterModelCallEvent {
                    invocation_state: state.clone(),
                    agent: self.agent_handle(),
                    attempt_count,
                    stop_data: Some(ModelStopData {
                        message: message.clone(),
                        stop_reason: StopReason::EndTurn,
                    }),
                    error: None,
                    retry: false,
                };
                hooks.invoke_callbacks(&mut after).await?;
                if after.retry {
                    attempt_count += 1;
                    continue;
                }
                return Ok(StreamAggregatedResult {
                    message,
                    stop_reason: StopReason::EndTurn,
                    usage: None,
                    metrics: None,
                });
            }

            // The model call runs through the InvokeModelStage middleware; the
            // terminal performs the actual call and owns the model span so the
            // span records the post-middleware request. Ports
            // `_invokeModelWithMiddleware`.
            let context = InvokeModelContext {
                messages: self.messages.snapshot(),
                system_prompt: self.system_prompt.clone(),
                tool_specs: self.tool_registry.tool_specs(),
                tool_choice: None,
                invocation_state: state.clone(),
            };
            let model = self.model.clone();
            let tracer = self.tracer.clone();
            let model_id = self.model.model_id().map(str::to_string);
            let terminal = move |context: InvokeModelContext| {
                let model = model.clone();
                let tracer = tracer.clone();
                let model_id = model_id.clone();
                async move {
                    let options = StreamOptions {
                        system_prompt: context.system_prompt,
                        tool_specs: context.tool_specs,
                        tool_choice: context.tool_choice,
                    };
                    let model_span = tracer.start_model_span(model_id.as_deref());
                    let result = model.stream_aggregated(&context.messages, &options).await;
                    match &result {
                        Ok(aggregated) => tracer.end_model_span(
                            &model_span,
                            aggregated.usage.as_ref(),
                            aggregated.metrics.as_ref(),
                            None,
                        ),
                        Err(error) => {
                            tracer.end_model_span(&model_span, None, None, Some(&error.to_string()))
                        }
                    }
                    result
                }
            };

            match self.invoke_model_mw.invoke(context, terminal).await {
                Ok(result) => {
                    for block in &result.message.content {
                        let mut content_block = ContentBlockEvent {
                            content_block: block.clone(),
                            invocation_state: state.clone(),
                            agent: self.agent_handle(),
                        };
                        hooks.invoke_callbacks(&mut content_block).await?;
                    }

                    let mut model_message = ModelMessageEvent {
                        message: result.message.clone(),
                        stop_reason: result.stop_reason.clone(),
                        invocation_state: state.clone(),
                        agent: self.agent_handle(),
                    };
                    hooks.invoke_callbacks(&mut model_message).await?;

                    let mut after = AfterModelCallEvent {
                        invocation_state: state.clone(),
                        agent: self.agent_handle(),
                        attempt_count,
                        stop_data: Some(ModelStopData {
                            message: result.message.clone(),
                            stop_reason: result.stop_reason.clone(),
                        }),
                        error: None,
                        retry: false,
                    };
                    hooks.invoke_callbacks(&mut after).await?;
                    if after.retry {
                        attempt_count += 1;
                        continue;
                    }
                    return Ok(result);
                }
                Err(error) => {
                    // Context-window overflow: ask the conversation manager to
                    // reduce the history, then retry. Bounded so a manager that
                    // reports success without shrinking cannot loop forever.
                    if matches!(error, StrandsError::ContextWindowOverflow(_)) {
                        if let Some(manager) = &self.conversation_manager {
                            if reduce_attempts < MAX_CONTEXT_REDUCTIONS
                                && manager
                                    .reduce_context(&self.agent_handle(), Some(&error.to_string()))
                                    .await
                                    .is_ok()
                            {
                                reduce_attempts += 1;
                                continue;
                            }
                        }
                    }

                    let mut after = AfterModelCallEvent {
                        invocation_state: state.clone(),
                        agent: self.agent_handle(),
                        attempt_count,
                        stop_data: None,
                        error: Some(error.to_string()),
                        retry: false,
                    };
                    hooks.invoke_callbacks(&mut after).await?;
                    if after.retry {
                        attempt_count += 1;
                        continue;
                    }
                    return Err(error);
                }
            }
        }
    }

    /// Runs every tool in the assistant message, firing [`BeforeToolsEvent`],
    /// per-tool events, [`ToolResultEvent`], and [`AfterToolsEvent`]. Ports
    /// `executeTools` and the sequential executor's core.
    async fn execute_tools(
        &self,
        assistant_message: &Message,
        state: &InvocationState,
        completed: Option<HashMap<String, ToolResultBlock>>,
    ) -> Result<ToolsExecutionResult, StrandsError> {
        let hooks = self.hooks.clone();
        let completed = completed.unwrap_or_default();

        let mut before = BeforeToolsEvent {
            message: assistant_message.clone(),
            invocation_state: state.clone(),
            agent: self.agent_handle(),
            cancel: None,
            interrupt_state: self.interrupt_state.clone(),
        };
        // A BeforeTools hook interrupt stores pending state with no completed
        // results before propagating, so the whole batch replays on resume.
        if let Err(error) = hooks.invoke_callbacks(&mut before).await {
            if matches!(error, StrandsError::Interrupt(_)) {
                self.interrupt_state
                    .set_pending_tool_execution(PendingToolExecution {
                        assistant_message: assistant_message.clone(),
                        completed_tool_results: completed,
                    });
            }
            return Err(error);
        }

        let tool_uses: Vec<ToolUseBlock> = assistant_message
            .content
            .iter()
            .filter_map(ContentBlock::as_tool_use)
            .cloned()
            .collect();

        let mut result_blocks: Vec<ToolResultBlock> = Vec::new();
        // Accumulates results as they complete, keyed by model-issued id, so an
        // interrupt mid-batch can store what finished for a clean resume.
        let mut results_by_id: HashMap<String, ToolResultBlock> = completed.clone();

        if let Some(cancel) = &before.cancel {
            let message = cancel.message("Tool cancelled by hook").to_string();
            for tool_use in &tool_uses {
                let result = error_result(&tool_use.tool_use_id, &message);
                let mut event = ToolResultEvent {
                    result: result.clone(),
                    invocation_state: state.clone(),
                    agent: self.agent_handle(),
                };
                hooks.invoke_callbacks(&mut event).await?;
                result_blocks.push(result);
            }
        } else {
            for tool_use in &tool_uses {
                // On resume, a tool that already completed is restored without
                // replaying its lifecycle events. Ports the sequential executor's
                // completed-result skip.
                if let Some(done) = completed.get(&tool_use.tool_use_id) {
                    result_blocks.push(done.clone());
                    continue;
                }

                match self.execute_single_tool(tool_use, state).await {
                    Ok(result) => {
                        let mut event = ToolResultEvent {
                            result: result.clone(),
                            invocation_state: state.clone(),
                            agent: self.agent_handle(),
                        };
                        hooks.invoke_callbacks(&mut event).await?;
                        results_by_id.insert(tool_use.tool_use_id.clone(), result.clone());
                        result_blocks.push(result);
                    }
                    Err(error) if matches!(error, StrandsError::Interrupt(_)) => {
                        self.interrupt_state
                            .set_pending_tool_execution(PendingToolExecution {
                                assistant_message: assistant_message.clone(),
                                completed_tool_results: results_by_id,
                            });
                        return Err(error);
                    }
                    Err(error) => return Err(error),
                }
            }
        }

        let tool_result_message = Message::new(
            Role::User,
            result_blocks
                .into_iter()
                .map(ContentBlock::ToolResult)
                .collect(),
        );

        let mut after = AfterToolsEvent {
            message: tool_result_message.clone(),
            invocation_state: state.clone(),
            agent: self.agent_handle(),
            end_turn: None,
        };
        hooks.invoke_callbacks(&mut after).await?;

        Ok(ToolsExecutionResult {
            message: tool_result_message,
            end_turn: after.end_turn,
        })
    }

    /// Runs one tool with its [`BeforeToolCallEvent`] / [`AfterToolCallEvent`]
    /// bracket, honoring `cancel`, `selected_tool`, `tool_use` mutation, and
    /// `retry`. Ports the per-tool `executeTool` loop.
    async fn execute_single_tool(
        &self,
        tool_use_block: &ToolUseBlock,
        state: &InvocationState,
    ) -> Result<ToolResultBlock, StrandsError> {
        let hooks = self.hooks.clone();
        let original_name = tool_use_block.name.clone();
        let registry_tool = self.tool_registry.resolve(&original_name).ok().cloned();

        let mut tool_use = ToolUseData {
            name: tool_use_block.name.clone(),
            tool_use_id: tool_use_block.tool_use_id.clone(),
            input: tool_use_block.input.clone(),
        };

        loop {
            let mut before = BeforeToolCallEvent {
                tool_use: tool_use.clone(),
                tool: registry_tool.clone(),
                invocation_state: state.clone(),
                agent: self.agent_handle(),
                cancel: None,
                selected_tool: None,
                interrupt_state: self.interrupt_state.clone(),
            };
            hooks.invoke_callbacks(&mut before).await?;

            // Adopt the (possibly mutated) tool_use but keep the model-issued id.
            tool_use = ToolUseData {
                tool_use_id: tool_use_block.tool_use_id.clone(),
                ..before.tool_use
            };

            // selected_tool wins; otherwise re-resolve a renamed tool, else keep
            // the original registry match — resolved before cancel so
            // AfterToolCallEvent reports the same effective tool on every path.
            let effective_tool = before.selected_tool.clone().or_else(|| {
                if tool_use.name != original_name {
                    self.tool_registry.resolve(&tool_use.name).ok().cloned()
                } else {
                    registry_tool.clone()
                }
            });

            if let Some(cancel) = &before.cancel {
                let message = cancel.message("Tool cancelled by hook").to_string();
                let result = error_result(&tool_use.tool_use_id, &message);
                let mut after = AfterToolCallEvent {
                    tool_use: tool_use.clone(),
                    tool: effective_tool.clone(),
                    result,
                    error: None,
                    invocation_state: state.clone(),
                    agent: self.agent_handle(),
                    retry: false,
                };
                hooks.invoke_callbacks(&mut after).await?;
                if after.retry {
                    continue;
                }
                return Ok(normalize_tool_result_id(
                    after.result,
                    &tool_use_block.tool_use_id,
                ));
            }

            // Tool execution runs through the ExecuteToolStage middleware; the
            // terminal performs the actual call and owns the tool span. Ports
            // `_executeToolWithMiddleware`.
            let context = ExecuteToolContext {
                tool: effective_tool.clone(),
                tool_use: tool_use.clone(),
                invocation_state: state.clone(),
            };
            let tracer = self.tracer.clone();
            let interrupt_state = self.interrupt_state.clone();
            let original_id = tool_use_block.tool_use_id.clone();
            let terminal = move |context: ExecuteToolContext| {
                let tracer = tracer.clone();
                let interrupt_state = interrupt_state.clone();
                let original_id = original_id.clone();
                async move {
                    let tool_span = tracer
                        .start_tool_span(&context.tool_use.name, &context.tool_use.tool_use_id);
                    let (result, error) = match &context.tool {
                        Some(tool) => {
                            let block = ToolUseBlock {
                                name: context.tool_use.name.clone(),
                                tool_use_id: context.tool_use.tool_use_id.clone(),
                                input: context.tool_use.input.clone(),
                                reasoning_signature: None,
                            };
                            // An interrupt raised inside the tool propagates as
                            // `Err`; an ordinary failure becomes an error result.
                            match crate::tools::execute_tool_reporting_error(
                                tool.as_ref(),
                                block,
                                interrupt_state,
                            )
                            .await
                            {
                                Ok(pair) => pair,
                                Err(error) => {
                                    tracer.end_tool_span(
                                        &tool_span,
                                        "error",
                                        Some(&error.to_string()),
                                    );
                                    return Err(error);
                                }
                            }
                        }
                        None => {
                            let message = format!("Tool '{}' not found", context.tool_use.name);
                            (
                                error_result(&context.tool_use.tool_use_id, &message),
                                Some(message),
                            )
                        }
                    };
                    let status = match result.status {
                        ToolResultStatus::Success => "success",
                        ToolResultStatus::Error => "error",
                    };
                    tracer.end_tool_span(&tool_span, status, error.as_deref());
                    let result = normalize_tool_result_id(result, &original_id);
                    Ok(ToolExecutionResult { result, error })
                }
            };

            let execution = self.execute_tool_mw.invoke(context, terminal).await?;
            let (result, error) = (execution.result, execution.error);

            let mut after = AfterToolCallEvent {
                tool_use: tool_use.clone(),
                tool: effective_tool.clone(),
                result,
                error,
                invocation_state: state.clone(),
                agent: self.agent_handle(),
                retry: false,
            };
            hooks.invoke_callbacks(&mut after).await?;
            if after.retry {
                continue;
            }
            return Ok(normalize_tool_result_id(
                after.result,
                &tool_use_block.tool_use_id,
            ));
        }
    }

    /// Pushes `message` into history and fires [`MessageAddedEvent`].
    async fn append_message(
        &mut self,
        message: Message,
        state: &InvocationState,
    ) -> Result<(), StrandsError> {
        let hooks = self.hooks.clone();
        self.messages.push(message.clone());
        let mut event = MessageAddedEvent {
            message,
            invocation_state: state.clone(),
            agent: self.agent_handle(),
        };
        hooks.invoke_callbacks(&mut event).await
    }

    fn last_message(&self) -> Message {
        self.messages
            .last()
            .unwrap_or_else(|| Message::new(Role::Assistant, vec![ContentBlock::text("")]))
    }
}

/// Builds an error tool-result block carrying `message` as its text.
fn error_result(tool_use_id: &str, message: &str) -> ToolResultBlock {
    ToolResultBlock {
        tool_use_id: tool_use_id.to_string(),
        status: ToolResultStatus::Error,
        content: vec![ToolResultContent::Text(message.to_string())],
    }
}

/// Ensures a tool result carries the model-issued `tool_use_id`, so resume and
/// provider correlation match the assistant's tool-use blocks. Ports
/// `_normalizeToolResultId`.
fn normalize_tool_result_id(result: ToolResultBlock, tool_use_id: &str) -> ToolResultBlock {
    if result.tool_use_id == tool_use_id {
        return result;
    }
    ToolResultBlock {
        tool_use_id: tool_use_id.to_string(),
        status: result.status,
        content: result.content,
    }
}
