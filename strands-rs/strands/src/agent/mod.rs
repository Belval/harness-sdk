//! Core agent: the event loop, builder, result type, and invocation state.
//!
//! Ports the essential agent loop from `agent/agent.ts` (`stream`'s resume loop,
//! `_stream`, `_invokeModel`, `executeTools`) and `AgentResult` from
//! `types/agent.ts`. The loop fires the [`crate::hooks`] lifecycle events at each
//! point and honors their control fields (`cancel`, `retry`, `selected_tool`,
//! `resume`, `end_turn`, and the mutable `tool_use` / `result`), raises and
//! resumes [`crate::interrupt`]s, emits [`crate::telemetry`] spans around the
//! agent, each cycle, model calls, and tool calls, and runs model calls and tool
//! executions through their [`crate::middleware`] stacks, and captures
//! structured output via a synthetic tool when a schema is configured.
//!
//! The checkpoint, session, and cancellation surfaces of the TypeScript loop are
//! out of scope for the vertical slice; the control flow they wrap is preserved.

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
use crate::session::SessionManager;
use crate::telemetry::{AttributeValue, Tracer};
use crate::tools::executor::{
    SequentialToolExecutor, ToolExecutionContext, ToolExecutor, ToolsExecutionResult,
};
use crate::tools::{Tool, ToolProvider, ToolRegistry};
use crate::types::interrupt::InterruptResponse;
use crate::types::messages::{
    ContentBlock, Message, Role, StopReason, SystemPrompt, ToolResultBlock, ToolResultContent,
    ToolResultStatus, ToolUseBlock,
};
use crate::types::tools::{ToolChoice, ToolSpec};

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

/// Name of the synthetic tool used to capture structured output. Ports
/// `STRUCTURED_OUTPUT_TOOL_NAME`.
const STRUCTURED_OUTPUT_TOOL_NAME: &str = "strands_structured_output";

/// Whether the loop should stop with a result or run another cycle.
enum CycleOutcome {
    /// The turn is complete; return this result.
    Done(AgentResult),
    /// Run another cycle.
    Continue,
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
    tool_providers: Vec<Arc<dyn ToolProvider>>,
    providers_loaded: bool,
    hooks: HookRegistry,
    interrupt_state: InterruptState,
    state: AgentState,
    conversation_manager: Option<Arc<dyn ConversationManager>>,
    session_manager: Option<Arc<dyn SessionManager>>,
    structured_output_schema: Option<serde_json::Value>,
    trace_attributes: HashMap<String, AttributeValue>,
    tracer: Tracer,
    invoke_model_mw: MiddlewareStack<InvokeModelContext, StreamAggregatedResult>,
    execute_tool_mw: MiddlewareStack<ExecuteToolContext, ToolExecutionResult>,
    tool_executor: Arc<dyn ToolExecutor>,
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
        tool_providers: Vec<Arc<dyn ToolProvider>>,
        hooks: HookRegistry,
        state: AgentState,
        conversation_manager: Option<Arc<dyn ConversationManager>>,
        session_manager: Option<Arc<dyn SessionManager>>,
        structured_output_schema: Option<serde_json::Value>,
        trace_attributes: HashMap<String, AttributeValue>,
        tool_executor: Arc<dyn ToolExecutor>,
    ) -> Self {
        Agent {
            messages: Messages::new(messages),
            system_prompt,
            name,
            id: crate::types::messages::generate_tracking_id(),
            model,
            tool_registry,
            tool_providers,
            providers_loaded: false,
            hooks,
            interrupt_state: InterruptState::new(),
            state,
            conversation_manager,
            session_manager,
            structured_output_schema,
            trace_attributes,
            tracer: Tracer::new(),
            invoke_model_mw: MiddlewareStack::new(),
            execute_tool_mw: MiddlewareStack::new(),
            tool_executor,
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

    /// The session manager, if one is configured. Ports `agent._session_manager`.
    pub fn session_manager(&self) -> Option<&Arc<dyn SessionManager>> {
        self.session_manager.as_ref()
    }

    /// Builds the hook-facing handle passed to events fired this loop.
    fn agent_handle(&self) -> AgentHandle {
        AgentHandle::new(
            self.state.clone(),
            self.messages.clone(),
            self.model.model_id().map(str::to_string),
            self.model.get_config(),
            self.session_manager.clone(),
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
    /// Appends the prompt as a user message, then drives the loop to completion
    /// with a fresh [`InvocationState`].
    pub async fn invoke(&mut self, prompt: impl Into<String>) -> Result<AgentResult, StrandsError> {
        self.invoke_message(Message::user(prompt.into())).await
    }

    /// Like [`Agent::invoke`] but threads a caller-supplied [`InvocationState`]
    /// through the invocation. Ports `invoke_async(..., invocation_state=...)`.
    ///
    /// The state is carried on every hook event of the invocation as
    /// `event.invocation_state`, letting callers seed request-scoped context
    /// (e.g. `user_id`, `trace_id`) that hooks and tools can read.
    pub async fn invoke_with_state(
        &mut self,
        prompt: impl Into<String>,
        state: InvocationState,
    ) -> Result<AgentResult, StrandsError> {
        self.invoke_message_with_state(Message::user(prompt.into()), state)
            .await
    }

    /// Runs the agent loop starting from a caller-constructed user message, with
    /// a fresh [`InvocationState`].
    ///
    /// # Errors
    /// Returns an error if the agent is waiting on an interrupt — resume with
    /// [`Agent::resume`] before starting a new turn.
    pub async fn invoke_message(&mut self, message: Message) -> Result<AgentResult, StrandsError> {
        self.invoke_message_with_state(message, InvocationState::new())
            .await
    }

    /// Like [`Agent::invoke_message`] but threads a caller-supplied
    /// [`InvocationState`] through the invocation.
    ///
    /// # Errors
    /// Returns an error if the agent is waiting on an interrupt — resume with
    /// [`Agent::resume`] before starting a new turn.
    pub async fn invoke_message_with_state(
        &mut self,
        message: Message,
        state: InvocationState,
    ) -> Result<AgentResult, StrandsError> {
        if self.interrupt_state.is_activated() {
            return Err(StrandsError::model(
                "Agent is in an interrupted state. Resume with `Agent::resume` before invoking.",
            ));
        }
        self.run(Some(message), state).await
    }

    /// Resumes a turn that halted on an interrupt, supplying the human responses,
    /// with a fresh [`InvocationState`].
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
        self.resume_with_state(responses, InvocationState::new())
            .await
    }

    /// Like [`Agent::resume`] but threads a caller-supplied [`InvocationState`]
    /// through the resumed invocation.
    ///
    /// # Errors
    /// Returns an error if the agent is not in an interrupted state, or if a
    /// response references an unknown interrupt id.
    pub async fn resume_with_state(
        &mut self,
        responses: Vec<InterruptResponse>,
        state: InvocationState,
    ) -> Result<AgentResult, StrandsError> {
        if !self.interrupt_state.is_activated() {
            return Err(StrandsError::model(
                "Agent is not in an interrupted state; call `invoke` to start a new turn.",
            ));
        }
        self.interrupt_state.resume(responses)?;
        self.run(None, state).await
    }

    /// The interrupt state backing this agent, for inspection and serialization.
    pub fn interrupt_state(&self) -> &InterruptState {
        &self.interrupt_state
    }

    /// Loads tools from the configured [`ToolProvider`]s into the registry, once,
    /// at the start of the first invocation. Ports the lazy `load_tools` behavior.
    async fn load_tool_providers(&mut self) -> Result<(), StrandsError> {
        if self.providers_loaded {
            return Ok(());
        }
        let providers = self.tool_providers.clone();
        for provider in &providers {
            for tool in provider.load_tools().await? {
                self.tool_registry.add(tool)?;
            }
        }
        self.providers_loaded = true;
        Ok(())
    }

    /// The resume loop: brackets each pass with [`BeforeInvocationEvent`] /
    /// [`AfterInvocationEvent`] and re-enters when a hook sets `resume`. Ports
    /// the `while (true)` resume loop in `stream()`.
    async fn run(
        &mut self,
        mut new_input: Option<Message>,
        state: InvocationState,
    ) -> Result<AgentResult, StrandsError> {
        self.load_tool_providers().await?;

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
            &self.trace_attributes,
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
        // Once the model returns plain text while a structured-output schema is
        // set, the next cycle forces the structured-output tool. Ports the
        // TypeScript `structuredOutputChoice` state.
        let mut force_structured = false;

        loop {
            if iterations >= MAX_LOOP_ITERATIONS {
                return Ok(AgentResult::new(StopReason::EndTurn, self.last_message()));
            }
            iterations += 1;

            let cycle_span = self.tracer.start_cycle_span(&format!("cycle-{iterations}"));
            match self
                .run_one_cycle(&mut new_input, state, &mut force_structured)
                .await
            {
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
        force_structured: &mut bool,
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

                let model_result = self.invoke_model(state, *force_structured).await?;
                let structured = self.structured_output_schema.is_some();

                if model_result.stop_reason != StopReason::ToolUse {
                    // With a schema set, plain text means the model ignored the
                    // structured-output tool: drop the turn and force it next
                    // cycle, or error if it already refused when forced. Ports
                    // the `structuredOutputChoice` fallback.
                    if structured {
                        if *force_structured {
                            return Err(StrandsError::StructuredOutput(
                                "The model failed to invoke the structured output tool even after it was forced."
                                    .to_string(),
                            ));
                        }
                        *force_structured = true;
                        return Ok(CycleOutcome::Continue);
                    }
                    self.append_message(model_result.message.clone(), state)
                        .await?;
                    return Ok(CycleOutcome::Done(AgentResult::new(
                        model_result.stop_reason,
                        model_result.message,
                    )));
                }

                // The model called the structured-output tool: capture its input
                // as the structured result, record the tool-use and a success
                // tool-result in history, and finish.
                if structured {
                    if let Some(output) = extract_structured_output(&model_result.message) {
                        let tool_use_id = structured_tool_use_id(&model_result.message);
                        self.append_message(model_result.message.clone(), state)
                            .await?;
                        if let Some(tool_use_id) = tool_use_id {
                            let result_message = Message::new(
                                Role::User,
                                vec![ContentBlock::ToolResult(ToolResultBlock {
                                    tool_use_id,
                                    status: ToolResultStatus::Success,
                                    content: vec![ToolResultContent::Json(output.clone())],
                                })],
                            );
                            self.append_message(result_message, state).await?;
                        }
                        return Ok(CycleOutcome::Done(AgentResult::with_structured_output(
                            StopReason::ToolUse,
                            model_result.message,
                            output,
                        )));
                    }
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
    ///
    /// When a structured-output schema is configured, a synthetic
    /// `strands_structured_output` tool carrying that schema is offered to the
    /// model; `force_structured` additionally forces the model to call it.
    async fn invoke_model(
        &self,
        state: &InvocationState,
        force_structured: bool,
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
            let mut tool_specs = self.tool_registry.tool_specs();
            let mut tool_choice = None;
            if let Some(schema) = &self.structured_output_schema {
                tool_specs.push(structured_output_tool_spec(schema));
                if force_structured {
                    tool_choice = Some(ToolChoice::Tool {
                        name: STRUCTURED_OUTPUT_TOOL_NAME.to_string(),
                    });
                }
            }
            let context = InvokeModelContext {
                messages: self.messages.snapshot(),
                system_prompt: self.system_prompt.clone(),
                tool_specs,
                tool_choice,
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

    /// Runs the tools for one turn via the configured [`ToolExecutor`], building
    /// the execution context from the agent's shared handles.
    async fn execute_tools(
        &self,
        assistant_message: &Message,
        state: &InvocationState,
        completed: Option<HashMap<String, ToolResultBlock>>,
    ) -> Result<ToolsExecutionResult, StrandsError> {
        let ctx = ToolExecutionContext {
            hooks: self.hooks.clone(),
            tool_registry: self.tool_registry.clone(),
            interrupt_state: self.interrupt_state.clone(),
            tool_middleware: self.execute_tool_mw.clone(),
            tracer: self.tracer.clone(),
            agent: self.agent_handle(),
        };
        self.tool_executor
            .execute_tools(&ctx, assistant_message, state, completed)
            .await
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

/// The synthetic tool spec offered to capture structured output: its input
/// schema is the caller's desired output schema. Ports `StructuredOutputTool`.
fn structured_output_tool_spec(schema: &serde_json::Value) -> ToolSpec {
    ToolSpec {
        name: STRUCTURED_OUTPUT_TOOL_NAME.to_string(),
        description:
            "This tool MUST only be invoked as the last and final tool before returning the completed result to the caller."
                .to_string(),
        input_schema: Some(schema.clone()),
        output_schema: None,
    }
}

/// Returns the input of a `strands_structured_output` tool-use in `message`, if
/// present — the captured structured output. Ports `_extractStructuredOutput`.
fn extract_structured_output(message: &Message) -> Option<serde_json::Value> {
    message.content.iter().find_map(|block| match block {
        ContentBlock::ToolUse(tool_use) if tool_use.name == STRUCTURED_OUTPUT_TOOL_NAME => {
            Some(tool_use.input.clone())
        }
        _ => None,
    })
}

/// The tool-use id of the `strands_structured_output` call in `message`, if any.
fn structured_tool_use_id(message: &Message) -> Option<String> {
    message.content.iter().find_map(|block| match block {
        ContentBlock::ToolUse(tool_use) if tool_use.name == STRUCTURED_OUTPUT_TOOL_NAME => {
            Some(tool_use.tool_use_id.clone())
        }
        _ => None,
    })
}
