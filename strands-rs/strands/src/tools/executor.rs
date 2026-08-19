//! Pluggable tool execution for one assistant turn. Ports the `ToolExecutor`
//! surface (`AgentKwargs.tool_executor`) and the sequential executor.
//!
//! The [`ToolExecutor`] trait owns the "run the tool-use blocks of one model
//! turn" phase. [`SequentialToolExecutor`] is the default and runs tools one at
//! a time, carrying the full per-tool lifecycle: `BeforeToolsEvent`, per-tool
//! `BeforeToolCallEvent` / `AfterToolCallEvent` (with `cancel`, `selected_tool`,
//! `tool_use` mutation, and `retry`), `ToolResultEvent`, the `ExecuteToolStage`
//! middleware, interrupt propagation with pending-execution storage, telemetry
//! spans, the completed-results skip on resume, and `AfterToolsEvent`.
//!
//! A custom executor receives the same [`ToolExecutionContext`] handles and may
//! change the batch strategy (e.g. concurrency) or short-circuit; the per-tool
//! lifecycle lives here so custom executors that delegate keep parity.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::agent::{AgentHandle, InvocationState};
use crate::errors::StrandsError;
use crate::hooks::{
    AfterToolCallEvent, AfterToolsEvent, BeforeToolCallEvent, BeforeToolsEvent, HookEndTurn,
    HookRegistry, ToolResultEvent, ToolUseData,
};
use crate::interrupt::{InterruptState, PendingToolExecution};
use crate::middleware::{ExecuteToolContext, MiddlewareStack, ToolExecutionResult};
use crate::telemetry::Tracer;
use crate::tools::ToolRegistry;
use crate::types::messages::{
    ContentBlock, Message, Role, ToolResultBlock, ToolResultContent, ToolResultStatus, ToolUseBlock,
};

/// The outcome of running the tools for one model turn.
pub struct ToolsExecutionResult {
    /// The user message carrying every tool result.
    pub message: Message,
    /// Set when a hook requested the turn end early after tools.
    pub end_turn: Option<HookEndTurn>,
}

/// The clonable agent handles a [`ToolExecutor`] needs. Built by the agent per
/// tool phase and passed to the executor, mirroring the TypeScript
/// `ToolExecutorOptions`.
#[derive(Clone)]
pub struct ToolExecutionContext {
    /// The hook registry, for firing the tool lifecycle events.
    pub hooks: HookRegistry,
    /// The tool registry, for resolving tools by name.
    pub tool_registry: ToolRegistry,
    /// The interrupt state, for storing pending execution on an interrupt.
    pub interrupt_state: InterruptState,
    /// The `ExecuteToolStage` middleware wrapping each tool call.
    pub tool_middleware: MiddlewareStack<ExecuteToolContext, ToolExecutionResult>,
    /// The tracer, for per-tool spans.
    pub tracer: Tracer,
    /// The hook-facing agent handle carried on the tool events.
    pub agent: AgentHandle,
}

/// Runs the tool-use blocks of one assistant turn. Ports `ToolExecutor`.
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Executes the tools requested in `assistant_message`, returning the
    /// tool-result message and the `AfterToolsEvent`'s `end_turn`. `completed`
    /// carries results already finished before a resume, which are restored
    /// without replaying their lifecycle events.
    async fn execute_tools(
        &self,
        ctx: &ToolExecutionContext,
        assistant_message: &Message,
        state: &InvocationState,
        completed: Option<HashMap<String, ToolResultBlock>>,
    ) -> Result<ToolsExecutionResult, StrandsError>;
}

/// Executes tools one at a time in source order. The default executor. Ports
/// `SequentialToolExecutor`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SequentialToolExecutor;

#[async_trait]
impl ToolExecutor for SequentialToolExecutor {
    async fn execute_tools(
        &self,
        ctx: &ToolExecutionContext,
        assistant_message: &Message,
        state: &InvocationState,
        completed: Option<HashMap<String, ToolResultBlock>>,
    ) -> Result<ToolsExecutionResult, StrandsError> {
        let hooks = ctx.hooks.clone();
        let completed = completed.unwrap_or_default();

        let mut before = BeforeToolsEvent {
            message: assistant_message.clone(),
            invocation_state: state.clone(),
            agent: ctx.agent.clone(),
            cancel: None,
            interrupt_state: ctx.interrupt_state.clone(),
        };
        // A BeforeTools hook interrupt stores pending state with no completed
        // results before propagating, so the whole batch replays on resume.
        if let Err(error) = hooks.invoke_callbacks(&mut before).await {
            if matches!(error, StrandsError::Interrupt(_)) {
                ctx.interrupt_state
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
                    agent: ctx.agent.clone(),
                };
                hooks.invoke_callbacks(&mut event).await?;
                result_blocks.push(result);
            }
        } else {
            for tool_use in &tool_uses {
                // On resume, a tool that already completed is restored without
                // replaying its lifecycle events. Ports the completed-result skip.
                if let Some(done) = completed.get(&tool_use.tool_use_id) {
                    result_blocks.push(done.clone());
                    continue;
                }

                match self.execute_single_tool(ctx, tool_use, state).await {
                    Ok(result) => {
                        let mut event = ToolResultEvent {
                            result: result.clone(),
                            invocation_state: state.clone(),
                            agent: ctx.agent.clone(),
                        };
                        hooks.invoke_callbacks(&mut event).await?;
                        results_by_id.insert(tool_use.tool_use_id.clone(), result.clone());
                        result_blocks.push(result);
                    }
                    Err(error) if matches!(error, StrandsError::Interrupt(_)) => {
                        ctx.interrupt_state
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
            agent: ctx.agent.clone(),
            end_turn: None,
        };
        hooks.invoke_callbacks(&mut after).await?;

        Ok(ToolsExecutionResult {
            message: tool_result_message,
            end_turn: after.end_turn,
        })
    }
}

impl SequentialToolExecutor {
    /// Runs one tool with its `BeforeToolCallEvent` / `AfterToolCallEvent`
    /// bracket, honoring `cancel`, `selected_tool`, `tool_use` mutation, and
    /// `retry`. Ports the per-tool `executeTool` loop.
    async fn execute_single_tool(
        &self,
        ctx: &ToolExecutionContext,
        tool_use_block: &ToolUseBlock,
        state: &InvocationState,
    ) -> Result<ToolResultBlock, StrandsError> {
        let hooks = ctx.hooks.clone();
        let original_name = tool_use_block.name.clone();
        let registry_tool = ctx.tool_registry.resolve(&original_name).ok().cloned();

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
                agent: ctx.agent.clone(),
                cancel: None,
                selected_tool: None,
                interrupt_state: ctx.interrupt_state.clone(),
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
                    ctx.tool_registry.resolve(&tool_use.name).ok().cloned()
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
                    agent: ctx.agent.clone(),
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
            let tracer = ctx.tracer.clone();
            let interrupt_state = ctx.interrupt_state.clone();
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

            let execution = ctx.tool_middleware.invoke(context, terminal).await?;
            let (result, error) = (execution.result, execution.error);

            let mut after = AfterToolCallEvent {
                tool_use: tool_use.clone(),
                tool: effective_tool.clone(),
                result,
                error,
                invocation_state: state.clone(),
                agent: ctx.agent.clone(),
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
}

/// Builds an error tool-result block carrying `message` as its text.
pub(crate) fn error_result(tool_use_id: &str, message: &str) -> ToolResultBlock {
    ToolResultBlock {
        tool_use_id: tool_use_id.to_string(),
        status: ToolResultStatus::Error,
        content: vec![ToolResultContent::Text(message.to_string())],
    }
}

/// Ensures a tool result carries the model-issued `tool_use_id`, so resume and
/// provider correlation match the assistant's tool-use blocks. Ports
/// `_normalizeToolResultId`.
pub(crate) fn normalize_tool_result_id(
    result: ToolResultBlock,
    tool_use_id: &str,
) -> ToolResultBlock {
    if result.tool_use_id == tool_use_id {
        return result;
    }
    ToolResultBlock {
        tool_use_id: tool_use_id.to_string(),
        status: result.status,
        content: result.content,
    }
}
