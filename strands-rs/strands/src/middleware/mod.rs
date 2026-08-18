//! Agent middleware — an onion of handlers wrapping a stage's execution. Ports
//! the `middleware/` subsystem (`types.ts`, `registry.ts`, and the two stable
//! stages of `stages.ts`).
//!
//! Each stage is a typed [`MiddlewareStack`] of handlers in three phases:
//! `Input` transforms the context before execution, `Output` transforms the
//! result after, and `Wrap` wraps the whole call with a `next` continuation it
//! may short-circuit (skip `next`) or retry (call `next` more than once).
//! Composition orders phases `input → output → wrap` and nests so the first
//! registered handler in a phase is outermost — identical to the TypeScript
//! registry's `compose`.
//!
//! # Deviations from the TypeScript port
//!
//! - **Non-streaming.** The TypeScript handler is an async generator that yields
//!   events while returning a result; the Rust agent loop does not stream events
//!   to consumers, so handlers here are plain async functions returning the
//!   result. The event-yielding form (and the `AgentStreamStage`) defer with the
//!   streaming agent API.
//! - **Typed per-stage stacks, not one token-keyed registry.** Rust keeps each
//!   stage a concrete `MiddlewareStack<Context, Result>` rather than a
//!   `Map<stageToken, …>`; the stage token is the stack itself.
//! - **The `ExecuteToolContext.interrupt` middleware bridge is deferred.**

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use crate::agent::InvocationState;
use crate::errors::StrandsError;
use crate::hooks::ToolUseData;
use crate::tools::Tool;
use crate::types::messages::{Message, SystemPrompt, ToolResultBlock};
use crate::types::tools::{ToolChoice, ToolSpec};

/// A boxed, `Send` future returning a stage result or error.
type BoxFut<T> = Pin<Box<dyn Future<Output = Result<T, StrandsError>> + Send>>;

/// The continuation passed to a wrap handler: runs the rest of the chain
/// (inner handlers and the terminal) for a context. Ports `MiddlewareNext`.
pub type MiddlewareNext<C, R> = Arc<dyn Fn(C) -> BoxFut<R> + Send + Sync>;

/// Removes a previously registered middleware handler. Safe to call repeatedly.
pub type MiddlewareCleanup = Box<dyn Fn() + Send + Sync>;

/// A wrap handler stored internally; input/output handlers are adapted to this
/// shape at registration, matching the TypeScript registry's adapters.
type WrapFn<C, R> = Arc<dyn Fn(C, MiddlewareNext<C, R>) -> BoxFut<R> + Send + Sync>;

/// Composition phase. Ordered `input < output < wrap` so wrap handlers sit
/// closest to the terminal. Ports `PHASE_ORDER`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Input,
    Output,
    Wrap,
}

impl Phase {
    fn order(self) -> u8 {
        match self {
            Phase::Input => 0,
            Phase::Output => 1,
            Phase::Wrap => 2,
        }
    }
}

struct Entry<C, R> {
    id: u64,
    phase: Phase,
    handler: WrapFn<C, R>,
}

struct Inner<C, R> {
    entries: Vec<Entry<C, R>>,
    next_id: u64,
}

impl<C, R> Default for Inner<C, R> {
    fn default() -> Self {
        Inner {
            entries: Vec::new(),
            next_id: 0,
        }
    }
}

/// A typed stack of middleware for one stage. Ports the per-stage slice of
/// `MiddlewareRegistry`.
///
/// Cloning yields another handle to the same handlers, so an agent and its
/// builder share one stack.
pub struct MiddlewareStack<C, R> {
    inner: Arc<Mutex<Inner<C, R>>>,
}

impl<C, R> Clone for MiddlewareStack<C, R> {
    fn clone(&self) -> Self {
        MiddlewareStack {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<C, R> Default for MiddlewareStack<C, R> {
    fn default() -> Self {
        MiddlewareStack {
            inner: Arc::new(Mutex::new(Inner::default())),
        }
    }
}

impl<C: Send + 'static, R: Send + 'static> MiddlewareStack<C, R> {
    /// Creates an empty stack.
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&self, phase: Phase, handler: WrapFn<C, R>) -> MiddlewareCleanup {
        let mut inner = self.inner.lock().expect("middleware stack mutex poisoned");
        let id = inner.next_id;
        inner.next_id += 1;
        inner.entries.push(Entry { id, phase, handler });

        let weak = Arc::downgrade(&self.inner);
        Box::new(move || {
            if let Some(inner) = weak.upgrade() {
                let mut inner = inner.lock().expect("middleware stack mutex poisoned");
                inner.entries.retain(|entry| entry.id != id);
            }
        })
    }

    /// Registers a wrap handler: it receives the context and a `next`
    /// continuation, and returns the result. Ports `MiddlewareRegistry.add`.
    pub fn add<F, Fut>(&self, handler: F) -> MiddlewareCleanup
    where
        F: Fn(C, MiddlewareNext<C, R>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<R, StrandsError>> + Send + 'static,
    {
        let handler: WrapFn<C, R> = Arc::new(move |context, next| Box::pin(handler(context, next)));
        self.push(Phase::Wrap, handler)
    }

    /// Registers an input handler that transforms the context before execution.
    /// Ports `MiddlewareRegistry.addInput`.
    pub fn add_input<F, Fut>(&self, handler: F) -> MiddlewareCleanup
    where
        F: Fn(C) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<C, StrandsError>> + Send + 'static,
    {
        let handler = Arc::new(handler);
        let adapted: WrapFn<C, R> = Arc::new(move |context, next: MiddlewareNext<C, R>| {
            let handler = handler.clone();
            Box::pin(async move {
                let transformed = handler(context).await?;
                next(transformed).await
            })
        });
        self.push(Phase::Input, adapted)
    }

    /// Registers an output handler that transforms the result after execution.
    /// Ports `MiddlewareRegistry.addOutput`.
    pub fn add_output<F, Fut>(&self, handler: F) -> MiddlewareCleanup
    where
        F: Fn(R) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<R, StrandsError>> + Send + 'static,
    {
        let handler = Arc::new(handler);
        let adapted: WrapFn<C, R> = Arc::new(move |context, next: MiddlewareNext<C, R>| {
            let handler = handler.clone();
            Box::pin(async move {
                let result = next(context).await?;
                handler(result).await
            })
        });
        self.push(Phase::Output, adapted)
    }

    /// Composes the registered handlers around `terminal` and runs the chain for
    /// `context`. Ports `MiddlewareRegistry.invoke`.
    pub async fn invoke<F, Fut>(&self, context: C, terminal: F) -> Result<R, StrandsError>
    where
        F: Fn(C) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<R, StrandsError>> + Send + 'static,
    {
        let terminal: MiddlewareNext<C, R> = Arc::new(move |context| Box::pin(terminal(context)));
        let chain = self.compose(terminal);
        chain(context).await
    }

    /// Composes the handlers around `terminal` into a single continuation.
    ///
    /// Handlers are stable-sorted by phase (`input → output → wrap`) then nested
    /// from the end, so the first registered handler within a phase is outermost.
    fn compose(&self, terminal: MiddlewareNext<C, R>) -> MiddlewareNext<C, R> {
        let mut handlers: Vec<WrapFn<C, R>> = {
            let inner = self.inner.lock().expect("middleware stack mutex poisoned");
            let mut entries: Vec<(u8, usize, WrapFn<C, R>)> = inner
                .entries
                .iter()
                .enumerate()
                .map(|(index, entry)| (entry.phase.order(), index, entry.handler.clone()))
                .collect();
            // Stable sort by phase order, ties keeping registration order.
            entries.sort_by_key(|(order, index, _)| (*order, *index));
            entries.into_iter().map(|(_, _, handler)| handler).collect()
        };

        let mut current = terminal;
        while let Some(handler) = handlers.pop() {
            let next = current.clone();
            current = Arc::new(move |context| handler(context, next.clone()));
        }
        current
    }
}

/// Context for the `InvokeModelStage`. Ports `InvokeModelContext` (the `agent`
/// back-reference and `projectedInputTokens` are omitted in the slice).
#[derive(Debug, Clone)]
pub struct InvokeModelContext {
    /// The conversation to send to the model.
    pub messages: Vec<Message>,
    /// The system prompt, if any.
    pub system_prompt: Option<SystemPrompt>,
    /// Tool specifications offered to the model.
    pub tool_specs: Vec<ToolSpec>,
    /// How the model should choose tools.
    pub tool_choice: Option<ToolChoice>,
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
}

/// Context for the `ExecuteToolStage`. Ports `ExecuteToolContext` (the `agent`
/// back-reference and the interrupt bridge are omitted in the slice).
#[derive(Clone)]
pub struct ExecuteToolContext {
    /// The resolved tool, or `None` when the name matched no registered tool.
    pub tool: Option<Arc<dyn Tool>>,
    /// The tool-use request being executed.
    pub tool_use: ToolUseData,
    /// Per-invocation shared state.
    pub invocation_state: InvocationState,
}

/// Result of the `ExecuteToolStage`. Ports `ExecuteToolResult`; the separate
/// `error` message stands in for the TypeScript `ToolResultBlock.error` field,
/// which the Rust `ToolResultBlock` does not carry.
#[derive(Debug, Clone)]
pub struct ToolExecutionResult {
    /// The tool result block.
    pub result: ToolResultBlock,
    /// The error message when the tool threw; `None` otherwise.
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    //! Ports the composition specs from `middleware/__tests__/registry.test.ts`:
    //! phase ordering, onion nesting, short-circuit, retry, and removal.

    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    type Stack = MiddlewareStack<Vec<&'static str>, Vec<&'static str>>;

    async fn terminal(mut log: Vec<&'static str>) -> Result<Vec<&'static str>, StrandsError> {
        log.push("terminal");
        Ok(log)
    }

    // "input runs before wrap; output runs after; wrap is innermost"
    #[tokio::test]
    async fn phase_ordering() {
        let stack = Stack::new();
        stack.add_input(|mut log: Vec<&'static str>| async move {
            log.push("input");
            Ok(log)
        });
        stack.add_output(|mut log: Vec<&'static str>| async move {
            log.push("output");
            Ok(log)
        });
        stack.add(
            |mut log: Vec<&'static str>, next: MiddlewareNext<_, _>| async move {
                log.push("wrap-before");
                let mut log = next(log).await?;
                log.push("wrap-after");
                Ok(log)
            },
        );

        let result = stack.invoke(Vec::new(), terminal).await.unwrap();
        // input transforms context on the way in; wrap brackets the terminal;
        // output transforms the result on the way out.
        assert_eq!(
            result,
            vec!["input", "wrap-before", "terminal", "wrap-after", "output"]
        );
    }

    // "a wrap handler that never calls next short-circuits the terminal"
    #[tokio::test]
    async fn short_circuit_skips_terminal() {
        let stack = Stack::new();
        stack.add(
            |_log: Vec<&'static str>, _next: MiddlewareNext<_, _>| async move {
                Ok(vec!["short-circuited"])
            },
        );
        let result = stack.invoke(Vec::new(), terminal).await.unwrap();
        assert_eq!(result, vec!["short-circuited"]);
    }

    // "a wrap handler may call next more than once (retry)"
    #[tokio::test]
    async fn retry_calls_terminal_twice() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_terminal = calls.clone();
        let stack = Stack::new();
        stack.add(move |log: Vec<&'static str>, next: MiddlewareNext<_, _>| {
            let next = next.clone();
            async move {
                let _first = next(log.clone()).await?;
                next(log).await
            }
        });
        let result = stack
            .invoke(Vec::new(), move |mut log: Vec<&'static str>| {
                let calls = calls_terminal.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    log.push("terminal");
                    Ok(log)
                }
            })
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(result, vec!["terminal"]);
    }

    // "cleanup removes a handler"
    #[tokio::test]
    async fn cleanup_removes_handler() {
        let stack = Stack::new();
        let cleanup = stack.add_output(|mut log: Vec<&'static str>| async move {
            log.push("output");
            Ok(log)
        });
        cleanup();
        let result = stack.invoke(Vec::new(), terminal).await.unwrap();
        assert_eq!(result, vec!["terminal"]);
    }

    // "an empty stack runs the terminal directly"
    #[tokio::test]
    async fn empty_stack_runs_terminal() {
        let stack = Stack::new();
        let result = stack.invoke(Vec::new(), terminal).await.unwrap();
        assert_eq!(result, vec!["terminal"]);
    }
}
