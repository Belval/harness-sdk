//! When to summarize an agent's older turns: at a fixed token count (`--compact-at`), not only
//! near the end of the model's window. A 1M-token window would otherwise let every call carry
//! up to ~850k tokens before anything is compacted.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::dashboard::ContextControl;
use async_trait::async_trait;
use strands_agents::context_manager::stash::Stash;
use strands_agents::context_manager::{ContextState, ContextStrategy, AUTO_SUMMARIZE_UTILIZATION};
use strands_agents::models::DEFAULT_CONTEXT_WINDOW_LIMIT;
use strands_agents::{Agent, Model, Result};

/// Default `--compact-at`: summarize once a call's input reaches 200k tokens.
pub const DEFAULT_COMPACT_AT: u64 = 200_000;

/// The input size (tokens) at which older turns are summarized: `compact_at`, or the context
/// manager's own 85% mark when the window is smaller.
pub fn compaction_point(window: u64, compact_at: u64) -> u64 {
    let auto = (window as f64 * AUTO_SUMMARIZE_UTILIZATION) as u64;
    auto.min(compact_at).max(1)
}

/// The utilization (share of `window`) at which to summarize:
/// `min(AUTO_SUMMARIZE_UTILIZATION, compact_at / window)`.
pub fn compaction_utilization(window: u64, compact_at: u64) -> f64 {
    compaction_point(window, compact_at) as f64 / window.max(1) as f64
}

/// The window the context manager assumes for `model`: its own, else the same default.
pub fn window_of(model: &dyn Model) -> u64 {
    model
        .context_window_limit()
        .filter(|window| *window > 0)
        .unwrap_or(DEFAULT_CONTEXT_WINDOW_LIMIT)
}

/// Runs `inner` (an ungated summarize strategy) once utilization reaches the compaction point of
/// the agent's current model, so a `/model` switch moves the point with the window, or when the
/// agent asked for it through its `context` tool.
pub struct CompactAt {
    inner: Arc<dyn ContextStrategy>,
    compact_at: u64,
    control: Option<Arc<ContextControl>>,
    /// Set by [`CompactAt::arm`]: run on this pipeline pass whatever utilization says by then.
    armed: AtomicBool,
}

impl CompactAt {
    /// Gate `inner` at `compact_at` tokens.
    pub fn new(inner: Arc<dyn ContextStrategy>, compact_at: u64) -> Self {
        Self {
            inner,
            compact_at,
            control: None,
            armed: AtomicBool::new(false),
        }
    }

    /// Also run `inner` once after each summarize request made through `control`.
    pub fn on_request(mut self, control: Arc<ContextControl>) -> Self {
        self.control = Some(control);
        self
    }

    /// Whether this pipeline pass will compact; if so, it will even if an earlier strategy
    /// brings utilization back under the point. Takes a pending summarize request.
    pub fn arm(&self, ctx: &ContextState<'_>) -> bool {
        let due = self.due(ctx);
        if due {
            self.armed.store(true, Ordering::SeqCst);
        }
        due
    }

    fn due(&self, ctx: &ContextState<'_>) -> bool {
        let requested = self.control.as_ref().is_some_and(|c| c.take_summarize_request());
        let window = window_of(ctx.agent.model().as_ref());
        requested || ctx.overflow || ctx.utilization >= compaction_utilization(window, self.compact_at)
    }
}

#[async_trait]
impl ContextStrategy for CompactAt {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn init(&self, agent: &Agent, stash: Option<Arc<Stash>>) {
        self.inner.init(agent, stash);
    }

    async fn apply(&self, ctx: &mut ContextState<'_>) -> Result<bool> {
        if !self.armed.swap(false, Ordering::SeqCst) && !self.due(ctx) {
            return Ok(false);
        }
        self.inner.apply(ctx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compacts_at_the_smaller_of_compact_at_and_85_percent() {
        assert_eq!(compaction_point(1_000_000, 200_000), 200_000);
        assert!((compaction_utilization(1_000_000, 200_000) - 0.2).abs() < 1e-9);
        assert_eq!(compaction_point(200_000, 200_000), 170_000);
        assert!((compaction_utilization(200_000, 200_000) - AUTO_SUMMARIZE_UTILIZATION).abs() < 1e-9);
        assert_eq!(compaction_point(200_000, 1_500_000), 170_000);
        assert_eq!(compaction_point(1_000_000, 50_000), 50_000);
    }
}
