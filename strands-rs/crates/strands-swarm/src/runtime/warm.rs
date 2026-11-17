//! Keep-warm: while a turn waits on a tool call (a `wait`, a slow build), re-send the last model
//! request every [`SwarmConfig::keep_warm`] and drop it at the first content, so the provider's
//! 5-minute prompt cache is read (0.1x input) instead of expiring and being rewritten (1.25x).

use strands_agents::models::StreamAggregatedResult;
use strands_agents::types::StopReason;

use super::*;

/// Refresh interval for the 5-minute cache, with margin for the call's own latency.
pub const KEEP_WARM_EVERY: Duration = Duration::from_secs(270);
/// Refreshes per pause, then the cache is left to expire: 12 × 4.5 min is under an hour.
pub const KEEP_WARM_MAX: usize = 12;

/// A model request as sent, to send again.
struct Request {
    model: Arc<dyn Model>,
    messages: Vec<Message>,
    options: StreamOptions,
}

impl Swarm {
    /// After a call that stopped for tool use and hit the prompt cache, refresh that cache until
    /// the agent's next call, a cancel or interrupt, or [`KEEP_WARM_MAX`] refreshes. Refreshes go
    /// through the model gate and count toward the agent's tokens (estimated as a full cache
    /// read, since a dropped stream reports no usage).
    pub(super) fn install_keep_warm(&self, agent: &Agent, node: &Arc<Node>) {
        let Some(every) = self.inner.config.keep_warm else {
            return;
        };
        let swarm = self.clone();
        let node = Arc::downgrade(node);
        let refresher: Arc<Mutex<Option<CancellationToken>>> = Arc::default();
        agent.add_middleware::<InvokeModelStage>(wrap_fn(
            move |context: InvokeModelContext, events, next| {
                let (swarm, node, refresher) = (swarm.clone(), node.clone(), refresher.clone());
                async move {
                    if let Some(previous) = refresher.lock().take() {
                        previous.cancel();
                    }
                    let request = Request {
                        model: context.model.clone(),
                        messages: context.messages.clone(),
                        options: StreamOptions {
                            system_prompt: context.system_prompt.clone(),
                            tool_specs: context.tool_specs.clone(),
                            tool_choice: context.tool_choice.clone(),
                            dynamic_trailing_blocks: Some(context.dynamic_trailing_blocks),
                            ..Default::default()
                        },
                    };
                    let stop = context.agent.cancel_token().child_token();
                    let result: StreamAggregatedResult = next.run(context, &events).await?;
                    let cached = result.usage.as_ref().map_or(0, |usage| {
                        usage.cache_read_input_tokens.unwrap_or(0)
                            + usage.cache_write_input_tokens.unwrap_or(0)
                    });
                    if let (StopReason::ToolUse, true, Some(node)) =
                        (&result.stop_reason, cached > 0, node.upgrade())
                    {
                        let usage = result.usage.clone().unwrap_or_default();
                        *refresher.lock() = Some(stop.clone());
                        tokio::spawn(swarm.keep_warm(node, request, usage, every, stop));
                    }
                    Ok(result)
                }
            },
        ));
    }

    async fn keep_warm(
        self,
        node: Arc<Node>,
        request: Request,
        usage: Usage,
        every: Duration,
        stop: CancellationToken,
    ) {
        // The whole request was cached by the call it repeats: a refresh reads all of it.
        let prefix = usage.input_tokens
            + usage.cache_read_input_tokens.unwrap_or(0)
            + usage.cache_write_input_tokens.unwrap_or(0);
        let options = StreamOptions {
            cancel: Some(stop.clone()),
            ..request.options
        };
        for _ in 0..KEEP_WARM_MAX {
            tokio::select! {
                _ = tokio::time::sleep(every) => {}
                _ = stop.cancelled() => return,
            }
            let Ok(_permit) = self.admit_model_call(&node.id, &stop).await else {
                return;
            };
            let read = tokio::select! {
                read = reads_prefix(request.model.stream(&request.messages, &options)) => read,
                _ = stop.cancelled() => return,
            };
            if !read {
                return;
            }
            if let Some(agent) = node.agent() {
                crate::pruning::note_model_call(&agent);
            }
            let refresh = Usage {
                input_tokens: 0,
                output_tokens: 1,
                total_tokens: prefix + 1,
                cache_read_input_tokens: Some(prefix),
                cache_write_input_tokens: None,
            };
            self.add_usage(&node, &refresh, request.model.model_id(), false);
            if let Some(journal) = &self.inner.journal {
                journal.event("keep_warm", serde_json::json!({"id": node.id, "tokens": prefix}));
            }
        }
    }
}

/// Read `events` until the first content (the provider has read the prompt, and so refreshed
/// its cache, before it streams any) and drop the rest. `false` when the call failed first.
async fn reads_prefix(mut events: ModelEventStream<'_>) -> bool {
    while let Some(event) = events.next().await {
        match event {
            Ok(ModelStreamEvent::ContentBlockDelta { .. } | ModelStreamEvent::MessageStop { .. }) => {
                return true
            }
            Ok(_) => {}
            Err(_) => return false,
        }
    }
    false
}
