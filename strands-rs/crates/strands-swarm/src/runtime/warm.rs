//! Keep-warm: while a turn waits on a tool call (a `wait`, a slow build), re-send the last model
//! request every [`SwarmConfig::keep_warm`] and drop it at the first content, so the provider's
//! 5-minute prompt cache is read (0.1x input) instead of expiring and being rewritten (1.25x).
//! An idle agent whose next schedule fires soon is kept warm the same way until it fires, when
//! that costs less than the rewrite ([`Swarm::refreshes_until_fire`]).

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

/// How a keep-warm loop runs.
struct Plan {
    every: Duration,
    refreshes: usize,
    /// Until a schedule fires: stops if none is left.
    scheduled: bool,
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
                    let node = node.upgrade().filter(|_| cached > 0);
                    let plan = node.and_then(|node| match result.stop_reason {
                        StopReason::ToolUse => Some((node, KEEP_WARM_MAX, false)),
                        StopReason::EndTurn => swarm
                            .refreshes_until_fire(&node, request.model.model_id(), every)
                            .map(|n| (node, n, true)),
                        _ => None,
                    });
                    if let Some((node, refreshes, scheduled)) = plan {
                        // The invocation's token ends with the turn; an idle agent's refreshes
                        // run on until its next call, the swarm's shutdown or its last schedule.
                        let stop = match scheduled {
                            true => swarm.inner.root_cancel.child_token(),
                            false => stop,
                        };
                        let usage = result.usage.clone().unwrap_or_default();
                        *refresher.lock() = Some(stop.clone());
                        let plan = Plan {
                            every,
                            refreshes,
                            scheduled,
                        };
                        tokio::spawn(swarm.keep_warm(node, request, usage, plan, stop));
                    }
                    Ok(result)
                }
            },
        ));
    }

    /// How many refreshes, `every` apart, keep `node`'s cache warm until its next schedule fires;
    /// `None` when none is due, or it is so far off that they cost more than the rewrite.
    ///
    /// With cache reads at 0.1x input and writes at 1.25x, a refresh costs 0.1 of the prefix and
    /// a rewrite instead of a read 1.15, so up to 11 refreshes (49 min at 4.5 min) pay: a 30
    /// minute poller spends 6 x 0.1 = 0.6 against 1.15, ~48% less per fire. Longer gaps are left
    /// to expire; `--cache-ttl 1h` is the answer for those.
    pub(super) fn refreshes_until_fire(
        &self,
        node: &Node,
        model: Option<String>,
        every: Duration,
    ) -> Option<usize> {
        let until = self.next_fire(&node.id)? - self.now();
        let count = (until.to_std().ok()?.as_secs() / every.as_secs().max(1)) as usize;
        let model = model
            .or_else(|| self.inner.config.default_model.clone())
            .unwrap_or_default();
        let price = self
            .inner
            .config
            .prices
            .price(&model)
            .unwrap_or(crate::cost::Price {
                input: 1.0,
                output: 0.0,
                cache_read: None,
                cache_write: Some(crate::cost::CACHE_WRITE_FACTOR),
            });
        (1..=price.warm_refreshes()).contains(&count).then_some(count)
    }

    /// When the soonest schedule posting to agent `id` next fires.
    fn next_fire(&self, id: &str) -> Option<DateTime<Utc>> {
        let list = self.inner.schedules.lock();
        list.iter().filter(|s| s.to == id).map(|s| s.next).min()
    }

    async fn keep_warm(
        self,
        node: Arc<Node>,
        request: Request,
        usage: Usage,
        plan: Plan,
        stop: CancellationToken,
    ) {
        let Plan {
            every,
            refreshes,
            scheduled,
        } = plan;
        // The whole request was cached by the call it repeats: a refresh reads all of it.
        let prefix = super::actor::usage_counts(&usage).input;
        let options = StreamOptions {
            cancel: Some(stop.clone()),
            ..request.options
        };
        for _ in 0..refreshes {
            // Biased: a refresh whose time comes as the agent's next call starts is not sent.
            tokio::select! {
                biased;
                _ = stop.cancelled() => return,
                _ = tokio::time::sleep(every) => {}
            }
            if scheduled && (self.next_fire(&node.id).is_none() || !node.status.lock().is_live()) {
                return;
            }
            let Ok(_permit) = self.admit_model_call(&node.id, &stop).await else {
                return;
            };
            if stop.is_cancelled() {
                return;
            }
            let read = tokio::select! {
                biased;
                _ = stop.cancelled() => return,
                read = reads_prefix(request.model.stream(&request.messages, &options)) => read,
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
            self.journal_event("keep_warm", serde_json::json!({"id": node.id, "tokens": prefix}));
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
