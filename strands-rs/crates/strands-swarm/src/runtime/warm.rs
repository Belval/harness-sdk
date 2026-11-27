//! Keep-warm: while a turn waits on a tool call (a `wait`, a slow build), re-send the last model
//! request every [`SwarmConfig::keep_warm`] and drop it at the first content, so the provider's
//! 5-minute prompt cache is read (0.1x input) instead of expiring and being rewritten (1.25x).
//! A model whose cache lives longer ([`cost::cache_lifetime`]: OpenAI's GPT-6 on Bedrock, 30
//! minutes) is refreshed proportionally less often.
//! An idle agent whose next schedule fires soon is kept warm the same way until it fires, when
//! that costs less than the rewrite ([`Swarm::refreshes_until_fire`]); any other idle agent for
//! [`KEEP_WARM_IDLE`] refreshes, in case a message wakes it.

use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

use strands_agents::models::{AgentMetadata, StreamAggregatedResult};
use strands_agents::types::{ContentBlock, Role, StopReason};

use super::*;

/// Refresh interval for the 5-minute cache ([`cost::CACHE_LIFETIME`]), with margin for the call's own latency.
pub const KEEP_WARM_EVERY: Duration = Duration::from_secs(270);
/// Refreshes per pause, then the cache is left to expire. The 13th, at 58.5 minutes, keeps it
/// until 63.5: past a `wait` at its 60-minute maximum, the commonest long pause. With 12 the
/// cache expired at 59 minutes and the call after such a wait rewrote it, 254 times in 19.5
/// hours of one swarm (~$430 of writes, against ~$70 for the extra refreshes).
pub const KEEP_WARM_MAX: usize = 13;
/// Refreshes after a turn ends, when no schedule keeps the agent warm: an idle agent is usually
/// woken by a message (a child's reply) within minutes. Measured over 3,800 idle gaps, of those
/// still idle at the k-th refresh the share woken before the next one was 29, 21, 13 and 11% for
/// k = 1..4 and under 8% after, against the 0.1 / 1.15 ≈ 8.7% at which a refresh pays.
pub const KEEP_WARM_IDLE: usize = 4;
/// [`KEEP_WARM_MAX`] for the one-hour cache (`--cache-ttl 1h`), whose refreshes are 12 times as
/// far apart: two, at 54 and 108 minutes, keep a cache warm until 168, past a `wait` at its
/// 60-minute maximum with the first alone.
pub const KEEP_WARM_MAX_1H: usize = 2;
/// [`KEEP_WARM_IDLE`] for the one-hour cache. Measured on a live run, of the agents still idle
/// at the k-th refresh about half were woken before the next (an hour later), well above the
/// 0.1 / 1.9 ≈ 5% at which a refresh pays against a 2x rewrite.
pub const KEEP_WARM_IDLE_1H: usize = 3;
/// How long the one-hour cache lives without a hit.
pub const CACHE_LIFETIME_1H: Duration = Duration::from_secs(60 * 60);

/// A model request as sent, to send again.
struct Request {
    model: Arc<dyn Model>,
    held: Held,
    options: StreamOptions,
}

/// The messages of a [`Request`]. A conversation reaches megabytes and a waiting agent holds its
/// own, so a copy kept for up to an hour would hold it twice.
enum Held {
    /// The agent's first `len` messages as the call sent them: a digest of what its history held
    /// then, and the per-call blocks that were added after it to the last message. A refresh is
    /// built from the history when it is sent, so only that is sent again that the call cached;
    /// an edited history (compaction, pruning) no longer matches and ends the refreshes.
    History {
        len: usize,
        digest: u64,
        trailing: Vec<ContentBlock>,
    },
    /// A request that history cannot rebuild: a hook edited an earlier message for this call only.
    Copy(Vec<Message>),
}

impl Held {
    /// The messages of a call of `agent` that sent `messages`, of which the last `trailing` blocks
    /// of the last message are per-call.
    fn capture(agent: &Agent, messages: &[Message], trailing: usize) -> Self {
        let Some((last, before)) = messages.split_last() else {
            return Self::Copy(Vec::new());
        };
        let keep = last.content.len().saturating_sub(trailing);
        let same = agent.with_messages(|own| {
            own.len() == messages.len()
                && own[..before.len()] == *before
                && own[before.len()].role == last.role
                && own[before.len()].content == last.content[..keep]
        });
        match same {
            true => Self::History {
                len: messages.len(),
                digest: digest(before, &last.role, &last.content[..keep]),
                trailing: last.content[keep..].to_vec(),
            },
            false => Self::Copy(messages.to_vec()),
        }
    }

    /// The messages to send now, `None` when the agent's history is no longer the one the call
    /// sent: its next call will not read this prefix.
    fn messages(&self, agent: Option<&Agent>) -> Option<Cow<'_, [Message]>> {
        match self {
            Self::Copy(messages) => Some(Cow::Borrowed(messages)),
            Self::History {
                len,
                digest: wanted,
                trailing,
            } => agent?.with_messages(|own| {
                let (last, before) = own.get(..*len)?.split_last()?;
                if digest(before, &last.role, &last.content) != *wanted {
                    return None;
                }
                let mut messages = own[..*len].to_vec();
                messages[*len - 1].content.extend(trailing.iter().cloned());
                Some(Cow::Owned(messages))
            }),
        }
    }
}

/// A hash of the roles and contents of `before` and of a last message with `role` and `content`:
/// what the provider reads of a conversation.
fn digest(before: &[Message], role: &Role, content: &[ContentBlock]) -> u64 {
    struct Hashing(DefaultHasher);
    impl std::io::Write for Hashing {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.write(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut hashing = Hashing(DefaultHasher::new());
    for message in before {
        let _ = serde_json::to_writer(&mut hashing, &(&message.role, &message.content));
    }
    let _ = serde_json::to_writer(&mut hashing, &(role, content));
    hashing.0.finish()
}

/// How a keep-warm loop runs.
struct Plan {
    every: Duration,
    refreshes: usize,
    /// The turn ended: runs on past it, stopping if the agent is cancelled.
    idle: bool,
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
                        held: Held::capture(
                            &context.agent,
                            &context.messages,
                            context.dynamic_trailing_blocks,
                        ),
                        options: StreamOptions {
                            system_prompt: context.system_prompt.clone(),
                            tool_specs: context.tool_specs.clone(),
                            tool_choice: context.tool_choice.clone(),
                            dynamic_trailing_blocks: Some(context.dynamic_trailing_blocks),
                            // The same cache key as the call itself, where the model takes one.
                            agent_metadata: context.agent.session_manager().map(|manager| AgentMetadata {
                                session_id: Some(manager.session_id().to_owned()),
                            }),
                            ..Default::default()
                        },
                    };
                    let stop = context.agent.cancel_token().child_token();
                    let every = swarm.refresh_every(every, request.model.model_id());
                    let (max, idle_max) = swarm.refresh_limits(request.model.model_id());
                    let result: StreamAggregatedResult = next.run(context, &events).await?;
                    let cached = result.usage.as_ref().map_or(0, |usage| {
                        usage.cache_read_input_tokens.unwrap_or(0)
                            + usage.cache_write_input_tokens.unwrap_or(0)
                    });
                    let node = node.upgrade().filter(|_| cached > 0);
                    let plan = node.and_then(|node| match result.stop_reason {
                        StopReason::ToolUse => Some((node, max, false, false)),
                        StopReason::EndTurn => {
                            match swarm.refreshes_until_fire(&node, request.model.model_id(), every) {
                                Some(n) => Some((node, n, true, true)),
                                None => Some((node, idle_max, true, false)),
                            }
                        }
                        _ => None,
                    });
                    if let Some((node, refreshes, idle, scheduled)) = plan {
                        // The invocation's token ends with the turn; an idle agent's refreshes
                        // run on until its next call, the swarm's shutdown or its last schedule.
                        let stop = match idle {
                            true => swarm.inner.root_cancel.child_token(),
                            false => stop,
                        };
                        let usage = result.usage.clone().unwrap_or_default();
                        *refresher.lock() = Some(stop.clone());
                        let plan = Plan {
                            every,
                            refreshes,
                            idle,
                            scheduled,
                        };
                        tokio::spawn(swarm.keep_warm(node, request, usage, plan, stop));
                    }
                    Ok(result)
                }
            },
        ));
    }

    /// The model named `model`, or the swarm's default model.
    fn model_name(&self, model: Option<String>) -> String {
        model
            .or_else(|| self.inner.config.default_model.clone())
            .unwrap_or_default()
    }

    /// The refresh interval for `model`: `every` is for the default 5-minute cache, and a cache
    /// that lives n times as long is refreshed n times as far apart.
    fn refresh_every(&self, every: Duration, model: Option<String>) -> Duration {
        let lifetime = self.cache_lifetime(model);
        every.mul_f64(lifetime.as_secs_f64() / crate::cost::CACHE_LIFETIME.as_secs_f64())
    }

    /// How long `model`'s prompt cache lives: the hour of `--cache-ttl 1h` where the model's own
    /// is the default 5 minutes (OpenAI's longer implicit cache ignores the TTL).
    fn cache_lifetime(&self, model: Option<String>) -> Duration {
        match crate::cost::cache_lifetime(&self.model_name(model)) {
            crate::cost::CACHE_LIFETIME if self.inner.config.cache_1h => CACHE_LIFETIME_1H,
            lifetime => lifetime,
        }
    }

    /// Refreshes of a turn waiting on a tool call, and of an idle agent, for `model`.
    fn refresh_limits(&self, model: Option<String>) -> (usize, usize) {
        match self.cache_lifetime(model) == CACHE_LIFETIME_1H {
            true => (KEEP_WARM_MAX_1H, KEEP_WARM_IDLE_1H),
            false => (KEEP_WARM_MAX, KEEP_WARM_IDLE),
        }
    }

    /// How many refreshes, `every` apart, keep `node`'s cache warm until its next schedule fires;
    /// `None` when none is due, or it is so far off that they cost more than the rewrite.
    ///
    /// With cache reads at 0.1x input and writes at 1.25x, a refresh costs 0.1 of the prefix and
    /// a rewrite instead of a read 1.15, so up to 11 refreshes (49 min at 4.5 min) pay: a 30
    /// minute poller spends 6 x 0.1 = 0.6 against 1.15, ~48% less per fire. Longer gaps are left
    /// to expire; `--cache-ttl 1h` is the answer for those. `every` is the model's own interval
    /// ([`Swarm::refresh_every`]), so a cache that lives longer pays for fewer refreshes per
    /// minute and is kept warm for correspondingly longer gaps.
    pub(super) fn refreshes_until_fire(
        &self,
        node: &Node,
        model: Option<String>,
        every: Duration,
    ) -> Option<usize> {
        let until = self.next_fire(&node.id)? - self.now();
        let count = (until.to_std().ok()?.as_secs() / every.as_secs().max(1)) as usize;
        let model = self.model_name(model);
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
            idle,
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
            if (scheduled && self.next_fire(&node.id).is_none()) || (idle && !node.status.lock().is_live()) {
                return;
            }
            let Ok(_permit) = self.admit_model_call(&node.id, &stop).await else {
                return;
            };
            if stop.is_cancelled() {
                return;
            }
            // Built now, from the history the agent's next call will send.
            let Some(messages) = request.held.messages(node.agent().as_ref()) else {
                return;
            };
            let read = tokio::select! {
                biased;
                _ = stop.cancelled() => return,
                read = reads_prefix(request.model.stream(&messages, &options)) => read,
            };
            drop(messages);
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

#[cfg(test)]
mod tests {
    use strands_agents::testing::MockMessageModel;

    use super::*;

    fn agent(messages: Vec<Message>) -> Agent {
        let agent = Agent::builder()
            .model(Arc::new(MockMessageModel::new(Vec::new())))
            .no_sandbox()
            .build()
            .unwrap();
        agent.set_messages(messages);
        agent
    }

    fn sent(held: &Held, agent: &Agent) -> Option<Vec<Message>> {
        held.messages(Some(agent)).map(Cow::into_owned)
    }

    #[test]
    fn a_request_over_the_agent_s_history_holds_none_of_it() {
        let history = vec![Message::user("a"), Message::assistant("b"), Message::user("c")];
        let agent = agent(history.clone());
        let mut call = history.clone();
        call[2].content.push(ContentBlock::text("per call"));
        let held = Held::capture(&agent, &call, 1);
        let Held::History { len, trailing, .. } = &held else {
            panic!("the history rebuilds this request");
        };
        assert_eq!((*len, trailing.len()), (3, 1));
        assert_eq!(sent(&held, &agent), Some(call));

        // Later messages are not part of it; an edit of the prefix ends it.
        agent.with_messages(|own| own.push(Message::assistant("d")));
        assert_eq!(sent(&held, &agent).map(|m| m.len()), Some(3));
        agent.with_messages(|own| own[0] = Message::user("edited"));
        assert_eq!(sent(&held, &agent), None);
        assert!(held.messages(None).is_none());
    }

    #[test]
    fn an_edit_of_an_earlier_message_for_one_call_is_kept_as_a_copy() {
        let agent = agent(vec![
            Message::user("a"),
            Message::assistant("b"),
            Message::user("c"),
        ]);
        let call = vec![Message::user("a!"), Message::assistant("b"), Message::user("c")];
        let held = Held::capture(&agent, &call, 0);
        assert!(matches!(held, Held::Copy(_)));
        assert_eq!(sent(&held, &agent), Some(call));
    }
}
