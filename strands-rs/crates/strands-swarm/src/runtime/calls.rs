//! The model-call gate: how many calls may be in flight at once, adjustable while the swarm runs.

use super::*;

/// Calls in flight and callers waiting, in arrival order.
#[derive(Default)]
struct Slots {
    in_flight: usize,
    next_ticket: u64,
    waiting: VecDeque<(u64, String)>,
}

/// The swarm-wide cap on model calls in flight. Waiters are admitted oldest first; every change
/// that may free a slot (a call ending, a raised cap, a waiter leaving) bumps `changed`.
pub(super) struct CallGate {
    slots: Mutex<Slots>,
    changed: tokio::sync::watch::Sender<u64>,
}

impl Default for CallGate {
    fn default() -> Self {
        Self {
            slots: Mutex::default(),
            changed: tokio::sync::watch::channel(0).0,
        }
    }
}

impl CallGate {
    fn wake(&self) {
        self.changed.send_modify(|version| *version += 1);
    }
}

/// A model call's slot, given back when dropped.
pub(super) struct CallPermit(Arc<CallGate>);

impl Drop for CallPermit {
    fn drop(&mut self) {
        self.0.slots.lock().in_flight -= 1;
        self.0.wake();
    }
}

impl Swarm {
    /// The model-call limit in force (`0`: none): the last [`Swarm::set_max_calls`], else the
    /// config's.
    pub fn max_calls(&self) -> usize {
        self.inner
            .max_calls
            .lock()
            .unwrap_or(self.inner.config.max_concurrent_calls)
    }

    /// Change the model-call limit for this run (kept across restarts, ahead of
    /// `--max-concurrent-calls`); `Some(0)` lifts it and `None` returns to the configured one.
    /// Raising it admits waiting calls at once; lowering it leaves calls in flight to finish.
    pub fn set_max_calls(&self, limit: Option<usize>) {
        *self.inner.max_calls.lock() = limit;
        self.persist();
        self.journal_event("max_calls", serde_json::json!({ "limit": self.max_calls() }));
        self.inner.calls.wake();
        // The waiters' text names the limit; say the new one.
        let (calls, waiting) = {
            let slots = self.inner.calls.slots.lock();
            (
                slots.in_flight,
                slots.waiting.iter().map(|(_, id)| id.clone()).collect::<Vec<_>>(),
            )
        };
        let limit = self.max_calls();
        for id in waiting {
            self.emit_queued(&id, true, calls, limit);
        }
    }

    /// Publish that `id` started (`queued`) or stopped waiting for a model-call slot.
    fn emit_queued(&self, id: &str, queued: bool, calls: usize, limit: usize) {
        let id = id.to_owned();
        self.emit(SwarmEvent::ModelQueued {
            id,
            queued,
            calls,
            limit,
        });
    }

    /// Model calls in flight now.
    pub fn calls_in_flight(&self) -> usize {
        self.inner.calls.slots.lock().in_flight
    }

    /// What agent `id` is waiting for, if the host holds it (`waiting: host load 180/128 …`) or it
    /// waits for a model-call slot.
    pub fn waiting_for_call(&self, id: &str) -> Option<String> {
        if let Some(why) = self.inner.host.held.lock().get(id) {
            return Some(format!("waiting: {why}"));
        }
        let calls = self.inner.calls.slots.lock();
        calls
            .waiting
            .iter()
            .any(|(_, waiting)| waiting == id)
            .then(|| crate::text::queued_for_call(calls.in_flight, self.max_calls()))
    }

    /// Refuse a model call of agent `id` once the budget is spent, else wait for a slot under the
    /// call limit, to be held for the call's duration. Waiting is published as
    /// [`SwarmEvent::ModelQueued`].
    pub(super) async fn admit_model_call(&self, id: &str, cancel: &CancellationToken) -> Result<CallPermit> {
        if let Some(message) = self.budget_exhausted(id) {
            let mut stopped = self.inner.over_budget.lock();
            if !stopped.iter().any(|stopped| stopped == id) {
                stopped.push(id.to_owned());
            }
            return Err(Error::other(message));
        }
        let gate = &self.inner.calls;
        let mut changes = gate.changed.subscribe();
        let mut ticket = None;
        // The counts the waiting event last told.
        let mut shown = None;
        loop {
            let (calls, limit) = {
                let mut slots = gate.slots.lock();
                let limit = self.max_calls();
                let free = if limit == 0 {
                    usize::MAX
                } else {
                    limit.saturating_sub(slots.in_flight)
                };
                let ahead = match ticket {
                    Some(ticket) => slots.waiting.iter().position(|(t, _)| *t == ticket),
                    None => Some(slots.waiting.len()),
                };
                if ahead.is_some_and(|ahead| ahead < free) {
                    if let Some(ticket) = ticket {
                        slots.waiting.retain(|(t, _)| *t != ticket);
                    }
                    slots.in_flight += 1;
                    let calls = slots.in_flight;
                    drop(slots);
                    if ticket.is_some() {
                        self.emit_queued(id, false, calls, limit);
                    }
                    return Ok(CallPermit(self.inner.calls.clone()));
                }
                if ticket.is_none() {
                    slots.next_ticket += 1;
                    let next = slots.next_ticket;
                    ticket = Some(next);
                    slots.waiting.push_back((next, id.to_owned()));
                }
                (slots.in_flight, limit)
            };
            if shown != Some((calls, limit)) {
                shown = Some((calls, limit));
                self.emit_queued(id, true, calls, limit);
            }
            let cancelled = tokio::select! {
                _ = changes.changed() => false,
                _ = cancel.cancelled() => true,
            };
            if cancelled {
                gate.slots.lock().waiting.retain(|(t, _)| Some(*t) != ticket);
                gate.wake();
                self.emit_queued(id, false, self.calls_in_flight(), self.max_calls());
                return Err(Error::Cancelled);
            }
        }
    }
}
