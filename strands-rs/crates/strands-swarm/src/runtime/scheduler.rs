//! Scheduled messages: creating, listing and cancelling schedules, and the loop that fires them.

use super::*;

impl Swarm {
    /// The swarm's clock: wall time, but advancing with tokio's clock (paused in tests).
    pub fn now(&self) -> DateTime<Utc> {
        let (wall, instant) = self.inner.epoch;
        let elapsed = chrono::Duration::from_std(instant.elapsed()).unwrap_or_default();
        Utc::now().max(wall + elapsed)
    }

    /// Create (or, with a taken id, replace) a schedule from `owner` (an agent id or [`USER`]):
    /// `message` is posted to agent `to` at `at` (default: one `every` from now), then every
    /// `every` if set. Persisted with the run; fires missed while it was down fire once on resume.
    pub fn schedule(&self, owner: &str, request: ScheduleRequest) -> Result<Schedule> {
        let ScheduleRequest {
            id,
            to,
            message,
            every,
            at,
        } = request;
        if owner != USER {
            self.known(owner)?;
        }
        let live = self.node(&to).is_some_and(|node| node.status.lock().is_live());
        if !live {
            return Err(Error::other(format!("{to} is not a live agent.")));
        }
        if every.is_some_and(|every| every < schedules::MIN_EVERY) {
            return Err(Error::other(
                "Schedules repeat at most once a minute (every >= 1m).",
            ));
        }
        let now = self.now();
        let next = match (at, every) {
            (Some(at), _) => at,
            (None, Some(every)) => now + chrono::Duration::from_std(every).map_err(Error::other)?,
            (None, None) => return Err(Error::other("Give `at`, `every`, or both.")),
        };
        let id = id.map(|id| id.trim().to_owned()).filter(|id| !id.is_empty());
        if id.as_ref().is_some_and(|id| id.len() > 40) {
            return Err(Error::other("Schedule ids are at most 40 characters."));
        }
        let schedule = {
            let mut list = self.inner.schedules.lock();
            let id = id.unwrap_or_else(|| {
                let taken = list
                    .iter()
                    .filter_map(|s| s.id.strip_prefix('s')?.parse::<u64>().ok());
                format!("s{}", taken.max().unwrap_or(0) + 1)
            });
            let schedule = Schedule {
                id,
                owner: owner.to_owned(),
                to,
                message,
                next,
                every_secs: every.map(|every| every.as_secs()),
                fired: None,
            };
            match list.iter_mut().find(|s| s.id == schedule.id) {
                Some(old) => *old = schedule.clone(),
                None => list.push(schedule.clone()),
            }
            schedule
        };
        self.journal_event("schedule", serde_json::to_value(&schedule).unwrap_or_default());
        self.clear_dormancy(&schedule.to);
        self.persist();
        self.start_scheduler();
        // An idle receiver now sleeps on its schedule, which frees its slot.
        self.start_queued();
        Ok(schedule)
    }

    /// Receivers of pending schedules.
    pub(super) fn scheduled_targets(&self) -> Vec<String> {
        self.inner.schedules.lock().iter().map(|s| s.to.clone()).collect()
    }

    /// Remove schedule `id`.
    pub fn unschedule(&self, id: &str) -> Result<Schedule> {
        let removed = {
            let mut list = self.inner.schedules.lock();
            let index = list.iter().position(|s| s.id == id);
            index.map(|index| list.remove(index))
        };
        let removed = removed.ok_or_else(|| Error::other(format!("No schedule {id}.")))?;
        self.journal_event("unschedule", serde_json::json!({ "id": id }));
        self.persist();
        self.inner.schedules_changed.notify_one();
        Ok(removed)
    }

    /// Pending schedules, soonest first; those of cancelled agents are dropped.
    pub fn schedules(&self) -> Vec<Schedule> {
        self.prune_schedules();
        let mut list = self.inner.schedules.lock().clone();
        list.sort_by_key(|s| s.next);
        list
    }

    /// Drop schedules whose receiver or (agent) owner is gone or cancelled.
    pub(super) fn prune_schedules(&self) {
        let alive = |id: &str| self.node(id).is_some_and(|node| node.status.lock().is_live());
        let list = self.inner.schedules.lock().clone();
        let dead: Vec<String> = list
            .iter()
            .filter(|s| !alive(&s.to) || (s.owner != USER && !alive(&s.owner)))
            .map(|s| s.id.clone())
            .collect();
        if !dead.is_empty() {
            self.inner.schedules.lock().retain(|s| !dead.contains(&s.id));
            self.persist();
        }
    }

    pub(super) fn start_scheduler(&self) {
        if self.inner.scheduler.swap(true, Ordering::SeqCst) {
            self.inner.schedules_changed.notify_one();
            return;
        }
        let swarm = self.clone();
        tokio::spawn(async move { swarm.run_scheduler().await });
    }

    /// Post due schedules, then sleep until the next one (re-checking at least every minute, so
    /// a suspended machine catches up), a change, or shutdown.
    pub(super) async fn run_scheduler(self) {
        loop {
            if self.inner.shutting_down.load(Ordering::SeqCst) {
                return;
            }
            self.prune_schedules();
            let now = self.now();
            let due: Vec<Schedule> = {
                let list = self.inner.schedules.lock();
                list.iter().filter(|s| s.next <= now).cloned().collect()
            };
            let cost = (!due.is_empty()).then(|| self.cost());
            let fired: std::collections::HashMap<String, schedules::Fired> = due
                .iter()
                .map(|s| {
                    let spend = cost.as_ref().and_then(|cost| cost.spend(&s.to));
                    let fired = schedules::Fired {
                        at: now,
                        tokens: spend.map_or(0, |spend| spend.own_tokens),
                        usd: spend.filter(|s| !s.own.unknown).map(|s| s.own.usd),
                    };
                    (s.id.clone(), fired)
                })
                .collect();
            self.inner.schedules.lock().retain_mut(|s| {
                if s.next > now {
                    return true;
                }
                s.fired = fired.get(&s.id).copied();
                s.advance(now)
            });
            for schedule in &due {
                let text = schedule.text(fired[&schedule.id]);
                if let Err(error) = self.post(&schedule.owner, &schedule.to, "schedule", text) {
                    tracing::warn!(%error, id = %schedule.id, "scheduled message not delivered");
                }
            }
            if !due.is_empty() {
                self.persist();
            }
            let next = self.inner.schedules.lock().iter().map(|s| s.next).min();
            let wait = next
                .map(|next| (next - self.now()).to_std().unwrap_or_default())
                .unwrap_or(Duration::MAX)
                .min(Duration::from_secs(60));
            tokio::select! {
                _ = tokio::time::sleep(wait) => {}
                _ = self.inner.schedules_changed.notified() => {}
                _ = self.inner.root_cancel.cancelled() => return,
            }
        }
    }
}
