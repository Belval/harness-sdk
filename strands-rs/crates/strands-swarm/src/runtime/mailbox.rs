//! Messaging: posting mail, draining a mailbox, and waiting for mail from given agents.

use super::*;

/// How often a `wait` re-checks whether it can still get mail ([`Swarm::unreachable`]).
const GIVE_UP_RECHECK: Duration = Duration::from_secs(1);

impl Swarm {
    /// Post `text` from `from` to `to`'s mailbox. `to` may be [`USER`].
    pub fn send(&self, from: &str, to: &str, text: impl Into<String>) -> Result<()> {
        if from == to {
            // It would only come back as the agent's own next input, an extra turn for nothing.
            return Err(Error::other(
                "You cannot message yourself; keep notes in your reply or in a file instead.",
            ));
        }
        self.post(from, to, "message", text.into())
    }

    /// [`Swarm::send`] without waking `to` ([`FYI`]): for acknowledgements and progress notes,
    /// which would otherwise cost an idle receiver a whole turn each.
    pub fn send_fyi(&self, from: &str, to: &str, text: impl Into<String>) -> Result<()> {
        if from == to {
            return Err(Error::other(
                "You cannot message yourself; keep notes in your reply or in a file instead.",
            ));
        }
        self.post(from, to, FYI, text.into())
    }

    /// [`Swarm::send`] with images attached. Only their paths are queued (and saved); each must
    /// pass [`images::check`] now, and is read when `to` takes the mail.
    pub fn send_with_images(
        &self,
        from: &str,
        to: &str,
        text: impl Into<String>,
        images: Vec<PathBuf>,
    ) -> Result<()> {
        for path in &images {
            images::check(path).map_err(Error::other)?;
        }
        self.post_mail(from, to, "message", text.into(), images)
    }

    /// Take all mail for `id`.
    pub fn drain(&self, id: &str) -> Vec<Mail> {
        let mail: Vec<Mail> = self
            .node(id)
            .map(|node| node.mailbox.lock().drain(..).collect())
            .unwrap_or_default();
        if !mail.is_empty() {
            if let Some(node) = self.node(id).filter(|_| mail.iter().any(|m| m.from == USER)) {
                node.user_asked.store(true, Ordering::SeqCst);
            }
            self.persist();
            self.emit(SwarmEvent::MailRead { id: id.to_owned() });
        }
        mail
    }

    /// Wait until `id` has mail matching `filter` (or any mail, when `filter` is empty), then
    /// take all of its mail. `None` on cancellation.
    pub async fn wait_for_mail(
        &self,
        id: &str,
        filter: &[String],
        cancel: &CancellationToken,
    ) -> Option<Vec<Mail>> {
        match self.wait_until(id, filter, cancel, true, |_, _| None).await {
            WaitOutcome::Mail(mail) => Some(mail),
            _ => None,
        }
    }

    /// Like [`Swarm::wait_for_mail`], for the `wait` tool: also ends, with a notice, once no mail
    /// can come any more because every agent in `from` (or, without `from`, every agent `id`
    /// spawned) is cancelled, or none can come until the waiter frees a slot for them.
    pub async fn wait(&self, id: &str, from: &[String], cancel: &CancellationToken) -> WaitOutcome {
        self.wait_or_peek(id, from, cancel, true).await
    }

    /// Like [`Swarm::wait`], but the mail stays in the mailbox (and so in `swarm.json`):
    /// `Mail` comes back empty. A `wait` with `all` takes the mail only once it returns, so a
    /// crash while it still waits on others loses none.
    pub async fn wait_for_arrival(
        &self,
        id: &str,
        from: &[String],
        cancel: &CancellationToken,
    ) -> WaitOutcome {
        self.wait_or_peek(id, from, cancel, false).await
    }

    async fn wait_or_peek(
        &self,
        id: &str,
        from: &[String],
        cancel: &CancellationToken,
        take: bool,
    ) -> WaitOutcome {
        let give_up = |swarm: &Self, node: &Node| swarm.unreachable(node, from);
        let _waiting = self.node(id).map(|node| dormant::Waiting::new(&node, from));
        self.wait_until(id, from, cancel, take, give_up).await
    }

    /// The mail settle window in force: the last [`Swarm::set_mail_settle`], else the config's.
    pub fn mail_settle(&self) -> Duration {
        match *self.inner.mail_settle.lock() {
            Some(secs) => Duration::from_secs(secs),
            None => self.inner.config.mail_settle,
        }
    }

    /// Change the mail settle window in seconds (`None`: back to the config's, `Some(0)`: none).
    pub fn set_mail_settle(&self, secs: Option<u64>) {
        *self.inner.mail_settle.lock() = secs;
        self.persist();
        self.journal_event(
            "mail_settle",
            serde_json::json!({ "secs": self.mail_settle().as_secs() }),
        );
    }

    /// Before an idle agent's turn starts on mail: let more agent mail arrive, so that one turn
    /// answers a burst instead of one report each. The window ([`Swarm::mail_settle`]) is fixed
    /// from the first mail and only applies while a child is still working, since otherwise
    /// nothing more is coming. The user's mail ends it (it is never made to wait), as does
    /// cancelling the agent.
    pub(super) async fn settle_mail(&self, node: &Node) {
        let window = self.mail_settle();
        if window.is_zero() || has_user_mail(node) || !self.has_working_child(&node.id) {
            return;
        }
        let deadline = tokio::time::Instant::now() + window;
        *node.settling.lock() = Some(deadline);
        loop {
            let notified = node.notify.notified();
            if has_user_mail(node) {
                break;
            }
            tokio::select! {
                _ = notified => {}
                _ = tokio::time::sleep_until(deadline) => break,
                _ = node.cancel.cancelled() => break,
            }
        }
        node.settling.lock().take();
    }

    /// How long until `id`'s turn starts, while it lets mail settle ([`Swarm::settle_mail`]).
    pub fn waking_in(&self, id: &str) -> Option<Duration> {
        let deadline = (*self.node(id)?.settling.lock())?;
        Some(deadline.saturating_duration_since(tokio::time::Instant::now()))
    }

    /// Whether one of `id`'s children is starting, queued or running: more of its mail may come.
    fn has_working_child(&self, id: &str) -> bool {
        let agents = self.inner.agents.lock();
        agents.values().filter(|node| node.parent == id).any(|node| {
            matches!(
                *node.status.lock(),
                AgentStatus::Starting | AgentStatus::Queued | AgentStatus::Running
            )
        })
    }

    /// Whether `id` has unread mail from one of `from` (from anyone when empty).
    pub fn has_mail_from(&self, id: &str, from: &[String]) -> bool {
        self.node(id).is_some_and(|node| has_matching_mail(&node, from))
    }

    /// The senders of `id`'s unread mail that wakes it.
    pub fn unread_senders(&self, id: &str) -> Vec<String> {
        self.node(id)
            .map(|node| {
                let mailbox = node.mailbox.lock();
                mailbox
                    .iter()
                    .filter(|mail| mail.wakes())
                    .map(|mail| mail.from.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub(super) async fn wait_until(
        &self,
        id: &str,
        filter: &[String],
        cancel: &CancellationToken,
        take: bool,
        give_up: impl Fn(&Self, &Node) -> Option<String>,
    ) -> WaitOutcome {
        let Some(node) = self.node(id) else {
            return WaitOutcome::Cancelled;
        };
        loop {
            let notified = node.notify.notified();
            if has_matching_mail(&node, filter) {
                return WaitOutcome::Mail(if take { self.drain(id) } else { Vec::new() });
            }
            if let Some(notice) = give_up(self, &node) {
                return WaitOutcome::Unreachable(notice);
            }
            tokio::select! {
                _ = notified => {}
                // Other agents' status changes (one going idle) notify no waiter.
                _ = tokio::time::sleep(GIVE_UP_RECHECK) => {}
                _ = cancel.cancelled() => return WaitOutcome::Cancelled,
                _ = node.cancel.cancelled() => return WaitOutcome::Cancelled,
            }
        }
    }

    /// Why `node` cannot get mail from `from` any more, if so.
    pub(super) fn unreachable(&self, node: &Node, from: &[String]) -> Option<String> {
        let agents = self.inner.agents.lock();
        let live = |id: &str| {
            id == USER || id == JOBS_SENDER || agents.get(id).is_some_and(|n| n.status.lock().is_live())
        };
        if from.is_empty() {
            let mut children = agents.values().filter(|n| n.parent == node.id).peekable();
            if children.peek().is_none() || children.any(|n| live(&n.id)) {
                return None;
            }
            return Some(
                "Every agent you spawned is cancelled, so no replies will come. To wait for anyone \
                 else, end your turn: idle agents wake up when a message arrives."
                    .to_owned(),
            );
        }
        if from.iter().any(|id| live(id)) {
            return None;
        }
        Some(format!(
            "No messages will come: {} cancelled.",
            from.iter()
                .map(|id| agents
                    .get(id)
                    .map_or(id.clone(), |n| format!("{} ({id})", n.name)))
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }

    /// Put a message to or from the user on record.
    pub(super) fn journal_mail(&self, mail: &Mail, to: &str) {
        let Some(journal) = &self.inner.journal else {
            return;
        };
        if mail.from == USER || to == USER {
            let to_label = self.label(to);
            let attached: String = mail
                .images
                .iter()
                .map(|path| format!("\n[attached image: {}]", path.display()))
                .collect();
            let text = format!("{}{attached}", mail.text);
            journal.mail(&mail.from, &mail.from_label, to, &to_label, mail.kind, &text);
        }
    }

    pub(super) fn post(&self, from: &str, to: &str, kind: &'static str, text: String) -> Result<()> {
        self.post_mail(from, to, kind, text, Vec::new())
    }

    pub(super) fn post_mail(
        &self,
        from: &str,
        to: &str,
        kind: &'static str,
        text: String,
        images: Vec<PathBuf>,
    ) -> Result<()> {
        let from_label = self.label(from);
        let mail = Mail {
            from: from.to_owned(),
            from_label,
            kind,
            text,
            images,
        };
        let event = SwarmEvent::Mail {
            to: to.to_owned(),
            mail: mail.clone(),
        };
        if to == USER {
            self.journal_mail(&mail, to);
            self.emit(event);
            return Ok(());
        }
        let node = self.known(to)?;
        if !node.status.lock().is_live() {
            return Err(Error::other(format!("Agent {to} is cancelled.")));
        }
        self.journal_mail(&mail, to);
        // Emitted under the mailbox lock, so it always precedes the `MailRead` of its drain.
        let mut mailbox = node.mailbox.lock();
        self.emit(event);
        mailbox.push_back(mail);
        drop(mailbox);
        self.persist();
        node.notify.notify_waiters();
        node.notify.notify_one();
        Ok(())
    }

    /// Deliver mail that arrives mid-turn by appending it to the agent's next tool result.
    pub(super) fn install_mail_hook(&self, agent: &Agent, id: &str) {
        let swarm = self.clone();
        let id = id.to_owned();
        agent
            .hooks()
            .add_sync_callback(move |event: &mut AfterToolCallEvent| {
                let mail = swarm.drain(&id);
                if !mail.is_empty() {
                    let text = format!(
                        "\n\n[New messages arrived while this tool ran]\n{}",
                        render_mail(&mail)
                    );
                    event.result.content.push(ToolResultContent::Text(text));
                    event.result.content.extend(images::result_content(&mail));
                }
                Ok(())
            });
    }
}

fn has_user_mail(node: &Node) -> bool {
    node.mailbox
        .lock()
        .iter()
        .any(|mail| mail.wakes() && mail.from == USER)
}

pub(super) fn has_matching_mail(node: &Node, filter: &[String]) -> bool {
    let mailbox = node.mailbox.lock();
    let mut waking = mailbox.iter().filter(|mail| mail.wakes());
    match filter.is_empty() {
        true => waking.next().is_some(),
        // The user is never filtered out: their message ends any wait.
        false => waking.any(|mail| mail.from == USER || filter.contains(&mail.from)),
    }
}
