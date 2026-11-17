//! Messaging: posting mail, draining a mailbox, and waiting for mail from given agents.

use super::*;

/// How often a `wait` re-checks whether it can still get mail ([`Swarm::stalled_on_queue`]).
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
        match self.wait_until(id, filter, cancel, |_, _| None).await {
            WaitOutcome::Mail(mail) => Some(mail),
            _ => None,
        }
    }

    /// Like [`Swarm::wait_for_mail`], for the `wait` tool: also ends, with a notice, once no mail
    /// can come any more because every agent in `from` (or, without `from`, every agent `id`
    /// spawned) is cancelled, or none can come until the waiter frees a slot for them.
    pub async fn wait(&self, id: &str, from: &[String], cancel: &CancellationToken) -> WaitOutcome {
        let give_up = |swarm: &Self, node: &Node| {
            swarm
                .unreachable(node, from)
                .or_else(|| swarm.stalled_on_queue(node, from))
        };
        self.wait_until(id, from, cancel, give_up).await
    }

    /// Whether `id` has unread mail from one of `from` (from anyone when empty).
    pub fn has_mail_from(&self, id: &str, from: &[String]) -> bool {
        self.node(id).is_some_and(|node| has_matching_mail(&node, from))
    }

    pub(super) async fn wait_until(
        &self,
        id: &str,
        filter: &[String],
        cancel: &CancellationToken,
        give_up: impl Fn(&Self, &Node) -> Option<String>,
    ) -> WaitOutcome {
        let Some(node) = self.node(id) else {
            return WaitOutcome::Cancelled;
        };
        loop {
            let notified = node.notify.notified();
            if has_matching_mail(&node, filter) {
                return WaitOutcome::Mail(self.drain(id));
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

    /// Why `node` cannot get mail from `from` (without `from`, its children) until it acts: every
    /// one still live is queued at the agent limit, which only a cancel frees, and no other agent
    /// is running (or has mail to run on) that could cancel one.
    pub(super) fn stalled_on_queue(&self, node: &Node, from: &[String]) -> Option<String> {
        let (limit, scheduled) = (self.max_agents(), self.scheduled_targets());
        let agents = self.inner.agents.lock();
        let awaited: Vec<&Arc<Node>> = match from.is_empty() {
            true => agents.values().filter(|n| n.parent == node.id).collect(),
            // The user and job reports can come at any time.
            false => from.iter().map(|id| agents.get(id)).collect::<Option<_>>()?,
        };
        let mut queued = Vec::new();
        for other in awaited {
            match other.status.lock().clone() {
                AgentStatus::Queued => queued.push(format!("{} ({})", other.name, other.id)),
                AgentStatus::Cancelled => {}
                _ => return None,
            }
        }
        let mut holding = Vec::new();
        for other in agents.values().filter(|n| n.id != node.id) {
            let status = other.status.lock().clone();
            match status {
                AgentStatus::Running | AgentStatus::Starting => return None,
                AgentStatus::Idle | AgentStatus::Errored(_) if !other.mailbox.lock().is_empty() => {
                    return None
                }
                AgentStatus::Idle | AgentStatus::Errored(_) if !scheduled.contains(&other.id) => {
                    holding.push(format!("{} ({})", other.name, other.id))
                }
                _ => {}
            }
        }
        if queued.is_empty() {
            return None;
        }
        let holding = match holding.is_empty() {
            true => String::new(),
            false => format!(" Idle agents holding slots: {}.", holding.join(", ")),
        };
        Some(format!(
            "No messages will come yet: {} queued at the agent limit ({limit}). A queued agent \
             starts only once a live one is cancelled, and no other agent is running to do that.\
             {holding} Cancel agents you are done with (or ask the user to raise /max-agents), \
             then wait again.",
            queued.join(", "),
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

pub(super) fn has_matching_mail(node: &Node, filter: &[String]) -> bool {
    let mailbox = node.mailbox.lock();
    match filter.is_empty() {
        true => !mailbox.is_empty(),
        // The user is never filtered out: their message ends any wait.
        false => mailbox
            .iter()
            .any(|mail| mail.from == USER || filter.contains(&mail.from)),
    }
}
