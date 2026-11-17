//! UI state: a fold over [`SwarmEvent`]s plus the input line. No terminal I/O here.

use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;

use indexmap::IndexMap;

use strands_agents::types::{ContentBlock, Message, Role, ToolResultStatus};

use crate::runtime::{summarize_input, AgentStatus, SwarmEvent, USER};
use crate::text::compact;

pub use super::commands::{parse_command, Command, SlashCommand, SLASH_COMMANDS};

/// Entries kept per agent log; the oldest are dropped beyond this.
pub const MAX_LOG_ENTRIES: usize = 2000;

/// Context share (of the compaction point) at which an agent's log notes that summarization is
/// near.
const COMPACTION_NOTICE_AT: f64 = 0.9;

/// One entry in an agent's log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogEntry {
    /// The task the agent was spawned with.
    Task(String),
    /// Assistant text, accumulated from deltas.
    Text(String),
    /// A tool call; `ok` is `None` while it runs.
    Tool {
        /// Tool name.
        name: String,
        /// Input summary.
        summary: String,
        /// Outcome.
        ok: Option<bool>,
    },
    /// A message the agent received (or, on the sender, a message it sent the user).
    MailIn {
        /// Sender label.
        from: String,
        /// `message`, `reply` or `error`.
        kind: String,
        /// Body.
        text: String,
        /// Whether the receiving agent has taken it from its mailbox; `None` for mail to the user.
        read: Option<bool>,
    },
    /// A status change worth showing (errors, cancellation).
    Status(String),
    /// User-side text restored from a resumed conversation (the task or received mail).
    Input(String),
    /// A quiet marker, such as where restored history begins.
    Note(String),
    /// An image from a tool result or attached to a message: shown as a thumbnail while
    /// [`App::images`] keeps it decoded, and always as its [`LogEntry::placeholder`] line.
    Image {
        /// Where it came from (a path or tool input).
        source: String,
        /// Pixel size and thumbnail key, once decoded; `None` when it could not be.
        decoded: Option<super::images::Decoded>,
    },
}

impl LogEntry {
    /// `[image: shot.png 640x480]`, the text an image entry stands for (also in plain text and
    /// copies, which never contain the thumbnail).
    pub fn placeholder(source: &str, decoded: Option<&super::images::Decoded>) -> String {
        match decoded {
            Some(d) => format!("[image: {source} {}x{}]", d.width, d.height),
            None => format!("[image: {source}, not decodable]"),
        }
    }
}

/// What the UI knows about one agent.
#[derive(Debug, Clone)]
pub struct AgentView {
    /// Id.
    pub id: String,
    /// Parent id.
    pub parent: String,
    /// Name.
    pub name: String,
    /// One line on what the agent is up to: its latest tool call, else the first line of its task.
    pub doing: String,
    /// Working directory.
    pub cwd: PathBuf,
    /// Own branch.
    pub branch: Option<String>,
    /// Status.
    pub status: AgentStatus,
    /// Cumulative input tokens, cache reads and writes included.
    pub input_tokens: u64,
    /// Cumulative output tokens.
    pub output_tokens: u64,
    /// The part of `input_tokens` read from the prompt cache.
    pub cache_read_tokens: u64,
    /// Model id, once known.
    pub model: Option<String>,
    /// The last model call's input as a share of the compaction point (`0..1+`), once known.
    pub context: Option<f64>,
    /// Log, oldest first, at most [`MAX_LOG_ENTRIES`] entries.
    pub log: VecDeque<LogEntry>,
    /// Log entries the user has not looked at.
    pub unseen: bool,
    /// Waiting for a free model-call slot.
    pub queued: bool,
}

impl AgentView {
    /// `parser (a2)`.
    pub fn label(&self) -> String {
        format!("{} ({})", self.name, self.id)
    }

    fn push(&mut self, entry: LogEntry) {
        self.log.push_back(entry);
        self.trim();
        self.unseen = true;
    }

    fn push_text(&mut self, delta: &str) {
        match self.log.back_mut() {
            Some(LogEntry::Text(text)) => text.push_str(delta),
            _ => self.log.push_back(LogEntry::Text(delta.to_owned())),
        }
        self.trim();
        self.unseen = true;
    }

    fn mark_read(&mut self) {
        for entry in &mut self.log {
            if let LogEntry::MailIn { read: Some(read), .. } = entry {
                *read = true;
            }
        }
    }

    /// What a starting or running agent is doing right now, for the log's live indicator.
    pub fn activity(&self) -> Option<String> {
        match self.status {
            AgentStatus::Queued => return Some("queued until an agent slot frees…".to_owned()),
            AgentStatus::Starting => return Some("starting…".to_owned()),
            AgentStatus::Running if self.queued => return Some("queued for a model slot…".to_owned()),
            AgentStatus::Running => {}
            _ => return None,
        }
        Some(match self.log.back() {
            Some(LogEntry::Tool {
                name,
                summary,
                ok: None,
            }) => format!("running {name}: {summary}…"),
            Some(LogEntry::Text(_)) => "writing…".to_owned(),
            _ => "thinking…".to_owned(),
        })
    }

    /// Insert a resumed agent's restored conversation before everything logged since the restart
    /// (after the task entry, which the conversation repeats as its first input).
    fn backfill(&mut self, mut entries: Vec<LogEntry>, omitted: usize) {
        let keep_task = matches!(self.log.front(), Some(LogEntry::Task(_)));
        let task = self.log.front().cloned().filter(|_| keep_task);
        if omitted == 0 {
            if let (Some(LogEntry::Task(task)), Some(LogEntry::Input(first))) = (&task, entries.first()) {
                if task.trim() == first.trim() {
                    entries.remove(0);
                }
            }
        }
        let note = match omitted {
            0 => "restored conversation".to_owned(),
            n => format!("restored conversation ({n} earlier messages not shown)"),
        };
        let since = self.log.drain(usize::from(keep_task)..).collect::<Vec<_>>();
        self.log.push_back(LogEntry::Note(note));
        self.log.extend(entries);
        self.log.push_back(LogEntry::Note("restarted".to_owned()));
        self.log.extend(since);
        self.trim();
        self.unseen = true;
    }

    /// Drop the oldest entries beyond [`MAX_LOG_ENTRIES`].
    fn trim(&mut self) {
        let excess = self.log.len().saturating_sub(MAX_LOG_ENTRIES);
        self.log.drain(..excess);
    }

    fn finish_tool(&mut self, name: &str, ok: bool) {
        let running = self.log.iter_mut().rev().find_map(|entry| match entry {
            LogEntry::Tool {
                name: n,
                ok: outcome @ None,
                ..
            } if n == name => Some(outcome),
            _ => None,
        });
        if let Some(outcome) = running {
            *outcome = Some(ok);
        }
    }
}

/// Everything the view renders.
#[derive(Debug, Default)]
pub struct App {
    /// Agents in spawn order.
    pub agents: IndexMap<String, AgentView>,
    /// Index into [`App::tree`] of the selected agent.
    pub selected: usize,
    /// The input box (may hold several lines).
    pub input: String,
    /// Cursor in [`App::input`] as a byte offset; `None` is the end (see [`App::cursor`]).
    pub cursor: Option<usize>,
    /// Lines sent from the input, oldest first, for ↑/↓.
    pub history: Vec<String>,
    /// Index into [`App::history`] being shown while browsing it.
    pub history_at: Option<usize>,
    /// What was typed before history browsing started.
    pub draft: String,
    /// Lines scrolled up from the bottom of the log; 0 follows new output.
    pub scroll: u16,
    /// Last error or notice, shown in the footer.
    pub notice: Option<String>,
    /// Set when the user asked to quit.
    pub quit: bool,
    /// Set when the user asked to read the selected log as plain text in the terminal's own
    /// scrollback (for native scrolling, search and copy-paste).
    pub show_plain: bool,
    /// Columns added to (or taken from) the agent tree's fitted width; the tree hides when
    /// shrunk below a few columns.
    pub tree_adjust: i16,
    /// Show cancelled agents in the tree (hidden by default).
    pub show_cancelled: bool,
    /// Highlighted entry of the slash-command popup.
    pub completion: usize,
    /// Swarm-wide token budget, if any.
    pub budget: Option<u64>,
    /// Tool calls an installed reload waits for ([`SwarmEvent::ReloadPending`]).
    pub reload_pending: Option<usize>,
    /// Decoded image thumbnails for [`LogEntry::Image`].
    pub images: super::images::Images,
    /// Set while the F1 key-binding popup is open.
    pub show_keys: bool,
    /// The `/status` popup's report while it is open, refreshed by the event loop.
    pub status: Option<crate::status::StatusReport>,
    /// Lines the `/status` or `/cost` popup is scrolled down.
    pub status_scroll: u16,
    /// The `/cost` popup's report while it is open, refreshed by the event loop.
    pub cost: Option<crate::cost::CostReport>,
    /// The swarm's estimated spend for the header, refreshed when token totals change.
    pub spend: Option<crate::cost::Estimate>,
    /// Ring the terminal bell when an agent messages the user (`--notify`).
    pub notify: bool,
    /// A bell is due; the event loop rings it.
    pub bell: bool,
    /// The mouse selection being made or last made, highlighted until the next click.
    pub selection: Option<super::select::Selection>,
    /// When Esc was last pressed on an empty input; a second press soon after stops the turn.
    pub last_esc: Option<std::time::Instant>,
}

impl App {
    /// Fold one swarm event into the state.
    pub fn apply(&mut self, event: SwarmEvent) {
        let selected = self.selected_id();
        self.apply_event(event);
        self.reselect(selected);
    }

    fn apply_event(&mut self, event: SwarmEvent) {
        match event {
            SwarmEvent::Spawned {
                id,
                parent,
                name,
                task,
                cwd,
                branch,
            } => {
                let doing = task
                    .lines()
                    .find(|line| !line.trim().is_empty())
                    .unwrap_or_default();
                let view = AgentView {
                    id: id.clone(),
                    parent,
                    name,
                    doing: doing.trim().to_owned(),
                    cwd,
                    branch,
                    status: AgentStatus::Starting,
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_tokens: 0,
                    model: None,
                    context: None,
                    log: VecDeque::from([LogEntry::Task(task)]),
                    unseen: true,
                    queued: false,
                };
                self.agents.insert(id, view);
            }
            SwarmEvent::Status { id, status } => self.apply_status(&id, status),
            SwarmEvent::Text { id, delta } => self.with(&id, |view| view.push_text(&delta)),
            SwarmEvent::ToolStarted { id, name, summary } => self.with(&id, |view| {
                view.doing = format!("{name} {summary}");
                view.push(LogEntry::Tool {
                    name,
                    summary,
                    ok: None,
                })
            }),
            SwarmEvent::ToolFinished { id, name, ok } => self.with(&id, |view| view.finish_tool(&name, ok)),
            SwarmEvent::Mail { to, mail } => self.apply_mail(&to, mail),
            SwarmEvent::Model { id, model } => self.with(&id, |view| view.model = Some(model)),
            SwarmEvent::MailRead { id } => self.with(&id, AgentView::mark_read),
            SwarmEvent::Image { id, source, bytes } => {
                let entry = self.image_entry(source, &bytes);
                self.with(&id, |view| view.push(entry));
            }
            SwarmEvent::ModelQueued { id, queued } => self.with(&id, |view| view.queued = queued),
            SwarmEvent::Budget { tokens } => self.budget = tokens,
            SwarmEvent::ReloadPending { tool_calls } => self.reload_pending = Some(tool_calls),
            SwarmEvent::Reloaded { from, to, changes } => {
                self.notice = Some(crate::version::notice(&from, &to, &changes));
            }
            SwarmEvent::Reclaimed { collected, requested } => {
                if requested || collected.iter().any(|agent| !agent.is_empty()) {
                    self.notice = Some(crate::gc::notice(&collected));
                }
            }
            SwarmEvent::History {
                id,
                messages,
                omitted,
            } => self.with(&id, |view| view.backfill(history_entries(&messages), omitted)),
            SwarmEvent::Usage {
                id,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                context,
                ..
            } => self.with(&id, |view| {
                view.input_tokens = input_tokens;
                view.output_tokens = output_tokens;
                view.cache_read_tokens = cache_read_tokens;
                // Note the crossing once, from a known lower value, so a restart does not repeat it.
                let crossed = |c: Option<f64>| c.is_some_and(|c| c >= COMPACTION_NOTICE_AT);
                if crossed(context) && view.context.is_some() && !crossed(view.context) {
                    let percent = (context.unwrap_or_default() * 100.0).round();
                    view.push(LogEntry::Status(format!(
                        "context {percent}% of the compaction point — older turns will be summarized soon"
                    )));
                }
                view.context = context.or(view.context);
            }),
            SwarmEvent::Compacted { id, before, after } => self.with(&id, |view| {
                view.push(LogEntry::Status(format!(
                    "context compacted: older turns summarized, {} → {} tokens",
                    compact(before),
                    compact(after)
                )))
            }),
        }
    }

    fn apply_status(&mut self, id: &str, status: AgentStatus) {
        let note = match &status {
            AgentStatus::Errored(error) => Some(format!("error: {error}")),
            AgentStatus::Cancelled => Some("cancelled".to_owned()),
            _ => None,
        };
        self.with(id, |view| {
            view.status = status;
            if let Some(note) = note {
                view.push(LogEntry::Status(note));
            }
        });
    }

    fn apply_mail(&mut self, to: &str, mail: crate::runtime::Mail) {
        if to == USER && mail.from != USER {
            self.bell |= self.notify;
        }
        // Replies to the user are already visible as the sender's streamed text.
        if to == USER && mail.kind == "reply" {
            return;
        }
        let (target, from) = match to {
            USER => (mail.from.clone(), format!("{} → user", mail.from_label)),
            _ => (to.to_owned(), mail.from_label.clone()),
        };
        let entry = LogEntry::MailIn {
            from,
            kind: mail.kind.to_owned(),
            text: mail.text,
            read: (to != USER).then_some(false),
        };
        let attached: Vec<LogEntry> = mail
            .images
            .iter()
            .map(|path| {
                let bytes = std::fs::read(path).unwrap_or_default();
                self.image_entry(path.display().to_string(), &bytes)
            })
            .collect();
        self.with(&target, |view| {
            view.push(entry);
            attached.into_iter().for_each(|image| view.push(image));
        });
    }

    fn image_entry(&mut self, source: String, bytes: &[u8]) -> LogEntry {
        LogEntry::Image {
            source,
            decoded: self.images.add(bytes).ok(),
        }
    }

    fn with(&mut self, id: &str, f: impl FnOnce(&mut AgentView)) {
        if let Some(view) = self.agents.get_mut(id) {
            f(view);
        }
    }

    /// Agents in tree order (depth-first by spawn order) with their depth.
    pub fn tree(&self) -> Vec<(usize, &AgentView)> {
        let mut out = Vec::with_capacity(self.agents.len());
        self.walk(USER, 0, &mut out);
        out
    }

    fn walk<'a>(&'a self, parent: &str, depth: usize, out: &mut Vec<(usize, &'a AgentView)>) {
        for view in self.agents.values().filter(|view| view.parent == parent) {
            if !self.show_cancelled && !self.has_live(view) {
                continue;
            }
            out.push((depth, view));
            self.walk(&view.id, depth + 1, out);
        }
    }

    /// Whether `view` or any of its descendants is not cancelled.
    fn has_live(&self, view: &AgentView) -> bool {
        view.status.is_live()
            || self
                .agents
                .values()
                .any(|child| child.parent == view.id && self.has_live(child))
    }

    /// Cancelled agents left out of the tree.
    pub fn hidden(&self) -> usize {
        self.agents.len().saturating_sub(self.tree().len())
    }

    /// Show or hide cancelled agents, keeping the selected agent selected when it stays visible.
    pub fn toggle_cancelled(&mut self) {
        let selected = self.selected_id();
        self.show_cancelled = !self.show_cancelled;
        self.reselect(selected);
    }

    /// Keep `previous` selected after the tree changed; when it is gone, stay on the same row.
    fn reselect(&mut self, previous: Option<String>) {
        let tree = self.tree();
        self.selected = previous
            .and_then(|id| tree.iter().position(|(_, view)| view.id == id))
            .unwrap_or_else(|| self.selected.min(tree.len().saturating_sub(1)));
    }

    /// Id of the selected agent.
    pub fn selected_id(&self) -> Option<String> {
        let tree = self.tree();
        let index = self.selected.min(tree.len().saturating_sub(1));
        tree.get(index).map(|(_, view)| view.id.clone())
    }

    /// Move the selection by `delta` rows, clamped.
    pub fn select_by(&mut self, delta: isize) {
        let len = self.tree().len();
        if len == 0 {
            return;
        }
        self.selected = self.selected.saturating_add_signed(delta).min(len - 1);
        self.scroll = 0;
    }

    /// Grow (or with a negative `delta`, shrink) the agent tree.
    pub fn resize_tree(&mut self, delta: i16) {
        self.tree_adjust = self.tree_adjust.saturating_add(delta).clamp(-200, 200);
    }

    /// Mark the selected agent's log as seen.
    pub fn mark_seen(&mut self) {
        if let Some(id) = self.selected_id() {
            self.with(&id, |view| view.unseen = false);
        }
    }

    /// Input tokens read from the prompt cache, across the swarm.
    pub fn total_cache_read(&self) -> u64 {
        self.agents.values().map(|view| view.cache_read_tokens).sum()
    }

    /// Total tokens across the swarm.
    pub fn total_tokens(&self) -> (u64, u64) {
        self.agents.values().fold((0, 0), |(i, o), view| {
            (i + view.input_tokens, o + view.output_tokens)
        })
    }

    /// Slash commands matching the input while the command word is being typed; empty (popup
    /// closed) otherwise.
    pub fn completions(&self) -> Vec<&'static SlashCommand> {
        let word = self.input.as_str();
        if !word.starts_with('/') || word.contains(char::is_whitespace) {
            return Vec::new();
        }
        SLASH_COMMANDS
            .iter()
            .filter(|command| command.name.starts_with(word))
            .collect()
    }

    /// Move the popup highlight, wrapping around.
    pub fn move_completion(&mut self, delta: isize) {
        let len = self.completions().len();
        if len > 0 {
            let current = self.completion.min(len - 1) as isize;
            self.completion = (current + delta).rem_euclid(len as isize) as usize;
        }
    }

    /// Replace the input with the highlighted command (plus a space when it takes arguments).
    /// Returns `false` when the popup is closed.
    pub fn complete(&mut self) -> bool {
        let completions = self.completions();
        let Some(command) = completions.get(self.completion.min(completions.len().saturating_sub(1))) else {
            return false;
        };
        self.input = match command.args.is_empty() {
            true => command.name.to_owned(),
            false => format!("{} ", command.name),
        };
        self.cursor = None;
        self.completion = 0;
        true
    }

    /// Whether the input is exactly a complete command name, so Enter should run it.
    pub fn input_is_command(&self) -> bool {
        SLASH_COMMANDS.iter().any(|command| command.name == self.input)
    }
}

/// Log entries for restored messages: text from either side and tool calls (tool results only
/// decide whether a call failed).
pub fn history_entries(messages: &[Message]) -> Vec<LogEntry> {
    let blocks = || {
        messages
            .iter()
            .flat_map(|m| m.content.iter().map(move |b| (m.role, b)))
    };
    let failed: HashSet<&str> = blocks()
        .filter_map(|(_, block)| match block {
            ContentBlock::ToolResult(result) if result.status == ToolResultStatus::Error => {
                Some(result.tool_use_id.as_str())
            }
            _ => None,
        })
        .collect();
    blocks()
        .filter_map(|(role, block)| match block {
            ContentBlock::Text(text) if text.trim().is_empty() => None,
            ContentBlock::Text(text) if role == Role::User => Some(LogEntry::Input(text.clone())),
            ContentBlock::Text(text) => Some(LogEntry::Text(text.clone())),
            ContentBlock::ToolUse(call) => Some(LogEntry::Tool {
                name: call.name.clone(),
                summary: summarize_input(&call.input),
                ok: Some(!failed.contains(call.tool_use_id.as_str())),
            }),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Mail;

    fn spawned(id: &str, parent: &str) -> SwarmEvent {
        SwarmEvent::Spawned {
            id: id.into(),
            parent: parent.into(),
            name: format!("name-{id}"),
            task: format!("\ntask {id}\nmore detail"),
            cwd: PathBuf::from("/w"),
            branch: None,
        }
    }

    #[test]
    fn a_message_with_an_image_logs_the_image_after_the_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shot.png");
        std::fs::write(&path, super::super::images::png(8, 6)).unwrap();
        let mut app = App::default();
        app.apply(spawned("a1", USER));
        app.apply(SwarmEvent::Mail {
            to: "a1".into(),
            mail: Mail {
                from: USER.into(),
                from_label: USER.into(),
                kind: "message",
                text: "what is this?".into(),
                images: vec![path.clone()],
            },
        });
        let log = &app.agents["a1"].log;
        assert!(matches!(&log[log.len() - 2], LogEntry::MailIn { text, .. } if text == "what is this?"));
        match log.back() {
            Some(LogEntry::Image {
                source,
                decoded: Some(decoded),
            }) => {
                assert_eq!(source, &path.display().to_string());
                assert_eq!((decoded.width, decoded.height), (8, 6));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn orders_agents_depth_first_under_their_parents() {
        let mut app = App::default();
        for (id, parent) in [
            ("a1", USER),
            ("a2", "a1"),
            ("a3", USER),
            ("a4", "a1"),
            ("a5", "a2"),
        ] {
            app.apply(spawned(id, parent));
        }
        let tree: Vec<(usize, &str)> = app.tree().iter().map(|(d, v)| (*d, v.id.as_str())).collect();
        assert_eq!(tree, vec![(0, "a1"), (1, "a2"), (2, "a5"), (1, "a4"), (0, "a3")]);
    }

    #[test]
    fn tracks_what_an_agent_is_doing() {
        let mut app = App::default();
        app.apply(spawned("a1", USER));
        assert_eq!(app.agents["a1"].label(), "name-a1 (a1)");
        assert_eq!(app.agents["a1"].doing, "task a1");
        app.apply(SwarmEvent::ToolStarted {
            id: "a1".into(),
            name: "shell".into(),
            summary: "cargo test".into(),
        });
        assert_eq!(app.agents["a1"].doing, "shell cargo test");
    }

    #[test]
    fn accumulates_text_deltas_into_one_entry_and_closes_tool_calls() {
        let mut app = App::default();
        app.apply(spawned("a1", USER));
        for delta in ["hel", "lo"] {
            app.apply(SwarmEvent::Text {
                id: "a1".into(),
                delta: delta.into(),
            });
        }
        app.apply(SwarmEvent::ToolStarted {
            id: "a1".into(),
            name: "shell".into(),
            summary: "ls".into(),
        });
        app.apply(SwarmEvent::ToolFinished {
            id: "a1".into(),
            name: "shell".into(),
            ok: false,
        });
        let log = &app.agents["a1"].log;
        assert_eq!(log[1], LogEntry::Text("hello".into()));
        assert_eq!(
            log[2],
            LogEntry::Tool {
                name: "shell".into(),
                summary: "ls".into(),
                ok: Some(false)
            }
        );
    }

    #[test]
    fn shows_agent_to_user_messages_on_the_sender_and_skips_replies() {
        let mut app = App::default();
        app.apply(spawned("a1", USER));
        let mail = |kind: &'static str| Mail {
            from: "a1".into(),
            from_label: "a1".into(),
            kind,
            text: "hi".into(),
            images: Vec::new(),
        };
        app.apply(SwarmEvent::Mail {
            to: USER.into(),
            mail: mail("reply"),
        });
        app.apply(SwarmEvent::Mail {
            to: USER.into(),
            mail: mail("message"),
        });
        assert_eq!(app.agents["a1"].log.len(), 2);
        assert!(matches!(&app.agents["a1"].log[1], LogEntry::MailIn { from, .. } if from == "a1 → user"));
    }

    #[test]
    fn notify_rings_the_bell_for_messages_to_the_user_only() {
        let mail = |from: &str| Mail {
            from: from.into(),
            from_label: from.into(),
            kind: "reply",
            text: "done".into(),
            images: Vec::new(),
        };
        let mut quiet = App::default();
        quiet.apply(spawned("a1", USER));
        quiet.apply(SwarmEvent::Mail {
            to: USER.into(),
            mail: mail("a1"),
        });
        assert!(!quiet.bell);

        let mut app = App {
            notify: true,
            ..App::default()
        };
        app.apply(spawned("a1", USER));
        app.apply(SwarmEvent::Mail {
            to: "a1".into(),
            mail: mail(USER),
        });
        assert!(!app.bell, "mail from the user does not ring");
        app.apply(SwarmEvent::Mail {
            to: USER.into(),
            mail: mail("a1"),
        });
        assert!(app.bell);
    }

    #[test]
    fn marks_mail_read_once_the_agent_takes_it() {
        let mut app = App::default();
        app.apply(spawned("a1", USER));
        let mail = Mail {
            from: USER.into(),
            from_label: USER.into(),
            kind: "message",
            text: "hi".into(),
            images: Vec::new(),
        };
        let read = |app: &App| match app.agents["a1"].log.back() {
            Some(LogEntry::MailIn { read, .. }) => *read,
            other => panic!("{other:?}"),
        };
        app.apply(SwarmEvent::Mail {
            to: "a1".into(),
            mail: mail.clone(),
        });
        assert_eq!(read(&app), Some(false));
        app.apply(SwarmEvent::MailRead { id: "a1".into() });
        assert_eq!(read(&app), Some(true));
        app.apply(SwarmEvent::Mail {
            to: "a1".into(),
            mail,
        });
        assert_eq!(read(&app), Some(false));
    }

    #[test]
    fn describes_the_live_activity_of_a_running_agent() {
        let mut app = App::default();
        app.apply(spawned("a1", USER));
        let activity = |app: &App| app.agents["a1"].activity();
        assert_eq!(activity(&app).as_deref(), Some("starting…"));
        app.apply(SwarmEvent::Status {
            id: "a1".into(),
            status: AgentStatus::Running,
        });
        assert_eq!(activity(&app).as_deref(), Some("thinking…"));
        app.apply(SwarmEvent::Text {
            id: "a1".into(),
            delta: "hm".into(),
        });
        assert_eq!(activity(&app).as_deref(), Some("writing…"));
        app.apply(SwarmEvent::ToolStarted {
            id: "a1".into(),
            name: "shell".into(),
            summary: "cargo test".into(),
        });
        assert_eq!(activity(&app).as_deref(), Some("running shell: cargo test…"));
        app.apply(SwarmEvent::ToolFinished {
            id: "a1".into(),
            name: "shell".into(),
            ok: true,
        });
        assert_eq!(activity(&app).as_deref(), Some("thinking…"));
        app.apply(SwarmEvent::Status {
            id: "a1".into(),
            status: AgentStatus::Idle,
        });
        assert_eq!(activity(&app), None);
        app.apply(SwarmEvent::Status {
            id: "a1".into(),
            status: AgentStatus::Running,
        });
        app.apply(SwarmEvent::ModelQueued {
            id: "a1".into(),
            queued: true,
        });
        assert_eq!(activity(&app).as_deref(), Some("queued for a model slot…"));
        app.apply(SwarmEvent::ModelQueued {
            id: "a1".into(),
            queued: false,
        });
        assert_eq!(activity(&app).as_deref(), Some("thinking…"));
    }

    #[test]
    fn logs_keep_only_the_newest_entries() {
        let mut app = App::default();
        app.apply(spawned("a1", USER));
        for i in 0..MAX_LOG_ENTRIES + 10 {
            app.apply(SwarmEvent::ToolStarted {
                id: "a1".into(),
                name: "shell".into(),
                summary: format!("cmd {i}"),
            });
        }
        app.apply(SwarmEvent::Text {
            id: "a1".into(),
            delta: "done".into(),
        });
        let log = &app.agents["a1"].log;
        assert_eq!(log.len(), MAX_LOG_ENTRIES);
        assert_eq!(log.back(), Some(&LogEntry::Text("done".into())));
        let first = LogEntry::Tool {
            name: "shell".into(),
            summary: "cmd 11".into(),
            ok: None,
        };
        assert_eq!(log.front(), Some(&first));
    }

    #[test]
    fn backfills_a_resumed_agents_conversation_after_its_task() {
        use strands_agents::types::{ToolResultBlock, ToolUseBlock};
        let mut app = App::default();
        app.apply(spawned("a1", USER));
        app.apply(SwarmEvent::Status {
            id: "a1".into(),
            status: AgentStatus::Errored("boom".into()),
        });
        let call = |id: &str, command: &str| {
            ContentBlock::ToolUse(ToolUseBlock::new(
                "shell",
                id,
                serde_json::json!({"command": command}),
            ))
        };
        let messages = vec![
            Message::new(Role::User, vec!["\ntask a1\nmore detail".into()]),
            Message::new(
                Role::Assistant,
                vec!["looking".into(), call("t1", "ls"), call("t2", "false")],
            ),
            Message::new(
                Role::User,
                vec![
                    ContentBlock::ToolResult(ToolResultBlock::success("t1", vec![])),
                    ContentBlock::ToolResult(ToolResultBlock::error("t2", "exit 1")),
                ],
            ),
            Message::new(Role::User, vec!["[message from a2] hi".into()]),
        ];
        app.apply(SwarmEvent::History {
            id: "a1".into(),
            messages,
            omitted: 0,
        });
        let tool = |summary: &str, ok| LogEntry::Tool {
            name: "shell".into(),
            summary: summary.into(),
            ok: Some(ok),
        };
        let log: Vec<_> = app.agents["a1"].log.iter().cloned().collect();
        assert_eq!(
            log,
            vec![
                LogEntry::Task("\ntask a1\nmore detail".into()),
                LogEntry::Note("restored conversation".into()),
                LogEntry::Text("looking".into()),
                tool("ls", true),
                tool("false", false),
                LogEntry::Input("[message from a2] hi".into()),
                LogEntry::Note("restarted".into()),
                LogEntry::Status("error: boom".into()),
            ]
        );
    }

    #[test]
    fn hides_cancelled_agents_and_keeps_the_selection() {
        let mut app = App::default();
        for (id, parent) in [
            ("a1", USER),
            ("a2", "a1"),
            ("a3", "a2"),
            ("a4", USER),
            ("a5", USER),
        ] {
            app.apply(spawned(id, parent));
        }
        let cancel = |app: &mut App, id: &str| {
            app.apply(SwarmEvent::Status {
                id: id.into(),
                status: AgentStatus::Cancelled,
            })
        };
        let ids = |app: &App| app.tree().iter().map(|(_, v)| v.id.clone()).collect::<Vec<_>>();
        app.selected = 3; // a4
        cancel(&mut app, "a2");
        assert_eq!(ids(&app), ["a1", "a2", "a3", "a4", "a5"], "a2 has a live child");
        cancel(&mut app, "a3");
        assert_eq!(ids(&app), ["a1", "a4", "a5"]);
        assert_eq!(
            app.selected_id().as_deref(),
            Some("a4"),
            "selection follows its agent"
        );
        assert_eq!(app.hidden(), 2);

        cancel(&mut app, "a4");
        assert_eq!(
            app.selected_id().as_deref(),
            Some("a5"),
            "a hidden selection moves on"
        );
        cancel(&mut app, "a5");
        assert_eq!(app.selected_id().as_deref(), Some("a1"));

        app.toggle_cancelled();
        assert_eq!(ids(&app), ["a1", "a2", "a3", "a4", "a5"]);
        assert_eq!(app.selected_id().as_deref(), Some("a1"));
        app.select_by(4);
        assert_eq!(app.selected_id().as_deref(), Some("a5"));
        app.toggle_cancelled();
        assert_eq!(app.selected_id().as_deref(), Some("a1"));
        app.select_by(4);
        assert_eq!(
            app.selected_id().as_deref(),
            Some("a1"),
            "only visible rows are selectable"
        );
    }

    #[test]
    fn the_slash_popup_filters_moves_and_completes() {
        let mut app = App {
            input: "/".into(),
            ..Default::default()
        };
        assert_eq!(app.completions().len(), SLASH_COMMANDS.len());
        app.move_completion(-1);
        assert_eq!(app.completions()[app.completion].name, "/quit");

        app.input = "/mo".into();
        app.completion = 0;
        let names: Vec<_> = app.completions().iter().map(|c| c.name).collect();
        assert_eq!(names, ["/model"]);
        assert!(!app.input_is_command());
        assert!(app.complete());
        assert_eq!(app.input, "/model ");
        assert!(app.completions().is_empty(), "closes once arguments start");
        assert!(!app.complete());

        app.input = "/cancel".into();
        assert!(app.input_is_command());
        app.input = "hello".into();
        assert!(app.completions().is_empty());
    }

    #[test]
    fn reclaimed_disk_shows_as_a_notice_and_an_empty_requested_pass_says_so() {
        let mut app = App::default();
        let reclaimed = crate::gc::Collected {
            id: "a3".into(),
            removed: vec![(PathBuf::from("/t-worktrees/r/a3"), 3 << 30)],
            kept: Vec::new(),
        };
        app.apply(SwarmEvent::Reclaimed {
            collected: vec![crate::gc::Collected::default()],
            requested: false,
        });
        assert_eq!(app.notice, None, "a quiet automatic pass stays quiet");
        app.apply(SwarmEvent::Reclaimed {
            collected: vec![reclaimed],
            requested: false,
        });
        let notice = app.notice.clone().unwrap();
        assert!(
            notice.starts_with("gc: reclaimed 3.0 GB from 1 cancelled agents"),
            "{notice}"
        );
        app.apply(SwarmEvent::Reclaimed {
            collected: Vec::new(),
            requested: true,
        });
        assert_eq!(app.notice.as_deref(), Some("gc: nothing to reclaim"));
        assert!(SLASH_COMMANDS.iter().any(|command| command.name == "/gc"));
    }

    #[test]
    fn a_compaction_is_noted_in_the_agents_log() {
        let mut app = App::default();
        app.apply(spawned("a1", USER));
        app.apply(SwarmEvent::Compacted {
            id: "a1".into(),
            before: 305_000,
            after: 98_100,
        });
        assert!(app.agents["a1"]
            .log
            .iter()
            .any(|entry| matches!(entry, LogEntry::Status(note)
            if note == "context compacted: older turns summarized, 305.0k → 98.1k tokens")));
    }

    #[test]
    fn crossing_the_summarize_threshold_is_noted_once() {
        let mut app = App::default();
        app.apply(spawned("a1", USER));
        let usage = |context: Option<f64>| SwarmEvent::Usage {
            id: "a1".into(),
            input_tokens: 1,
            output_tokens: 1,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            context,
        };
        let notes = |app: &App| {
            app.agents["a1"]
                .log
                .iter()
                .filter(|entry| matches!(entry, LogEntry::Status(note) if note.contains("summarized")))
                .count()
        };
        // A restored value above the threshold is not a crossing.
        app.apply(usage(Some(0.95)));
        assert_eq!(notes(&app), 0);
        app.apply(usage(Some(0.3)));
        app.apply(usage(Some(0.91)));
        app.apply(usage(Some(0.97)));
        assert_eq!(notes(&app), 1);
        assert!(app.agents["a1"]
            .log
            .iter()
            .any(|entry| matches!(entry, LogEntry::Status(note)
            if note == "context 91% of the compaction point — older turns will be summarized soon")));
    }
}
