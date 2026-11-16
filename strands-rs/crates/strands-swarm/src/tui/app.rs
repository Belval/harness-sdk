//! UI state: a fold over [`SwarmEvent`]s plus the input line. No terminal I/O here.

use std::collections::VecDeque;
use std::path::PathBuf;

use indexmap::IndexMap;

use crate::runtime::{AgentStatus, SwarmEvent, USER};

/// Entries kept per agent log; the oldest are dropped beyond this.
pub const MAX_LOG_ENTRIES: usize = 2000;

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
    /// Cumulative input tokens.
    pub input_tokens: u64,
    /// Cumulative output tokens.
    pub output_tokens: u64,
    /// Model id, once known.
    pub model: Option<String>,
    /// Log, oldest first, at most [`MAX_LOG_ENTRIES`] entries.
    pub log: VecDeque<LogEntry>,
    /// Log entries the user has not looked at.
    pub unseen: bool,
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
            AgentStatus::Starting => return Some("starting…".to_owned()),
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
    /// The input line.
    pub input: String,
    /// Lines scrolled up from the bottom of the log; 0 follows new output.
    pub scroll: u16,
    /// Last error or notice, shown in the footer.
    pub notice: Option<String>,
    /// Set when the user asked to quit.
    pub quit: bool,
}

impl App {
    /// Fold one swarm event into the state.
    pub fn apply(&mut self, event: SwarmEvent) {
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
                    model: None,
                    log: VecDeque::from([LogEntry::Task(task)]),
                    unseen: true,
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
            SwarmEvent::Usage {
                id,
                input_tokens,
                output_tokens,
            } => self.with(&id, |view| {
                view.input_tokens = input_tokens;
                view.output_tokens = output_tokens;
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
        self.with(&target, |view| view.push(entry));
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
            out.push((depth, view));
            self.walk(&view.id, depth + 1, out);
        }
    }

    /// Id of the selected agent.
    pub fn selected_id(&self) -> Option<String> {
        let tree = self.tree();
        let index = self.selected.min(tree.len().saturating_sub(1));
        tree.get(index).map(|(_, view)| view.id.clone())
    }

    /// Move the selection by `delta` rows, clamped.
    pub fn select_by(&mut self, delta: isize) {
        let len = self.agents.len();
        if len == 0 {
            return;
        }
        self.selected = self.selected.saturating_add_signed(delta).min(len - 1);
        self.scroll = 0;
    }

    /// Mark the selected agent's log as seen.
    pub fn mark_seen(&mut self) {
        if let Some(id) = self.selected_id() {
            self.with(&id, |view| view.unseen = false);
        }
    }

    /// Total tokens across the swarm.
    pub fn total_tokens(&self) -> (u64, u64) {
        self.agents.values().fold((0, 0), |(i, o), view| {
            (i + view.input_tokens, o + view.output_tokens)
        })
    }
}

/// What the input line asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Start a new root agent with this task.
    NewRoot(String),
    /// Start a new root agent that keeps working until this goal is met.
    Goal(String),
    /// Send a message to the selected agent.
    Message(String),
    /// Cancel the selected agent.
    Cancel,
    /// Show the selected agent's model.
    ShowModel,
    /// Switch the selected agent's model; `None` returns it to the default.
    SetModel(Option<String>),
    /// Quit.
    Quit,
}

/// Parse a submitted input line. `has_agents` decides whether plain text starts a root agent.
pub fn parse_command(line: &str, has_agents: bool) -> Option<Command> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    if let Some(goal) = line.strip_prefix("/goal ") {
        return Some(Command::Goal(goal.trim().to_owned()));
    }
    if let Some(task) = line.strip_prefix("/new ") {
        return Some(Command::NewRoot(task.trim().to_owned()));
    }
    if let Some(model) = line.strip_prefix("/model ") {
        return Some(Command::SetModel(match model.trim() {
            "default" => None,
            id => Some(id.to_owned()),
        }));
    }
    match line {
        "/cancel" => return Some(Command::Cancel),
        "/model" => return Some(Command::ShowModel),
        "/quit" | "/q" => return Some(Command::Quit),
        _ => {}
    }
    match has_agents {
        true => Some(Command::Message(line.to_owned())),
        false => Some(Command::NewRoot(line.to_owned())),
    }
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
    fn marks_mail_read_once_the_agent_takes_it() {
        let mut app = App::default();
        app.apply(spawned("a1", USER));
        let mail = Mail {
            from: USER.into(),
            from_label: USER.into(),
            kind: "message",
            text: "hi".into(),
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
    fn parses_input_lines() {
        assert_eq!(
            parse_command("fix it", false),
            Some(Command::NewRoot("fix it".into()))
        );
        assert_eq!(
            parse_command("fix it", true),
            Some(Command::Message("fix it".into()))
        );
        assert_eq!(
            parse_command("/new other", true),
            Some(Command::NewRoot("other".into()))
        );
        assert_eq!(parse_command("/cancel", true), Some(Command::Cancel));
        assert_eq!(
            parse_command("/goal ship it", true),
            Some(Command::Goal("ship it".into()))
        );
        assert_eq!(parse_command("/model", true), Some(Command::ShowModel));
        assert_eq!(
            parse_command("/model  anthropic/claude-x ", true),
            Some(Command::SetModel(Some("anthropic/claude-x".into())))
        );
        assert_eq!(
            parse_command("/model default", true),
            Some(Command::SetModel(None))
        );
        assert_eq!(parse_command("  ", true), None);
    }
}
