//! UI state: a fold over [`SwarmEvent`]s plus the input line. No terminal I/O here.

use std::path::PathBuf;

use indexmap::IndexMap;

use crate::runtime::{AgentStatus, SwarmEvent, USER};

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
    /// A message the agent received.
    MailIn {
        /// Sender label.
        from: String,
        /// `message`, `reply` or `error`.
        kind: String,
        /// Body.
        text: String,
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
    /// Log.
    pub log: Vec<LogEntry>,
    /// Log entries the user has not looked at.
    pub unseen: bool,
}

impl AgentView {
    /// `parser (a2)`.
    pub fn label(&self) -> String {
        format!("{} ({})", self.name, self.id)
    }

    fn push(&mut self, entry: LogEntry) {
        self.log.push(entry);
        self.unseen = true;
    }

    fn push_text(&mut self, delta: &str) {
        match self.log.last_mut() {
            Some(LogEntry::Text(text)) => text.push_str(delta),
            _ => self.log.push(LogEntry::Text(delta.to_owned())),
        }
        self.unseen = true;
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
                    log: vec![LogEntry::Task(task)],
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
    match line {
        "/cancel" => return Some(Command::Cancel),
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
        assert_eq!(parse_command("  ", true), None);
    }
}
