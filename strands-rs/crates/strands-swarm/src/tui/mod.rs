//! Terminal UI: the spawn tree on the left, the selected agent's live log on the right, and an
//! input line that messages the selected agent (or starts a root agent).

pub mod app;
pub mod view;

use std::time::Duration;

use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use futures::StreamExt;
use strands_agents::Result;
use tokio::sync::mpsc::UnboundedReceiver;

use crate::runtime::{SpawnRequest, Swarm, SwarmEvent, USER};
pub use app::{parse_command, App, Command};

/// Why the TUI returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// The user quit.
    Quit,
    /// An agent installed a new binary; the host should exit with
    /// [`crate::RELOAD_EXIT_CODE`].
    Reload,
}

/// Run the TUI until the user quits, then cancel every agent. `root`, when given, is spawned
/// immediately.
pub async fn run(
    swarm: Swarm,
    mut events: UnboundedReceiver<SwarmEvent>,
    root: Option<SpawnRequest>,
) -> Result<Exit> {
    let mut terminal = ratatui::init();
    let mut app = App::default();
    if let Some(request) = root {
        let outcome = swarm.spawn(USER, request).await;
        app.notice = outcome.err().map(|error| format!("error: {error}"));
    }
    let mut keys = EventStream::new();
    let mut ticker = tokio::time::interval(Duration::from_millis(120));
    let mut tick = 0usize;

    // While the plain log is shown on the main screen the TUI is not drawn; any key returns.
    let mut plain = false;
    let outcome = loop {
        app.mark_seen();
        if !plain {
            if let Err(error) = terminal.draw(|frame| view::draw(frame, &app, swarm.run_id(), tick)) {
                break Err(strands_agents::Error::other(format!("terminal error: {error}")));
            }
        }
        tokio::select! {
            Some(event) = events.recv() => {
                app.apply(event);
                while let Ok(event) = events.try_recv() {
                    app.apply(event);
                }
            }
            Some(Ok(event)) = keys.next() => match plain {
                true if matches!(event, Event::Key(key) if key.kind == KeyEventKind::Press) => {
                    plain = false;
                    if let Err(error) = leave_plain(&mut terminal) {
                        break Err(strands_agents::Error::other(format!("terminal error: {error}")));
                    }
                }
                true => {}
                false => on_terminal_event(&swarm, &mut app, event).await,
            },
            _ = ticker.tick() => tick = tick.wrapping_add(1),
            _ = swarm.reload_requested() => break Ok(Exit::Reload),
        }
        if app.quit {
            break Ok(Exit::Quit);
        }
        if std::mem::take(&mut app.show_plain) {
            plain = true;
            if let Err(error) = enter_plain(&app) {
                break Err(strands_agents::Error::other(format!("terminal error: {error}")));
            }
        }
    };

    ratatui::restore();
    swarm.shutdown().await;
    outcome
}

/// Leave the alternate screen and print the selected agent's log as plain text, so it lands in
/// the terminal's own scrollback where it can be scrolled, searched and copied natively.
fn enter_plain(app: &App) -> std::io::Result<()> {
    use std::io::Write;
    let text = app
        .selected_id()
        .and_then(|id| app.agents.get(&id))
        .map_or_else(|| "(no agent selected)\n".to_owned(), view::plain_log);
    let mut out = std::io::stdout();
    crossterm::execute!(out, crossterm::terminal::LeaveAlternateScreen)?;
    crossterm::terminal::disable_raw_mode()?;
    write!(out, "\n{text}\n— press any key to return to the swarm —")?;
    out.flush()?;
    crossterm::terminal::enable_raw_mode()
}

fn leave_plain(terminal: &mut ratatui::DefaultTerminal) -> std::io::Result<()> {
    crossterm::execute!(std::io::stdout(), crossterm::terminal::EnterAlternateScreen)?;
    terminal.clear()
}

async fn on_terminal_event(swarm: &Swarm, app: &mut App, event: Event) {
    let Event::Key(key) = event else { return };
    if key.kind != KeyEventKind::Press {
        return;
    }
    if let Some(command) = on_key(app, key) {
        submit(swarm, app, command).await;
    }
}

/// Apply a key press to the state; returns a command when the input line was submitted.
fn on_key(app: &mut App, key: KeyEvent) -> Option<Command> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Char('c') if ctrl => app.quit = true,
        KeyCode::Char(c) => app.input.push(c),
        KeyCode::Backspace => drop(app.input.pop()),
        KeyCode::Up => app.select_by(-1),
        KeyCode::Down => app.select_by(1),
        KeyCode::PageUp => app.scroll = app.scroll.saturating_add(10),
        KeyCode::PageDown => app.scroll = app.scroll.saturating_sub(10),
        KeyCode::End => app.scroll = 0,
        KeyCode::Esc => app.input.clear(),
        KeyCode::F(2) => app.show_plain = true,
        KeyCode::Enter => {
            let line = std::mem::take(&mut app.input);
            return parse_command(&line, !app.agents.is_empty());
        }
        _ => {}
    }
    None
}

async fn submit(swarm: &Swarm, app: &mut App, command: Command) {
    let selected = app.selected_id();
    let outcome = match (command, selected) {
        (Command::Quit, _) => {
            app.quit = true;
            Ok(String::new())
        }
        (Command::NewRoot(task), _) => {
            let request = SpawnRequest {
                task,
                ..Default::default()
            };
            swarm
                .spawn(USER, request)
                .await
                .map(|id| format!("started {}", swarm.label(&id)))
        }
        (Command::Goal(goal), _) => {
            let request = SpawnRequest {
                task: goal.clone(),
                goal: Some(goal),
                ..Default::default()
            };
            swarm
                .spawn(USER, request)
                .await
                .map(|id| format!("started {} with a goal", swarm.label(&id)))
        }
        (Command::Message(text), Some(id)) => swarm
            .send(USER, &id, text)
            .map(|()| format!("sent to {}", swarm.label(&id))),
        (Command::Cancel, Some(id)) => swarm
            .cancel(&id)
            .map(|()| format!("cancelled {}", swarm.label(&id))),
        (Command::ShowModel, Some(id)) => Ok(match swarm.agent(&id).and_then(|info| info.model) {
            Some(model) => format!("{} uses {model}", swarm.label(&id)),
            None => format!("{} uses the default model", swarm.label(&id)),
        }),
        (Command::SetModel(model), Some(id)) => swarm
            .set_model(&id, model)
            .await
            .map(|model| format!("{} now uses {model}", swarm.label(&id))),
        (_, None) => Ok("no agent selected".to_owned()),
    };
    app.notice = Some(outcome.unwrap_or_else(|error| format!("error: {error}")));
}

/// End-to-end: drive [`App`] with key events and swarm events, render into a [`TestBackend`]
/// and assert on what is on screen.
#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use strands_agents::types::{ContentBlock, Message, Role, ToolUseBlock};

    use super::*;
    use crate::runtime::{AgentStatus, Mail};

    const WIDTH: u16 = 100;
    const HEIGHT: u16 = 24;

    fn screen(app: &App) -> String {
        let mut terminal = Terminal::new(TestBackend::new(WIDTH, HEIGHT)).unwrap();
        terminal.draw(|frame| view::draw(frame, app, "run1", 0)).unwrap();
        let buffer = terminal.backend().buffer();
        (0..HEIGHT)
            .map(|y| (0..WIDTH).map(|x| buffer[(x, y)].symbol()).collect::<String>() + "\n")
            .collect()
    }

    fn press(app: &mut App, code: KeyCode) -> Option<Command> {
        on_key(app, KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn type_line(app: &mut App, line: &str) -> Option<Command> {
        for c in line.chars() {
            assert_eq!(press(app, KeyCode::Char(c)), None);
        }
        press(app, KeyCode::Enter)
    }

    fn spawn(app: &mut App, id: &str, parent: &str) {
        app.apply(SwarmEvent::Spawned {
            id: id.into(),
            parent: parent.into(),
            name: format!("name-{id}"),
            task: format!("task for {id}"),
            cwd: PathBuf::from("/w"),
            branch: None,
        });
        app.apply(SwarmEvent::Status {
            id: id.into(),
            status: AgentStatus::Idle,
        });
    }

    #[test]
    fn an_empty_swarm_asks_for_a_root_task_and_submits_it() {
        let mut app = App::default();
        let before = screen(&app);
        assert!(before.contains("Type a task below"), "{before}");
        assert!(before.contains("new root task"), "{before}");

        for c in "fix the bug".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        assert!(screen(&app).contains("fix the bug"));
        press(&mut app, KeyCode::Backspace);
        assert!(screen(&app).contains("fix the bu "));

        let command = press(&mut app, KeyCode::Enter);
        assert_eq!(command, Some(Command::NewRoot("fix the bu".into())));
        assert!(app.input.is_empty());
    }

    #[test]
    fn the_tree_shows_children_under_parents_and_arrows_select() {
        let mut app = App::default();
        spawn(&mut app, "a1", USER);
        spawn(&mut app, "a2", "a1");
        app.mark_seen();
        let shown = screen(&app);
        assert!(shown.contains("● name-a1 a1"), "{shown}");
        assert!(shown.contains("  ● name-a2 a2"), "{shown}");
        assert!(shown.contains("message → name-a1 (a1)"), "{shown}");
        assert!(shown.contains("task for a1"), "{shown}");

        press(&mut app, KeyCode::Down);
        let shown = screen(&app);
        assert!(shown.contains("message → name-a2 (a2)"), "{shown}");
        assert!(shown.contains("task for a2"), "{shown}");
        assert_eq!(
            type_line(&mut app, "status?"),
            Some(Command::Message("status?".into()))
        );
        assert_eq!(type_line(&mut app, "/cancel"), Some(Command::Cancel));
    }

    #[test]
    fn the_log_shows_text_tool_calls_and_mail() {
        let mut app = App::default();
        spawn(&mut app, "a1", USER);
        app.apply(SwarmEvent::Text {
            id: "a1".into(),
            delta: "Reading the ".into(),
        });
        app.apply(SwarmEvent::Text {
            id: "a1".into(),
            delta: "code.".into(),
        });
        app.apply(SwarmEvent::ToolStarted {
            id: "a1".into(),
            name: "shell".into(),
            summary: "cargo test".into(),
        });
        app.apply(SwarmEvent::ToolFinished {
            id: "a1".into(),
            name: "shell".into(),
            ok: false,
        });
        app.apply(SwarmEvent::Mail {
            to: "a1".into(),
            mail: Mail {
                from: "a2".into(),
                from_label: "a2".into(),
                kind: "message",
                text: "all green".into(),
            },
        });
        let shown = screen(&app);
        for expected in [
            "Reading the code.",
            "✗ shell cargo test",
            "✉ a2 (message)",
            "  all green",
        ] {
            assert!(shown.contains(expected), "missing {expected:?} in\n{shown}");
        }
    }

    #[test]
    fn a_resumed_agent_shows_its_restored_conversation() {
        let mut app = App::default();
        spawn(&mut app, "a1", USER);
        let call = ToolUseBlock::new("shell", "t1", serde_json::json!({"command": "ls -la"}));
        app.apply(SwarmEvent::History {
            id: "a1".into(),
            messages: vec![
                Message::new(Role::User, vec!["task for a1".into()]),
                Message::new(
                    Role::Assistant,
                    vec!["Listing files.".into(), ContentBlock::ToolUse(call)],
                ),
                Message::new(Role::User, vec!["[message from a2] done".into()]),
            ],
            omitted: 7,
        });
        let shown = screen(&app);
        for expected in [
            "task for a1",
            "restored conversation (7 earlier messages not shown)",
            "Listing files.",
            "✓ shell ls -la",
            "◀ input",
            "[message from a2] done",
            "— restarted —",
        ] {
            assert!(shown.contains(expected), "missing {expected:?} in\n{shown}");
        }
    }

    #[test]
    fn scrolling_pages_through_a_capped_log() {
        let mut app = App::default();
        spawn(&mut app, "a1", USER);
        for i in 0..app::MAX_LOG_ENTRIES + 50 {
            app.apply(SwarmEvent::ToolStarted {
                id: "a1".into(),
                name: "shell".into(),
                summary: format!("step {i}"),
            });
        }
        let last = format!("⋯ shell step {}", app::MAX_LOG_ENTRIES + 49);
        let shown = screen(&app);
        let last_log_row = shown.lines().nth(usize::from(HEIGHT) - 4).unwrap();
        assert!(
            last_log_row.contains(&last),
            "newest entry on the pane's last row\n{shown}"
        );

        press(&mut app, KeyCode::PageUp);
        let shown = screen(&app);
        assert!(!shown.contains(&last), "{shown}");
        assert!(shown.contains(&format!("step {}", app::MAX_LOG_ENTRIES + 30)));

        app.scroll = u16::MAX;
        let top = screen(&app);
        assert!(
            top.contains("⋯ shell step 50 "),
            "oldest kept entry at the top\n{top}"
        );
        assert!(!top.contains("step 49 "), "{top}");

        press(&mut app, KeyCode::End);
        assert!(screen(&app).contains(&last));
    }

    #[test]
    fn f2_asks_for_the_selected_log_as_plain_text() {
        let mut app = App::default();
        spawn(&mut app, "a1", USER);
        app.apply(SwarmEvent::Text {
            id: "a1".into(),
            delta: "a long line that the pane would wrap\nsecond".into(),
        });
        app.apply(SwarmEvent::ToolStarted {
            id: "a1".into(),
            name: "shell".into(),
            summary: "ls".into(),
        });
        press(&mut app, KeyCode::F(2));
        assert!(app.show_plain);
        let text = view::plain_log(&app.agents["a1"]);
        assert!(text.starts_with("name-a1 (a1) · idle · /w"), "{text}");
        assert!(
            text.contains(
                "▶ task\n  task for a1\n\na long line that the pane would wrap\nsecond\n\n⋯ shell ls\n"
            ),
            "{text}"
        );
        assert!(!text.contains('│') && !text.contains('─'), "{text}");
    }

    #[test]
    fn log_rows_carry_no_border_glyphs() {
        let mut app = App::default();
        spawn(&mut app, "a1", USER);
        app.apply(SwarmEvent::Text {
            id: "a1".into(),
            delta: "copy me".into(),
        });
        let shown = screen(&app);
        let row = shown.lines().find(|l| l.contains("copy me")).unwrap();
        let log_part = &row[row.find("copy me").unwrap()..];
        assert_eq!(log_part.trim_end(), "copy me", "{shown}");
    }

    #[test]
    fn ctrl_c_quits() {
        let mut app = App::default();
        on_key(&mut app, KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(app.quit);
    }
}
