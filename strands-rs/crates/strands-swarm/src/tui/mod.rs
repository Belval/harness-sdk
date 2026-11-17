//! Terminal UI: the spawn tree on the left, the selected agent's live log on the right, and an
//! input line that messages the selected agent (or starts a root agent).

pub mod app;
pub mod commands;
mod edit;
pub mod images;
pub mod select;
pub mod view;

use std::time::Duration;

use crossterm::event::{
    Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent,
    MouseEventKind,
};
use futures::StreamExt;
use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect};
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
    notify: bool,
) -> Result<Exit> {
    let mut terminal = ratatui::init();
    set_mouse_capture(true);
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        set_mouse_capture(false);
        hook(info);
    }));
    // The last frame, for copying a mouse selection out of it.
    let mut screen = Buffer::empty(Rect::default());
    // Ask the terminal which graphics protocol it speaks: after raw mode, before reading events.
    let mut app = App {
        images: images::Images::detect(),
        notify,
        ..Default::default()
    };
    if let Some(request) = root {
        let outcome = swarm.spawn(USER, request).await;
        app.notice = outcome.err().map(|error| format!("error: {error}"));
    }
    let mut keys = EventStream::new();
    let mut ticker = tokio::time::interval(Duration::from_millis(120));
    let mut tick = 0usize;

    // While the plain log is shown on the main screen the TUI is not drawn; any key returns.
    let mut plain = false;
    // Token totals the header's cost estimate was last computed for.
    let mut spent = None;
    let outcome = loop {
        app.mark_seen();
        if app.status.is_some() {
            app.status = Some(swarm.status());
        }
        let tokens = app.total_tokens();
        if app.cost.is_some() || spent != Some(tokens) {
            let report = swarm.cost();
            app.spend = Some(report.total());
            app.cost = app.cost.is_some().then_some(report);
            spent = Some(tokens);
        }
        if std::mem::take(&mut app.bell) {
            ring_bell();
        }
        if !plain {
            match terminal.draw(|frame| view::draw(frame, &app, swarm.run_id(), tick)) {
                Ok(frame) => screen.clone_from(frame.buffer),
                Err(error) => break Err(strands_agents::Error::other(format!("terminal error: {error}"))),
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
                false => on_terminal_event(&swarm, &mut app, event, &screen).await,
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

    set_mouse_capture(false);
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
    set_mouse_capture(false);
    crossterm::execute!(out, crossterm::terminal::LeaveAlternateScreen)?;
    crossterm::terminal::disable_raw_mode()?;
    write!(out, "\n{text}\n— press any key to return to the swarm —")?;
    out.flush()?;
    crossterm::terminal::enable_raw_mode()
}

/// BEL: terminals (and tmux) flag the window or raise a notification.
fn ring_bell() {
    use std::io::Write;
    let mut out = std::io::stdout();
    let _ = out.write_all(b"\x07").and_then(|()| out.flush());
}

fn leave_plain(terminal: &mut ratatui::DefaultTerminal) -> std::io::Result<()> {
    crossterm::execute!(std::io::stdout(), crossterm::terminal::EnterAlternateScreen)?;
    set_mouse_capture(true);
    terminal.clear()
}

async fn on_terminal_event(swarm: &Swarm, app: &mut App, event: Event, screen: &Buffer) {
    match event {
        Event::Key(key) if key.kind == KeyEventKind::Press => {
            if let Some(command) = on_key(app, key) {
                submit(swarm, app, command).await;
            }
        }
        Event::Mouse(mouse) => {
            if let Some(text) = on_mouse(app, mouse, screen) {
                copy_to_clipboard(app, &text);
            }
        }
        _ => {}
    }
}

/// Turn mouse capture on (selection and wheel go to the TUI) or off (the terminal's own).
fn set_mouse_capture(on: bool) {
    let mut out = std::io::stdout();
    let _ = match on {
        true => crossterm::execute!(out, crossterm::event::EnableMouseCapture),
        false => crossterm::execute!(out, crossterm::event::DisableMouseCapture),
    };
}

fn copy_to_clipboard(app: &mut App, text: &str) {
    use std::io::Write;
    let mut out = std::io::stdout();
    let written = out
        .write_all(select::osc52(text).as_bytes())
        .and_then(|()| out.flush());
    app.notice = Some(match written {
        Ok(()) => format!("copied {} chars", text.chars().count()),
        Err(error) => format!("copy failed: {error}"),
    });
}

/// Rows the mouse wheel scrolls the log by.
const WHEEL_STEP: u16 = 3;

/// Apply a mouse event to the state: a left drag selects inside the pane it started in, and
/// releasing it returns the selected text of `screen` (the last frame) to copy. The wheel scrolls
/// the log.
fn on_mouse(app: &mut App, mouse: MouseEvent, screen: &Buffer) -> Option<String> {
    let at = Position::new(mouse.column, mouse.row);
    let areas = view::areas(app, screen.area);
    match mouse.kind {
        MouseEventKind::ScrollUp => app.scroll = app.scroll.saturating_add(WHEEL_STEP),
        MouseEventKind::ScrollDown => app.scroll = app.scroll.saturating_sub(WHEEL_STEP),
        MouseEventKind::Down(MouseButton::Left) => {
            app.selection = areas.pane_at(at).map(|pane| select::Selection {
                pane,
                anchor: at,
                cursor: at,
            });
        }
        MouseEventKind::Drag(MouseButton::Left) => {
            if let Some(selection) = &mut app.selection {
                selection.cursor = select::clamp(at, areas.inner(selection.pane));
            }
        }
        MouseEventKind::Up(MouseButton::Left) => {
            let selection = app.selection.filter(|s| !s.is_empty());
            let Some(selection) = selection else {
                app.selection = None;
                return None;
            };
            let inner = areas.inner(selection.pane);
            let continues = match selection.pane {
                select::Pane::Log => view::log_continuations(app, inner),
                select::Pane::Tree => Vec::new(),
            };
            // The log is copied from its text alone, without thumbnails.
            let text = match selection.pane {
                select::Pane::Log => {
                    let log = view::log_text(app, screen.area);
                    select::selected_text(&log, inner, &selection, &continues)
                }
                select::Pane::Tree => select::selected_text(screen, inner, &selection, &continues),
            };
            return (!text.is_empty()).then_some(text);
        }
        _ => {}
    }
    None
}

/// Two Esc presses this close together stop the selected agent's turn.
const DOUBLE_ESC: std::time::Duration = std::time::Duration::from_secs(1);

/// Columns one Alt-←/→ press resizes the agent tree by.
const TREE_STEP: i16 = 4;

/// Apply a key press to the state; returns a command when the input line was submitted.
fn on_key(app: &mut App, key: KeyEvent) -> Option<Command> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    let up_down = |code: KeyCode| if code == KeyCode::Up { -1 } else { 1 };
    if app.status.is_some() || app.cost.is_some() {
        // The status and cost popups are modal: they scroll, any other key closes them.
        match key.code {
            KeyCode::Up => app.status_scroll = app.status_scroll.saturating_sub(1),
            KeyCode::Down => app.status_scroll = app.status_scroll.saturating_add(1),
            KeyCode::PageUp => app.status_scroll = app.status_scroll.saturating_sub(10),
            KeyCode::PageDown => app.status_scroll = app.status_scroll.saturating_add(10),
            KeyCode::Char('c') if ctrl => app.quit = true,
            _ => (app.status, app.cost) = (None, None),
        }
        return None;
    }
    match key.code {
        KeyCode::Char('c') if ctrl => app.quit = true,
        KeyCode::Char('a') if ctrl => app.home(),
        KeyCode::Char('e') if ctrl => app.end(),
        KeyCode::Char('w') if ctrl => app.delete_word(),
        KeyCode::Char('u') if ctrl => app.delete_to_line_start(),
        KeyCode::Char('p') if ctrl => app.browse_history(-1),
        KeyCode::Char('n') if ctrl => app.browse_history(1),
        KeyCode::Char(_) if ctrl => {}
        KeyCode::Char(c) => app.insert(c),
        KeyCode::Backspace if alt => app.delete_word(),
        KeyCode::Backspace => app.backspace(),
        KeyCode::Delete => app.delete(),
        KeyCode::Tab => drop(app.complete()),
        KeyCode::Up | KeyCode::Down if !app.completions().is_empty() => {
            app.move_completion(up_down(key.code))
        }
        KeyCode::Up | KeyCode::Down if app.wants_history() => app.browse_history(up_down(key.code)),
        KeyCode::Up | KeyCode::Down => app.select_by(up_down(key.code)),
        KeyCode::PageUp => app.scroll = app.scroll.saturating_add(10),
        KeyCode::PageDown => app.scroll = app.scroll.saturating_sub(10),
        KeyCode::Home if !app.input.is_empty() => app.home(),
        KeyCode::End if !app.input.is_empty() => app.end(),
        KeyCode::End => app.scroll = 0,
        KeyCode::Esc if app.show_keys => app.show_keys = false,
        KeyCode::Esc if app.input.is_empty() && app.selection.is_none() => {
            let now = std::time::Instant::now();
            match app.last_esc.take() {
                Some(at) if now.duration_since(at) <= DOUBLE_ESC => return Some(Command::Stop),
                _ => {
                    app.last_esc = Some(now);
                    app.notice = Some("press esc again to stop the selected agent's turn".to_owned());
                }
            }
        }
        KeyCode::Esc => {
            app.clear_input();
            app.selection = None;
        }
        KeyCode::F(1) => app.show_keys = !app.show_keys,
        KeyCode::F(2) => app.show_plain = true,
        KeyCode::F(3) => app.toggle_cancelled(),
        KeyCode::Left | KeyCode::Right if alt => app.resize_tree(if key.code == KeyCode::Left {
            -TREE_STEP
        } else {
            TREE_STEP
        }),
        KeyCode::Left | KeyCode::Right if ctrl => {
            app.move_word(if key.code == KeyCode::Left { -1 } else { 1 })
        }
        KeyCode::Left => app.move_char(-1),
        KeyCode::Right => app.move_char(1),
        // Shift-Enter (where the terminal reports it) or Alt-Enter starts a new line.
        KeyCode::Enter if shift || alt => app.insert('\n'),
        // Enter on a partly typed command picks it from the popup instead of sending it.
        KeyCode::Enter if !app.input_is_command() && app.complete() => {}
        KeyCode::Enter => {
            let line = app.take_input();
            return parse_command(&line, app.selected_id().is_some());
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
        (Command::To { text, .. }, _) if text.is_empty() => {
            Err(strands_agents::Error::other("usage: /to <agent> <message>"))
        }
        (Command::To { who, text }, _) => swarm.resolve(&who).and_then(|id| {
            swarm
                .send(USER, &id, text)
                .map(|()| format!("sent to {}", swarm.label(&id)))
        }),
        (Command::Message(text), Some(id)) => swarm
            .send(USER, &id, text)
            .map(|()| format!("sent to {}", swarm.label(&id))),
        (Command::Stop, Some(id)) => swarm
            .interrupt(&id)
            .map(|()| format!("stopping {}'s turn", swarm.label(&id))),
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
        (Command::MaxAgents(None), _) => Ok(max_agents_notice(swarm)),
        (Command::MaxAgents(Some(limit)), _) => match limit.as_str() {
            "default" => Ok(None),
            n => n
                .parse::<usize>()
                .map(Some)
                .map_err(|_| strands_agents::Error::other(format!("Not a number: {n}"))),
        }
        .and_then(|limit| swarm.set_max_agents(limit))
        .map(|()| max_agents_notice(swarm)),
        (Command::Budget(None), _) => Ok(budget_notice(swarm)),
        (Command::Budget(Some(tokens)), _) => match tokens.as_str() {
            "off" | "none" => Ok(None),
            tokens => crate::budget::parse_tokens(tokens).map(Some),
        }
        .map(|tokens| {
            swarm.set_budget(tokens);
            budget_notice(swarm)
        })
        .map_err(strands_agents::Error::other),
        (Command::Remember(Some(text)), _) => swarm
            .remember(&text)
            .await
            .map(|line| format!("remembered {line}")),
        (Command::Remember(None), _) => Ok(memory_notice(swarm).await),
        (Command::Forget(text), _) => swarm
            .forget(&text)
            .await
            .map(|removed| format!("forgot {} line(s): {}", removed.len(), removed.join(" | "))),
        (Command::Schedules, _) => Ok(schedules_notice(swarm)),
        (Command::Cost, _) => {
            app.cost = Some(swarm.cost());
            app.status_scroll = 0;
            Ok("cost (estimates from list prices): ↑↓ scroll, any other key closes".to_owned())
        }
        (Command::Status, _) => {
            app.status = Some(swarm.status());
            app.status_scroll = 0;
            Ok("status: ↑↓ scroll, any other key closes".to_owned())
        }
        (Command::Gc, _) => {
            swarm.gc_in_background();
            Ok("gc: measuring and reclaiming in the background…".to_owned())
        }
        (Command::Image { path, text }, Some(id)) => {
            let path = std::path::absolute(&path).unwrap_or(path);
            swarm
                .send_with_images(USER, &id, text, vec![path.clone()])
                .map(|()| format!("sent {} to {}", path.display(), swarm.label(&id)))
        }
        (_, None) => Ok("no agent selected".to_owned()),
    };
    app.notice = Some(outcome.unwrap_or_else(|error| format!("error: {error}")));
}

/// Where the memory file is and how many entries it holds.
async fn memory_notice(swarm: &Swarm) -> String {
    let Some(memory) = swarm.memory() else {
        return "memory is off (--no-memory)".to_owned();
    };
    match memory.read().await {
        Ok(text) => format!(
            "memory: {} entries in {}",
            text.lines().filter(|line| !line.trim().is_empty()).count(),
            memory.path().display()
        ),
        Err(error) => format!("error: {error}"),
    }
}

/// Every pending schedule on one line, soonest first.
fn schedules_notice(swarm: &Swarm) -> String {
    let schedules = swarm.schedules();
    if schedules.is_empty() {
        return "no schedules".to_owned();
    }
    let all: Vec<String> = schedules
        .iter()
        .map(|s| crate::tools::describe_schedule(swarm, s))
        .collect();
    format!("schedules: {}", all.join(" | "))
}

fn max_agents_notice(swarm: &Swarm) -> String {
    format!("up to {} agents at once", swarm.max_agents())
}

fn budget_notice(swarm: &Swarm) -> String {
    let used = crate::budget::compact(swarm.tokens_used());
    match swarm.budget() {
        Some(budget) => format!("{used} of {} tokens used", crate::budget::compact(budget)),
        None => format!("{used} tokens used, no budget"),
    }
}

/// End-to-end: drive [`App`] with key events and swarm events, render into a [`TestBackend`]
/// and assert on what is on screen.
#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ratatui::backend::TestBackend;
    use ratatui::style::Modifier;
    use ratatui::Terminal;
    use strands_agents::types::{ContentBlock, Message, Role, ToolUseBlock};

    use super::*;
    use crate::runtime::{AgentStatus, Mail};

    const WIDTH: u16 = 100;
    const HEIGHT: u16 = 24;

    fn render(app: &App) -> Buffer {
        let mut terminal = Terminal::new(TestBackend::new(WIDTH, HEIGHT)).unwrap();
        terminal.draw(|frame| view::draw(frame, app, "run1", 0)).unwrap();
        terminal.backend().buffer().clone()
    }

    fn mouse(app: &mut App, kind: MouseEventKind, x: u16, y: u16) -> Option<String> {
        let screen = render(app);
        let event = MouseEvent {
            kind,
            column: x,
            row: y,
            modifiers: KeyModifiers::NONE,
        };
        on_mouse(app, event, &screen)
    }

    fn drag(app: &mut App, from: (u16, u16), to: (u16, u16)) -> Option<String> {
        let left = MouseButton::Left;
        assert_eq!(mouse(app, MouseEventKind::Down(left), from.0, from.1), None);
        assert_eq!(mouse(app, MouseEventKind::Drag(left), to.0, to.1), None);
        mouse(app, MouseEventKind::Up(left), to.0, to.1)
    }

    fn row_of(app: &App, needle: &str) -> u16 {
        let shown = screen(app);
        let row = shown.lines().position(|l| l.contains(needle));
        u16::try_from(row.unwrap_or_else(|| panic!("{needle:?} not on screen\n{shown}"))).unwrap()
    }

    fn log_pane(app: &App) -> Rect {
        view::areas(app, Rect::new(0, 0, WIDTH, HEIGHT)).inner(select::Pane::Log)
    }

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
    fn esc_twice_on_an_empty_input_stops_the_turn() {
        let mut app = App {
            input: "draft".into(),
            ..Default::default()
        };
        assert_eq!(press(&mut app, KeyCode::Esc), None);
        assert!(app.input.is_empty(), "the first esc clears the input");
        assert_eq!(press(&mut app, KeyCode::Esc), None);
        assert!(app.notice.as_deref().unwrap().contains("esc again"));
        assert_eq!(press(&mut app, KeyCode::Esc), Some(Command::Stop));
        // A slow second press only arms again.
        app.last_esc = Some(std::time::Instant::now() - 2 * DOUBLE_ESC);
        assert_eq!(press(&mut app, KeyCode::Esc), None);
        assert!(app.last_esc.is_some());
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
                images: Vec::new(),
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

    fn tree_column(shown: &str) -> Option<usize> {
        let row = shown.lines().nth(2)?;
        row.chars().position(|c| c == '│')
    }

    #[test]
    fn the_agent_tree_fits_its_labels_and_resizes_with_alt_arrows() {
        let mut app = App::default();
        spawn(&mut app, "a1", USER);
        spawn(&mut app, "a2", "a1");
        let fitted = tree_column(&screen(&app)).unwrap();
        assert!(fitted < usize::from(WIDTH) / 4, "fits short labels: {fitted}");

        let alt = |code| KeyEvent::new(code, KeyModifiers::ALT);
        on_key(&mut app, alt(KeyCode::Right));
        assert_eq!(tree_column(&screen(&app)), Some(fitted + 4));
        for _ in 0..5 {
            on_key(&mut app, alt(KeyCode::Left));
        }
        let hidden = screen(&app);
        assert_eq!(tree_column(&hidden), None, "{hidden}");
        assert!(!hidden.contains(" agents ─"), "{hidden}");
        assert!(
            hidden.contains("name-a1 (a1) · idle"),
            "the log still names the agent\n{hidden}"
        );
        for _ in 0..100 {
            on_key(&mut app, KeyEvent::new(KeyCode::Right, KeyModifiers::ALT));
        }
        let widest = tree_column(&screen(&app)).unwrap();
        assert!(widest <= usize::from(WIDTH) * 6 / 10, "capped: {widest}");
    }

    #[test]
    fn the_input_box_edits_like_a_line_editor_and_recalls_sent_lines() {
        let mut app = App::default();
        spawn(&mut app, "a1", USER);
        let key = |code, modifiers| KeyEvent::new(code, modifiers);
        let none = KeyModifiers::NONE;
        for c in "fix bug".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        on_key(&mut app, key(KeyCode::Left, KeyModifiers::CONTROL));
        for c in "the ".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        on_key(&mut app, key(KeyCode::Enter, KeyModifiers::ALT));
        press(&mut app, KeyCode::Char('!'));
        assert_eq!(app.input, "fix the \n!bug");
        let shown = screen(&app);
        assert!(shown.contains("fix the"), "multi-line input is drawn\n{shown}");
        on_key(&mut app, key(KeyCode::Char('a'), KeyModifiers::CONTROL));
        press(&mut app, KeyCode::Delete);
        on_key(&mut app, key(KeyCode::Char('e'), KeyModifiers::CONTROL));
        assert_eq!(app.input, "fix the \nbug");
        assert_eq!(
            on_key(&mut app, key(KeyCode::Enter, none)),
            Some(Command::Message("fix the \nbug".into()))
        );

        // An empty input keeps ↑/↓ for selecting agents; with text, they walk the history.
        press(&mut app, KeyCode::Char('x'));
        press(&mut app, KeyCode::Up);
        assert_eq!(app.input, "fix the \nbug");
        press(&mut app, KeyCode::Down);
        assert_eq!(app.input, "x");
        on_key(&mut app, key(KeyCode::Char('u'), KeyModifiers::CONTROL));
        assert!(app.input.is_empty());
        on_key(&mut app, key(KeyCode::Char('p'), KeyModifiers::CONTROL));
        assert_eq!(app.input, "fix the \nbug", "ctrl-p recalls from an empty input");
    }

    #[test]
    fn cancelled_agents_are_hidden_until_f3() {
        let mut app = App::default();
        spawn(&mut app, "a1", USER);
        spawn(&mut app, "a2", USER);
        app.apply(SwarmEvent::Status {
            id: "a2".into(),
            status: AgentStatus::Cancelled,
        });
        let shown = screen(&app);
        assert!(!shown.contains("name-a2"), "{shown}");
        assert!(shown.contains("1 cancelled hidden (f3)"), "{shown}");

        press(&mut app, KeyCode::F(3));
        let shown = screen(&app);
        assert!(shown.contains("⊘ name-a2 a2"), "{shown}");
        assert!(!shown.contains("cancelled hidden"), "{shown}");
    }

    #[test]
    fn a_drag_in_the_log_copies_only_log_text_and_is_highlighted() {
        let mut app = App::default();
        spawn(&mut app, "a1", USER);
        app.apply(SwarmEvent::Text {
            id: "a1".into(),
            delta: "first line\nsecond line".into(),
        });
        let log = log_pane(&app);
        let first = row_of(&app, "first line");
        let copied = drag(&mut app, (log.x, first), (log.x + 10, first + 1));
        assert_eq!(copied.as_deref(), Some("first line\nsecond line"));

        let buffer = render(&app);
        let reversed = |x, y| buffer[(x, y)].modifier.contains(Modifier::REVERSED);
        assert!(reversed(log.x, first) && reversed(log.right() - 1, first));
        assert!(reversed(log.x + 10, first + 1) && !reversed(log.x + 11, first + 1));
        assert!(!reversed(0, first), "the tree is not highlighted");

        // Dragging over the tree stays clamped to the log.
        let copied = drag(&mut app, (log.x, first), (0, first + 1)).unwrap();
        assert_eq!(copied, "first line\ns");
        assert!(!copied.contains("name-a1"), "{copied}");
    }

    #[test]
    fn a_wrapped_line_is_copied_as_one_line() {
        let mut app = App::default();
        spawn(&mut app, "a1", USER);
        let long: String = (0..40).map(|i| format!("word{i} ")).collect();
        let long = long.trim_end().to_owned();
        app.apply(SwarmEvent::Text {
            id: "a1".into(),
            delta: long.clone(),
        });
        let log = log_pane(&app);
        let (start, end) = (row_of(&app, "word0 "), row_of(&app, "word39"));
        assert!(end > start, "the line wraps");
        let copied = drag(&mut app, (log.x, start), (log.right() - 1, end));
        assert_eq!(copied, Some(long));
    }

    #[test]
    fn a_drag_in_the_tree_copies_only_the_tree() {
        let mut app = App::default();
        spawn(&mut app, "a1", USER);
        spawn(&mut app, "a2", "a1");
        let tree = view::areas(&app, Rect::new(0, 0, WIDTH, HEIGHT)).inner(select::Pane::Tree);
        let copied = drag(&mut app, (0, tree.y), (WIDTH - 1, tree.y + 2)).unwrap();
        assert!(copied.starts_with("● name-a1 a1"), "{copied}");
        assert!(copied.contains("● name-a2 a2"), "{copied}");
        assert!(!copied.contains("▶ task") && !copied.contains('│'), "{copied}");
    }

    #[test]
    fn a_click_selects_nothing_and_the_wheel_scrolls_the_log() {
        let mut app = App::default();
        spawn(&mut app, "a1", USER);
        let log = log_pane(&app);
        assert_eq!(drag(&mut app, (log.x, log.y), (log.x, log.y)), None);
        assert_eq!(app.selection, None);

        mouse(&mut app, MouseEventKind::ScrollUp, log.x, log.y);
        mouse(&mut app, MouseEventKind::ScrollUp, log.x, log.y);
        assert_eq!(app.scroll, 2 * WHEEL_STEP);
        mouse(&mut app, MouseEventKind::ScrollDown, log.x, log.y);
        assert_eq!(app.scroll, WHEEL_STEP);
    }

    #[test]
    fn ctrl_c_quits() {
        let mut app = App::default();
        on_key(&mut app, KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(app.quit);
    }
}
