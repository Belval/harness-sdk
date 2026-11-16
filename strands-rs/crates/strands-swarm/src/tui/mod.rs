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

    let outcome = loop {
        app.mark_seen();
        if let Err(error) = terminal.draw(|frame| view::draw(frame, &app, swarm.run_id(), tick)) {
            break Err(strands_agents::Error::other(format!("terminal error: {error}")));
        }
        tokio::select! {
            Some(event) = events.recv() => {
                app.apply(event);
                while let Ok(event) = events.try_recv() {
                    app.apply(event);
                }
            }
            Some(Ok(event)) = keys.next() => on_terminal_event(&swarm, &mut app, event).await,
            _ = ticker.tick() => tick = tick.wrapping_add(1),
            _ = swarm.reload_requested() => break Ok(Exit::Reload),
        }
        if app.quit {
            break Ok(Exit::Quit);
        }
    };

    ratatui::restore();
    swarm.shutdown().await;
    outcome
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
            swarm.spawn(USER, request).await.map(|id| format!("started {id}"))
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
                .map(|id| format!("started {id} with a goal"))
        }
        (Command::Message(text), Some(id)) => swarm.send(USER, &id, text).map(|()| format!("sent to {id}")),
        (Command::Cancel, Some(id)) => swarm.cancel(&id).map(|()| format!("cancelled {id}")),
        (Command::ShowModel, Some(id)) => Ok(match swarm.agent(&id).and_then(|info| info.model) {
            Some(model) => format!("{id} uses {model}"),
            None => format!("{id} uses the default model"),
        }),
        (Command::SetModel(model), Some(id)) => swarm
            .set_model(&id, model)
            .await
            .map(|model| format!("{id} now uses {model}")),
        (_, None) => Ok("no agent selected".to_owned()),
    };
    app.notice = Some(outcome.unwrap_or_else(|error| format!("error: {error}")));
}
