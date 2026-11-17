//! End-to-end: a real [`Swarm`] on scripted mock models, driven through the TUI's own key
//! handling ([`on_terminal_event`]) with its events fed back into [`App`], asserting on the
//! rendered [`TestBackend`] buffer. The unit tests in `mod.rs` feed [`App`] hand-made events;
//! this checks that what the runtime really emits shows up on screen.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use serde_json::json;
use strands_agents::testing::{MockMessageModel, MockResponse};
use strands_agents::types::{ContentBlock, Message, Role, ToolUseBlock};
use strands_agents::Agent;

use super::*;
use crate::compaction::{CompactedSink, Compaction};
use crate::runtime::{AgentFactory, AgentSpec, SwarmConfig};

const WIDTH: u16 = 120;
const HEIGHT: u16 = 30;

/// Each agent's compaction sink, as the factory was handed it.
type Sinks = Arc<Mutex<HashMap<String, CompactedSink>>>;

/// Agents on scripted mock models; each saves its conversation when the swarm has a state dir,
/// as the harness factory does.
fn factory(scripts: HashMap<&'static str, Vec<MockResponse>>, sinks: &Sinks) -> AgentFactory {
    let models: HashMap<String, Arc<MockMessageModel>> = scripts
        .into_iter()
        .map(|(id, responses)| (id.to_owned(), MockMessageModel::shared(responses)))
        .collect();
    let sinks = sinks.clone();
    Arc::new(move |spec: AgentSpec| {
        let model = models[&spec.id].clone();
        if let Some(sink) = spec.on_compacted.clone() {
            sinks.lock().insert(spec.id.clone(), sink);
        }
        Box::pin(async move {
            let mut builder = Agent::builder()
                .id(&spec.id)
                .model(model)
                .system_prompt(spec.instructions)
                .tools(spec.tools)
                .no_sandbox();
            if let Some(dir) = &spec.session_dir {
                builder = builder.session_manager(crate::agent_session_manager(dir, &spec.id)?);
            }
            builder.build()
        })
    })
}

fn tool_call(name: &str, id: &str, input: serde_json::Value) -> MockResponse {
    MockResponse::message(Message::new(
        Role::Assistant,
        vec![ContentBlock::ToolUse(ToolUseBlock::new(name, id, input))],
    ))
}

fn say(text: &str) -> MockResponse {
    MockResponse::message(Message::assistant(text))
}

/// The TUI as the user sees it: draws `app` and keeps the frame, like the event loop does.
struct Screen {
    terminal: Terminal<TestBackend>,
    buffer: Buffer,
}

impl Screen {
    fn new() -> Self {
        Self {
            terminal: Terminal::new(TestBackend::new(WIDTH, HEIGHT)).unwrap(),
            buffer: Buffer::empty(Rect::default()),
        }
    }

    fn draw(&mut self, app: &mut App) -> String {
        app.mark_seen();
        let frame = self
            .terminal
            .draw(|frame| view::draw(frame, app, "run1", 0))
            .unwrap();
        self.buffer.clone_from(frame.buffer);
        (0..HEIGHT)
            .map(|y| {
                (0..WIDTH)
                    .map(|x| self.buffer[(x, y)].symbol())
                    .collect::<String>()
                    + "\n"
            })
            .collect()
    }

    /// Press `code`, through the TUI's terminal-event handler.
    async fn press(&mut self, swarm: &Swarm, app: &mut App, code: KeyCode) -> String {
        let key = Event::Key(KeyEvent::new(code, KeyModifiers::NONE));
        on_terminal_event(swarm, app, key, &self.buffer).await;
        self.draw(app)
    }

    /// Type `line` and press enter, each key going through the TUI's terminal-event handler.
    async fn type_line(&mut self, swarm: &Swarm, app: &mut App, line: &str) {
        let keys = line.chars().map(KeyCode::Char).chain([KeyCode::Enter]);
        for code in keys {
            let key = Event::Key(KeyEvent::new(code, KeyModifiers::NONE));
            on_terminal_event(swarm, app, key, &self.buffer).await;
            self.draw(app);
        }
    }

    /// Feed swarm events into `app`, as the event loop does, until the screen shows `needle`.
    async fn until(
        &mut self,
        app: &mut App,
        events: &mut UnboundedReceiver<SwarmEvent>,
        needle: &str,
    ) -> String {
        let wait = async {
            loop {
                let shown = self.draw(app);
                if shown.contains(needle) {
                    return shown;
                }
                let event = events.recv().await.expect("event stream closed");
                app.apply(event);
            }
        };
        match tokio::time::timeout(Duration::from_secs(10), wait).await {
            Ok(shown) => shown,
            Err(_) => panic!("{needle:?} never showed up\n{}", self.draw(app)),
        }
    }
}

#[tokio::test]
async fn a_real_swarm_shows_spawns_replies_compaction_and_status_on_screen() {
    let dir = tempfile::tempdir().unwrap();
    let sinks = Sinks::default();
    let scripts = HashMap::from([
        (
            "a1",
            vec![
                tool_call(
                    "spawn_agent",
                    "t1",
                    json!({"task": "check the parser", "name": "parser"}),
                ),
                tool_call("wait", "t2", json!({"from": ["parser"], "timeout_seconds": 10})),
                say("root: the parser is fine"),
                say("root: second answer"),
            ],
        ),
        ("a2", vec![say("parser: all green")]),
    ]);
    let (swarm, mut events) = Swarm::new(SwarmConfig::new(factory(scripts, &sinks), dir.path()));
    let mut app = App::default();
    let mut screen = Screen::new();

    // An empty swarm takes the first line as the root task.
    screen.type_line(&swarm, &mut app, "fix the bug").await;
    assert!(
        app.notice.as_deref().is_some_and(|n| n.starts_with("started ")),
        "{:?}",
        app.notice
    );

    // The root spawns a child, waits for it and replies to the user.
    let shown = screen
        .until(&mut app, &mut events, "root: the parser is fine")
        .await;
    assert!(
        shown.contains(" fix-bug test-model a1"),
        "the root is in the tree\n{shown}"
    );
    assert!(
        shown.contains("  ● parser test-model a2"),
        "its child under it\n{shown}"
    );
    assert!(shown.contains("fix the bug"), "the task heads the log\n{shown}");

    // A compaction reported by the agent's context pipeline is noted in its log; a failed one
    // also in the footer.
    let sink = sinks.lock()["a1"].clone();
    sink(Compaction::Done {
        before: 305_000,
        after: 98_100,
    });
    screen
        .until(
            &mut app,
            &mut events,
            "context compacted: older turns summarized, 305.0k → 98.1k tokens",
        )
        .await;
    sink(Compaction::Failed {
        tokens: 290_000,
        error: "throttled".into(),
    });
    let shown = screen
        .until(&mut app, &mut events, "a1: context not compacted")
        .await;
    assert!(
        shown.contains("summarizing older turns failed (throttled)"),
        "{shown}"
    );

    // A plain line goes to the selected agent (the root) and its answer shows up.
    screen.type_line(&swarm, &mut app, "anything else?").await;
    assert_eq!(
        app.notice.as_deref().map(|n| n.starts_with("sent to ")),
        Some(true)
    );
    screen.until(&mut app, &mut events, "root: second answer").await;

    // `/model` names the selected agent's model; `/status` opens the overlay over every agent.
    screen.type_line(&swarm, &mut app, "/model").await;
    assert_eq!(app.notice.as_deref(), Some("fix-bug (a1) uses test-model"));
    screen.until(&mut app, &mut events, "2 agents · 0 running").await;
    screen.type_line(&swarm, &mut app, "/status").await;
    let shown = screen.draw(&mut app);
    for expected in [
        "┌ status",
        "2 live agents",
        "fix-bug (a1) · idle",
        "parser (a2) · idle",
    ] {
        assert!(shown.contains(expected), "missing {expected:?}\n{shown}");
    }
    let key = Event::Key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
    on_terminal_event(&swarm, &mut app, key, &screen.buffer).await;
    assert!(
        !screen.draw(&mut app).contains("┌ status"),
        "any key closes the overlay"
    );
    swarm.shutdown().await;
}

#[tokio::test]
async fn after_a_restart_every_agent_shows_its_past_including_cancelled_ones() {
    let dir = tempfile::tempdir().unwrap();
    let state_dir = dir.path().join("run");
    let sinks = Sinks::default();
    let scripts = HashMap::from([
        (
            "a1",
            vec![
                tool_call(
                    "spawn_agent",
                    "t1",
                    json!({"task": "check the parser", "name": "parser"}),
                ),
                tool_call("wait", "t2", json!({"from": ["parser"], "timeout_seconds": 10})),
                say("root: the parser is fine"),
            ],
        ),
        ("a2", vec![say("parser: all green")]),
    ]);
    let mut config = SwarmConfig::new(factory(scripts, &sinks), dir.path());
    config.state_dir = Some(state_dir.clone());
    let (swarm, mut events) = Swarm::new(config.clone());
    let mut app = App::default();
    let mut screen = Screen::new();
    screen.type_line(&swarm, &mut app, "fix the bug").await;
    screen
        .until(&mut app, &mut events, "root: the parser is fine")
        .await;
    swarm.cancel("a2").unwrap();
    screen.until(&mut app, &mut events, "1 cancelled hidden").await;
    swarm.shutdown().await;

    // A new process: a fresh screen, the same run.
    let saved = crate::state::load(&state_dir)
        .await
        .unwrap()
        .expect("state saved");
    let resumed_factory = factory(HashMap::from([("a1", vec![]), ("a2", vec![])]), &sinks);
    let config = SwarmConfig {
        factory: resumed_factory,
        ..config
    };
    let (swarm, mut events) = Swarm::resume(config, saved).await.unwrap();
    let mut app = App::default();
    let mut screen = Screen::new();
    let shown = screen
        .until(&mut app, &mut events, "root: the parser is fine")
        .await;
    assert!(shown.contains("restored conversation"), "{shown}");
    assert!(
        shown.contains("spawn_agent"),
        "past tool calls are listed\n{shown}"
    );

    // The cancelled child has no actor, yet its log shows what it did once it is listed.
    screen.press(&swarm, &mut app, KeyCode::F(3)).await;
    screen.press(&swarm, &mut app, KeyCode::Down).await;
    assert_eq!(app.selected_id().as_deref(), Some("a2"));
    let shown = screen.until(&mut app, &mut events, "parser: all green").await;
    assert!(shown.contains("restored conversation"), "{shown}");
    swarm.shutdown().await;
}
