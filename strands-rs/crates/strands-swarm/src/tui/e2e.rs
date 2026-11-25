//! End-to-end: a real [`Swarm`] on scripted mock models, driven through the TUI's own key
//! handling ([`on_terminal_event`]) with its events fed back into [`App`], asserting on the
//! rendered [`TestBackend`] buffer. The unit tests in `mod.rs` feed [`App`] hand-made events;
//! this checks that what the runtime really emits shows up on screen.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use parking_lot::Mutex;
use ratatui::backend::TestBackend;
use ratatui::style::Color;
use ratatui::Terminal;
use serde_json::json;
use strands_agents::models::{Model, ModelStreamEvent, StreamOptions};
use strands_agents::testing::{MockMessageModel, MockResponse};
use strands_agents::types::{ContentBlock, Message, Role, ToolUseBlock};
use strands_agents::Agent;
use tokio::sync::Semaphore;

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
    gated_factory(scripts, &[], &Arc::new(Semaphore::new(0)), sinks)
}

/// [`factory`], where each model call of the `gated` agents first waits for a permit of `gate`.
fn gated_factory(
    scripts: HashMap<&'static str, Vec<MockResponse>>,
    gated: &[&str],
    gate: &Arc<Semaphore>,
    sinks: &Sinks,
) -> AgentFactory {
    let models: HashMap<String, Arc<Scripted>> = scripts
        .into_iter()
        .map(|(id, responses)| {
            let model = Scripted {
                inner: MockMessageModel::new(responses),
                gate: gated.contains(&id).then(|| gate.clone()),
            };
            (id.to_owned(), Arc::new(model))
        })
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

/// A scripted model whose calls each wait for a permit of `gate`, when it has one.
struct Scripted {
    inner: MockMessageModel,
    gate: Option<Arc<Semaphore>>,
}

impl Model for Scripted {
    fn model_id(&self) -> Option<String> {
        self.inner.model_id()
    }

    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        options: &'a StreamOptions,
    ) -> BoxStream<'a, strands_agents::Result<ModelStreamEvent>> {
        let Some(gate) = &self.gate else {
            return self.inner.stream(messages, options);
        };
        let opened = futures::stream::once(async move { gate.acquire().await.expect("open").forget() });
        Box::pin(
            opened
                .filter_map(|()| async { None })
                .chain(self.inner.stream(messages, options)),
        )
    }
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

/// The root (`a1`) spawns `parser` (`a2`), waits for its reply and then says each of `answers`.
fn root_spawns_parser(answers: &[&str]) -> HashMap<&'static str, Vec<MockResponse>> {
    let spawn = json!({"task": "check the parser", "name": "parser"});
    let wait = json!({"from": ["parser"], "timeout_seconds": 10});
    let root = [
        tool_call("spawn_agent", "t1", spawn),
        tool_call("wait", "t2", wait),
    ];
    HashMap::from([
        (
            "a1",
            root.into_iter()
                .chain(answers.iter().map(|text| say(text)))
                .collect(),
        ),
        ("a2", vec![say("parser: all green")]),
    ])
}

/// The TUI as the user sees it: draws `app` and keeps the frame, like the event loop does.
struct Screen {
    terminal: Terminal<TestBackend>,
    buffer: Buffer,
}

impl Screen {
    fn new() -> Self {
        Self::sized(WIDTH, HEIGHT)
    }

    fn sized(width: u16, height: u16) -> Self {
        Self {
            terminal: Terminal::new(TestBackend::new(width, height)).unwrap(),
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
        view::buffer_text(&self.buffer)
    }

    /// Press `code`, through the TUI's terminal-event handler.
    async fn press(&mut self, swarm: &Swarm, app: &mut App, code: KeyCode) -> String {
        let key = Event::Key(KeyEvent::new(code, KeyModifiers::NONE));
        on_terminal_event(swarm, app, key, &self.buffer).await;
        self.draw(app)
    }

    /// Type `line` and press enter.
    async fn type_line(&mut self, swarm: &Swarm, app: &mut App, line: &str) {
        for code in line.chars().map(KeyCode::Char).chain([KeyCode::Enter]) {
            self.press(swarm, app, code).await;
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
    let scripts = root_spawns_parser(&["root: the parser is fine", "root: second answer"]);
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
    // The whole report fits, so scrolling down does not move it off screen.
    let shown = screen.press(&swarm, &mut app, KeyCode::PageDown).await;
    assert_eq!(app.status_scroll, 0);
    assert!(shown.contains("2 live agents"), "{shown}");
    let shown = screen.press(&swarm, &mut app, KeyCode::Char('x')).await;
    assert!(!shown.contains("┌ status"), "any key closes the overlay");
    assert!(!shown.contains("any other key closes"), "and its hint\n{shown}");
    swarm.shutdown().await;
}

#[tokio::test]
async fn after_a_restart_every_agent_shows_its_past_including_cancelled_ones() {
    let dir = tempfile::tempdir().unwrap();
    let state_dir = dir.path().join("run");
    let sinks = Sinks::default();
    let scripts = root_spawns_parser(&["root: the parser is fine"]);
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

#[tokio::test]
async fn the_tree_tells_working_agents_from_waiting_ones() {
    let dir = tempfile::tempdir().unwrap();
    // The root spawns two children and waits for them; one call runs at a time, so the child
    // whose call holds the slot works while the other waits for it.
    let root = Message::new(
        Role::Assistant,
        vec![
            ContentBlock::ToolUse(ToolUseBlock::new(
                "spawn_agent",
                "t1",
                json!({"task": "check the parser", "name": "parser"}),
            )),
            ContentBlock::ToolUse(ToolUseBlock::new(
                "spawn_agent",
                "t2",
                json!({"task": "run the linter", "name": "linter"}),
            )),
            ContentBlock::ToolUse(ToolUseBlock::new(
                "wait",
                "t3",
                json!({"from": ["parser", "linter"], "all": true, "timeout_seconds": 30}),
            )),
        ],
    );
    let scripts = HashMap::from([
        ("a1", vec![MockResponse::message(root), say("root: done")]),
        ("a2", vec![say("parser: all green")]),
        ("a3", vec![say("linter: all green")]),
    ]);
    let gate = Arc::new(Semaphore::new(0));
    let factory = gated_factory(scripts, &["a2", "a3"], &gate, &Sinks::default());
    let mut config = SwarmConfig::new(factory, dir.path());
    config.max_concurrent_calls = 1;
    let (swarm, mut events) = Swarm::new(config);
    let mut app = App::default();
    let mut screen = Screen::new();
    screen.type_line(&swarm, &mut app, "fix the bug").await;

    let shown = screen
        .until(&mut app, &mut events, "3 agents · 1 running · 2 waiting")
        .await;
    assert!(shown.contains("⧗ fix-bug"), "the root waits for mail\n{shown}");
    let (working, queued) = match shown.contains("⠋ parser") {
        true => ("parser", "linter"),
        false => ("linter", "parser"),
    };
    assert!(shown.contains(&format!("⠋ {working}")), "{shown}");
    assert!(shown.contains(&format!("◌ {queued}")), "{shown}");
    for glyph in ["⧗", "⠋", "◌"] {
        let cell = screen.buffer.content().iter().find(|cell| cell.symbol() == glyph);
        assert_eq!(cell.map(|cell| cell.fg), Some(Color::Cyan), "{glyph}\n{shown}");
    }
    gate.add_permits(2);
    screen.until(&mut app, &mut events, "root: done").await;
    swarm.shutdown().await;
}

#[tokio::test]
async fn max_calls_shows_and_changes_the_model_call_cap_and_status_shows_both_caps() {
    let dir = tempfile::tempdir().unwrap();
    let spawn = |name: &str, id: &str| {
        ContentBlock::ToolUse(ToolUseBlock::new(
            "spawn_agent",
            id,
            json!({"task": format!("run the {name}"), "name": name}),
        ))
    };
    let wait = ContentBlock::ToolUse(ToolUseBlock::new(
        "wait",
        "t3",
        json!({"from": ["parser", "linter"], "all": true, "timeout_seconds": 30}),
    ));
    let root = Message::new(
        Role::Assistant,
        vec![spawn("parser", "t1"), spawn("linter", "t2"), wait],
    );
    let scripts = HashMap::from([
        ("a1", vec![MockResponse::message(root), say("root: done")]),
        ("a2", vec![say("parser: all green")]),
        ("a3", vec![say("linter: all green")]),
    ]);
    let gate = Arc::new(Semaphore::new(0));
    let factory = gated_factory(scripts, &["a2", "a3"], &gate, &Sinks::default());
    let mut config = SwarmConfig::new(factory, dir.path());
    config.max_concurrent_calls = 1;
    let (swarm, mut events) = Swarm::new(config);
    let mut app = App::default();
    let mut screen = Screen::new();
    screen.type_line(&swarm, &mut app, "fix the bug").await;
    screen
        .until(&mut app, &mut events, "3 agents · 1 running · 2 waiting")
        .await;

    // The cap is shown, and `/status` says how full both caps are and who waits for a slot.
    screen.type_line(&swarm, &mut app, "/max-calls").await;
    assert_eq!(
        app.notice.as_deref(),
        Some("up to 1 model calls at once (1 in flight)")
    );
    screen.type_line(&swarm, &mut app, "/status").await;
    let shown = screen.draw(&mut app);
    for expected in [
        "2 of 16 agent slots in use, 1 waiting · calls 1/1",
        "queued for a model slot (1/1 calls in flight; /max-calls to raise)",
    ] {
        assert!(shown.contains(expected), "missing {expected:?}\n{shown}");
    }
    screen.press(&swarm, &mut app, KeyCode::Char('x')).await;

    // Raising it lets the waiting call through; a bad number changes nothing.
    screen.type_line(&swarm, &mut app, "/max-calls two").await;
    assert_eq!(app.notice.as_deref(), Some("error: Not a number: two"));
    screen.type_line(&swarm, &mut app, "/max-calls 2").await;
    assert_eq!(
        app.notice.as_deref(),
        Some("up to 2 model calls at once (1 in flight)")
    );
    screen
        .until(&mut app, &mut events, "3 agents · 2 running · 1 waiting")
        .await;
    assert_eq!(swarm.calls_in_flight(), 2);
    // The header counts working agents against the agent cap, the root in `wait` apart;
    // `/max-agents` says so too.
    assert!(shown_header(&screen.draw(&mut app)).contains("2 of 16 agent slots in use, 1 waiting"));
    screen.type_line(&swarm, &mut app, "/max-agents").await;
    assert_eq!(
        app.notice.as_deref(),
        Some("2 of 16 agent slots in use, 1 waiting (working agents; idle and waiting ones hold none)")
    );
    screen.type_line(&swarm, &mut app, "/max-calls 0").await;
    assert_eq!(app.notice.as_deref(), Some("model calls unlimited (2 in flight)"));
    screen.type_line(&swarm, &mut app, "/status").await;
    let shown = screen.draw(&mut app);
    assert!(
        shown.contains("2 of 16 agent slots in use, 1 waiting · calls 2/unlimited"),
        "{shown}"
    );
    screen.press(&swarm, &mut app, KeyCode::Char('x')).await;
    screen.type_line(&swarm, &mut app, "/max-calls default").await;
    assert_eq!(
        app.notice.as_deref(),
        Some("up to 1 model calls at once (2 in flight)")
    );
    gate.add_permits(2);
    screen.until(&mut app, &mut events, "root: done").await;
    swarm.shutdown().await;
}

#[tokio::test]
async fn a_tool_line_breaks_a_long_path_live_and_restored() {
    let dir = tempfile::tempdir().unwrap();
    let state_dir = dir.path().join("run");
    let sinks = Sinks::default();
    let path =
        "/home/user/workplace/project.swarm/self-improve/a64/strands-rs/crates/strands-swarm/src/tui/view.rs";
    let scripts = HashMap::from([(
        "a1",
        vec![
            tool_call("read", "t1", json!({ "path": path })),
            say("root: read it"),
        ],
    )]);
    let mut config = SwarmConfig::new(factory(scripts, &sinks), dir.path());
    config.state_dir = Some(state_dir.clone());
    let (swarm, mut events) = Swarm::new(config.clone());
    let mut app = App::default();
    let mut screen = Screen::sized(80, 24);
    screen.type_line(&swarm, &mut app, "read the view").await;
    let shown = screen.until(&mut app, &mut events, "root: read it").await;
    assert!(shown.contains("✗ read /home/user/"), "live\n{shown}");
    swarm.shutdown().await;

    let saved = crate::state::load(&state_dir)
        .await
        .unwrap()
        .expect("state saved");
    let config = SwarmConfig {
        factory: factory(HashMap::from([("a1", vec![])]), &sinks),
        ..config
    };
    let (swarm, mut events) = Swarm::resume(config, saved).await.unwrap();
    let mut app = App::default();
    let mut screen = Screen::sized(80, 24);
    let shown = screen.until(&mut app, &mut events, "restored conversation").await;
    let shown = match shown.contains("root: read it") {
        true => shown,
        false => screen.until(&mut app, &mut events, "root: read it").await,
    };
    assert!(shown.contains("✗ read /home/user/"), "restored\n{shown}");
    swarm.shutdown().await;
}

#[tokio::test]
async fn the_cost_popup_wraps_its_prose_at_80_columns() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = HashMap::from([("a1", vec![say("root: done")])]);
    let (swarm, mut events) = Swarm::new(SwarmConfig::new(factory(scripts, &Sinks::default()), dir.path()));
    let mut app = App::default();
    let mut screen = Screen::sized(80, 24);
    screen.type_line(&swarm, &mut app, "fix the bug").await;
    screen.until(&mut app, &mut events, "root: done").await;
    screen.type_line(&swarm, &mut app, "/cost").await;
    let shown = screen.until(&mut app, &mut events, "┌ estimated cost").await;
    // The popup's text, its rows joined: the prose is all there, wrapped rather than clipped.
    let text = shown
        .lines()
        .filter_map(|row| row.split('│').nth(1))
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" ");
    for prose in [
        "(list prices per 1M tokens; cache read 0.1×, write 1.25× input; not a bill)",
        "Unpriced models show as unknown; --prices FILE adds or overrides prices.",
    ] {
        assert!(text.contains(prose), "{prose:?}\n{shown}");
    }
    swarm.shutdown().await;
}

#[tokio::test]
async fn a_long_input_line_wraps_in_a_growing_box() {
    let dir = tempfile::tempdir().unwrap();
    let scripts = HashMap::from([("a1", vec![say("root: done")])]);
    let (swarm, mut events) = Swarm::new(SwarmConfig::new(factory(scripts, &Sinks::default()), dir.path()));
    let mut app = App::default();
    let mut screen = Screen::sized(80, 24);
    screen.type_line(&swarm, &mut app, "fix the bug").await;
    screen.until(&mut app, &mut events, "root: done").await;
    let words: Vec<String> = (1..=30).map(|n| format!("word{n:02}")).collect();
    let mut shown = String::new();
    for code in words.join(" ").chars().map(KeyCode::Char) {
        shown = screen.press(&swarm, &mut app, code).await;
    }
    let rows: Vec<&str> = shown.lines().collect();
    let top = rows
        .iter()
        .position(|row| row.starts_with(" message → "))
        .expect(&shown);
    // 30 words of 7 columns wrap onto three rows at whole words, all in view.
    let input: Vec<&str> = rows[top + 1..rows.len() - 1]
        .iter()
        .map(|row| row.trim_end())
        .collect();
    assert_eq!(input.len(), 3, "{shown}");
    assert!(input[0].starts_with("word01 word02"), "{shown}");
    assert!(
        input
            .iter()
            .all(|row| row.ends_with(|c: char| c.is_ascii_digit())),
        "{shown}"
    );
    assert_eq!(input.join(" "), words.join(" "), "{shown}");
    // The cursor sits after the last word.
    let cursor = screen.terminal.get_cursor_position().unwrap();
    assert_eq!(usize::from(cursor.x), input[2].chars().count(), "{shown}");
    assert_eq!(usize::from(cursor.y), top + 3, "{shown}");
    swarm.shutdown().await;
}

/// Usage just over the budget reads as over it in `/budget`, not as "1.3M of 1.2M".
#[tokio::test]
async fn budget_shows_usage_just_over_it_as_over_budget() {
    let dir = tempfile::tempdir().unwrap();
    let saved: crate::state::SwarmState = serde_json::from_value(json!({
        "run_id": "r",
        "cwd": dir.path(),
        "next_id": 2,
        "agents": [{"id": "a1", "name": null, "parent": "user", "cwd": dir.path(), "branch": null,
                    "task": "t", "model": null, "goal": null, "status": "cancelled", "mailbox": [],
                    "input_tokens": 1_200_000, "output_tokens": 60_000}],
        "budget_tokens": 1_240_000,
    }))
    .unwrap();
    let config = SwarmConfig::new(factory(HashMap::new(), &Sinks::default()), dir.path());
    let (swarm, _events) = Swarm::resume(config, saved).await.unwrap();
    let mut app = App::default();
    let mut screen = Screen::new();
    screen.type_line(&swarm, &mut app, "/budget").await;
    let shown = screen.draw(&mut app);
    assert!(
        shown.contains("1.3M tokens used, over the 1.2M budget"),
        "{shown}"
    );

    // A budget at or below the spend would stop everyone at once: refused, the old one stays.
    screen.type_line(&swarm, &mut app, "/budget 1M").await;
    let notice = app.notice.clone().unwrap();
    assert!(
        notice.starts_with("Not set: the swarm has already used 1.3M") && notice.contains("a 1.0M budget"),
        "{notice}"
    );
    assert_eq!(swarm.budget(), Some(1_240_000));
    screen.type_line(&swarm, &mut app, "/budget 5M").await;
    assert_eq!(app.notice.as_deref(), Some("1.3M of 5.0M tokens used"));
    assert_eq!(swarm.budget(), Some(5_000_000));
    swarm.shutdown().await;
}

#[tokio::test]
async fn priority_shows_and_replaces_the_priority_lane_and_status_marks_it() {
    let dir = tempfile::tempdir().unwrap();
    let factory = factory(root_spawns_parser(&["root: done"]), &Sinks::default());
    let mut config = SwarmConfig::new(factory, dir.path());
    config.priority = vec!["relay".into()];
    let (swarm, mut events) = Swarm::new(config);
    let mut app = App::default();
    let mut screen = Screen::new();
    screen.type_line(&swarm, &mut app, "fix the bug").await;
    screen.until(&mut app, &mut events, "root: done").await;

    // A `--priority` name no agent has yet shows as given; an unknown agent changes nothing.
    screen.type_line(&swarm, &mut app, "/priority").await;
    assert_eq!(app.notice.as_deref(), Some("priority lane: root agents, relay"));
    screen
        .type_line(&swarm, &mut app, "/priority parser nobody")
        .await;
    assert_eq!(
        app.notice.as_deref(),
        Some("error: No agent with id or name `nobody`; call list_agents to see who exists.")
    );
    screen.type_line(&swarm, &mut app, "/priority parser").await;
    assert_eq!(
        app.notice.as_deref(),
        Some("priority lane: root agents, parser (a2)")
    );
    screen.type_line(&swarm, &mut app, "/status").await;
    let shown = screen.draw(&mut app);
    assert!(shown.contains("fix-bug (a1) · idle · priority"), "{shown}");
    assert!(shown.contains("parser (a2) · idle · priority"), "{shown}");
    screen.press(&swarm, &mut app, KeyCode::Char('x')).await;

    // `off` leaves the roots; `default` returns to the flag.
    screen.type_line(&swarm, &mut app, "/priority off").await;
    assert_eq!(app.notice.as_deref(), Some("priority lane: root agents"));
    assert!(!swarm.is_priority("a2"));
    screen.type_line(&swarm, &mut app, "/priority default").await;
    assert_eq!(app.notice.as_deref(), Some("priority lane: root agents, relay"));
    swarm.shutdown().await;
}

/// ← and → with an empty input fold and unfold the selected agent's subtree; with text they
/// move the cursor.
#[tokio::test]
async fn arrows_fold_the_tree_with_an_empty_input_and_edit_it_otherwise() {
    let dir = tempfile::tempdir().unwrap();
    let spawn = |name: &str, id: &str| {
        ContentBlock::ToolUse(ToolUseBlock::new(
            "spawn_agent",
            id,
            json!({"task": format!("run the {name}"), "name": name}),
        ))
    };
    let wait = ContentBlock::ToolUse(ToolUseBlock::new(
        "wait",
        "t3",
        json!({"from": ["parser", "linter"], "all": true, "timeout_seconds": 30}),
    ));
    let root = Message::new(
        Role::Assistant,
        vec![spawn("parser", "t1"), spawn("linter", "t2"), wait],
    );
    let scripts = HashMap::from([
        ("a1", vec![MockResponse::message(root), say("root: done")]),
        ("a2", vec![say("parser: all green")]),
        ("a3", vec![say("linter: all green")]),
    ]);
    let gate = Arc::new(Semaphore::new(0));
    let factory = gated_factory(scripts, &["a2", "a3"], &gate, &Sinks::default());
    let mut config = SwarmConfig::new(factory, dir.path());
    config.max_concurrent_calls = 1;
    let (swarm, mut events) = Swarm::new(config);
    let mut app = App::default();
    let mut screen = Screen::new();
    screen.type_line(&swarm, &mut app, "fix the bug").await;
    let before = screen
        .until(&mut app, &mut events, "3 agents · 1 running · 2 waiting")
        .await;
    assert!(!before.contains('▸'), "an unfolded tree has no marker\n{before}");
    let child_rows = |shown: &str| {
        ["  ⠋ parser", "  ◌ parser", "  ⠋ linter", "  ◌ linter"]
            .iter()
            .filter(|row| shown.contains(*row))
            .count()
    };
    assert_eq!(child_rows(&before), 2, "{before}");

    // ← folds the root: its children leave the list, its row counts them and who works.
    let folded = screen.press(&swarm, &mut app, KeyCode::Left).await;
    assert!(folded.contains("⧗ fix-bug ▸ (+2, 1 running)"), "{folded}");
    assert_eq!(child_rows(&folded), 0, "{folded}");
    assert!(
        folded.contains("3 agents · 1 running · 2 waiting"),
        "the header still counts them\n{folded}"
    );
    assert!(
        !folded.contains("hidden (f3)"),
        "folded agents are not cancelled ones\n{folded}"
    );
    // ↓ skips the hidden agents; ← on a folded root has no parent to go to.
    screen.press(&swarm, &mut app, KeyCode::Down).await;
    assert_eq!(app.selected_id().as_deref(), Some("a1"));
    screen.press(&swarm, &mut app, KeyCode::Left).await;
    assert_eq!(app.selected_id().as_deref(), Some("a1"));

    // → unfolds it, and the screen is as it was.
    let unfolded = screen.press(&swarm, &mut app, KeyCode::Right).await;
    assert_eq!(unfolded, before);
    // → on an unfolded agent selects its first child; ← on a leaf, its parent.
    screen.press(&swarm, &mut app, KeyCode::Right).await;
    assert_eq!(app.selected_id().as_deref(), Some("a2"));
    screen.press(&swarm, &mut app, KeyCode::Down).await;
    assert_eq!(app.selected_id().as_deref(), Some("a3"));
    screen.press(&swarm, &mut app, KeyCode::Left).await;
    assert_eq!(app.selected_id().as_deref(), Some("a1"));

    // With text in the input, the arrows move the cursor and fold nothing.
    for code in [
        KeyCode::Char('a'),
        KeyCode::Char('b'),
        KeyCode::Left,
        KeyCode::Char('X'),
        KeyCode::Right,
        KeyCode::Char('Y'),
    ] {
        screen.press(&swarm, &mut app, code).await;
    }
    assert_eq!(app.input, "aXbY");
    assert!(app.folded.is_empty());
    let shown = screen.press(&swarm, &mut app, KeyCode::Esc).await;
    assert_eq!(child_rows(&shown), 2, "{shown}");

    // Folded agents' events still arrive: the count follows them as they finish.
    screen.press(&swarm, &mut app, KeyCode::Left).await;
    gate.add_permits(2);
    let shown = screen.until(&mut app, &mut events, "fix-bug ▸ (+2) ").await;
    assert_eq!(child_rows(&shown), 0, "{shown}");
    screen.until(&mut app, &mut events, "root: done").await;
    screen.press(&swarm, &mut app, KeyCode::Right).await;
    let selected = |app: &App| app.selected_id();
    screen.press(&swarm, &mut app, KeyCode::Down).await;
    assert_eq!(selected(&app).as_deref(), Some("a2"));
    let shown = screen.until(&mut app, &mut events, "parser: all green").await;
    assert!(shown.contains("  ● parser"), "{shown}");
    swarm.shutdown().await;
}

/// The header line of a drawn screen.
fn shown_header(shown: &str) -> &str {
    shown.lines().next().unwrap_or_default()
}

/// Ctrl-C twice, or `/quit`, ends the loop with a reason, and the swarm's `shutdown` event in
/// `events.jsonl` carries it.
#[tokio::test]
async fn a_tui_quit_records_why_in_the_shutdown_event() {
    for (keys, reason) in [("^C^C", "tui-ctrl-c"), ("/quit", "tui-quit-command")] {
        let dir = tempfile::tempdir().unwrap();
        let state_dir = dir.path().join("run");
        let mut config = SwarmConfig::new(factory(HashMap::new(), &Sinks::default()), dir.path());
        config.state_dir = Some(state_dir.clone());
        let (swarm, _events) = Swarm::new(config);
        let mut app = App::default();
        let mut screen = Screen::new();
        match keys {
            "^C^C" => {
                for _ in 0..2 {
                    let key = Event::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
                    on_terminal_event(&swarm, &mut app, key, &screen.buffer).await;
                }
            }
            line => screen.type_line(&swarm, &mut app, line).await,
        }
        assert_eq!(app.quit, Some(reason));
        swarm
            .shutdown_because(&shutdown_reason(&Ok(Exit::Quit), app.quit))
            .await;
        let events = std::fs::read_to_string(state_dir.join(crate::journal::EVENTS_FILE)).unwrap();
        let shutdown = events
            .lines()
            .find(|l| l.contains("\"event\":\"shutdown\""))
            .unwrap();
        assert!(
            shutdown.contains(&format!("\"reason\":\"{reason}\"")),
            "{shutdown}"
        );
    }
}
