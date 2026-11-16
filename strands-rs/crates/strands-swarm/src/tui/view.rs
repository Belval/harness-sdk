//! Rendering [`App`] with ratatui.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;

use super::app::{AgentView, App, LogEntry};
use crate::runtime::AgentStatus;

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const HELP: &str =
    "enter send · /new <task> · /goal <goal> · /model [<id>|default] · /cancel · ↑↓ select · pgup/pgdn scroll · ctrl-c quit";

/// Draw the whole screen. `tick` animates spinners.
pub fn draw(frame: &mut Frame, app: &App, run_id: &str, tick: usize) {
    let [header, body, input, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(5),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());
    let [agents, log] =
        Layout::horizontal([Constraint::Percentage(28), Constraint::Percentage(72)]).areas(body);

    draw_header(frame, header, app, run_id);
    draw_agents(frame, agents, app, tick);
    draw_log(frame, log, app);
    draw_input(frame, input, app);
    let footer_text = app.notice.clone().unwrap_or_else(|| HELP.to_owned());
    frame.render_widget(Paragraph::new(footer_text).dim(), footer);
}

fn draw_header(frame: &mut Frame, area: Rect, app: &App, run_id: &str) {
    let running = app
        .agents
        .values()
        .filter(|v| v.status == AgentStatus::Running)
        .count();
    let (input, output) = app.total_tokens();
    let line = Line::from(vec![
        Span::styled(" strands-swarm ", Style::new().bold().reversed()),
        Span::raw(format!(
            "  run {run_id} · {} agents · {running} running · {} in / {} out tokens",
            app.agents.len(),
            compact(input),
            compact(output)
        )),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn draw_agents(frame: &mut Frame, area: Rect, app: &App, tick: usize) {
    let items: Vec<ListItem> = app
        .tree()
        .into_iter()
        .map(|(depth, view)| ListItem::new(agent_item(depth, view, tick)))
        .collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" agents "))
        .highlight_style(Style::new().add_modifier(Modifier::REVERSED));
    let mut state = ListState::default().with_selected((!app.agents.is_empty()).then_some(app.selected));
    frame.render_stateful_widget(list, area, &mut state);
}

/// Two lines per agent: `⠋ parser a2 ⎇ •`, then what it is doing, dimmed. Both are clipped by
/// the list's width, so the name stays visible on narrow terminals.
fn agent_item(depth: usize, view: &AgentView, tick: usize) -> Text<'static> {
    let (glyph, color) = status_glyph(&view.status, tick);
    let indent = "  ".repeat(depth);
    let mut spans = vec![
        Span::raw(indent.clone()),
        Span::styled(format!("{glyph} "), Style::new().fg(color)),
        Span::styled(view.name.clone(), Style::new().bold()),
        Span::styled(format!(" {}", view.id), Style::new().dim()),
    ];
    if view.branch.is_some() {
        spans.push(Span::styled(" ⎇", Style::new().fg(Color::Magenta)));
    }
    if view.unseen {
        spans.push(Span::styled(" •", Style::new().fg(Color::Yellow)));
    }
    let doing = Line::from(Span::styled(
        format!("{indent}  {}", view.doing),
        Style::new().dim(),
    ));
    Text::from(vec![Line::from(spans), doing])
}

fn status_glyph(status: &AgentStatus, tick: usize) -> (&'static str, Color) {
    match status {
        AgentStatus::Starting => ("…", Color::Gray),
        AgentStatus::Running => (SPINNER[tick % SPINNER.len()], Color::Cyan),
        AgentStatus::Idle => ("●", Color::Green),
        AgentStatus::Errored(_) => ("✗", Color::Red),
        AgentStatus::Cancelled => ("⊘", Color::DarkGray),
    }
}

fn draw_log(frame: &mut Frame, area: Rect, app: &App) {
    let selected = app.selected_id().and_then(|id| app.agents.get(&id));
    let Some(view) = selected else {
        let hint = Paragraph::new("Type a task below and press enter to start the root agent.")
            .block(Block::default().borders(Borders::ALL).title(" log "));
        frame.render_widget(hint, area);
        return;
    };
    let title = log_title(view);
    let text = Text::from(view.log.iter().flat_map(entry_lines).collect::<Vec<_>>());
    let paragraph = Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title(title));
    let inner_height = area.height.saturating_sub(2);
    let total = u16::try_from(paragraph.line_count(area.width.saturating_sub(2))).unwrap_or(u16::MAX);
    let bottom = total.saturating_sub(inner_height);
    let offset = bottom.saturating_sub(app.scroll);
    frame.render_widget(paragraph.scroll((offset, 0)), area);
}

fn log_title(view: &AgentView) -> String {
    let branch = view
        .branch
        .as_ref()
        .map(|b| format!(" · {b}"))
        .unwrap_or_default();
    let model = view.model.as_ref().map(|m| format!(" · {m}")).unwrap_or_default();
    format!(
        " {} · {} · {}{branch} · {}/{} tok{model} ",
        view.label(),
        view.status,
        view.cwd.display(),
        compact(view.input_tokens),
        compact(view.output_tokens)
    )
}

fn entry_lines(entry: &LogEntry) -> Vec<Line<'static>> {
    match entry {
        LogEntry::Task(task) => block_lines("▶ task", task, Color::Blue),
        LogEntry::Text(text) => text
            .lines()
            .map(|l| Line::raw(l.to_owned()))
            .chain([Line::raw("")])
            .collect(),
        LogEntry::Tool { name, summary, ok } => {
            let (mark, color) = match ok {
                None => ("⋯", Color::Gray),
                Some(true) => ("✓", Color::Green),
                Some(false) => ("✗", Color::Red),
            };
            vec![Line::from(vec![
                Span::styled(format!("{mark} {name} "), Style::new().fg(color)),
                Span::styled(summary.clone(), Style::new().dim()),
            ])]
        }
        LogEntry::MailIn { from, kind, text } => {
            let color = if kind == "error" {
                Color::Red
            } else {
                Color::Yellow
            };
            block_lines(&format!("✉ {from} ({kind})"), text, color)
        }
        LogEntry::Status(note) => vec![Line::styled(format!("— {note}"), Style::new().fg(Color::Red))],
    }
}

fn block_lines(heading: &str, body: &str, color: Color) -> Vec<Line<'static>> {
    let mut lines = vec![Line::styled(heading.to_owned(), Style::new().fg(color).bold())];
    lines.extend(
        body.lines()
            .map(|l| Line::styled(format!("  {l}"), Style::new().fg(color))),
    );
    lines.push(Line::raw(""));
    lines
}

fn draw_input(frame: &mut Frame, area: Rect, app: &App) {
    let target = app
        .selected_id()
        .and_then(|id| app.agents.get(&id).map(AgentView::label))
        .map_or_else(
            || " new root task ".to_owned(),
            |label| format!(" message → {label} "),
        );
    let input =
        Paragraph::new(app.input.as_str()).block(Block::default().borders(Borders::ALL).title(target));
    frame.render_widget(input, area);
    let cursor_x = area.x + 1 + u16::try_from(app.input.chars().count()).unwrap_or(u16::MAX);
    frame.set_cursor_position((cursor_x.min(area.right().saturating_sub(2)), area.y + 1));
}

fn compact(n: u64) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=999_999 => format!("{:.1}k", n as f64 / 1e3),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}
