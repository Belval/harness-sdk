//! Rendering [`App`] with ratatui.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Padding, Paragraph, Wrap};
use ratatui::Frame;

use super::app::{AgentView, App, LogEntry};
use crate::runtime::{AgentStatus, USER};

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// Narrowest agent tree; shrinking it further hides it.
const MIN_TREE: usize = 10;
const HELP: &str = "enter send · / commands · ↑↓ select · pgup/pgdn scroll · ctrl-←→ tree width · \
     f2 plain text · f3 cancelled · ctrl-c quit";

/// Draw the whole screen. `tick` animates spinners.
pub fn draw(frame: &mut Frame, app: &App, run_id: &str, tick: usize) {
    let [header, body, input, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(5),
        Constraint::Length(2),
        Constraint::Length(1),
    ])
    .areas(frame.area());
    let tree = tree_width(app, body.width);
    let [agents, log] = Layout::horizontal([Constraint::Length(tree), Constraint::Min(0)]).areas(body);

    draw_header(frame, header, app, run_id);
    if tree > 0 {
        draw_agents(frame, agents, app, tick);
    }
    draw_log(frame, log, app, tick);
    draw_input(frame, input, app);
    draw_completions(frame, input, app);
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
    let hidden = match app.hidden() {
        0 => String::new(),
        n => format!(" · {n} cancelled hidden (f3)"),
    };
    let budget = app
        .budget
        .map(|budget| format!(" · budget {}", compact(budget)))
        .unwrap_or_default();
    let line = Line::from(vec![
        Span::styled(" strands-swarm ", Style::new().bold().reversed()),
        Span::raw(format!(
            "  run {run_id} · {} agents · {running} running · {} in / {} out tokens{budget}{hidden}",
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
        .block(
            Block::default()
                .borders(Borders::TOP | Borders::RIGHT)
                .title(" agents "),
        )
        .highlight_style(Style::new().add_modifier(Modifier::REVERSED));
    let mut state = ListState::default().with_selected(app.selected_id().map(|_| app.selected));
    frame.render_stateful_widget(list, area, &mut state);
}

/// The agent tree's width: its longest name line plus room for the unseen marker, at most 30% of
/// `total`, adjusted by the user (up to 60%); 0 hides it.
fn tree_width(app: &App, total: u16) -> u16 {
    let fit = app
        .tree()
        .into_iter()
        .map(|(depth, view)| agent_item(depth, view, 0).lines[0].width() + if view.unseen { 1 } else { 3 })
        .max()
        .unwrap_or(0)
        .max(MIN_TREE);
    let fit = i32::try_from(fit)
        .unwrap_or(i32::MAX)
        .min(i32::from(total) * 3 / 10);
    let width = (fit + i32::from(app.tree_adjust)).clamp(0, i32::from(total) * 6 / 10);
    match u16::try_from(width) {
        Ok(width) if width >= MIN_TREE as u16 => width,
        _ => 0,
    }
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

fn draw_log(frame: &mut Frame, area: Rect, app: &App, tick: usize) {
    let selected = app.selected_id().and_then(|id| app.agents.get(&id));
    let Some(view) = selected else {
        let hint = Paragraph::new("Type a task below and press enter to start the root agent.")
            .block(Block::default().borders(Borders::TOP).title(" log "));
        frame.render_widget(hint, area);
        return;
    };
    let title = log_title(view);
    let mut lines: Vec<Line> = view.log.iter().flat_map(entry_lines).collect();
    if let Some(activity) = view.activity() {
        let glyph = SPINNER[tick % SPINNER.len()];
        lines.push(Line::styled(
            format!("{glyph} {activity}"),
            Style::new().fg(Color::Cyan),
        ));
    }
    let text = Text::from(lines);
    let paragraph = Paragraph::new(text).wrap(Wrap { trim: false }).block(
        Block::default()
            .borders(Borders::TOP)
            .padding(Padding::left(1))
            .title(title),
    );
    // `line_count` takes the inner width but counts the block's top border as a line.
    let inner_height = area.height.saturating_sub(1);
    let count = paragraph
        .line_count(area.width.saturating_sub(1))
        .saturating_sub(1);
    let total = u16::try_from(count).unwrap_or(u16::MAX);
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

/// An agent's log as plain text: the same lines the log pane shows, without styling, borders or
/// wrapping, so the terminal can scroll, search and copy it natively.
pub fn plain_log(view: &AgentView) -> String {
    let mut out = format!("{}\n\n", log_title(view).trim());
    for line in view.log.iter().flat_map(entry_lines) {
        out.extend(line.spans.iter().map(|span| span.content.as_ref()));
        out.push('\n');
    }
    out
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
        LogEntry::MailIn {
            from,
            kind,
            text,
            read,
        } => {
            // The user's own messages stand apart from agent traffic.
            let mut lines = match (from.as_str(), kind.as_str()) {
                (USER, _) => block_lines("▶ you", text, Color::Cyan),
                (_, "error") => block_lines(&format!("✉ {from} ({kind})"), text, Color::Red),
                _ => block_lines(&format!("✉ {from} ({kind})"), text, Color::Yellow),
            };
            match read {
                Some(false) => lines[0].push_span(Span::styled(" · queued", Style::new().dim())),
                Some(true) => lines[0].push_span(Span::styled(" · read ✓", Style::new().fg(Color::Green))),
                None => {}
            }
            lines
        }
        LogEntry::Status(note) => vec![Line::styled(format!("— {note}"), Style::new().fg(Color::Red))],
        LogEntry::Input(text) => block_lines("◀ input", text, Color::Gray),
        LogEntry::Note(note) => vec![Line::styled(format!("— {note} —"), Style::new().dim())],
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
        Paragraph::new(app.input.as_str()).block(Block::default().borders(Borders::TOP).title(target));
    frame.render_widget(input, area);
    let cursor_x = area.x + u16::try_from(app.input.chars().count()).unwrap_or(u16::MAX);
    frame.set_cursor_position((cursor_x.min(area.right().saturating_sub(1)), area.y + 1));
}

/// The slash-command popup, just above the input box.
fn draw_completions(frame: &mut Frame, input: Rect, app: &App) {
    let completions = app.completions();
    if completions.is_empty() {
        return;
    }
    let height = (completions.len() as u16 + 2).min(input.y);
    let width = input.width.min(72);
    let area = Rect::new(input.x, input.y - height, width, height);
    let items: Vec<ListItem> = completions
        .iter()
        .map(|command| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("{} ", command.name), Style::new().bold()),
                Span::raw(format!("{:<16}", command.args)),
                Span::styled(command.help, Style::new().dim()),
            ]))
        })
        .collect();
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" commands · tab/enter pick "),
        )
        .highlight_style(Style::new().reversed());
    let mut state = ListState::default().with_selected(Some(app.completion.min(completions.len() - 1)));
    frame.render_widget(Clear, area);
    frame.render_stateful_widget(list, area, &mut state);
}

fn compact(n: u64) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=999_999 => format!("{:.1}k", n as f64 / 1e3),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mail(from: &str) -> LogEntry {
        LogEntry::MailIn {
            from: from.into(),
            kind: "message".into(),
            text: "hi".into(),
            read: Some(true),
        }
    }

    #[test]
    fn the_users_messages_have_their_own_color() {
        let user = entry_lines(&mail(USER));
        let agent = entry_lines(&mail("a2 (parser)"));
        assert_eq!(user[0].to_string(), "▶ you · read ✓");
        assert_eq!(user[0].style.fg, Some(Color::Cyan));
        assert_eq!(agent[0].style.fg, Some(Color::Yellow));
    }

    #[test]
    fn typing_a_slash_shows_the_matching_commands_above_the_input() {
        let mut terminal = ratatui::Terminal::new(ratatui::backend::TestBackend::new(100, 20)).unwrap();
        let app = App {
            input: "/c".into(),
            ..Default::default()
        };
        terminal.draw(|frame| draw(frame, &app, "run", 0)).unwrap();
        let screen: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(screen.contains("/cancel"), "{screen}");
        assert!(!screen.contains("/model"), "{screen}");
    }
}
