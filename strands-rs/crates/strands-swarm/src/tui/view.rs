//! Rendering [`App`] with ratatui.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Padding, Paragraph, Wrap};
use ratatui::Frame;

use super::app::{AgentView, App, LogEntry};
use super::select::{self, Pane};
use crate::runtime::{AgentStatus, USER};

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// Narrowest agent tree; shrinking it further hides it.
const MIN_TREE: usize = 10;
const HINT: &str = "/ commands · f1 keys";
/// Key bindings, for the F1 popup.
const KEYS: &[(&str, &str)] = &[
    ("enter", "send to the selected agent"),
    ("/", "commands"),
    ("↑ ↓", "select an agent (or a command)"),
    ("pgup pgdn end", "scroll the log (or the mouse wheel)"),
    ("drag", "select in one pane and copy (shift-drag: terminal's own)"),
    ("ctrl-← →", "tree width (hides when narrow)"),
    ("f2", "plain-text log, for native scroll and copy"),
    ("f3", "show or hide cancelled agents"),
    ("esc", "clear the input and selection, close this"),
    ("esc esc", "stop the selected agent's turn (it stays alive)"),
    ("ctrl-c", "quit"),
];

/// Where each part of the screen goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Areas {
    /// Run summary line.
    pub header: Rect,
    /// Tree and log together.
    pub body: Rect,
    /// Agent tree, empty when hidden.
    pub tree: Rect,
    /// Selected agent's log.
    pub log: Rect,
    /// Input line.
    pub input: Rect,
    /// Help or notice line.
    pub footer: Rect,
}

impl Areas {
    /// The text area of `pane`, inside its borders.
    pub fn inner(&self, pane: Pane) -> Rect {
        match pane {
            // Top border; right border.
            Pane::Tree => Rect {
                y: self.tree.y + 1,
                height: self.tree.height.saturating_sub(1),
                width: self.tree.width.saturating_sub(1),
                ..self.tree
            },
            // Top border; one column of left padding.
            Pane::Log => Rect {
                x: self.log.x + 1,
                y: self.log.y + 1,
                width: self.log.width.saturating_sub(1),
                height: self.log.height.saturating_sub(1),
            },
        }
    }

    /// The pane whose text area contains `position`.
    pub fn pane_at(&self, position: ratatui::layout::Position) -> Option<Pane> {
        [Pane::Log, Pane::Tree]
            .into_iter()
            .find(|pane| self.inner(*pane).contains(position))
    }
}

/// Lay out a screen of `area` for `app`.
pub fn areas(app: &App, area: Rect) -> Areas {
    let [header, body, input, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(5),
        Constraint::Length(2),
        Constraint::Length(1),
    ])
    .areas(area);
    let tree = tree_width(app, body.width);
    let [tree, log] = Layout::horizontal([Constraint::Length(tree), Constraint::Min(0)]).areas(body);
    Areas {
        header,
        body,
        tree,
        log,
        input,
        footer,
    }
}

/// Draw the whole screen. `tick` animates spinners.
pub fn draw(frame: &mut Frame, app: &App, run_id: &str, tick: usize) {
    let areas = areas(app, frame.area());
    draw_header(frame, areas.header, app, run_id);
    if areas.tree.width > 0 {
        draw_agents(frame, areas.tree, app, tick);
    }
    draw_log(frame, areas.log, app, tick);
    draw_input(frame, areas.input, app);
    if let Some(selection) = &app.selection {
        select::highlight(frame.buffer_mut(), areas.inner(selection.pane), selection);
    }
    draw_completions(frame, areas.input, app);
    if app.show_keys {
        draw_keys(frame, areas.body);
    }
    let footer_text = app.notice.clone().unwrap_or_else(|| HINT.to_owned());
    frame.render_widget(Paragraph::new(footer_text).dim(), areas.footer);
}

fn draw_header(frame: &mut Frame, area: Rect, app: &App, run_id: &str) {
    let running = app
        .agents
        .values()
        .filter(|v| v.status == AgentStatus::Running)
        .count();
    let (input, output) = app.total_tokens();
    let cached = cached(app.total_cache_read(), input)
        .map(|percent| format!(" ({percent} cached)"))
        .unwrap_or_default();
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
            "  run {run_id} · {} agents · {running} running · in {}{cached} · out {}{budget}{hidden}",
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

/// One line per agent: `⠋ parser opus-5-5 1.2M/34k 34% a2 ⎇ •`. The list clips it at its
/// width, so the trailing id goes first on narrow terminals.
fn agent_item(depth: usize, view: &AgentView, tick: usize) -> Text<'static> {
    let (glyph, color) = status_glyph(&view.status, tick);
    let mut spans = vec![
        Span::raw("  ".repeat(depth)),
        Span::styled(format!("{glyph} "), Style::new().fg(color)),
        Span::styled(view.name.clone(), Style::new().bold()),
    ];
    if let Some(model) = &view.model {
        spans.push(Span::styled(
            format!(" {}", short_model(model)),
            Style::new().dim(),
        ));
    }
    if view.input_tokens + view.output_tokens > 0 {
        spans.push(Span::styled(format!(" {}", tokens(view)), Style::new().dim()));
    }
    if let Some((percent, color)) = context(view) {
        spans.push(Span::styled(format!(" {percent}"), Style::new().fg(color)));
    }
    spans.push(Span::styled(format!(" {}", view.id), Style::new().dim()));
    if view.branch.is_some() {
        spans.push(Span::styled(" ⎇", Style::new().fg(Color::Magenta)));
    }
    if view.unseen {
        spans.push(Span::styled(" •", Style::new().fg(Color::Yellow)));
    }
    Text::from(Line::from(spans))
}

/// The share of `input` read from the prompt cache, e.g. `97%`; `None` without input.
fn cached(cache_read: u64, input: u64) -> Option<String> {
    (input > 0).then(|| format!("{}%", (cache_read as f64 * 100.0 / input as f64).round() as u64))
}

/// `1.2M/34k`: cumulative input/output tokens.
fn tokens(view: &AgentView) -> String {
    format!("{}/{}", compact(view.input_tokens), compact(view.output_tokens))
}

/// The last model call's context window use, e.g. `34%`, colored green/yellow/red at 50%/80%.
fn context(view: &AgentView) -> Option<(String, Color)> {
    let context = view.context?;
    let color = match context {
        c if c >= 0.8 => Color::Red,
        c if c >= 0.5 => Color::Yellow,
        _ => Color::Green,
    };
    Some((format!("{}%", (context * 100.0).round() as u64), color))
}

/// A model id without provider, region, vendor and version noise:
/// `bedrock/global.anthropic.claude-opus-5-5` → `opus-5-5`, `amazon.nova-pro-v1:0` → `nova-pro`.
fn short_model(id: &str) -> &str {
    let mut name = id.rsplit('/').next().unwrap_or(id);
    while let Some((head, rest)) = name.split_once('.') {
        match !head.is_empty() && head.chars().all(|c| c.is_ascii_lowercase()) {
            true => name = rest,
            false => break,
        }
    }
    name = name.split(':').next().unwrap_or(name);
    name = name.strip_prefix("claude-").unwrap_or(name);
    while let Some((rest, tail)) = name.rsplit_once('-') {
        let version =
            tail.len() > 1 && tail.starts_with('v') && tail[1..].chars().all(|c| c.is_ascii_digit());
        let date = tail.len() == 8 && tail.chars().all(|c| c.is_ascii_digit());
        match version || date {
            true => name = rest,
            false => break,
        }
    }
    name
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
    let (lines, thumbnails) = log_lines(view, app, tick);
    let (paragraph, offset) = log_paragraph(app, view, lines.clone(), area);
    frame.render_widget(paragraph, area);
    if !thumbnails.is_empty() {
        draw_thumbnails(frame, app, &lines, &thumbnails, offset, log_inner(area));
    }
}

/// The scrolled log paragraph for `area` and its first visible row.
fn log_paragraph(
    app: &App,
    view: &AgentView,
    lines: Vec<Line<'static>>,
    area: Rect,
) -> (Paragraph<'static>, u16) {
    let paragraph = Paragraph::new(Text::from(lines))
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::TOP)
                .padding(Padding::left(1))
                .title(log_title(view)),
        );
    // `line_count` takes the inner width but counts the block's top border as a line.
    let count = paragraph
        .line_count(area.width.saturating_sub(1))
        .saturating_sub(1);
    let offset = log_offset(app, count, area.height.saturating_sub(1));
    (paragraph.scroll((offset, 0)), offset)
}

/// The log's text area inside `area` (below the title border, after the left padding).
fn log_inner(area: Rect) -> Rect {
    Rect::new(
        area.x.saturating_add(1),
        area.y.saturating_add(1),
        area.width.saturating_sub(1),
        area.height.saturating_sub(1),
    )
}

/// Draw each kept thumbnail over the blank rows reserved for it, when they are fully in view
/// (a graphics protocol cannot be cut at the top).
fn draw_thumbnails(
    frame: &mut Frame,
    app: &App,
    lines: &[Line<'static>],
    thumbnails: &[(usize, u64)],
    offset: u16,
    inner: Rect,
) {
    let mut row = 0usize;
    let mut starts = Vec::with_capacity(lines.len());
    for line in lines {
        starts.push(row);
        row += Paragraph::new(line.clone())
            .wrap(Wrap { trim: false })
            .line_count(inner.width)
            .max(1);
    }
    for (line, id) in thumbnails {
        let Some(protocol) = app.images.get(*id) else {
            continue;
        };
        let size = protocol.size();
        let Some(top) = starts[*line].checked_sub(usize::from(offset)) else {
            continue;
        };
        if top + usize::from(size.height) > usize::from(inner.height) || size.width > inner.width {
            continue;
        }
        let at = Rect::new(inner.x, inner.y + top as u16, size.width, size.height);
        frame.render_widget(ratatui_image::Image::new(protocol), at);
    }
}

/// The log pane of `area` as text only, the way a copy should see it: thumbnails left out (their
/// placeholder line stays), so selecting over one never picks up graphics cells.
pub fn log_text(app: &App, area: Rect) -> ratatui::buffer::Buffer {
    let log = areas(app, area).log;
    let mut buffer = ratatui::buffer::Buffer::empty(area);
    if let Some(view) = app.selected_id().and_then(|id| app.agents.get(&id)) {
        let (lines, _) = log_lines(view, app, 0);
        let (paragraph, _) = log_paragraph(app, view, lines, log);
        ratatui::widgets::Widget::render(paragraph, log, &mut buffer);
    }
    buffer
}

/// The log's lines, with blank rows reserved under each kept thumbnail's placeholder, and the
/// index of each reserved block's first row with its image.
fn log_lines(view: &AgentView, app: &App, tick: usize) -> (Vec<Line<'static>>, Vec<(usize, u64)>) {
    let mut lines: Vec<Line> = Vec::new();
    let mut thumbnails = Vec::new();
    for entry in &view.log {
        lines.extend(entry_lines(entry));
        if let LogEntry::Image {
            decoded: Some(decoded),
            ..
        } = entry
        {
            let rows = app.images.rows(decoded.id);
            if rows > 0 {
                thumbnails.push((lines.len(), decoded.id));
                lines.extend(std::iter::repeat_n(Line::raw(""), usize::from(rows)));
            }
        }
    }
    if let Some(activity) = view.activity() {
        let glyph = SPINNER[tick % SPINNER.len()];
        lines.push(Line::styled(
            format!("{glyph} {activity}"),
            Style::new().fg(Color::Cyan),
        ));
    }
    (lines, thumbnails)
}

/// First visible row of a log of `rows` wrapped rows in a pane `height` rows tall.
fn log_offset(app: &App, rows: usize, height: u16) -> u16 {
    let total = u16::try_from(rows).unwrap_or(u16::MAX);
    total.saturating_sub(height).saturating_sub(app.scroll)
}

/// For each row of the log's text area `inner`: whether it continues the row above (a wrapped
/// line), so a copied selection can rejoin it.
pub fn log_continuations(app: &App, inner: Rect) -> Vec<bool> {
    let Some(view) = app.selected_id().and_then(|id| app.agents.get(&id)) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for line in log_lines(view, app, 0).0 {
        let wrapped = Paragraph::new(line)
            .wrap(Wrap { trim: false })
            .line_count(inner.width);
        rows.push(false);
        rows.extend(std::iter::repeat_n(true, wrapped.max(1) - 1));
    }
    let offset = usize::from(log_offset(app, rows.len(), inner.height));
    rows.into_iter()
        .skip(offset)
        .take(usize::from(inner.height))
        .collect()
}

/// `parser a2 · running · opus-5-5 · 1.2M/34k tok · 34% · …/a2/repo`.
fn log_title(view: &AgentView) -> String {
    let status = match &view.status {
        AgentStatus::Errored(_) => "errored".to_owned(),
        status => status.to_string(),
    };
    let mut parts = vec![view.label(), status];
    parts.extend(view.model.as_deref().map(|model| short_model(model).to_owned()));
    if view.input_tokens + view.output_tokens > 0 {
        let cached = cached(view.cache_read_tokens, view.input_tokens)
            .map(|percent| format!(" ({percent} cached)"))
            .unwrap_or_default();
        parts.push(format!("{} tok{cached}", tokens(view)));
    }
    parts.extend(context(view).map(|(percent, _)| percent));
    parts.push(short_path(&view.cwd));
    format!(" {} ", parts.join(" · "))
}

/// The last two components of `path`, e.g. `…/a2/repo`.
fn short_path(path: &std::path::Path) -> String {
    let parts: Vec<_> = path.iter().map(|part| part.to_string_lossy()).collect();
    match parts.len() {
        0..=3 => path.display().to_string(),
        n => format!("…/{}/{}", parts[n - 2], parts[n - 1]),
    }
}

/// The F1 key-binding popup, centered over the body.
fn draw_keys(frame: &mut Frame, body: Rect) {
    let lines: Vec<Line> = KEYS
        .iter()
        .map(|(key, help)| {
            Line::from(vec![
                Span::styled(format!(" {key:<14}"), Style::new().bold()),
                Span::styled(*help, Style::new().dim()),
            ])
        })
        .collect();
    let width = 60.min(body.width);
    let height = (KEYS.len() as u16 + 2).min(body.height);
    let area = Rect::new(
        body.x + (body.width - width) / 2,
        body.y + (body.height - height) / 2,
        width,
        height,
    );
    let popup =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" keys · esc close "));
    frame.render_widget(Clear, area);
    frame.render_widget(popup, area);
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
        LogEntry::Image { source, decoded } => vec![Line::styled(
            LogEntry::placeholder(source, decoded.as_ref()),
            Style::new().fg(Color::Magenta),
        )],
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

    /// An app with agent `a1` whose log holds a 40x30 image from `read`, thumbnails on `picker`.
    fn app_with_an_image(picker: Option<ratatui_image::picker::Picker>) -> App {
        let mut app = App {
            images: picker.map_or_else(Default::default, super::super::images::Images::with_picker),
            ..Default::default()
        };
        app.apply(crate::runtime::SwarmEvent::Spawned {
            id: "a1".into(),
            parent: USER.into(),
            name: "shots".into(),
            task: "look".into(),
            cwd: "/w".into(),
            branch: None,
        });
        app.apply(crate::runtime::SwarmEvent::Image {
            id: "a1".into(),
            source: "/w/shot.png".into(),
            bytes: std::sync::Arc::new(super::super::images::png(40, 30)),
        });
        app
    }

    fn screen_text(buffer: &ratatui::buffer::Buffer) -> String {
        let area = buffer.area;
        (area.top()..area.bottom())
            .map(|y| {
                (area.left()..area.right())
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn an_image_shows_as_a_thumbnail_under_its_placeholder_but_copies_as_text() {
        let app = app_with_an_image(Some(ratatui_image::picker::Picker::halfblocks()));
        let mut terminal = ratatui::Terminal::new(ratatui::backend::TestBackend::new(100, 30)).unwrap();
        terminal.draw(|frame| draw(frame, &app, "run", 0)).unwrap();
        let screen = screen_text(terminal.backend().buffer());
        assert!(screen.contains("[image: /w/shot.png 40x30]"), "{screen}");
        let halfblock = |text: &str| text.contains('▀') || text.contains('▄');
        assert!(halfblock(&screen), "no halfblock thumbnail:\n{screen}");

        let copy = screen_text(&log_text(&app, Rect::new(0, 0, 100, 30)));
        assert!(copy.contains("[image: /w/shot.png 40x30]"), "{copy}");
        assert!(!halfblock(&copy), "{copy}");
        assert!(plain_log(&app.agents["a1"]).contains("[image: /w/shot.png 40x30]\n"));
    }

    #[test]
    fn without_a_picker_an_image_is_only_its_placeholder() {
        let app = app_with_an_image(None);
        let mut terminal = ratatui::Terminal::new(ratatui::backend::TestBackend::new(100, 30)).unwrap();
        terminal.draw(|frame| draw(frame, &app, "run", 0)).unwrap();
        let screen = screen_text(terminal.backend().buffer());
        assert!(screen.contains("[image: /w/shot.png 40x30]"), "{screen}");
        assert!(!screen.contains('▀') && !screen.contains('▄'), "{screen}");
        assert_eq!(
            LogEntry::placeholder("x.png", None),
            "[image: x.png, not decodable]"
        );
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

    #[test]
    fn model_ids_are_shortened_for_the_agent_tree() {
        for (id, short) in [
            ("bedrock/global.anthropic.claude-opus-5-5", "opus-5-5"),
            ("us.anthropic.claude-3-5-sonnet-20240620-v1:0", "3-5-sonnet"),
            ("amazon.nova-pro-v1:0", "nova-pro"),
            ("openai/gpt-4.1", "gpt-4.1"),
            ("test-model", "test-model"),
        ] {
            assert_eq!(short_model(id), short, "{id}");
        }
    }

    #[test]
    fn an_agent_is_one_compact_line_with_model_tokens_and_context() {
        use crate::runtime::SwarmEvent;
        let mut app = App::default();
        app.apply(SwarmEvent::Spawned {
            id: "a2".into(),
            parent: USER.into(),
            name: "parser".into(),
            task: "parse".into(),
            cwd: "/home/me/work/a2/repo".into(),
            branch: None,
        });
        app.apply(SwarmEvent::Model {
            id: "a2".into(),
            model: "global.anthropic.claude-opus-5-5".into(),
        });
        app.apply(SwarmEvent::Usage {
            id: "a2".into(),
            input_tokens: 1_200_000,
            output_tokens: 34_000,
            cache_read_tokens: 1_164_000,
            cache_write_tokens: 0,
            context: Some(0.34),
        });
        let view = &app.agents["a2"];
        let item = agent_item(0, view, 0);
        assert_eq!(item.lines.len(), 1);
        assert!(
            item.lines[0]
                .to_string()
                .contains("parser opus-5-5 1.2M/34.0k 34% a2"),
            "{}",
            item.lines[0]
        );
        assert_eq!(
            log_title(view),
            " parser (a2) · starting · opus-5-5 · 1.2M/34.0k tok (97% cached) · 34% · …/a2/repo "
        );
    }
}
