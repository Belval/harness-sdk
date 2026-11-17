//! Rendering [`App`] with ratatui.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Padding, Paragraph, Wrap};
use ratatui::Frame;

use super::app::{AgentView, App, LogEntry, SlashCommand, Waiting};
use super::select::{self, Pane};
use crate::runtime::{AgentStatus, USER};
use crate::text::compact;

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// Narrowest agent tree; shrinking it further hides it.
const MIN_TREE: usize = 10;
const HINT: &str = "/ commands · f1 keys";
/// Most input rows shown at once (lines wrap); a longer message scrolls with the cursor.
const MAX_INPUT_LINES: usize = 6;

/// Key bindings, for the F1 popup.
const KEYS: &[(&str, &str)] = &[
    ("enter", "send to the selected agent"),
    ("/", "commands"),
    ("↑ ↓", "select an agent (or a command)"),
    ("pgup pgdn end", "scroll the log (or the mouse wheel)"),
    ("drag", "select in one pane and copy (shift-drag: terminal's own)"),
    (
        "← → home end",
        "move in the input (ctrl-a ctrl-e: line start, end)",
    ),
    ("ctrl-← →", "move by word"),
    ("ctrl-w ctrl-u", "delete the word, or to the line start"),
    (
        "alt-enter",
        "new line (shift-enter where the terminal reports it)",
    ),
    ("↑ ↓ with text", "earlier and later sent messages (ctrl-p ctrl-n)"),
    ("alt-← →", "tree width (hides when narrow)"),
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
            Pane::Log => log_inner(self.log),
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
    let (rows, (cursor, _)) = input_rows(app, usize::from(area.width));
    let lines = rows.len().max(cursor + 1).clamp(1, MAX_INPUT_LINES) as u16;
    let [header, body, input, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(5),
        Constraint::Length(1 + lines),
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
    if let Some(popup) = popup(app, areas.body) {
        let (area, _) = popup_area(areas.body, &popup);
        frame.render_widget(Clear, area);
        frame.render_widget(popup.scroll((app.status_scroll, 0)), area);
    }
    let footer_text = app.notice.clone().unwrap_or_else(|| HINT.to_owned());
    frame.render_widget(Paragraph::new(footer_text).dim(), areas.footer);
}

fn draw_header(frame: &mut Frame, area: Rect, app: &App, run_id: &str) {
    // Running agents that wait (for a model slot or for mail) are not counted as running.
    let (waiting, running) = app
        .agents
        .values()
        .filter(|v| v.status == AgentStatus::Running)
        .fold((0, 0), |(waiting, running), v| match v.waiting() {
            Some(_) => (waiting + 1, running),
            None => (waiting, running + 1),
        });
    let waiting = match waiting {
        0 => String::new(),
        n => format!(" · {n} waiting"),
    };
    let (input, output) = app.total_tokens();
    let cached = cached(app.total_cache_read(), input)
        .map(|percent| format!(" ({percent} cached)"))
        .unwrap_or_default();
    let hidden = match app.hidden() {
        0 => String::new(),
        n => format!(" · {n} cancelled hidden (f3)"),
    };
    let spend = match app.spend {
        Some(spend) if spend.any => format!(" · est. {}", spend.render()),
        _ => String::new(),
    };
    let budget = app
        .budget
        .map(|budget| format!(" · budget {}", compact(budget)))
        .unwrap_or_default();
    let tokens = format!(" · in {}{cached} · out {}", compact(input), compact(output));
    let header = |tokens: &str| {
        let mut line = Line::from(vec![
            Span::styled(" strands-swarm ", Style::new().bold().reversed()),
            Span::raw(format!(
                "  run {run_id} · {} agents · {running} running{waiting}{tokens}{spend}{budget}{hidden}",
                app.agents.len(),
            )),
        ]);
        if let Some(calls) = app.reload_pending {
            let text = format!(" · reload pending (waiting for {calls} tool calls)");
            line.push_span(Span::styled(text, Style::new().yellow()));
        }
        line
    };
    // A narrow terminal drops the token totals first: spend, budget, hidden agents and a
    // pending reload matter more.
    let mut line = header(&tokens);
    if line.width() > usize::from(area.width) {
        line = header("");
    }
    frame.render_widget(Paragraph::new(line), area);
}

fn draw_agents(frame: &mut Frame, area: Rect, app: &App, tick: usize) {
    let width = usize::from(area.width.saturating_sub(1));
    let items: Vec<ListItem> = app
        .tree()
        .into_iter()
        .map(|(depth, view)| ListItem::new(fit_row(agent_item(depth, view, tick), width, view.unseen)))
        .collect();
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::TOP | Borders::RIGHT)
                .title(" agents "),
        )
        .highlight_style(Style::new().add_modifier(Modifier::REVERSED));
    let mut state = ListState::default().with_selected(app.selected_view().map(|_| app.selected));
    frame.render_stateful_widget(list, area, &mut state);
}

/// The agent tree's width: its longest name line plus room for the unseen marker, at most 30% of
/// `total`, adjusted by the user (up to 60%); 0 hides it.
fn tree_width(app: &App, total: u16) -> u16 {
    let fit = app
        .tree()
        .into_iter()
        .map(|(depth, view)| agent_item(depth, view, 0).width() + if view.unseen { 1 } else { 3 })
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
fn agent_item(depth: usize, view: &AgentView, tick: usize) -> Line<'static> {
    let (glyph, color) = status_glyph(view, tick);
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
    Line::from(spans)
}

/// Clip an agent row to `width` columns, keeping its trailing unseen marker (the last span) in
/// view: it is what tells the user to look, so it must not fall off a narrow tree.
fn fit_row(mut line: Line<'static>, width: usize, unseen: bool) -> Line<'static> {
    if !unseen || line.width() <= width {
        return line;
    }
    let marker = line.spans.pop().unwrap_or_default();
    let mut room = width.saturating_sub(marker.width());
    let mut spans = Vec::new();
    for span in line.spans {
        let chars = span.content.chars().count();
        if chars > room {
            let head: String = span.content.chars().take(room).collect();
            spans.push(Span::styled(head, span.style));
            break;
        }
        room -= chars;
        spans.push(span);
    }
    spans.push(marker);
    Line::from(spans)
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

/// A working agent spins; one waiting for a model slot or for mail shows a still glyph.
fn status_glyph(view: &AgentView, tick: usize) -> (&'static str, Color) {
    match &view.status {
        AgentStatus::Queued => ("◌", Color::DarkGray),
        AgentStatus::Starting => ("…", Color::Gray),
        AgentStatus::Running => match view.waiting() {
            Some(Waiting::Slot) => ("◌", Color::Cyan),
            Some(Waiting::Mail) => ("⧗", Color::Cyan),
            None => (SPINNER[tick % SPINNER.len()], Color::Cyan),
        },
        AgentStatus::Idle => ("●", Color::Green),
        AgentStatus::Errored(_) => ("✗", Color::Red),
        AgentStatus::Cancelled => ("⊘", Color::DarkGray),
    }
}

fn draw_log(frame: &mut Frame, area: Rect, app: &App, tick: usize) {
    let Some(view) = app.selected_view() else {
        let hint = Paragraph::new("Type a task below and press enter to start the root agent.")
            .block(Block::default().borders(Borders::TOP).title(" log "));
        frame.render_widget(hint, area);
        return;
    };
    let window = log_window(view, app, tick, log_inner(area));
    frame.render_widget(log_paragraph(view, &window), area);
    unglue(frame.buffer_mut(), area);
    if !window.thumbnails.is_empty() {
        draw_thumbnails(frame, app, &window, log_inner(area));
    }
}

/// The scrolled log paragraph drawn in the log pane.
fn log_paragraph(view: &AgentView, window: &LogWindow) -> Paragraph<'static> {
    Paragraph::new(Text::from(window.lines.clone()))
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::TOP)
                .padding(Padding::left(1))
                .title(log_title(view)),
        )
        .scroll((window.offset, 0))
}

/// The end of a log that the pane shows: enough of its newest lines to fill the pane at the
/// current scroll, not the whole log: the frame is redrawn several times a second, too often to
/// wrap all of up to [`super::app::MAX_LOG_ENTRIES`] entries each time.
struct LogWindow {
    /// The log's last lines, with blank rows reserved under each kept thumbnail's placeholder.
    lines: Vec<Line<'static>>,
    /// Rows each of `lines` takes when wrapped.
    rows: Vec<usize>,
    /// The index in `lines` of each reserved block's first row, with its image.
    thumbnails: Vec<(usize, u64)>,
    /// The first visible row of `lines`.
    offset: u16,
}

/// The [`LogWindow`] of `view`'s log for the text area `inner`.
fn log_window(view: &AgentView, app: &App, tick: usize, inner: Rect) -> LogWindow {
    let wanted = usize::from(inner.height) + usize::from(app.scroll);
    let rows_of = |lines: &[Line<'static>]| -> Vec<usize> {
        lines.iter().map(|line| wrapped_rows(line, inner.width)).collect()
    };
    // Newest first: each entry's lines, their wrapped rows, and its thumbnail with the length of
    // its placeholder (the lines before the reserved rows).
    let mut groups = Vec::new();
    if let Some(activity) = view.activity() {
        // Only work spins; waiting (for mail, a slot, a reload) holds still, so an idle swarm's
        // screen does not change between ticks.
        let glyph = match view.spins() {
            true => SPINNER[tick % SPINNER.len()],
            false => status_glyph(view, tick).0,
        };
        let line = Line::styled(format!("{glyph} {activity}"), Style::new().fg(Color::Cyan));
        let lines = vec![line];
        groups.push((rows_of(&lines), lines, None));
    }
    let mut total = 0;
    for entry in view.log.iter().rev() {
        if total >= wanted {
            break;
        }
        let mut lines = entry_lines(entry);
        let mut thumbnail = None;
        if let LogEntry::Image {
            decoded: Some(decoded),
            ..
        } = entry
        {
            let rows = app.images.rows(decoded.id);
            if rows > 0 {
                thumbnail = Some((lines.len(), decoded.id));
                lines.extend(std::iter::repeat_n(Line::raw(""), usize::from(rows)));
            }
        }
        let rows = rows_of(&lines);
        total += rows.iter().sum::<usize>();
        groups.push((rows, lines, thumbnail));
    }
    let mut window = LogWindow {
        lines: Vec::new(),
        rows: Vec::new(),
        thumbnails: Vec::new(),
        offset: 0,
    };
    for (rows, lines, thumbnail) in groups.into_iter().rev() {
        if let Some((placeholder, id)) = thumbnail {
            window.thumbnails.push((window.lines.len() + placeholder, id));
        }
        window.rows.extend(rows);
        window.lines.extend(lines);
    }
    let shown = window.rows.iter().sum();
    window.offset = log_offset(app, shown, inner.height);
    window
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
fn draw_thumbnails(frame: &mut Frame, app: &App, window: &LogWindow, inner: Rect) {
    let mut row = 0usize;
    let mut starts = Vec::with_capacity(window.rows.len());
    for rows in &window.rows {
        starts.push(row);
        row += rows;
    }
    let offset = window.offset;
    for (line, id) in &window.thumbnails {
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
    if let Some(view) = app.selected_view() {
        let window = log_window(view, app, 0, log_inner(log));
        ratatui::widgets::Widget::render(log_paragraph(view, &window), log, &mut buffer);
        unglue(&mut buffer, log);
    }
    buffer
}

/// Keeps a tool line's mark, name and summary together when wrapping: the wrapper breaks a
/// too-long word at the edge rather than leaving `✗ read` alone above a long path.
const GLUE: char = '\u{a0}';

/// Turn the [`GLUE`] in `area` back into spaces, so copies get plain text.
fn unglue(buffer: &mut ratatui::buffer::Buffer, area: Rect) {
    let glue = GLUE.to_string();
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            if let Some(cell) = buffer.cell_mut((x, y)).filter(|cell| cell.symbol() == glue) {
                cell.set_symbol(" ");
            }
        }
    }
}

/// First visible row of a log of `rows` wrapped rows in a pane `height` rows tall.
fn log_offset(app: &App, rows: usize, height: u16) -> u16 {
    let total = u16::try_from(rows).unwrap_or(u16::MAX);
    total.saturating_sub(height).saturating_sub(app.scroll)
}

/// Rows `line` takes when wrapped at `width` (an empty line takes one).
fn wrapped_rows(line: &Line<'static>, width: u16) -> usize {
    Paragraph::new(line.clone())
        .wrap(Wrap { trim: false })
        .line_count(width)
        .max(1)
}

/// For each row of the log's text area `inner`: whether it continues the row above (a wrapped
/// line), so a copied selection can rejoin it.
pub fn log_continuations(app: &App, inner: Rect) -> Vec<bool> {
    let Some(view) = app.selected_view() else {
        return Vec::new();
    };
    let window = log_window(view, app, 0, inner);
    let mut rows = Vec::new();
    for wrapped in &window.rows {
        rows.push(false);
        rows.extend(std::iter::repeat_n(true, wrapped - 1));
    }
    rows.into_iter()
        .skip(usize::from(window.offset))
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
    let longest = lines.iter().map(Line::width).max().unwrap_or(0);
    let width = to_u16(longest + 2).min(body.width);
    let height = (KEYS.len() as u16 + 2).min(body.height);
    let area = centered(body, width, height);
    let popup =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" keys · esc close "));
    frame.render_widget(Clear, area);
    frame.render_widget(popup, area);
}

/// The `/status` or `/cost` popup, when one is open, unscrolled.
fn popup(app: &App, body: Rect) -> Option<Paragraph<'static>> {
    let (title, lines, wrap) = if let Some(report) = &app.status {
        let doing = |id: &str| {
            let view = app.agents.get(id)?;
            let queued = view.queued.filter(|_| view.status == AgentStatus::Running);
            if let Some((calls, limit)) = queued {
                return Some(crate::text::queued_for_call(calls, limit));
            }
            matches!(view.status, AgentStatus::Starting | AgentStatus::Running).then(|| view.doing.clone())
        };
        let lines: Vec<Line> = report
            .lines(&doing)
            .into_iter()
            .map(|line| match line.strip_prefix("  ") {
                Some(_) => Line::from(line).dim(),
                None => Line::from(line).bold(),
            })
            .collect();
        (" status · ↑↓ scroll · any key close ", lines, true)
    } else {
        // The first and last lines are prose (intro, footnote): those wrap by word; the table
        // between does not wrap, so its columns stay aligned.
        let width = usize::from(popup_width(body).saturating_sub(2));
        let lines = app.cost.as_ref()?.lines();
        let last = lines.len().saturating_sub(1);
        let lines = lines
            .into_iter()
            .enumerate()
            .flat_map(|(at, line)| match at == 0 || at == last {
                true => wrap_words(&line, width),
                false => vec![line],
            })
            .map(Line::from)
            .collect();
        (" estimated cost · ↑↓ scroll · any key close ", lines, false)
    };
    let popup = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title));
    Some(match wrap {
        true => popup.wrap(Wrap { trim: false }),
        false => popup,
    })
}

/// Where `popup` goes: centred over most of `body`, as tall as its lines need. Also returns
/// its height in rows when not clipped (borders included).
fn popup_area(body: Rect, popup: &Paragraph) -> (Rect, u16) {
    let width = popup_width(body);
    // `line_count` takes the inner width and counts both borders.
    let rows = to_u16(popup.line_count(width.saturating_sub(2)));
    let height = rows.min(body.height);
    (centered(body, width, height), rows)
}

/// A `width` × `height` rect in the middle of `body` (which it must fit in).
fn centered(body: Rect, width: u16, height: u16) -> Rect {
    Rect::new(
        body.x + (body.width - width) / 2,
        body.y + (body.height - height) / 2,
        width,
        height,
    )
}

fn popup_width(body: Rect) -> u16 {
    body.width.saturating_sub(4).max(body.width.min(20))
}

/// `text` split at spaces into lines of at most `width` characters (a longer word gets its own).
fn wrap_words(text: &str, width: usize) -> Vec<String> {
    let mut lines = vec![String::new()];
    for word in text.split(' ') {
        let line = lines.last_mut().expect("never empty");
        if line.is_empty() {
            line.push_str(word);
        } else if line.chars().count() + 1 + word.chars().count() <= width {
            line.push(' ');
            line.push_str(word);
        } else {
            lines.push(word.to_owned());
        }
    }
    lines
}

/// How far the open popup can scroll on a `screen`-sized terminal: until its last line shows.
pub fn popup_scroll_limit(app: &App, screen: Rect) -> u16 {
    let body = areas(app, screen).body;
    popup(app, body).map_or(0, |popup| {
        let (area, rows) = popup_area(body, &popup);
        rows - area.height
    })
}

/// An agent's log as plain text: the same lines the log pane shows, without styling, borders or
/// wrapping, so the terminal can scroll, search and copy it natively.
pub fn plain_log(view: &AgentView) -> String {
    let mut out = format!("{}\n\n", log_title(view).trim());
    for line in view.log.iter().flat_map(entry_lines) {
        out.extend(line.spans.iter().map(|span| span.content.replace(GLUE, " ")));
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
                Span::styled(format!("{mark}{GLUE}{name}{GLUE}"), Style::new().fg(color)),
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
    let target = app.selected_view().map(AgentView::label).map_or_else(
        || " new root task ".to_owned(),
        |label| format!(" message → {label} "),
    );
    // Wrapped at the box's width, scrolled so the cursor's row stays in view.
    let (rows, (row, col)) = input_rows(app, usize::from(area.width));
    let height = usize::from(area.height.saturating_sub(1)).max(1);
    let top = row.saturating_sub(height - 1);
    let lines: Vec<Line> = rows
        .iter()
        .skip(top)
        .take(height)
        .map(|range| Line::raw(&app.input[range.clone()]))
        .collect();
    let input = Paragraph::new(lines).block(Block::default().borders(Borders::TOP).title(target));
    frame.render_widget(input, area);
    let x = area.x.saturating_add(to_u16(col));
    let y = area.y.saturating_add(1).saturating_add(to_u16(row - top));
    frame.set_cursor_position((x.min(area.right().saturating_sub(1)), y));
}

/// The input wrapped at `width` columns: each row's byte range in it, and the cursor's row and
/// column. A line breaks after its last space that fits, else mid-word. A cursor after a full
/// row goes to the start of the next, which may be one past the last.
fn input_rows(app: &App, width: usize) -> (Vec<std::ops::Range<usize>>, (usize, usize)) {
    let (input, width) = (app.input.as_str(), width.max(1));
    let mut rows = Vec::new();
    let mut start = 0;
    for line in input.split('\n') {
        let end = start + line.len();
        let mut at = start;
        while let Some((cut, _)) = input[at..end].char_indices().nth(width) {
            let len = input[at..at + cut].rfind(' ').map_or(cut, |space| space + 1);
            rows.push(at..at + len);
            at += len;
        }
        rows.push(at..end);
        start = end + 1;
    }
    let cursor = app.cursor();
    let row = rows.iter().rposition(|row| row.start <= cursor).unwrap_or(0);
    let col = input[rows[row].start..cursor].chars().count();
    let cursor = match col >= width {
        true => (row + 1, 0),
        false => (row, col),
    };
    (rows, cursor)
}

fn to_u16(n: usize) -> u16 {
    u16::try_from(n).unwrap_or(u16::MAX)
}

/// The slash-command popup, just above the input box.
fn draw_completions(frame: &mut Frame, input: Rect, app: &App) {
    let completions = app.completions();
    if completions.is_empty() {
        return;
    }
    // `name args` padded to one column, so every help text starts at the same place.
    let args = |command: &SlashCommand| match command.args {
        "" => String::new(),
        args => format!(" {args}"),
    };
    let column = completions
        .iter()
        .map(|command| command.name.len() + args(command).len() + 2)
        .max()
        .unwrap_or(0);
    let help = completions
        .iter()
        .map(|command| command.help.len())
        .max()
        .unwrap_or(0);
    let height = (completions.len() as u16 + 2).min(input.y);
    let width = input.width.min(to_u16(column + help + 2));
    let area = Rect::new(input.x, input.y - height, width, height);
    let items: Vec<ListItem> = completions
        .iter()
        .map(|command| {
            let pad = column - command.name.len();
            ListItem::new(Line::from(vec![
                Span::styled(command.name, Style::new().bold()),
                Span::raw(format!("{:<pad$}", args(command))),
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

#[cfg(test)]
/// `buffer` as text, one line per row.
pub fn buffer_text(buffer: &ratatui::buffer::Buffer) -> String {
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::super::app::SLASH_COMMANDS;
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

    fn render(app: &App, width: u16, height: u16) -> String {
        let mut terminal = ratatui::Terminal::new(ratatui::backend::TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| draw(frame, app, "r", 0)).unwrap();
        buffer_text(terminal.backend().buffer())
    }

    /// The event loop ticks only while [`App::animating`]: otherwise every tick would draw the
    /// same screen. A swarm whose agents wait (on mail, a model slot) or are idle stays still.
    #[test]
    fn the_screen_changes_between_ticks_only_while_the_app_animates() {
        use crate::runtime::{AgentStatus, SwarmEvent};
        let frames = |app: &App| -> Vec<String> {
            let mut terminal = ratatui::Terminal::new(ratatui::backend::TestBackend::new(80, 12)).unwrap();
            (0..SPINNER.len())
                .map(|tick| {
                    terminal.draw(|frame| draw(frame, app, "r", tick)).unwrap();
                    buffer_text(terminal.backend().buffer())
                })
                .collect()
        };
        let mut app = App::default();
        for (id, parent) in [("a1", USER), ("a2", "a1")] {
            app.apply(SwarmEvent::Spawned {
                id: id.into(),
                parent: parent.into(),
                name: format!("name-{id}"),
                task: "t".into(),
                cwd: "/w".into(),
                branch: None,
            });
            app.apply(SwarmEvent::Status {
                id: id.into(),
                status: AgentStatus::Running,
            });
        }
        let tool = |id: &str, name: &str| SwarmEvent::ToolStarted {
            id: id.into(),
            name: name.into(),
            summary: String::new(),
        };
        // The selected root waits for mail while its child works, then the child waits too.
        app.apply(tool("a1", "wait"));
        assert!(app.animating());
        let shown = frames(&app);
        assert!(
            shown.windows(2).any(|pair| pair[0] != pair[1]),
            "a working agent spins"
        );
        app.apply(SwarmEvent::ModelQueued {
            id: "a2".into(),
            queued: true,
            calls: 8,
            limit: 8,
        });
        let still = |app: &App, case: &str| {
            assert!(!app.animating(), "{case}");
            let shown = frames(app);
            assert!(
                shown.windows(2).all(|pair| pair[0] == pair[1]),
                "{case}\n{}",
                shown.join("\n")
            );
        };
        still(&app, "waiting on mail and a model slot");
        let idle = |id: &str| SwarmEvent::Status {
            id: id.into(),
            status: AgentStatus::Idle,
        };
        app.apply(idle("a2"));
        still(&app, "one idle");
        app.apply(idle("a1"));
        app.selected = 1;
        still(&app, "both idle, the child selected");
        app.status = Some(Default::default());
        assert!(app.animating(), "an open /status popup is refreshed");
    }

    #[test]
    fn a_clipped_tree_row_keeps_its_unseen_marker() {
        let mut app = App::default();
        for (id, name) in [("a1", "root"), ("a2", "lazy-mcp-discovery-and-enable-tools")] {
            app.apply(crate::runtime::SwarmEvent::Spawned {
                id: id.into(),
                parent: if id == "a1" { USER.into() } else { "a1".into() },
                name: name.into(),
                task: "t".into(),
                cwd: "/w".into(),
                branch: Some(format!("swarm/{id}")),
            });
        }
        app.agents.get_mut("a1").unwrap().unseen = false;
        let screen = render(&app, 80, 12);
        let rows: Vec<&str> = screen.lines().collect();
        let tree = usize::from(tree_width(&app, 80)) - 1;
        let row = |y: usize| rows[y].chars().take(tree).collect::<String>();
        assert_eq!(row(3), "  … lazy-mcp-discover •", "{screen}");
        assert!(!row(2).contains('•'), "{screen}");
    }

    #[test]
    fn the_slash_popup_aligns_help_in_one_column() {
        let app = App {
            input: "/".into(),
            ..Default::default()
        };
        let screen = render(&app, 120, 30);
        let columns: HashSet<usize> = SLASH_COMMANDS
            .iter()
            .map(|command| {
                let row = screen
                    .lines()
                    .find(|row| row.contains(command.help))
                    .expect(&screen);
                row.find(command.help).unwrap()
            })
            .collect();
        assert_eq!(columns.len(), 1, "{screen}");
        let longest = SLASH_COMMANDS
            .iter()
            .map(|c| c.help)
            .max_by_key(|h| h.len())
            .unwrap();
        assert!(screen.contains(&format!("{longest}│")), "{screen}");
    }

    #[test]
    fn a_narrow_header_drops_token_totals_before_spend_and_hidden_agents() {
        let mut app = App::default();
        for (id, status) in [("a1", AgentStatus::Running), ("a2", AgentStatus::Cancelled)] {
            app.apply(crate::runtime::SwarmEvent::Spawned {
                id: id.into(),
                parent: USER.into(),
                name: format!("agent-{id}"),
                task: "t".into(),
                cwd: "/w".into(),
                branch: None,
            });
            app.apply(crate::runtime::SwarmEvent::Status {
                id: id.into(),
                status,
            });
        }
        app.agents.get_mut("a1").unwrap().input_tokens = 14_100_000;
        app.budget = Some(50_000_000);
        let header = |width: u16| {
            render(&app, width, 10)
                .lines()
                .next()
                .unwrap()
                .trim_end()
                .to_owned()
        };
        assert_eq!(
            header(90),
            " strands-swarm   run r · 2 agents · 1 running · budget 50.0M · 1 cancelled hidden (f3)"
        );
        let wide = header(160);
        assert!(
            wide.contains("· in 14.1M (0% cached) · out 0 · budget 50.0M · 1 cancelled hidden (f3)"),
            "{wide}"
        );
    }

    #[test]
    fn a_running_tool_call_shows_once_above_its_spinner() {
        let mut app = App::default();
        app.apply(crate::runtime::SwarmEvent::Spawned {
            id: "a1".into(),
            parent: USER.into(),
            name: "root".into(),
            task: "t".into(),
            cwd: "/w".into(),
            branch: None,
        });
        app.apply(crate::runtime::SwarmEvent::Status {
            id: "a1".into(),
            status: AgentStatus::Running,
        });
        app.apply(crate::runtime::SwarmEvent::ToolStarted {
            id: "a1".into(),
            name: "wait".into(),
            summary: "{\"from\":[\"parser\"]}".into(),
        });
        let screen = render(&app, 80, 12);
        assert_eq!(screen.matches("parser").count(), 1, "{screen}");
        assert!(screen.contains("running wait…"), "{screen}");
    }

    #[test]
    fn the_keys_popup_fits_every_help_text() {
        let app = App {
            show_keys: true,
            ..Default::default()
        };
        let screen = render(&app, 100, 30);
        for (_, help) in KEYS {
            assert!(screen.contains(help), "{help}:\n{screen}");
        }
    }

    #[test]
    fn every_overlay_renders_on_a_tiny_terminal() {
        let mut app = app_with_an_image(None);
        app.input = "/".into();
        app.show_keys = true;
        app.status = Some(crate::status::StatusReport::default());
        app.cost = Some(crate::cost::CostReport {
            agents: vec![],
            prices: crate::cost::Prices::default(),
        });
        for (width, height) in [(1, 1), (5, 3), (20, 5), (40, 10)] {
            render(&app, width, height);
        }
    }

    #[test]
    fn a_long_popup_scrolls_until_its_last_line_shows() {
        let mut app = app_with_an_image(None);
        let agent = |n: usize| crate::status::AgentReport {
            id: format!("a{n}"),
            label: format!("agent-{n} (a{n})"),
            status: AgentStatus::Idle,
            unread: 0,
            branch: None,
            git: None,
            quiet: None,
            todos: vec![],
            background: vec![],
            schedules: vec![],
        };
        app.status = Some(crate::status::StatusReport {
            limits: None,
            base: None,
            agents: (1..=40).map(agent).collect(),
        });
        let size = Rect::new(0, 0, 80, 24);
        let limit = popup_scroll_limit(&app, size);
        assert!(limit > 0);
        app.status_scroll = limit;
        let screen = render(&app, 80, 24);
        let last = screen
            .lines()
            .rfind(|row| row.contains("agent-"))
            .unwrap_or_default();
        assert!(last.contains("agent-40 (a40)"), "{screen}");
        let below = screen
            .lines()
            .skip_while(|row| !row.contains("agent-40 (a40)"))
            .nth(1);
        assert!(
            below.unwrap_or_default().contains('└'),
            "nothing blank after the end:\n{screen}"
        );
        app.status = None;
        assert_eq!(popup_scroll_limit(&app, size), 0);
    }

    #[test]
    fn a_tool_line_breaks_a_long_summary_instead_of_orphaning_its_name() {
        let mut app = app_with_an_image(None);
        let path = "/home/user/workplace/project.swarm/self-improve/a19/strands-rs/crates/view.rs";
        app.agents.get_mut("a1").unwrap().log.push_back(LogEntry::Tool {
            name: "read".into(),
            summary: path.into(),
            ok: Some(false),
        });
        let screen = render(&app, 80, 24);
        let row = screen.lines().find(|row| row.contains("✗ read")).expect(&screen);
        assert!(row.contains("✗ read /home/user/"), "{screen}");
        assert!(!screen.contains('\u{a0}'), "the glue renders as spaces");
        let plain = plain_log(&app.agents["a1"]);
        assert!(plain.contains(&format!("✗ read {path}")), "{plain}");
    }

    #[test]
    fn the_status_popup_lists_agents_over_the_log_and_wraps_long_lines() {
        let mut app = app_with_an_image(None);
        let subject = "abc1234 feat: thumbnails for every image in the log pane, scaled to fit the cell grid";
        app.status = Some(crate::status::StatusReport {
            limits: None,
            base: Some("main".into()),
            agents: vec![crate::status::AgentReport {
                id: "a1".into(),
                label: "shots (a1)".into(),
                status: AgentStatus::Idle,
                unread: 0,
                branch: Some("swarm/r/a1".into()),
                git: Some(crate::status::BranchState {
                    ahead: vec![subject.into()],
                    ..Default::default()
                }),
                quiet: None,
                todos: vec![],
                background: vec![],
                schedules: vec![],
            }],
        });
        let wide = render(&app, 100, 20);
        for expected in ["shots (a1) · idle", "abc1234 feat: thumbnails", "1 not on main"] {
            assert!(wide.contains(expected), "{expected}:\n{wide}");
        }
        let narrow = render(&app, 60, 20);
        assert!(narrow.contains("scaled to fit the cell grid"), "{narrow}");
        let tiny = render(&app, 30, 20);
        assert!(tiny.contains("shots (a1) · idle"), "{tiny}");
    }

    #[test]
    fn the_cost_popup_and_header_show_labelled_estimates() {
        let mut app = app_with_an_image(None);
        let mut ledger = crate::cost::Ledger::default();
        let counts = crate::cost::Counts {
            calls: 1,
            input: 2_000_000,
            output: 0,
            cache_read: 0,
            cache_write: 0,
        };
        ledger.record("claude-sonnet-4-5", counts, true);
        let report = crate::cost::CostReport {
            agents: vec![crate::cost::AgentCost {
                id: "a1".into(),
                parent: crate::runtime::USER.into(),
                label: "shots (a1)".into(),
                totals: counts,
                ledger,
                fallback_model: None,
            }],
            prices: crate::cost::Prices::default(),
        };
        app.spend = Some(report.total());
        app.cost = Some(report);
        let screen = render(&app, 120, 20);
        assert!(screen.contains("est. $6.00"), "{screen}");
        assert!(screen.contains("estimated cost"), "{screen}");
        assert!(screen.contains("Estimated cost: $6.00"), "{screen}");
        assert!(screen.contains("shots (a1)"), "{screen}");
        assert!(screen.contains("claude-sonnet-4-5"), "{screen}");

        // An 80-column terminal clips the table, but not the estimate.
        let screen = render(&app, 80, 20);
        let row = screen
            .lines()
            .find(|row| row.contains("│shots (a1)"))
            .expect(&screen);
        assert!(row.contains("$6.00"), "{screen}");
        // Its intro and footnote wrap instead of running off the edge.
        let flat = screen
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .replace("│ │", "");
        for prose in [
            "write 1.25× input; not a bill)",
            "--prices FILE adds or overrides prices.",
        ] {
            assert!(flat.contains(prose), "{prose}:\n{screen}");
        }
    }

    #[test]
    fn an_image_shows_as_a_thumbnail_under_its_placeholder_but_copies_as_text() {
        let app = app_with_an_image(Some(ratatui_image::picker::Picker::halfblocks()));
        let screen = render(&app, 100, 30);
        assert!(screen.contains("[image: /w/shot.png 40x30]"), "{screen}");
        let halfblock = |text: &str| text.contains('▀') || text.contains('▄');
        assert!(halfblock(&screen), "no halfblock thumbnail:\n{screen}");

        let copy = buffer_text(&log_text(&app, Rect::new(0, 0, 100, 30)));
        assert!(copy.contains("[image: /w/shot.png 40x30]"), "{copy}");
        assert!(!halfblock(&copy), "{copy}");
        assert!(plain_log(&app.agents["a1"]).contains("[image: /w/shot.png 40x30]\n"));
    }

    #[test]
    fn without_a_graphics_protocol_an_image_is_only_its_placeholder() {
        let mut app = app_with_an_image(None);
        // A terminal (or tmux) that only manages halfblocks gets the placeholder line alone.
        app.images =
            super::super::images::Images::for_terminal(Some(ratatui_image::picker::Picker::halfblocks()));
        app.apply(crate::runtime::SwarmEvent::Image {
            id: "a1".into(),
            source: "/w/diagram.png".into(),
            bytes: std::sync::Arc::new(super::super::images::png(1024, 768)),
        });
        let screen = render(&app, 100, 30);
        assert!(screen.contains("[image: /w/shot.png 40x30]"), "{screen}");
        assert!(screen.contains("[image: /w/diagram.png 1024x768]"), "{screen}");
        assert!(!screen.contains('▀') && !screen.contains('▄'), "{screen}");
        assert_eq!(
            LogEntry::placeholder("x.png", None),
            "[image: x.png, not decodable]"
        );
    }

    #[test]
    fn a_pending_reload_shows_in_the_header() {
        let mut app = App::default();
        app.apply(crate::runtime::SwarmEvent::ReloadPending { tool_calls: 2 });
        let screen = render(&app, 140, 10);
        assert!(
            screen.contains("reload pending (waiting for 2 tool calls)"),
            "{screen}"
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
        let app = App {
            input: "/c".into(),
            ..Default::default()
        };
        let screen = render(&app, 100, 20);
        assert!(screen.contains("/cancel"), "{screen}");
        assert!(!screen.contains("/model"), "{screen}");
    }

    #[test]
    fn the_input_grows_with_its_lines_and_shows_the_cursor_where_it_is() {
        let mut terminal = ratatui::Terminal::new(ratatui::backend::TestBackend::new(40, 20)).unwrap();
        let mut app = App {
            input: "first line\nsecond".into(),
            cursor: Some(3),
            ..Default::default()
        };
        terminal.draw(|frame| draw(frame, &app, "run", 0)).unwrap();
        let input = areas(&app, Rect::new(0, 0, 40, 20)).input;
        assert_eq!(input.height, 3, "a border and two lines");
        let cursor = terminal.get_cursor_position().unwrap();
        assert_eq!((cursor.x, cursor.y), (3, input.y + 1));

        // A long line wraps, growing the box, with the cursor after its last character.
        app.input = "x".repeat(100);
        app.cursor = None;
        terminal.draw(|frame| draw(frame, &app, "run", 0)).unwrap();
        let input = areas(&app, Rect::new(0, 0, 40, 20)).input;
        assert_eq!(input.height, 4, "a border and rows of 40, 40 and 20");
        let cursor = terminal.get_cursor_position().unwrap();
        assert_eq!((cursor.x, cursor.y), (20, input.y + 3));
        // After a full row, the cursor starts the next.
        app.input = "x".repeat(80);
        terminal.draw(|frame| draw(frame, &app, "run", 0)).unwrap();
        let input = areas(&app, Rect::new(0, 0, 40, 20)).input;
        assert_eq!(input.height, 4);
        let cursor = terminal.get_cursor_position().unwrap();
        assert_eq!((cursor.x, cursor.y), (0, input.y + 3));
        // Past the most rows, the box scrolls to keep the cursor in view.
        app.input = "x".repeat(400);
        app.cursor = Some(45);
        terminal.draw(|frame| draw(frame, &app, "run", 0)).unwrap();
        let input = areas(&app, Rect::new(0, 0, 40, 20)).input;
        assert_eq!(input.height, 1 + MAX_INPUT_LINES as u16);
        let cursor = terminal.get_cursor_position().unwrap();
        assert_eq!((cursor.x, cursor.y), (5, input.y + 2));
        app.cursor = None;
        terminal.draw(|frame| draw(frame, &app, "run", 0)).unwrap();
        let cursor = terminal.get_cursor_position().unwrap();
        assert_eq!((cursor.x, cursor.y), (0, input.bottom() - 1));
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
        let item = agent_item(0, view, 0).to_string();
        assert!(item.contains("parser opus-5-5 1.2M/34.0k 34% a2"), "{item}");
        assert_eq!(
            log_title(view),
            " parser (a2) · starting · opus-5-5 · 1.2M/34.0k tok (97% cached) · 34% · …/a2/repo "
        );
    }

    /// An app whose agent `a1` has a log of `entries` entries: wrapped text, tool calls and mail.
    fn app_with_a_long_log(entries: usize) -> App {
        let mut app = App::default();
        app.apply(crate::runtime::SwarmEvent::Spawned {
            id: "a1".into(),
            parent: USER.into(),
            name: "root".into(),
            task: "t".into(),
            cwd: "/w".into(),
            branch: None,
        });
        let log = &mut app.agents.get_mut("a1").unwrap().log;
        for i in 0..entries {
            log.push_back(match i % 3 {
                0 => LogEntry::Text(format!("{i} {}", "words that wrap ".repeat(i % 9))),
                1 => LogEntry::Tool {
                    name: "shell".into(),
                    summary: format!("cargo test {i}"),
                    ok: Some(true),
                },
                _ => mail("p1 (a2)"),
            });
        }
        app
    }

    #[test]
    fn the_log_pane_shows_the_same_rows_as_the_whole_log_would() {
        let app = app_with_a_long_log(300);
        let area = Rect::new(0, 0, 40, 12);
        let inner = log_inner(area);
        let view = &app.agents["a1"];
        let mut all: Vec<Line> = view.log.iter().flat_map(entry_lines).collect();
        let activity = view.activity().expect("a starting agent shows its activity");
        all.push(Line::styled(
            format!("{} {activity}", SPINNER[0]),
            Style::new().fg(Color::Cyan),
        ));
        let whole = Paragraph::new(Text::from(all.clone())).wrap(Wrap { trim: false });
        let rows = whole.line_count(inner.width);
        for scroll in [0, 1, 7, 37, 200, u16::MAX] {
            let app = App {
                scroll,
                ..app_with_a_long_log(300)
            };
            let window = log_window(&app.agents["a1"], &app, 0, inner);
            let mut shown = ratatui::buffer::Buffer::empty(inner);
            let windowed = Paragraph::new(Text::from(window.lines.clone()))
                .wrap(Wrap { trim: false })
                .scroll((window.offset, 0));
            ratatui::widgets::Widget::render(windowed, inner, &mut shown);
            let mut expected = ratatui::buffer::Buffer::empty(inner);
            let offset = log_offset(&app, rows, inner.height);
            let scrolled = whole.clone().scroll((offset, 0));
            ratatui::widgets::Widget::render(scrolled, inner, &mut expected);
            assert_eq!(buffer_text(&shown), buffer_text(&expected), "scrolled {scroll}");
        }
    }

    #[test]
    fn a_frame_lays_out_only_the_end_of_a_long_log() {
        let app = app_with_a_long_log(super::super::app::MAX_LOG_ENTRIES);
        let inner = log_inner(Rect::new(0, 0, 40, 12));
        let window = log_window(&app.agents["a1"], &app, 0, inner);
        assert!(window.lines.len() < 40, "{} lines laid out", window.lines.len());
        assert!(window.rows.iter().sum::<usize>() >= usize::from(inner.height));
    }
}
