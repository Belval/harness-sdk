//! Mouse selection inside one pane (the log or the agent tree) and copying it to the system
//! clipboard with an OSC 52 escape, so a drag never picks up the other pane.

use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect};
use ratatui::style::Modifier;

/// A pane that can be selected in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pane {
    /// The agent tree.
    Tree,
    /// The selected agent's log.
    Log,
}

/// A drag from `anchor` to `cursor` (screen cells) inside `pane`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selection {
    /// Where the drag started.
    pub pane: Pane,
    /// First cell.
    pub anchor: Position,
    /// Current cell, clamped to the pane.
    pub cursor: Position,
}

impl Selection {
    /// Start and end in reading order.
    fn ordered(&self) -> (Position, Position) {
        let key = |p: Position| (p.y, p.x);
        match key(self.anchor) <= key(self.cursor) {
            true => (self.anchor, self.cursor),
            false => (self.cursor, self.anchor),
        }
    }

    /// The selected columns of row `y` in `area`, as an inclusive range.
    fn columns(&self, area: Rect, y: u16) -> Option<(u16, u16)> {
        let (start, end) = self.ordered();
        if y < start.y || y > end.y || area.width == 0 {
            return None;
        }
        let first = if y == start.y { start.x } else { area.x };
        let last = if y == end.y { end.x } else { area.right() - 1 };
        Some((first, last))
    }

    /// Whether it covers more than the cell it started on.
    pub fn is_empty(&self) -> bool {
        self.anchor == self.cursor
    }
}

/// `position` moved inside `area`.
pub fn clamp(position: Position, area: Rect) -> Position {
    Position {
        x: position
            .x
            .clamp(area.x, area.right().saturating_sub(1).max(area.x)),
        y: position
            .y
            .clamp(area.y, area.bottom().saturating_sub(1).max(area.y)),
    }
}

/// Show `selection` reversed.
pub fn highlight(buffer: &mut Buffer, area: Rect, selection: &Selection) {
    for y in area.top()..area.bottom() {
        let Some((first, last)) = selection.columns(area, y) else {
            continue;
        };
        for x in first..=last {
            if let Some(cell) = buffer.cell_mut((x, y)) {
                cell.modifier.toggle(Modifier::REVERSED);
            }
        }
    }
}

/// The selected text of `area` in `buffer`: trailing spaces trimmed, one line per row except that
/// rows marked in `continues` (indexed from `area.y`) join the row above, as wrapped rows do.
pub fn selected_text(buffer: &Buffer, area: Rect, selection: &Selection, continues: &[bool]) -> String {
    let mut out = String::new();
    let mut previous_full = false;
    for y in area.top()..area.bottom() {
        let Some((first, last)) = selection.columns(area, y) else {
            continue;
        };
        let row: String = (first..=last)
            .filter_map(|x| buffer.cell((x, y)).map(|cell| cell.symbol()))
            .collect();
        let row = row.trim_end();
        let continued = continues.get(usize::from(y - area.y)).copied().unwrap_or(false);
        if y > selection.ordered().0.y {
            // A word-wrapped row lost the space it broke at, unless the word itself was split.
            out.push_str(match (continued, previous_full) {
                (true, true) => "",
                (true, false) => " ",
                (false, _) => "\n",
            });
        }
        previous_full = last == area.right() - 1 && row.chars().count() == usize::from(last - first + 1);
        out.push_str(row);
    }
    out
}

/// The OSC 52 escape that puts `text` on the system clipboard (terminals, ssh, tmux with
/// `set-clipboard on`).
pub fn osc52(text: &str) -> String {
    format!("\x1b]52;c;{}\x07", base64(text.as_bytes()))
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, b)| n | u32::from(*b) << (16 - 8 * i));
        for i in 0..4 {
            match i <= chunk.len() {
                true => out.push(char::from(ALPHABET[(n >> (18 - 6 * i) & 63) as usize])),
                false => out.push('='),
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_the_rfc_vectors() {
        let cases = [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ];
        for (plain, encoded) in cases {
            assert_eq!(base64(plain.as_bytes()), encoded);
        }
        assert_eq!(osc52("hi"), "\x1b]52;c;aGk=\x07");
    }

    #[test]
    fn clamps_into_the_pane() {
        let area = Rect::new(10, 2, 5, 3);
        assert_eq!(clamp(Position::new(0, 0), area), Position::new(10, 2));
        assert_eq!(clamp(Position::new(99, 99), area), Position::new(14, 4));
        assert_eq!(clamp(Position::new(12, 3), area), Position::new(12, 3));
    }
}
