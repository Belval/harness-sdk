//! Line editing for the input box: a cursor, word motions and deletions, newlines for multi-line
//! messages, and a history of sent lines. The text stays a plain `String` ([`App::input`]); the
//! cursor is a byte offset into it, `None` meaning "at the end", so code that replaces the text
//! wholesale leaves the cursor at its end.

use super::app::App;

/// Most sent lines kept for ↑/↓.
const MAX_HISTORY: usize = 100;

impl App {
    /// Byte offset of the cursor, clamped to the text and to a char boundary.
    pub fn cursor(&self) -> usize {
        let mut at = self.cursor.unwrap_or(self.input.len()).min(self.input.len());
        while !self.input.is_char_boundary(at) {
            at -= 1;
        }
        at
    }

    fn set_cursor(&mut self, at: usize) {
        self.cursor = Some(at);
    }

    /// The edit changed the text: the popup and history browsing start over.
    fn edited(&mut self) {
        self.completion = 0;
        self.history_at = None;
    }

    /// Insert `c` at the cursor.
    pub fn insert(&mut self, c: char) {
        let at = self.cursor();
        self.input.insert(at, c);
        self.set_cursor(at + c.len_utf8());
        self.edited();
    }

    /// Delete the char before the cursor.
    pub fn backspace(&mut self) {
        let at = self.cursor();
        if let Some(c) = self.input[..at].chars().next_back() {
            self.input.replace_range(at - c.len_utf8()..at, "");
            self.set_cursor(at - c.len_utf8());
            self.edited();
        }
    }

    /// Delete the char under the cursor.
    pub fn delete(&mut self) {
        let at = self.cursor();
        if let Some(c) = self.input[at..].chars().next() {
            self.input.replace_range(at..at + c.len_utf8(), "");
            self.set_cursor(at);
            self.edited();
        }
    }

    /// Move the cursor one char left (`-1`) or right (`1`).
    pub fn move_char(&mut self, delta: isize) {
        let at = self.cursor();
        let next = match delta < 0 {
            true => self.input[..at].chars().next_back().map(|c| at - c.len_utf8()),
            false => self.input[at..].chars().next().map(|c| at + c.len_utf8()),
        };
        self.set_cursor(next.unwrap_or(at));
    }

    /// Move to the start of the previous word (`-1`) or past the end of the next one (`1`).
    pub fn move_word(&mut self, delta: isize) {
        let at = self.cursor();
        let target = match delta < 0 {
            true => word_start(&self.input, at),
            false => word_end(&self.input, at),
        };
        self.set_cursor(target);
    }

    /// Move to the start of the current line.
    pub fn home(&mut self) {
        let at = self.cursor();
        self.set_cursor(line_start(&self.input, at));
    }

    /// Move to the end of the current line.
    pub fn end(&mut self) {
        let at = self.cursor();
        let end = self.input[at..].find('\n').map_or(self.input.len(), |i| at + i);
        self.set_cursor(end);
    }

    /// Delete the word before the cursor (Ctrl-W).
    pub fn delete_word(&mut self) {
        let at = self.cursor();
        let start = word_start(&self.input, at);
        self.cut(start, at);
    }

    /// Delete from the start of the line to the cursor (Ctrl-U).
    pub fn delete_to_line_start(&mut self) {
        let at = self.cursor();
        let start = line_start(&self.input, at);
        // At a line start, join with the previous line instead of doing nothing.
        let start = match start == at && at > 0 {
            true => at - 1,
            false => start,
        };
        self.cut(start, at);
    }

    fn cut(&mut self, start: usize, end: usize) {
        if start < end {
            self.input.replace_range(start..end, "");
            self.set_cursor(start);
            self.edited();
        }
    }

    /// Empty the input.
    pub fn clear_input(&mut self) {
        self.input.clear();
        self.cursor = None;
        self.edited();
    }

    /// Take the input to send it, remembering it for ↑/↓.
    pub fn take_input(&mut self) -> String {
        let line = std::mem::take(&mut self.input);
        self.cursor = None;
        self.edited();
        if !line.trim().is_empty() && self.history.last() != Some(&line) {
            self.history.push(line.clone());
            if self.history.len() > MAX_HISTORY {
                self.history.remove(0);
            }
        }
        line
    }

    /// Whether ↑/↓ should walk the history rather than select agents: while browsing it, or
    /// when there is text in the input.
    pub fn wants_history(&self) -> bool {
        self.history_at.is_some() || !self.input.is_empty()
    }

    /// Show an older (`-1`) or newer (`1`) sent line. Newer than the newest returns to what was
    /// being typed before browsing started.
    pub fn browse_history(&mut self, delta: isize) {
        if self.history.is_empty() {
            return;
        }
        let newest = self.history.len();
        let current = self.history_at.unwrap_or(newest);
        if self.history_at.is_none() {
            if delta > 0 {
                return;
            }
            self.draft = self.input.clone();
        }
        let next = current.saturating_add_signed(delta).min(newest);
        self.input = match next == newest {
            true => std::mem::take(&mut self.draft),
            false => self.history[next].clone(),
        };
        self.history_at = (next < newest).then_some(next);
        self.cursor = None;
        self.completion = 0;
    }

    /// `(line, column)` of the cursor, in chars, for drawing.
    pub fn cursor_line_col(&self) -> (usize, usize) {
        let before = &self.input[..self.cursor()];
        let line = before.matches('\n').count();
        let col = before.rsplit('\n').next().unwrap_or_default().chars().count();
        (line, col)
    }
}

fn line_start(text: &str, at: usize) -> usize {
    text[..at].rfind('\n').map_or(0, |i| i + 1)
}

/// Start of the word before `at`, skipping the whitespace in between.
fn word_start(text: &str, at: usize) -> usize {
    let before = &text[..at];
    let trimmed = before.trim_end();
    trimmed
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_whitespace())
        .map_or(0, |(i, c)| i + c.len_utf8())
}

/// End of the word after `at`, skipping the whitespace in between.
fn word_end(text: &str, at: usize) -> usize {
    let after = &text[at..];
    let skipped = after.len() - after.trim_start().len();
    let word = &after[skipped..];
    let len = word.find(char::is_whitespace).unwrap_or(word.len());
    at + skipped + len
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app(text: &str) -> App {
        App {
            input: text.into(),
            ..Default::default()
        }
    }

    /// The text with `|` at the cursor.
    fn shown(app: &App) -> String {
        let mut text = app.input.clone();
        text.insert(app.cursor(), '|');
        text
    }

    #[test]
    fn inserts_and_deletes_at_the_cursor() {
        let mut app = app("helo");
        app.move_char(-1);
        app.insert('l');
        assert_eq!(shown(&app), "hell|o");
        app.delete();
        assert_eq!(shown(&app), "hell|");
        app.home();
        app.delete();
        app.backspace();
        assert_eq!(shown(&app), "|ell");
        app.end();
        app.backspace();
        assert_eq!(shown(&app), "el|");
        // Multi-byte chars move and delete whole.
        let mut app = app_with_cursor("añb", 3);
        app.backspace();
        assert_eq!(shown(&app), "a|b");
    }

    fn app_with_cursor(text: &str, at: usize) -> App {
        let mut app = app(text);
        app.cursor = Some(at);
        app
    }

    #[test]
    fn word_motions_and_deletions() {
        let mut app = app("fix the  parser now");
        app.move_word(-1);
        assert_eq!(shown(&app), "fix the  parser |now");
        app.move_word(-1);
        app.move_word(-1);
        assert_eq!(shown(&app), "fix |the  parser now");
        app.move_word(1);
        assert_eq!(shown(&app), "fix the|  parser now");
        app.move_word(1);
        assert_eq!(shown(&app), "fix the  parser| now");
        app.delete_word();
        assert_eq!(shown(&app), "fix the  | now");
        app.delete_word();
        assert_eq!(shown(&app), "fix | now");
        app.delete_to_line_start();
        assert_eq!(shown(&app), "| now");
    }

    #[test]
    fn lines_are_edited_one_at_a_time() {
        let mut app = app("first\nsecond");
        app.home();
        assert_eq!(shown(&app), "first\n|second");
        assert_eq!(app.cursor_line_col(), (1, 0));
        app.move_char(-1);
        assert_eq!(app.cursor_line_col(), (0, 5));
        app.home();
        app.end();
        assert_eq!(shown(&app), "first|\nsecond");
        app.move_char(1);
        app.move_char(1);
        app.move_char(1);
        app.delete_to_line_start();
        assert_eq!(shown(&app), "first\n|cond");
        app.delete_to_line_start();
        assert_eq!(shown(&app), "first|cond", "at a line start, joins the lines");
    }

    #[test]
    fn history_walks_sent_lines_and_returns_to_the_draft() {
        let mut app = App::default();
        for line in ["one", "two", "two", "  "] {
            app.input = line.into();
            app.take_input();
        }
        assert_eq!(app.history, ["one", "two"], "no repeats or blanks");
        assert!(!app.wants_history());
        app.input = "dra".into();
        assert!(app.wants_history());
        app.browse_history(-1);
        assert_eq!(app.input, "two");
        app.browse_history(-1);
        app.browse_history(-1);
        assert_eq!(app.input, "one", "stops at the oldest");
        app.browse_history(1);
        assert_eq!(app.input, "two");
        app.browse_history(1);
        assert_eq!(shown(&app), "dra|", "back to what was typed");
        assert!(app.history_at.is_none());
        app.browse_history(1);
        assert_eq!(app.input, "dra", "nothing newer than the draft");
        // Typing while browsing keeps the recalled text and ends browsing.
        app.browse_history(-1);
        app.insert('!');
        assert_eq!(app.input, "two!");
        assert!(app.history_at.is_none());
    }
}
