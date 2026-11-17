//! The hint `edit` adds when `old_str` matches nothing: why, and where the closest text is.
//!
//! Models most often miss by JSON escapes copied from a tool result (`\"` for `"`, `\n` for a
//! line break), by whitespace (indentation, trailing spaces), or by a line that changed since they
//! read it. Each gets its own line, with the file's text numbered as `read` shows it.
//!
//! The work is bounded for any file: the whitespace check compares lazily, and the closest-text
//! search compares only `old_str`'s first [`COMPARED_LINES`] non-blank lines, skips lines whose
//! length rules them out, and gives up after [`MAX_CELLS`] edit-distance cells.

/// Lines of the file shown around a near match, at most.
const MAX_SHOWN: usize = 8;
/// Below this similarity (0–1) a line is not worth showing as the closest one.
const MIN_SIMILARITY: f64 = 0.5;
/// Characters of a line compared for similarity; longer lines are cut.
const COMPARED_CHARS: usize = 200;
/// Lines of `old_str` (the first non-blank ones) looked for in the file.
const COMPARED_LINES: usize = 3;
/// Edit-distance cells computed at most per hint, so a huge file cannot stall the tool.
const MAX_CELLS: usize = 2_000_000;

/// What to tell the model when `old_str` does not occur in `content`.
pub(super) fn no_match_hint(content: &str, old_str: &str) -> String {
    let unescaped = unescape(old_str);
    if unescaped != old_str && content.contains(&unescaped) {
        return "old_str holds escapes (\\\" or \\n) where the file has the literal characters; \
                send the quotes and line breaks themselves."
            .to_owned();
    }
    let file: Vec<&str> = content.lines().collect();
    let wanted: Vec<&str> = old_str.lines().collect();
    if let Some(start) = whitespace_match(&file, &wanted) {
        return format!(
            "Lines {}-{} match except for whitespace (indentation, tabs, trailing spaces):\n{}",
            start + 1,
            start + wanted.len(),
            numbered(&file, start, wanted.len())
        );
    }
    let note = "Check whitespace and escaping: old_str must hold the file's literal characters.";
    match closest_line(&file, &wanted) {
        Closest::Found(at, offset) => {
            let start = at.saturating_sub(offset);
            format!(
                "Closest text in the file (it may have changed since you read it):\n{}\n{note}",
                numbered(&file, start, wanted.len().max(1))
            )
        }
        Closest::Nothing => {
            format!("No line resembles it; read the file for its current content. {note}")
        }
        Closest::TooLarge => format!(
            "The file is too large to search for the closest text; read the part you meant to \
             edit. {note}"
        ),
    }
}

/// `old_str` with the JSON string escapes a model copies from a tool result turned back into
/// characters.
fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some(other) => out.extend(['\\', other]),
            None => out.push('\\'),
        }
    }
    out
}

/// The first window of `file` whose lines equal `wanted`'s once all whitespace is removed.
fn whitespace_match(file: &[&str], wanted: &[&str]) -> Option<usize> {
    let squash = |line: &str| -> Vec<u8> { line.bytes().filter(|b| !b.is_ascii_whitespace()).collect() };
    let wanted: Vec<Vec<u8>> = wanted.iter().map(|line| squash(line)).collect();
    if wanted.iter().all(Vec::is_empty) || wanted.len() > file.len() {
        return None;
    }
    // Compared lazily: most windows fail on their first line's first bytes.
    let same = |line: &str, squashed: &[u8]| {
        let bytes = line.bytes().filter(|b| !b.is_ascii_whitespace());
        bytes.eq(squashed.iter().copied())
    };
    (0..=file.len() - wanted.len()).find(|&start| {
        file[start..]
            .iter()
            .zip(&wanted)
            .all(|(line, squashed)| same(line, squashed))
    })
}

/// Where [`closest_line`] found the text most like `old_str`.
#[derive(Debug, PartialEq)]
enum Closest {
    /// The file line's index, and the index of the `old_str` line it resembles.
    Found(usize, usize),
    /// No line is at least [`MIN_SIMILARITY`] alike.
    Nothing,
    /// The search would take more than [`MAX_CELLS`].
    TooLarge,
}

/// The file line most like one of `wanted`'s first [`COMPARED_LINES`] non-blank lines, compared
/// as bytes (trimmed, at most [`COMPARED_CHARS`]).
fn closest_line(file: &[&str], wanted: &[&str]) -> Closest {
    let bytes = |line: &str| -> Vec<u8> {
        let line = line.trim().as_bytes();
        line[..line.len().min(COMPARED_CHARS)].to_vec()
    };
    let wanted: Vec<(usize, Vec<u8>)> = (wanted.iter().enumerate())
        .filter(|(_, line)| !line.trim().is_empty())
        .take(COMPARED_LINES)
        .map(|(offset, line)| (offset, bytes(line)))
        .collect();
    let mut cells = 0;
    let mut row = [0u16; COMPARED_CHARS + 1];
    let mut best: Option<(f64, usize, usize)> = None;
    for (offset, line) in &wanted {
        for (at, candidate) in file.iter().enumerate() {
            let candidate = candidate.trim().as_bytes();
            let candidate = &candidate[..candidate.len().min(COMPARED_CHARS)];
            // The distance is at least the length difference, so similarity is at most
            // short / long: skip lines that cannot reach MIN_SIMILARITY or beat the best so far.
            let (short, long) = (line.len().min(candidate.len()), line.len().max(candidate.len()));
            let floor = best.map_or(MIN_SIMILARITY, |(top, ..)| top.max(MIN_SIMILARITY));
            if long == 0 || (short as f64) < floor * long as f64 {
                continue;
            }
            cells += line.len() * candidate.len();
            if cells > MAX_CELLS {
                return Closest::TooLarge;
            }
            let score = similarity(line, candidate, &mut row);
            if best.is_none_or(|(top, ..)| score > top) {
                best = Some((score, at, *offset));
            }
        }
    }
    match best {
        Some((score, at, offset)) if score >= MIN_SIMILARITY => Closest::Found(at, offset),
        _ => Closest::Nothing,
    }
}

/// 1 − edit distance / longer length, both at most [`COMPARED_CHARS`] long; `row` is scratch
/// for the one row of the distance table kept at a time.
fn similarity(a: &[u8], b: &[u8], row: &mut [u16; COMPARED_CHARS + 1]) -> f64 {
    let longest = a.len().max(b.len());
    if longest == 0 {
        return 1.0;
    }
    for (j, cell) in row.iter_mut().enumerate().take(b.len() + 1) {
        *cell = j as u16;
    }
    for (i, &ca) in a.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i as u16 + 1;
        for (j, &cb) in b.iter().enumerate() {
            let mut cell = diagonal + u16::from(ca != cb);
            diagonal = row[j + 1];
            if row[j] + 1 < cell {
                cell = row[j] + 1;
            }
            if diagonal + 1 < cell {
                cell = diagonal + 1;
            }
            row[j + 1] = cell;
        }
    }
    1.0 - f64::from(row[b.len()]) / longest as f64
}

/// `count` lines of `file` from `start`, at most [`MAX_SHOWN`], numbered like `read`.
fn numbered(file: &[&str], start: usize, count: usize) -> String {
    let end = (start + count.min(MAX_SHOWN)).min(file.len());
    (start..end)
        .map(|i| format!("{:>6}\t{}", i + 1, file[i]))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = "def f():\n    return \"a b\"\n\n\ndef g(x):\n    return x + 1\n";

    #[test]
    fn json_escapes_are_named() {
        let hint = no_match_hint(FILE, "return \\\"a b\\\"");
        assert!(hint.contains("escapes"), "{hint}");
    }

    #[test]
    fn a_whitespace_difference_shows_the_matching_lines() {
        let hint = no_match_hint(FILE, "def g(x):\n  return x+1");
        assert!(
            hint.starts_with("Lines 5-6 match except for whitespace"),
            "{hint}"
        );
        assert!(
            hint.contains("     5\tdef g(x):\n     6\t    return x + 1"),
            "{hint}"
        );
    }

    #[test]
    fn a_changed_line_shows_the_closest_text_numbered() {
        let hint = no_match_hint(FILE, "def g(y):\n    return y + 2");
        assert!(hint.starts_with("Closest text in the file"), "{hint}");
        assert!(
            hint.contains("     5\tdef g(x):\n     6\t    return x + 1"),
            "{hint}"
        );
        assert!(hint.contains("Check whitespace and escaping"), "{hint}");
    }

    #[test]
    fn the_window_starts_where_old_str_would() {
        // Only the second line of old_str resembles anything: the window starts one line earlier.
        let hint = no_match_hint(FILE, "zzzzzzzz\n    return x + 2");
        assert!(hint.contains("     5\tdef g(x):\n     6\t"), "{hint}");
    }

    #[test]
    fn nothing_alike_says_to_read_the_file() {
        let hint = no_match_hint(FILE, "class Unrelated(Base):");
        assert!(hint.starts_with("No line resembles it"), "{hint}");
    }

    #[test]
    fn similarity_is_one_minus_the_normalized_edit_distance() {
        let row = &mut [0; COMPARED_CHARS + 1];
        assert_eq!(similarity(b"abc", b"abc", row), 1.0);
        assert_eq!(similarity(b"abcd", b"abed", row), 0.75);
        assert_eq!(similarity(b"kitten", b"sitting", row), 1.0 - 3.0 / 7.0);
        assert_eq!(similarity(b"", b"abc", row), 0.0);
    }

    #[test]
    fn a_huge_file_gives_up_on_the_closest_text_quickly() {
        let file: String = (0..50_000)
            .map(|i| format!("    value_{i} = compute(value_{i}, 'some text')\n"))
            .collect();
        let old_str: String = (0..50)
            .map(|i| format!("    valeu_{i} = compote(valeu_{i}, 'sum text')\n"))
            .collect();
        let started = std::time::Instant::now();
        let hint = no_match_hint(&file, &old_str);
        let took = started.elapsed();
        assert!(hint.starts_with("The file is too large"), "{hint}");
        assert!(took < std::time::Duration::from_secs(2), "{took:?}");
    }

    #[test]
    fn lines_of_very_different_length_cost_nothing() {
        // 50k one-character lines are all skipped by length, so the budget is never reached.
        let file = "x\n".repeat(50_000) + "    return compute(value)\n";
        let started = std::time::Instant::now();
        let hint = no_match_hint(&file, "    return compute(valeu)");
        assert!(hint.contains(" 50001\t    return compute(value)"), "{hint}");
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }
}
