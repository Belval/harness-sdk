//! Pattern and line-range search over offloaded text
//! (`strands-ts/src/vended-plugins/context-offloader/search.ts`).

use std::collections::BTreeSet;
use std::sync::LazyLock;

use regex::Regex;

/// A line range, 1-indexed and inclusive.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LineRange {
    pub start: usize,
    pub end: usize,
}

/// What to search for.
#[derive(Debug, Clone, Default)]
pub(crate) struct SearchInput<'a> {
    pub pattern: Option<&'a str>,
    pub line_range: Option<LineRange>,
    pub context_lines: usize,
}

const MAX_PATTERN_LENGTH: usize = 200;
// Best-effort heuristic, kept from TS; does not catch all catastrophic patterns.
static NESTED_QUANTIFIER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[+*]\s*\)\s*[+*]").expect("valid regex"));

/// Cuts output at the last newline before `max_chars` and appends a truncation message.
fn truncate(output: &str, max_chars: usize, message: &str) -> String {
    if output.chars().count() <= max_chars {
        return output.to_owned();
    }
    let limit = output
        .char_indices()
        .nth(max_chars)
        .map_or(output.len(), |(index, _)| index);
    // `lastIndexOf('\n', maxChars)` also accepts a newline at `max_chars` itself.
    let window = &output[..(limit + 1).min(output.len())];
    let end = match window.rfind('\n') {
        Some(cut) if cut > 0 => cut,
        _ => limit,
    };
    format!("{}\n\n[{message}]", &output[..end])
}

/// Formats line indices with line numbers, `>` prefixes for matches, and `---` separators for
/// gaps.
fn format_lines(lines: &[&str], indices: &[usize], matched: &BTreeSet<usize>) -> String {
    let Some(last) = indices.last() else {
        return String::new();
    };
    let width = (last + 1).to_string().len();
    let mut output = Vec::with_capacity(indices.len());
    for (i, &index) in indices.iter().enumerate() {
        if i > 0 && index > indices[i - 1] + 1 {
            output.push("---".to_owned());
        }
        let prefix = if matched.contains(&index) { '>' } else { ' ' };
        output.push(format!("{prefix} {:>width$}| {}", index + 1, lines[index]));
    }
    output.join("\n")
}

/// The pattern as a regex, or as a literal when it does not compile (or looks catastrophic).
fn compile(pattern: &str) -> Regex {
    let truncated: String = pattern.chars().take(MAX_PATTERN_LENGTH).collect();
    let compiled = (!NESTED_QUANTIFIER.is_match(&truncated))
        .then(|| Regex::new(&truncated).ok())
        .flatten();
    compiled.unwrap_or_else(|| Regex::new(&regex::escape(&truncated)).expect("escaped literal"))
}

/// Finds lines matching a pattern, expands with context, and formats with truncation.
fn search_by_pattern(
    lines: &[&str],
    pattern: &str,
    (scope_start, scope_end): (usize, usize),
    context_lines: usize,
    max_chars: usize,
    scope_label: &str,
) -> String {
    let regex = compile(pattern);
    let matched: BTreeSet<usize> = (scope_start..=scope_end)
        .filter(|&i| regex.is_match(lines[i]))
        .collect();
    if matched.is_empty() {
        return format!(
            "No matches found for pattern '{pattern}'{scope_label} (searched {} lines).",
            scope_end - scope_start + 1
        );
    }
    let mut visible = BTreeSet::new();
    for &index in &matched {
        let from = index.saturating_sub(context_lines).max(scope_start);
        let to = (index + context_lines).min(scope_end);
        visible.extend(from..=to);
    }
    let safe: String = pattern
        .chars()
        .map(|c| {
            if matches!(c, '\n' | '\r' | '/' | ']') {
                ' '
            } else {
                c
            }
        })
        .take(50)
        .collect();
    let count = matched.len();
    let plural = if count > 1 { "es" } else { "" };
    let header = format!("[{count} match{plural} for /{safe}/{scope_label}]");
    let indices: Vec<usize> = visible.into_iter().collect();
    let body = format_lines(lines, &indices, &matched);
    format!(
        "{header}\n\n{}",
        truncate(&body, max_chars, "output truncated, narrow your search")
    )
}

/// Formats a contiguous range of lines with truncation.
fn search_by_line_range(lines: &[&str], start: usize, end: usize, max_chars: usize) -> String {
    let header = format!("[Lines {}-{} of {}]", start + 1, end + 1, lines.len());
    let indices: Vec<usize> = (start..=end).collect();
    let body = format_lines(lines, &indices, &BTreeSet::new());
    format!(
        "{header}\n\n{}",
        truncate(&body, max_chars, "output truncated, narrow your range")
    )
}

/// Search offloaded text by pattern or line range: formatted results with line numbers, or an
/// error or no-match message. Output beyond `max_chars` is truncated.
pub(crate) fn search_content(text: &str, input: &SearchInput<'_>, max_chars: usize) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    let total = lines.len();
    if text.is_empty() {
        return "Content is empty (0 lines).".to_owned();
    }
    let (mut scope_start, mut scope_end) = (0, total - 1);
    if let Some(range) = input.line_range {
        if range.start > range.end {
            return format!(
                "Error: line_range.start ({}) must be <= line_range.end ({}).",
                range.start, range.end
            );
        }
        if range.start > total {
            return format!(
                "Error: line_range.start ({}) is beyond content length ({total} lines).",
                range.start
            );
        }
        scope_start = range.start.saturating_sub(1);
        scope_end = range.end.saturating_sub(1).min(total - 1);
    }
    if let Some(pattern) = input.pattern.filter(|p| !p.is_empty()) {
        let label = match input.line_range {
            Some(range) => format!(" in lines {}-{}", range.start, scope_end + 1),
            None => String::new(),
        };
        return search_by_pattern(
            &lines,
            pattern,
            (scope_start, scope_end),
            input.context_lines,
            max_chars,
            &label,
        );
    }
    search_by_line_range(&lines, scope_start, scope_end, max_chars)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX: usize = 10_000;

    fn pattern(pattern: &str, context_lines: usize) -> SearchInput<'_> {
        SearchInput {
            pattern: Some(pattern),
            context_lines,
            ..Default::default()
        }
    }

    fn range(start: usize, end: usize) -> SearchInput<'static> {
        SearchInput {
            line_range: Some(LineRange { start, end }),
            context_lines: 5,
            ..Default::default()
        }
    }

    fn numbered(prefix: &str, count: usize) -> String {
        (1..=count)
            .map(|i| format!("{prefix} {i}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn empty_content() {
        assert_eq!(
            search_content("", &pattern("x", 5), MAX),
            "Content is empty (0 lines)."
        );
        assert!(!search_content("\n", &range(1, 1), MAX).contains("Content is empty"));
    }

    #[test]
    fn line_range_validation() {
        let text = "line 1\nline 2\nline 3\nline 4\nline 5";
        assert!(search_content(text, &range(5, 2), MAX).contains("must be <= line_range.end"));
        assert!(search_content(text, &range(100, 200), MAX).contains("beyond content length (5 lines)"));
        let clamped = search_content(text, &range(3, 999), MAX);
        assert!(clamped.contains("[Lines 3-5 of 5]"), "{clamped}");
        assert!(clamped.contains("line 3") && clamped.contains("line 5"));
    }

    #[test]
    fn finds_a_single_match_with_context() {
        let result = search_content(&numbered("line", 20), &pattern("line 10", 2), MAX);
        assert!(result.contains("[1 match for /line 10/]"), "{result}");
        assert!(result.contains("> 10| line 10"));
        assert!(result.contains("   8| line 8"));
        assert!(result.contains("  12| line 12"));
        assert!(!result.contains("line 7"));
    }

    #[test]
    fn patterns_are_regexes() {
        let text = numbered("line", 20);
        let result = search_content(&text, &pattern("line [12]0", 0), MAX);
        assert!(result.contains("2 matches"), "{result}");
        assert!(result.contains("> 10| line 10") && result.contains("> 20| line 20"));
        // An escaped bracket, as models write for `SPEC[name]`.
        let spec = "a\n# SPEC[orders]: total\nb";
        let result = search_content(spec, &pattern(r"SPEC\[orders\]", 0), MAX);
        assert!(result.contains("> 2| # SPEC[orders]: total"), "{result}");
    }

    #[test]
    fn says_when_nothing_matches() {
        let result = search_content(&numbered("line", 20), &pattern("nonexistent", 5), MAX);
        assert!(
            result.contains("No matches found for pattern 'nonexistent'"),
            "{result}"
        );
        assert!(result.contains("searched 20 lines"));
    }

    #[test]
    fn context_lines_zero_and_groups() {
        let text = numbered("line", 20);
        let result = search_content(&text, &pattern("line 5", 0), MAX);
        assert!(result.contains("> 5| line 5"));
        assert!(!result.contains("line 4") && !result.contains("line 6"));
        let merged = search_content(&text, &pattern("line [67]", 2), MAX);
        assert!(
            merged.contains("2 matches") && !merged.contains("---"),
            "{merged}"
        );
        let apart = search_content(&text, &pattern("line (1|20)", 0), MAX);
        assert!(apart.contains("---"), "{apart}");
    }

    #[test]
    fn falls_back_to_a_literal_match_on_an_invalid_regex() {
        let text = "foo (bar\nbaz\nfoo (bar again";
        let result = search_content(text, &pattern("foo (bar", 0), MAX);
        assert!(result.contains("2 matches"), "{result}");
        assert!(result.contains("> 1| foo (bar") && result.contains("> 3| foo (bar again"));
    }

    #[test]
    fn sanitizes_the_pattern_in_the_header() {
        let result = search_content("test line\nanother line", &pattern("test]\nline", 0), MAX);
        let header = result.split('\n').next().unwrap();
        assert!(!header.contains("]/"), "{header}");
    }

    #[test]
    fn a_pattern_within_a_line_range() {
        let text = numbered("item", 30);
        let within = |p| SearchInput {
            pattern: Some(p),
            line_range: Some(LineRange { start: 10, end: 20 }),
            context_lines: 0,
        };
        let result = search_content(&text, &within("item 1"), MAX);
        assert!(result.contains("in lines 10-20"), "{result}");
        assert!(result.contains("> 10| item 10") && result.contains("> 11| item 11"));
        assert!(!result.contains("> 1|"));
        let none = search_content(&text, &within("item 5"), MAX);
        assert!(
            none.contains("No matches found") && none.contains("in lines 10-20"),
            "{none}"
        );
    }

    #[test]
    fn a_line_range_without_a_pattern() {
        let text = numbered("line", 50);
        let result = search_content(&text, &range(5, 10), MAX);
        assert!(result.contains("[Lines 5-10 of 50]"), "{result}");
        assert!(result.contains("  5| line 5") && result.contains(" 10| line 10"));
        assert!(!result.contains("line 4") && !result.contains("line 11"));
        assert!(!search_content(&text, &range(1, 10), MAX).contains("---"));
    }

    #[test]
    fn truncation() {
        let text = numbered("match line", 500);
        let result = search_content(&text, &pattern("match", 0), 200);
        assert!(
            result.contains("output truncated, narrow your search"),
            "{result}"
        );
        assert!(result.chars().count() <= 280);
        let text = numbered("line", 500);
        let result = search_content(&text, &range(1, 500), 200);
        assert!(result.contains("output truncated, narrow your range"), "{result}");
        assert!(result.chars().count() <= 280);
        let long = format!("{{\"key\":\"{}\"}}", "x".repeat(5000));
        for input in [pattern("key", 0), range(1, 50)] {
            let result = search_content(&long, &input, 500);
            assert!(result.contains("output truncated") && result.contains("{\"key\":\""));
        }
        assert!(!search_content("short\ncontent", &range(1, 2), MAX).contains("truncated"));
    }
}
