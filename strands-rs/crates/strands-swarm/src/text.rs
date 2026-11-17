//! Small text helpers: token counts parsed and shown compactly, clipping to a few chars, and
//! terminal output reduced to the text it shows.

/// Parse a token count: `500000`, `500k`, `2.5M`, `1G` (case-insensitive, `_` allowed).
pub fn parse_tokens(text: &str) -> Result<u64, String> {
    let text = text.trim().replace('_', "");
    let (number, scale) = match text.chars().last().map(|c| c.to_ascii_lowercase()) {
        Some('k') => (&text[..text.len() - 1], 1e3),
        Some('m') => (&text[..text.len() - 1], 1e6),
        Some('g') => (&text[..text.len() - 1], 1e9),
        _ => (text.as_str(), 1.0),
    };
    match number.parse::<f64>() {
        Ok(n) if n.is_finite() && n > 0.0 => Ok((n * scale).round() as u64),
        _ => Err(format!("`{text}` is not a token count (e.g. 500000, 500k, 2M)")),
    }
}

/// Parse a size in bytes: `0`, `512M`, `20G`, `1.5TB` (binary units, case-insensitive).
pub fn parse_bytes(text: &str) -> Result<u64, String> {
    let lower = text.trim().to_ascii_lowercase();
    let number = lower.strip_suffix('b').unwrap_or(&lower);
    let (number, shift) = match number.chars().last() {
        Some('k') => (&number[..number.len() - 1], 10),
        Some('m') => (&number[..number.len() - 1], 20),
        Some('g') => (&number[..number.len() - 1], 30),
        Some('t') => (&number[..number.len() - 1], 40),
        _ => (number, 0),
    };
    match number.trim().parse::<f64>() {
        Ok(n) if n.is_finite() && n >= 0.0 => Ok((n * (1u64 << shift) as f64).round() as u64),
        _ => Err(format!("`{text}` is not a size (e.g. 20G, 512M, 0)")),
    }
}

/// `950`, `12.3k`, `4.5M`.
pub fn compact(n: u64) -> String {
    compact_to(n, 1)
}

/// [`compact`] with `decimals` places; beyond one, trailing zeros are dropped (`1.25M`).
fn compact_to(n: u64, decimals: usize) -> String {
    let (scaled, unit) = match n {
        0..=999 => return n.to_string(),
        1_000..=999_999 => (n as f64 / 1e3, "k"),
        _ => (n as f64 / 1e6, "M"),
    };
    let mut number = format!("{scaled:.decimals$}");
    if decimals > 1 {
        let kept = number
            .trim_end_matches('0')
            .len()
            .max(number.find('.').map_or(0, |dot| dot + 2));
        number.truncate(kept);
    }
    number + unit
}

/// Token usage against a budget, e.g. `1.1M of 1.2M tokens used`, `all 1.2M tokens used` or
/// What an agent waits for when the model-call cap is full, e.g.
/// `queued for a model slot (8/8 calls in flight; /max-calls to raise)`.
pub fn queued_for_call(calls: usize, limit: usize) -> String {
    format!("queued for a model slot ({calls}/{limit} calls in flight; /max-calls to raise)")
}

/// `1.3M tokens used, over the 1.2M budget`. Two different counts are shown with enough places to
/// tell them apart, so usage just under the budget never reads as equal to it or over it.
pub fn usage(used: u64, budget: u64) -> String {
    let shown = |decimals| (compact_to(used, decimals), compact_to(budget, decimals));
    let (u, b) = (1..=6)
        .map(shown)
        .find(|(u, b)| used == budget || u != b)
        .unwrap_or_else(|| (used.to_string(), budget.to_string()));
    match used.cmp(&budget) {
        std::cmp::Ordering::Less => format!("{u} of {b} tokens used"),
        std::cmp::Ordering::Equal => format!("all {b} tokens used"),
        std::cmp::Ordering::Greater => format!("{u} tokens used, over the {b} budget"),
    }
}

/// `text` cut to its first `max` chars, with "…" appended when that dropped anything.
pub fn clip(text: &str, max: usize) -> String {
    match text.chars().count() > max {
        true => text.chars().take(max).chain(['…']).collect(),
        false => text.to_owned(),
    }
}

/// Terminal output as the text a terminal leaves on screen: without ANSI escape sequences
/// (colours, cursor moves, OSC titles and hyperlinks) and with each `\r`-overwritten line (progress
/// bars, spinners) reduced to what was written last. Test runners and build tools colour their
/// output even into a pipe, in tokens the model pays for and has to read around.
pub fn plain(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains(['\x1b', '\r']) {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\x1b' => match chars.next() {
                // CSI: parameters and intermediates up to a final byte in `@`..=`~`.
                Some('[') => {
                    for c in chars.by_ref() {
                        if ('@'..='~').contains(&c) {
                            break;
                        }
                    }
                }
                // OSC (and the other string sequences): up to BEL or ST (`ESC \`).
                Some(']' | 'P' | '_' | '^' | 'X') => {
                    while let Some(c) = chars.next() {
                        if c == '\x07' || (c == '\x1b' && chars.next_if_eq(&'\\').is_some()) {
                            break;
                        }
                    }
                }
                // Two-character sequences (`ESC 7`, `ESC (B`'s first part, …).
                _ => {}
            },
            '\r' if chars.peek() == Some(&'\n') => {}
            // A carriage return starts the line over; a trailing one overwrites nothing.
            '\r' if chars.peek().is_some() => {
                let start = out.rfind('\n').map_or(0, |i| i + 1);
                out.truncate(start);
            }
            '\r' => {}
            c => out.push(c),
        }
    }
    std::borrow::Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_drops_escape_sequences_and_overwritten_progress() {
        assert!(matches!(
            plain("no escapes\n\tat all"),
            std::borrow::Cow::Borrowed(_)
        ));
        assert_eq!(
            plain("\x1b[31m\x1b[1mError\x1b[22m: boom\x1b[39m\n\x1b[2K\x1b[1Gdone"),
            "Error: boom\ndone"
        );
        assert_eq!(
            plain("\x1b]8;;http://x\x07link\x1b]8;;\x07 \x1b]0;title\x1b\\ok \x1b(Bplain"),
            "link ok Bplain"
        );
        assert_eq!(plain("a\r\nb\r\n"), "a\nb\n", "CRLF is a line break");
        assert_eq!(plain("start\n 10%\r 50%\r100%\nnext\r"), "start\n100%\nnext");
        assert_eq!(
            plain("cut \x1b[3"),
            "cut ",
            "an unterminated sequence ends the text"
        );
    }

    #[test]
    fn parses_sizes_in_binary_units() {
        assert_eq!(parse_bytes("20G"), Ok(20 << 30));
        assert_eq!(parse_bytes("1.5gb"), Ok(3 << 29));
        assert_eq!(parse_bytes("512M"), Ok(512 << 20));
        assert_eq!(parse_bytes("0"), Ok(0));
        assert!(parse_bytes("lots").is_err());
    }

    #[test]
    fn parses_token_counts_with_suffixes() {
        assert_eq!(parse_tokens("500000"), Ok(500_000));
        assert_eq!(parse_tokens("500k"), Ok(500_000));
        assert_eq!(parse_tokens(" 2.5M "), Ok(2_500_000));
        assert_eq!(parse_tokens("1g"), Ok(1_000_000_000));
        assert_eq!(parse_tokens("10_000"), Ok(10_000));
        assert!(parse_tokens("0").is_err());
        assert!(parse_tokens("lots").is_err());
        assert!(parse_tokens("").is_err());
    }

    #[test]
    fn formats_counts_compactly() {
        assert_eq!(compact(950), "950");
        assert_eq!(compact(12_345), "12.3k");
        assert_eq!(compact(4_500_000), "4.5M");
    }

    /// Usage just over or under a budget never rounds to a contradiction like "1.3M of 1.2M".
    #[test]
    fn usage_against_a_budget_reads_consistently() {
        assert_eq!(
            usage(1_260_000, 1_240_000),
            "1.3M tokens used, over the 1.2M budget"
        );
        assert_eq!(
            usage(1_240_001, 1_240_000),
            "1.240001M tokens used, over the 1.24M budget"
        );
        assert_eq!(usage(1_249_000, 1_250_000), "1.249M of 1.25M tokens used");
        assert_eq!(usage(1_250_000, 1_250_000), "all 1.2M tokens used");
        assert_eq!(usage(300_000, 1_000_000), "300.0k of 1.0M tokens used");
        assert_eq!(usage(999, 1_000), "999 of 1.0k tokens used");
    }

    #[test]
    fn clips_to_a_char_count_marking_the_cut() {
        assert_eq!(clip("short", 5), "short");
        assert_eq!(clip("élan vital", 4), "élan…");
    }
}
