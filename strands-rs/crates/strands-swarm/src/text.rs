//! Small text helpers: token counts parsed and shown compactly, and clipping to a few chars.

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
    match n {
        0..=999 => n.to_string(),
        1_000..=999_999 => format!("{:.1}k", n as f64 / 1e3),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

/// `text` cut to its first `max` chars, with "…" appended when that dropped anything.
pub fn clip(text: &str, max: usize) -> String {
    match text.chars().count() > max {
        true => text.chars().take(max).chain(['…']).collect(),
        false => text.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn clips_to_a_char_count_marking_the_cut() {
        assert_eq!(clip("short", 5), "short");
        assert_eq!(clip("élan vital", 4), "élan…");
    }
}
