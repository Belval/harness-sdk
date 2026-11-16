//! Token budget helpers: parsing and formatting token counts for `--budget-tokens` and `/budget`.

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

/// `950`, `12.3k`, `4.5M`.
pub fn compact(n: u64) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=999_999 => format!("{:.1}k", n as f64 / 1e3),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

/// The error a model call fails with once the budget is spent.
pub fn exhausted_message(used: u64, budget: u64) -> String {
    format!(
        "Token budget exhausted: the swarm has used {} of {} tokens, so no further model calls run. \
         Report this to whoever is waiting on you; you will be told when the user raises the budget \
         (TUI `/budget N`).",
        compact(used),
        compact(budget)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
