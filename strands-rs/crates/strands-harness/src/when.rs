//! Durations (`90s`, `1h30m`) and points in time (`in 2h`, `2026-11-17T09:00Z`) as agents write
//! them, for todo deadlines and anything else a tool schedules.

use std::time::Duration;

use chrono::{DateTime, NaiveDateTime, Utc};

/// A duration like `90s`, `30m`, `2h`, `1d` or `1h30m`.
pub fn parse_duration(text: &str) -> Result<Duration, String> {
    let bad = || format!("Not a duration: {text:?} (use e.g. 90s, 30m, 2h, 1d or 1h30m).");
    let mut total = 0u64;
    let mut digits = String::new();
    for c in text.trim().chars() {
        if c.is_ascii_digit() {
            digits.push(c);
            continue;
        }
        let unit = match c {
            's' => 1,
            'm' => 60,
            'h' => 3600,
            'd' => 86_400,
            'w' => 604_800,
            _ => return Err(bad()),
        };
        let amount: u64 = digits.parse().map_err(|_| bad())?;
        total = amount
            .checked_mul(unit)
            .and_then(|secs| total.checked_add(secs))
            .ok_or_else(bad)?;
        digits.clear();
    }
    if !digits.is_empty() || total == 0 {
        return Err(bad());
    }
    Ok(Duration::from_secs(total))
}

/// A point in time: `in 2h`, or an ISO 8601 UTC/offset time such as `2026-11-17T09:00Z`,
/// `2026-11-17T09:00:00+01:00` or `2026-11-17 09:00` (taken as UTC).
pub fn parse_at(text: &str, now: DateTime<Utc>) -> Result<DateTime<Utc>, String> {
    let text = text.trim();
    if let Some(delay) = text.strip_prefix("in ") {
        let delay = chrono::Duration::from_std(parse_duration(delay)?).map_err(|e| e.to_string())?;
        return Ok(now + delay);
    }
    if let Ok(time) = DateTime::parse_from_rfc3339(text) {
        return Ok(time.with_timezone(&Utc));
    }
    // RFC 3339 needs seconds and an offset; accept the shorter forms people write.
    let (local, offset) = match text.strip_suffix('Z').or_else(|| text.strip_suffix("UTC")) {
        Some(local) => (local.trim(), "+00:00"),
        None => match text.len().checked_sub(6).filter(|&i| text.is_char_boundary(i)) {
            Some(i) if matches!(&text[i..i + 1], "+" | "-") && text[i..].contains(':') => {
                (&text[..i], &text[i..])
            }
            _ => (text, "+00:00"),
        },
    };
    let local = local.replacen(' ', "T", 1);
    for format in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(&local, format) {
            let stamp = format!("{}{offset}", naive.format("%Y-%m-%dT%H:%M:%S"));
            if let Ok(time) = DateTime::parse_from_rfc3339(&stamp) {
                return Ok(time.with_timezone(&Utc));
            }
        }
    }
    Err(format!(
        "Not a time: {text:?} (use e.g. \"in 2h\" or \"2026-11-17T09:00Z\")."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utc(text: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(text).unwrap().with_timezone(&Utc)
    }

    #[test]
    fn durations_parse_in_the_usual_units() {
        assert_eq!(parse_duration("30m"), Ok(Duration::from_secs(1800)));
        assert_eq!(parse_duration("1h30m"), Ok(Duration::from_secs(5400)));
        assert_eq!(parse_duration(" 90s "), Ok(Duration::from_secs(90)));
        assert_eq!(parse_duration("2d"), Ok(Duration::from_secs(172_800)));
        for bad in ["", "30", "m", "5 minutes", "0m", "1.5h"] {
            assert!(parse_duration(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn times_parse_relative_and_absolute() {
        let now = utc("2026-11-16T10:00:00Z");
        assert_eq!(parse_at("in 2h", now), Ok(utc("2026-11-16T12:00:00Z")));
        let nine = utc("2026-11-17T09:00:00Z");
        for text in [
            "2026-11-17T09:00Z",
            "2026-11-17T09:00:00Z",
            "2026-11-17 09:00",
            "2026-11-17T10:00+01:00",
            "2026-11-17T04:00:00-05:00",
            "2026-11-17 09:00 UTC",
        ] {
            assert_eq!(parse_at(text, now), Ok(nine), "{text}");
        }
        assert!(parse_at("tomorrow", now).is_err());
        assert!(parse_at("in a bit", now).is_err());
    }
}
