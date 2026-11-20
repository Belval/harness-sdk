//! Delayed and recurring messages: a [`Schedule`] posts its message to an agent's mailbox at
//! `next`, then every `every` (if set). Times are UTC; parsing and display are here, firing is in
//! the runtime.

use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Shortest repeat interval: a faster loop would mostly burn tokens.
pub const MIN_EVERY: Duration = Duration::from_secs(60);

/// A message to deliver later, possibly repeatedly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Schedule {
    /// Unique id (`s1`, … or chosen by the creator).
    pub id: String,
    /// Who created it: an agent id, or the user. Posts come from it.
    pub owner: String,
    /// Receiving agent id.
    pub to: String,
    /// Text to post.
    pub message: String,
    /// When it fires next.
    pub next: DateTime<Utc>,
    /// Repeat interval in seconds; `None` fires once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub every_secs: Option<u64>,
    /// When it last fired and what the receiver had spent by then, so the next fire can say what
    /// the receiver spent in between.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fired: Option<Fired>,
}

/// A fire of a [`Schedule`] and the receiver's spend so far at that moment.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Fired {
    /// When.
    pub at: DateTime<Utc>,
    /// The receiver's input + output tokens so far.
    pub tokens: u64,
    /// Its estimated USD so far; `None` when its model has no price.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usd: Option<f64>,
}

impl Schedule {
    /// After firing at `now`: move `next` to the first repeat after `now`, so repeats missed while
    /// the swarm was down collapse into the one fire. `false` for a one-shot (drop it).
    pub fn advance(&mut self, now: DateTime<Utc>) -> bool {
        let Some(every) = self.every_secs.filter(|secs| *secs > 0) else {
            return false;
        };
        let every = chrono::Duration::seconds(every as i64);
        let missed = ((now - self.next).num_seconds() / every.num_seconds()).max(0);
        self.next += every * (missed as i32 + 1);
        true
    }

    /// The text this posts at `now`, given the receiver's spend so far. It carries the time, which
    /// a polling agent otherwise runs `date` for on every fire. After a repeat's first fire it
    /// says what the receiver spent since the last one (and per hour at this interval), so a
    /// polling agent sees what its loop costs.
    pub fn text(&self, now: Fired) -> String {
        let at = now.at.format("%Y-%m-%d %H:%M UTC");
        let text = format!("[schedule {}, {at}] {}", self.id, self.message);
        let (Some(every), Some(last)) = (self.every_secs, self.fired) else {
            return text;
        };
        let tokens = now.tokens.saturating_sub(last.tokens);
        if tokens == 0 {
            return text;
        }
        let ago = human((now.at - last.at).to_std().unwrap_or_default());
        let tokens = crate::text::compact(tokens);
        let spent = match (now.usd, last.usd) {
            (Some(now), Some(last)) => {
                let usd = (now - last).max(0.0);
                let rate = usd * 3600.0 / every.max(1) as f64;
                format!(
                    "{tokens} tokens, est. {}, since this schedule last fired {ago} ago: about {}/h at this interval",
                    crate::cost::dollars(usd),
                    crate::cost::dollars(rate)
                )
            }
            _ => format!("{tokens} tokens since this schedule last fired {ago} ago"),
        };
        format!("{text}\n(You used {spent}.)")
    }

    /// One line: `s1 every 30m, next in 12m (2026-11-17 09:00 UTC): check the queue`.
    pub fn describe(&self, now: DateTime<Utc>) -> String {
        let when = match self.every_secs {
            Some(secs) => format!("every {}, next", human(Duration::from_secs(secs))),
            None => "once".to_owned(),
        };
        let wait = (self.next - now).to_std().unwrap_or_default();
        let first = self.message.lines().next().unwrap_or_default();
        format!(
            "{} {when} in {} ({}): {first}",
            self.id,
            human(wait),
            self.next.format("%Y-%m-%d %H:%M UTC")
        )
    }
}

pub use strands_harness::when::{parse_at, parse_duration};

/// `45s`, `12m`, `2h5m`, `3d4h`: the two largest units, to the nearest minute from a minute on.
pub fn human(duration: Duration) -> String {
    let secs = match duration.as_secs() {
        secs if secs >= 60 => (secs + 30) / 60 * 60,
        secs => secs,
    };
    let units = [(86_400, "d"), (3600, "h"), (60, "m"), (1, "s")];
    let parts: Vec<String> = units
        .iter()
        .scan(secs, |left, (size, unit)| {
            let n = *left / size;
            *left %= size;
            Some((n, unit))
        })
        .skip_while(|(n, _)| *n == 0)
        .take(2)
        .filter(|(n, _)| *n > 0)
        .map(|(n, unit)| format!("{n}{unit}"))
        .collect();
    match parts.is_empty() {
        true => "0s".to_owned(),
        false => parts.concat(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utc(text: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(text).unwrap().with_timezone(&Utc)
    }

    #[test]
    fn a_late_repeat_fires_once_and_keeps_its_grid() {
        let mut schedule = Schedule {
            id: "s1".into(),
            owner: "a1".into(),
            to: "a1".into(),
            message: "check".into(),
            next: utc("2026-11-16T10:00:00Z"),
            every_secs: Some(1800),
            fired: None,
        };
        // Down from 10:00 to 11:40: one fire now, the next on the 30-minute grid.
        assert!(schedule.advance(utc("2026-11-16T11:40:00Z")));
        assert_eq!(schedule.next, utc("2026-11-16T12:00:00Z"));
        assert!(schedule.advance(utc("2026-11-16T12:00:00Z")));
        assert_eq!(schedule.next, utc("2026-11-16T12:30:00Z"));
        schedule.every_secs = None;
        assert!(!schedule.advance(utc("2026-11-16T12:30:00Z")));
    }

    #[test]
    fn a_schedule_describes_itself_on_one_line() {
        let schedule = Schedule {
            id: "s1".into(),
            owner: "a1".into(),
            to: "a2".into(),
            message: "check the queue\nand report".into(),
            next: utc("2026-11-17T09:00:00Z"),
            every_secs: Some(1800),
            fired: None,
        };
        let now = utc("2026-11-17T08:48:00Z");
        assert_eq!(
            schedule.describe(now),
            "s1 every 30m, next in 12m (2026-11-17 09:00 UTC): check the queue"
        );
        let once = Schedule {
            every_secs: None,
            ..schedule
        };
        assert_eq!(
            once.describe(utc("2026-11-16T07:00:00Z")),
            "s1 once in 1d2h (2026-11-17 09:00 UTC): check the queue"
        );
        assert_eq!(human(Duration::from_secs(3 * 86_400 + 61)), "3d");
        assert_eq!(human(Duration::from_secs(45)), "45s");
        assert_eq!(human(Duration::from_secs(29 * 60 + 59)), "30m");
    }

    #[test]
    fn a_fire_says_when_it_is_and_a_repeat_what_its_receiver_spent_since_the_last() {
        let mut schedule = Schedule {
            id: "poll".into(),
            owner: "a1".into(),
            to: "a1".into(),
            message: "poll slack".into(),
            next: utc("2026-11-17T09:00:00Z"),
            every_secs: Some(120),
            fired: None,
        };
        let at = |time: &str, tokens, usd| Fired {
            at: utc(time),
            tokens,
            usd,
        };
        let first = at("2026-11-17T09:00:00Z", 1_000_000, Some(10.0));
        assert_eq!(
            schedule.text(first),
            "[schedule poll, 2026-11-17 09:00 UTC] poll slack",
            "nothing to compare yet"
        );
        schedule.fired = Some(first);
        assert_eq!(
            schedule.text(at("2026-11-17T09:02:00Z", 1_330_000, Some(10.21))),
            "[schedule poll, 2026-11-17 09:02 UTC] poll slack\n(You used 330.0k tokens, est. $0.21, since this schedule last \
             fired 2m ago: about $6.30/h at this interval.)"
        );
        assert_eq!(
            schedule.text(at("2026-11-17T09:02:00Z", 1_000_000, Some(10.0))),
            "[schedule poll, 2026-11-17 09:02 UTC] poll slack",
            "nothing spent, nothing said"
        );
        schedule.fired = Some(at("2026-11-17T09:00:00Z", 1_000_000, None));
        assert_eq!(
            schedule.text(at("2026-11-17T09:04:00Z", 1_002_000, None)),
            "[schedule poll, 2026-11-17 09:04 UTC] poll slack\n(You used 2.0k tokens since this schedule last fired 4m ago.)",
            "unpriced: tokens only"
        );
        schedule.every_secs = None;
        assert_eq!(
            schedule.text(at("2026-11-17T09:04:00Z", 1_002_000, None)),
            "[schedule poll, 2026-11-17 09:04 UTC] poll slack",
            "a one-shot has no loop to price"
        );
    }
}
