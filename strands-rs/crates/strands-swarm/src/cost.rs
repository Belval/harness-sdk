//! Estimated spend: tokens per agent and model, priced from a small table of list prices
//! (`--prices FILE` overrides it), plus per-call stats. Everything here is an estimate from
//! the usage the models report, not a bill.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Calls [`Ledger::call_stats`] looks back over.
pub const RECENT_CALLS: usize = 50;

/// Cache reads cost this share of the input price unless a price says otherwise.
pub const CACHE_READ_FACTOR: f64 = 0.1;

/// Cache writes cost this share of the input price unless a price says otherwise.
pub const CACHE_WRITE_FACTOR: f64 = 1.25;

/// Built-in USD list prices per 1M input/output tokens, by model-id substring. Unknown newer
/// versions fall back to their tier's latest known price.
const BUILTIN: &[(&str, f64, f64)] = &[
    ("opus", 5.0, 25.0),
    ("opus-4-1", 15.0, 75.0),
    ("opus-4-2", 15.0, 75.0),
    ("3-opus", 15.0, 75.0),
    ("sonnet", 3.0, 15.0),
    ("haiku", 1.0, 5.0),
    ("3-5-haiku", 0.8, 4.0),
    ("3-haiku", 0.25, 1.25),
];

/// Tokens of one agent on one model. `input` counts all input, cache reads and writes included.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counts {
    /// Model calls.
    #[serde(default)]
    pub calls: u64,
    /// Input tokens, cache reads and writes included.
    pub input: u64,
    /// Output tokens.
    pub output: u64,
    /// Input read from the prompt cache.
    #[serde(default)]
    pub cache_read: u64,
    /// Input written to the prompt cache.
    #[serde(default)]
    pub cache_write: u64,
}

impl Counts {
    fn add(&mut self, other: &Counts) {
        self.calls += other.calls;
        self.input += other.input;
        self.output += other.output;
        self.cache_read += other.cache_read;
        self.cache_write += other.cache_write;
    }

    fn minus(&self, other: &Counts) -> Counts {
        Counts {
            calls: self.calls.saturating_sub(other.calls),
            input: self.input.saturating_sub(other.input),
            output: self.output.saturating_sub(other.output),
            cache_read: self.cache_read.saturating_sub(other.cache_read),
            cache_write: self.cache_write.saturating_sub(other.cache_write),
        }
    }

    fn is_empty(&self) -> bool {
        self.input + self.output == 0
    }
}

/// One of an agent's own model calls, for [`Ledger::call_stats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Call {
    /// Input tokens, cache included.
    pub input: u64,
    /// Of which read from the cache.
    pub cache_read: u64,
}

/// An agent's tokens by model plus its most recent calls; persisted with the agent.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ledger {
    /// Model id → tokens spent on it.
    #[serde(default)]
    pub by_model: BTreeMap<String, Counts>,
    /// The last [`RECENT_CALLS`] own calls (side calls such as the goal judge left out).
    #[serde(default)]
    pub recent: VecDeque<Call>,
}

/// Median input and cache hit rate over recent calls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CallStats {
    /// Calls looked at.
    pub calls: usize,
    /// Median input tokens per call.
    pub median_input: u64,
    /// Share of their input read from the cache, `0..=1`.
    pub cache_hit: f64,
}

impl Ledger {
    /// Whether nothing was recorded.
    pub fn is_empty(&self) -> bool {
        self.by_model.is_empty() && self.recent.is_empty()
    }

    /// Count one model call on `model`; `own` calls also feed [`Ledger::call_stats`].
    pub fn record(&mut self, model: &str, counts: Counts, own: bool) {
        self.by_model.entry(model.to_owned()).or_default().add(&counts);
        if own {
            self.recent.push_back(Call {
                input: counts.input,
                cache_read: counts.cache_read,
            });
            while self.recent.len() > RECENT_CALLS {
                self.recent.pop_front();
            }
        }
    }

    /// Stats over the recent calls; `None` before the first.
    pub fn call_stats(&self) -> Option<CallStats> {
        if self.recent.is_empty() {
            return None;
        }
        let mut inputs: Vec<u64> = self.recent.iter().map(|call| call.input).collect();
        inputs.sort_unstable();
        let mid = inputs.len() / 2;
        let median_input = match inputs.len() % 2 {
            0 => (inputs[mid - 1] + inputs[mid]) / 2,
            _ => inputs[mid],
        };
        let input: u64 = self.recent.iter().map(|call| call.input).sum();
        let read: u64 = self.recent.iter().map(|call| call.cache_read).sum();
        Some(CallStats {
            calls: self.recent.len(),
            median_input,
            cache_hit: if input == 0 {
                0.0
            } else {
                read as f64 / input as f64
            },
        })
    }
}

/// USD per 1M tokens.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Price {
    /// Uncached input.
    pub input: f64,
    /// Output.
    pub output: f64,
    /// Cache reads; default [`CACHE_READ_FACTOR`] × `input`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<f64>,
    /// Cache writes; default [`CACHE_WRITE_FACTOR`] × `input`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<f64>,
}

impl Price {
    /// Estimated USD for `counts`.
    pub fn cost(&self, counts: &Counts) -> f64 {
        let uncached = counts
            .input
            .saturating_sub(counts.cache_read + counts.cache_write);
        let read = self.cache_read.unwrap_or(self.input * CACHE_READ_FACTOR);
        let write = self.cache_write.unwrap_or(self.input * CACHE_WRITE_FACTOR);
        (uncached as f64 * self.input
            + counts.output as f64 * self.output
            + counts.cache_read as f64 * read
            + counts.cache_write as f64 * write)
            / 1e6
    }
}

/// Prices by model-id substring: the user's (`--prices`) first, then the built-in table; the
/// longest matching key wins within each.
#[derive(Debug, Clone, PartialEq)]
pub struct Prices {
    user: BTreeMap<String, Price>,
    builtin: BTreeMap<String, Price>,
}

impl Default for Prices {
    fn default() -> Self {
        let builtin = BUILTIN
            .iter()
            .map(|&(key, input, output)| {
                let price = Price {
                    input,
                    output,
                    cache_read: None,
                    cache_write: None,
                };
                (key.to_owned(), price)
            })
            .collect();
        Self {
            user: BTreeMap::new(),
            builtin,
        }
    }
}

impl Prices {
    /// The built-in table overridden by a JSON object `{"<model-id substring>": {"input": 5,
    /// "output": 25, "cache_read"?: 0.5, "cache_write"?: 6.25}}` (USD per 1M tokens).
    pub fn with_overrides(json: &str) -> Result<Self, String> {
        let user: BTreeMap<String, Price> =
            serde_json::from_str(json).map_err(|e| format!("bad price table: {e}"))?;
        let user = user.into_iter().map(|(k, v)| (k.to_lowercase(), v)).collect();
        Ok(Self {
            user,
            ..Self::default()
        })
    }

    /// [`Prices::with_overrides`] from a file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::with_overrides(&json).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The price of `model`, if any key is part of its id (case-insensitive).
    pub fn price(&self, model: &str) -> Option<Price> {
        let model = model.to_lowercase();
        let best = |table: &BTreeMap<String, Price>| {
            table
                .iter()
                .filter(|(key, _)| model.contains(key.as_str()))
                .max_by_key(|(key, _)| key.len())
                .map(|(_, price)| *price)
        };
        best(&self.user).or_else(|| best(&self.builtin))
    }
}

/// An estimate in USD; `unknown` when some tokens were on models without a price.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Estimate {
    /// USD for the priced tokens.
    pub usd: f64,
    /// Some tokens could not be priced.
    pub unknown: bool,
    /// Any tokens at all.
    pub any: bool,
}

impl Estimate {
    fn add(&mut self, other: Estimate) {
        self.usd += other.usd;
        self.unknown |= other.unknown;
        self.any |= other.any;
    }

    /// `$12.30`, `$12.30+?` when partly unknown, `unknown` when nothing was priced.
    pub fn render(&self) -> String {
        match (self.unknown, self.usd > 0.0 || !self.any) {
            (false, _) => dollars(self.usd),
            (true, true) => format!("{}+?", dollars(self.usd)),
            (true, false) => "unknown".to_owned(),
        }
    }
}

fn dollars(usd: f64) -> String {
    match usd {
        usd if usd >= 100.0 => format!("${usd:.0}"),
        usd if usd >= 0.01 || usd == 0.0 => format!("${usd:.2}"),
        usd => format!("${usd:.4}"),
    }
}

/// One agent's spend, for [`CostReport`].
#[derive(Debug, Clone, PartialEq)]
pub struct AgentCost {
    /// `name (id)`.
    pub label: String,
    /// Totals across restarts (also from before per-model tracking).
    pub totals: Counts,
    /// Tokens by model and recent calls.
    pub ledger: Ledger,
    /// Model that tokens not in the ledger (runs saved before it) are priced as.
    pub fallback_model: Option<String>,
}

impl AgentCost {
    /// Estimated USD of everything this agent spent.
    pub fn estimate(&self, prices: &Prices) -> Estimate {
        let mut total = Estimate::default();
        let mut priced = |model: Option<&str>, counts: &Counts| {
            if counts.is_empty() {
                return;
            }
            let estimate = match model.and_then(|model| prices.price(model)) {
                Some(price) => Estimate {
                    usd: price.cost(counts),
                    unknown: false,
                    any: true,
                },
                None => Estimate {
                    usd: 0.0,
                    unknown: true,
                    any: true,
                },
            };
            total.add(estimate);
        };
        let mut tracked = Counts::default();
        for (model, counts) in &self.ledger.by_model {
            priced(Some(model), counts);
            tracked.add(counts);
        }
        priced(self.fallback_model.as_deref(), &self.totals.minus(&tracked));
        total
    }

    /// The model this agent spent most input on.
    pub fn main_model(&self) -> Option<&str> {
        let most = self.ledger.by_model.iter().max_by_key(|(_, counts)| counts.input);
        most.map(|(model, _)| model.as_str())
            .or(self.fallback_model.as_deref())
    }
}

/// What `/cost` and `--cost` show.
#[derive(Debug, Clone, PartialEq)]
pub struct CostReport {
    /// Agents in spawn order.
    pub agents: Vec<AgentCost>,
    /// Prices used.
    pub prices: Prices,
}

impl CostReport {
    /// The report of a saved run (`--cost`); `default_model` prices tokens of agents without a
    /// model of their own that predate per-model tracking.
    pub fn from_state(state: &crate::state::SwarmState, prices: Prices, default_model: Option<&str>) -> Self {
        let agents = state
            .agents
            .iter()
            .map(|agent| AgentCost {
                label: match &agent.name {
                    Some(name) => format!("{name} ({})", agent.id),
                    None => agent.id.clone(),
                },
                totals: Counts {
                    calls: 0,
                    input: agent.input_tokens,
                    output: agent.output_tokens,
                    cache_read: agent.cache_read_tokens,
                    cache_write: agent.cache_write_tokens,
                },
                ledger: agent.ledger.clone(),
                fallback_model: agent.model.clone().or(default_model.map(str::to_owned)),
            })
            .collect();
        Self { agents, prices }
    }

    /// The swarm-wide estimate.
    pub fn total(&self) -> Estimate {
        let mut total = Estimate::default();
        for agent in &self.agents {
            total.add(agent.estimate(&self.prices));
        }
        total
    }

    /// The report as a text table.
    pub fn lines(&self) -> Vec<String> {
        let row = |cells: [&str; 8]| {
            format!(
                "{:<24} {:<22} {:>7} {:>7} {:>8} {:>8} {:>9} {:>14}",
                cells[0], cells[1], cells[2], cells[3], cells[4], cells[5], cells[6], cells[7]
            )
            .trim_end()
            .to_owned()
        };
        let mut lines = vec![
            format!(
                "Estimated cost: {} (list prices per 1M tokens; cache read {CACHE_READ_FACTOR}×, \
                 write {CACHE_WRITE_FACTOR}× input; not a bill)",
                self.total().render()
            ),
            String::new(),
            row([
                "agent",
                "model",
                "in",
                "out",
                "cache rd",
                "cache wr",
                "est. $",
                &format!("last {RECENT_CALLS} calls"),
            ]),
        ];
        let mut sum = Counts::default();
        for agent in &self.agents {
            let t = &agent.totals;
            sum.add(t);
            let stats = agent.ledger.call_stats().map_or_else(String::new, |stats| {
                format!(
                    "{}/call {:.0}%",
                    crate::budget::compact(stats.median_input),
                    stats.cache_hit * 100.0
                )
            });
            let model = agent.main_model().map(short_model).unwrap_or("-");
            lines.push(row([
                &truncate(&agent.label, 24),
                &truncate(model, 22),
                &crate::budget::compact(t.input),
                &crate::budget::compact(t.output),
                &crate::budget::compact(t.cache_read),
                &crate::budget::compact(t.cache_write),
                &agent.estimate(&self.prices).render(),
                &stats,
            ]));
        }
        lines.push(row([
            "total",
            "",
            &crate::budget::compact(sum.input),
            &crate::budget::compact(sum.output),
            &crate::budget::compact(sum.cache_read),
            &crate::budget::compact(sum.cache_write),
            &self.total().render(),
            "",
        ]));
        lines.push(String::new());
        lines.push(format!(
            "last {RECENT_CALLS} calls: median input tokens per call and the share of it read \
             from the cache. Unpriced models show as unknown; --prices FILE adds or overrides \
             prices."
        ));
        lines
    }
}

/// `claude-opus-5` for `bedrock/global.anthropic.claude-opus-5`.
pub fn short_model(model: &str) -> &str {
    let model = model.rsplit('/').next().unwrap_or(model);
    model.rsplit_once("anthropic.").map_or(model, |(_, rest)| rest)
}

fn truncate(text: &str, max: usize) -> String {
    match text.chars().count() > max {
        true => text.chars().take(max - 1).chain(['…']).collect(),
        false => text.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(input: u64, output: u64, cache_read: u64, cache_write: u64) -> Counts {
        Counts {
            calls: 1,
            input,
            output,
            cache_read,
            cache_write,
        }
    }

    #[test]
    fn the_builtin_table_prices_claude_tiers_by_their_most_specific_key() {
        let prices = Prices::default();
        let input = |model: &str| prices.price(model).map(|price| price.input);
        assert_eq!(input("global.anthropic.claude-opus-5"), Some(5.0));
        assert_eq!(input("us.anthropic.claude-opus-4-1-20250805-v1:0"), Some(15.0));
        assert_eq!(input("claude-opus-4-20250514"), Some(15.0));
        assert_eq!(input("claude-3-opus-20240229"), Some(15.0));
        assert_eq!(input("bedrock/global.anthropic.claude-sonnet-4-5"), Some(3.0));
        assert_eq!(input("claude-3-5-haiku-latest"), Some(0.8));
        assert_eq!(input("anthropic.claude-3-haiku-20240307-v1:0"), Some(0.25));
        assert_eq!(input("us.anthropic.claude-haiku-4-5"), Some(1.0));
        assert_eq!(input("CLAUDE-SONNET-5"), Some(3.0), "case-insensitive");
        assert_eq!(input("openai.gpt-oss-120b"), None);
    }

    #[test]
    fn cache_reads_and_writes_cost_a_share_of_the_input_price() {
        let price = Prices::default().price("claude-sonnet-4-5").unwrap();
        // 1M uncached at $3, 1M out at $15, 2M read at $0.30, 1M written at $3.75.
        let cost = price.cost(&counts(4_000_000, 1_000_000, 2_000_000, 1_000_000));
        assert!((cost - (3.0 + 15.0 + 0.6 + 3.75)).abs() < 1e-9, "{cost}");
    }

    #[test]
    fn a_price_file_overrides_and_extends_the_table() {
        let json = r#"{"opus": {"input": 1, "output": 2, "cache_read": 0.5}, "gpt-oss": {"input": 0.15, "output": 0.6}}"#;
        let prices = Prices::with_overrides(json).unwrap();
        let opus = prices.price("claude-opus-4-1").unwrap();
        assert_eq!((opus.input, opus.cache_read), (1.0, Some(0.5)), "user keys first");
        assert_eq!(prices.price("openai.gpt-oss-120b").unwrap().output, 0.6);
        assert_eq!(
            prices.price("claude-sonnet-4").unwrap().input,
            3.0,
            "rest built in"
        );
        let error = Prices::with_overrides(r#"{"x": {"input": "cheap"}}"#).unwrap_err();
        assert!(error.contains("bad price table"), "{error}");
    }

    #[test]
    fn call_stats_cover_the_last_own_calls_only() {
        let mut ledger = Ledger::default();
        for input in 1..=60u64 {
            ledger.record("m", counts(input * 1000, 10, input * 500, 0), true);
        }
        ledger.record("judge", counts(1_000_000, 10, 0, 0), false);
        let stats = ledger.call_stats().unwrap();
        assert_eq!(stats.calls, RECENT_CALLS);
        // Calls 11..=60: median of 50 values is (35k + 36k) / 2.
        assert_eq!(stats.median_input, 35_500);
        assert!((stats.cache_hit - 0.5).abs() < 1e-9);
        assert_eq!(ledger.by_model["m"].calls, 60);
        assert_eq!(ledger.by_model["judge"].input, 1_000_000);
        assert_eq!(Ledger::default().call_stats(), None);
    }

    #[test]
    fn untracked_tokens_use_the_fallback_model_and_unpriced_ones_are_unknown() {
        let prices = Prices::default();
        let mut ledger = Ledger::default();
        ledger.record("claude-sonnet-4-5", counts(1_000_000, 0, 0, 0), true);
        let agent = AgentCost {
            label: "a1".into(),
            // 1M more input than the ledger has: saved before per-model tracking.
            totals: counts(2_000_000, 0, 0, 0),
            ledger: ledger.clone(),
            fallback_model: Some("claude-opus-5".into()),
        };
        assert_eq!(agent.estimate(&prices).render(), "$8.00");
        assert_eq!(agent.main_model(), Some("claude-sonnet-4-5"));

        ledger.record("mystery-model", counts(1_000_000, 0, 0, 0), true);
        let partly = AgentCost {
            totals: counts(2_000_000, 0, 0, 0),
            ledger,
            fallback_model: None,
            ..agent.clone()
        };
        assert_eq!(partly.estimate(&prices).render(), "$3.00+?");
        let none = AgentCost {
            totals: counts(5, 5, 0, 0),
            ledger: Ledger::default(),
            fallback_model: None,
            ..agent
        };
        assert_eq!(none.estimate(&prices).render(), "unknown");
        assert_eq!(Estimate::default().render(), "$0.00");
    }

    #[test]
    fn the_report_is_a_labelled_table_with_a_total() {
        let mut ledger = Ledger::default();
        ledger.record(
            "global.anthropic.claude-opus-5",
            counts(1_000_000, 100_000, 800_000, 0),
            true,
        );
        let report = CostReport {
            agents: vec![AgentCost {
                label: "lazy-mcp (a2)".into(),
                totals: counts(1_000_000, 100_000, 800_000, 0),
                ledger,
                fallback_model: None,
            }],
            prices: Prices::default(),
        };
        let lines = report.lines();
        // 200k uncached × $5 + 100k out × $25 + 800k read × $0.50 = 1 + 2.5 + 0.4.
        assert!(lines[0].starts_with("Estimated cost: $3.90 ("), "{}", lines[0]);
        assert!(lines[0].contains("not a bill"));
        let row = &lines[3];
        assert!(row.starts_with("lazy-mcp (a2)"), "{row}");
        assert!(row.contains("claude-opus-5"), "{row}");
        assert!(row.contains("$3.90"), "{row}");
        assert!(row.ends_with("1.0M/call 80%"), "{row}");
        assert!(lines[4].starts_with("total"), "{}", lines[4]);
    }

    #[test]
    fn model_ids_are_shortened_for_display() {
        assert_eq!(
            short_model("bedrock/global.anthropic.claude-opus-5"),
            "claude-opus-5"
        );
        assert_eq!(short_model("anthropic/claude-sonnet-4-5"), "claude-sonnet-4-5");
        assert_eq!(short_model("gpt-5"), "gpt-5");
    }
}
