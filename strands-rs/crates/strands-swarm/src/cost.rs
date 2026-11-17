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

/// [`CACHE_WRITE_FACTOR`] for writes with the one-hour cache TTL (`--cache-ttl 1h`).
pub const CACHE_WRITE_1H_FACTOR: f64 = 2.0;

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
    // Bedrock's global GPT-6 Astra price (AWS Price List); GPT-6.2 Astra's is unlisted yet.
    ("-astra", 10.0, 50.0),
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
    /// When it ended (Unix seconds); 0 for calls saved before this was kept.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub at: i64,
}

fn is_zero(at: &i64) -> bool {
    *at == 0
}

/// How far back [`CostReport::cache_hit_since`] looks for the header and `/cost`: the swarm's
/// cache hit rate now, not one weighed down by every call since the run started.
pub const RECENT_HIT_SECS: i64 = 3600;

/// An agent's tokens by model plus its most recent calls; persisted with the agent.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ledger {
    /// Model id → tokens spent on it.
    #[serde(default)]
    pub by_model: BTreeMap<String, Counts>,
    /// The last [`RECENT_CALLS`] own calls (side calls such as the goal judge left out).
    #[serde(default)]
    pub recent: VecDeque<Call>,
    /// Side calls by purpose ([`SidePurpose`]) and model; also counted in `by_model`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub side: BTreeMap<String, BTreeMap<String, Counts>>,
}

/// What a side call (one not carrying the agent's own context) was for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidePurpose {
    /// The goal judge reviewing a finished turn.
    Judge,
    /// Summarizing older turns ([`crate::compaction::CompactAt`]).
    Summarizer,
    /// Anything else on the agent's auxiliary model.
    Other,
}

impl SidePurpose {
    /// Told apart by the system prompt each one sends.
    pub fn of(system_prompt: Option<&str>) -> Self {
        use strands_agents::context_manager::strategies::SUMMARIZATION_SYSTEM_PROMPT;
        use strands_agents::plugins::vended::goal::JUDGE_SYSTEM_PROMPT;
        match system_prompt {
            Some(prompt) if prompt.starts_with(JUDGE_SYSTEM_PROMPT) => Self::Judge,
            Some(prompt) if prompt.starts_with(SUMMARIZATION_SYSTEM_PROMPT) => Self::Summarizer,
            _ => Self::Other,
        }
    }

    /// `judge`, `summarizer` or `other`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Judge => "judge",
            Self::Summarizer => "summarizer",
            Self::Other => "other",
        }
    }
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

    /// Count one side call for `purpose` on `model` (in `by_model` too).
    pub fn record_side(&mut self, purpose: SidePurpose, model: &str, counts: Counts) {
        self.record(model, counts, None);
        let by_model = self.side.entry(purpose.name().to_owned()).or_default();
        by_model.entry(model.to_owned()).or_default().add(&counts);
    }

    /// Count one model call on `model`. `own` is when one of the agent's own calls ended (Unix
    /// seconds); those also feed [`Ledger::call_stats`] and [`CostReport::cache_hit_since`].
    pub fn record(&mut self, model: &str, counts: Counts, own: Option<i64>) {
        self.by_model.entry(model.to_owned()).or_default().add(&counts);
        if let Some(at) = own {
            self.recent.push_back(Call {
                input: counts.input,
                cache_read: counts.cache_read,
                at,
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
    write_factor: f64,
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
            write_factor: CACHE_WRITE_FACTOR,
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

    /// Price cache writes without a price of their own at `factor` × input, e.g.
    /// [`CACHE_WRITE_1H_FACTOR`].
    pub fn with_cache_write_factor(mut self, factor: f64) -> Self {
        self.write_factor = factor;
        self
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
        let mut price = best(&self.user).or_else(|| best(&self.builtin))?;
        price.cache_write = price.cache_write.or(Some(price.input * self.write_factor));
        Some(price)
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
    /// `counts` on `model` (none for no tokens; unknown without a model or a price).
    fn of(prices: &Prices, model: Option<&str>, counts: &Counts) -> Self {
        if counts.is_empty() {
            return Self::default();
        }
        let price = model.and_then(|model| prices.price(model));
        Self {
            usd: price.map_or(0.0, |price| price.cost(counts)),
            unknown: price.is_none(),
            any: true,
        }
    }

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

/// `$12.30`, `$123`, `$0.0012`.
pub(crate) fn dollars(usd: f64) -> String {
    match usd {
        usd if usd >= 100.0 => format!("${usd:.0}"),
        usd if usd >= 0.01 || usd == 0.0 => format!("${usd:.2}"),
        usd => format!("${usd:.4}"),
    }
}

/// One agent's spend, for [`CostReport`].
#[derive(Debug, Clone, PartialEq)]
pub struct AgentCost {
    /// Id.
    pub id: String,
    /// Parent id.
    pub parent: String,
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
        let mut priced =
            |model: Option<&str>, counts: &Counts| total.add(Estimate::of(prices, model, counts));
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
                id: agent.id.clone(),
                parent: agent.parent.clone(),
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

    /// Share of input read from the prompt cache over own calls that ended at or after `since`
    /// (Unix seconds), across agents; `None` without any. Each agent keeps only its last
    /// [`RECENT_CALLS`], so a busier one counts with fewer of its calls.
    pub fn cache_hit_since(&self, since: i64) -> Option<f64> {
        let (input, read) = self
            .agents
            .iter()
            .flat_map(|agent| &agent.ledger.recent)
            .filter(|call| call.at >= since)
            .fold((0, 0), |(input, read), call| {
                (input + call.input, read + call.cache_read)
            });
        (input > 0).then(|| read as f64 / input as f64)
    }

    /// Recent and run-long cache hit rates at `now` (Unix seconds), e.g. `Cache reads: 97% of
    /// input over the last hour, 91% since the run started.`
    pub fn cache_line(&self, now: i64) -> Option<String> {
        let (input, read) = self.agents.iter().fold((0, 0), |(input, read), agent| {
            (input + agent.totals.input, read + agent.totals.cache_read)
        });
        let percent = |share: f64| format!("{:.0}%", share * 100.0);
        let total = (input > 0).then(|| percent(read as f64 / input as f64))?;
        Some(match self.cache_hit_since(now - RECENT_HIT_SECS) {
            Some(recent) => format!(
                "Cache reads: {} of input over the last hour, {total} since the run started.",
                percent(recent)
            ),
            None => format!("Cache reads: {total} of input since the run started."),
        })
    }

    /// What `id` and its subtree (descendants, cancelled ones included) spent; `None` for an
    /// unknown id.
    pub fn spend(&self, id: &str) -> Option<Spend> {
        let own = self.agents.iter().find(|agent| agent.id == id)?;
        let mut subtree = vec![own];
        let mut next = 0;
        while let Some(agent) = subtree.get(next) {
            let parent = agent.id.clone();
            subtree.extend(self.agents.iter().filter(|agent| agent.parent == parent));
            next += 1;
        }
        let mut spend = Spend {
            own: own.estimate(&self.prices),
            own_tokens: own.totals.input + own.totals.output,
            descendants: subtree.len() - 1,
            ..Spend::default()
        };
        for agent in subtree {
            spend.subtree.add(agent.estimate(&self.prices));
            spend.subtree_tokens += agent.totals.input + agent.totals.output;
        }
        Some(spend)
    }

    /// Side calls across the swarm by purpose: tokens and estimate.
    pub fn side_calls(&self) -> BTreeMap<String, (Counts, Estimate)> {
        let mut out: BTreeMap<String, (Counts, Estimate)> = BTreeMap::new();
        for agent in &self.agents {
            for (purpose, by_model) in &agent.ledger.side {
                let (total, estimate) = out.entry(purpose.clone()).or_default();
                for (model, counts) in by_model {
                    total.add(counts);
                    estimate.add(Estimate::of(&self.prices, Some(model), counts));
                }
            }
        }
        out
    }

    /// The report as a text table.
    pub fn lines(&self) -> Vec<String> {
        // The estimate follows the model, so it stays in view when a narrow terminal clips the
        // table.
        let row = |[name, model, estimate, input, output, read, write, stats]: [&str; 8]| {
            format!("{name:<24} {model:<22} {estimate:>9} {input:>7} {output:>7} {read:>8} {write:>8} {stats:>14}")
                .trim_end()
                .to_owned()
        };
        let counted = |name: &str, model: &str, estimate: Estimate, counts: &Counts, stats: &str| {
            let [input, output, read, write] =
                [counts.input, counts.output, counts.cache_read, counts.cache_write]
                    .map(crate::text::compact);
            row([
                name,
                model,
                &estimate.render(),
                &input,
                &output,
                &read,
                &write,
                stats,
            ])
        };
        let mut lines = vec![
            format!(
                "Estimated cost: {} (list prices per 1M tokens; cache read {CACHE_READ_FACTOR}×, \
                 write {}× input; not a bill)",
                self.total().render(),
                self.prices.write_factor
            ),
            String::new(),
            row([
                "agent",
                "model",
                "est. $",
                "in",
                "out",
                "cache rd",
                "cache wr",
                &format!("last {RECENT_CALLS} calls"),
            ]),
        ];
        let mut sum = Counts::default();
        for agent in &self.agents {
            sum.add(&agent.totals);
            let stats = agent.ledger.call_stats().map_or_else(String::new, |stats| {
                format!(
                    "{}/call {:.0}%",
                    crate::text::compact(stats.median_input),
                    stats.cache_hit * 100.0
                )
            });
            let model = agent.main_model().map(short_model).unwrap_or("-");
            lines.push(counted(
                &crate::text::clip(&agent.label, 23),
                &crate::text::clip(model, 21),
                agent.estimate(&self.prices),
                &agent.totals,
                &stats,
            ));
        }
        lines.push(counted("total", "", self.total(), &sum, ""));
        if let Some(line) = self.cache_line(chrono::Utc::now().timestamp()) {
            lines.push(line);
        }
        let side = self.side_calls();
        if !side.is_empty() {
            lines.push(String::new());
            lines.push("side calls, part of the totals above:".to_owned());
            for (purpose, (counts, estimate)) in &side {
                let calls = format!("{} calls", counts.calls);
                lines.push(counted(&format!("  {purpose}"), &calls, *estimate, counts, ""));
            }
        }
        lines.push(String::new());
        lines.push(format!(
            "last {RECENT_CALLS} calls: median input tokens per call and the share of it read \
             from the cache. Unpriced models show as unknown; --prices FILE adds or overrides \
             prices."
        ));
        lines
    }
}

/// One agent's spend and its subtree's, from [`CostReport::spend`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Spend {
    /// The agent's own estimate.
    pub own: Estimate,
    /// Its input + output tokens.
    pub own_tokens: u64,
    /// The agent and its descendants.
    pub subtree: Estimate,
    /// Their input + output tokens.
    pub subtree_tokens: u64,
    /// How many descendants.
    pub descendants: usize,
}

impl Spend {
    /// `est. $4.10 / 12.3M tokens (with 2 descendants $9.80 / 30.1M)`.
    pub fn describe(&self) -> String {
        let own = format!(
            "est. {} / {} tokens",
            self.own.render(),
            crate::text::compact(self.own_tokens)
        );
        match self.descendants {
            0 => own,
            n => format!(
                "{own} (with {n} descendant{} {} / {})",
                if n == 1 { "" } else { "s" },
                self.subtree.render(),
                crate::text::compact(self.subtree_tokens)
            ),
        }
    }
}

/// `claude-opus-5` for `bedrock/global.anthropic.claude-opus-5`.
pub fn short_model(model: &str) -> &str {
    let model = model.rsplit('/').next().unwrap_or(model);
    model.rsplit_once("anthropic.").map_or(model, |(_, rest)| rest)
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

    /// Agent `id` (also its label) under `parent`, without a fallback model.
    fn agent(id: &str, parent: &str, totals: Counts, ledger: Ledger) -> AgentCost {
        AgentCost {
            id: id.into(),
            parent: parent.into(),
            label: id.into(),
            totals,
            ledger,
            fallback_model: None,
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
        assert_eq!(input("bedrock-mantle/openai.gpt-6-astra"), Some(10.0));
        assert_eq!(
            input("bedrock/us.openai.gpt-6.2-astra"),
            Some(10.0),
            "falls back to 6"
        );
        assert_eq!(prices.price("openai.gpt-6-astra").unwrap().output, 50.0);
    }

    #[test]
    fn cache_reads_and_writes_cost_a_share_of_the_input_price() {
        let price = Prices::default().price("claude-sonnet-4-5").unwrap();
        // 1M uncached at $3, 1M out at $15, 2M read at $0.30, 1M written at $3.75.
        let cost = price.cost(&counts(4_000_000, 1_000_000, 2_000_000, 1_000_000));
        assert!((cost - (3.0 + 15.0 + 0.6 + 3.75)).abs() < 1e-9, "{cost}");
    }

    #[test]
    fn one_hour_cache_writes_cost_twice_the_input_price_unless_priced() {
        let prices = Prices::default().with_cache_write_factor(CACHE_WRITE_1H_FACTOR);
        let price = prices.price("claude-sonnet-4-5").unwrap();
        assert_eq!(price.cache_write, Some(6.0));
        let json = r#"{"sonnet": {"input": 3, "output": 15, "cache_write": 5}}"#;
        let priced = Prices::with_overrides(json)
            .unwrap()
            .with_cache_write_factor(CACHE_WRITE_1H_FACTOR);
        assert_eq!(priced.price("claude-sonnet-4-5").unwrap().cache_write, Some(5.0));
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
            ledger.record("m", counts(input * 1000, 10, input * 500, 0), Some(0));
        }
        ledger.record("judge", counts(1_000_000, 10, 0, 0), None);
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
        ledger.record("claude-sonnet-4-5", counts(1_000_000, 0, 0, 0), Some(0));
        let old = AgentCost {
            // 1M more input than the ledger has: saved before per-model tracking.
            fallback_model: Some("claude-opus-5".into()),
            ..agent("a1", "user", counts(2_000_000, 0, 0, 0), ledger.clone())
        };
        assert_eq!(old.estimate(&prices).render(), "$8.00");
        assert_eq!(old.main_model(), Some("claude-sonnet-4-5"));

        ledger.record("mystery-model", counts(1_000_000, 0, 0, 0), Some(0));
        let partly = agent("a1", "user", counts(2_000_000, 0, 0, 0), ledger);
        assert_eq!(partly.estimate(&prices).render(), "$3.00+?");
        let none = agent("a1", "user", counts(5, 5, 0, 0), Ledger::default());
        assert_eq!(none.estimate(&prices).render(), "unknown");
        assert_eq!(Estimate::default().render(), "$0.00");
    }

    #[test]
    fn the_report_is_a_labelled_table_with_a_total() {
        let mut ledger = Ledger::default();
        ledger.record(
            "global.anthropic.claude-opus-5",
            counts(1_000_000, 100_000, 800_000, 0),
            Some(0),
        );
        let report = CostReport {
            agents: vec![AgentCost {
                label: "lazy-mcp (a2)".into(),
                ..agent("a2", "a1", counts(1_000_000, 100_000, 800_000, 0), ledger)
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
        // The one call is dated 0, long before the last hour.
        assert_eq!(lines[5], "Cache reads: 80% of input since the run started.");
    }

    #[test]
    fn the_recent_cache_hit_rate_counts_only_calls_of_the_last_hour() {
        let now = 1_000_000;
        let mut old = Ledger::default();
        // Hours ago, before caching worked: drags the run-long rate down.
        old.record("m", counts(900, 0, 0, 0), Some(now - 7200));
        old.record("m", counts(100, 0, 90, 0), Some(now - 60));
        let mut busy = Ledger::default();
        busy.record("m", counts(100, 0, 100, 0), Some(now - 3600));
        // A call saved before calls were dated counts in no window.
        busy.recent.push_back(Call {
            input: 100,
            cache_read: 0,
            at: 0,
        });
        let report = CostReport {
            agents: vec![
                agent("a1", "user", counts(1000, 0, 90, 0), old),
                agent("a2", "a1", counts(200, 0, 100, 0), busy),
            ],
            prices: Prices::default(),
        };
        assert_eq!(report.cache_hit_since(now - RECENT_HIT_SECS), Some(0.95));
        assert_eq!(report.cache_hit_since(now + 1), None);
        assert_eq!(
            report.cache_line(now).unwrap(),
            "Cache reads: 95% of input over the last hour, 16% since the run started."
        );
        let saved = serde_json::to_string(&report.agents[1].ledger).unwrap();
        assert!(saved.contains(r#"{"input":100,"cache_read":0}"#), "{saved}");
        let back: Ledger = serde_json::from_str(&saved).unwrap();
        assert_eq!(back, report.agents[1].ledger);
    }

    #[test]
    fn side_calls_are_told_apart_by_their_system_prompt() {
        use strands_agents::context_manager::strategies::SUMMARIZATION_SYSTEM_PROMPT;
        use strands_agents::plugins::vended::goal::JUDGE_SYSTEM_PROMPT;
        let judge = format!("{JUDGE_SYSTEM_PROMPT}\n\nGoal: ship it");
        assert_eq!(SidePurpose::of(Some(&judge)), SidePurpose::Judge);
        assert_eq!(
            SidePurpose::of(Some(SUMMARIZATION_SYSTEM_PROMPT)),
            SidePurpose::Summarizer
        );
        assert_eq!(SidePurpose::of(Some("You are a poet.")), SidePurpose::Other);
        assert_eq!(SidePurpose::of(None), SidePurpose::Other);
    }

    #[test]
    fn side_calls_count_in_the_totals_and_get_their_own_rows() {
        let opus = "global.anthropic.claude-opus-5";
        let mut ledger = Ledger::default();
        ledger.record(opus, counts(1_000_000, 0, 0, 0), Some(0));
        ledger.record_side(SidePurpose::Judge, opus, counts(200_000, 0, 0, 0));
        ledger.record_side(SidePurpose::Judge, opus, counts(200_000, 0, 0, 0));
        ledger.record_side(SidePurpose::Summarizer, opus, counts(600_000, 0, 0, 0));
        assert_eq!(
            ledger.by_model[opus].input, 2_000_000,
            "side calls are in the totals"
        );
        assert_eq!(
            ledger.call_stats().unwrap().calls,
            1,
            "but not in the own-call stats"
        );
        let report = CostReport {
            agents: vec![agent("a1", "user", counts(2_000_000, 0, 0, 0), ledger.clone())],
            prices: Prices::default(),
        };
        let side = report.side_calls();
        assert_eq!(side["judge"].0.calls, 2);
        assert_eq!(side["judge"].0.input, 400_000);
        assert_eq!(side["judge"].1.render(), "$2.00");
        assert_eq!(side["summarizer"].1.render(), "$3.00");
        let lines = report.lines();
        let at = lines
            .iter()
            .position(|line| line == "side calls, part of the totals above:")
            .expect("no side-call section");
        assert!(lines[at + 1].starts_with("  judge"), "{}", lines[at + 1]);
        assert!(lines[at + 1].contains("2 calls"), "{}", lines[at + 1]);
        assert!(lines[at + 1].contains("$2.00"), "{}", lines[at + 1]);
        assert!(lines[at + 2].starts_with("  summarizer"), "{}", lines[at + 2]);
        // The ledger persists with its side map and reads back.
        let json = serde_json::to_string(&ledger).unwrap();
        let back: Ledger = serde_json::from_str(&json).unwrap();
        assert_eq!(back.side, ledger.side);
        let old: Ledger = serde_json::from_str(r#"{"by_model":{}}"#).unwrap();
        assert!(old.side.is_empty(), "ledgers saved before side calls still load");
    }

    #[test]
    fn spend_adds_up_an_agent_and_all_its_descendants() {
        let sonnet = |id: &str, parent: &str, input: u64| {
            let mut ledger = Ledger::default();
            ledger.record("claude-sonnet-4-5", counts(input, 0, 0, 0), Some(0));
            agent(id, parent, counts(input, 0, 0, 0), ledger)
        };
        let report = CostReport {
            agents: vec![
                sonnet("a1", "user", 1_000_000),
                sonnet("a2", "a1", 2_000_000),
                sonnet("a3", "a2", 1_000_000),
                sonnet("a4", "user", 5_000_000),
            ],
            prices: Prices::default(),
        };
        let root = report.spend("a1").unwrap();
        assert_eq!((root.descendants, root.subtree_tokens), (2, 4_000_000));
        assert_eq!(
            root.describe(),
            "est. $3.00 / 1.0M tokens (with 2 descendants $12.00 / 4.0M)"
        );
        assert_eq!(report.spend("a3").unwrap().describe(), "est. $3.00 / 1.0M tokens");
        assert_eq!(report.spend("a9"), None);
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
