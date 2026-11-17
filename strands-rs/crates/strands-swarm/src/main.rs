//! `strands-swarm`: run a model-driven agent swarm in the terminal.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use strands_agents::tools::Tool;
use strands_harness::Effort;
use strands_swarm::logging;
use strands_swarm::mcp::MCP_CACHE_FILE;
use strands_swarm::state;
use strands_swarm::tui::Exit;
use strands_swarm::RELOAD_EXIT_CODE;
use strands_swarm::{
    discover_skills, harness_factory, harness_model_resolver, load_shared_mcp, HarnessFactoryOptions,
    McpOptions, Memory, Skills, SpawnRequest, Swarm, SwarmConfig, SwarmEvent, USER,
};
use tokio::sync::mpsc::UnboundedReceiver;

/// Run a swarm of cooperating coding agents. The root agent decides whether and how to spawn
/// others.
#[derive(Parser, Debug)]
#[command(version = env!("STRANDS_SWARM_VERSION"))]
struct Cli {
    /// Task for the root agent. Optional in the TUI, where you can type it.
    task: Option<String>,
    /// Default model for every agent (`provider/name` or a Bedrock model id).
    #[arg(long)]
    model: Option<String>,
    /// Model for side calls (the goal judge, context summarization), e.g. a cheaper, faster one.
    /// Default: each agent's own model.
    #[arg(long, alias = "judge-model")]
    side_model: Option<String>,
    /// Reasoning effort.
    #[arg(long, default_value = "auto")]
    effort: Effort,
    /// Prompt-cache TTL on Bedrock/Anthropic: `5m` (the provider default) or `1h`. One-hour
    /// entries survive long waits and reloads but cost 2x input to write instead of 1.25x, so
    /// they pay off only when an agent often idles 5-60 minutes between calls.
    #[arg(long, value_parser = ["5m", "1h"])]
    cache_ttl: Option<String>,
    /// Do not keep the 5-minute prompt cache warm while an agent waits on a tool call (by
    /// default its last request is re-sent every 4.5 minutes, up to 12 times, and dropped at the
    /// first token: a cache read at 0.1x input instead of a rewrite at 1.25x). Off with `1h`.
    #[arg(long)]
    no_keep_warm: bool,
    /// Working directory of the root agent.
    #[arg(long, default_value = ".")]
    cwd: PathBuf,
    /// Maximum number of live agents, root included; further spawns queue until a slot frees.
    #[arg(long, default_value_t = 16)]
    max_agents: usize,
    /// Maximum number of model calls in flight across the swarm; more queue. 0 means no limit.
    #[arg(long, default_value_t = 8)]
    max_concurrent_calls: usize,
    /// Ring the terminal bell when an agent messages you (TUI only; tmux and most terminals turn
    /// it into a window flag or desktop notification).
    #[arg(long)]
    notify: bool,
    /// Branch that `/status` and `list_agents` compare agent branches against, and that a
    /// cancelled agent's branch must be merged into to be deleted; default: the branch checked
    /// out where the swarm started.
    #[arg(long)]
    status_base: Option<String>,
    /// Give agents the `land` tool: rebase their commits onto a branch they do not own in a
    /// scratch worktree, run the gates there, and fast-forward it on success.
    #[arg(long)]
    allow_land: bool,
    /// A further branch (glob: `*`, `?`) `land` may fast-forward besides the root agent's
    /// checked-out branch (repeatable), e.g. `--land-into 'release/*'`.
    #[arg(long = "land-into", value_name = "GLOB")]
    land_into: Vec<String>,
    /// Check command `land` runs when the caller names none (repeatable), e.g.
    /// `--gate 'cargo test'`.
    #[arg(long = "gate", value_name = "CMD")]
    gates: Vec<String>,
    /// Carry out one landing from a plan file, then exit (how `land` jobs run).
    #[arg(long, value_name = "PLAN", hide = true)]
    land: Option<PathBuf>,
    /// Swarm-wide token budget (input + output, e.g. `5M`); once spent, model calls stop until it
    /// is raised (TUI `/budget N`). A resumed run keeps its last budget unless given a different
    /// `--budget-tokens` than it was last started with.
    #[arg(long, value_parser = strands_swarm::text::parse_tokens)]
    budget_tokens: Option<u64>,
    /// Tell the root agents (and the TUI) once free space on the run's, the worktrees' or the
    /// cargo target dirs' filesystem drops below this (e.g. `20G`, `512M`); `0` turns it off.
    #[arg(long, value_parser = strands_swarm::text::parse_bytes, default_value = "20G")]
    min_free_disk: u64,
    /// Summarize an agent's older turns once a call's input reaches this many tokens (e.g. `200k`,
    /// `1.5M`), or 85% of the model's window if that is smaller.
    #[arg(long, value_parser = strands_swarm::text::parse_tokens, default_value = "200k")]
    compact_at: u64,
    /// File with extra instructions given to every agent.
    #[arg(long)]
    instructions: Option<PathBuf>,
    /// The user's memory file, shown to every agent and written by `remember` / `/remember`
    /// (default `$XDG_CONFIG_HOME/strands-swarm/memory.md`, `XDG_CONFIG_HOME` defaulting to
    /// `~/.config`).
    #[arg(long, conflicts_with = "no_memory")]
    memory: Option<PathBuf>,
    /// Run without the memory file and the `remember`/`forget` tools.
    #[arg(long)]
    no_memory: bool,
    /// Give the root agent a goal: a judge re-checks it after every turn and sends the agent back
    /// to work until it is met. Without a task, the goal is also the task.
    #[arg(long)]
    goal: Option<String>,
    /// `mcp.json`-shaped file (or `~/.claude.json`) whose servers every agent can use. Repeatable.
    #[arg(long = "mcp-config")]
    mcp_configs: Vec<PathBuf>,
    /// Regex over `<server>_<tool>` names to hide from agents. Repeatable.
    #[arg(long = "mcp-deny")]
    mcp_deny: Vec<String>,
    /// Register every MCP tool on every agent up front instead of on demand through `find_tools`.
    #[arg(long)]
    mcp_eager: bool,
    /// Print the MCP tools agents would get (and those denied), then exit.
    #[arg(long)]
    list_tools: bool,
    /// Resume a throwaway copy of a run directory on a stub model, check that every agent replies
    /// and the TUI renders, then exit (0 on success). Used by `reload_swarm` on a new binary.
    #[arg(long, value_name = "RUN_DIR_COPY")]
    self_test: Option<PathBuf>,
    /// Directory of skills (`<name>/SKILL.md`), or one skill directory, offered to every agent
    /// ahead of the default locations. Repeatable.
    #[arg(long = "skills-dir")]
    skills_dirs: Vec<PathBuf>,
    /// Skip the default skill locations: the project's `.claude/skills`, `.agents/skills` and
    /// `.agent/skills`, `~/.claude/skills`, enabled Claude Code plugins, `~/.codex/skills` and
    /// `~/.agents/skills`.
    #[arg(long)]
    no_default_skills: bool,
    /// Print the skills agents would get (and those shadowed by an earlier one), then exit.
    #[arg(long)]
    list_skills: bool,
    /// Name of a persistent run. State lives in `$XDG_STATE_HOME/strands-swarm/<name>`
    /// (`XDG_STATE_HOME` defaulting to `~/.local/state`); starting again with the same name
    /// resumes it (agents, conversations, unread messages).
    #[arg(long)]
    run: Option<String>,
    /// No TUI: print events to stderr and messages to the user to stdout; exit once the root has
    /// replied and no agent is busy (non-zero when that reply is an error). Needs a task or
    /// `--goal` unless resuming.
    #[arg(long)]
    headless: bool,
    /// Print the run's journal (every message to and from the user) and exit. Needs `--run`.
    #[arg(long, requires = "run")]
    journal: bool,
    /// Reclaim the disk of the run's cancelled worktree agents (their private `CARGO_TARGET_DIR`,
    /// their worktree when it has no uncommitted changes, their branch once merged), report, exit.
    /// Needs `--run`. A live swarm does this by itself; the TUI's `/gc` retries.
    #[arg(long, requires = "run")]
    gc: bool,
    /// Print the run's estimated cost per agent (tokens priced from a built-in list-price table,
    /// see `--prices`) and exit. Needs `--run`.
    #[arg(long, requires = "run")]
    cost: bool,
    /// JSON price table (USD per 1M tokens) over the built-in one, keyed by model-id substring:
    /// `{"opus": {"input": 5, "output": 25, "cache_read": 0.5, "cache_write": 6.25}}`; cache
    /// prices default to 0.1x and 1.25x input.
    #[arg(long, value_name = "FILE")]
    prices: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    strands_swarm::watchdog::spawn(Duration::from_secs(10), |report| {
        // Under swarmctl stderr goes to the run's stderr.log.
        release_terminal();
        eprintln!("strands-swarm: {report}");
        std::process::exit(strands_swarm::watchdog::DEADLOCK_EXIT_CODE);
    });
    match run(cli).await {
        Ok(Exit::Quit) => ExitCode::SUCCESS,
        Ok(Exit::Reload) => ExitCode::from(RELOAD_EXIT_CODE),
        Err(error) => {
            eprintln!("strands-swarm: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<Exit, Box<dyn std::error::Error>> {
    if let Some(path) = &cli.land {
        let plan: strands_swarm::land::LandPlan = serde_json::from_slice(&std::fs::read(path)?)?;
        let stop = strands_swarm::land::stop_signal();
        let outcome = strands_swarm::land::run_until(&plan, stop).await;
        // Done with: only a job killed before this leaves its plan (see `land::tidy`).
        let _ = std::fs::remove_file(path);
        return match outcome {
            Ok(report) => {
                println!("{report}");
                Ok(Exit::Quit)
            }
            Err(report) => {
                println!("{report}");
                Err("not landed".into())
            }
        };
    }
    if let Some(dir) = &cli.self_test {
        eprintln!("strands-swarm: {}", strands_swarm::selftest::run(dir).await?);
        return Ok(Exit::Quit);
    }
    if cli.gc {
        let (dir, saved) = load_run(cli.run.as_deref().unwrap_or_default()).await?;
        let mut collected = strands_swarm::gc::collect_run(&saved, cli.status_base.as_deref()).await;
        let jobs = strands_swarm::jobs::Jobs::new(dir.join("jobs"));
        let shared = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from);
        let target =
            strands_swarm::land::target_dir(shared.as_deref(), &saved.run_id, &jobs.dir().join("land"));
        collected.extend(strands_swarm::land::tidy_when_idle(&jobs, Some(&target)).await);
        let journal = strands_swarm::journal::Journal::open(&dir);
        for agent in collected.iter().filter(|c| !c.is_empty()) {
            journal.event("gc", agent.journal_fields());
            agent.lines().iter().for_each(|line| println!("{line}"));
        }
        journal.flush().await;
        println!("{}", strands_swarm::gc::summary(&collected));
        return Ok(Exit::Quit);
    }
    let prices = match &cli.prices {
        Some(path) => strands_swarm::cost::Prices::load(path)?,
        None => strands_swarm::cost::Prices::default(),
    };
    // Writes are priced at the TTL the run uses now; a run that switched TTL is estimated as if
    // it had always used this one.
    let prices = match cli.cache_ttl.as_deref() {
        Some("1h") => prices.with_cache_write_factor(strands_swarm::cost::CACHE_WRITE_1H_FACTOR),
        _ => prices,
    };
    let default_model = cli.model.as_deref().map_or_else(
        || strands_harness::defaults::DEFAULT_MODEL.to_owned(),
        |model| strands_swarm::runtime::expand_model(model, Some(strands_harness::defaults::DEFAULT_MODEL)),
    );
    if let (true, Some(name)) = (cli.cost, &cli.run) {
        let (_, saved) = load_run(name).await?;
        let report = strands_swarm::cost::CostReport::from_state(&saved, prices, Some(&default_model));
        report.lines().iter().for_each(|line| println!("{line}"));
        return Ok(Exit::Quit);
    }
    if let (true, Some(name)) = (cli.journal, &cli.run) {
        for entry in strands_swarm::journal::read(&state::run_dir(name)?)? {
            println!("{}", entry.render());
        }
        return Ok(Exit::Quit);
    }
    let tui = !(cli.headless || cli.list_tools || cli.list_skills);
    // Before anything resumes: agents must not start for a TUI that cannot.
    if tui && !has_terminal() {
        return Err(
            "the TUI needs a terminal and there is none (stdout is not a tty, or no tty is \
                    attached); run with --headless, or in tmux (swarmctl start)"
                .into(),
        );
    }
    if !(cli.list_tools || cli.list_skills) {
        let timeout = strands_swarm::watchdog::STARTUP_TIMEOUT;
        strands_swarm::watchdog::watch_startup(timeout, strands_swarm::watchdog::is_ready, move || {
            release_terminal();
            eprintln!(
                "strands-swarm: not up after {}s; exiting so that the supervisor restarts it (or rolls a new binary back)",
                timeout.as_secs()
            );
            std::process::exit(strands_swarm::watchdog::STARTUP_HANG_EXIT_CODE);
        });
    }
    let cwd = std::fs::canonicalize(&cli.cwd)?;
    let instructions = cli
        .instructions
        .as_ref()
        .map(std::fs::read_to_string)
        .transpose()?;
    let run_dir = cli.run.as_deref().map(state::run_dir).transpose()?;
    // The TUI owns the terminal, so warnings and MCP server logs (which would paint over it) go
    // to files; other modes keep them on stderr.
    let private_path = |in_run: &str, suffix: &str| match &run_dir {
        Some(dir) => dir.join(in_run),
        None => std::env::temp_dir().join(format!("strands-swarm-{}{suffix}", std::process::id())),
    };
    let log = tui.then(|| private_path(logging::LOG_FILE, ".log"));
    logging::init(
        log.as_deref()
            .map(|path| logging::open_log(path, logging::MAX_LOG_BYTES))
            .transpose()?,
    );
    if let Some(path) = &log {
        eprintln!("strands-swarm: warnings in {}", path.display());
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let skills = discover_skills(&cli.skills_dirs, !cli.no_default_skills, &cwd, home.as_deref())?;
    if cli.list_skills {
        print_skills(&skills);
        return Ok(Exit::Quit);
    }
    eprintln!(
        "strands-swarm: {} skills ({} shadowed; --list-skills shows them)",
        skills.skills.len(),
        skills.shadowed.len()
    );
    let mcp_logs = tui.then(|| private_path("logs", ""));
    let started = std::time::Instant::now();
    let mcp_options = McpOptions {
        deny: cli.mcp_deny.clone(),
        stderr_dir: mcp_logs.clone(),
        cache: run_dir.as_ref().map(|dir| dir.join(MCP_CACHE_FILE)),
        connect: cli.mcp_eager || cli.list_tools,
    };
    let mcp = load_shared_mcp(&cli.mcp_configs, mcp_options).await?;
    if !cli.mcp_configs.is_empty() {
        eprintln!(
            "strands-swarm: {} MCP servers ({} connected in {:.1}s, the rest on first use), {} tools shared ({}), {} denied",
            mcp.servers.len(),
            mcp.connected().await,
            started.elapsed().as_secs_f64(),
            mcp.tools.len(),
            if cli.mcp_eager {
                "eager"
            } else {
                "on demand via find_tools"
            },
            mcp.denied.len()
        );
        if let Some(dir) = &mcp_logs {
            eprintln!("strands-swarm: MCP server logs in {}", dir.display());
        }
    }
    if cli.list_tools {
        mcp.tools
            .iter()
            .for_each(|tool| println!("allow {}", tool.name()));
        mcp.denied.iter().for_each(|name| println!("deny  {name}"));
        return Ok(Exit::Quit);
    }
    let (tools, catalog) = match cli.mcp_eager {
        true => (mcp.tools.clone(), None),
        false => (Vec::new(), Some(mcp.catalog())),
    };
    let options = HarnessFactoryOptions {
        model: cli.model,
        effort: cli.effort,
        instructions,
        tools,
        catalog,
        skills: std::sync::Arc::new(skills.for_agents()),
        compact_at: Some(cli.compact_at),
        cache_ttl: cli.cache_ttl.clone(),
    };
    let (task, goal) = (cli.task, cli.goal);
    let mut config = SwarmConfig::new(harness_factory(options.clone()), cwd);
    let models = harness_model_resolver(options);
    if let Some(id) = cli.side_model {
        config.side_model = Some(models(Some(id)).await?);
    }
    config.models = Some(models);
    config.max_agents = cli.max_agents;
    config.max_concurrent_calls = cli.max_concurrent_calls;
    config.keep_warm = (!cli.no_keep_warm && cli.cache_ttl.as_deref() != Some("1h"))
        .then_some(strands_swarm::runtime::KEEP_WARM_EVERY);
    config.status_base = cli.status_base;
    config.allow_land = cli.allow_land;
    config.gates = cli.gates;
    config.land_into = cli.land_into;
    config.budget_tokens = cli.budget_tokens;
    config.task_flag = task.clone().filter(|task| !task.is_empty());
    config.compact_at = cli.compact_at;
    config.prices = Arc::new(prices);
    config.default_model = Some(default_model);
    if !cli.no_memory {
        config.memory = cli
            .memory
            .clone()
            .or_else(Memory::default_path)
            .map(|path| Arc::new(Memory::new(path)));
    }
    config.cargo_target_dir = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from);
    let saved = match (&cli.run, run_dir) {
        (Some(name), Some(dir)) => {
            config.run_id = name.clone();
            config.state_dir = Some(dir.clone());
            state::load(&dir).await?
        }
        _ => None,
    };
    let resumed = saved.is_some();
    let last_task = saved.as_ref().and_then(|saved| saved.task_flag.clone());
    let (swarm, events) = match saved {
        Some(saved) => {
            config.cwd = saved.cwd.clone();
            let agents = match saved.agents.len() {
                1 => "1 agent".to_owned(),
                n => format!("{n} agents"),
            };
            eprintln!("strands-swarm: resuming run {} ({agents})", saved.run_id);
            Swarm::resume(config, saved).await?
        }
        None => Swarm::new(config),
    };
    if cli.min_free_disk > 0 {
        let free = std::sync::Arc::new(strands_swarm::runtime::free_space);
        let guard =
            swarm
                .clone()
                .guard_disk(cli.min_free_disk, strands_swarm::runtime::DISK_CHECK_EVERY, free);
        tokio::spawn(guard);
    }
    let live_root = swarm
        .agents()
        .into_iter()
        .find(|agent| agent.parent == USER && agent.status.is_live());
    let plan = start_plan(task, goal, live_root.is_some(), last_task.as_deref());
    let (root, expect_reply) = match (plan, live_root) {
        (StartPlan::Message(task), Some(agent)) => {
            swarm.send(USER, &agent.id, task)?;
            (None, true)
        }
        (StartPlan::Spawn(request), _) => (Some(request), true),
        _ => (None, false),
    };

    if !cli.headless {
        return Ok(strands_swarm::tui::run(swarm, events, root, cli.notify).await?);
    }
    match root {
        Some(root) => drop(swarm.spawn(USER, root).await?),
        None if !resumed => return Err("--headless needs a task or a --goal".into()),
        None => {}
    }
    strands_swarm::watchdog::ready();
    let exit = run_headless(&swarm, events, expect_reply).await;
    swarm.shutdown().await;
    exit
}

/// Whether the TUI can take a terminal: stdout is one, and so is stdin or the process's
/// controlling terminal (where raw mode is set).
fn has_terminal() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
        && (std::io::stdin().is_terminal() || std::fs::File::open("/dev/tty").is_ok())
}

/// Give the screen back if the TUI holds it, before a watchdog exits the process.
fn release_terminal() {
    if crossterm::terminal::is_raw_mode_enabled().unwrap_or(false) {
        ratatui::restore();
    }
}

/// The saved run `name` and its directory.
async fn load_run(name: &str) -> Result<(PathBuf, state::SwarmState), Box<dyn std::error::Error>> {
    let dir = state::run_dir(name)?;
    let saved = state::load(&dir)
        .await?
        .ok_or_else(|| format!("no saved run {name}"))?;
    Ok((dir, saved))
}

fn print_skills(skills: &Skills) {
    let rows = skills
        .skills
        .iter()
        .map(|s| ("skill   ", s))
        .chain(skills.shadowed.iter().map(|s| ("shadowed", s)));
    for (kind, found) in rows {
        let path = found
            .skill
            .path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        println!("{kind} {} [{}] {path}", found.skill.name, found.origin);
    }
}

/// What a start does with its TASK and `--goal`.
#[derive(Debug)]
enum StartPlan {
    /// Send the TASK to the live root of the resumed run.
    Message(String),
    /// Start a root agent.
    Spawn(SpawnRequest),
    Nothing,
}

/// A resumed run that still has a live root gets only a TASK, as a message, and only one other
/// than `last_task` (the one the run was last started with): a supervisor replays its arguments
/// on every restart. `--goal` seeds a new root alone, for the same reason. Otherwise the TASK
/// (or the goal) starts a root.
fn start_plan(
    task: Option<String>,
    goal: Option<String>,
    live_root: bool,
    last_task: Option<&str>,
) -> StartPlan {
    if live_root {
        return match task.filter(|task| !task.is_empty() && Some(task.as_str()) != last_task) {
            Some(task) => StartPlan::Message(task),
            None => StartPlan::Nothing,
        };
    }
    match task.or_else(|| goal.clone()) {
        Some(task) => StartPlan::Spawn(SpawnRequest {
            task,
            goal,
            ..Default::default()
        }),
        None => StartPlan::Nothing,
    }
}

/// Print events until the root has replied (or, with `expect_reply` unset, at once) and no agent
/// is busy. Fails when the last reply to the user was an error.
async fn run_headless(
    swarm: &Swarm,
    mut events: UnboundedReceiver<SwarmEvent>,
    expect_reply: bool,
) -> Result<Exit, Box<dyn std::error::Error>> {
    let mut replied = !expect_reply;
    let mut failed = false;
    let mut ticker = tokio::time::interval(Duration::from_millis(500));
    loop {
        tokio::select! {
            Some(event) = events.recv() => {
                if let SwarmEvent::Mail { to, mail } = &event {
                    if to == USER {
                        replied = true;
                        failed = mail.kind == "error";
                    }
                }
                print_event(&event);
            }
            _ = ticker.tick() => {}
            _ = tokio::signal::ctrl_c() => return Ok(Exit::Quit),
            _ = swarm.reload_requested() => return Ok(Exit::Reload),
        }
        let all_stopped = swarm.agents().iter().all(|agent| !agent.status.is_live());
        if all_stopped || (replied && !swarm.is_busy()) {
            return match failed {
                true => Err("the root agent's last turn failed".into()),
                false => Ok(Exit::Quit),
            };
        }
    }
}

fn print_event(event: &SwarmEvent) {
    match event {
        SwarmEvent::Spawned {
            id,
            parent,
            name,
            task,
            cwd,
            ..
        } => {
            eprintln!(
                "[{id}] {name}, spawned by {parent} in {}: {}",
                cwd.display(),
                first_line(task)
            )
        }
        SwarmEvent::Status { id, status } => eprintln!("[{id}] {status}"),
        SwarmEvent::Compacted { id, before, after } => {
            eprintln!("[{id}] context compacted: {before} -> {after} tokens")
        }
        SwarmEvent::CompactFailed { id, tokens, error } => {
            eprintln!("[{id}] context not compacted ({tokens} tokens): {error}")
        }
        SwarmEvent::Text { .. }
        | SwarmEvent::Usage { .. }
        | SwarmEvent::MailRead { .. }
        | SwarmEvent::Image { .. }
        | SwarmEvent::ModelQueued { .. }
        | SwarmEvent::HeldForReload { .. }
        | SwarmEvent::History { .. } => {}
        SwarmEvent::Reloaded { from, to, changes } => {
            eprintln!("{}", strands_swarm::version::notice(from, to, changes));
        }
        SwarmEvent::ReloadPending { tool_calls } => {
            eprintln!("reload pending (waiting for {tool_calls} tool calls)")
        }
        SwarmEvent::Model { id, model } => eprintln!("[{id}] model {model}"),
        SwarmEvent::Reclaimed { collected, .. } => collected
            .iter()
            .flat_map(|agent| agent.lines())
            .for_each(|line| eprintln!("gc {line}")),
        SwarmEvent::Budget { tokens } => match tokens {
            Some(tokens) => eprintln!("token budget {tokens}"),
            None => eprintln!("no token budget"),
        },
        SwarmEvent::ToolStarted { id, name, summary } => eprintln!("[{id}] {name} {summary}"),
        SwarmEvent::ToolFinished { id, name, ok: false } => eprintln!("[{id}] {name} failed"),
        SwarmEvent::ToolFinished { .. } => {}
        SwarmEvent::Mail { to, mail } if to == USER => {
            println!("--- {} ---\n{}\n", mail.from_label, mail.text)
        }
        SwarmEvent::Mail { to, mail } => eprintln!(
            "[{to}] ✉ {} ({}): {}",
            mail.from_label,
            mail.kind,
            first_line(&mail.text)
        ),
    }
}

fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_resume_sends_its_live_root_the_task_but_never_the_goal() {
        let goal = || Some("all tests pass".to_owned());
        assert!(matches!(start_plan(None, goal(), true, None), StartPlan::Nothing));
        assert!(matches!(
            start_plan(Some("also fix X".into()), goal(), true, None),
            StartPlan::Message(task) if task == "also fix X"
        ));
        assert!(matches!(
            start_plan(Some(String::new()), None, true, None),
            StartPlan::Nothing
        ));
    }

    #[test]
    fn a_replayed_task_is_not_sent_again_but_a_new_one_is() {
        let last = Some("build it");
        assert!(matches!(
            start_plan(Some("build it".into()), None, true, last),
            StartPlan::Nothing
        ));
        assert!(matches!(
            start_plan(Some("also fix X".into()), None, true, last),
            StartPlan::Message(task) if task == "also fix X"
        ));
    }

    #[test]
    fn without_a_live_root_the_task_or_goal_starts_one() {
        let StartPlan::Spawn(request) = start_plan(None, Some("g".into()), false, None) else {
            panic!("a goal starts a root");
        };
        assert_eq!((request.task.as_str(), request.goal.as_deref()), ("g", Some("g")));
        let StartPlan::Spawn(request) = start_plan(Some("t".into()), None, false, Some("t")) else {
            panic!("a task starts a root");
        };
        assert_eq!((request.task.as_str(), request.goal), ("t", None));
        assert!(matches!(start_plan(None, None, false, None), StartPlan::Nothing));
    }
}
