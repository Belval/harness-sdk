//! `strands-swarm`: run a model-driven agent swarm in the terminal.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use strands_agents::tools::Tool;
use strands_harness::Effort;
use strands_swarm::state;
use strands_swarm::tui::Exit;
use strands_swarm::RELOAD_EXIT_CODE;
use strands_swarm::{
    harness_factory, harness_model_resolver, load_shared_mcp, HarnessFactoryOptions, SpawnRequest, Swarm,
    SwarmConfig, SwarmEvent, USER,
};
use tokio::sync::mpsc::UnboundedReceiver;

/// Run a swarm of cooperating coding agents. The root agent decides whether and how to spawn
/// others.
#[derive(Parser, Debug)]
#[command(version)]
struct Cli {
    /// Task for the root agent. Optional in the TUI, where you can type it.
    task: Option<String>,
    /// Default model for every agent (`provider/name` or a Bedrock model id).
    #[arg(long)]
    model: Option<String>,
    /// Reasoning effort.
    #[arg(long, default_value = "auto")]
    effort: Effort,
    /// Working directory of the root agent.
    #[arg(long, default_value = ".")]
    cwd: PathBuf,
    /// Maximum number of live agents, root included.
    #[arg(long, default_value_t = 16)]
    max_agents: usize,
    /// File with extra instructions given to every agent.
    #[arg(long)]
    instructions: Option<PathBuf>,
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
    /// Name of a persistent run. State lives in `~/.local/state/strands-swarm/<name>`; starting
    /// again with the same name resumes it (agents, conversations, unread messages).
    #[arg(long)]
    run: Option<String>,
    /// No TUI: stream events to stderr and exit when the swarm goes quiet.
    #[arg(long)]
    headless: bool,
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    strands_swarm::watchdog::spawn(Duration::from_secs(10), |report| {
        // The TUI may own the screen; under swarmctl stderr goes to the run's stderr.log.
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
    let cwd = std::fs::canonicalize(&cli.cwd)?;
    let instructions = cli
        .instructions
        .as_ref()
        .map(std::fs::read_to_string)
        .transpose()?;
    let mcp = load_shared_mcp(&cli.mcp_configs, &cli.mcp_deny).await?;
    if !cli.mcp_configs.is_empty() {
        eprintln!(
            "strands-swarm: {} MCP servers, {} tools shared ({}), {} denied",
            mcp.clients.len(),
            mcp.tools.len(),
            if cli.mcp_eager {
                "eager"
            } else {
                "on demand via find_tools"
            },
            mcp.denied.len()
        );
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
    };
    let root = root_request(cli.task, cli.goal);
    let mut config = SwarmConfig::new(harness_factory(options.clone()), cwd);
    config.models = Some(harness_model_resolver(options));
    config.max_agents = cli.max_agents;
    config.cargo_target_dir = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from);
    let saved = match &cli.run {
        Some(name) => {
            let dir = state::run_dir(name)?;
            config.run_id = name.clone();
            config.state_dir = Some(dir.clone());
            state::load(&dir).await?
        }
        None => None,
    };
    let resumed = saved.is_some();
    let (swarm, events) = match saved {
        Some(saved) => {
            config.cwd = saved.cwd.clone();
            eprintln!(
                "strands-swarm: resuming run {} ({} agents)",
                saved.run_id,
                saved.agents.len()
            );
            Swarm::resume(config, saved).await?
        }
        None => Swarm::new(config),
    };
    // A resumed run already has its root; the task and goal only seed a new one.
    let root = if resumed { None } else { root };

    if !cli.headless {
        return Ok(strands_swarm::tui::run(swarm, events, root).await?);
    }
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    match root {
        Some(root) => drop(swarm.spawn(USER, root).await?),
        None if !resumed => return Err("--headless needs a task or a --goal".into()),
        None => {}
    }
    let exit = run_headless(&swarm, events).await;
    swarm.shutdown().await;
    Ok(exit)
}

fn root_request(task: Option<String>, goal: Option<String>) -> Option<SpawnRequest> {
    let task = task.or_else(|| goal.clone())?;
    Some(SpawnRequest {
        task,
        goal,
        ..Default::default()
    })
}

/// Print events until the root has replied and no agent is busy.
async fn run_headless(swarm: &Swarm, mut events: UnboundedReceiver<SwarmEvent>) -> Exit {
    let mut replied = false;
    let mut ticker = tokio::time::interval(Duration::from_millis(500));
    loop {
        tokio::select! {
            Some(event) = events.recv() => {
                replied |= matches!(&event, SwarmEvent::Mail { to, .. } if to == USER);
                print_event(&event);
            }
            _ = ticker.tick() => {}
            _ = tokio::signal::ctrl_c() => return Exit::Quit,
            _ = swarm.reload_requested() => return Exit::Reload,
        }
        let all_stopped = swarm.agents().iter().all(|agent| !agent.status.is_live());
        if all_stopped || (replied && !swarm.is_busy()) {
            return Exit::Quit;
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
        SwarmEvent::Text { .. } | SwarmEvent::Usage { .. } | SwarmEvent::MailRead { .. } => {}
        SwarmEvent::Model { id, model } => eprintln!("[{id}] model {model}"),
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
