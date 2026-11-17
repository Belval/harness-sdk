//! The swarm tools every agent carries: `spawn_agent`, `send_message`, `wait`, `list_agents`,
//! `cancel_agent`, `reload_swarm`, `schedule`, `run_background`, `jobs`, plus `land` (with
//! `--allow-land`) and `remember`/`forget` (with a memory file). Each instance is bound to the
//! agent that owns it.

use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use strands_agents::tools::{FunctionTool, ToolContext, ToolRef};
use strands_agents::Result;

use crate::jobs::{Job, JobState};
use crate::runtime::{
    render_mail, AgentInfo, AgentStatus, LandRequest, ReloadWhen, ScheduleRequest, SpawnRequest, Swarm,
    WaitOutcome, USER,
};
use crate::schedule::{self, Schedule};

const DEFAULT_WAIT_SECONDS: u64 = 600;
const MAX_WAIT_SECONDS: u64 = 3600;

const SPAWN_DESCRIPTION: &str = concat!(
    "Start a new agent that works on `task` concurrently with you, and return its id immediately. ",
    "It has the same tools you have, including spawning its own agents. Its final reply at the end of each ",
    "turn is sent to you as a message; you can message it at any time with send_message and it continues ",
    "with its full context. The task is its only context, so include what it needs to know. ",
    "Set worktree=true to give it its own git worktree and branch, cut from your HEAD (uncommitted changes ",
    "are not included), so parallel edits do not collide; integrate its branch yourself with git. ",
    "At the agent limit the agent is queued (status `queued`; it can already be messaged or cancelled) and ",
    "starts when a slot frees."
);

const SEND_DESCRIPTION: &str = concat!(
    "Send a message to another agent by id or name, or to `parent` (whoever spawned you; the user for the root ",
    "agent). A busy agent sees it after its current tool call; an idle one starts a new turn."
);

const WAIT_DESCRIPTION: &str = concat!(
    "Block until a message arrives, then return every unread message. With `from`, keep waiting until one ",
    "of those agents has written. Returns early with a status summary on timeout."
);

const LIST_DESCRIPTION: &str = concat!(
    "List the agents in the swarm with name, id, parent, status, working directory, branch, running ",
    "background jobs, model and estimated spend (own, and with descendants). ",
    "Cancelled agents are only counted unless include_cancelled is set."
);

const CANCEL_DESCRIPTION: &str =
    "Stop an agent and every agent it spawned, and kill their background jobs. Its worktree and branch are left in place.";

#[derive(Deserialize, schemars::JsonSchema)]
struct SpawnInput {
    /// Everything the new agent needs to know: goal, context, constraints, what to report back.
    task: String,
    /// Short role name shown to the user and other agents, e.g. `lazy-mcp` or `tui-tests`. Agents can
    /// be addressed by it; a suffix is added if it is taken.
    name: String,
    /// Give the agent its own git worktree and branch.
    #[serde(default)]
    worktree: bool,
    /// Model to use instead of the default (`provider/name` or a Bedrock model id). List-price cost
    /// relative to opus: sonnet ≈ 0.6×, haiku ≈ 0.2×; cheaper models suit mechanical subtasks.
    #[serde(default)]
    model: Option<String>,
    /// A checkable completion criterion. When set, a judge reviews each finished turn and sends the
    /// agent back to work until the criterion is met.
    #[serde(default)]
    goal: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct SendInput {
    /// Receiver: an agent id or name, or `parent`.
    to: String,
    /// Message text.
    message: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct WaitInput {
    /// Only return once one of these agents (ids or names) has written. Omit to wait for anyone.
    #[serde(default)]
    from: Vec<String>,
    /// Give up after this many seconds (default 600, max 3600).
    #[serde(default)]
    timeout_seconds: Option<u64>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct CancelInput {
    /// Id or name of the agent to stop.
    id: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ListInput {
    /// Also list cancelled agents.
    #[serde(default)]
    include_cancelled: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ReloadInput {
    /// Absolute path of the freshly built `strands-swarm` binary.
    binary: String,
    /// `idle` (default): restart once no agent has a tool call in flight (`wait` does not count),
    /// at most 10 minutes later. `now`: restart right away, interrupting in-flight tool calls.
    #[serde(default)]
    when: ReloadWhen,
}

const RELOAD_DESCRIPTION: &str = concat!(
    "Restart the whole swarm into a new strands-swarm binary (hot reload). The binary must already be ",
    "built; it is checked (--version, --self-test) and installed right away, and the restart follows ",
    "per `when`. A newer reload supersedes a pending one. Every agent is stopped, saved, and resumed ",
    "under the new binary with its conversation and unread messages; agents that were mid-turn get a ",
    "restart notice. If the new binary crashes on startup the supervisor rolls back to the previous one."
);

fn reload_tool(swarm: Swarm) -> ToolRef {
    let tool = FunctionTool::builder("reload_swarm")
        .description(RELOAD_DESCRIPTION)
        .typed_handler(move |input: ReloadInput, _ctx: ToolContext| {
            let swarm = swarm.clone();
            async move {
                swarm
                    .reload(std::path::Path::new(&input.binary), input.when)
                    .await?;
                Result::Ok(match input.when {
                    ReloadWhen::Now => "Reload scheduled; the swarm is restarting now.".to_owned(),
                    ReloadWhen::Idle => concat!(
                        "Installed and self-tested. The swarm restarts once no agent has a tool call ",
                        "in flight (at most 10 minutes from now); carry on until then."
                    )
                    .to_owned(),
                })
            }
        });
    Arc::new(tool)
}

const RUN_BACKGROUND_DESCRIPTION: &str = concat!(
    "Run a shell command in the background, in your working directory and environment, and return its ",
    "job id at once. It runs detached from the swarm, so it survives swarm restarts. When it ends you get ",
    "a message with its exit code and the last lines of its output (an idle agent wakes up for it): ",
    "use it for builds, test suites and anything slow instead of polling with sleep. Output goes to a log ",
    "file you can read meanwhile; `jobs` lists or kills jobs."
);

const JOBS_DESCRIPTION: &str =
    "List the background jobs of the swarm (yours first) with status and log path, or kill one by id.";

#[derive(Deserialize, schemars::JsonSchema)]
struct RunBackgroundInput {
    /// Shell command (run with `sh -c`).
    command: String,
    /// Short label, e.g. `gates`.
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct JobsInput {
    /// Id of a job to kill (with everything it started); omit to list jobs.
    #[serde(default)]
    kill: Option<String>,
}

const LAND_DESCRIPTION: &str = concat!(
    "Land your commits on the branch your root agent has checked out (others only if the user ",
    "allowed them), instead of asking someone to merge them. You can land your own branch or ",
    "that of an agent you spawned. In a scratch worktree the branch is ",
    "rebased onto it, the checks run there (default: the swarm's gate commands) with a private ",
    "CARGO_TARGET_DIR, and only if all pass is the target fast-forwarded. Runs as a background ",
    "job; you get a `job` message with the outcome. On a conflict (with the conflicted files), a ",
    "failing check (with its output tail) or a target checkout with uncommitted changes, nothing ",
    "is changed. Your own branch is never modified."
);

#[derive(Deserialize, schemars::JsonSchema)]
struct LandInput {
    /// Branch to land: your own (the default) or that of an agent you spawned.
    #[serde(default)]
    branch: Option<String>,
    /// Branch to fast-forward; default (and usually the only one allowed): the branch your
    /// root agent has checked out.
    #[serde(default)]
    into: Option<String>,
    /// Shell commands that must pass on the rebased commits, run from the same subdirectory as
    /// your working directory; default: the swarm's gates.
    #[serde(default)]
    check: Option<Vec<String>>,
}

fn land_tool(swarm: Swarm, me: String) -> ToolRef {
    let tool = FunctionTool::builder("land")
        .description(LAND_DESCRIPTION)
        .typed_handler(move |input: LandInput, _ctx: ToolContext| {
            let swarm = swarm.clone();
            let me = me.clone();
            async move {
                let request = LandRequest {
                    branch: input.branch,
                    into: input.into,
                    check: input.check,
                };
                let job = swarm.land(&me, request).await?;
                Result::Ok(format!(
                    "Started job {}; output: {}. You will get a message when it ends.",
                    job.label(),
                    swarm.job_log(&job.meta.id).display()
                ))
            }
        });
    Arc::new(tool)
}

fn run_background_tool(swarm: Swarm, me: String) -> ToolRef {
    let tool = FunctionTool::builder("run_background")
        .description(RUN_BACKGROUND_DESCRIPTION)
        .typed_handler(move |input: RunBackgroundInput, _ctx: ToolContext| {
            let swarm = swarm.clone();
            let me = me.clone();
            async move {
                let job = swarm.start_job(&me, &input.command, input.name).await?;
                Result::Ok(format!(
                    "Started job {} in {}; output: {}. You will get a message when it ends.",
                    job.label(),
                    job.meta.cwd.display(),
                    swarm.job_log(&job.meta.id).display()
                ))
            }
        });
    Arc::new(tool)
}

fn jobs_tool(swarm: Swarm, me: String) -> ToolRef {
    let tool = FunctionTool::builder("jobs")
        .description(JOBS_DESCRIPTION)
        .typed_handler(move |input: JobsInput, _ctx: ToolContext| {
            let swarm = swarm.clone();
            let me = me.clone();
            async move {
                if let Some(id) = input.kill {
                    let job = swarm.kill_job(&id).await?;
                    return Result::Ok(format!("Job {}: {}.", job.label(), job.status()));
                }
                Result::Ok(list_jobs(&swarm, &me, swarm.jobs().await))
            }
        });
    Arc::new(tool)
}

/// One line per job, the caller's first, then the newest first within each group.
fn list_jobs(swarm: &Swarm, me: &str, mut jobs: Vec<Job>) -> String {
    if jobs.is_empty() {
        return "No background jobs.".to_owned();
    }
    jobs.reverse();
    jobs.sort_by_key(|job| job.meta.owner != me);
    jobs.iter()
        .map(|job| {
            format!(
                "{}: {}, agent {}, `{}`, log {}",
                job.label(),
                job.status(),
                swarm.label(&job.meta.owner),
                job.meta.command,
                swarm.job_log(&job.meta.id).display()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ScheduleInput {
    /// Text to post when it fires. Omit (and `cancel`) to list every schedule.
    #[serde(default)]
    message: Option<String>,
    /// Receiver: an agent id or name, or `parent`. Default: yourself.
    #[serde(default)]
    to: Option<String>,
    /// Repeat interval, e.g. `30m`, `2h`, `1d` (at least `1m`). Omit for a one-shot.
    #[serde(default)]
    every: Option<String>,
    /// First (or only) time: `in 2h`, or ISO 8601 such as `2026-11-17T09:00Z` (UTC unless an offset
    /// is given). Default: one `every` from now.
    #[serde(default)]
    at: Option<String>,
    /// Id to create, or to replace an existing schedule. Default: a new `sN`.
    #[serde(default)]
    id: Option<String>,
    /// Id of a schedule to remove.
    #[serde(default)]
    cancel: Option<String>,
}

const SCHEDULE_DESCRIPTION: &str = concat!(
    "Post `message` to an agent's mailbox (default: your own) later, repeatedly, or both: e.g. ",
    "`every: \"30m\"` to poll something, `at: \"in 2h\"` for a reminder. For periodic work, schedule ",
    "yourself and end your turn instead of sitting in a long `wait`: each fire starts a new turn, and an ",
    "idle agent waiting only on a schedule does not count toward the agent limit. Schedules survive ",
    "restarts; a fire missed while the swarm was down is delivered once on resume. `cancel: id` removes ",
    "one; no arguments lists them all. Schedules of a cancelled agent are dropped."
);

fn schedule_tool(swarm: Swarm, me: String) -> ToolRef {
    let tool = FunctionTool::builder("schedule")
        .description(SCHEDULE_DESCRIPTION)
        .typed_handler(move |input: ScheduleInput, _ctx: ToolContext| {
            let swarm = swarm.clone();
            let me = me.clone();
            async move {
                if let Some(id) = input.cancel {
                    let removed = swarm.unschedule(&id)?;
                    return Result::Ok(format!("Removed {}.", describe_schedule(&swarm, &removed)));
                }
                let Some(message) = input.message else {
                    return Result::Ok(list_schedules(&swarm));
                };
                let to = match &input.to {
                    Some(who) => resolve(&swarm, &me, who)?,
                    None => me.clone(),
                };
                let every = input
                    .every
                    .as_deref()
                    .map(schedule::parse_duration)
                    .transpose()
                    .map_err(strands_agents::Error::other)?;
                let now = swarm.now();
                let at = input
                    .at
                    .as_deref()
                    .map(|at| schedule::parse_at(at, now))
                    .transpose()
                    .map_err(strands_agents::Error::other)?;
                let request = ScheduleRequest {
                    id: input.id,
                    to,
                    message,
                    every,
                    at,
                };
                let created = swarm.schedule(&me, request)?;
                Result::Ok(format!("Scheduled {}.", describe_schedule(&swarm, &created)))
            }
        });
    Arc::new(tool)
}

/// `s1 every 30m, next in 12m (…) → monitor (a3): check the queue`.
pub fn describe_schedule(swarm: &Swarm, schedule: &Schedule) -> String {
    let line = schedule.describe(swarm.now());
    let (head, message) = line.split_once(": ").unwrap_or((line.as_str(), ""));
    format!("{head} → {}: {message}", swarm.label(&schedule.to))
}

/// Every pending schedule, soonest first, one per line.
pub fn list_schedules(swarm: &Swarm) -> String {
    let lines: Vec<String> = swarm
        .schedules()
        .iter()
        .map(|s| describe_schedule(swarm, s))
        .collect();
    match lines.is_empty() {
        true => "No schedules.".to_owned(),
        false => lines.join("\n"),
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct RememberInput {
    /// The fact, preference or instruction to keep, as one self-contained sentence.
    text: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ForgetInput {
    /// Text identifying the entries to drop; every memory line containing it (case-insensitive)
    /// is removed.
    text: String,
}

const REMEMBER_DESCRIPTION: &str = concat!(
    "Save something the user asked you to remember to the user's memory file, shared by every agent ",
    "and every future run (shown under \"# Memory\" in the system prompt from the next turn on). Use it ",
    "only for things the user wants kept, not for your own notes."
);

const FORGET_DESCRIPTION: &str =
    "Remove outdated or wrong entries from the user's memory file: every line containing the text.";

fn remember_tool(swarm: Swarm) -> ToolRef {
    let tool = FunctionTool::builder("remember")
        .description(REMEMBER_DESCRIPTION)
        .typed_handler(move |input: RememberInput, _ctx: ToolContext| {
            let swarm = swarm.clone();
            async move {
                let line = swarm.remember(&input.text).await?;
                Result::Ok(format!("Remembered: {line}"))
            }
        });
    Arc::new(tool)
}

fn forget_tool(swarm: Swarm) -> ToolRef {
    let tool = FunctionTool::builder("forget")
        .description(FORGET_DESCRIPTION)
        .typed_handler(move |input: ForgetInput, _ctx: ToolContext| {
            let swarm = swarm.clone();
            async move {
                let removed = swarm.forget(&input.text).await?;
                Result::Ok(format!("Forgot:\n{}", removed.join("\n")))
            }
        });
    Arc::new(tool)
}

/// The swarm tools bound to agent `id`; `remember` and `forget` only when memory is on.
pub fn swarm_tools(swarm: Swarm, id: &str) -> Vec<ToolRef> {
    let mut tools = vec![
        spawn_tool(swarm.clone(), id.to_owned()),
        send_tool(swarm.clone(), id.to_owned()),
        wait_tool(swarm.clone(), id.to_owned()),
        list_tool(swarm.clone()),
        cancel_tool(swarm.clone(), id.to_owned()),
        reload_tool(swarm.clone()),
        schedule_tool(swarm.clone(), id.to_owned()),
        run_background_tool(swarm.clone(), id.to_owned()),
        jobs_tool(swarm.clone(), id.to_owned()),
    ];
    if swarm.allows_land() {
        tools.push(land_tool(swarm.clone(), id.to_owned()));
    }
    if swarm.memory().is_some() {
        tools.push(remember_tool(swarm.clone()));
        tools.push(forget_tool(swarm));
    }
    tools
}

fn spawn_tool(swarm: Swarm, me: String) -> ToolRef {
    let tool = FunctionTool::builder("spawn_agent")
        .description(SPAWN_DESCRIPTION)
        .typed_handler(move |input: SpawnInput, _ctx: ToolContext| {
            let swarm = swarm.clone();
            let me = me.clone();
            async move {
                let request = SpawnRequest {
                    task: input.task,
                    name: Some(input.name),
                    worktree: input.worktree,
                    model: input.model,
                    goal: input.goal,
                };
                let id = swarm.spawn(&me, request).await?;
                let Some(info) = swarm.agent(&id) else {
                    return Result::Ok(format!("Spawned {id}."));
                };
                let described = describe(&swarm, &info);
                if info.status != AgentStatus::Queued {
                    return Result::Ok(format!(
                        "Spawned {described}. Its replies will arrive as messages."
                    ));
                }
                Result::Ok(format!(
                    "Queued {described}: the agent limit is reached, so it starts (and its replies \
                     arrive as messages) once a slot frees. To free one, cancel an agent you no \
                     longer need, or cancel this one to drop it. {}",
                    idle_agents(&swarm)
                ))
            }
        });
    Arc::new(tool)
}

fn send_tool(swarm: Swarm, me: String) -> ToolRef {
    let tool = FunctionTool::builder("send_message")
        .description(SEND_DESCRIPTION)
        .typed_handler(move |input: SendInput, _ctx: ToolContext| {
            let swarm = swarm.clone();
            let me = me.clone();
            async move {
                let to = resolve(&swarm, &me, &input.to)?;
                swarm.send(&me, &to, input.message)?;
                Result::Ok(format!("Sent to {}.", swarm.label(&to)))
            }
        });
    Arc::new(tool)
}

fn wait_tool(swarm: Swarm, me: String) -> ToolRef {
    let tool = FunctionTool::builder("wait")
        .description(WAIT_DESCRIPTION)
        .typed_handler(move |input: WaitInput, ctx: ToolContext| {
            let swarm = swarm.clone();
            let me = me.clone();
            async move {
                let seconds = input
                    .timeout_seconds
                    .unwrap_or(DEFAULT_WAIT_SECONDS)
                    .clamp(1, MAX_WAIT_SECONDS);
                let from = input
                    .from
                    .iter()
                    .map(|who| resolve(&swarm, &me, who))
                    .collect::<Result<Vec<_>>>()?;
                let wait = swarm.wait(&me, &from, &ctx.cancel);
                let text = match tokio::time::timeout(Duration::from_secs(seconds), wait).await {
                    Ok(WaitOutcome::Mail(mail)) => render_mail(&mail),
                    Ok(WaitOutcome::Cancelled) => "Wait cancelled.".to_owned(),
                    Ok(WaitOutcome::Unreachable(notice)) => notice,
                    Err(_) => timeout_summary(&swarm, &from, seconds),
                };
                Result::Ok(text)
            }
        });
    Arc::new(tool)
}

fn list_tool(swarm: Swarm) -> ToolRef {
    let tool = FunctionTool::builder("list_agents")
        .description(LIST_DESCRIPTION)
        .typed_handler(move |input: ListInput, _ctx: ToolContext| {
            let swarm = swarm.clone();
            async move {
                let jobs = swarm.jobs().await;
                Result::Ok(list_agents(&swarm, input.include_cancelled, &jobs))
            }
        });
    Arc::new(tool)
}

fn cancel_tool(swarm: Swarm, me: String) -> ToolRef {
    let tool = FunctionTool::builder("cancel_agent")
        .description(CANCEL_DESCRIPTION)
        .typed_handler(move |input: CancelInput, _ctx: ToolContext| {
            let swarm = swarm.clone();
            let me = me.clone();
            async move {
                let id = swarm.resolve(&input.id)?;
                if id == me {
                    return Result::Ok("You cannot cancel yourself; end your turn instead.".to_owned());
                }
                swarm.cancel(&id)?;
                Result::Ok(format!("Cancelled {} and its descendants.", swarm.label(&id)))
            }
        });
    Arc::new(tool)
}

/// The id `who` refers to: `parent`, the user, or an agent id or name.
fn resolve(swarm: &Swarm, me: &str, who: &str) -> Result<String> {
    match who {
        "parent" => Ok(swarm
            .agent(me)
            .map_or_else(|| USER.to_owned(), |info| info.parent)),
        USER => Ok(USER.to_owned()),
        other => swarm.resolve(other),
    }
}

/// `parser (a2): running, parent root (a1), cwd /repo, branch swarm/run/a2, jobs j3 (gates), model
/// claude-sonnet-4-5, est. $1.20 / 3.4M tokens`. One line per agent; cancelled ones are only counted
/// unless `include_cancelled`; then the swarm's total.
fn list_agents(swarm: &Swarm, include_cancelled: bool, jobs: &[Job]) -> String {
    let agents = swarm.agents();
    let (shown, hidden): (Vec<_>, Vec<_>) = agents
        .iter()
        .partition(|info| include_cancelled || info.status.is_live());
    let running = |id: &str| -> String {
        let labels: Vec<String> = jobs
            .iter()
            .filter(|job| job.meta.owner == id && job.state == JobState::Running)
            .map(Job::label)
            .collect();
        match labels.is_empty() {
            true => String::new(),
            false => format!(", jobs {}", labels.join(", ")),
        }
    };
    let schedules = swarm.schedules();
    let cost = swarm.cost();
    let spend = |info: &AgentInfo| -> String {
        let model = info.model.as_deref().map(crate::cost::short_model);
        let model = model.map(|m| format!(", model {m}")).unwrap_or_default();
        let spend = cost.spend(&info.id).map(|s| format!(", {}", s.describe()));
        model + &spend.unwrap_or_default()
    };
    let mut lines = Vec::new();
    for info in &shown {
        lines.push(describe(swarm, info) + &running(&info.id) + &spend(info));
        let upcoming = schedules.iter().filter(|s| s.to == info.id);
        lines.extend(upcoming.map(|s| format!("  scheduled: {}", s.describe(swarm.now()))));
    }
    if !hidden.is_empty() {
        lines.push(format!(
            "({} cancelled agents not shown; include_cancelled=true lists them)",
            hidden.len()
        ));
    }
    let tokens: u64 = cost.agents.iter().map(|a| a.totals.input + a.totals.output).sum();
    lines.push(format!(
        "Swarm total: est. {} / {} tokens (estimates from list prices).",
        cost.total().render(),
        crate::text::compact(tokens)
    ));
    lines.join("\n")
}

/// The idle agents (cancellation candidates), for a queued spawn's reply.
fn idle_agents(swarm: &Swarm) -> String {
    let idle: Vec<String> = swarm
        .agents()
        .iter()
        .filter(|info| matches!(info.status, AgentStatus::Idle | AgentStatus::Errored(_)))
        .map(|info| {
            format!(
                "{} ({}, parent {})",
                info.name,
                info.id,
                swarm.label(&info.parent)
            )
        })
        .collect();
    match idle.is_empty() {
        true => "No agent is idle right now.".to_owned(),
        false => format!("Idle agents: {}.", idle.join(", ")),
    }
}

fn describe(swarm: &Swarm, info: &AgentInfo) -> String {
    let branch = info
        .branch
        .as_ref()
        .map(|b| format!(", branch {b}"))
        .unwrap_or_default();
    format!(
        "{} ({}): {}, parent {}, cwd {}{branch}",
        info.name,
        info.id,
        info.status,
        swarm.label(&info.parent),
        info.cwd.display()
    )
}

fn timeout_summary(swarm: &Swarm, from: &[String], seconds: u64) -> String {
    let agents: Vec<String> = swarm
        .agents()
        .iter()
        .filter(|info| from.is_empty() || from.contains(&info.id))
        .map(|info| describe(swarm, info))
        .collect();
    format!(
        "No messages within {seconds}s. Current status:\n{}",
        agents.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_agent_requires_a_name() {
        let schema = strands_agents::tools::input_schema_for::<SpawnInput>();
        let required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(
            required.contains(&"name") && required.contains(&"task"),
            "{schema}"
        );
        assert_eq!(schema["properties"]["name"]["type"], "string");
    }
}
