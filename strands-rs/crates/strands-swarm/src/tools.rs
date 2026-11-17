//! The swarm tools every agent carries: `spawn_agent`, `send_message`, `wait`, `list_agents`,
//! `cancel_agent`, `reload_swarm`, `schedule`, `run_background`, `jobs`, plus `land` (with
//! `--allow-land`) and `remember`/`forget` (with a memory file). Each instance is bound to the
//! agent that owns it.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use strands_agents::tools::{FunctionTool, ToolContext, ToolRef};
use strands_agents::Result;

use crate::jobs::{Job, JobState};
use crate::runtime::{
    render_mail, AgentInfo, AgentStatus, LandRequest, ReloadWhen, ScheduleRequest, SpawnRequest, Swarm,
    WaitOutcome, JOBS_SENDER, USER,
};
use crate::schedule::{self, Schedule};
use crate::status::BranchState;
use tokio_util::sync::CancellationToken;

/// `T`'s input schema without what tells a model nothing: the root `title` and `description`
/// (the tool description says it) and per-field `format` and trivial `default`s (null, false,
/// empty). Every tool on every call pays for its schema.
pub(crate) fn lean_schema<T: schemars::JsonSchema>() -> serde_json::Value {
    fn lean(schema: &mut serde_json::Value) {
        let serde_json::Value::Object(map) = schema else {
            return;
        };
        map.remove("format");
        let trivial = |value: &serde_json::Value| match value {
            serde_json::Value::Null | serde_json::Value::Bool(false) => true,
            serde_json::Value::Array(items) => items.is_empty(),
            serde_json::Value::String(text) => text.is_empty(),
            _ => false,
        };
        if map.get("default").is_some_and(trivial) {
            map.remove("default");
        }
        if let Some(serde_json::Value::Object(properties)) = map.get_mut("properties") {
            properties.values_mut().for_each(lean);
        }
        if let Some(items) = map.get_mut("items") {
            lean(items);
        }
    }
    let mut schema = strands_agents::tools::input_schema_for::<T>();
    if let serde_json::Value::Object(map) = &mut schema {
        map.remove("title");
        map.remove("description");
    }
    lean(&mut schema);
    schema
}

const DEFAULT_WAIT_SECONDS: u64 = 600;
const MAX_WAIT_SECONDS: u64 = 3600;

const SPAWN_DESCRIPTION: &str = concat!(
    "Start an agent working on `task` concurrently; returns its id at once. It has your tools. Its final ",
    "reply each turn comes to you as a message; message it any time and it continues with its context. ",
    "The task is all it knows. worktree=true gives it its own git worktree and branch off your HEAD ",
    "(uncommitted changes excluded); you integrate the branch. At the agent limit it is `queued` until a ",
    "slot frees."
);

const SEND_DESCRIPTION: &str = concat!(
    "Message an agent (id or name) or `parent` (who spawned you; the user for the root agent). A busy agent ",
    "sees it after its current tool call; an idle one starts a turn."
);

const WAIT_DESCRIPTION: &str =
    "Block until a message arrives and return all unread ones; on timeout, a status summary.";

const LIST_DESCRIPTION: &str = concat!(
    "List agents: status, cwd, branch (ahead/behind the base, clean merge or conflicting files), ",
    "jobs, model, estimated spend."
);

const CANCEL_DESCRIPTION: &str = concat!(
    "Stop an agent, its descendants and their jobs. A clean worktree is removed, its branch too once ",
    "the base has all its commits (merged or cherry-picked); the result says what stays and why."
);

/// How long `cancel_agent` waits to report what reclaiming the cancelled worktrees did.
const CANCEL_RECLAIM_WAIT: Duration = Duration::from_secs(10);

#[derive(Deserialize, schemars::JsonSchema)]
struct SpawnInput {
    /// Goal, context, constraints, what to report back.
    task: String,
    /// Short role name, e.g. `tui-tests`; usable as its address.
    name: String,
    #[serde(default)]
    worktree: bool,
    /// Model id, or opus/sonnet/haiku, instead of the default. Cost vs opus: sonnet ≈ 0.6×, haiku ≈ 0.2× (mechanical work).
    #[serde(default)]
    model: Option<String>,
    /// Checkable completion criterion: a judge sends the agent back to work until it is met.
    #[serde(default)]
    goal: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct SendInput {
    to: String,
    message: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct WaitInput {
    /// Only return for mail from these (agents, or `jobs`) or the user; default anyone.
    #[serde(default)]
    from: Vec<String>,
    /// Default 600, max 3600.
    #[serde(default)]
    timeout_seconds: Option<u64>,
    /// Wait for every one of `from` (or until the user writes), not just the first.
    #[serde(default)]
    all: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct CancelInput {
    /// Id or name.
    id: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ListInput {
    #[serde(default)]
    include_cancelled: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ReloadInput {
    /// Absolute path of the built `strands-swarm` binary.
    binary: String,
    /// `idle` (default): once no tool call but `wait` is in flight, at most 10 min later. `now`:
    /// at once, interrupting tool calls.
    #[serde(default)]
    #[schemars(schema_with = "reload_when_schema")]
    when: ReloadWhen,
}

const RELOAD_DESCRIPTION: &str = concat!(
    "Hot-reload the whole swarm into a built strands-swarm binary: it is self-tested and installed now, ",
    "then every agent is saved and resumed under it with its conversation and mail (mid-turn agents get ",
    "a restart notice). A newer reload supersedes a pending one; a binary that crashes on start is rolled back."
);

fn reload_when_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({ "type": "string", "enum": ["idle", "now"] })
}

async fn reload_tool(swarm: Swarm, _me: String, input: ReloadInput, _ctx: ToolContext) -> Result<String> {
    swarm
        .reload(std::path::Path::new(&input.binary), input.when)
        .await?;
    Ok(match input.when {
        ReloadWhen::Now => "Reload scheduled; the swarm is restarting now.".to_owned(),
        ReloadWhen::Idle => concat!(
            "Installed and self-tested. The swarm restarts once no agent has a tool call ",
            "in flight (at most 10 minutes from now); carry on until then."
        )
        .to_owned(),
    })
}

const RUN_BACKGROUND_DESCRIPTION: &str = concat!(
    "Run a shell command in your cwd, detached (survives swarm restarts); returns its job id. When it ends you ",
    "get a message from `jobs` with the exit code and output tail, waking you if idle. For builds, tests, ",
    "anything slow: start it here, then `wait` or end your turn; do not sleep-poll its log."
);

const JOBS_DESCRIPTION: &str = "List background jobs (yours first) with status and log path, or kill one.";

#[derive(Deserialize, schemars::JsonSchema)]
struct RunBackgroundInput {
    /// Run with `sh -c`.
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

async fn land_tool(swarm: Swarm, me: String, input: LandInput, _ctx: ToolContext) -> Result<String> {
    let request = LandRequest {
        branch: input.branch,
        into: input.into,
        check: input.check,
    };
    let job = swarm.land(&me, request).await?;
    Ok(format!(
        "Started job {}; output: {}. You will get a message when it ends.",
        job.label(),
        swarm.job_log(&job.meta.id).display()
    ))
}

async fn run_background_tool(
    swarm: Swarm,
    me: String,
    input: RunBackgroundInput,
    _ctx: ToolContext,
) -> Result<String> {
    let job = swarm.start_job(&me, &input.command, input.name).await?;
    Ok(format!(
        "Started job {} in {}; output: {}. You will get a message when it ends.",
        job.label(),
        job.meta.cwd.display(),
        swarm.job_log(&job.meta.id).display()
    ))
}

async fn jobs_tool(swarm: Swarm, me: String, input: JobsInput, _ctx: ToolContext) -> Result<String> {
    if let Some(id) = input.kill {
        let job = swarm.kill_job(&id).await?;
        return Ok(format!("Job {}: {}.", job.label(), job.status()));
    }
    Ok(list_jobs(&swarm, &me, swarm.jobs().await))
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
    /// Text to post; omit (with `cancel`) to list schedules.
    #[serde(default)]
    message: Option<String>,
    /// Agent id or name, or `parent`; default yourself.
    #[serde(default)]
    to: Option<String>,
    /// Repeat interval, e.g. `30m`, `1d` (min `1m`); omit for a one-shot.
    #[serde(default)]
    every: Option<String>,
    /// First time: `in 2h` or ISO 8601 (UTC unless offset); default one `every` from now.
    #[serde(default)]
    at: Option<String>,
    /// Id to create or replace; default a new `sN`.
    #[serde(default)]
    id: Option<String>,
    /// Id of a schedule to remove.
    #[serde(default)]
    cancel: Option<String>,
}

const SCHEDULE_DESCRIPTION: &str = concat!(
    "Post `message` to a mailbox later and/or repeatedly. For periodic work, schedule yourself and end ",
    "your turn rather than a long `wait`: each fire starts a turn, and an agent idle on a schedule holds ",
    "no agent slot. Survives restarts (a missed fire is delivered once)."
);

async fn schedule_tool(swarm: Swarm, me: String, input: ScheduleInput, _ctx: ToolContext) -> Result<String> {
    if let Some(id) = input.cancel {
        let removed = swarm.unschedule(&id)?;
        return Ok(format!("Removed {}.", describe_schedule(&swarm, &removed)));
    }
    let Some(message) = input.message else {
        return Ok(list_schedules(&swarm));
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
    Ok(format!("Scheduled {}.", describe_schedule(&swarm, &created)))
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
    /// One self-contained sentence.
    text: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ForgetInput {
    text: String,
}

const REMEMBER_DESCRIPTION: &str = concat!(
    "Save what the user asked you to remember to their memory file (\"# Memory\" in every agent's ",
    "system prompt, this run and future ones). Not for your own notes."
);

const FORGET_DESCRIPTION: &str = "Remove every memory-file line containing `text` (case-insensitive).";

async fn remember_tool(swarm: Swarm, _me: String, input: RememberInput, _ctx: ToolContext) -> Result<String> {
    let line = swarm.remember(&input.text).await?;
    Ok(format!("Remembered: {line}"))
}

async fn forget_tool(swarm: Swarm, _me: String, input: ForgetInput, _ctx: ToolContext) -> Result<String> {
    let removed = swarm.forget(&input.text).await?;
    Ok(format!("Forgot:\n{}", removed.join("\n")))
}

/// The swarm tools bound to agent `id`; `reload_swarm` only in a persistent run, `land` only
/// when allowed, `remember` and `forget` only when memory is on.
pub fn swarm_tools(swarm: Swarm, id: &str) -> Vec<ToolRef> {
    let s = &swarm;
    let mut tools = vec![
        swarm_tool(s, id, "spawn_agent", SPAWN_DESCRIPTION, spawn_tool),
        swarm_tool(s, id, "send_message", SEND_DESCRIPTION, send_tool),
        swarm_tool(s, id, "wait", WAIT_DESCRIPTION, wait_tool),
        swarm_tool(s, id, "list_agents", LIST_DESCRIPTION, list_tool),
        swarm_tool(s, id, "cancel_agent", CANCEL_DESCRIPTION, cancel_tool),
        swarm_tool(s, id, "schedule", SCHEDULE_DESCRIPTION, schedule_tool),
        swarm_tool(
            s,
            id,
            "run_background",
            RUN_BACKGROUND_DESCRIPTION,
            run_background_tool,
        ),
        swarm_tool(s, id, "jobs", JOBS_DESCRIPTION, jobs_tool),
    ];
    if swarm.can_reload() {
        tools.push(swarm_tool(s, id, "reload_swarm", RELOAD_DESCRIPTION, reload_tool));
    }
    if swarm.allows_land() {
        tools.push(swarm_tool(s, id, "land", LAND_DESCRIPTION, land_tool));
    }
    if swarm.memory().is_some() {
        tools.push(swarm_tool(s, id, "remember", REMEMBER_DESCRIPTION, remember_tool));
        tools.push(swarm_tool(s, id, "forget", FORGET_DESCRIPTION, forget_tool));
    }
    tools
}

/// Tool `name` taking `I` (as its [`lean_schema`]): each call runs `handler` with the swarm, the
/// owning agent's id, the input and the call's context.
fn swarm_tool<I, F, Fut>(swarm: &Swarm, me: &str, name: &str, description: &str, handler: F) -> ToolRef
where
    I: serde::de::DeserializeOwned + schemars::JsonSchema + Send + 'static,
    F: Fn(Swarm, String, I, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<String>> + Send + 'static,
{
    let (swarm, me) = (swarm.clone(), me.to_owned());
    let tool = FunctionTool::builder(name)
        .description(description)
        .input_schema(lean_schema::<I>())
        .typed_handler(move |input: I, ctx: ToolContext| handler(swarm.clone(), me.clone(), input, ctx));
    Arc::new(tool)
}

async fn spawn_tool(swarm: Swarm, me: String, input: SpawnInput, _ctx: ToolContext) -> Result<String> {
    let requested = input.name.clone();
    let request = SpawnRequest {
        task: input.task,
        name: Some(input.name),
        worktree: input.worktree,
        model: input.model,
        goal: input.goal,
    };
    let id = swarm.spawn(&me, request).await?;
    let Some(info) = swarm.agent(&id) else {
        return Ok(format!("Spawned {id}."));
    };
    let mut described = describe(&swarm, &info);
    if let Some(note) = crate::names::renamed(&requested, &info.name) {
        described.push_str(&format!(" ({note})"));
    }
    if info.status != AgentStatus::Queued {
        return Ok(format!(
            "Spawned {described}. Its replies will arrive as messages."
        ));
    }
    Ok(format!(
        "Queued {described}: the agent limit is reached, so it starts (and its replies \
         arrive as messages) once a slot frees. To free one, cancel an agent you no \
         longer need, or cancel this one to drop it. {}",
        idle_agents(&swarm)
    ))
}

async fn send_tool(swarm: Swarm, me: String, input: SendInput, _ctx: ToolContext) -> Result<String> {
    let to = resolve(&swarm, &me, &input.to)?;
    swarm.send(&me, &to, input.message)?;
    Ok(format!("Sent to {}.", swarm.label(&to)))
}

async fn wait_tool(swarm: Swarm, me: String, input: WaitInput, ctx: ToolContext) -> Result<String> {
    let seconds = input
        .timeout_seconds
        .unwrap_or(DEFAULT_WAIT_SECONDS)
        .clamp(1, MAX_WAIT_SECONDS);
    let from = input
        .from
        .iter()
        .map(|who| match who.as_str() {
            JOBS_SENDER => Ok(JOBS_SENDER.to_owned()),
            _ => resolve(&swarm, &me, who),
        })
        .collect::<Result<Vec<_>>>()?;
    if from == [JOBS_SENDER] && !swarm.has_mail_from(&me, &from) && !has_unreported_jobs(&swarm, &me).await {
        return Ok(NO_JOBS.to_owned());
    }
    if input.all {
        if from.is_empty() {
            return Err(strands_agents::Error::other(
                "`all` needs `from`: the agents to wait for.",
            ));
        }
        return Ok(wait_all(&swarm, &me, from, seconds, &ctx.cancel).await);
    }
    let wait = swarm.wait(&me, &from, &ctx.cancel);
    let text = match tokio::time::timeout(Duration::from_secs(seconds), wait).await {
        Ok(WaitOutcome::Mail(mail)) => render_mail(&mail),
        Ok(WaitOutcome::Cancelled) => "Wait cancelled.".to_owned(),
        Ok(WaitOutcome::Unreachable(notice)) => notice,
        Err(_) => timeout_summary(&swarm, &from, seconds),
    };
    Ok(text)
}

/// `wait` with `all`: collect mail until each of `from` has sent something, the user writes, or
/// `seconds` pass; every message taken on the way is returned, with a note on who is missing.
async fn wait_all(
    swarm: &Swarm,
    me: &str,
    mut remaining: Vec<String>,
    seconds: u64,
    cancel: &CancellationToken,
) -> String {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(seconds);
    let mut mail = Vec::new();
    let note = loop {
        let wait = swarm.wait(me, &remaining, cancel);
        match tokio::time::timeout_at(deadline, wait).await {
            Ok(WaitOutcome::Mail(batch)) => {
                remaining.retain(|id| !batch.iter().any(|m| &m.from == id));
                let from_user = batch.iter().any(|m| m.from == USER);
                mail.extend(batch);
                match (remaining.is_empty(), from_user) {
                    (true, _) => break None,
                    (false, true) => break Some(still_waiting(swarm, &remaining, "The user wrote")),
                    (false, false) => {}
                }
            }
            Ok(WaitOutcome::Cancelled) => break Some("Wait cancelled.".to_owned()),
            Ok(WaitOutcome::Unreachable(notice)) => break Some(notice),
            Err(_) => {
                let why = format!("No messages within {seconds}s");
                break Some(still_waiting(swarm, &remaining, &why));
            }
        }
    };
    let mut text = render_mail(&mail);
    if let Some(note) = note {
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        text.push_str(&note);
    }
    text
}

/// `<why> before all of them sent something; still waiting on:` and their status lines.
fn still_waiting(swarm: &Swarm, remaining: &[String], why: &str) -> String {
    let lines: Vec<String> = swarm
        .agents()
        .iter()
        .filter(|info| remaining.contains(&info.id))
        .map(|info| describe(swarm, info))
        .chain(remaining.iter().filter(|id| *id == JOBS_SENDER).cloned())
        .collect();
    format!(
        "{why} before all of them sent something; still waiting on:\n{}",
        lines.join("\n")
    )
}

/// What `wait` from `jobs` says when the agent has no job left to report.
const NO_JOBS: &str = "You have no running or unreported background jobs, so no job message will come.";

/// Whether `me` has a job still running or whose end it has not been told yet.
async fn has_unreported_jobs(swarm: &Swarm, me: &str) -> bool {
    swarm
        .jobs()
        .await
        .iter()
        .any(|job| job.meta.owner == me && (job.state == JobState::Running || !job.reported))
}

async fn list_tool(swarm: Swarm, _me: String, input: ListInput, _ctx: ToolContext) -> Result<String> {
    let jobs = swarm.jobs().await;
    let branches = swarm.branch_states().await;
    let left = match input.include_cancelled {
        true => cancelled_branches(&swarm, branches.0.as_deref()).await,
        false => HashMap::new(),
    };
    let text = list_agents(&swarm, input.include_cancelled, &jobs, &branches, &left);
    Ok(text)
}

async fn cancel_tool(swarm: Swarm, me: String, input: CancelInput, _ctx: ToolContext) -> Result<String> {
    let id = swarm.resolve(&input.id)?;
    if id == me {
        return Ok("You cannot cancel yourself; end your turn instead.".to_owned());
    }
    swarm.cancel(&id)?;
    let mut lines = vec![format!("Cancelled {} and its descendants.", swarm.label(&id))];
    for (agent, reclaimed) in swarm.reclaimed_after_cancel(&id, CANCEL_RECLAIM_WAIT).await {
        match reclaimed {
            Some(collected) => lines.extend(collected.lines()),
            None => lines.push(format!(
                "{agent}: still stopping; its worktree and branch are reclaimed once it stops if safe."
            )),
        }
    }
    Ok(lines.join("\n"))
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

/// `parser (a2): running, parent root (a1), cwd /repo, branch swarm/run/a2 (2 not on main, 1
/// behind, merges cleanly), jobs j3 (gates), model claude-sonnet-4-5, est. $1.20 / 3.4M tokens`.
/// One line per agent; cancelled ones are only counted unless `include_cancelled`; then the
/// swarm's total.
fn list_agents(
    swarm: &Swarm,
    include_cancelled: bool,
    jobs: &[Job],
    (base, branches): &(Option<String>, HashMap<String, BranchState>),
    cancelled: &HashMap<String, Option<BranchState>>,
) -> String {
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
    let merge = |id: &str| -> String {
        let base = base.as_deref().unwrap_or("the base");
        match (branches.get(id), cancelled.get(id)) {
            (Some(state), _) => format!(" ({})", state.describe(base)),
            (None, Some(Some(state))) => format!(" (kept: {})", state.describe(base)),
            (None, Some(None)) => " (deleted)".to_owned(),
            (None, None) => String::new(),
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
        let line = describe(swarm, info) + &merge(&info.id) + &running(&info.id) + &spend(info);
        lines.push(line);
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

/// How the branches of cancelled worktree agents stand, seen from the root's checkout: `None` for
/// one that no longer exists (deleted once merged); absent when git could not tell.
async fn cancelled_branches(swarm: &Swarm, base: Option<&str>) -> HashMap<String, Option<BranchState>> {
    let agents = swarm.agents();
    let root = agents
        .iter()
        .find(|info| info.parent == USER && info.branch.is_none() && info.cwd.exists());
    let (Some(root), Some(base)) = (root, base) else {
        return HashMap::new();
    };
    let mut states = HashMap::new();
    for info in agents.iter().filter(|info| !info.status.is_live()) {
        let Some(branch) = &info.branch else { continue };
        let exists = [
            "rev-parse",
            "--verify",
            "--quiet",
            &format!("refs/heads/{branch}"),
        ];
        if crate::worktree::git(&root.cwd, &exists).await.is_err() {
            states.insert(info.id.clone(), None);
        } else if let Some(state) = crate::status::branch_state(&root.cwd, base, branch).await {
            states.insert(info.id.clone(), Some(state));
        }
    }
    states
}

fn describe(swarm: &Swarm, info: &AgentInfo) -> String {
    let removed = match info.status.is_live() || info.cwd.exists() {
        true => "",
        false => " (removed)",
    };
    let branch = info
        .branch
        .as_ref()
        .map(|b| format!(", branch {b}"))
        .unwrap_or_default();
    format!(
        "{} ({}): {}, parent {}, cwd {}{removed}{branch}",
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
