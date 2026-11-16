//! The swarm tools every agent carries: `spawn_agent`, `send_message`, `wait`, `list_agents`,
//! `cancel_agent`. Each instance is bound to the agent that owns it.

use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use strands_agents::tools::{FunctionTool, ToolContext, ToolRef};
use strands_agents::Result;

use crate::runtime::{render_mail, AgentInfo, SpawnRequest, Swarm, USER};

/// Tool names, in registration order.
pub const SWARM_TOOL_NAMES: &[&str] = &[
    "spawn_agent",
    "send_message",
    "wait",
    "list_agents",
    "cancel_agent",
    "reload_swarm",
];

const DEFAULT_WAIT_SECONDS: u64 = 600;
const MAX_WAIT_SECONDS: u64 = 3600;

const SPAWN_DESCRIPTION: &str = concat!(
    "Start a new agent that works on `task` concurrently with you, and return its id immediately. ",
    "It has the same tools you have, including spawning its own agents. Its final reply at the end of each ",
    "turn is sent to you as a message; you can message it at any time with send_message and it continues ",
    "with its full context. The task is its only context, so include what it needs to know. ",
    "Set worktree=true to give it its own git worktree and branch, cut from your HEAD (uncommitted changes ",
    "are not included), so parallel edits do not collide; integrate its branch yourself with git."
);

const SEND_DESCRIPTION: &str = concat!(
    "Send a message to another agent by id or name, or to `parent` (whoever spawned you; the user for the root ",
    "agent). A busy agent sees it after its current tool call; an idle one starts a new turn."
);

const WAIT_DESCRIPTION: &str = concat!(
    "Block until a message arrives, then return every unread message. With `from`, keep waiting until one ",
    "of those agents has written. Returns early with a status summary on timeout."
);

const LIST_DESCRIPTION: &str =
    "List every agent in the swarm with its name, id, parent, status, working directory and branch.";

const CANCEL_DESCRIPTION: &str =
    "Stop an agent and every agent it spawned. Its worktree and branch are left in place.";

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
    /// Model to use instead of the default (`provider/name` or a Bedrock model id).
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
struct NoInput {}

#[derive(Deserialize, schemars::JsonSchema)]
struct ReloadInput {
    /// Absolute path of the freshly built `strands-swarm` binary.
    binary: String,
}

const RELOAD_DESCRIPTION: &str = concat!(
    "Restart the whole swarm into a new strands-swarm binary (hot reload). The binary must already be ",
    "built and must answer --version. Every agent is stopped, saved, and resumed under the new binary with ",
    "its conversation and unread messages; agents that were mid-turn get a restart notice. If the new ",
    "binary crashes on startup the supervisor rolls back to the previous one."
);

fn reload_tool(swarm: Swarm) -> ToolRef {
    let tool = FunctionTool::builder("reload_swarm")
        .description(RELOAD_DESCRIPTION)
        .typed_handler(move |input: ReloadInput, _ctx: ToolContext| {
            let swarm = swarm.clone();
            async move {
                swarm.reload(std::path::Path::new(&input.binary)).await?;
                Result::Ok("Reload scheduled; the swarm is restarting now.".to_owned())
            }
        });
    Arc::new(tool)
}

/// The swarm tools bound to agent `id`.
pub fn swarm_tools(swarm: Swarm, id: &str) -> Vec<ToolRef> {
    vec![
        spawn_tool(swarm.clone(), id.to_owned()),
        send_tool(swarm.clone(), id.to_owned()),
        wait_tool(swarm.clone(), id.to_owned()),
        list_tool(swarm.clone()),
        cancel_tool(swarm.clone(), id.to_owned()),
        reload_tool(swarm),
    ]
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
                let info = swarm
                    .agent(&id)
                    .map(|info| describe(&swarm, &info))
                    .unwrap_or_default();
                Result::Ok(format!("Spawned {info}. Its replies will arrive as messages."))
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
                let wait = swarm.wait_for_mail(&me, &from, &ctx.cancel);
                let text = match tokio::time::timeout(Duration::from_secs(seconds), wait).await {
                    Ok(Some(mail)) => render_mail(&mail),
                    Ok(None) => "Wait cancelled.".to_owned(),
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
        .typed_handler(move |_input: NoInput, _ctx: ToolContext| {
            let swarm = swarm.clone();
            async move {
                let lines: Vec<String> = swarm.agents().iter().map(|info| describe(&swarm, info)).collect();
                Result::Ok(lines.join("\n"))
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

/// `parser (a2): running, parent root (a1), cwd /repo, branch swarm/run/a2`.
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
