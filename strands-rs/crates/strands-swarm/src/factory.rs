//! The default [`AgentFactory`]: every agent is a [`create_harness`] agent rooted at its own
//! working directory, with the swarm tools in place of the harness's `subagent`.

use std::sync::Arc;

use strands_agents::context_manager::strategies::{
    Offload, OffloadConditions, OffloadTarget, TruncateConfig,
};
use strands_agents::context_manager::{
    ContextManagerConfig, ContextManagerOption, StashOption, AGENTIC_TRUNCATE_THRESHOLD,
    AUTO_SUMMARIZE_UTILIZATION,
};
use strands_agents::plugins::vended::GoalLoop;
use strands_agents::sandbox::LocalSandbox;
use strands_agents::session::{SaveLatestStrategy, SessionManagerOptions};
use strands_agents::tools::ToolRef;
use strands_agents::Plugin;
use strands_agents::{FileStorage, Result, SessionManager};
use strands_harness::defaults::DEFAULT_BUILTIN_TOOLS;
use strands_harness::{
    create_harness, BuiltinToolsConfig, Effort, EnvironmentContext, EnvironmentContextConfig,
    HarnessAgentOptions, MemoryOption, ModelOption, SandboxOption, SessionOption,
};

use crate::runtime::{AgentFactory, AgentSpec};

/// `subagent` is replaced by the swarm tools; `web_search` fails construction on models without
/// native search (every Bedrock model), so it is opt-in through MCP instead.
const EXCLUDED_BUILTINS: &[&str] = &["subagent", "web_search"];

/// Preview kept in context when a large tool result is offloaded to the stash.
const TRUNCATE_PREVIEW_TOKENS: usize = 2_000;

/// Settings shared by every agent the factory builds.
#[derive(Clone, Default)]
pub struct HarnessFactoryOptions {
    /// Default model (`provider/name` or a Bedrock id); the harness default when `None`.
    pub model: Option<String>,
    /// Reasoning effort.
    pub effort: Effort,
    /// Extra instructions given to every agent, before the swarm section.
    pub instructions: Option<String>,
    /// Tools shared by every agent, e.g. MCP tools loaded once for the whole swarm.
    pub tools: Vec<ToolRef>,
}

/// A factory that builds harness agents.
///
/// Sessions are saved only for runs with a state directory; long-term memory is off, since a shared
/// memory directory would interleave writes from concurrent agents.
pub fn harness_factory(options: HarnessFactoryOptions) -> AgentFactory {
    Arc::new(move |spec: AgentSpec| {
        let options = options.clone();
        Box::pin(async move { create_harness(harness_options(&options, spec)?).await })
    })
}

/// A session manager for `agent_id` under `dir`, saved after every message so a crash loses at
/// most the in-flight model call.
pub fn agent_session_manager(dir: &std::path::Path, agent_id: &str) -> Result<SessionManager> {
    let options = SessionManagerOptions {
        save_latest_on: SaveLatestStrategy::Message,
        ..Default::default()
    };
    SessionManager::with_options(agent_id, Arc::new(FileStorage::new(dir)), options)
}

fn agent_session(dir: &std::path::Path, agent_id: &str) -> Result<SessionOption> {
    Ok(SessionOption::Instance(Arc::new(agent_session_manager(
        dir, agent_id,
    )?)))
}

/// The `auto` pipeline with coding-sized limits: the preset offloads any tool result over 1,500
/// tokens, which stashes most file reads and command outputs and costs a retrieval call each.
/// Coding agents keep results up to the `agentic` threshold inline, with a larger preview, and
/// still summarize old turns near the context limit.
fn context_manager() -> ContextManagerOption {
    let truncate = Offload::truncate(
        OffloadTarget::ToolResults,
        TruncateConfig {
            preview_tokens: Some(TRUNCATE_PREVIEW_TOKENS),
            preview: None,
        },
    )
    .when(OffloadConditions {
        threshold: Some(AGENTIC_TRUNCATE_THRESHOLD),
        ..Default::default()
    });
    let summarize = Offload::summarize(OffloadTarget::All).when(OffloadConditions {
        utilization: Some(AUTO_SUMMARIZE_UTILIZATION),
        preserve_recent: Some(4.0),
        ..Default::default()
    });
    ContextManagerOption::Config(ContextManagerConfig {
        strategies: Some(vec![Arc::new(truncate), Arc::new(summarize)]),
        stash: StashOption::Enabled,
    })
}

fn harness_options(options: &HarnessFactoryOptions, spec: AgentSpec) -> Result<HarnessAgentOptions> {
    let session = match &spec.session_dir {
        Some(dir) => agent_session(dir, &spec.id)?,
        None => SessionOption::Off,
    };
    let builtins = DEFAULT_BUILTIN_TOOLS
        .iter()
        .copied()
        .filter(|name| !EXCLUDED_BUILTINS.contains(name));
    let environment = EnvironmentContext::new(EnvironmentContextConfig {
        name: None,
        root: Some(spec.cwd.clone()),
    });
    let mut plugins: Vec<Arc<dyn Plugin>> = vec![Arc::new(environment)];
    if let Some(goal) = &spec.goal {
        plugins.push(Arc::new(GoalLoop::goal(goal.clone())));
    }
    let mut sandbox = LocalSandbox::with_base_dir(spec.cwd);
    if !spec.env.is_empty() {
        sandbox = sandbox.with_env(spec.env.into_iter().collect());
    }
    let mut tools = spec.tools;
    tools.extend(options.tools.iter().cloned());
    let instructions = match &options.instructions {
        Some(extra) => format!("{extra}\n\n{}", spec.instructions),
        None => spec.instructions,
    };
    Ok(HarnessAgentOptions {
        model: spec.model.or_else(|| options.model.clone()).map(ModelOption::Id),
        effort: options.effort,
        instructions: Some(instructions),
        tools,
        plugins,
        builtin_tools: Some(BuiltinToolsConfig::list(builtins)),
        sandbox: SandboxOption::Instance(Arc::new(sandbox)),
        session,
        memory: MemoryOption::Off,
        context_manager: context_manager(),
        id: Some(spec.id),
        name: spec.name,
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use strands_agents::context_manager::ContextManager;
    use strands_agents::testing::MockMessageModel;
    use strands_agents::types::{
        ContentBlock, Message, Role, ToolResultBlock, ToolResultContent, ToolUseBlock,
    };
    use strands_agents::Agent;

    fn exchange(id: &str, chars: usize) -> [Message; 2] {
        let call = ToolUseBlock {
            tool_use_id: id.to_owned(),
            name: "shell".to_owned(),
            input: serde_json::json!({}),
            reasoning_signature: None,
        };
        let result = ToolResultBlock::success(id, vec![ToolResultContent::Text("x".repeat(chars))]);
        [
            Message::new(Role::Assistant, vec![ContentBlock::ToolUse(call)]),
            Message::new(Role::User, vec![ContentBlock::ToolResult(result)]),
        ]
    }

    fn result_len(message: &Message) -> usize {
        message.tool_results().next().unwrap().text().chars().count()
    }

    #[tokio::test]
    async fn keeps_coding_sized_tool_results_inline_and_offloads_huge_ones() {
        let agent = Agent::builder()
            .model(Arc::new(MockMessageModel::new(Vec::new())))
            .build()
            .unwrap();
        let manager = ContextManager::from(context_manager()).unwrap();
        let mut messages = vec![Message::user("start")];
        messages.extend(exchange("file-read", 16_000));
        messages.extend(exchange("huge-log", 200_000));

        manager
            .run_strategies_on(&agent, &mut messages, Some(0), false)
            .await
            .unwrap();

        assert_eq!(result_len(&messages[2]), 16_000);
        assert!(result_len(&messages[4]) < 20_000);
    }
}
