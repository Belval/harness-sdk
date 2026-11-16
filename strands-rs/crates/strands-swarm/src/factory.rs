//! The default [`AgentFactory`]: every agent is a [`create_harness`] agent rooted at its own
//! working directory, with the swarm tools in place of the harness's `subagent`.

use std::sync::Arc;

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
        sandbox: SandboxOption::Instance(Arc::new(LocalSandbox::with_base_dir(spec.cwd))),
        session,
        memory: MemoryOption::Off,
        id: Some(spec.id),
        name: spec.name,
        ..Default::default()
    })
}
