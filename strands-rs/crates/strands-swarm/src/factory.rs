//! The default [`AgentFactory`]: every agent is a [`create_harness`] agent rooted at its own
//! working directory, with the swarm tools in place of the harness's `subagent`.

use std::sync::Arc;

use strands_agents::context_manager::retrieval_tool::RETRIEVAL_TOOL_NAME;
use strands_agents::context_manager::strategies::{
    Offload, OffloadConditions, OffloadTarget, TruncateConfig,
};
use strands_agents::context_manager::{
    ContextManagerConfig, ContextManagerOption, StashConfig, StashOption, AGENTIC_TRUNCATE_THRESHOLD,
};
use strands_agents::plugins::vended::{GoalLoop, JudgeConfig, Skill};
use strands_agents::sandbox::LocalSandbox;
use strands_agents::session::{SaveLatestStrategy, SessionManagerOptions};
use strands_agents::storage::{namespace, LocalFileStorage};
use strands_agents::tools::{Tool, ToolContext, ToolRef};
use strands_agents::types::{ToolResultBlock, ToolSpec};
use strands_agents::Plugin;
use strands_agents::{Agent, FileStorage, Result, SessionManager};
use strands_harness::defaults::{DEFAULT_BUILTIN_TOOLS, DEFAULT_MODEL};
use strands_harness::{
    create_harness, resolve_model, BackgroundTasksOption, BuiltinToolsConfig, Effort, EnvironmentContext,
    EnvironmentContextConfig, HarnessAgentOptions, MemoryOption, ModelOption, ResolveModelOptions,
    SandboxOption, SessionOption, SkillsOption,
};

use crate::catalog::ToolCatalog;
use crate::compaction::{CompactAt, CompactedSink, DEFAULT_COMPACT_AT};
use crate::dashboard::ContextControl;
use crate::pruning::{CachePrune, CACHE_TTL, CACHE_TTL_1H};
use crate::runtime::{AgentFactory, AgentSpec, ModelResolver, GOAL_MAX_ATTEMPTS};
use crate::skills::{skills_prompt, skills_tool};

/// `subagent` is replaced by the swarm tools; `web_search` fails construction on models without
/// native search (every Bedrock model), so it is opt-in through MCP instead.
const EXCLUDED_BUILTINS: &[&str] = &["subagent", "web_search"];

/// Preview kept in context when a large tool result is offloaded to the stash.
/// How to use the file tools: agents otherwise script edits with `python3`/`sed` through `shell`,
/// which hides the change from the log and skips the uniqueness check `edit` makes.
const FILE_TOOLS_PROMPT: &str = "# Files\n\
Read files with `read` (use `offset`/`limit` to page through large ones) instead of `cat`/`sed -n`; \
change them with `edit` (replaces one exact, unique string) or `write` (whole file) instead of \
`python3`/`sed` scripts in `shell`. Search with `rg -n` (or `grep -n`) in `shell`. The file tools \
take absolute paths; relative ones do not resolve against your working directory.";

const TRUNCATE_PREVIEW_TOKENS: usize = 2_000;

/// Transcript the goal judge sees, in characters (about 10k tokens): the task, the latest
/// messages and the reply under judgement. A long-running agent's whole conversation would be
/// resent on every turn it ends.
const JUDGE_TRANSCRIPT_CHARS: usize = 40_000;

/// Settings shared by every agent the factory builds.
#[derive(Clone, Default)]
pub struct HarnessFactoryOptions {
    /// Default model (`provider/name` or a Bedrock id); the harness default when `None`.
    pub model: Option<String>,
    /// Reasoning effort.
    pub effort: Effort,
    /// Extra instructions given to every agent, before the swarm section.
    pub instructions: Option<String>,
    /// Tools registered on every agent up front.
    pub tools: Vec<ToolRef>,
    /// Tools every agent can enable on demand through `find_tools`, e.g. the shared MCP tools.
    /// `find_tools` is registered only when the catalog has tools.
    pub catalog: Option<Arc<ToolCatalog>>,
    /// Skills offered to every agent: one line each in the system prompt, loaded with the
    /// `skills` tool. Neither when empty.
    pub skills: Arc<Vec<Skill>>,
    /// Summarize older turns once a call's input reaches this many tokens (or 85% of the
    /// model's window, if smaller); [`DEFAULT_COMPACT_AT`] when `None`.
    pub compact_at: Option<u64>,
    /// TTL of the prompt-cache points (`"5m"`, `"1h"`); the provider default (5m) when `None`.
    pub cache_ttl: Option<String>,
}

/// A factory that builds harness agents.
///
/// Sessions are saved only for runs with a state directory; long-term memory is off, since a shared
/// memory directory would interleave writes from concurrent agents.
pub fn harness_factory(options: HarnessFactoryOptions) -> AgentFactory {
    Arc::new(move |spec: AgentSpec| {
        let options = options.clone();
        Box::pin(async move { build(&options, harness_options(&options, spec)?).await })
    })
}

async fn build(options: &HarnessFactoryOptions, harness: HarnessAgentOptions) -> Result<Agent> {
    let agent = create_harness(harness).await?;
    agent.with_tool_registry(|registry| {
        let Some(tool) = registry.get(RETRIEVAL_TOOL_NAME) else {
            return Ok(());
        };
        registry.add_or_replace([Reworded::tool(tool, RETRIEVAL_DESCRIPTION)])
    })?;
    // The harness has restored the session by now; give back the tools enabled before.
    if let Some(catalog) = &options.catalog {
        catalog.restore(&agent);
    }
    Ok(agent)
}

/// `retrieve_context`'s description without the SDK's options list and examples, which repeat
/// its input schema; every call pays for it.
const RETRIEVAL_DESCRIPTION: &str = concat!(
    "Read back content offloaded from your context (truncated, dropped or summarized) by the ref ",
    "left in its place: all of it, the lines matching `pattern`, or a `line_range`."
);

/// A tool with its description replaced.
struct Reworded {
    inner: ToolRef,
    spec: ToolSpec,
}

impl Reworded {
    fn tool(inner: ToolRef, description: &str) -> ToolRef {
        let mut spec = inner.spec().clone();
        spec.description = description.to_owned();
        Arc::new(Self { inner, spec })
    }
}

#[async_trait::async_trait]
impl Tool for Reworded {
    fn name(&self) -> &str {
        &self.spec.name
    }

    fn description(&self) -> &str {
        &self.spec.description
    }

    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn invoke(&self, ctx: ToolContext) -> Result<ToolResultBlock> {
        self.inner.invoke(ctx).await
    }
}

/// The [`ModelResolver`] matching [`harness_factory`]: resolves an id (`None`: the factory's
/// default) exactly as `create_harness` does, with the same effort and caching, so a switched
/// agent runs as if it had been built with that model.
pub fn harness_model_resolver(options: HarnessFactoryOptions) -> ModelResolver {
    Arc::new(move |model: Option<String>| {
        let model = model_option(&options, model);
        let effort = options.effort;
        let cache_ttl = options.cache_ttl.clone();
        Box::pin(async move {
            resolve_model(ResolveModelOptions {
                model,
                default: DEFAULT_MODEL.to_owned(),
                effort,
                cache_ttl,
                // `web_search` is excluded from the builtins, so the harness never asks for it.
                web_search: false,
                ..Default::default()
            })
            .await
        })
    })
}

fn model_option(options: &HarnessFactoryOptions, model: Option<String>) -> Option<ModelOption> {
    model.or_else(|| options.model.clone()).map(ModelOption::Id)
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
/// summarize old turns at `compact_at` tokens (see [`CompactAt`]). Earlier turns' reasoning and
/// older large `shell`/`read` results are pruned only when the prompt cache is lost anyway (see
/// [`CachePrune`]). With `stash_dir` the full
/// outputs behind truncated results are kept there, so `retrieve_context` still finds them after
/// a restart.
/// `control` (the agent's `context` tool) gets the stash and can request a summary; each
/// summary is reported to `on_compacted`.
fn context_manager(
    compact_at: u64,
    stash_dir: Option<&std::path::Path>,
    control: &Arc<ContextControl>,
    on_compacted: Option<CompactedSink>,
    cache_ttl: Option<&str>,
) -> ContextManagerOption {
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
    let summarize = Arc::new(
        CompactAt::new(compact_at)
            .on_request(Arc::clone(control))
            .on_compacted(on_compacted),
    );
    let prune = CachePrune::new(Some(Arc::clone(&summarize))).with_cache_ttl(match cache_ttl {
        Some("1h") => CACHE_TTL_1H,
        _ => CACHE_TTL,
    });
    ContextManagerOption::Config(ContextManagerConfig {
        strategies: Some(vec![
            Arc::new(prune.clone()),
            Arc::new(truncate),
            summarize,
            control.strategy(),
            // Last: what the call sends, for the next call's gate.
            prune.mark(),
        ]),
        stash: match stash_dir {
            // A namespaced view is used as the stash root as is; plain storage would be keyed by
            // the agent's session id, which is new in every process.
            Some(dir) => StashOption::Config(StashConfig {
                storage: Some(namespace(
                    Arc::new(LocalFileStorage::new(dir.to_string_lossy())),
                    "stash",
                )),
                retrieval_tool: true,
            }),
            None => StashOption::Enabled,
        },
    })
}

/// `goal` with the swarm's attempt cap ([`GOAL_MAX_ATTEMPTS`]).
pub fn bounded_goal(goal: GoalLoop) -> GoalLoop {
    goal.with_max_attempts(GOAL_MAX_ATTEMPTS)
}

fn harness_options(options: &HarnessFactoryOptions, spec: AgentSpec) -> Result<HarnessAgentOptions> {
    let session = match &spec.session_dir {
        Some(dir) => agent_session(dir, &spec.id)?,
        None => SessionOption::Off,
    };
    let stash_dir = spec.session_dir.as_ref().map(|dir| dir.join(&spec.id));
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
        let judge = JudgeConfig::default().with_max_transcript_chars(JUDGE_TRANSCRIPT_CHARS);
        plugins.push(Arc::new(bounded_goal(
            GoalLoop::goal(goal.clone()).with_judge(judge),
        )));
    }
    let mut sandbox = LocalSandbox::with_base_dir(spec.cwd);
    if !spec.env.is_empty() {
        sandbox = sandbox.with_env(spec.env.into_iter().collect());
    }
    let compact_at = options.compact_at.unwrap_or(DEFAULT_COMPACT_AT);
    let control = ContextControl::new(compact_at);
    let mut tools = spec.tools;
    tools.extend(options.tools.iter().cloned());
    tools.push(control.tool());
    if let Some(catalog) = options.catalog.as_ref().filter(|catalog| !catalog.is_empty()) {
        tools.push(catalog.find_tool());
    }
    let mut instructions = match &options.instructions {
        Some(extra) => format!("{extra}\n\n{}", spec.instructions),
        None => spec.instructions,
    };
    instructions = format!("{instructions}\n\n{FILE_TOOLS_PROMPT}");
    if !options.skills.is_empty() {
        instructions = format!("{instructions}\n\n{}", skills_prompt(&options.skills));
        tools.push(skills_tool(options.skills.clone()));
    }
    Ok(HarnessAgentOptions {
        model: model_option(options, spec.model),
        effort: options.effort,
        cache_ttl: options.cache_ttl.clone(),
        instructions: Some(instructions),
        tools,
        plugins,
        builtin_tools: Some(BuiltinToolsConfig::list(builtins)),
        sandbox: SandboxOption::Instance(Arc::new(sandbox)),
        session,
        memory: MemoryOption::Off,
        // `run_background` covers slow shell commands and agents run concurrently anyway; the SDK's
        // per-tool `_background_execution` flag and its manage tool cost every call ~700 tokens.
        background_tasks: BackgroundTasksOption::Off,
        // Discovery happens once at startup (crate::skills); the harness's own `./.agent/skills`
        // lookup, relative to the process rather than the agent, stays off.
        skills: SkillsOption::Off,
        context_manager: context_manager(
            compact_at,
            stash_dir.as_deref(),
            &control,
            spec.on_compacted,
            options.cache_ttl.as_deref(),
        ),
        id: Some(spec.id),
        name: Some(spec.name),
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
        let manager = ContextManager::from(context_manager(
            DEFAULT_COMPACT_AT,
            None,
            &ContextControl::new(DEFAULT_COMPACT_AT),
            None,
            None,
        ))
        .unwrap();
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

    fn chat(turns: usize) -> Vec<Message> {
        (0..turns)
            .flat_map(|i| {
                [
                    Message::user(format!("question {i} {}", "q".repeat(400))),
                    Message::assistant(format!("answer {i} {}", "a".repeat(400))),
                ]
            })
            .collect()
    }

    #[tokio::test]
    async fn summarizes_older_turns_once_input_reaches_compact_at() {
        use strands_agents::testing::MockResponse;
        let model = MockMessageModel::shared(vec![MockResponse::message(Message::assistant(
            "SUMMARY of the early turns",
        ))]);
        let agent = Agent::builder().model(model.clone()).build().unwrap();
        // The mock has no known window, so the context manager assumes 200k: 5k tokens is 2.5%,
        // far below the 85% mark, but past `compact_at`.
        let manager = ContextManager::from(context_manager(
            4_000,
            None,
            &ContextControl::new(4_000),
            None,
            None,
        ))
        .unwrap();

        let mut below = chat(6);
        let changed = manager
            .run_strategies_on(&agent, &mut below, Some(3_000), false)
            .await
            .unwrap();
        assert!(!changed);
        assert_eq!(below.len(), 12);
        assert_eq!(model.call_count(), 0);

        let mut above = chat(6);
        let changed = manager
            .run_strategies_on(&agent, &mut above, Some(5_000), false)
            .await
            .unwrap();
        assert!(changed);
        assert_eq!(model.call_count(), 1);
        assert!(above.len() < 12, "{} messages", above.len());
        assert!(above
            .iter()
            .any(|m| m.text().contains("SUMMARY of the early turns")));
        assert!(
            above.last().unwrap().text().starts_with("answer 5"),
            "recent turns kept"
        );
    }

    #[tokio::test]
    async fn the_context_tool_can_request_a_summary_below_compact_at() {
        use strands_agents::testing::MockResponse;
        let model = MockMessageModel::shared(vec![MockResponse::message(Message::assistant(
            "SUMMARY on request",
        ))]);
        let agent = Agent::builder().model(model.clone()).build().unwrap();
        let control = ContextControl::new(DEFAULT_COMPACT_AT);
        let manager =
            ContextManager::from(context_manager(DEFAULT_COMPACT_AT, None, &control, None, None)).unwrap();
        control.request_summary();

        let mut messages = chat(6);
        let changed = manager
            .run_strategies_on(&agent, &mut messages, Some(3_000), false)
            .await
            .unwrap();
        assert!(changed, "summarized although far below the point");
        assert!(messages.iter().any(|m| m.text().contains("SUMMARY on request")));

        let mut again = chat(6);
        let changed = manager
            .run_strategies_on(&agent, &mut again, Some(3_000), false)
            .await
            .unwrap();
        assert!(!changed, "a request is used once");
        assert_eq!(model.call_count(), 1);
    }

    #[tokio::test]
    async fn stashed_tool_output_outlives_the_process_with_a_stash_dir() {
        use strands_agents::Plugin;
        let dir = tempfile::tempdir().unwrap();
        let stash_of = |agent: Agent| {
            let dir = dir.path().to_owned();
            async move {
                let manager = ContextManager::from(context_manager(
                    DEFAULT_COMPACT_AT,
                    Some(&dir),
                    &ContextControl::new(DEFAULT_COMPACT_AT),
                    None,
                    None,
                ))
                .unwrap();
                manager.init_agent(&agent).await.unwrap();
                manager.stash().unwrap()
            }
        };
        let agent = || {
            Agent::builder()
                .id("a1")
                .model(Arc::new(MockMessageModel::new(Vec::new())))
                .build()
                .unwrap()
        };
        let [_, result] = exchange("t1", 3_000);
        stash_of(agent())
            .await
            .store_message(&result, &Default::default())
            .await;

        let restarted = stash_of(agent()).await;
        let full = restarted
            .retrieve("t1_0")
            .await
            .expect("the output is still stashed");
        assert!(full.to_string().contains(&"x".repeat(3_000)), "{full}");
    }

    use strands_agents::tools::FunctionTool;

    fn spec() -> AgentSpec {
        AgentSpec {
            id: "a1".into(),
            name: "t".into(),
            cwd: std::env::temp_dir(),
            model: None,
            instructions: String::new(),
            tools: Vec::new(),
            goal: None,
            session_dir: None,
            env: Default::default(),
            on_compacted: None,
        }
    }

    fn tool_names(catalog: Option<Arc<ToolCatalog>>) -> Vec<String> {
        let options = HarnessFactoryOptions {
            catalog,
            ..Default::default()
        };
        let harness = harness_options(&options, spec()).unwrap();
        harness.tools.iter().map(|t| t.name().to_owned()).collect()
    }

    #[test]
    fn find_tools_is_registered_only_for_a_non_empty_catalog() {
        let tool: ToolRef = Arc::new(
            FunctionTool::builder("srv_echo")
                .description("Echo.")
                .handler(|input: serde_json::Value, _ctx| async move { Result::Ok(input) }),
        );
        let full = ToolCatalog::new(vec![tool], vec![("srv".into(), 1)]);
        assert_eq!(tool_names(Some(Arc::new(full))), ["context", "find_tools"]);
        let empty = ToolCatalog::new(Vec::new(), vec![("srv".into(), 0)]);
        assert_eq!(tool_names(Some(Arc::new(empty))), ["context"]);
        assert_eq!(tool_names(None), ["context"]);
    }

    /// What an agent sends besides its messages, serialized: the prompt cache is keyed on these
    /// bytes, so they must not change between processes or across a resume.
    fn request_prefix(options: &strands_agents::models::StreamOptions) -> String {
        format!(
            "{}\n{}",
            serde_json::to_string(&options.system_prompt).unwrap(),
            serde_json::to_string(&options.tool_specs).unwrap()
        )
    }

    /// Builds the agent `harness_factory` would, on `model`, and runs one turn.
    async fn run_turn(
        options: &HarnessFactoryOptions,
        spec: AgentSpec,
        responses: Vec<strands_agents::testing::MockResponse>,
    ) -> Vec<String> {
        let mut harness = harness_options(options, spec).unwrap();
        let model = MockMessageModel::shared(responses);
        harness.model = Some(ModelOption::Instance(model.clone()));
        let agent = build(options, harness).await.unwrap();
        agent.invoke("go").await.unwrap();
        model
            .calls()
            .iter()
            .map(|(_, options)| request_prefix(options))
            .collect()
    }

    #[tokio::test]
    async fn the_system_prompt_and_tool_specs_are_identical_across_builds_and_resumes() {
        use strands_agents::testing::MockResponse;
        let dir = tempfile::tempdir().unwrap();
        let tool = |name: &str| -> ToolRef {
            Arc::new(
                FunctionTool::builder(name)
                    .description(format!("Tool {name}."))
                    .handler(|input: serde_json::Value, _ctx| async move { Result::Ok(input) }),
            )
        };
        let names = ["srv_d", "srv_b", "srv_a", "srv_c", "other_x"];
        let catalog = ToolCatalog::new(
            names.iter().map(|name| tool(name)).collect(),
            vec![("srv".into(), 4), ("other".into(), 1)],
        );
        let mut skill = Skill::new("review", "Review a change.", "Read the diff.");
        skill.path = Some(dir.path().join("review"));
        let options = HarnessFactoryOptions {
            instructions: Some("Be brief.".into()),
            tools: vec![tool("shared_one"), tool("shared_two")],
            catalog: Some(Arc::new(catalog)),
            skills: Arc::new(vec![skill]),
            ..Default::default()
        };
        let spec = || AgentSpec {
            instructions: "# Swarm\nYou are a1.".into(),
            tools: vec![tool("spawn_agent"), tool("wait")],
            ..spec()
        };
        let fresh = || async {
            run_turn(
                &options,
                spec(),
                vec![MockResponse::message(Message::assistant("done"))],
            )
            .await
        };
        assert_eq!(fresh().await, fresh().await, "two fresh builds");

        let sessions = dir.path().join("sessions");
        let saved = || AgentSpec {
            session_dir: Some(sessions.clone()),
            ..spec()
        };
        let enable = ToolUseBlock::new(
            "find_tools",
            "f1",
            serde_json::json!({ "enable": ["srv_c", "srv_a", "other_x"] }),
        );
        let before = run_turn(
            &options,
            saved(),
            vec![
                MockResponse::message(Message::new(Role::Assistant, vec![ContentBlock::ToolUse(enable)])),
                MockResponse::message(Message::assistant("done")),
            ],
        )
        .await;
        assert_ne!(before[0], before[1], "enabling tools changes the specs");
        let after = run_turn(
            &options,
            saved(),
            vec![MockResponse::message(Message::assistant("again"))],
        )
        .await;
        assert_eq!(after[0], before[1], "a resumed agent sends what it sent last");
    }

    #[tokio::test]
    async fn agents_see_the_skill_catalog_and_load_a_skill_on_demand() {
        use strands_agents::testing::MockResponse;
        let dir = tempfile::tempdir().unwrap();
        let mut skill = Skill::new("review", "Review a change.", "Step 1: read the diff.");
        skill.path = Some(dir.path().join("review"));
        let options = HarnessFactoryOptions {
            skills: Arc::new(vec![skill]),
            ..Default::default()
        };
        let mut harness = harness_options(&options, spec()).unwrap();
        assert!(harness.instructions.as_deref().unwrap().contains("`skills` tool"));
        let call = ToolUseBlock::new("skills", "s1", serde_json::json!({ "skill_name": "review" }));
        let model = MockMessageModel::shared(vec![
            MockResponse::message(Message::new(Role::Assistant, vec![ContentBlock::ToolUse(call)])),
            MockResponse::message(Message::assistant("done")),
        ]);
        harness.model = Some(ModelOption::Instance(model.clone()));
        let agent = create_harness(harness).await.unwrap();
        agent.invoke("review my change").await.unwrap();

        let calls = model.calls();
        let prompt = calls[0].1.system_prompt.as_ref().unwrap().text();
        assert!(prompt.contains("\n- review: Review a change."), "{prompt}");
        assert!(!prompt.contains("Step 1"));
        assert!(!prompt.contains("<available_skills>"), "the SDK plugin stays off");
        assert!(
            !prompt.contains(&dir.path().display().to_string()),
            "no paths in the prompt"
        );
        let loaded = calls[1].0.last().unwrap().tool_results().next().unwrap().text();
        assert!(loaded.contains("Step 1: read the diff."), "{loaded}");
        assert!(loaded.contains("review/SKILL.md"), "{loaded}");
    }

    /// The swarm prompt tells agents not to `cd` into their working directory; that only holds
    /// while the shell tool runs in the agent's sandbox.
    #[tokio::test]
    async fn shell_commands_start_in_the_agents_working_directory() {
        use strands_agents::testing::MockResponse;
        let dir = tempfile::tempdir().unwrap();
        let cwd = dir.path().canonicalize().unwrap();
        let mut harness = harness_options(
            &HarnessFactoryOptions::default(),
            AgentSpec {
                cwd: cwd.clone(),
                ..spec()
            },
        )
        .unwrap();
        let call = ToolUseBlock::new("shell", "s1", serde_json::json!({ "command": "pwd" }));
        let model = MockMessageModel::shared(vec![
            MockResponse::message(Message::new(Role::Assistant, vec![ContentBlock::ToolUse(call)])),
            MockResponse::message(Message::assistant("done")),
        ]);
        harness.model = Some(ModelOption::Instance(model.clone()));
        let agent = create_harness(harness).await.unwrap();
        agent.invoke("where am I").await.unwrap();

        let calls = model.calls();
        // The shell result is a JSON block (`{"output": ..., "exit_code": ...}`), not text.
        let result = calls[1].0.last().unwrap().tool_results().next().unwrap();
        let output = format!("{:?}", result.content);
        assert!(output.contains(&cwd.display().to_string()), "{output}");
    }

    /// Through the whole harness (every plugin and hook), a call that has no reason to prune
    /// re-sends the previous call's messages unchanged, so the prompt cache matches all of them.
    #[tokio::test]
    async fn the_next_call_resends_history_unchanged() {
        use strands_agents::testing::MockResponse;
        let dir = tempfile::tempdir().unwrap();
        let mut harness = harness_options(
            &HarnessFactoryOptions::default(),
            AgentSpec {
                cwd: dir.path().to_path_buf(),
                ..spec()
            },
        )
        .unwrap();
        let call = ToolUseBlock::new("shell", "s1", serde_json::json!({ "command": "echo hi" }));
        let model = MockMessageModel::shared(vec![
            MockResponse::message(Message::new(Role::Assistant, vec![ContentBlock::ToolUse(call)])),
            MockResponse::message(Message::assistant("done")),
            MockResponse::message(Message::assistant("again")),
        ]);
        harness.model = Some(ModelOption::Instance(model.clone()));
        let agent = create_harness(harness).await.unwrap();
        agent
            .invoke(format!("first task {}", "x".repeat(4_000)))
            .await
            .unwrap();
        agent.invoke("second task").await.unwrap();

        let calls = model.calls();
        // Without the per-call environment block the injector folds into the last user message
        // (after the cache point, by design).
        let json = |messages: &[Message]| {
            let mut messages = messages.to_vec();
            for message in &mut messages {
                message.content.retain(
                    |block| !matches!(block, ContentBlock::Text(text) if text.contains("<system-reminder>")),
                );
            }
            serde_json::to_string(&messages).unwrap()
        };
        for pair in calls.windows(2) {
            let (before, after) = (&pair[0].0, &pair[1].0);
            assert_eq!(
                json(&after[..before.len()]),
                json(before),
                "the sent prefix changed"
            );
        }
        assert_eq!(calls.len(), 3);
    }

    #[test]
    fn agents_are_told_to_prefer_the_file_tools_over_shell_edits() {
        let harness = harness_options(&HarnessFactoryOptions::default(), spec()).unwrap();
        let names: Vec<_> = DEFAULT_BUILTIN_TOOLS
            .iter()
            .filter(|name| !EXCLUDED_BUILTINS.contains(name))
            .collect();
        for tool in ["read", "edit", "write", "shell"] {
            assert!(names.contains(&&tool), "{tool} is a builtin");
        }
        let instructions = harness.instructions.unwrap();
        assert!(instructions.contains("`edit`"), "{instructions}");
        assert!(instructions.contains("`offset`/`limit`"), "{instructions}");
    }

    #[test]
    fn no_skills_means_no_skills_tool_or_instructions() {
        let harness = harness_options(&HarnessFactoryOptions::default(), spec()).unwrap();
        assert!(matches!(harness.skills, SkillsOption::Off));
        assert!(!harness.instructions.unwrap().contains("skills"));
        assert!(harness.tools.iter().all(|tool| tool.name() != "skills"));
    }

    #[tokio::test]
    async fn the_resolver_rejects_an_unknown_provider_and_falls_back_to_the_default() {
        let resolve = harness_model_resolver(HarnessFactoryOptions::default());
        let error = resolve(Some("nope/some-model".into()))
            .await
            .err()
            .expect("rejected");
        assert!(error.to_string().contains("nope"), "{error}");

        let resolve = harness_model_resolver(HarnessFactoryOptions {
            model: Some("nope/default".into()),
            ..Default::default()
        });
        let error = resolve(None).await.err().expect("default used");
        assert!(error.to_string().contains("nope"), "{error}");
    }

    /// The tree's context % and the summarization gate both need the window of the models the
    /// swarm runs; an unknown one would silently fall back to a 200k default.
    #[tokio::test]
    async fn the_swarms_models_have_a_known_context_window() {
        let resolve = harness_model_resolver(HarnessFactoryOptions::default());
        for id in [
            "bedrock/global.anthropic.claude-opus-5-5",
            "bedrock/global.anthropic.claude-sonnet-5-5",
            "bedrock/us.anthropic.claude-haiku-5-5",
            "bedrock/us.anthropic.claude-opus-4-6-v1",
            "global.anthropic.claude-opus-5",
        ] {
            let model = resolve(Some(id.into())).await.unwrap();
            assert!(model.context_window_limit().is_some(), "{id}");
        }
        let default = resolve(None).await.unwrap();
        assert!(default.context_window_limit().is_some(), "{DEFAULT_MODEL}");
        for id in ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"] {
            assert!(
                strands_agents::models::defaults::get_context_window_limit(id).is_some(),
                "anthropic/{id}"
            );
        }
    }

    /// The fixed prefix a root agent sends on every call, built the way a real run builds it
    /// (persistent, swarm prompt and tools, memory, an MCP catalog, two skills), split
    /// into `(part, bytes)`: the system prompt sections, then one entry per tool spec.
    async fn fixed_prefix() -> Vec<(String, usize)> {
        use crate::runtime::{SpawnRequest, Swarm, SwarmConfig, USER};
        use strands_agents::testing::MockResponse;
        let dir = tempfile::tempdir().unwrap();
        let tool = |name: &str| -> ToolRef {
            Arc::new(
                FunctionTool::builder(name)
                    .description("x")
                    .handler(|input: serde_json::Value, _ctx| async move { Result::Ok(input) }),
            )
        };
        let servers = [
            ("builder-mcp", 55),
            ("chorus-mcp", 8),
            ("slack-mcp", 115),
            ("quick-evals", 95),
        ];
        let catalog = ToolCatalog::new(
            vec![tool("builder-mcp_x")],
            servers.iter().map(|(name, n)| (name.to_string(), *n)).collect(),
        );
        let skills = ["review", "deploy"].map(|name| {
            let mut skill = Skill::new(name, "Do the thing. Use when asked to.", "Steps.");
            skill.path = Some(dir.path().join(name));
            skill
        });
        let options = HarnessFactoryOptions {
            catalog: Some(Arc::new(catalog)),
            skills: Arc::new(skills.to_vec()),
            ..Default::default()
        };
        let model = MockMessageModel::shared(vec![MockResponse::message(Message::assistant("ok"))]);
        let shared = model.clone();
        let factory: AgentFactory = Arc::new(move |spec: AgentSpec| {
            let options = options.clone();
            let model = shared.clone();
            Box::pin(async move {
                let mut harness = harness_options(&options, spec)?;
                harness.model = Some(ModelOption::Instance(model));
                build(&options, harness).await
            })
        });
        let mut config = SwarmConfig::new(factory, dir.path());
        config.memory = Some(Arc::new(crate::Memory::new(dir.path().join("memory.md"))));
        config.state_dir = Some(dir.path().join("run"));
        let (swarm, _events) = Swarm::new(config);
        let request = SpawnRequest {
            task: "go".into(),
            ..Default::default()
        };
        swarm.spawn(USER, request).await.unwrap();
        while model.call_count() == 0 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        swarm.shutdown().await;
        let calls = model.calls();
        let options = &calls[0].1;
        let prompt = options.system_prompt.as_ref().unwrap().text();
        let path = dir.path().display().to_string();
        let prompt = prompt.replace(&path, "/home/user/project");
        if let Ok(out) = std::env::var("STRANDS_SWARM_DUMP_PREFIX") {
            let specs = serde_json::to_string_pretty(&options.tool_specs).unwrap();
            std::fs::write(&out, format!("{prompt}\n\n=== TOOLS ===\n{specs}")).unwrap();
        }
        let mut parts = Vec::new();
        let mut section = String::from("(preamble)");
        let mut text = String::new();
        for line in prompt.split_inclusive('\n') {
            if let Some(title) = line.strip_prefix("# ") {
                parts.push((section, text.len()));
                section = title.trim().to_owned();
                text.clear();
            }
            text.push_str(line);
        }
        parts.push((section, text.len()));
        for spec in &options.tool_specs {
            let json = serde_json::to_string(spec).unwrap();
            parts.push((format!("tool {}", spec.name), json.len()));
        }
        parts
    }

    /// The prefix is paid on every model call of every agent; this pins its size so it cannot
    /// regrow unnoticed. Raise a budget only for something agents need on every call.
    /// `cargo test -p strands-swarm --lib fixed_prefix -- --nocapture` prints the breakdown;
    /// `STRANDS_SWARM_DUMP_PREFIX=<file>` also writes the prompt and tool specs out.
    #[tokio::test]
    async fn the_fixed_prefix_stays_within_its_budget() {
        // About 4.6k tokens by Bedrock's count for Claude, down from 6.6k.
        const BUDGET_BYTES: usize = 15_000;
        const TOOL_BUDGET_BYTES: usize = 1_100;
        let parts = fixed_prefix().await;
        let total: usize = parts.iter().map(|(_, bytes)| bytes).sum();
        for (part, bytes) in &parts {
            println!("{bytes:>6}  {part}");
        }
        println!("{total:>6}  total");
        assert!(
            total <= BUDGET_BYTES,
            "the fixed prefix grew to {total} bytes: {parts:#?}"
        );
        let big: Vec<_> = parts
            .iter()
            .filter(|(part, bytes)| part.starts_with("tool ") && *bytes > TOOL_BUDGET_BYTES)
            .collect();
        assert!(
            big.is_empty(),
            "tool specs over {TOOL_BUDGET_BYTES} bytes: {big:?}"
        );
        let names: Vec<&str> = parts
            .iter()
            .filter_map(|(part, _)| part.strip_prefix("tool "))
            .collect();
        assert!(
            !names.contains(&"strands_manage_background_task"),
            "the SDK's background tasks stay off: {names:?}"
        );
        assert!(
            names.contains(&"reload_swarm") && names.contains(&"find_tools"),
            "{names:?}"
        );
    }

    #[test]
    fn lean_schemas_drop_what_tells_a_model_nothing() {
        /// Input doc.
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Input {
            /// Kept.
            text: String,
            #[serde(default)]
            flag: bool,
            #[serde(default)]
            limit: Option<u64>,
            #[serde(default)]
            names: Vec<String>,
        }
        let schema = crate::tools::lean_schema::<Input>();
        assert_eq!(
            schema,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Kept."},
                    "flag": {"type": "boolean"},
                    "limit": {"type": ["integer", "null"], "minimum": 0},
                    "names": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text"],
            })
        );
    }

    #[tokio::test]
    async fn retrieve_context_keeps_its_schema_under_the_shorter_description() {
        let options = HarnessFactoryOptions::default();
        let mut harness = harness_options(&options, spec()).unwrap();
        harness.model = Some(ModelOption::Instance(MockMessageModel::shared(vec![])));
        let agent = build(&options, harness).await.unwrap();
        let tool = agent
            .with_tool_registry(|registry| registry.get(RETRIEVAL_TOOL_NAME))
            .unwrap();
        assert_eq!(tool.description(), RETRIEVAL_DESCRIPTION);
        let schema = tool.spec().input_schema.clone().unwrap();
        assert!(schema["properties"]["line_range"].is_object(), "{schema}");
    }
}
