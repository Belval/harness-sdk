//! The default [`AgentFactory`]: every agent is a [`create_harness`] agent rooted at its own
//! working directory, with the swarm tools in place of the harness's `subagent`.

use std::sync::Arc;

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
use strands_agents::tools::ToolRef;
use strands_agents::Plugin;
use strands_agents::{FileStorage, Result, SessionManager};
use strands_harness::defaults::{DEFAULT_BUILTIN_TOOLS, DEFAULT_MODEL};
use strands_harness::{
    create_harness, resolve_model, BuiltinToolsConfig, Effort, EnvironmentContext, EnvironmentContextConfig,
    HarnessAgentOptions, MemoryOption, ModelOption, ResolveModelOptions, SandboxOption, SessionOption,
    SkillsOption,
};

use crate::catalog::ToolCatalog;
use crate::compaction::{CompactAt, DEFAULT_COMPACT_AT};
use crate::pruning::{DropOldReasoning, TrimOldResults};
use crate::runtime::{AgentFactory, AgentSpec, ModelResolver};
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
}

/// A factory that builds harness agents.
///
/// Sessions are saved only for runs with a state directory; long-term memory is off, since a shared
/// memory directory would interleave writes from concurrent agents.
pub fn harness_factory(options: HarnessFactoryOptions) -> AgentFactory {
    Arc::new(move |spec: AgentSpec| {
        let options = options.clone();
        Box::pin(async move {
            let agent = create_harness(harness_options(&options, spec)?).await?;
            // The harness has restored the session by now; give back the tools enabled before.
            if let Some(catalog) = &options.catalog {
                catalog.restore(&agent);
            }
            Ok(agent)
        })
    })
}

/// The [`ModelResolver`] matching [`harness_factory`]: resolves an id (`None`: the factory's
/// default) exactly as `create_harness` does, with the same effort and caching, so a switched
/// agent runs as if it had been built with that model.
pub fn harness_model_resolver(options: HarnessFactoryOptions) -> ModelResolver {
    Arc::new(move |model: Option<String>| {
        let model = model_option(&options, model);
        let effort = options.effort;
        Box::pin(async move {
            resolve_model(ResolveModelOptions {
                model,
                default: DEFAULT_MODEL.to_owned(),
                effort,
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
/// summarize old turns at `compact_at` tokens (see [`CompactAt`]). Earlier turns' reasoning is
/// dropped first (see [`DropOldReasoning`]) and older large `shell`/`read` results are cut to a
/// preview (see [`TrimOldResults`]). With `stash_dir` the full outputs behind those previews are
/// kept there, so `retrieve_context` still finds them after a restart.
fn context_manager(compact_at: u64, stash_dir: Option<&std::path::Path>) -> ContextManagerOption {
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
    // Ungated (utilization 0) and message-level; `CompactAt` decides when it runs.
    let summarize = Offload::summarize(OffloadTarget::All).when(OffloadConditions {
        utilization: Some(0.0),
        preserve_recent: Some(4.0),
        ..Default::default()
    });
    let summarize = CompactAt::new(Arc::new(summarize), compact_at);
    ContextManagerOption::Config(ContextManagerConfig {
        strategies: Some(vec![
            Arc::new(DropOldReasoning),
            Arc::new(truncate),
            Arc::new(TrimOldResults::default()),
            Arc::new(summarize),
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
        plugins.push(Arc::new(GoalLoop::goal(goal.clone()).with_judge(judge)));
    }
    let mut sandbox = LocalSandbox::with_base_dir(spec.cwd);
    if !spec.env.is_empty() {
        sandbox = sandbox.with_env(spec.env.into_iter().collect());
    }
    let mut tools = spec.tools;
    tools.extend(options.tools.iter().cloned());
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
        instructions: Some(instructions),
        tools,
        plugins,
        builtin_tools: Some(BuiltinToolsConfig::list(builtins)),
        sandbox: SandboxOption::Instance(Arc::new(sandbox)),
        session,
        memory: MemoryOption::Off,
        // Discovery happens once at startup (crate::skills); the harness's own `./.agent/skills`
        // lookup, relative to the process rather than the agent, stays off.
        skills: SkillsOption::Off,
        context_manager: context_manager(
            options.compact_at.unwrap_or(DEFAULT_COMPACT_AT),
            stash_dir.as_deref(),
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
        let manager = ContextManager::from(context_manager(DEFAULT_COMPACT_AT, None)).unwrap();
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
        let manager = ContextManager::from(context_manager(4_000, None)).unwrap();

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
    async fn stashed_tool_output_outlives_the_process_with_a_stash_dir() {
        use strands_agents::Plugin;
        let dir = tempfile::tempdir().unwrap();
        let stash_of = |agent: Agent| {
            let dir = dir.path().to_owned();
            async move {
                let manager = ContextManager::from(context_manager(DEFAULT_COMPACT_AT, Some(&dir))).unwrap();
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
        assert_eq!(tool_names(Some(Arc::new(full))), ["find_tools"]);
        let empty = ToolCatalog::new(Vec::new(), vec![("srv".into(), 0)]);
        assert!(tool_names(Some(Arc::new(empty))).is_empty());
        assert!(tool_names(None).is_empty());
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
}
