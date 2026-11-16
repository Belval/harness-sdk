//! Lazy MCP tool discovery. Agents start without MCP tools; `find_tools` searches the shared
//! catalog and enables tools for the calling agent at runtime, so only the tools an agent uses are
//! sent to the model on every call.

use std::sync::Arc;

use serde::Deserialize;
use strands_agents::tools::{FunctionTool, ToolContext, ToolRef};
use strands_agents::{Agent, Result};

/// Name of the discovery tool.
pub const FIND_TOOLS_NAME: &str = "find_tools";

/// Agent-state key listing the MCP tools an agent enabled; the session saves it, so a resumed
/// agent gets its tools back.
pub const ENABLED_STATE_KEY: &str = "swarm_mcp_enabled";

const DEFAULT_LIMIT: usize = 15;
const MAX_LIMIT: usize = 50;
const SUMMARY_CHARS: usize = 160;

/// The MCP tools agents can enable, and how many each server offers.
pub struct ToolCatalog {
    tools: Vec<ToolRef>,
    servers: Vec<(String, usize)>,
}

impl ToolCatalog {
    /// A catalog of `tools`; `servers` lists `(server name, tool count)` for the tool description.
    pub fn new(tools: Vec<ToolRef>, servers: Vec<(String, usize)>) -> Self {
        Self { tools, servers }
    }

    /// Whether the catalog has no tools (then `find_tools` is not worth registering).
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// The tool named `name`.
    pub fn get(&self, name: &str) -> Option<&ToolRef> {
        self.tools.iter().find(|tool| tool.name() == name)
    }

    /// Tools matching the keywords in `query`, best first: most keywords matched, then name hits
    /// over description hits, then catalog order. Matching is case-insensitive substring search.
    pub fn search(&self, query: &str) -> Vec<&ToolRef> {
        let terms = keywords(query);
        if terms.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(usize, usize, usize, &ToolRef)> = self
            .tools
            .iter()
            .enumerate()
            .filter_map(|(index, tool)| {
                let name = tool.name().to_lowercase();
                let description = tool.description().to_lowercase();
                let (mut matched, mut score) = (0, 0);
                for term in &terms {
                    if name.contains(term.as_str()) {
                        (matched, score) = (matched + 1, score + 3);
                    } else if description.contains(term.as_str()) {
                        (matched, score) = (matched + 1, score + 1);
                    }
                }
                (matched > 0).then_some((matched, score, index, tool))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
        scored.into_iter().map(|(.., tool)| tool).collect()
    }

    /// The `find_tools` tool over this catalog.
    pub fn find_tool(self: &Arc<Self>) -> ToolRef {
        let catalog = self.clone();
        let tool = FunctionTool::builder(FIND_TOOLS_NAME)
            .description(self.description())
            .typed_handler(move |input: FindInput, ctx: ToolContext| {
                let catalog = catalog.clone();
                async move { Result::Ok(catalog.run(&ctx.agent, input)) }
            });
        Arc::new(tool)
    }

    /// Re-enable the tools `agent` enabled before (as recorded in its state), e.g. after its
    /// session was restored. Names no longer in the catalog are skipped.
    pub fn restore(&self, agent: &Agent) {
        let names: Vec<String> = agent.state().get_as(ENABLED_STATE_KEY).unwrap_or_default();
        let (tools, _) = self.resolve(&names);
        if !tools.is_empty() {
            add_tools(agent, tools);
        }
    }

    fn description(&self) -> String {
        let servers: Vec<String> = self
            .servers
            .iter()
            .filter(|(_, count)| *count > 0)
            .map(|(name, count)| format!("{name} ({count})"))
            .collect();
        format!(
            "Search and enable MCP tools. {} MCP tools are available but not loaded, from these servers: {}. \
             Pass `query` (keywords) to search tool names and descriptions; pass `enable` (exact tool names) \
             to load them for yourself: they become callable from your next step on and stay enabled across \
             restarts. Tool names are `<server>_<tool>`.",
            self.tools.len(),
            servers.join(", ")
        )
    }

    fn run(&self, agent: &Agent, input: FindInput) -> String {
        let mut out = Vec::new();
        if !input.enable.is_empty() {
            let (tools, unknown) = self.resolve(&input.enable);
            if !tools.is_empty() {
                let names: Vec<String> = tools.iter().map(|tool| tool.name().to_owned()).collect();
                add_tools(agent, tools);
                out.push(format!("Enabled: {}.", names.join(", ")));
            }
            if !unknown.is_empty() {
                out.push(format!(
                    "Unknown (search for the exact name): {}.",
                    unknown.join(", ")
                ));
            }
        }
        let query = input.query.unwrap_or_default();
        if !query.trim().is_empty() {
            out.push(self.render_search(agent, &query, input.limit));
        }
        if out.is_empty() {
            out.push("Pass `query` to search or `enable` to load tools.".to_owned());
        }
        out.join("\n")
    }

    fn render_search(&self, agent: &Agent, query: &str, limit: Option<usize>) -> String {
        let limit = limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
        let hits = self.search(query);
        if hits.is_empty() {
            return format!("No MCP tools match \"{query}\".");
        }
        let enabled: Vec<String> = agent.tools().iter().map(|t| t.name().to_owned()).collect();
        let mut lines: Vec<String> = hits
            .iter()
            .take(limit)
            .map(|tool| {
                let mark = match enabled.iter().any(|name| name == tool.name()) {
                    true => " (enabled)",
                    false => "",
                };
                format!("- {}{mark}: {}", tool.name(), summary(tool.description()))
            })
            .collect();
        if hits.len() > limit {
            lines.push(format!(
                "({} more matches; refine the query or raise `limit`.)",
                hits.len() - limit
            ));
        }
        lines.join("\n")
    }

    /// Catalog tools for `names` (deduplicated), and the names not in the catalog.
    fn resolve(&self, names: &[String]) -> (Vec<ToolRef>, Vec<String>) {
        let mut tools: Vec<ToolRef> = Vec::new();
        let mut unknown = Vec::new();
        for name in names {
            match self.get(name) {
                Some(tool) if !tools.iter().any(|t| t.name() == name) => tools.push(tool.clone()),
                Some(_) => {}
                None => unknown.push(name.clone()),
            }
        }
        (tools, unknown)
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct FindInput {
    /// Keywords to search for in tool names and descriptions, e.g. "slack search messages".
    #[serde(default)]
    query: Option<String>,
    /// Exact tool names to enable for yourself.
    #[serde(default)]
    enable: Vec<String>,
    /// Maximum search results (default 15, max 50).
    #[serde(default)]
    limit: Option<usize>,
}

/// Register `tools` on `agent` and record them in its state for resume.
fn add_tools(agent: &Agent, tools: Vec<ToolRef>) {
    let mut names: Vec<String> = agent.state().get_as(ENABLED_STATE_KEY).unwrap_or_default();
    for tool in &tools {
        if !names.iter().any(|name| name == tool.name()) {
            names.push(tool.name().to_owned());
        }
    }
    // Catalog names are unique, so replacing only ever refreshes the same tool.
    let _ = agent.with_tool_registry(|registry| registry.add_or_replace(tools));
    agent.state().set(ENABLED_STATE_KEY, serde_json::json!(names));
}

fn keywords(query: &str) -> Vec<String> {
    let mut terms: Vec<String> = Vec::new();
    for term in query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(str::to_lowercase)
    {
        if !terms.contains(&term) {
            terms.push(term);
        }
    }
    terms
}

/// First line, cut at the first sentence end and at [`SUMMARY_CHARS`].
fn summary(description: &str) -> String {
    let line = description.trim().lines().next().unwrap_or_default();
    let sentence = match line.find(". ") {
        Some(end) => &line[..=end],
        None => line,
    };
    match sentence.char_indices().nth(SUMMARY_CHARS) {
        Some((cut, _)) => format!("{}…", &sentence[..cut]),
        None => sentence.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use strands_agents::testing::{MockMessageModel, MockResponse};
    use strands_agents::types::{ContentBlock, Message, Role, ToolResultStatus, ToolUseBlock};

    fn tool(name: &str, description: &str) -> ToolRef {
        let reply = format!("{name} ran");
        Arc::new(FunctionTool::builder(name).description(description).handler(
            move |_input: serde_json::Value, _ctx: ToolContext| {
                let reply = reply.clone();
                async move { Result::Ok(reply) }
            },
        ))
    }

    fn catalog() -> Arc<ToolCatalog> {
        Arc::new(ToolCatalog::new(
            vec![
                tool("slack_search", "Search Slack messages. Supports modifiers."),
                tool("slack_post_message", "Post a message to a channel."),
                tool("wiki_read", "Read a wiki page; can search inside it."),
                tool("tickets_get", "Get a ticket by id."),
            ],
            vec![("slack".into(), 2), ("wiki".into(), 1), ("tickets".into(), 1)],
        ))
    }

    fn tool_call(name: &str, id: &str, input: serde_json::Value) -> MockResponse {
        MockResponse::message(Message::new(
            Role::Assistant,
            vec![ContentBlock::ToolUse(ToolUseBlock::new(name, id, input))],
        ))
    }

    fn agent(model: Arc<MockMessageModel>, catalog: &Arc<ToolCatalog>) -> Agent {
        Agent::builder()
            .model(model)
            .tools(vec![catalog.find_tool()])
            .no_sandbox()
            .build()
            .unwrap()
    }

    /// Registered tool names that come from `catalog()` or are `find_tools` (the SDK adds its own).
    fn names(agent: &Agent) -> Vec<String> {
        let specs: Vec<_> = agent.tools().iter().map(|t| t.spec().clone()).collect();
        ours(&specs)
    }

    fn ours(specs: &[strands_agents::types::ToolSpec]) -> Vec<String> {
        let catalog = catalog();
        specs
            .iter()
            .map(|spec| spec.name.clone())
            .filter(|name| name == FIND_TOOLS_NAME || catalog.get(name).is_some())
            .collect()
    }

    #[test]
    fn search_ranks_more_keywords_and_name_hits_first() {
        let catalog = catalog();
        let hits: Vec<&str> = catalog.search("Slack search").iter().map(|t| t.name()).collect();
        // Both keywords in the name, then one name hit, then a description-only hit.
        assert_eq!(hits, ["slack_search", "slack_post_message", "wiki_read"]);
        assert!(catalog.search("  ").is_empty());
        assert!(catalog.search("calendar").is_empty());
    }

    #[test]
    fn summary_keeps_the_first_sentence() {
        assert_eq!(summary("Search Slack. More.\nDetails"), "Search Slack.");
        assert_eq!(summary(&"x".repeat(200)).chars().count(), SUMMARY_CHARS + 1);
    }

    #[test]
    fn description_lists_servers_with_counts() {
        let description = catalog().description();
        assert!(description.contains("4 MCP tools"), "{description}");
        assert!(description.contains("slack (2), wiki (1), tickets (1)"));
    }

    #[tokio::test]
    async fn search_marks_enabled_tools_and_reports_unknown_names() {
        let catalog = catalog();
        let agent = agent(MockMessageModel::shared(vec![]), &catalog);
        let text = catalog.run(
            &agent,
            FindInput {
                query: Some("slack".into()),
                enable: vec!["slack_search".into(), "nope".into()],
                limit: Some(1),
            },
        );
        assert!(text.contains("Enabled: slack_search."), "{text}");
        assert!(text.contains("Unknown (search for the exact name): nope."));
        assert!(text.contains("- slack_search (enabled): Search Slack messages."));
        assert!(text.contains("(1 more matches"));
    }

    #[tokio::test]
    async fn enabled_tool_is_sent_and_callable_on_the_next_model_call() {
        let catalog = catalog();
        let model = MockMessageModel::shared(vec![
            tool_call("find_tools", "t1", json!({ "enable": ["slack_search"] })),
            tool_call("slack_search", "t2", json!({})),
            MockResponse::message(Message::assistant("done")),
        ]);
        let agent = agent(model.clone(), &catalog);
        agent.invoke("go").await.unwrap();

        let calls = model.calls();
        let specs = |i: usize| ours(&calls[i].1.tool_specs);
        assert_eq!(specs(0), ["find_tools"]);
        assert_eq!(specs(1), ["find_tools", "slack_search"]);
        let result = calls[2]
            .0
            .iter()
            .flat_map(|m| m.content.iter())
            .find_map(|block| match block {
                ContentBlock::ToolResult(result) if result.tool_use_id == "t2" => Some(result.clone()),
                _ => None,
            })
            .expect("slack_search result");
        assert_eq!(result.status, ToolResultStatus::Success);
        assert_eq!(
            agent.state().get(ENABLED_STATE_KEY),
            Some(json!(["slack_search"]))
        );
    }

    #[tokio::test]
    async fn resumed_session_gets_its_enabled_tools_back() {
        let dir = tempfile::tempdir().unwrap();
        let catalog = catalog();
        let session = |model: Arc<MockMessageModel>| {
            Agent::builder()
                .model(model)
                .tools(vec![catalog.find_tool()])
                .no_sandbox()
                .session_manager(crate::agent_session_manager(dir.path(), "a1").unwrap())
                .build()
                .unwrap()
        };
        let first = session(MockMessageModel::shared(vec![
            tool_call(
                "find_tools",
                "t1",
                json!({ "enable": ["wiki_read", "tickets_get"] }),
            ),
            MockResponse::message(Message::assistant("done")),
        ]));
        first.invoke("go").await.unwrap();
        drop(first);

        let resumed = session(MockMessageModel::shared(vec![]));
        resumed.initialize().await.unwrap();
        assert_eq!(names(&resumed), ["find_tools"]);
        catalog.restore(&resumed);
        assert_eq!(names(&resumed), ["find_tools", "wiki_read", "tickets_get"]);
    }
}
