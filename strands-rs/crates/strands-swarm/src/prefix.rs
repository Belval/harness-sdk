//! The fixed prefix: the system prompt and tool specs a root agent sends at the start of every
//! call, built the way a real run builds them (swarm prompt and tools, memory, an MCP catalog, 20
//! skills) on a stub model, in a throwaway directory. Providers cache tools, then the system
//! prompt, then messages, so a binary whose prefix differs makes each agent's next call rewrite
//! its whole cache: on a live run of ~80 busy agents, ~12M tokens and ~$70 per reload.
//! `--print-prefix` prints it; `reload_swarm` compares the running binary's with the new one's.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use strands_agents::models::{ContentBlockDelta, Model, ModelEventStream, ModelStreamEvent, StreamOptions};
use strands_agents::plugins::vended::Skill;
use strands_agents::tools::{FunctionTool, ToolRef};
use strands_agents::types::{Message, Role, StopReason};
use strands_agents::{Error, Result};
use strands_harness::ModelOption;

use crate::catalog::ToolCatalog;
use crate::factory::{build, harness_options, HarnessFactoryOptions};
use crate::runtime::{AgentFactory, AgentSpec, SpawnRequest, Swarm, SwarmConfig, USER};

/// Stands in for the throwaway directory in the prompt, so two builds compare equal.
const PROJECT: &str = "/home/user/project";

/// A root agent's system prompt and tool specs, in the order sent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prefix {
    /// The system prompt's text.
    pub system: String,
    /// `(name, spec as JSON)` per tool.
    pub tools: Vec<(String, String)>,
}

impl Prefix {
    /// The system prompt split at its `# ` headings into `(title, text)`, the text before the
    /// first one titled `(preamble)`.
    pub fn sections(&self) -> Vec<(String, String)> {
        let mut sections = Vec::new();
        let mut title = String::from("(preamble)");
        let mut text = String::new();
        for line in self.system.split_inclusive('\n') {
            if let Some(next) = line.strip_prefix("# ") {
                sections.push((
                    std::mem::replace(&mut title, next.trim().to_owned()),
                    std::mem::take(&mut text),
                ));
            }
            text.push_str(line);
        }
        sections.push((title, text));
        sections
    }

    /// What differs in `new`, e.g. `["tool send_message", "new tool land", "prompt section Swarm"]`;
    /// empty when the two are the same.
    pub fn changes(&self, new: &Prefix) -> Vec<String> {
        let mut changes = Vec::new();
        let find = |tools: &[(String, String)], name: &str| {
            tools
                .iter()
                .find(|(other, _)| other == name)
                .map(|(_, spec)| spec.clone())
        };
        for (name, spec) in &new.tools {
            match find(&self.tools, name) {
                None => changes.push(format!("new tool {name}")),
                Some(old) if old != *spec => changes.push(format!("tool {name}")),
                Some(_) => {}
            }
        }
        for (name, _) in &self.tools {
            if find(&new.tools, name).is_none() {
                changes.push(format!("tool {name} removed"));
            }
        }
        let names =
            |tools: &[(String, String)]| tools.iter().map(|(name, _)| name.clone()).collect::<Vec<_>>();
        if changes.is_empty() && names(&self.tools) != names(&new.tools) {
            changes.push("tool order".to_owned());
        }
        let (old_sections, new_sections) = (self.sections(), new.sections());
        for section in &new_sections {
            if !old_sections.contains(section) {
                changes.push(format!("prompt section {}", section.0));
            }
        }
        for (title, _) in &old_sections {
            if !new_sections.iter().any(|(other, _)| other == title) {
                changes.push(format!("prompt section {title} removed"));
            }
        }
        if self.system != new.system && !changes.iter().any(|change| change.starts_with("prompt")) {
            changes.push("system prompt".to_owned());
        }
        changes
    }
}

/// The prefix of this binary's root agent; `allow_land` adds the `land` tool, as `--allow-land`
/// does.
pub async fn canonical(allow_land: bool) -> Result<Prefix> {
    let dir = std::env::temp_dir().join(format!("strands-swarm-prefix-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).map_err(|e| Error::other(format!("{}: {e}", dir.display())))?;
    let prefix = canonical_in(&dir, allow_land).await;
    let _ = std::fs::remove_dir_all(&dir);
    prefix
}

async fn canonical_in(dir: &Path, allow_land: bool) -> Result<Prefix> {
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
    // Shaped like real plugin skills: a long first sentence, then when to use it.
    let description = "Diagnose failed, stuck or blocked deployments of a service pipeline, \
                       including infrastructure stack failures and rollbacks. Use when a \
                       deployment failed, a promotion is blocked or an alarm rolled it back.";
    let skills: Vec<Skill> = (0..20)
        .map(|n| {
            let name = format!("skill-{n}");
            let mut skill = Skill::new(&name, description, "Steps.");
            skill.path = Some(dir.join(name));
            skill
        })
        .collect();
    let options = HarnessFactoryOptions {
        catalog: Some(Arc::new(catalog)),
        skills: Arc::new(skills),
        ..Default::default()
    };
    let seen: Arc<Mutex<Option<StreamOptions>>> = Arc::default();
    let model = Arc::new(Capture(Arc::clone(&seen)));
    let factory: AgentFactory = Arc::new(move |spec: AgentSpec| {
        let (options, model) = (options.clone(), Arc::clone(&model));
        Box::pin(async move {
            let mut harness = harness_options(&options, spec)?;
            harness.model = Some(ModelOption::Instance(model));
            build(&options, harness).await
        })
    });
    let mut config = SwarmConfig::new(factory, dir);
    config.memory = Some(Arc::new(crate::Memory::new(dir.join("memory.md"))));
    config.state_dir = Some(dir.join("run"));
    config.allow_land = allow_land;
    config.reclaim_disk = false;
    let (swarm, _events) = Swarm::new(config);
    let request = SpawnRequest {
        task: "go".into(),
        ..Default::default()
    };
    swarm.spawn(USER, request).await?;
    let sent = async {
        loop {
            if let Some(options) = seen.lock().take() {
                return options;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    let sent = tokio::time::timeout(Duration::from_secs(30), sent).await;
    swarm.shutdown().await;
    let sent = sent.map_err(|_| Error::other("the stub agent made no model call within 30s"))?;
    let system = sent
        .system_prompt
        .as_ref()
        .map(|prompt| prompt.text())
        .unwrap_or_default();
    let tools = sent
        .tool_specs
        .iter()
        .map(|spec| Ok((spec.name.clone(), serde_json::to_string(spec)?)))
        .collect::<std::result::Result<_, serde_json::Error>>()
        .map_err(|e| Error::other(e.to_string()))?;
    Ok(Prefix {
        system: system.replace(&dir.display().to_string(), PROJECT),
        tools,
    })
}

/// Keeps the first call's options and answers `ok`.
struct Capture(Arc<Mutex<Option<StreamOptions>>>);

impl Model for Capture {
    fn model_id(&self) -> Option<String> {
        Some("prefix-stub".into())
    }

    fn stream<'a>(&'a self, _messages: &'a [Message], options: &'a StreamOptions) -> ModelEventStream<'a> {
        self.0.lock().get_or_insert_with(|| options.clone());
        let events = [
            ModelStreamEvent::MessageStart {
                role: Role::Assistant,
            },
            ModelStreamEvent::ContentBlockStart { start: None },
            ModelStreamEvent::ContentBlockDelta {
                delta: ContentBlockDelta::Text { text: "ok".into() },
            },
            ModelStreamEvent::ContentBlockStop {},
            ModelStreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
                additional_model_response_fields: None,
            },
        ];
        Box::pin(futures::stream::iter(events.into_iter().map(Ok)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefix(system: &str, tools: &[(&str, &str)]) -> Prefix {
        Prefix {
            system: system.into(),
            tools: tools
                .iter()
                .map(|(n, s)| (n.to_string(), s.to_string()))
                .collect(),
        }
    }

    #[test]
    fn changes_name_the_tools_and_prompt_sections_that_differ() {
        let old = prefix(
            "intro\n# Swarm\nyou are a\n# Files\nread\n",
            &[("a", "1"), ("b", "2"), ("c", "3")],
        );
        assert!(old.changes(&old.clone()).is_empty());
        let new = prefix(
            "intro\n# Swarm\nyou are b\n# Files\nread\n",
            &[("a", "1"), ("b", "9"), ("d", "4")],
        );
        assert_eq!(
            old.changes(&new),
            ["tool b", "new tool d", "tool c removed", "prompt section Swarm"]
        );
        let reordered = prefix(&old.system, &[("b", "2"), ("a", "1"), ("c", "3")]);
        assert_eq!(old.changes(&reordered), ["tool order"]);
        let preamble = prefix(
            "intro!\n# Swarm\nyou are a\n# Files\nread\n",
            &[("a", "1"), ("b", "2"), ("c", "3")],
        );
        assert_eq!(old.changes(&preamble), ["prompt section (preamble)"]);
        let dropped = prefix(
            "intro\n# Swarm\nyou are a\n",
            &[("a", "1"), ("b", "2"), ("c", "3")],
        );
        assert_eq!(old.changes(&dropped), ["prompt section Files removed"]);
        let swapped = prefix(
            "intro\n# Files\nread\n# Swarm\nyou are a\n",
            &[("a", "1"), ("b", "2"), ("c", "3")],
        );
        assert_eq!(old.changes(&swapped), ["system prompt"]);
    }

    #[tokio::test]
    async fn the_prefix_is_the_same_on_every_build_and_land_adds_its_tool() {
        let (first, second) = (canonical(false).await.unwrap(), canonical(false).await.unwrap());
        assert!(first.changes(&second).is_empty(), "{:?}", first.changes(&second));
        assert_eq!(first, second);
        assert!(
            !first.system.contains("strands-swarm-prefix-"),
            "the temp dir is replaced"
        );
        let names: Vec<&str> = first.tools.iter().map(|(name, _)| name.as_str()).collect();
        assert!(
            names.contains(&"reload_swarm") && !names.contains(&"land"),
            "{names:?}"
        );
        let landing = canonical(true).await.unwrap();
        assert_eq!(first.changes(&landing), ["new tool land"]);
    }
}
