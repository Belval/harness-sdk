//! MCP servers shared by the whole swarm: connected once, their tools handed to every agent.

use std::path::Path;

use regex::Regex;
use strands_agents::mcp::{ListOptions, LoadOptions, McpClient};
use strands_agents::tools::ToolRef;
use strands_agents::{Error, Result};

/// Connected MCP servers and the tools that survived the deny filter. Keep this alive for as long
/// as agents use the tools.
pub struct SharedMcp {
    /// The connected clients.
    pub clients: Vec<McpClient>,
    /// Tools offered to agents, named `<server>_<tool>`.
    pub tools: Vec<ToolRef>,
    /// Names removed by the deny filter.
    pub denied: Vec<String>,
}

/// Connect every server in each `mcp.json`-shaped file (a Claude Code `~/.claude.json` works too:
/// only its `mcpServers` key is read). Servers that fail to connect are skipped with a warning.
/// Tools whose prefixed name matches any `deny` pattern are dropped.
pub async fn load_shared_mcp(configs: &[impl AsRef<Path>], deny: &[String]) -> Result<SharedMcp> {
    let deny = deny
        .iter()
        .map(|pattern| {
            Regex::new(pattern).map_err(|e| Error::other(format!("Invalid --mcp-deny {pattern}: {e}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut shared = SharedMcp {
        clients: Vec::new(),
        tools: Vec::new(),
        denied: Vec::new(),
    };
    for config in configs {
        let mut load = LoadOptions::default();
        load.continue_on_error = true;
        let mut list = ListOptions::default();
        list.prefix_with_server_name = true;
        let clients = McpClient::load_servers(config.as_ref(), load, list).await?;
        for client in clients {
            add_client_tools(&mut shared, client, &deny).await;
        }
    }
    Ok(shared)
}

async fn add_client_tools(shared: &mut SharedMcp, client: McpClient, deny: &[Regex]) {
    match client.list_tools().await {
        Ok(tools) => {
            for tool in tools {
                match deny.iter().any(|pattern| pattern.is_match(tool.name())) {
                    true => shared.denied.push(tool.name().to_owned()),
                    false => shared.tools.push(tool),
                }
            }
        }
        Err(error) => {
            tracing::warn!(server = %client.name(), %error, "skipping MCP server | listing tools failed")
        }
    }
    shared.clients.push(client);
}
