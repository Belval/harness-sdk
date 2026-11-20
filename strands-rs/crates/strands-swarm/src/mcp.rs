//! MCP servers shared by the whole swarm. A server connects the first time an agent enables or
//! calls one of its tools (or up front with `connect`), not at startup: its tool list comes from
//! a per-run cache written the last time it was connected, keyed by the server's config. A
//! server that died is reconnected on the next call.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use strands_agents::mcp::{
    ConfigSource, IndexMap, ListOptions, LoadOptions, McpClient, McpConnectionState, McpServerConfig,
};
use strands_agents::tools::{create_error_result, Tool, ToolContext, ToolRef};
use strands_agents::types::{ToolResultBlock, ToolSpec};
use strands_agents::{Error, Result};

use crate::catalog::ToolCatalog;

/// Longest a server may take to start and list its tools.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

/// Name of the tool cache in a run directory.
pub const MCP_CACHE_FILE: &str = "mcp-tools.json";

/// The swarm's MCP servers and the tools that survived the deny filter. Keep this alive for as
/// long as agents use the tools.
pub struct SharedMcp {
    /// Every configured server, connected or not.
    pub servers: Vec<Arc<McpServer>>,
    /// Tools offered to agents, named `<server>_<tool>`; each connects its server on first call.
    pub tools: Vec<ToolRef>,
    /// Names removed by the deny filter.
    pub denied: Vec<String>,
}

impl SharedMcp {
    /// The tools as a catalog agents enable from on demand with `find_tools`; enabling a tool
    /// starts its server in the background, so the first call does not wait for it.
    pub fn catalog(&self) -> Arc<ToolCatalog> {
        let owners: HashMap<String, Arc<McpServer>> = self
            .servers
            .iter()
            .flat_map(|server| server.tool_names().into_iter().map(|name| (name, server.clone())))
            .collect();
        let warm = move |names: &[String]| {
            let mut started: Vec<&str> = Vec::new();
            for server in names.iter().filter_map(|name| owners.get(name)) {
                if !started.contains(&server.name()) {
                    started.push(server.name());
                    let server = server.clone();
                    tokio::spawn(async move { server.warm().await });
                }
            }
        };
        let servers = self
            .servers
            .iter()
            .map(|s| (s.name().to_owned(), s.offered))
            .collect();
        Arc::new(ToolCatalog::new(self.tools.clone(), servers).with_on_enable(Arc::new(warm)))
    }

    /// How many servers are connected right now.
    pub async fn connected(&self) -> usize {
        let mut count = 0;
        for server in &self.servers {
            count += usize::from(server.is_connected().await);
        }
        count
    }
}

/// One configured MCP server.
pub struct McpServer {
    name: String,
    config: McpServerConfig,
    stderr_dir: Option<PathBuf>,
    cache: Option<Arc<ToolCache>>,
    /// Tool specs as last listed (cached or live), before the deny filter.
    specs: Vec<ToolSpec>,
    /// How many of its tools survived the deny filter.
    offered: usize,
    live: tokio::sync::Mutex<Option<Live>>,
}

struct Live {
    client: McpClient,
    /// In the order the server listed them.
    tools: Vec<ToolRef>,
}

impl McpServer {
    /// The server's name (its `mcp.json` key).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether the server is connected and its transport is still open.
    pub async fn is_connected(&self) -> bool {
        match &*self.live.lock().await {
            Some(live) => live.client.is_alive().await,
            None => false,
        }
    }

    fn tool_names(&self) -> Vec<String> {
        self.specs.iter().map(|spec| spec.name.clone()).collect()
    }

    /// Connect now unless connected; a failure is only logged (the next call retries).
    async fn warm(&self) {
        if let Err(error) = self.live_tool(None).await {
            tracing::warn!(server = %self.name, %error, "MCP server failed to start");
        }
    }

    /// The live tool `name` (`None`: just make sure the server is up), connecting or
    /// reconnecting the server first when needed.
    async fn live_tool(&self, name: Option<&str>) -> Result<Option<ToolRef>> {
        let mut live = self.live.lock().await;
        let alive = match &*live {
            Some(current) => current.client.is_alive().await,
            None => false,
        };
        if !alive {
            if let Some(dead) = live.take() {
                tracing::info!(server = %self.name, "MCP server is gone | reconnecting");
                let _ = dead.client.disconnect().await;
            }
            *live = Some(self.connect().await?);
        }
        let Some(name) = name else {
            return Ok(None);
        };
        let tools = &live.as_ref().expect("connected above").tools;
        let tool = tools.iter().find(|tool| tool.name() == name);
        tool.cloned()
            .map(Some)
            .ok_or_else(|| Error::other(format!("MCP server \"{}\" no longer offers {name}.", self.name)))
    }

    async fn connect(&self) -> Result<Live> {
        let (client, tools) = tokio::time::timeout(
            CONNECT_TIMEOUT,
            connect_server(&self.name, &self.config, self.stderr_dir.as_deref()),
        )
        .await
        .map_err(|_| {
            Error::other(format!(
                "MCP server \"{}\" did not start within {}s.",
                self.name,
                CONNECT_TIMEOUT.as_secs()
            ))
        })??;
        if let Some(cache) = &self.cache {
            let specs = tools.iter().map(|tool| tool.spec().clone()).collect();
            cache.store(&self.name, &self.config, specs).await;
        }
        Ok(Live { client, tools })
    }
}

/// Connect one server and list its tools.
async fn connect_server(
    name: &str,
    config: &McpServerConfig,
    stderr_dir: Option<&Path>,
) -> Result<(McpClient, Vec<ToolRef>)> {
    let mut load = LoadOptions::default();
    load.stderr_dir = stderr_dir.map(Path::to_path_buf);
    let servers: IndexMap<String, McpServerConfig> = IndexMap::from([(name.to_owned(), config.clone())]);
    let client = McpClient::load_servers(ConfigSource::Inline(servers), load, ListOptions::default())
        .await?
        .pop()
        .ok_or_else(|| Error::other(format!("MCP server \"{name}\" is disabled.")))?;
    if client.connection_state() == McpConnectionState::Failed {
        return Err(Error::other(format!("MCP server \"{name}\" failed to connect.")));
    }
    let tools = client.list_tools().await?;
    Ok((client, tools))
}

/// A server's tool, offered from its (possibly cached) spec; calls go to the live tool.
struct LazyMcpTool {
    spec: ToolSpec,
    server: Arc<McpServer>,
}

#[async_trait]
impl Tool for LazyMcpTool {
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
        match self.server.live_tool(Some(&self.spec.name)).await {
            Ok(Some(tool)) => {
                let mut result = tool.invoke(ctx).await?;
                crate::text::defang_content(&mut result.content);
                Ok(result)
            }
            Ok(None) => unreachable!("a named lookup returns the tool"),
            Err(error) => Ok(create_error_result(error, &ctx.tool_use.tool_use_id)),
        }
    }
}

/// `<run>/mcp-tools.json`: each server's tool specs as last listed, with a hash of the config
/// they were listed under (an entry whose config changed is ignored). A hash, not the config, so
/// secrets in `env` or headers stay out of the file.
struct ToolCache {
    path: PathBuf,
    entries: tokio::sync::Mutex<BTreeMap<String, CacheEntry>>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct CacheEntry {
    config: String,
    tools: Vec<ToolSpec>,
}

/// FNV-1a over the config's JSON: stable across builds, unlike `std`'s hasher.
fn config_hash(config: &McpServerConfig) -> Option<String> {
    let json = serde_json::to_vec(config).ok()?;
    let hash = json.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    Some(format!("{hash:016x}"))
}

impl ToolCache {
    async fn open(path: PathBuf) -> Self {
        let entries = match tokio::fs::read_to_string(&path).await {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|error| {
                tracing::warn!(path = %path.display(), %error, "ignoring unreadable MCP tool cache");
                BTreeMap::new()
            }),
            Err(_) => BTreeMap::new(),
        };
        Self {
            path,
            entries: tokio::sync::Mutex::new(entries),
        }
    }

    async fn get(&self, name: &str, config: &McpServerConfig) -> Option<Vec<ToolSpec>> {
        let key = config_hash(config)?;
        let entries = self.entries.lock().await;
        entries
            .get(name)
            .filter(|entry| entry.config == key)
            .map(|entry| entry.tools.clone())
    }

    async fn store(&self, name: &str, config: &McpServerConfig, tools: Vec<ToolSpec>) {
        let Some(config) = config_hash(config) else {
            return;
        };
        let mut entries = self.entries.lock().await;
        let entry = CacheEntry { config, tools };
        // Each process connects its servers again: most listings repeat the cached one.
        if entries.get(name) == Some(&entry) {
            return;
        }
        entries.insert(name.to_owned(), entry);
        let text = serde_json::to_vec_pretty(&*entries).unwrap_or_default();
        if let Err(error) = crate::files::replace_async(&self.path, text, false).await {
            tracing::warn!(path = %self.path.display(), %error, "cannot write the MCP tool cache");
        }
    }
}

/// How [`load_shared_mcp`] finds the servers' tools.
#[derive(Debug, Clone, Default)]
pub struct McpOptions {
    /// Regexes over `<server>_<tool>` names to hide from agents.
    pub deny: Vec<String>,
    /// Each stdio server's stderr goes to `<dir>/mcp-<server>.log` instead of ours.
    pub stderr_dir: Option<PathBuf>,
    /// Tool cache file (normally `<run>/mcp-tools.json`); `None` connects every server up front.
    pub cache: Option<PathBuf>,
    /// Connect every server up front even when its tools are cached.
    pub connect: bool,
}

/// Read every server in each `mcp.json`-shaped file (a Claude Code `~/.claude.json` works too:
/// only its `mcpServers` key is read). A server whose tools are cached for its current config
/// stays unconnected until first use; any other one is connected now to list its tools, and
/// skipped with a warning when that fails. Tools whose prefixed name matches a `deny` pattern
/// are dropped.
pub async fn load_shared_mcp(configs: &[impl AsRef<Path>], options: McpOptions) -> Result<SharedMcp> {
    let deny = options
        .deny
        .iter()
        .map(|pattern| {
            Regex::new(pattern).map_err(|e| Error::other(format!("Invalid --mcp-deny {pattern}: {e}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let cache = match &options.cache {
        Some(path) => Some(Arc::new(ToolCache::open(path.clone()).await)),
        None => None,
    };
    let mut shared = SharedMcp {
        servers: Vec::new(),
        tools: Vec::new(),
        denied: Vec::new(),
    };
    for config in configs {
        let servers = ConfigSource::Path(config.as_ref().to_path_buf()).load().await?;
        for (name, config) in servers {
            if config.disabled {
                continue;
            }
            let config = normalized(&name, config);
            let cached = match (&cache, options.connect) {
                (Some(cache), false) => cache.get(&name, &config).await,
                _ => None,
            };
            let mut server = McpServer {
                name: name.clone(),
                config,
                stderr_dir: options.stderr_dir.clone(),
                cache: cache.clone(),
                specs: Vec::new(),
                offered: 0,
                live: tokio::sync::Mutex::new(None),
            };
            server.specs = match cached {
                Some(specs) => specs,
                None => match server.connect().await {
                    Ok(live) => {
                        let specs = live.tools.iter().map(|tool| tool.spec().clone()).collect();
                        *server.live.get_mut() = Some(live);
                        specs
                    }
                    Err(error) => {
                        tracing::warn!(server = %name, %error, "skipping MCP server");
                        continue;
                    }
                },
            };
            add_server(&mut shared, server, &deny);
        }
    }
    Ok(shared)
}

/// The config as it is connected (and cached) under: prefixed with the server's name unless it
/// sets its own prefix, and tolerant of over-long tool names (skipped, not fatal).
fn normalized(name: &str, mut config: McpServerConfig) -> McpServerConfig {
    if config.prefix.is_none() {
        let prefix = name
            .chars()
            .map(|c| match c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                true => c,
                false => '_',
            })
            .collect();
        config.prefix = Some(prefix);
    }
    config.continue_on_error.get_or_insert(true);
    config
}

fn add_server(shared: &mut SharedMcp, mut server: McpServer, deny: &[Regex]) {
    let (denied, offered): (Vec<ToolSpec>, Vec<ToolSpec>) = server
        .specs
        .iter()
        .cloned()
        .partition(|spec| deny.iter().any(|pattern| pattern.is_match(&spec.name)));
    shared.denied.extend(denied.into_iter().map(|spec| spec.name));
    server.offered = offered.len();
    server.specs = offered.clone();
    let server = Arc::new(server);
    shared.tools.extend(offered.into_iter().map(|spec| {
        Arc::new(LazyMcpTool {
            spec,
            server: server.clone(),
        }) as ToolRef
    }));
    shared.servers.push(server);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str) -> ToolSpec {
        serde_json::from_value(serde_json::json!({
            "name": name,
            "description": "d",
            "inputSchema": {"type": "object"},
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn an_unchanged_listing_does_not_rewrite_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp-tools.json");
        let config: McpServerConfig = serde_json::from_value(serde_json::json!({"command": "x"})).unwrap();
        let cache = ToolCache::open(path.clone()).await;
        cache.store("s", &config, vec![spec("a")]).await;
        assert!(std::fs::read_to_string(&path).unwrap().contains("\"a\""));
        // A marker the next write would replace.
        std::fs::write(&path, "marker").unwrap();
        cache.store("s", &config, vec![spec("a")]).await;
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "marker",
            "same tools, no write"
        );
        cache.store("s", &config, vec![spec("b")]).await;
        assert!(
            std::fs::read_to_string(&path).unwrap().contains("\"b\""),
            "changed tools are written"
        );
        // The next process reads the file and lists the same tools.
        let reopened = ToolCache::open(path.clone()).await;
        std::fs::write(&path, "marker").unwrap();
        reopened.store("s", &config, vec![spec("b")]).await;
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "marker");
    }
}
