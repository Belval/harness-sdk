//! [`McpClient`] — connect to an MCP server and expose its tools (`strands-ts/src/mcp/client.ts`).

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, Weak};

use indexmap::IndexMap;
use parking_lot::RwLock;
use regex::Regex;
use rmcp::model::{CallToolRequestParams, CallToolResponse, CallToolResult};
use rmcp::service::{RoleClient, RunningService};
use rmcp::{ClientHandler, ServiceExt};
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::sync::CancellationToken;

use super::config::{ConfigSource, McpServerConfig, McpToolFilters, ResolvedTransport};
use super::tool::McpTool;
use crate::error::{Error, Result};
use crate::tools::{ToolRef, MAX_TOOL_NAME_LENGTH};
use crate::types::JsonValue;

const NOT_CONNECTED: &str = "MCP server failed to connect. Call connect(true) to retry.";

/// Connection state of an [`McpClient`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McpConnectionState {
    /// Never connected, or disconnected again.
    Disconnected,
    /// Connected and usable.
    Connected,
    /// A connect attempt failed and was swallowed (`continue_on_error`).
    Failed,
}

/// Defaults applied to every server in [`McpClient::load_servers`].
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct LoadOptions {
    /// Swallow per-server connect/list failures instead of failing the whole load.
    pub continue_on_error: bool,
    /// Append each stdio server's stderr to `<dir>/mcp-<name>.log` (the directory is created)
    /// instead of inheriting this process's stderr. Useful when stderr is a terminal UI.
    pub stderr_dir: Option<PathBuf>,
}

/// Tool naming/filtering options for [`McpClient::load_servers`] and
/// [`McpClient::list_tools_with`].
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct ListOptions {
    /// `None` inherits the client's prefix; `Some("")` disables prefixing.
    pub prefix: Option<String>,
    /// `None` inherits the client's filters.
    pub tool_filters: Option<McpToolFilters>,
    /// `load_servers` only: derive each prefix from its config key.
    pub prefix_with_server_name: bool,
}

/// Called after a `tools/list_changed` notification with the previous tool names and the new tools.
pub type ToolsChangedCallback = Arc<dyn Fn(Vec<String>, Vec<ToolRef>) + Send + Sync>;

#[derive(Debug)]
struct CompiledFilters {
    allowed: Vec<Regex>,
    rejected: Vec<Regex>,
}

impl CompiledFilters {
    fn compile(server: &str, filters: &McpToolFilters) -> Result<Self> {
        Ok(Self {
            allowed: compile_patterns(server, "allowed", &filters.allowed)?,
            rejected: compile_patterns(server, "rejected", &filters.rejected)?,
        })
    }

    fn should_include(&self, server_tool_name: &str) -> bool {
        if self.rejected.iter().any(|r| r.is_match(server_tool_name)) {
            return false;
        }
        self.allowed.is_empty() || self.allowed.iter().any(|r| r.is_match(server_tool_name))
    }
}

/// Size at which a server's stderr log is rotated to `<log>.1`, so a chatty server cannot fill
/// the disk: at most about twice this per server.
pub const STDERR_LOG_MAX: u64 = 4 * 1024 * 1024;

/// Open `path` for appending (creating it and its directory), rotating it first when full.
fn open_log(path: &Path, max: u64) -> std::io::Result<(std::fs::File, u64)> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut size = std::fs::metadata(path).map_or(0, |meta| meta.len());
    if size >= max {
        std::fs::rename(path, rotated(path))?;
        size = 0;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| std::io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    Ok((file, size))
}

fn rotated(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".1");
    PathBuf::from(name)
}

/// Copy a server's stderr into `file` (already `size` bytes) until it closes, rotating to
/// `<path>.1` whenever the file reaches `max`. Output that cannot be written is dropped: the
/// server must never block on a full pipe.
async fn copy_log(
    mut from: impl tokio::io::AsyncRead + Unpin,
    path: PathBuf,
    file: std::fs::File,
    mut size: u64,
    max: u64,
) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut file = Some(tokio::fs::File::from_std(file));
    let mut buf = vec![0; 8192];
    loop {
        let n = match from.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        if size >= max {
            if let Some(mut full) = file.take() {
                let _ = full.flush().await;
            }
            file = open_log(&path, max).ok().map(|(f, s)| {
                size = s;
                tokio::fs::File::from_std(f)
            });
        }
        if let Some(out) = file.as_mut() {
            if out.write_all(&buf[..n]).await.is_ok() {
                size += n as u64;
            }
        }
    }
    if let Some(mut out) = file {
        let _ = out.flush().await;
    }
}

/// Patterns are matched from the start of the name, like TS `new RegExp(p)` + `^`-anchored test.
fn compile_patterns(server: &str, key: &str, patterns: &[String]) -> Result<Vec<Regex>> {
    patterns
        .iter()
        .map(|pattern| {
            Regex::new(&format!("^(?:{pattern})")).map_err(|e| {
                Error::other(format!(
                    "Server \"{server}\": invalid regex in toolFilters.{key}: \"{pattern}\": {e}"
                ))
            })
        })
        .collect()
}

struct Inner {
    name: String,
    transport: ResolvedTransport,
    prefix: Option<String>,
    filters: Option<CompiledFilters>,
    continue_on_error: bool,
    /// File a stdio server's stderr is appended to; `None` inherits ours.
    stderr_log: Option<PathBuf>,
    state: RwLock<McpConnectionState>,
    service: AsyncMutex<Option<RunningService<RoleClient, Handler>>>,
    on_tools_changed: RwLock<Option<ToolsChangedCallback>>,
    registered_tool_names: RwLock<Vec<String>>,
}

/// A connected MCP server. Cloning shares the connection.
#[derive(Clone)]
pub struct McpClient {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpClient")
            .field("name", &self.inner.name)
            .field("state", &self.connection_state())
            .finish()
    }
}

impl McpClient {
    /// Connect to a single server; the client is named after its command or URL.
    pub async fn connect(config: McpServerConfig) -> Result<Self> {
        let name = config
            .command
            .clone()
            .or_else(|| config.url.clone())
            .unwrap_or_else(|| "mcp".to_owned());
        Self::connect_named(name, config, None).await
    }

    /// Connect to a server under an explicit name (the `mcp.json` key), optionally overriding the
    /// tool-name prefix.
    pub async fn connect_named(
        name: impl Into<String>,
        config: McpServerConfig,
        prefix: Option<String>,
    ) -> Result<Self> {
        Self::connect_inner(name.into(), config, prefix, None).await
    }

    async fn connect_inner(
        name: String,
        config: McpServerConfig,
        prefix: Option<String>,
        stderr_log: Option<PathBuf>,
    ) -> Result<Self> {
        let config = config.interpolate_env()?;
        let transport = config.resolve()?;
        let filters = config
            .tool_filters
            .as_ref()
            .filter(|f| !f.is_empty())
            .map(|f| CompiledFilters::compile(&name, f))
            .transpose()?;
        let continue_on_error = config.continue_on_error.unwrap_or(false);
        let client = Self {
            inner: Arc::new(Inner {
                name,
                transport,
                prefix: prefix.or(config.prefix),
                filters,
                continue_on_error,
                stderr_log,
                state: RwLock::new(McpConnectionState::Disconnected),
                service: AsyncMutex::new(None),
                on_tools_changed: RwLock::new(None),
                registered_tool_names: RwLock::new(Vec::new()),
            }),
        };
        match client.do_connect().await {
            Ok(()) => Ok(client),
            Err(error) if continue_on_error => {
                tracing::warn!(server = %client.inner.name, %error, "MCP server failed to connect");
                *client.inner.state.write() = McpConnectionState::Failed;
                Ok(client)
            }
            Err(error) => Err(error),
        }
    }

    /// Connect every enabled server in an `mcp.json`-shaped config (a path, an inline map, or raw
    /// JSON). With `load.continue_on_error`, a server that fails to connect is kept in the
    /// `Failed` state instead of aborting the load.
    pub async fn load_servers(
        source: impl Into<ConfigSource>,
        load: LoadOptions,
        list: ListOptions,
    ) -> Result<Vec<Self>> {
        let servers: IndexMap<String, McpServerConfig> = source.into().load().await?;
        let mut clients = Vec::new();
        for (name, mut config) in servers {
            if config.disabled {
                continue;
            }
            if config.continue_on_error.is_none() && load.continue_on_error {
                config.continue_on_error = Some(true);
            }
            // An explicit server prefix wins; otherwise the config key, else the shared default.
            let prefix = match (&config.prefix, list.prefix_with_server_name) {
                (Some(_), _) => None,
                (None, true) => Some(sanitize_prefix(&name)),
                (None, false) => list.prefix.clone(),
            };
            if config.tool_filters.is_none() {
                config.tool_filters = list.tool_filters.clone();
            }
            let stderr_log = load
                .stderr_dir
                .as_ref()
                .map(|dir| dir.join(format!("mcp-{name}.log")));
            match Self::connect_inner(name.clone(), config, prefix, stderr_log).await {
                Ok(client) => clients.push(client),
                Err(error) if load.continue_on_error => {
                    tracing::warn!(server = %name, %error, "skipping MCP server (continue_on_error)");
                }
                Err(error) => return Err(error),
            }
        }
        Ok(clients)
    }

    /// The server's name (its `mcp.json` key).
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Current connection state.
    pub fn connection_state(&self) -> McpConnectionState {
        *self.inner.state.read()
    }

    /// Whether the connection is up: connected and the transport still open (a stdio server that
    /// exited closes it). Rust-only: lets a long-lived host reconnect a server that died.
    pub async fn is_alive(&self) -> bool {
        let service = self.inner.service.lock().await;
        service
            .as_ref()
            .is_some_and(|service| !service.is_closed() && !service.peer().is_transport_closed())
    }

    /// What the server reported at `initialize`, once connected.
    pub async fn server_info(&self) -> Option<McpServerInfo> {
        let peer = self.inner.service.lock().await.as_ref()?.peer_info()?;
        Some(McpServerInfo::from(peer.as_ref()))
    }

    /// The server's tools as Strands tools, with the client's prefix and filters applied.
    pub async fn list_tools(&self) -> Result<Vec<ToolRef>> {
        self.list_tools_with(ListOptions::default()).await
    }

    /// [`McpClient::list_tools`] with per-call prefix/filter overrides (`None` inherits).
    pub async fn list_tools_with(&self, options: ListOptions) -> Result<Vec<ToolRef>> {
        if self.connection_state() == McpConnectionState::Failed {
            return Ok(Vec::new());
        }
        let server_tools = {
            let service = self.inner.service.lock().await;
            let service = service.as_ref().ok_or_else(|| Error::other(NOT_CONNECTED))?;
            service
                .list_all_tools()
                .await
                .map_err(|e| self.rpc_error("tools/list", e))?
        };

        let prefix = options.prefix.as_ref().or(self.inner.prefix.as_ref());
        let filters = options
            .tool_filters
            .as_ref()
            .filter(|f| !f.is_empty())
            .map(|f| CompiledFilters::compile(&self.inner.name, f))
            .transpose()?;
        let filters = filters.as_ref().or(self.inner.filters.as_ref());

        let mut tools: Vec<ToolRef> = Vec::with_capacity(server_tools.len());
        for tool in &server_tools {
            if filters.is_some_and(|f| !f.should_include(&tool.name)) {
                continue;
            }
            let agent_name = match prefix {
                Some(prefix) if !prefix.is_empty() => format!("{prefix}_{}", tool.name),
                _ => tool.name.to_string(),
            };
            if agent_name.len() > MAX_TOOL_NAME_LENGTH {
                let message = format!(
                    "server={}, tool={}, name={agent_name}, length={}, limit={MAX_TOOL_NAME_LENGTH} \
                     | tool name exceeds registry limit | use a shorter prefix or tool name",
                    self.inner.name,
                    tool.name,
                    agent_name.len()
                );
                if self.inner.continue_on_error {
                    tracing::warn!("{message} | skipping tool (continueOnError)");
                    continue;
                }
                return Err(Error::ToolValidation(message));
            }
            tools.push(Arc::new(McpTool::new(self.clone(), tool, agent_name)));
        }

        // Only re-baseline the tracked names when this call used the client's own options.
        if options.prefix.is_none() && options.tool_filters.is_none() {
            *self.inner.registered_tool_names.write() = tools.iter().map(|t| t.name().to_owned()).collect();
        }
        Ok(tools)
    }

    /// Register the `tools/list_changed` callback, replacing any previous one.
    pub fn on_tools_changed(&self, callback: impl Fn(Vec<String>, Vec<ToolRef>) + Send + Sync + 'static) {
        *self.inner.on_tools_changed.write() = Some(Arc::new(callback));
    }

    /// Drop the `tools/list_changed` callback.
    pub fn clear_on_tools_changed(&self) {
        *self.inner.on_tools_changed.write() = None;
    }

    /// Call a tool by its **server-side** name.
    pub async fn call_tool(
        &self,
        server_tool_name: &str,
        arguments: JsonValue,
        cancel: CancellationToken,
    ) -> Result<CallToolResult> {
        let arguments = match arguments {
            JsonValue::Null => None,
            JsonValue::Object(map) => Some(map),
            other => {
                return Err(Error::other(format!(
                    "MCP tool arguments must be a JSON object, got {other}"
                )))
            }
        };
        let mut params = CallToolRequestParams::new(server_tool_name.to_owned());
        params.arguments = arguments;
        let service = self.inner.service.lock().await;
        let service = service.as_ref().ok_or_else(|| Error::other(NOT_CONNECTED))?;
        // `call_tool_once` keeps rmcp from silently driving `input_required` rounds, which the
        // TS client has no equivalent for.
        let call = service.call_tool_once(params);
        let response = tokio::select! {
            biased;
            () = cancel.cancelled() => return Err(Error::Cancelled),
            response = call => response.map_err(|e| self.rpc_error("tools/call", e))?,
        };
        match response {
            CallToolResponse::Complete(result) => Ok(result),
            CallToolResponse::InputRequired(_) => Err(Error::other(format!(
                "MCP server \"{}\" requested additional input for tool \"{server_tool_name}\"; \
                 multi-round tool responses are not supported",
                self.inner.name
            ))),
            _ => Err(Error::other(format!(
                "MCP server \"{}\" returned an unsupported response for tool \"{server_tool_name}\"",
                self.inner.name
            ))),
        }
    }

    /// Close the connection. Idempotent.
    pub async fn disconnect(&self) -> Result<()> {
        let service = self.inner.service.lock().await.take();
        *self.inner.state.write() = McpConnectionState::Disconnected;
        if let Some(service) = service {
            let _ = service.cancel().await;
        }
        Ok(())
    }

    async fn do_connect(&self) -> Result<()> {
        let handler = Handler {
            inner: Arc::new(OnceLock::new()),
        };
        let _ = handler.inner.set(Arc::downgrade(&self.inner));
        let service = match &self.inner.transport {
            ResolvedTransport::Stdio {
                command,
                args,
                env,
                cwd,
            } => {
                // Off the terminal: a server that opens `/dev/tty` (a login prompt, progress
                // output) would otherwise draw over a full-screen UI.
                let path = env.get("PATH").map(std::ffi::OsStr::new);
                let dir = cwd.as_deref().map(std::path::Path::new);
                if !crate::sandbox::program_exists(std::ffi::OsStr::new(command), path, dir) {
                    return Err(self.connect_error(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("{command}: command not found"),
                    )));
                }
                let mut cmd = crate::sandbox::session_command(command);
                cmd.args(args);
                cmd.envs(env);
                if let Some(cwd) = cwd {
                    cmd.current_dir(cwd);
                }
                let log = match &self.inner.stderr_log {
                    Some(path) => Some(open_log(path, STDERR_LOG_MAX).map_err(|e| self.connect_error(e))?),
                    None => None,
                };
                let stderr = match log {
                    Some(_) => std::process::Stdio::piped(),
                    None => std::process::Stdio::inherit(),
                };
                let (transport, pipe) = rmcp::transport::TokioChildProcess::builder(cmd)
                    .stderr(stderr)
                    .spawn()
                    .map_err(|e| self.connect_error(e))?;
                if let (Some(pipe), Some((file, size)), Some(path)) = (pipe, log, &self.inner.stderr_log) {
                    tokio::spawn(copy_log(pipe, path.clone(), file, size, STDERR_LOG_MAX));
                }
                handler
                    .serve(transport)
                    .await
                    .map_err(|e| self.connect_error(e))?
            }
            ResolvedTransport::StreamableHttp { url, headers } => {
                let mut custom = std::collections::HashMap::new();
                for (key, value) in headers {
                    let name = http::HeaderName::from_bytes(key.as_bytes())
                        .map_err(|e| Error::other(format!("Invalid MCP header name \"{key}\": {e}")))?;
                    let value = http::HeaderValue::from_str(value)
                        .map_err(|e| Error::other(format!("Invalid MCP header value for \"{key}\": {e}")))?;
                    custom.insert(name, value);
                }
                let config =
                    rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig::with_uri(
                        url.as_str(),
                    )
                    .custom_headers(custom);
                let transport = rmcp::transport::StreamableHttpClientTransport::from_config(config);
                handler
                    .serve(transport)
                    .await
                    .map_err(|e| self.connect_error(e))?
            }
        };
        *self.inner.service.lock().await = Some(service);
        *self.inner.state.write() = McpConnectionState::Connected;
        Ok(())
    }

    /// Re-list tools after a `tools/list_changed` notification and fire the callback.
    async fn handle_tools_changed(&self) {
        let Some(callback) = self.inner.on_tools_changed.read().clone() else {
            return;
        };
        let old_names = self.inner.registered_tool_names.read().clone();
        match self.list_tools().await {
            Ok(tools) => callback(old_names, tools),
            Err(error) => tracing::warn!(
                server = %self.inner.name,
                %error,
                "failed to refresh tools after toolsChanged notification"
            ),
        }
    }

    fn connect_error(&self, error: impl std::fmt::Display) -> Error {
        Error::other(format!(
            "MCP server \"{}\" failed to connect: {error}",
            self.inner.name
        ))
    }

    fn rpc_error(&self, method: &str, error: impl std::fmt::Display) -> Error {
        Error::other(format!(
            "MCP server \"{}\" {method} failed: {error}",
            self.inner.name
        ))
    }

    /// A never-connected client, for unit tests that only exercise naming/mapping.
    #[cfg(test)]
    pub(crate) fn disconnected_for_test(name: &str) -> Self {
        Self {
            inner: Arc::new(Inner {
                name: name.to_owned(),
                transport: ResolvedTransport::Stdio {
                    command: "true".to_owned(),
                    args: Vec::new(),
                    env: Default::default(),
                    cwd: None,
                },
                prefix: None,
                filters: None,
                continue_on_error: false,
                stderr_log: None,
                state: RwLock::new(McpConnectionState::Disconnected),
                service: AsyncMutex::new(None),
                on_tools_changed: RwLock::new(None),
                registered_tool_names: RwLock::new(Vec::new()),
            }),
        }
    }
}

/// `[^A-Za-z0-9_-]` → `_`, so a config key like `awslabs.foo` becomes a legal tool-name prefix.
pub(crate) fn sanitize_prefix(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// rmcp client handler: the only notification we act on is `tools/list_changed`.
#[derive(Clone)]
struct Handler {
    inner: Arc<OnceLock<Weak<Inner>>>,
}

impl ClientHandler for Handler {
    fn on_tool_list_changed(
        &self,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        let client = self
            .inner
            .get()
            .and_then(Weak::upgrade)
            .map(|inner| McpClient { inner });
        async move {
            if let Some(client) = client {
                client.handle_tools_changed().await;
            }
        }
    }
}

/// What an MCP server reported during `initialize`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct McpServerInfo {
    /// Server implementation name, when the server provided one.
    pub name: Option<String>,
    /// Server implementation version, when the server provided one.
    pub version: Option<String>,
    /// The negotiated MCP protocol version.
    pub protocol_version: String,
    /// Human-readable usage instructions from the server.
    pub instructions: Option<String>,
}

impl From<&rmcp::model::ServerPeerInfo> for McpServerInfo {
    fn from(info: &rmcp::model::ServerPeerInfo) -> Self {
        Self {
            name: info.server_info.as_ref().map(|i| i.name.clone()),
            version: info.server_info.as_ref().map(|i| i.version.clone()),
            protocol_version: info.protocol_version.to_string(),
            instructions: info.instructions.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn a_servers_stderr_log_rotates_when_full() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("logs/mcp-x.log");
        let (file, size) = open_log(&path, 100).unwrap();
        // A pipe that hands out at most 16 bytes per read, like a server writing line by line.
        let (mut server, pipe) = tokio::io::duplex(16);
        let writer = tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            server.write_all(&[b'e'; 1000]).await.unwrap();
        });
        copy_log(pipe, path.clone(), file, size, 100).await;
        writer.await.unwrap();
        let current = std::fs::metadata(&path).unwrap().len();
        let previous = std::fs::metadata(rotated(&path)).unwrap().len();
        assert!(
            current <= 116 && (100..=116).contains(&previous),
            "{current}, {previous}"
        );

        // A full log is rotated on the next connect as well.
        std::fs::write(&path, vec![b'x'; 150]).unwrap();
        let (_, size) = open_log(&path, 100).unwrap();
        assert_eq!(size, 0);
        assert_eq!(std::fs::read(rotated(&path)).unwrap(), vec![b'x'; 150]);
    }

    use super::*;

    #[test]
    fn sanitizes_server_name_prefixes() {
        assert_eq!(sanitize_prefix("awslabs.foo"), "awslabs_foo");
        assert_eq!(sanitize_prefix("fs-1_a"), "fs-1_a");
        assert_eq!(sanitize_prefix("a b/c"), "a_b_c");
    }

    #[test]
    fn filters_match_from_the_start_of_the_server_name() {
        let filters = CompiledFilters::compile(
            "srv",
            &McpToolFilters {
                allowed: vec!["read_.*".to_owned()],
                rejected: vec!["read_secret".to_owned()],
            },
        )
        .unwrap();
        assert!(filters.should_include("read_file"));
        assert!(!filters.should_include("read_secret"));
        assert!(!filters.should_include("write_file"));
        assert!(!filters.should_include("x_read_file"));
    }

    #[test]
    fn allow_list_is_optional() {
        let filters = CompiledFilters::compile(
            "srv",
            &McpToolFilters {
                allowed: vec![],
                rejected: vec!["danger".to_owned()],
            },
        )
        .unwrap();
        assert!(filters.should_include("anything"));
        assert!(!filters.should_include("danger_zone"));
    }

    #[test]
    fn invalid_filter_regex_is_reported_with_the_server_and_key() {
        let err = CompiledFilters::compile(
            "srv",
            &McpToolFilters {
                allowed: vec!["(".to_owned()],
                rejected: vec![],
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.starts_with("Server \"srv\": invalid regex in toolFilters.allowed: \"(\": "),
            "{err}"
        );
    }

    #[tokio::test]
    async fn failed_clients_list_no_tools() {
        let config = McpServerConfig {
            continue_on_error: Some(true),
            ..McpServerConfig::stdio("strands-definitely-not-a-real-binary", Vec::<String>::new())
        };
        let client = McpClient::connect_named("broken", config, None).await.unwrap();
        assert_eq!(client.connection_state(), McpConnectionState::Failed);
        assert!(client.list_tools().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn connect_failures_propagate_without_continue_on_error() {
        let config = McpServerConfig::stdio("strands-definitely-not-a-real-binary", Vec::<String>::new());
        let err = McpClient::connect_named("broken", config, None)
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .starts_with("MCP server \"broken\" failed to connect:"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn disabled_servers_are_skipped() {
        let servers = serde_json::json!({
            "mcpServers": {"off": {"command": "strands-nope", "disabled": true}}
        });
        let clients = McpClient::load_servers(servers, LoadOptions::default(), ListOptions::default())
            .await
            .unwrap();
        assert!(clients.is_empty());
    }
}
