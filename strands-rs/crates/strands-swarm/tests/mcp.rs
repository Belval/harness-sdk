//! Lazy MCP servers against a small fake stdio server (a Python script): tool lists come from the
//! per-run cache without connecting, the first call connects, a dead server reconnects, and a
//! changed config invalidates its cache entry.

use std::path::PathBuf;
use std::time::Duration;

use serde_json::json;
use strands_agents::agent::InvocationState;
use strands_agents::testing::MockMessageModel;
use strands_agents::tools::{ToolContext, ToolRef};
use strands_agents::types::{ToolResultStatus, ToolUse};
use strands_agents::Agent;
use strands_swarm::mcp::{load_shared_mcp, McpOptions, SharedMcp, MCP_CACHE_FILE};
use tokio_util::sync::CancellationToken;

const SERVER: &str = r#"
import json, os, sys
with open(sys.argv[1], "a") as log:
    log.write("start\n")
def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()
for line in sys.stdin:
    message = json.loads(line)
    id, method = message.get("id"), message.get("method")
    if id is None:
        continue
    params = message.get("params") or {}
    if method == "initialize":
        result = {"protocolVersion": params["protocolVersion"], "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fake", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [
            {"name": "echo", "description": "Echo the text back",
             "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
            {"name": "die", "description": "Exit the server", "inputSchema": {"type": "object"}},
        ]}
    elif method == "tools/call" and params["name"] == "die":
        send({"jsonrpc": "2.0", "id": id, "result": {"content": [{"type": "text", "text": "bye"}]}})
        os._exit(0)
    elif method == "tools/call":
        text = (params.get("arguments") or {}).get("text", "")
        result = {"content": [{"type": "text", "text": "echo: " + text}]}
    elif method == "ping":
        result = {}
    else:
        send({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "unknown method"}})
        continue
    send({"jsonrpc": "2.0", "id": id, "result": result})
"#;

struct Fixture {
    dir: tempfile::TempDir,
}

impl Fixture {
    /// `None` when there is no `python3` to run the fake server.
    fn new() -> Option<Self> {
        let python = std::process::Command::new("python3").arg("--version").output();
        if !python.is_ok_and(|out| out.status.success()) {
            eprintln!("skipping: python3 not found");
            return None;
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("server.py"), SERVER).unwrap();
        let fixture = Self { dir };
        fixture.write_config("v1");
        Some(fixture)
    }

    /// The config; `version` is an unused extra argument, so changing it changes the config.
    fn write_config(&self, version: &str) {
        let config = json!({"mcpServers": {"fake": {
            "command": "python3",
            "args": [self.path("server.py"), self.path("starts.log"), version],
        }}});
        std::fs::write(self.config(), config.to_string()).unwrap();
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    fn config(&self) -> PathBuf {
        self.path("mcp.json")
    }

    fn starts(&self) -> usize {
        std::fs::read_to_string(self.path("starts.log"))
            .map(|log| log.lines().count())
            .unwrap_or(0)
    }

    async fn load(&self, connect: bool) -> SharedMcp {
        self.load_denying(connect, &[]).await
    }

    async fn load_denying(&self, connect: bool, deny: &[&str]) -> SharedMcp {
        let options = McpOptions {
            deny: deny.iter().map(|pattern| (*pattern).to_owned()).collect(),
            stderr_dir: Some(self.path("logs")),
            cache: Some(self.path(MCP_CACHE_FILE)),
            connect,
        };
        load_shared_mcp(&[self.config()], options).await.unwrap()
    }
}

/// What agents are sent for these tools: their specs and the `find_tools` spec, serialized.
fn sent_specs(mcp: &SharedMcp) -> String {
    let specs: Vec<_> = mcp.tools.iter().map(|tool| tool.spec().clone()).collect();
    let find = mcp.catalog().find_tool().spec().clone();
    serde_json::to_string(&(specs, find)).unwrap()
}

fn names(mcp: &SharedMcp) -> Vec<&str> {
    mcp.tools.iter().map(|tool| tool.name()).collect()
}

async fn call(tool: &ToolRef, text: &str) -> (ToolResultStatus, String) {
    let agent = Agent::builder()
        .model(MockMessageModel::shared(Vec::new()))
        .no_sandbox()
        .build()
        .unwrap();
    let tool_use = ToolUse {
        name: tool.name().to_owned(),
        tool_use_id: "t1".into(),
        input: json!({"text": text}),
    };
    let ctx = ToolContext::new(
        tool_use,
        agent,
        InvocationState::default(),
        CancellationToken::new(),
    );
    let result = tool.invoke(ctx).await.unwrap();
    (result.status, format!("{:?}", result.content))
}

fn tool(mcp: &SharedMcp, name: &str) -> ToolRef {
    mcp.tools
        .iter()
        .find(|tool| tool.name() == name)
        .cloned()
        .unwrap()
}

fn echo(mcp: &SharedMcp) -> ToolRef {
    tool(mcp, "fake_echo")
}

async fn wait_until_disconnected(mcp: &SharedMcp) {
    let gone = async {
        while mcp.connected().await > 0 {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), gone)
        .await
        .expect("the server never went away");
}

#[tokio::test]
async fn a_cached_server_starts_on_first_call_and_restarts_after_dying() {
    let Some(fixture) = Fixture::new() else {
        return;
    };
    let cold = fixture.load(false).await;
    assert_eq!(fixture.starts(), 1, "no cache yet: listed live");
    assert_eq!(cold.connected().await, 1);
    assert_eq!(names(&cold), ["fake_echo", "fake_die"]);
    assert!(fixture.path(MCP_CACHE_FILE).is_file());
    let listed = sent_specs(&cold);
    drop(cold);

    let warm = fixture.load(false).await;
    assert_eq!(
        fixture.starts(),
        1,
        "listed from the cache without starting the server"
    );
    assert_eq!(warm.connected().await, 0);
    assert_eq!(names(&warm), ["fake_echo", "fake_die"]);
    assert_eq!(
        sent_specs(&warm),
        listed,
        "cached specs are sent byte for byte as listed"
    );
    let catalog = warm.catalog();
    assert!(catalog.get("fake_echo").is_some());

    let (status, text) = call(&echo(&warm), "hi").await;
    assert_eq!(status, ToolResultStatus::Success, "{text}");
    assert!(text.contains("echo: hi"), "{text}");
    assert_eq!(fixture.starts(), 2, "the first call started it");
    call(&echo(&warm), "again").await;
    assert_eq!(fixture.starts(), 2, "and it stays up");

    call(&tool(&warm, "fake_die"), "").await;
    wait_until_disconnected(&warm).await;
    let (status, text) = call(&echo(&warm), "back").await;
    assert_eq!(status, ToolResultStatus::Success, "{text}");
    assert!(text.contains("echo: back"), "{text}");
    assert_eq!(fixture.starts(), 3, "reconnected after the server died");
}

#[tokio::test]
async fn a_changed_config_or_connect_lists_live_again() {
    let Some(fixture) = Fixture::new() else {
        return;
    };
    drop(fixture.load(false).await);
    assert_eq!(fixture.starts(), 1);
    let denied = fixture.load_denying(false, &["fake_d.*"]).await;
    assert_eq!(fixture.starts(), 1, "the deny filter applies to cached tools too");
    assert_eq!(
        (names(&denied), denied.denied.clone()),
        (vec!["fake_echo"], vec!["fake_die".to_owned()])
    );
    drop(denied);
    drop(fixture.load(true).await);
    assert_eq!(
        fixture.starts(),
        2,
        "connect starts every server despite the cache"
    );
    fixture.write_config("v2");
    let changed = fixture.load(false).await;
    assert_eq!(
        fixture.starts(),
        3,
        "a changed config is not served from the cache"
    );
    assert_eq!(changed.connected().await, 1);
    drop(changed);
    drop(fixture.load(false).await);
    assert_eq!(fixture.starts(), 3, "the new config was cached");
}

#[tokio::test]
async fn a_server_that_cannot_start_is_skipped_and_retried_on_the_next_run() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("mcp.json");
    let missing = json!({"mcpServers": {"gone": {"command": "/nonexistent/server"}}});
    std::fs::write(&config, missing.to_string()).unwrap();
    let options = McpOptions {
        cache: Some(dir.path().join(MCP_CACHE_FILE)),
        ..Default::default()
    };
    let mcp = load_shared_mcp(&[config.as_path()], options).await.unwrap();
    assert!(mcp.servers.is_empty() && mcp.tools.is_empty());
    let cached = std::fs::read_to_string(dir.path().join(MCP_CACHE_FILE)).unwrap_or_default();
    assert!(!cached.contains("gone"), "failures are not cached: {cached}");
}

#[tokio::test]
async fn enabling_a_tool_starts_its_server_in_the_background() {
    let Some(fixture) = Fixture::new() else {
        return;
    };
    drop(fixture.load(false).await);
    let mcp = fixture.load(false).await;
    assert_eq!(fixture.starts(), 1);
    let find = mcp.catalog().find_tool();
    let agent = Agent::builder()
        .model(MockMessageModel::shared(Vec::new()))
        .no_sandbox()
        .build()
        .unwrap();
    let tool_use = ToolUse {
        name: find.name().to_owned(),
        tool_use_id: "f1".into(),
        input: json!({"enable": ["fake_echo"]}),
    };
    let ctx = ToolContext::new(
        tool_use,
        agent,
        InvocationState::default(),
        CancellationToken::new(),
    );
    let result = find.invoke(ctx).await.unwrap();
    assert!(format!("{:?}", result.content).contains("Enabled: fake_echo"));
    let started = async {
        while mcp.connected().await == 0 {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), started)
        .await
        .expect("enabling did not start the server");
    assert_eq!(fixture.starts(), 2);
    call(&echo(&mcp), "x").await;
    assert_eq!(fixture.starts(), 2, "the call reuses the warmed connection");
}

#[tokio::test]
async fn the_cache_keeps_secrets_out() {
    let Some(fixture) = Fixture::new() else {
        return;
    };
    let config = json!({"mcpServers": {"fake": {
        "command": "python3",
        "args": [fixture.path("server.py"), fixture.path("starts.log")],
        "env": {"TOKEN": "s3cr3t-token"},
    }}});
    std::fs::write(fixture.config(), config.to_string()).unwrap();
    drop(fixture.load(false).await);
    let cache = std::fs::read_to_string(fixture.path(MCP_CACHE_FILE)).unwrap();
    assert!(
        cache.contains("fake_echo") && !cache.contains("s3cr3t"),
        "{cache}"
    );
}
