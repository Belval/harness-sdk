//! Minimal stdio MCP server used by the MCP end-to-end tests (`tests/mcp_e2e.rs`,
//! `tests/mcp_router_e2e.rs`). Exposes `echo`, which returns its `text` argument, and `fail`,
//! which returns a tool-level error result (`isError`) so the error path can be tested.
//! `--exit-after-ms <n>` makes the server exit that long after startup, for the dead-server tests.
//! `--stderr <text>` writes `text` to stderr at startup, for the stderr-redirect tests.

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{tool, tool_router, ServiceExt};

struct EchoServer;

#[derive(serde::Deserialize, schemars::JsonSchema, Default)]
struct EchoInput {
    /// Text to echo back.
    text: String,
}

#[tool_router(server_handler)]
impl EchoServer {
    #[tool(name = "echo", description = "Echo the given text back")]
    fn echo(&self, Parameters(EchoInput { text }): Parameters<EchoInput>) -> CallToolResult {
        CallToolResult::success(vec![ContentBlock::text(text)])
    }

    #[tool(name = "fail", description = "Always returns a tool-level error")]
    fn fail(&self) -> CallToolResult {
        CallToolResult::error(vec![ContentBlock::text("boom: the tool failed")])
    }
}

fn arg_value(flag: &str) -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == flag {
            return args.next();
        }
    }
    None
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let exit_after = arg_value("--exit-after-ms").and_then(|ms| ms.parse::<u64>().ok());
    if let Some(text) = arg_value("--stderr") {
        eprintln!("{text}");
    }
    let service = EchoServer.serve(rmcp::transport::io::stdio()).await?;
    match exit_after {
        Some(ms) => {
            tokio::select! {
                result = service.waiting() => { result?; }
                () = tokio::time::sleep(std::time::Duration::from_millis(ms)) => {
                    std::process::exit(0);
                }
            }
        }
        None => {
            service.waiting().await?;
        }
    }
    Ok(())
}
