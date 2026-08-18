# Strands Agents - Rust SDK

A Rust implementation of the [Strands Agents](https://strandsagents.com/) SDK for building AI agents with a model-driven approach. Derived from the canonical TypeScript SDK (`strands-ts`).

> **Status: foundational vertical slice.** This crate ports a working end-to-end
> slice of the SDK — the core type system, the `Model` trait with streaming
> aggregation, a Bedrock provider, the tool system with a `#[tool]` macro, and
> the agent event loop. Many subsystems present in the TypeScript SDK are not yet
> ported; see [Scope](#scope) below.

## Quick start

```rust
use strands_agents::models::BedrockModel;
use strands_agents::{tool, Agent};

/// Get the current weather for a location.
#[tool]
async fn get_weather(location: String) -> String {
    format!("Weather in {location}: 72F, Sunny")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = BedrockModel::default_model().await;

    let mut agent = Agent::builder()
        .model(model)
        .system_prompt("You are a helpful assistant.")
        .tool(GetWeatherTool::new())
        .build();

    let result = agent.invoke("What's the weather in Seattle?").await?;
    println!("{result}");
    Ok(())
}
```

Run the example (requires AWS credentials in the environment):

```bash
cargo run --example weather_agent
```

## Workspace layout

```
strands-rs/
├── strands/          # the `strands-agents` crate (the SDK)
│   ├── src/
│   │   ├── types/    # messages, content blocks, media, tool specs, streaming events
│   │   ├── models/   # Model trait + stream aggregation, Bedrock provider
│   │   ├── tools/    # Tool trait, FunctionTool, registry, executor
│   │   ├── agent/    # Agent, builder, event loop, result
│   │   └── errors.rs
│   ├── tests/        # integration tests (agent loop)
│   └── examples/
├── strands-macros/   # the `#[tool]` procedural macro
└── docs/             # PORTING.md (construct mapping), TESTING.md
```

## Cargo features

| Feature   | Default | Description                                    |
|-----------|---------|------------------------------------------------|
| `macros`  | yes     | The `#[tool]` procedural macro.                |
| `bedrock` | yes     | The AWS Bedrock model provider.                |

## Scope

**Ported:** core message/content types, `Model` trait + `stream_aggregated`,
Bedrock provider (Converse Stream API), tool system (`Tool`, `FunctionTool`,
registry, sequential execution), the `#[tool]` macro, the agent loop
(`Agent::invoke`), the lifecycle **hook system** (`HookRegistry`, the
`Before*`/`After*` events with their `cancel` / `retry` / `selected_tool` /
`resume` / `end_turn` / mutable `tool_use` / `result` control fields, and
`InvocationState`), the **interrupt system** (human-in-the-loop `interrupt()`
on tools and hooks, `InterruptState`, and `Agent::resume`), and **prompt
caching** (`CachePointBlock`, structured `SystemPrompt`, and Bedrock
`BedrockCacheConfig` auto-injection + manual cache points).

**Not yet ported** (present in the TypeScript SDK): middleware, checkpointing,
telemetry/tracing, sessions, memory, structured output, tool progress-streaming,
guardrails, citations, the streaming agent API, multi-agent orchestration, and
providers other than Bedrock. A couple of streaming-update hook events are
deferred with the streaming API — see [`docs/PORTING.md`](docs/PORTING.md).

See [`docs/PORTING.md`](docs/PORTING.md) for the TypeScript→Rust construct
mapping and the per-file translation record.

## License

Apache-2.0.
