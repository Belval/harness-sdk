# Porting guide: TypeScript → Rust

This document records how the Rust SDK (`strands-rs`) is derived from the
canonical TypeScript SDK (`strands-ts`), so the two stay mechanically
translatable. It is the Rust counterpart to the construct-mapping guidance the
`port` skill applies.

## Construct mapping

| TypeScript | Rust | Notes |
|---|---|---|
| `interface` (data shape) | `struct` with `#[derive(Serialize, Deserialize)]` | serde supplies the wire form; no separate `Data`/class split is needed. |
| discriminated union by object key (`{ toolUse: ... }`) | `enum` with `#[serde(rename_all = "camelCase")]` (external tagging) | The JSON key selects the variant, exactly as `'toolUse' in block` does in TS. |
| `type X = 'a' \| 'b'` (string union) | `enum` with `#[serde(rename_all = ...)]` | Single-word values stay byte-identical (`'user'` → `User` → `"user"`). |
| open string union (`StopReason`) | `enum` with an `Other(String)` variant + custom `Serialize`/`Deserialize` | Preserves unknown provider values, mirroring the TS `(string & {})` escape hatch. |
| `abstract class Model` | `#[async_trait] trait Model` | `stream` is the one required method; `stream_aggregated` is a provided default. |
| `async *stream(): AsyncIterable<T>` | `fn stream(...) -> Pin<Box<dyn Stream<Item = Result<T, E>> + Send>>` | Built with `async-stream`. |
| `class` error hierarchy (`ModelError` + subclasses) | one `#[non_exhaustive] enum StrandsError` (thiserror) | Rust models error hierarchies with variants; `instanceof` → `matches!`. `{ cause }` → `#[source]`. |
| `class FunctionTool` (callback union) | `struct FunctionTool` holding a boxed async closure | The tool progress-streaming surface is not ported. |
| `Map`-backed `ToolRegistry` | `struct` wrapping `Vec<(String, Arc<dyn Tool>)>` | `Vec` preserves insertion order like the JS `Map`. |
| `tool()` factory (Zod schema) | `#[tool]` proc macro (signature-derived schema) | Schema derived from the fn signature at compile time rather than a runtime Zod schema. `#[tool(name = "…", description = "…")]` overrides the derived name/description. A tool can also be built at runtime from an explicit `ToolSpec` via `FunctionTool::from_spec`. |
| `crypto.randomUUID()` | `uuid::Uuid::new_v4()` | |
| `Uint8Array` field, base64 in `toJSON` | `Vec<u8>` with `#[serde(with = "base64_bytes")]` | Keeps the base64 wire form identical. |
| `AbortSignal` cancellation | (not ported in the slice) | The TS loop's cancellation path is out of scope. |
| `Tracer` + OpenTelemetry spans | `Tracer` emitting `tracing` spans | Rust telemetry is `tracing`; a subscriber (e.g. an OTel layer) is the backend. Span *names* are static (`invoke_agent`, `chat`, …) with the dynamic OTel name in a `name` field. |
| `MiddlewareHandler` async generator (`yield events, return result`) | `Fn(C, next) -> Future<R>` (no events) | The Rust loop is non-streaming, so handlers return the result without yielding events. |
| one `MiddlewareRegistry` keyed by stage token | one typed `MiddlewareStack<C, R>` per stage | The stage token is the typed stack; no `TypeId` erasure. |
| `InterruptError` (thrown) | `StrandsError::Interrupt(InterruptError)` (an `Err`) | Rust has no exceptions; `interrupt()` returns `Result` and the loop matches the variant. |
| `interruptFromAgent(agent, …)` | `interrupt_from_state(&InterruptState, …)` | Events/tool context carry an `Arc`-backed `InterruptState` handle instead of an `agent` back-reference. |
| resume via `invoke([InterruptResponseContent])` | `Agent::resume(Vec<InterruptResponse>)` | A dedicated resume entry point; `invoke` while activated errors (input gating). |
| `InterruptState` (per-agent, mutable) | `InterruptState` wrapping `Arc<Mutex<…>>` | Owned by the agent, cloned onto interrupt-raising events/contexts so they share one state. |
| `HookRegistry` (`Map<constructor, entries>`) | `HookRegistry` wrapping `Arc<Mutex<HashMap<TypeId, Vec<entry>>>>` | The event *type* is the key (`TypeId` ↔ constructor). `Arc<Mutex<…>>` lets `add_callback` return a real cleanup closure, mirroring the TS one. |
| `HookableEvent` base class + `_shouldReverseCallbacks` | `trait HookEvent` with `should_reverse_callbacks` default | Each event `impl HookEvent`; `After*` events override the default. |
| `HookCallback = (event) => void \| Promise<void>` | `Fn(&mut E) -> Result<(), StrandsError>` | Synchronous only in the slice (see deviations); a thrown error ↔ `Err`. |
| `cancel: boolean \| string` | `Option<HookCancel>` (`Default` / `WithMessage`) | Rust has no truthiness, so the signal is an explicit `Option`; the string carries a reason. `AfterToolsEvent.endTurn` maps the same way to `Option<HookEndTurn>`, where the string is literal content. |
| `InvocationState` (mutable bag by reference) | `InvocationState` wrapping `Arc<Mutex<HashMap>>` | A cheap-clone handle so every event can carry one and share the same map. |

## Cross-SDK naming parity

Following the monorepo's cross-SDK rules:

- **Identifiers** re-case to Rust idiom (`toolUseId` ↔ `tool_use_id`, `snake_case`).
- **Single-word string-literal values** are byte-identical (`"user"`, `"success"`).
- **Multi-word string-literal values** stay `camelCase` on the wire (`toolUse`,
  `endTurn`) via serde `rename_all`, matching the TS values — never emitted as
  `snake_case`.
- **Wire field names** exchanged with a provider keep their wire format
  (`inputSchema`, `tool_use_id`).
- **Directory/file stems** match word-for-word with the Rust separator
  (`function-tool.ts` ↔ `function_tool.rs`, `tool-registry.ts` ↔ `registry.rs`).

## Per-file translation record

| TypeScript source | Rust target |
|---|---|
| `types/messages.ts` | `types/messages.rs` |
| `types/media.ts` | `types/media.rs` |
| `models/streaming.ts` | `types/streaming.rs` |
| `tools/types.ts` | `types/tools.rs` |
| `errors.ts` | `errors.rs` |
| `models/model.ts` | `models/mod.rs` |
| `models/bedrock.ts` | `models/bedrock.rs` |
| `tools/tool.ts`, `tools/function-tool.ts` | `tools/mod.rs`, `tools/function_tool.rs` |
| `registry/tool-registry.ts` | `tools/registry.rs` |
| `tools/executors/*.ts` (`ToolExecutor`, sequential) | `tools/executor.rs` (`ToolExecutor`, `SequentialToolExecutor`, `ToolExecutionContext`) |
| `agent/agent.ts` (`_stream` core) | `agent/mod.rs` |
| `types/agent.ts` (`AgentResult`) | `agent/result.rs` |
| `types/agent.ts` (`InvocationState`) | `agent/invocation.rs` |
| `agent/state.ts` (`AgentState`) + `event.agent` | `agent/state.rs` (`AgentState`, `AgentHandle`, `Messages`) |
| `agent/conversation-manager/` (`ConversationManager`) | `conversation_manager/mod.rs` |
| `session/` (`SessionManager`) | `session/mod.rs` |
| `ToolRegistry` dynamic tools (`register_dynamic_tool`, `dynamic_tools`, `get_all_tool_specs`) | `tools/registry.rs` (`Arc`-shared base + per-instance dynamic layer, `fork()`) |
| `tools/tool-provider.ts` (`ToolProvider`) | `tools/tool_provider.rs` |
| `tools/decorator.ts` runtime `DecoratedFunctionTool` / `FunctionToolMetadata` | `tools/function_tool.rs` (`FunctionTool::from_spec`) |
| `tools/structured-output-tool.ts` + structured-output loop branch | `agent/mod.rs` (`structured_output_tool_spec`, loop) + `AgentResult::structured_output` |
| `hooks/registry.ts`, `hooks/types.ts` | `hooks/mod.rs` |
| `hooks/events.ts` | `hooks/events.rs` |
| `HookProvider` (`register_hooks`) | `hooks/provider.rs` |
| `interrupt.ts` | `interrupt.rs` |
| `types/interrupt.ts` | `types/interrupt.rs` |
| `SystemPrompt` / `SystemContentBlock` / `CacheConfig` (`types/messages.ts`, `models/model.ts`) | `types/messages.rs` (`SystemPrompt`, `SystemContentBlock`), `models/mod.rs` (`CacheStrategy`) |
| Bedrock caching (`models/bedrock.ts`) | `models/bedrock.rs` (`BedrockCacheConfig`, injection, wire lowering) |
| `telemetry/tracer.ts` (span surface) | `telemetry/mod.rs` (`Tracer`, `tracing` spans) |
| `middleware/{types,registry,stages}.ts` | `middleware/mod.rs` (`MiddlewareStack`, contexts) |
| `tools/tool-factory.ts` (`tool()`) | `strands-macros/src/lib.rs` (`#[tool]`) |

## Known deviations from a literal port

- **Empty text-block filtering** drops *all* whitespace-only text blocks before
  building the message, matching the TS filter (which is not strictly "trailing").
- **`stream_aggregated` error precedence:** a malformed tool-input JSON parse is
  deferred (not thrown immediately) so the `maxTokens` check keeps precedence,
  matching the TS ordering.
- **`MAX_LOOP_ITERATIONS`** in the agent loop is a slice-local substitute for the
  TS hook-driven `InvokeOptions.limits`, which are not ported.
- **Hook callbacks may be sync or async.** `add_callback` / `builder.hook` /
  `agent.add_hook` take a sync `Fn(&mut E) -> Result<(), StrandsError>`;
  `add_callback_async` / `builder.hook_async` / `agent.add_hook_async` take a
  `for<'a> Fn(&'a mut E) -> HookFuture<'a>` (callers write
  `|event| Box::pin(async move { … })`). `invoke_callbacks` is `async`; sync
  callbacks are stored as ready futures so both share one dispatch path.
- **Hook events carry an `AgentHandle`, not a full `&Agent`.** The loop holds the
  agent as `&mut self` while dispatching, so callbacks receive a handle over the
  agent's shared, interior-mutable surfaces — the persisted `AgentState`, the
  conversation `Messages` (read + rewrite), and the model id — rather than a
  borrow of the whole agent. The handle grows as more surfaces (metrics,
  conversation manager) are shared. The agent's `messages` is a shared `Messages`
  handle; read a snapshot via `Agent::messages()`.
- **Empty-string cancel / end-turn is not falsy.** TS treats `cancel = ""` /
  `endTurn = ""` as not-triggered (JS truthiness); Rust uses `Option`, so
  `Some(HookCancel::WithMessage("".into()))` genuinely cancels with an empty
  message.
- **Deferred hook events:** `ModelStreamUpdateEvent` / `ToolStreamUpdateEvent`
  (need the streaming loop and tool-progress streaming) are not part of the
  slice. `ContentBlockEvent` fires per aggregated block rather than per streamed
  block.
- **Interrupts resume via `Agent::resume(Vec<InterruptResponse>)`**, not by
  passing `interruptResponseContent` blocks to `invoke`. `invoke` while the agent
  is interrupted returns an error (the gate the TS SDK expresses as a
  `TypeError`). The `InterruptResponseContent` block type exists for session
  serialization parity but is not a member of the model-facing `ContentBlock`
  union.
- **Concurrent-executor interrupt semantics are not ported** — only sequential
  tool execution exists, so the deferred-interrupt (let in-flight siblings
  finish) behavior does not apply. `PendingToolExecution` holds live messages and
  is not serialized in the slice.
- **Prompt caching is Bedrock-only.** The neutral `CachePointBlock` /
  `SystemContentBlock` / `Usage` cache fields exist regardless, but lowering is
  implemented only for Bedrock (the sole ported provider). The Anthropic
  `cache_control` path and the OpenAI/Google/Vercel warn-and-drop stubs port
  once those providers do. Bedrock auto-mode injection (strip-then-inject after
  tools and into the last user message, non-PDF-document placement rule) and
  manual cache-point passthrough are ported; TTLs are passed through as strings
  (`CacheTtl::from`).
- **Telemetry emits `tracing` spans, not OpenTelemetry directly**, and does not
  wire an exporter — a `tracing` subscriber is the backend (per the Rust SDK
  `AGENTS.md`). Because `tracing` span names are static, the exact OTel span-name
  string (`invoke_agent {name}`) lives in a `name` field. The in-memory
  `AgentTrace` tree / `AgentResult.traces`, the metrics `Meter`, and multi-agent /
  node / memory spans are not ported. The `gen_ai.*` attribute keys/values,
  operation names, span hierarchy, the STABLE/LATEST semconv switch
  (`gen_ai.system` vs `gen_ai.provider.name`), and `OTEL_SERVICE_NAME` are in
  parity.
- **Middleware is non-streaming and typed per stage.** Handlers are async
  functions (`Input`: `C -> C`, `Output`: `R -> R`, `Wrap`: `(C, next) -> R`)
  rather than event-yielding async generators, and each stage is its own
  `MiddlewareStack<C, R>` (registered via `Agent::invoke_model_middleware()` /
  `execute_tool_middleware()`) rather than a token-keyed registry. Only the two
  stable stages (`InvokeModelStage`, `ExecuteToolStage`) are ported; the
  `AgentStreamStage`, the event-yielding handler form, and the middleware
  interrupt bridge defer with the streaming API. The model becomes
  `Arc<dyn Model>` so a stage terminal can own it, and `ToolExecutionResult`
  carries a separate `error` string in place of TypeScript's
  `ToolResultBlock.error`.
- **`ConversationManager` reduces reactively.** The loop calls `reduce_context`
  (async) on a model `ContextWindowOverflow` and retries (bounded by
  `MAX_CONTEXT_REDUCTIONS`). Proactive `apply_management` is on the trait and the
  agent exposes the manager; with async hook callbacks now available, a
  `BeforeModelCall` hook can call it and await, as the Python `ContextManager`
  does.
- **`HookProvider` registers one typed callback per event.** The trait's
  `register_hooks(&self, &HookRegistry)` bundles registrations (via
  `add_callback` / `add_callback_async`) as a unit, matching Python. The Python
  `_StreamingHook` pattern of one callback registered across many event types and
  handled polymorphically as a base event is deferred — strands-rs callbacks are
  typed per concrete event, so a provider registers one callback per event type.
- **`InvocationState` is caller-supplied via `invoke_with_state` /
  `invoke_message_with_state` / `resume_with_state`.** Ports the
  `invoke_async(..., invocation_state=...)` half of the Python surface — the
  seeded state flows to every hook event as `event.invocation_state`. The other
  half, a streaming/async-iterator API that yields the loop's events to the
  caller (the package's websocket `_streaming` push), is still deferred with the
  larger streaming-agent-API milestone.
- **Structured output captures the tool-use input directly.** With a schema set
  (`AgentBuilder::structured_output_schema`), the loop offers a synthetic
  `strands_structured_output` tool carrying that schema, forces it if the model
  replies with plain text, and — when the model calls it — captures the tool-use
  *input* as `AgentResult::structured_output`, recording a success tool-result in
  history. Simplifications vs. TypeScript: no schema *validation* of the input
  (the JSON is captured as-is rather than validated/retried via a Zod-backed
  tool), and the structured tool is assumed to be the sole/final tool call in its
  turn (co-called normal tools in the same turn are not executed on the capture
  path). A model that refuses even when forced yields `StrandsError::StructuredOutput`.
- **`ToolRegistry` has an `Arc`-shared base + per-instance dynamic layer.**
  `register_dynamic_tool` adds runtime tools (overriding a base tool of the same
  name); `fork()` shares the read-only base with a fresh empty dynamic layer (the
  Python warm-registry fast path). `get_all_tool_specs` returns the full base +
  dynamic union; the Python progressive-disclosure spec filter (a live
  `hidden_tools` set toggled on skill activation) is the skills subsystem's
  concern and is deferred — the registry offers no built-in spec filtering yet.
- **`SessionManager` is a minimal surface.** Only `sync_agent(agent)` is ported
  (readable surfaces via `AgentHandle`), exposed on the agent and the hook handle
  so an async hook can persist after changing the conversation. Full session
  persistence (initialize / per-message append / restore) and the manager's own
  auto-sync-on-`MessageAddedEvent` hook registration are deferred; the consumer
  drives `sync_agent` explicitly for now.
- **`ToolExecutor` owns the whole tool phase of a turn.** A custom executor
  (`AgentBuilder::tool_executor`) receives a `ToolExecutionContext` of the shared
  handles (hooks, tool registry, interrupt state, the `ExecuteToolStage`
  middleware, tracer, agent handle) and fully owns the turn's tool execution —
  it can change the batch strategy (e.g. concurrency) or short-circuit. The
  default `SequentialToolExecutor` carries the complete per-tool lifecycle
  (Before/After tool-call events, `cancel`/`selected_tool`/`tool_use`-mutation/
  `retry`, `ToolResultEvent`, middleware, interrupt + pending-execution storage,
  telemetry spans, completed-results skip, `AfterToolsEvent`); a custom executor
  that wants those must reproduce them (the per-tool lifecycle is not yet exposed
  as a standalone reusable helper).
