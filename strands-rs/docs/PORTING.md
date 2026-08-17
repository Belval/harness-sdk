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
| `tool()` factory (Zod schema) | `#[tool]` proc macro (signature-derived schema) | Schema derived from the fn signature at compile time rather than a runtime Zod schema. |
| `crypto.randomUUID()` | `uuid::Uuid::new_v4()` | |
| `Uint8Array` field, base64 in `toJSON` | `Vec<u8>` with `#[serde(with = "base64_bytes")]` | Keeps the base64 wire form identical. |
| `AbortSignal` cancellation | (not ported in the slice) | The TS loop's cancellation path is out of scope. |
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
| `tools/executors/sequential.ts` | `tools/mod.rs` (`execute_tools`) |
| `agent/agent.ts` (`_stream` core) | `agent/mod.rs` |
| `types/agent.ts` (`AgentResult`) | `agent/result.rs` |
| `types/agent.ts` (`InvocationState`) | `agent/invocation.rs` |
| `hooks/registry.ts`, `hooks/types.ts` | `hooks/mod.rs` |
| `hooks/events.ts` | `hooks/events.rs` |
| `tools/tool-factory.ts` (`tool()`) | `strands-macros/src/lib.rs` (`#[tool]`) |

## Known deviations from a literal port

- **Empty text-block filtering** drops *all* whitespace-only text blocks before
  building the message, matching the TS filter (which is not strictly "trailing").
- **`stream_aggregated` error precedence:** a malformed tool-input JSON parse is
  deferred (not thrown immediately) so the `maxTokens` check keeps precedence,
  matching the TS ordering.
- **`MAX_LOOP_ITERATIONS`** in the agent loop is a slice-local substitute for the
  TS hook-driven `InvokeOptions.limits`, which are not ported.
- **Hook callbacks are synchronous.** The TS SDK awaits sync-or-async callbacks;
  the Rust slice takes `Fn(&mut E) -> Result<(), StrandsError>`. Async-callback
  ergonomics (borrowing the event across an `await`) are deferred.
- **Hook events carry no `agent` back-reference.** The loop holds the agent as
  `&mut self` while dispatching and cannot also lend it to callbacks, so events
  expose only their data plus control fields. The agent-callback path is deferred.
- **Empty-string cancel / end-turn is not falsy.** TS treats `cancel = ""` /
  `endTurn = ""` as not-triggered (JS truthiness); Rust uses `Option`, so
  `Some(HookCancel::WithMessage("".into()))` genuinely cancels with an empty
  message.
- **Deferred hook events:** `ModelStreamUpdateEvent` / `ToolStreamUpdateEvent`
  (need the streaming loop and tool-progress streaming) and `InterruptEvent` +
  the `Interruptible.interrupt()` surface on tool-call events (need the interrupt
  feature) are not part of the hooks slice. `ContentBlockEvent` fires per
  aggregated block rather than per streamed block.
