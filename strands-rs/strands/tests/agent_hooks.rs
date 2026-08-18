//! Integration tests for the agent hook system.
//!
//! Ports the behavior specs from `agent/__tests__/agent.hook.test.ts`. Each test
//! names the `it` it mirrors. A scripted mock model and manually-implemented
//! tools stand in for the TypeScript fixtures.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream;
use serde_json::json;

use strands_agents::hooks::{
    AfterModelCallEvent, AfterToolCallEvent, AfterToolsEvent, BeforeInvocationEvent,
    BeforeModelCallEvent, BeforeToolCallEvent, BeforeToolsEvent, HookCancel, HookEndTurn,
    InitializedEvent, MessageAddedEvent, ToolResultEvent,
};
use strands_agents::models::{Model, ModelEventStream, StreamOptions};
use strands_agents::tools::ToolContext;
use strands_agents::types::messages::{Role, ToolResultContent, ToolResultStatus};
use strands_agents::types::streaming::{ContentBlockDelta, ModelStreamEvent, ToolUseStart};
use strands_agents::{Agent, ContentBlock, Message, StopReason, StrandsError, Tool, ToolSpec};

// --- Scripted model ---------------------------------------------------------

/// The events one scripted model turn emits.
struct Turn {
    events: Vec<ModelStreamEvent>,
}

impl Turn {
    fn text(text: &str) -> Self {
        Turn {
            events: vec![
                ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                },
                ModelStreamEvent::ContentBlockStart { start: None },
                ModelStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::Text(text.to_string()),
                },
                ModelStreamEvent::ContentBlockStop,
                ModelStreamEvent::MessageStop {
                    stop_reason: StopReason::EndTurn,
                },
            ],
        }
    }

    fn tool_use(name: &str, tool_use_id: &str, input: serde_json::Value) -> Self {
        Turn {
            events: vec![
                ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                },
                ModelStreamEvent::ContentBlockStart {
                    start: Some(ToolUseStart {
                        name: name.to_string(),
                        tool_use_id: tool_use_id.to_string(),
                        reasoning_signature: None,
                    }),
                },
                ModelStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::ToolUseInput(input.to_string()),
                },
                ModelStreamEvent::ContentBlockStop,
                ModelStreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ],
        }
    }
}

/// A model that replays scripted turns in order and counts its calls.
struct ScriptedModel {
    turns: Mutex<std::collections::VecDeque<Turn>>,
    calls: AtomicUsize,
}

impl ScriptedModel {
    fn new(turns: Vec<Turn>) -> Self {
        ScriptedModel {
            turns: Mutex::new(turns.into_iter().collect()),
            calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl Model for ScriptedModel {
    fn model_id(&self) -> Option<&str> {
        Some("scripted-model")
    }

    fn stream<'a>(
        &'a self,
        _messages: &'a [Message],
        _options: &'a StreamOptions,
    ) -> ModelEventStream<'a> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let turn = self
            .turns
            .lock()
            .unwrap()
            .pop_front()
            .expect("model called more times than scripted");
        Box::pin(stream::iter(turn.events.into_iter().map(Ok)))
    }
}

// --- Tools ------------------------------------------------------------------

/// Adds `a` and `b` from the tool input.
struct AddTool;

#[async_trait]
impl Tool for AddTool {
    fn name(&self) -> &str {
        "add"
    }
    fn description(&self) -> &str {
        "adds two numbers"
    }
    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "add".to_string(),
            description: "adds two numbers".to_string(),
            input_schema: Some(json!({ "type": "object" })),
            output_schema: None,
        }
    }
    async fn invoke(&self, context: ToolContext) -> Result<serde_json::Value, StrandsError> {
        let a = context
            .tool_use
            .input
            .get("a")
            .and_then(|value| value.as_i64())
            .unwrap_or(0);
        let b = context
            .tool_use
            .input
            .get("b")
            .and_then(|value| value.as_i64())
            .unwrap_or(0);
        Ok(json!(a + b))
    }
}

/// A tool that always fails, to exercise the error path.
struct FailingTool;

#[async_trait]
impl Tool for FailingTool {
    fn name(&self) -> &str {
        "boom"
    }
    fn description(&self) -> &str {
        "always fails"
    }
    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "boom".to_string(),
            description: "always fails".to_string(),
            input_schema: Some(json!({ "type": "object" })),
            output_schema: None,
        }
    }
    async fn invoke(&self, _context: ToolContext) -> Result<serde_json::Value, StrandsError> {
        Err(StrandsError::model("tool exploded"))
    }
}

/// Returns a fixed marker, used to prove `selected_tool` ran instead of `add`.
struct ReplacementTool;

#[async_trait]
impl Tool for ReplacementTool {
    fn name(&self) -> &str {
        "replacement"
    }
    fn description(&self) -> &str {
        "replacement tool"
    }
    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "replacement".to_string(),
            description: "replacement tool".to_string(),
            input_schema: Some(json!({ "type": "object" })),
            output_schema: None,
        }
    }
    async fn invoke(&self, _context: ToolContext) -> Result<serde_json::Value, StrandsError> {
        Ok(json!("replacement-ran"))
    }
}

// --- Helpers ----------------------------------------------------------------

type Log = Arc<Mutex<Vec<&'static str>>>;

fn log() -> Log {
    Arc::new(Mutex::new(Vec::new()))
}

fn record(log: &Log, name: &'static str) {
    log.lock().unwrap().push(name);
}

fn first_tool_result_text(message: &Message) -> String {
    let ContentBlock::ToolResult(block) = &message.content[0] else {
        panic!("expected a tool result block");
    };
    match &block.content[0] {
        ToolResultContent::Text(text) => text.clone(),
        other => panic!("expected text content, got {other:?}"),
    }
}

// --- Tests ------------------------------------------------------------------

// invocation lifecycle: "fires hooks during invoke"
#[tokio::test]
async fn fires_lifecycle_hooks_during_invoke() {
    let events = log();
    let model = ScriptedModel::new(vec![Turn::text("Hello")]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .hook::<InitializedEvent, _>({
            let events = events.clone();
            move |_| {
                record(&events, "initialized");
                Ok(())
            }
        })
        .hook::<BeforeInvocationEvent, _>({
            let events = events.clone();
            move |_| {
                record(&events, "beforeInvocation");
                Ok(())
            }
        })
        .hook::<BeforeModelCallEvent, _>({
            let events = events.clone();
            move |_| {
                record(&events, "beforeModelCall");
                Ok(())
            }
        })
        .hook::<AfterModelCallEvent, _>({
            let events = events.clone();
            move |_| {
                record(&events, "afterModelCall");
                Ok(())
            }
        })
        .hook::<MessageAddedEvent, _>({
            let events = events.clone();
            move |_| {
                record(&events, "messageAdded");
                Ok(())
            }
        })
        .build();

    agent.invoke("Hi").await.unwrap();

    let events = events.lock().unwrap().clone();
    assert_eq!(
        events,
        vec![
            "initialized",
            "beforeInvocation",
            "messageAdded", // user prompt
            "beforeModelCall",
            "afterModelCall",
            "messageAdded", // assistant reply
        ]
    );
}

// runtime hook registration: "allows adding hooks after agent creation via addHook"
#[tokio::test]
async fn allows_adding_hooks_after_creation() {
    let events = log();
    let model = ScriptedModel::new(vec![Turn::text("Hello")]);
    let mut agent = Agent::builder().model_boxed(Box::new(model)).build();

    agent.add_hook::<BeforeInvocationEvent, _>({
        let events = events.clone();
        move |_| {
            record(&events, "beforeInvocation");
            Ok(())
        }
    });

    agent.invoke("Hi").await.unwrap();
    assert_eq!(*events.lock().unwrap(), vec!["beforeInvocation"]);
}

// tool execution hooks: "fires tool hooks during tool execution"
#[tokio::test]
async fn fires_tool_hooks_during_execution() {
    let events = log();
    let model = ScriptedModel::new(vec![
        Turn::tool_use("add", "t1", json!({ "a": 1, "b": 2 })),
        Turn::text("done"),
    ]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(AddTool)
        .hook::<BeforeToolsEvent, _>({
            let events = events.clone();
            move |_| {
                record(&events, "beforeTools");
                Ok(())
            }
        })
        .hook::<BeforeToolCallEvent, _>({
            let events = events.clone();
            move |_| {
                record(&events, "beforeToolCall");
                Ok(())
            }
        })
        .hook::<AfterToolCallEvent, _>({
            let events = events.clone();
            move |_| {
                record(&events, "afterToolCall");
                Ok(())
            }
        })
        .hook::<ToolResultEvent, _>({
            let events = events.clone();
            move |_| {
                record(&events, "toolResult");
                Ok(())
            }
        })
        .hook::<AfterToolsEvent, _>({
            let events = events.clone();
            move |_| {
                record(&events, "afterTools");
                Ok(())
            }
        })
        .build();

    agent.invoke("add them").await.unwrap();
    assert_eq!(
        *events.lock().unwrap(),
        vec![
            "beforeTools",
            "beforeToolCall",
            "afterToolCall",
            "toolResult",
            "afterTools"
        ]
    );
}

// tool execution hooks: "fires AfterToolCallEvent with error when tool fails"
#[tokio::test]
async fn after_tool_call_carries_error_on_failure() {
    let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let model = ScriptedModel::new(vec![
        Turn::tool_use("boom", "t1", json!({})),
        Turn::text("recovered"),
    ]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(FailingTool)
        .hook::<AfterToolCallEvent, _>({
            let captured = captured.clone();
            move |event| {
                *captured.lock().unwrap() = event.error.clone();
                Ok(())
            }
        })
        .build();

    agent.invoke("run boom").await.unwrap();

    let error = captured
        .lock()
        .unwrap()
        .clone()
        .expect("error should be captured");
    assert!(error.contains("tool exploded"));
    // The failure also becomes an error tool-result the model can react to.
    let tool_result_message = &agent.messages()[2];
    let ContentBlock::ToolResult(block) = &tool_result_message.content[0] else {
        panic!("expected a tool result block");
    };
    assert_eq!(block.status, ToolResultStatus::Error);
}

// MessageAddedEvent: "fires for initial user input"
#[tokio::test]
async fn message_added_fires_for_initial_user_input() {
    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let model = ScriptedModel::new(vec![Turn::text("Hello")]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .hook::<MessageAddedEvent, _>({
            let captured = captured.clone();
            move |event| {
                captured
                    .lock()
                    .unwrap()
                    .push(event.message.role.to_string());
                Ok(())
            }
        })
        .build();

    agent.invoke("Hi").await.unwrap();
    assert_eq!(*captured.lock().unwrap(), vec!["user", "assistant"]);
}

// AfterModelCallEvent retry: "retries model call when hook sets retry"
#[tokio::test]
async fn retries_model_call_when_hook_sets_retry() {
    let model = ScriptedModel::new(vec![Turn::text("first"), Turn::text("second")]);
    let attempts = Arc::new(AtomicUsize::new(0));
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .hook::<AfterModelCallEvent, _>({
            let attempts = attempts.clone();
            move |event| {
                let count = attempts.fetch_add(1, Ordering::SeqCst);
                // Retry exactly once (after the first attempt).
                event.retry = count == 0;
                Ok(())
            }
        })
        .build();

    let result = agent.invoke("Hi").await.unwrap();
    // Retried once: two model attempts, final answer from the second turn.
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert_eq!(result.text(), "second");
    // No duplicated user message: history is user + final assistant.
    assert_eq!(agent.messages().len(), 2);
}

// AfterToolCallEvent retry: "fires BeforeToolCallEvent on each retry"
#[tokio::test]
async fn retries_tool_call_and_refires_before_tool_call() {
    let before_calls = Arc::new(AtomicUsize::new(0));
    let after_calls = Arc::new(AtomicUsize::new(0));
    let model = ScriptedModel::new(vec![
        Turn::tool_use("add", "t1", json!({ "a": 2, "b": 3 })),
        Turn::text("done"),
    ]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(AddTool)
        .hook::<BeforeToolCallEvent, _>({
            let before_calls = before_calls.clone();
            move |_| {
                before_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        })
        .hook::<AfterToolCallEvent, _>({
            let after_calls = after_calls.clone();
            move |event| {
                let count = after_calls.fetch_add(1, Ordering::SeqCst);
                event.retry = count == 0;
                Ok(())
            }
        })
        .build();

    agent.invoke("add").await.unwrap();
    assert_eq!(before_calls.load(Ordering::SeqCst), 2);
    assert_eq!(after_calls.load(Ordering::SeqCst), 2);
}

// cancel tool via hooks: "cancels individual tool call with custom message"
#[tokio::test]
async fn cancels_individual_tool_with_custom_message() {
    let model = ScriptedModel::new(vec![
        Turn::tool_use("add", "t1", json!({ "a": 1, "b": 2 })),
        Turn::text("done"),
    ]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(AddTool)
        .hook::<BeforeToolCallEvent, _>(|event| {
            event.cancel = Some(HookCancel::WithMessage("nope".to_string()));
            Ok(())
        })
        .build();

    agent.invoke("add").await.unwrap();
    let tool_result_message = &agent.messages()[2];
    assert_eq!(first_tool_result_text(tool_result_message), "nope");
}

// cancel all tools via BeforeToolsEvent: "cancels all tools with default message"
#[tokio::test]
async fn cancels_all_tools_via_before_tools() {
    let ran = Arc::new(AtomicUsize::new(0));
    let model = ScriptedModel::new(vec![
        Turn::tool_use("add", "t1", json!({ "a": 1, "b": 2 })),
        Turn::text("done"),
    ]);
    // A tool that would record if it ran; cancellation must prevent that.
    struct Recording(Arc<AtomicUsize>);
    #[async_trait]
    impl Tool for Recording {
        fn name(&self) -> &str {
            "add"
        }
        fn description(&self) -> &str {
            "records"
        }
        fn tool_spec(&self) -> ToolSpec {
            ToolSpec {
                name: "add".to_string(),
                description: "records".to_string(),
                input_schema: Some(json!({ "type": "object" })),
                output_schema: None,
            }
        }
        async fn invoke(&self, _context: ToolContext) -> Result<serde_json::Value, StrandsError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(json!(0))
        }
    }
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(Recording(ran.clone()))
        .hook::<BeforeToolsEvent, _>(|event| {
            event.cancel = Some(HookCancel::Default);
            Ok(())
        })
        .build();

    agent.invoke("add").await.unwrap();
    assert_eq!(ran.load(Ordering::SeqCst), 0);
    assert_eq!(
        first_tool_result_text(&agent.messages()[2]),
        "Tool cancelled by hook"
    );
}

// AfterToolCallEvent: "allows hooks to replace result on AfterToolCallEvent"
#[tokio::test]
async fn replaces_result_on_after_tool_call() {
    let model = ScriptedModel::new(vec![
        Turn::tool_use("add", "t1", json!({ "a": 1, "b": 2 })),
        Turn::text("done"),
    ]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(AddTool)
        .hook::<AfterToolCallEvent, _>(|event| {
            event.result.content = vec![ToolResultContent::Text("redacted".to_string())];
            Ok(())
        })
        .build();

    agent.invoke("add").await.unwrap();
    assert_eq!(first_tool_result_text(&agent.messages()[2]), "redacted");
}

// BeforeToolCallEvent selectedTool: "invokes the replacement tool instead of the registry tool"
#[tokio::test]
async fn selected_tool_replaces_registry_tool() {
    let model = ScriptedModel::new(vec![
        Turn::tool_use("add", "t1", json!({ "a": 1, "b": 2 })),
        Turn::text("done"),
    ]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(AddTool)
        .hook::<BeforeToolCallEvent, _>(|event| {
            event.selected_tool = Some(Arc::new(ReplacementTool));
            Ok(())
        })
        .build();

    agent.invoke("add").await.unwrap();
    assert_eq!(
        first_tool_result_text(&agent.messages()[2]),
        "replacement-ran"
    );
}

// BeforeToolCallEvent selectedTool: "cancel wins over selectedTool"
#[tokio::test]
async fn cancel_wins_over_selected_tool() {
    let model = ScriptedModel::new(vec![
        Turn::tool_use("add", "t1", json!({ "a": 1, "b": 2 })),
        Turn::text("done"),
    ]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(AddTool)
        .hook::<BeforeToolCallEvent, _>(|event| {
            event.selected_tool = Some(Arc::new(ReplacementTool));
            event.cancel = Some(HookCancel::Default);
            Ok(())
        })
        .build();

    agent.invoke("add").await.unwrap();
    assert_eq!(
        first_tool_result_text(&agent.messages()[2]),
        "Tool cancelled by hook"
    );
}

// BeforeToolCallEvent toolUse mutation: "passes mutated input to the tool"
#[tokio::test]
async fn mutated_tool_use_input_reaches_tool() {
    let model = ScriptedModel::new(vec![
        Turn::tool_use("add", "t1", json!({ "a": 1, "b": 2 })),
        Turn::text("done"),
    ]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(AddTool)
        .hook::<BeforeToolCallEvent, _>(|event| {
            event.tool_use.input = json!({ "a": 10, "b": 20 });
            Ok(())
        })
        .build();

    agent.invoke("add").await.unwrap();
    // 10 + 20 = 30, not 1 + 2.
    assert_eq!(first_tool_result_text(&agent.messages()[2]), "30");
}

// cancel invocation via hooks: "cancels invocation" + "does not append user message when cancelled"
#[tokio::test]
async fn cancels_invocation_and_skips_user_message() {
    let model = ScriptedModel::new(vec![]); // model must never be called
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .hook::<BeforeInvocationEvent, _>(|event| {
            event.cancel = Some(HookCancel::WithMessage("denied".to_string()));
            Ok(())
        })
        .build();

    let result = agent.invoke("Hi").await.unwrap();
    assert_eq!(result.stop_reason, StopReason::EndTurn);
    assert_eq!(result.text(), "denied");
    // Only the cancel assistant message is in history — the user prompt was not appended.
    assert_eq!(agent.messages().len(), 1);
    assert_eq!(agent.messages()[0].role, Role::Assistant);
}

// cancel model call via hooks: "cancels model call" + "does not emit ModelMessageEvent when cancelled"
#[tokio::test]
async fn cancels_model_call_with_message() {
    let model = ScriptedModel::new(vec![]); // model must never be called
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .hook::<BeforeModelCallEvent, _>(|event| {
            event.cancel = Some(HookCancel::WithMessage("no model".to_string()));
            Ok(())
        })
        .build();

    let result = agent.invoke("Hi").await.unwrap();
    assert_eq!(result.text(), "no model");
}

// AfterToolsEvent.endTurn: "halts the loop with custom assistant message when endTurn is a string"
#[tokio::test]
async fn end_turn_halts_with_custom_message() {
    let model = ScriptedModel::new(vec![
        // Only one tool turn is scripted; if the loop continued it would panic
        // ("model called more times than scripted"), proving the halt.
        Turn::tool_use("add", "t1", json!({ "a": 1, "b": 2 })),
    ]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(AddTool)
        .hook::<AfterToolsEvent, _>(|event| {
            event.end_turn = Some(HookEndTurn::WithContent("stopping here".to_string()));
            Ok(())
        })
        .build();

    let result = agent.invoke("add").await.unwrap();
    assert_eq!(result.stop_reason, StopReason::EndTurn);
    assert_eq!(result.text(), "stopping here");
}

// AfterInvocationEvent resume: "re-invokes the agent with the resume args"
#[tokio::test]
async fn resume_reinvokes_the_agent() {
    let before_count = Arc::new(AtomicUsize::new(0));
    let result_count = Arc::new(AtomicUsize::new(0));
    let model = ScriptedModel::new(vec![Turn::text("first"), Turn::text("second")]);
    let resumed = Arc::new(AtomicUsize::new(0));

    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .hook::<BeforeInvocationEvent, _>({
            let before_count = before_count.clone();
            move |_| {
                before_count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        })
        .hook::<strands_agents::hooks::AfterInvocationEvent, _>({
            let resumed = resumed.clone();
            move |event| {
                // Resume exactly once.
                if resumed.fetch_add(1, Ordering::SeqCst) == 0 {
                    event.resume = Some(Message::user("again"));
                }
                Ok(())
            }
        })
        .hook::<strands_agents::hooks::AgentResultEvent, _>({
            let result_count = result_count.clone();
            move |_| {
                result_count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        })
        .build();

    let result = agent.invoke("start").await.unwrap();
    // Two invocation passes, but only one final AgentResultEvent.
    assert_eq!(before_count.load(Ordering::SeqCst), 2);
    assert_eq!(result_count.load(Ordering::SeqCst), 1);
    assert_eq!(result.text(), "second");
}

// AfterInvocationEvent resume: "does not resume when resume is left undefined"
#[tokio::test]
async fn does_not_resume_when_left_unset() {
    let before_count = Arc::new(AtomicUsize::new(0));
    let model = ScriptedModel::new(vec![Turn::text("only")]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .hook::<BeforeInvocationEvent, _>({
            let before_count = before_count.clone();
            move |_| {
                before_count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        })
        .build();

    agent.invoke("start").await.unwrap();
    assert_eq!(before_count.load(Ordering::SeqCst), 1);
}
