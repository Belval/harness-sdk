//! Integration tests for the human-in-the-loop interrupt system.
//!
//! Ports the sequential-execution specs from `agent/__tests__/agent.interrupt.test.ts`.
//! Concurrent-execution, cancellation, and stream-break specs are out of scope
//! for the slice. Each test names the `it` it mirrors.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream;
use serde_json::json;

use strands_agents::hooks::{
    AfterToolCallEvent, BeforeToolCallEvent, BeforeToolsEvent, InterruptEvent,
};
use strands_agents::models::{Model, ModelEventStream, StreamOptions};
use strands_agents::tools::ToolContext;
use strands_agents::types::messages::Role;
use strands_agents::types::streaming::{ContentBlockDelta, ModelStreamEvent, ToolUseStart};
use strands_agents::{
    Agent, ContentBlock, InterruptParams, InterruptResponse, InterruptSource, Message, StopReason,
    StrandsError, Tool, ToolSpec,
};

// --- Scripted model ---------------------------------------------------------

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

    /// A turn requesting one or more tool calls: each `(name, id, input)`.
    fn tool_uses(calls: &[(&str, &str, serde_json::Value)]) -> Self {
        let mut events = vec![ModelStreamEvent::MessageStart {
            role: Role::Assistant,
        }];
        for (name, id, input) in calls {
            events.push(ModelStreamEvent::ContentBlockStart {
                start: Some(ToolUseStart {
                    name: name.to_string(),
                    tool_use_id: id.to_string(),
                    reasoning_signature: None,
                }),
            });
            events.push(ModelStreamEvent::ContentBlockDelta {
                delta: ContentBlockDelta::ToolUseInput(input.to_string()),
            });
            events.push(ModelStreamEvent::ContentBlockStop);
        }
        events.push(ModelStreamEvent::MessageStop {
            stop_reason: StopReason::ToolUse,
        });
        Turn { events }
    }
}

struct ScriptedModel {
    turns: Mutex<std::collections::VecDeque<Turn>>,
    calls: Arc<AtomicUsize>,
}

impl ScriptedModel {
    fn new(turns: Vec<Turn>) -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            ScriptedModel {
                turns: Mutex::new(turns.into_iter().collect()),
                calls: calls.clone(),
            },
            calls,
        )
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

/// A tool that raises an interrupt named `approval`, then returns the human
/// response once resumed.
struct ConfirmTool;

#[async_trait]
impl Tool for ConfirmTool {
    fn name(&self) -> &str {
        "confirm"
    }
    fn description(&self) -> &str {
        "asks for confirmation"
    }
    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "confirm".to_string(),
            description: "asks for confirmation".to_string(),
            input_schema: Some(json!({ "type": "object" })),
            output_schema: None,
        }
    }
    async fn invoke(&self, context: ToolContext) -> Result<serde_json::Value, StrandsError> {
        let response = context.interrupt(InterruptParams::new("approval"))?;
        Ok(json!({ "approved": response }))
    }
}

/// Records how many times it runs; used to prove completed tools are not replayed.
struct CountingAddTool(Arc<AtomicUsize>);

#[async_trait]
impl Tool for CountingAddTool {
    fn name(&self) -> &str {
        "add"
    }
    fn description(&self) -> &str {
        "adds and counts runs"
    }
    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "add".to_string(),
            description: "adds and counts runs".to_string(),
            input_schema: Some(json!({ "type": "object" })),
            output_schema: None,
        }
    }
    async fn invoke(&self, context: ToolContext) -> Result<serde_json::Value, StrandsError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        let a = context
            .tool_use
            .input
            .get("a")
            .and_then(|value| value.as_i64())
            .unwrap_or(0);
        Ok(json!(a))
    }
}

// --- Tests ------------------------------------------------------------------

// interrupt from tool callback: "returns stopReason interrupt when tool calls interrupt()"
#[tokio::test]
async fn tool_interrupt_stops_the_turn() {
    let (model, calls) = ScriptedModel::new(vec![Turn::tool_uses(&[("confirm", "t1", json!({}))])]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(ConfirmTool)
        .build();

    let result = agent.invoke("go").await.unwrap();

    assert_eq!(result.stop_reason, StopReason::Interrupt);
    assert_eq!(result.interrupts.len(), 1);
    assert_eq!(result.interrupts[0].name, "approval");
    assert_eq!(result.interrupts[0].source, InterruptSource::Tool);
    assert_eq!(result.interrupts[0].id, "tool:t1:approval");
    assert!(agent.interrupt_state().is_activated());
    // The model was called exactly once; the tool cycle never completed.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

// resume flow: "resumes tool callback execution without re-calling model"
#[tokio::test]
async fn resume_returns_response_without_recalling_model_for_the_tool_cycle() {
    let (model, calls) = ScriptedModel::new(vec![
        Turn::tool_uses(&[("confirm", "t1", json!({}))]),
        Turn::text("done"),
    ]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(ConfirmTool)
        .build();

    let interrupted = agent.invoke("go").await.unwrap();
    let interrupt_id = interrupted.interrupts[0].id.clone();
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let result = agent
        .resume(vec![InterruptResponse::new(interrupt_id, json!("yes"))])
        .await
        .unwrap();

    assert_eq!(result.stop_reason, StopReason::EndTurn);
    assert_eq!(result.text(), "done");
    assert!(!agent.interrupt_state().is_activated());
    // The tool cycle replayed from pending state (no extra model call); only the
    // follow-up turn called the model again.
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    // The tool result carries the human response.
    let messages = agent.messages();
    let tool_result_message = messages.iter().find(|message| {
        message
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolResult(_)))
    });
    assert!(tool_result_message.is_some());
}

// interrupt from BeforeToolCallEvent hook: "returns stopReason interrupt when hook calls interrupt()"
#[tokio::test]
async fn before_tool_call_hook_interrupt_stops_the_turn() {
    let (model, _calls) =
        ScriptedModel::new(vec![Turn::tool_uses(&[("add", "t1", json!({ "a": 1 }))])]);
    let ran = Arc::new(AtomicUsize::new(0));
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(CountingAddTool(ran.clone()))
        .hook::<BeforeToolCallEvent, _>(|event| {
            event.interrupt(InterruptParams::new("gate")).map(|_| ())
        })
        .build();

    let result = agent.invoke("go").await.unwrap();
    assert_eq!(result.stop_reason, StopReason::Interrupt);
    assert_eq!(result.interrupts[0].id, "hook:beforeToolCall:t1:gate");
    assert_eq!(result.interrupts[0].source, InterruptSource::Hook);
    // The tool never ran — the interrupt fired before execution.
    assert_eq!(ran.load(Ordering::SeqCst), 0);
}

// interrupt from BeforeToolsEvent hook: "returns stopReason interrupt when hook calls interrupt()"
#[tokio::test]
async fn before_tools_hook_interrupt_stops_the_turn() {
    let (model, _calls) =
        ScriptedModel::new(vec![Turn::tool_uses(&[("add", "t1", json!({ "a": 1 }))])]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(CountingAddTool(Arc::new(AtomicUsize::new(0))))
        .hook::<BeforeToolsEvent, _>(|event| {
            event.interrupt(InterruptParams::new("batch")).map(|_| ())
        })
        .build();

    let result = agent.invoke("go").await.unwrap();
    assert_eq!(result.stop_reason, StopReason::Interrupt);
    assert_eq!(result.interrupts[0].id, "hook:beforeTools:batch");
}

// resume flow: "preserves completed tool results when interrupt fires on a later tool"
#[tokio::test]
async fn preserves_completed_results_when_later_tool_interrupts() {
    let (model, _calls) = ScriptedModel::new(vec![
        Turn::tool_uses(&[
            ("add", "t1", json!({ "a": 5 })),
            ("confirm", "t2", json!({})),
        ]),
        Turn::text("done"),
    ]);
    let runs = Arc::new(AtomicUsize::new(0));
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(CountingAddTool(runs.clone()))
        .tool(ConfirmTool)
        .build();

    let interrupted = agent.invoke("go").await.unwrap();
    assert_eq!(interrupted.stop_reason, StopReason::Interrupt);
    // The `add` tool completed before `confirm` interrupted.
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    let interrupt_id = interrupted.interrupts[0].id.clone();
    assert_eq!(interrupt_id, "tool:t2:approval");

    let result = agent
        .resume(vec![InterruptResponse::new(interrupt_id, json!("ok"))])
        .await
        .unwrap();
    assert_eq!(result.stop_reason, StopReason::EndTurn);
    // `add` was NOT re-run on resume — its result was restored from pending state.
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

// resume error: "throws TypeError when sending a new message while in interrupted state"
#[tokio::test]
async fn invoke_while_interrupted_errors() {
    let (model, _calls) =
        ScriptedModel::new(vec![Turn::tool_uses(&[("confirm", "t1", json!({}))])]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(ConfirmTool)
        .build();

    agent.invoke("go").await.unwrap();
    let error = agent.invoke("another").await.unwrap_err();
    assert!(error.to_string().contains("interrupted state"));
}

// resume error: resuming when not interrupted is rejected
#[tokio::test]
async fn resume_when_not_interrupted_errors() {
    let (model, _calls) = ScriptedModel::new(vec![Turn::text("hi")]);
    let mut agent = Agent::builder().model_boxed(Box::new(model)).build();
    let error = agent
        .resume(vec![InterruptResponse::new("nope", json!(1))])
        .await
        .unwrap_err();
    assert!(error.to_string().contains("not in an interrupted state"));
}

// event contract: "does not fire AfterToolCallEvent when tool callback interrupts"
#[tokio::test]
async fn after_tool_call_not_fired_on_tool_interrupt() {
    let (model, _calls) =
        ScriptedModel::new(vec![Turn::tool_uses(&[("confirm", "t1", json!({}))])]);
    let after_calls = Arc::new(AtomicUsize::new(0));
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(ConfirmTool)
        .hook::<AfterToolCallEvent, _>({
            let after_calls = after_calls.clone();
            move |_| {
                after_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        })
        .build();

    agent.invoke("go").await.unwrap();
    assert_eq!(after_calls.load(Ordering::SeqCst), 0);
}

// InterruptEvent emission: "yields one InterruptEvent per unanswered interrupt at stop, tagged with source"
#[tokio::test]
async fn interrupt_event_emitted_with_source() {
    let (model, _calls) =
        ScriptedModel::new(vec![Turn::tool_uses(&[("confirm", "t1", json!({}))])]);
    let captured: Arc<Mutex<Vec<(String, InterruptSource)>>> = Arc::new(Mutex::new(Vec::new()));
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(ConfirmTool)
        .hook::<InterruptEvent, _>({
            let captured = captured.clone();
            move |event| {
                captured
                    .lock()
                    .unwrap()
                    .push((event.interrupt.name.clone(), event.interrupt.source));
                Ok(())
            }
        })
        .build();

    agent.invoke("go").await.unwrap();
    let captured = captured.lock().unwrap().clone();
    assert_eq!(
        captured,
        vec![("approval".to_string(), InterruptSource::Tool)]
    );
}

// multi-cycle interrupts: "interrupts again on cycle 2 after resuming from cycle 1"
#[tokio::test]
async fn interrupts_again_on_a_later_cycle() {
    let (model, _calls) = ScriptedModel::new(vec![
        Turn::tool_uses(&[("confirm", "t1", json!({}))]),
        Turn::tool_uses(&[("confirm", "t2", json!({}))]),
        Turn::text("done"),
    ]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(ConfirmTool)
        .build();

    // Cycle 1 interrupts.
    let first = agent.invoke("go").await.unwrap();
    assert_eq!(first.interrupts[0].id, "tool:t1:approval");

    // Resume cycle 1; the model then requests another tool that interrupts again.
    let second = agent
        .resume(vec![InterruptResponse::new(
            first.interrupts[0].id.clone(),
            json!("yes"),
        )])
        .await
        .unwrap();
    assert_eq!(second.stop_reason, StopReason::Interrupt);
    assert_eq!(second.interrupts[0].id, "tool:t2:approval");
    assert!(agent.interrupt_state().is_activated());

    // Resume cycle 2 through to completion.
    let third = agent
        .resume(vec![InterruptResponse::new(
            second.interrupts[0].id.clone(),
            json!("yes"),
        )])
        .await
        .unwrap();
    assert_eq!(third.stop_reason, StopReason::EndTurn);
    assert_eq!(third.text(), "done");
    assert!(!agent.interrupt_state().is_activated());
}
