//! Integration tests for agent telemetry.
//!
//! Ports the span-emission specs from `agent/__tests__/agent.tracer.test.node.ts`
//! by capturing the `tracing` spans the loop emits and asserting their names and
//! `gen_ai.*` attributes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream;
use serde_json::json;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::Layer;

use strands_agents::models::{Model, ModelEventStream, StreamOptions};
use strands_agents::tools::ToolContext;
use strands_agents::types::messages::Role;
use strands_agents::types::streaming::{
    ContentBlockDelta, Metrics, ModelStreamEvent, ToolUseStart, Usage,
};
use strands_agents::{Agent, Message, StopReason, StrandsError, Tool, ToolSpec};

// --- Capturing subscriber ---------------------------------------------------

#[derive(Clone, Default)]
struct CapturedSpan {
    name: String,
    fields: HashMap<String, String>,
}

#[derive(Default)]
struct FieldVisitor(HashMap<String, String>);

impl Visit for FieldVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        // `%value` (Display) is recorded through the debug path; format without
        // the surrounding quotes a `{:?}` on a string would add.
        self.0.insert(
            field.name().to_string(),
            format!("{value:?}").trim_matches('"').to_string(),
        );
    }
}

#[derive(Clone, Default)]
struct CapturingLayer {
    spans: Arc<Mutex<HashMap<u64, CapturedSpan>>>,
}

impl<S: Subscriber> Layer<S> for CapturingLayer {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        attrs.record(&mut visitor);
        self.spans.lock().unwrap().insert(
            id.into_u64(),
            CapturedSpan {
                name: attrs.metadata().name().to_string(),
                fields: visitor.0,
            },
        );
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        values.record(&mut visitor);
        let mut spans = self.spans.lock().unwrap();
        if let Some(span) = spans.get_mut(&id.into_u64()) {
            span.fields.extend(visitor.0);
        }
    }
}

impl CapturingLayer {
    fn spans(&self) -> Vec<CapturedSpan> {
        self.spans.lock().unwrap().values().cloned().collect()
    }
}

// --- Scripted model + tool --------------------------------------------------

struct Turn {
    events: Vec<ModelStreamEvent>,
}

impl Turn {
    fn text_with_usage(text: &str) -> Self {
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
                ModelStreamEvent::Metadata {
                    usage: Some(Usage {
                        input_tokens: 12,
                        output_tokens: 8,
                        total_tokens: 20,
                        ..Usage::default()
                    }),
                    metrics: Some(Metrics {
                        latency_ms: 42,
                        time_to_first_byte_ms: None,
                    }),
                },
            ],
        }
    }

    fn tool_use(name: &str, id: &str) -> Self {
        Turn {
            events: vec![
                ModelStreamEvent::MessageStart {
                    role: Role::Assistant,
                },
                ModelStreamEvent::ContentBlockStart {
                    start: Some(ToolUseStart {
                        name: name.to_string(),
                        tool_use_id: id.to_string(),
                        reasoning_signature: None,
                    }),
                },
                ModelStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::ToolUseInput("{}".to_string()),
                },
                ModelStreamEvent::ContentBlockStop,
                ModelStreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ],
        }
    }
}

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
        Some("test-model-id")
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
            .expect("model over-called");
        Box::pin(stream::iter(turn.events.into_iter().map(Ok)))
    }
}

struct NoopTool;

#[async_trait]
impl Tool for NoopTool {
    fn name(&self) -> &str {
        "adder"
    }
    fn description(&self) -> &str {
        "no-op"
    }
    fn tool_spec(&self) -> ToolSpec {
        ToolSpec {
            name: "adder".to_string(),
            description: "no-op".to_string(),
            input_schema: Some(json!({ "type": "object" })),
            output_schema: None,
        }
    }
    async fn invoke(&self, _context: ToolContext) -> Result<serde_json::Value, StrandsError> {
        Ok(json!(42))
    }
}

// --- Tests ------------------------------------------------------------------

fn find<'a>(spans: &'a [CapturedSpan], name: &str) -> Option<&'a CapturedSpan> {
    spans.iter().find(|span| span.name == name)
}

// "startAgentSpan / startModelInvokeSpan": emits the gen_ai span hierarchy
#[tokio::test]
async fn emits_agent_and_model_spans_with_gen_ai_attributes() {
    let layer = CapturingLayer::default();
    let subscriber = tracing_subscriber::registry().with(layer.clone());
    let _guard = tracing::subscriber::set_default(subscriber);

    let model = ScriptedModel::new(vec![Turn::text_with_usage("hello")]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .name("Weather")
        .build();
    agent.invoke("hi").await.unwrap();

    let spans = layer.spans();

    let agent_span = find(&spans, "invoke_agent").expect("agent span");
    assert_eq!(
        agent_span.fields.get("gen_ai.operation.name").unwrap(),
        "invoke_agent"
    );
    assert_eq!(
        agent_span.fields.get("gen_ai.agent.name").unwrap(),
        "Weather"
    );
    assert_eq!(
        agent_span.fields.get("name").unwrap(),
        "invoke_agent Weather"
    );
    // Stable conventions (the default) emit the service name as gen_ai.system.
    assert_eq!(
        agent_span.fields.get("gen_ai.system").unwrap(),
        "strands-agents"
    );

    assert!(find(&spans, "execute_agent_loop_cycle").is_some());

    let model_span = find(&spans, "chat").expect("model span");
    assert_eq!(
        model_span.fields.get("gen_ai.operation.name").unwrap(),
        "chat"
    );
    assert_eq!(
        model_span.fields.get("gen_ai.request.model").unwrap(),
        "test-model-id"
    );
    // Usage recorded at span end, under both legacy and new attribute names.
    assert_eq!(
        model_span.fields.get("gen_ai.usage.total_tokens").unwrap(),
        "20"
    );
    assert_eq!(
        model_span.fields.get("gen_ai.usage.input_tokens").unwrap(),
        "12"
    );
    assert_eq!(
        model_span.fields.get("gen_ai.usage.prompt_tokens").unwrap(),
        "12"
    );
    assert_eq!(
        model_span
            .fields
            .get("gen_ai.server.request.duration")
            .unwrap(),
        "42"
    );
}

// "startToolCallSpan": emits an execute_tool span with the tool name and status
#[tokio::test]
async fn emits_tool_span_with_status() {
    let layer = CapturingLayer::default();
    let subscriber = tracing_subscriber::registry().with(layer.clone());
    let _guard = tracing::subscriber::set_default(subscriber);

    let model = ScriptedModel::new(vec![
        Turn::tool_use("adder", "t1"),
        Turn::text_with_usage("done"),
    ]);
    let mut agent = Agent::builder()
        .model_boxed(Box::new(model))
        .tool(NoopTool)
        .build();
    agent.invoke("go").await.unwrap();

    let spans = layer.spans();
    let tool_span = find(&spans, "execute_tool").expect("tool span");
    assert_eq!(
        tool_span.fields.get("gen_ai.operation.name").unwrap(),
        "execute_tool"
    );
    assert_eq!(tool_span.fields.get("gen_ai.tool.name").unwrap(), "adder");
    assert_eq!(tool_span.fields.get("gen_ai.tool.call.id").unwrap(), "t1");
    assert_eq!(tool_span.fields.get("name").unwrap(), "execute_tool adder");
    // Status recorded at span end.
    assert_eq!(
        tool_span.fields.get("gen_ai.tool.status").unwrap(),
        "success"
    );
}
