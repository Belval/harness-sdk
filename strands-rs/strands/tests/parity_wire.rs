//! Wire-parity golden tests.
//!
//! These pin the cross-SDK-sensitive strings and JSON shapes that MUST stay
//! byte-identical to the canonical strands SDKs (per the cross-SDK rules in the
//! root `AGENTS.md`): stop-reason wire values, content-block/system-block JSON
//! keys, the cache-point shape, usage cache-token keys, role/status literals,
//! and interrupt-source values. They trip on drift even when the surrounding
//! Rust code legitimately diverges from the port.
//!
//! The `gen_ai.*` telemetry attribute keys, operation names, and span names are
//! the other wire-sensitive surface; they are asserted (via a capturing
//! subscriber) in `tests/agent_telemetry.rs`, which serves as their golden test.

use serde_json::json;

use strands_agents::interrupt::InterruptSource;
use strands_agents::types::interrupt::InterruptResponse;
use strands_agents::types::messages::{
    CachePointBlock, ReasoningBlock, Role, SystemContentBlock, ToolResultBlock, ToolResultContent,
    ToolResultStatus, ToolUseBlock,
};
use strands_agents::types::streaming::Usage;
use strands_agents::{ContentBlock, StopReason};

// StopReason wire values (camelCase, unknown passthrough). Byte-identical to TS.
#[test]
fn stop_reason_wire_values() {
    let cases = [
        (StopReason::EndTurn, "endTurn"),
        (StopReason::ToolUse, "toolUse"),
        (StopReason::MaxTokens, "maxTokens"),
        (StopReason::StopSequence, "stopSequence"),
        (StopReason::ContentFiltered, "contentFiltered"),
        (StopReason::GuardrailIntervened, "guardrailIntervened"),
        (StopReason::Cancelled, "cancelled"),
        (
            StopReason::ModelContextWindowExceeded,
            "modelContextWindowExceeded",
        ),
        (StopReason::Interrupt, "interrupt"),
    ];
    for (reason, wire) in cases {
        assert_eq!(
            reason.as_str(),
            wire,
            "StopReason::{reason:?} wire value drifted"
        );
        assert_eq!(
            StopReason::from_wire(wire),
            reason,
            "from_wire({wire}) drifted"
        );
    }
    // Unknown values pass through verbatim (the `(string & {})` escape hatch).
    assert_eq!(
        StopReason::from_wire("somethingNew").as_str(),
        "somethingNew"
    );
}

// Content-block JSON keys (serde external tagging, camelCase multi-word).
#[test]
fn content_block_json_keys() {
    let key = |block: &ContentBlock| -> String {
        serde_json::to_value(block)
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .next()
            .unwrap()
            .clone()
    };
    assert_eq!(key(&ContentBlock::text("hi")), "text");
    assert_eq!(
        key(&ContentBlock::ToolUse(ToolUseBlock {
            name: "t".into(),
            tool_use_id: "1".into(),
            input: json!({}),
            reasoning_signature: None,
        })),
        "toolUse"
    );
    assert_eq!(
        key(&ContentBlock::ToolResult(ToolResultBlock {
            tool_use_id: "1".into(),
            status: ToolResultStatus::Success,
            content: vec![ToolResultContent::Text("ok".into())],
        })),
        "toolResult"
    );
    assert_eq!(
        key(&ContentBlock::Reasoning(ReasoningBlock::default())),
        "reasoning"
    );
    assert_eq!(
        key(&ContentBlock::CachePoint(CachePointBlock::default_point())),
        "cachePoint"
    );
}

// Cache-point JSON shape: { "cachePoint": { "cacheType": "default", "ttl"? } }.
#[test]
fn cache_point_shape() {
    assert_eq!(
        serde_json::to_value(ContentBlock::CachePoint(CachePointBlock::default_point())).unwrap(),
        json!({ "cachePoint": { "cacheType": "default" } })
    );
    assert_eq!(
        serde_json::to_value(ContentBlock::CachePoint(CachePointBlock::with_ttl("1h"))).unwrap(),
        json!({ "cachePoint": { "cacheType": "default", "ttl": "1h" } })
    );
    // System-prompt cache point uses the same shape.
    assert_eq!(
        serde_json::to_value(SystemContentBlock::CachePoint(
            CachePointBlock::default_point()
        ))
        .unwrap(),
        json!({ "cachePoint": { "cacheType": "default" } })
    );
    assert_eq!(
        serde_json::to_value(SystemContentBlock::Text("guide".into())).unwrap(),
        json!({ "text": "guide" })
    );
}

// Tool-result content JSON keys.
#[test]
fn tool_result_content_keys() {
    let key = |c: &ToolResultContent| {
        serde_json::to_value(c)
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .next()
            .unwrap()
            .clone()
    };
    assert_eq!(key(&ToolResultContent::Text("x".into())), "text");
    assert_eq!(key(&ToolResultContent::Json(json!({}))), "json");
}

// Usage cache-token keys are the wire names shared with the provider.
#[test]
fn usage_cache_token_keys() {
    let usage = Usage {
        input_tokens: 1,
        output_tokens: 2,
        total_tokens: 3,
        cache_read_input_tokens: Some(4),
        cache_write_input_tokens: Some(5),
    };
    let value = serde_json::to_value(usage).unwrap();
    for key in [
        "inputTokens",
        "outputTokens",
        "totalTokens",
        "cacheReadInputTokens",
        "cacheWriteInputTokens",
    ] {
        assert!(
            value.get(key).is_some(),
            "Usage wire key {key} drifted: {value}"
        );
    }
}

// Single-word literal values are byte-identical across SDKs.
#[test]
fn role_and_status_literals() {
    assert_eq!(serde_json::to_value(Role::User).unwrap(), json!("user"));
    assert_eq!(
        serde_json::to_value(Role::Assistant).unwrap(),
        json!("assistant")
    );
    assert_eq!(
        serde_json::to_value(ToolResultStatus::Success).unwrap(),
        json!("success")
    );
    assert_eq!(
        serde_json::to_value(ToolResultStatus::Error).unwrap(),
        json!("error")
    );
}

// InterruptSource wire values, incl. the hyphenated multi-word case.
#[test]
fn interrupt_source_values() {
    assert_eq!(
        serde_json::to_value(InterruptSource::Tool).unwrap(),
        json!("tool")
    );
    assert_eq!(
        serde_json::to_value(InterruptSource::Hook).unwrap(),
        json!("hook")
    );
    assert_eq!(
        serde_json::to_value(InterruptSource::Middleware).unwrap(),
        json!("middleware")
    );
    assert_eq!(
        serde_json::to_value(InterruptSource::MultiagentHook).unwrap(),
        json!("multiagent-hook")
    );
}

// InterruptResponse wire keys (the resume payload).
#[test]
fn interrupt_response_keys() {
    let value = serde_json::to_value(InterruptResponse::new("id-1", json!("yes"))).unwrap();
    assert_eq!(value, json!({ "interruptId": "id-1", "response": "yes" }));
}
