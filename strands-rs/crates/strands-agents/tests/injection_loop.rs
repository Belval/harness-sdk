//! Context injection through the public API: the rendered text must reach the model call that
//! triggered it, folded into the last user message, while the durable history stays untouched.

use std::sync::Arc;

use strands_agents::plugins::ContextInjector;
use strands_agents::test_support::{MockMessageModel, MockResponse};
use strands_agents::types::{ContentBlock, Role, ToolUseBlock};
use strands_agents::{Agent, Message};

/// Text of the last user message the model was handed on `call`.
fn last_user_text(model: &MockMessageModel, call: usize) -> String {
    let (messages, _) = model.calls()[call].clone();
    messages
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .expect("a user message")
        .text()
}

#[tokio::test]
async fn folds_injected_text_into_the_call_that_triggered_it_without_touching_history() {
    let model = MockMessageModel::shared(vec![MockResponse::message(Message::assistant("done"))]);
    let agent = Agent::builder()
        .model(Arc::clone(&model) as Arc<dyn strands_agents::models::Model>)
        .plugin(ContextInjector::inject("remember: the sky is green").with_name("notes"))
        .build()
        .expect("agent builds");

    agent.invoke("what colour is the sky?").await.expect("invoke");

    let (_, options) = model.calls()[0].clone();
    assert_eq!(options.dynamic_trailing_blocks, Some(1));

    let seen = last_user_text(&model, 0);
    assert!(seen.contains("what colour is the sky?"), "seen: {seen}");
    assert!(seen.contains("remember: the sky is green"), "seen: {seen}");
    assert!(seen.ends_with("\n\nremember: the sky is green"), "seen: {seen}");

    let history = agent.messages();
    assert_eq!(history[0].text(), "what colour is the sky?");
    assert!(!history
        .iter()
        .any(|message| message.text().contains("the sky is green")));
}

#[tokio::test]
async fn the_default_user_turn_trigger_skips_a_tool_result_turn() {
    let model = MockMessageModel::shared(vec![
        MockResponse::message(Message::new(
            Role::Assistant,
            vec![ContentBlock::ToolUse(ToolUseBlock::new(
                "missing_tool",
                "call-1",
                serde_json::json!({}),
            ))],
        )),
        MockResponse::message(Message::assistant("done")),
    ]);
    let agent = Agent::builder()
        .model(Arc::clone(&model) as Arc<dyn strands_agents::models::Model>)
        .plugin(ContextInjector::inject("ephemeral note"))
        .build()
        .expect("agent builds");

    agent.invoke("go").await.expect("invoke");

    assert_eq!(model.call_count(), 2);
    assert!(last_user_text(&model, 0).contains("ephemeral note"));
    // Second call's last user message is the tool result, not a fresh ask: no injection.
    let (messages, _) = model.calls()[1].clone();
    let last = messages.last().expect("a message");
    assert!(matches!(last.content.first(), Some(ContentBlock::ToolResult(_))));
    assert!(!messages
        .iter()
        .any(|message| message.text().contains("ephemeral note")));
}

/// Edits the history on every call, as a context strategy that offloads or trims would, and
/// records what each call's pipeline saw.
struct EditEveryCall(Arc<parking_lot::Mutex<Vec<String>>>);

#[async_trait::async_trait]
impl strands_agents::context_manager::ContextStrategy for EditEveryCall {
    fn name(&self) -> &str {
        "edit-every-call"
    }

    async fn apply(
        &self,
        ctx: &mut strands_agents::context_manager::ContextState<'_>,
    ) -> strands_agents::Result<bool> {
        let last = ctx.messages.last_mut().expect("a message");
        self.0.lock().push(last.text());
        last.content.push(ContentBlock::text(" [edited]"));
        Ok(true)
    }
}

#[tokio::test]
async fn an_injection_registered_first_still_stays_out_of_the_context_managers_edits() {
    use strands_agents::context_manager::{ContextManagerConfig, ContextManagerOption};
    let model = MockMessageModel::shared(vec![MockResponse::message(Message::assistant("done"))]);
    let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let config = ContextManagerConfig {
        strategies: Some(vec![Arc::new(EditEveryCall(Arc::clone(&seen)))]),
        ..Default::default()
    };
    let agent = Agent::builder()
        .model(Arc::clone(&model) as Arc<dyn strands_agents::models::Model>)
        .plugin(ContextInjector::inject("ephemeral note"))
        .context_manager(ContextManagerOption::Config(config))
        .build()
        .expect("agent builds");

    agent.invoke("go").await.expect("invoke");

    assert_eq!(
        seen.lock().as_slice(),
        ["go"],
        "the strategy sees the history, not the injection"
    );
    let sent = last_user_text(&model, 0);
    assert!(
        sent.contains("[edited]") && sent.ends_with("ephemeral note"),
        "sent: {sent}"
    );
    let history = agent.messages();
    assert_eq!(
        history[0].text(),
        "go [edited]",
        "the edit is kept, the injection is not"
    );
}
