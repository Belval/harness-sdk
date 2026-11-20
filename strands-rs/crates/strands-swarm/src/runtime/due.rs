//! Todo items with a `due` time: an agent is messaged when one of its open items comes due,
//! and until then the item is a way for it to be woken, so it is not dormant.

use super::*;
use strands_harness::plugins::{TodoItem, TodoStatus};

/// Agent-state key of the latest due time already reminded about, unix seconds: an item is
/// reminded once, and again only if its `due` moves later.
const DUE_SEEN_KEY: &str = "swarm_due_seen";

impl Swarm {
    /// `id`'s todo list, if it has one.
    fn todos(&self, id: &str) -> Vec<TodoItem> {
        let agent = self.node(id).and_then(|node| node.agent());
        agent
            .and_then(|agent| agent.state().get_as(crate::status::TODOS_KEY))
            .unwrap_or_default()
    }

    /// `id`'s open todo items past their due time.
    pub fn overdue(&self, id: &str) -> Vec<TodoItem> {
        let now = self.now();
        self.todos(id)
            .into_iter()
            .filter(|item| item.overdue(now))
            .collect()
    }

    /// Whether `id` has an open todo item that is due later, which will wake it.
    pub(super) fn due_later(&self, id: &str) -> bool {
        let now = self.now();
        let open = |item: &TodoItem| item.status != TodoStatus::Completed;
        self.todos(id)
            .iter()
            .any(|item| open(item) && item.due.is_some_and(|due| due > now))
    }

    /// Message each live agent whose open todo items have come due since it was last told, once
    /// per item and due time.
    pub(super) fn remind_due(&self) {
        let now = self.now();
        for node in self.nodes() {
            if !node.status.lock().is_live() {
                continue;
            }
            let Some(agent) = node.agent() else { continue };
            let state = agent.state();
            let Some(todos) = state.get_as::<Vec<TodoItem>>(crate::status::TODOS_KEY) else {
                continue;
            };
            let seen = state.get_as::<i64>(DUE_SEEN_KEY).unwrap_or(i64::MIN);
            let fresh = |item: &&TodoItem| item.due.is_some_and(|due| due.timestamp() > seen);
            let due: Vec<&TodoItem> = todos
                .iter()
                .filter(|item| item.overdue(now))
                .filter(fresh)
                .collect();
            let Some(latest) = due.iter().filter_map(|item| item.due).max() else {
                continue;
            };
            state.set(DUE_SEEN_KEY, serde_json::json!(latest.timestamp()));
            let items: Vec<String> = due
                .iter()
                .map(|item| format!("- {}{}", item.content, item.due_note(now)))
                .collect();
            let text = format!(
                "Due now and still open on your todo list:\n{}\nDeliver it, or tell whoever is \
                 waiting when it will come and give the item a new `due`.",
                items.join("\n")
            );
            let _ = self.post(&node.id, &node.id, "due", text);
        }
    }
}
