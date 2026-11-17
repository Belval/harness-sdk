//! Estimated spend behind `/cost` and the header's `$` field.

use super::*;

impl Swarm {
    /// Every agent's tokens with an estimated cost from [`SwarmConfig::prices`], cancelled
    /// agents included (their tokens were spent too). No model call.
    pub fn cost(&self) -> crate::cost::CostReport {
        let nodes = self.nodes();
        let default = self.inner.config.default_model.clone();
        let agents = nodes
            .iter()
            .map(|node| {
                let usage = *node.usage.lock();
                // Not under another lock of the node (see `Node::info`).
                let ledger = node.ledger.lock().clone();
                crate::cost::AgentCost {
                    id: node.id.clone(),
                    parent: node.parent.clone(),
                    label: node.label(),
                    totals: crate::cost::Counts {
                        calls: 0,
                        input: usage.input,
                        output: usage.output,
                        cache_read: usage.cache_read,
                        cache_write: usage.cache_write,
                    },
                    ledger,
                    fallback_model: node.model_id().or_else(|| default.clone()),
                }
            })
            .collect();
        crate::cost::CostReport {
            agents,
            prices: (*self.inner.config.prices).clone(),
        }
    }
}
