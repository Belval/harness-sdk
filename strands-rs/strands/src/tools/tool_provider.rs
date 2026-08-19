//! Lazily-loaded tool providers. Ports `tools/tool-provider.ts`.

use std::sync::Arc;

use async_trait::async_trait;

use crate::errors::StrandsError;
use crate::tools::Tool;

/// Supplies tools to an agent lazily, at the start of an invocation. Ports the
/// `ToolProvider` interface.
///
/// A provider bundles zero or more tools that are only materialized when the
/// agent first runs (e.g. an `AIFunction` that builds a tool from its signature,
/// or a provider that fetches remote tool definitions). Implementations usually
/// construct their tools with [`crate::tools::FunctionTool::from_spec`].
///
/// The consumer-tracking hooks (`add_consumer` / `remove_consumer`) default to
/// no-ops; override them if a provider must know which agents use it.
#[async_trait]
pub trait ToolProvider: Send + Sync {
    /// Loads the tools this provider supplies. Ports `load_tools`.
    async fn load_tools(&self) -> Result<Vec<Arc<dyn Tool>>, StrandsError>;

    /// Notifies the provider that `consumer` started using it. Ports `add_consumer`.
    fn add_consumer(&self, _consumer: &str) {}

    /// Notifies the provider that `consumer` stopped using it. Ports `remove_consumer`.
    fn remove_consumer(&self, _consumer: &str) {}
}
