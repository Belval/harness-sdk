//! Hook providers — objects that bundle related hook registrations. Ports
//! `HookProvider`.

use super::HookRegistry;

/// A component that registers a related set of hook callbacks as one unit. Ports
/// the `HookProvider` interface (its `register_hooks`).
///
/// An implementation registers any mix of synchronous ([`HookRegistry::add_callback`])
/// and asynchronous ([`HookRegistry::add_callback_async`]) callbacks inside
/// [`HookProvider::register_hooks`], letting a caller add a cohesive feature —
/// metering, permissions, context management — with a single call.
///
/// # Example
/// ```
/// use strands_agents::hooks::{BeforeModelCallEvent, HookProvider, HookRegistry};
///
/// struct Logging;
/// impl HookProvider for Logging {
///     fn register_hooks(&self, registry: &HookRegistry) {
///         registry.add_callback::<BeforeModelCallEvent, _>(|_event| Ok(()));
///     }
/// }
/// ```
pub trait HookProvider {
    /// Registers this provider's callbacks on `registry`. Ports `register_hooks`
    /// (the TypeScript/Python `**kwargs` parameter is dropped).
    fn register_hooks(&self, registry: &HookRegistry);
}
