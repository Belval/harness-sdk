//! Agent lifecycle hooks — event-driven extensibility.
//!
//! Ports the `hooks/` subsystem: a [`HookRegistry`] holds callbacks keyed by
//! event type, and the agent loop dispatches [`HookEvent`]s through it at each
//! lifecycle point. Callbacks read the event's data and may mutate its control
//! fields (`cancel`, `retry`, `selected_tool`, `resume`, `end_turn`, and the
//! mutable `tool_use` / `result`) to steer the loop.
//!
//! # Deviations from the TypeScript port
//!
//! - **Callbacks may be synchronous or asynchronous.** `add_callback` takes a
//!   sync `Fn(&mut E) -> Result<(), StrandsError>`; `add_callback_async` takes a
//!   `for<'a> Fn(&'a mut E) -> HookFuture<'a>` whose future may borrow the event
//!   across `await` points (callers write `|event| Box::pin(async move { … })`).
//!   Both share one dispatch path — sync callbacks are stored as ready futures —
//!   and `invoke_callbacks` is `async`.
//! - **Events carry an [`crate::agent::AgentHandle`], not the whole agent.** The
//!   loop owns the agent as `&mut self` while dispatching, so instead of a full
//!   `&Agent` reference callbacks receive a handle over the agent's shared,
//!   interior-mutable surfaces (today: the persisted [`crate::agent::AgentState`]),
//!   which grows as more surfaces are shared.
//! - **The streaming update events** (`ModelStreamUpdateEvent`,
//!   `ToolStreamUpdateEvent`) are deferred with the streaming feature.

mod events;
mod provider;

pub use events::*;
pub use provider::HookProvider;

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use crate::errors::StrandsError;

/// The future a hook callback resolves to, borrowing the event for its duration.
pub type HookFuture<'a> = Pin<Box<dyn Future<Output = Result<(), StrandsError>> + Send + 'a>>;

/// Preset hook execution orders. Lower values run first. Ports `HookOrder`.
///
/// Any integer is a valid order — these presets are reference points, not
/// bounds. `SDK_FIRST` / `SDK_LAST` mark where the SDK's own hooks run so a
/// caller can position theirs relative to them.
pub struct HookOrder;

impl HookOrder {
    /// Runs with the SDK's earliest hooks.
    pub const SDK_FIRST: i32 = -100;
    /// Reserved for intervention output hooks.
    pub const INTERVENTION_OUTPUT: i32 = -90;
    /// The default order applied when a caller does not specify one.
    pub const DEFAULT: i32 = 0;
    /// Reserved for intervention input hooks.
    pub const INTERVENTION_INPUT: i32 = 90;
    /// Runs with the SDK's latest hooks.
    pub const SDK_LAST: i32 = 100;
}

/// An event that can be dispatched to hook callbacks. Ports `HookableEvent`.
///
/// The agent loop constructs these at each lifecycle point and passes them to
/// [`HookRegistry::invoke_callbacks`].
pub trait HookEvent: Any + Send {
    /// Whether callbacks run in reverse for this event, giving `After*` events
    /// LIFO cleanup semantics. Ports `_shouldReverseCallbacks`.
    fn should_reverse_callbacks(&self) -> bool {
        false
    }
}

/// A registered synchronous hook callback. Ports `HookCallback`.
///
/// Returning `Err` propagates out of the loop, the Rust counterpart to a
/// callback throwing in TypeScript.
pub type HookCallback<E> = dyn Fn(&mut E) -> Result<(), StrandsError> + Send + Sync;

/// Removes a previously registered callback. Ports `HookCleanup`.
///
/// Safe to call more than once; a second call is a no-op.
pub type HookCleanup = Box<dyn Fn() + Send + Sync>;

/// A type-erased callback stored in the registry. Downcasts the event back to
/// its concrete type, then returns the (possibly async) work as a future.
/// Synchronous callbacks are stored as ready futures, so both forms share one
/// dispatch path.
type ErasedCallback = Arc<dyn for<'a> Fn(&'a mut (dyn Any + Send)) -> HookFuture<'a> + Send + Sync>;

struct CallbackEntry {
    id: u64,
    order: i32,
    callback: ErasedCallback,
}

#[derive(Default)]
struct Inner {
    callbacks: HashMap<TypeId, Vec<CallbackEntry>>,
    next_id: u64,
}

/// Registry of hook callbacks keyed by event type. Ports `HookRegistryImplementation`.
///
/// Cloning a registry yields another handle to the same underlying callbacks, so
/// the agent and its builder can share one registry.
#[derive(Clone, Default)]
pub struct HookRegistry {
    inner: Arc<Mutex<Inner>>,
}

impl HookRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a synchronous `callback` for event type `E` at
    /// [`HookOrder::DEFAULT`].
    ///
    /// Returns a [`HookCleanup`] that removes the callback when invoked.
    pub fn add_callback<E, F>(&self, callback: F) -> HookCleanup
    where
        E: HookEvent + 'static,
        F: Fn(&mut E) -> Result<(), StrandsError> + Send + Sync + 'static,
    {
        self.add_callback_with_order(callback, HookOrder::DEFAULT)
    }

    /// Registers a synchronous `callback` for event type `E` with an explicit
    /// `order`.
    ///
    /// Lower orders run first; callbacks with the same order run in registration
    /// order (reversed for `After*` events).
    pub fn add_callback_with_order<E, F>(&self, callback: F, order: i32) -> HookCleanup
    where
        E: HookEvent + 'static,
        F: Fn(&mut E) -> Result<(), StrandsError> + Send + Sync + 'static,
    {
        // A sync callback runs to completion when the future is created, then
        // resolves immediately — the shared dispatch path just awaits it.
        let erased: ErasedCallback = Arc::new(move |event: &mut (dyn Any + Send)| {
            let event = event
                .downcast_mut::<E>()
                .expect("hook callback invoked with mismatched event type");
            Box::pin(std::future::ready(callback(event))) as HookFuture<'_>
        });
        self.register::<E>(order, erased)
    }

    /// Registers an asynchronous `callback` for event type `E` at
    /// [`HookOrder::DEFAULT`].
    ///
    /// The callback returns a boxed future that may borrow the event across
    /// `await` points, e.g.
    /// `registry.add_callback_async::<MyEvent, _>(|event| Box::pin(async move { … }))`.
    pub fn add_callback_async<E, F>(&self, callback: F) -> HookCleanup
    where
        E: HookEvent + 'static,
        F: for<'a> Fn(&'a mut E) -> HookFuture<'a> + Send + Sync + 'static,
    {
        self.add_callback_async_with_order(callback, HookOrder::DEFAULT)
    }

    /// Registers an asynchronous `callback` for event type `E` with an explicit
    /// `order`.
    pub fn add_callback_async_with_order<E, F>(&self, callback: F, order: i32) -> HookCleanup
    where
        E: HookEvent + 'static,
        F: for<'a> Fn(&'a mut E) -> HookFuture<'a> + Send + Sync + 'static,
    {
        let erased: ErasedCallback = Arc::new(move |event: &mut (dyn Any + Send)| {
            let event = event
                .downcast_mut::<E>()
                .expect("hook callback invoked with mismatched event type");
            callback(event)
        });
        self.register::<E>(order, erased)
    }

    /// Stores an erased callback for event type `E`, returning its cleanup.
    fn register<E: 'static>(&self, order: i32, callback: ErasedCallback) -> HookCleanup {
        let type_id = TypeId::of::<E>();
        let mut inner = self.inner.lock().expect("hook registry mutex poisoned");
        let id = inner.next_id;
        inner.next_id += 1;
        inner
            .callbacks
            .entry(type_id)
            .or_default()
            .push(CallbackEntry {
                id,
                order,
                callback,
            });

        let weak = Arc::downgrade(&self.inner);
        Box::new(move || {
            if let Some(inner) = weak.upgrade() {
                let mut inner = inner.lock().expect("hook registry mutex poisoned");
                if let Some(entries) = inner.callbacks.get_mut(&type_id) {
                    entries.retain(|entry| entry.id != id);
                }
            }
        })
    }

    /// Invokes every callback registered for `event`, in order, mutating the
    /// event in place. Ports `invokeCallbacks`.
    ///
    /// For `After*` events (those whose [`HookEvent::should_reverse_callbacks`]
    /// returns `true`), callbacks run reversed then re-sorted by order, so a
    /// lower order still runs first but same-order callbacks run in reverse
    /// registration order.
    ///
    /// [`StrandsError::Interrupt`] errors are collected across callbacks rather
    /// than propagated immediately, so every hook can raise its interrupt; a
    /// combined interrupt error is returned after all callbacks run. A duplicate
    /// interrupt name across callbacks is a hard error. Any non-interrupt error
    /// propagates immediately and stops later callbacks. Ports `invokeCallbacks`.
    pub async fn invoke_callbacks<E: HookEvent + 'static>(
        &self,
        event: &mut E,
    ) -> Result<(), StrandsError> {
        // Snapshot the callbacks under the lock, then release it before invoking
        // any of them: a callback may register or remove hooks, which re-enters
        // the registry and would otherwise deadlock on the same mutex. The lock
        // is never held across an `await`.
        let ordered = {
            let inner = self.inner.lock().expect("hook registry mutex poisoned");
            let Some(entries) = inner.callbacks.get(&TypeId::of::<E>()) else {
                return Ok(());
            };
            let mut selected: Vec<(u64, i32, ErasedCallback)> = entries
                .iter()
                .map(|entry| (entry.id, entry.order, entry.callback.clone()))
                .collect();
            if event.should_reverse_callbacks() {
                selected.reverse();
            }
            // Stable sort by order: ascending order overall, ties keep the
            // (possibly reversed) registration sequence.
            selected.sort_by_key(|(_, order, _)| *order);
            selected
        };

        let mut collected: Vec<crate::interrupt::Interrupt> = Vec::new();
        for (_, _, callback) in ordered {
            match callback(event).await {
                Ok(()) => {}
                Err(StrandsError::Interrupt(interrupt_error)) => {
                    collected.extend(interrupt_error.interrupts);
                }
                Err(other) => return Err(other),
            }
        }

        if collected.is_empty() {
            return Ok(());
        }

        // A name raised by more than one callback is ambiguous on resume.
        let mut seen: Vec<&str> = Vec::new();
        let mut duplicates: Vec<&str> = Vec::new();
        for interrupt in &collected {
            if seen.contains(&interrupt.name.as_str()) {
                if !duplicates.contains(&interrupt.name.as_str()) {
                    duplicates.push(&interrupt.name);
                }
            } else {
                seen.push(&interrupt.name);
            }
        }
        if !duplicates.is_empty() {
            let names = duplicates.join(", ");
            return Err(StrandsError::model(format!(
                "interrupt_names=<{names}> | duplicate interrupt names"
            )));
        }

        Err(StrandsError::Interrupt(
            crate::interrupt::InterruptError::new(collected),
        ))
    }
}

#[cfg(test)]
mod tests {
    //! Ports the registry composition specs from `hooks/__tests__/registry.test.ts`:
    //! ordered dispatch, same-order registration order, reverse ordering for
    //! `After*` events, cleanup, and error propagation.

    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct ForwardEvent {
        log: Vec<&'static str>,
    }
    impl HookEvent for ForwardEvent {}

    #[derive(Default)]
    struct ReverseEvent {
        log: Vec<&'static str>,
    }
    impl HookEvent for ReverseEvent {
        fn should_reverse_callbacks(&self) -> bool {
            true
        }
    }

    // "runs callbacks in registration order for same order value"
    #[tokio::test]
    async fn runs_in_registration_order() {
        let registry = HookRegistry::new();
        registry.add_callback::<ForwardEvent, _>(|event| {
            event.log.push("a");
            Ok(())
        });
        registry.add_callback::<ForwardEvent, _>(|event| {
            event.log.push("b");
            Ok(())
        });
        let mut event = ForwardEvent::default();
        registry.invoke_callbacks(&mut event).await.unwrap();
        assert_eq!(event.log, vec!["a", "b"]);
    }

    // "lower order runs first regardless of registration order"
    #[tokio::test]
    async fn sorts_by_order() {
        let registry = HookRegistry::new();
        registry.add_callback_with_order::<ForwardEvent, _>(
            |event| {
                event.log.push("late");
                Ok(())
            },
            HookOrder::SDK_LAST,
        );
        registry.add_callback_with_order::<ForwardEvent, _>(
            |event| {
                event.log.push("early");
                Ok(())
            },
            HookOrder::SDK_FIRST,
        );
        let mut event = ForwardEvent::default();
        registry.invoke_callbacks(&mut event).await.unwrap();
        assert_eq!(event.log, vec!["early", "late"]);
    }

    // "After* events run same-order callbacks in reverse registration order"
    #[tokio::test]
    async fn reverses_same_order_for_after_events() {
        let registry = HookRegistry::new();
        registry.add_callback::<ReverseEvent, _>(|event| {
            event.log.push("first");
            Ok(())
        });
        registry.add_callback::<ReverseEvent, _>(|event| {
            event.log.push("second");
            Ok(())
        });
        let mut event = ReverseEvent::default();
        registry.invoke_callbacks(&mut event).await.unwrap();
        assert_eq!(event.log, vec!["second", "first"]);
    }

    // "cleanup removes the callback and is idempotent"
    #[tokio::test]
    async fn cleanup_removes_callback() {
        let registry = HookRegistry::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_in = calls.clone();
        let cleanup = registry.add_callback::<ForwardEvent, _>(move |_| {
            calls_in.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });

        registry
            .invoke_callbacks(&mut ForwardEvent::default())
            .await
            .unwrap();
        cleanup();
        cleanup(); // idempotent
        registry
            .invoke_callbacks(&mut ForwardEvent::default())
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    // "a callback error propagates and stops later callbacks"
    #[tokio::test]
    async fn error_propagates() {
        let registry = HookRegistry::new();
        registry.add_callback::<ForwardEvent, _>(|_| Err(StrandsError::model("boom")));
        registry.add_callback::<ForwardEvent, _>(|event| {
            event.log.push("unreached");
            Ok(())
        });
        let mut event = ForwardEvent::default();
        let error = registry.invoke_callbacks(&mut event).await.unwrap_err();
        assert!(matches!(error, StrandsError::Model { .. }));
        assert!(event.log.is_empty());
    }

    // "no callbacks registered is a no-op"
    #[tokio::test]
    async fn no_callbacks_is_noop() {
        let registry = HookRegistry::new();
        let mut event = ForwardEvent::default();
        registry.invoke_callbacks(&mut event).await.unwrap();
        assert!(event.log.is_empty());
    }

    // Async callbacks run and interleave in order with sync callbacks, mutating
    // the event after awaiting.
    #[tokio::test]
    async fn async_callback_runs_and_interleaves_with_sync() {
        let registry = HookRegistry::new();
        registry.add_callback::<ForwardEvent, _>(|event| {
            event.log.push("sync-first");
            Ok(())
        });
        registry.add_callback_async::<ForwardEvent, _>(|event| {
            Box::pin(async move {
                // Yield to the runtime, then mutate the event after the await.
                tokio::task::yield_now().await;
                event.log.push("async-second");
                Ok(())
            })
        });
        registry.add_callback::<ForwardEvent, _>(|event| {
            event.log.push("sync-third");
            Ok(())
        });

        let mut event = ForwardEvent::default();
        registry.invoke_callbacks(&mut event).await.unwrap();
        assert_eq!(event.log, vec!["sync-first", "async-second", "sync-third"]);
    }
}
