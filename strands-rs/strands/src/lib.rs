//! # Strands Agents - Rust SDK
//!
//! A model-driven SDK for building AI agents. This is the Rust port of the
//! Strands Agents SDK, derived from the canonical TypeScript implementation.
//!
//! ## Quick start
//!
//! ```no_run
//! use strands_agents::{Agent, models::BedrockModel};
//!
//! # async fn run() -> Result<(), strands_agents::StrandsError> {
//! let model = BedrockModel::new("anthropic.claude-3-5-sonnet-20241022-v2:0").await;
//! let mut agent = Agent::builder().model(model).build();
//! let result = agent.invoke("Hello!").await?;
//! println!("{}", result);
//! # Ok(())
//! # }
//! ```

pub mod agent;
pub mod errors;
pub mod hooks;
pub mod interrupt;
pub mod models;
pub mod telemetry;
pub mod tools;
pub mod types;

pub use agent::{Agent, AgentBuilder, AgentResult, InvocationState};
pub use errors::StrandsError;
pub use hooks::{HookCleanup, HookEvent, HookOrder, HookRegistry};
pub use interrupt::{Interrupt, InterruptError, InterruptSource, InterruptState};
pub use models::{CacheStrategy, Model};
pub use telemetry::Tracer;
pub use tools::{Tool, ToolContext};
pub use types::interrupt::{InterruptParams, InterruptResponse, InterruptResponseContent};
pub use types::messages::{
    CachePointBlock, ContentBlock, Message, Role, StopReason, SystemContentBlock, SystemPrompt,
};
pub use types::tools::{ToolChoice, ToolSpec};

#[cfg(feature = "macros")]
pub use strands_agents_macros::tool;

/// Re-exports of dependencies the generated macro code relies on.
///
/// Not part of the stable public API; exists so `#[tool]`-generated code can
/// reference `async_trait` without the caller depending on it directly.
#[doc(hidden)]
pub mod reexport {
    pub use async_trait::async_trait;
}

/// Convenient re-exports for common imports.
pub mod prelude {
    pub use crate::agent::{Agent, AgentBuilder, AgentResult, InvocationState};
    pub use crate::errors::StrandsError;
    pub use crate::hooks::{HookOrder, HookRegistry};
    pub use crate::models::Model;
    pub use crate::tools::{Tool, ToolContext};
    pub use crate::types::messages::{ContentBlock, Message, Role};

    #[cfg(feature = "macros")]
    pub use strands_agents_macros::tool;
}
