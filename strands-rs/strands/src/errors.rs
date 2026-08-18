//! Error types for the Strands Agents Rust SDK.
//!
//! These variants represent specific error conditions that can occur during
//! agent execution and model provider interactions. They mirror the TypeScript
//! SDK's error classes (`ModelError` and its subclasses, `ToolValidationError`,
//! `ToolNotFoundError`, etc.), collapsed into a single `thiserror` enum because
//! Rust models error hierarchies with enum variants rather than subclassing.

use crate::types::messages::Message;

/// The error type returned throughout the SDK.
///
/// Model-provider failures are represented by the `Model`, `ContextWindowOverflow`,
/// `MaxTokens`, and `ModelThrottled` variants; a consumer distinguishes them by
/// matching on the variant, which is the Rust counterpart to the TypeScript
/// `instanceof ModelError` check.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StrandsError {
    /// Base variant for errors originating from model provider interactions.
    #[error("{message}")]
    Model {
        /// Human-readable description of the model error.
        message: String,
        /// The underlying cause, preserved for error chaining.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Raised when the combined input exceeds the model's context window.
    #[error("{0}")]
    ContextWindowOverflow(String),

    /// Raised when the model stops because it reached its maximum output token
    /// limit. Carries the partial assistant message generated before the limit.
    #[error("{message}")]
    MaxTokens {
        /// Description of the max-tokens condition.
        message: String,
        /// The partial assistant message produced before hitting the limit.
        partial_message: Box<Message>,
    },

    /// Raised when a model provider returns a throttling or rate-limit error.
    #[error("{message}")]
    ModelThrottled {
        /// Description of the throttling condition.
        message: String,
        /// The underlying cause, preserved for error chaining.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Raised when a tool fails validation during registration.
    #[error("{0}")]
    ToolValidation(String),

    /// Raised when a tool cannot be found by name.
    #[error("Tool '{0}' not found")]
    ToolNotFound(String),

    /// Raised when the model fails to produce structured output.
    #[error("{0}")]
    StructuredOutput(String),

    /// Raised when a tool or hook requests a human-in-the-loop interrupt.
    ///
    /// Carries the interrupts to surface. The agent loop catches this variant to
    /// stop the turn with [`crate::StopReason::Interrupt`] rather than
    /// propagating it to the caller.
    #[error("{0}")]
    Interrupt(#[from] crate::interrupt::InterruptError),
}

impl StrandsError {
    /// Creates a [`StrandsError::Model`] with no underlying cause.
    pub fn model(message: impl Into<String>) -> Self {
        Self::Model {
            message: message.into(),
            source: None,
        }
    }

    /// Creates a [`StrandsError::Model`] that wraps an underlying cause.
    pub fn model_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Model {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}
