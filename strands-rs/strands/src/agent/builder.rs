//! Builder for [`Agent`]. Ports the constructor-config surface of `agent/agent.ts`
//! into the idiomatic Rust builder pattern.

use std::sync::Arc;

use crate::agent::Agent;
use crate::errors::StrandsError;
use crate::hooks::{HookEvent, HookRegistry, InitializedEvent};
use crate::models::Model;
use crate::tools::{Tool, ToolRegistry};
use crate::types::messages::{Message, SystemPrompt};

/// Builder for constructing an [`Agent`].
#[derive(Default)]
pub struct AgentBuilder {
    model: Option<Box<dyn Model>>,
    system_prompt: Option<SystemPrompt>,
    messages: Vec<Message>,
    tool_registry: ToolRegistry,
    hooks: HookRegistry,
}

impl AgentBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        AgentBuilder::default()
    }

    /// Sets the model provider that drives the agent loop.
    pub fn model(mut self, model: impl Model + 'static) -> Self {
        self.model = Some(Box::new(model));
        self
    }

    /// Sets the model provider from an already-boxed trait object. Useful when
    /// the concrete model type is chosen at runtime.
    pub fn model_boxed(mut self, model: Box<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the system prompt.
    pub fn system_prompt(mut self, system_prompt: impl Into<SystemPrompt>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Seeds the conversation with initial messages.
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    /// Registers a tool. Ignores a tool whose name fails registry validation;
    /// use [`AgentBuilder::try_tool`] to surface the error.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        let _ = self.tool_registry.add(Arc::new(tool));
        self
    }

    /// Registers a tool, returning a validation error if the name is invalid or
    /// conflicts with an existing tool.
    pub fn try_tool(mut self, tool: impl Tool + 'static) -> Result<Self, StrandsError> {
        self.tool_registry.add(Arc::new(tool))?;
        Ok(self)
    }

    /// Registers a hook callback for event type `E`, fired at the default order.
    pub fn hook<E, F>(self, callback: F) -> Self
    where
        E: HookEvent + 'static,
        F: Fn(&mut E) -> Result<(), StrandsError> + Send + Sync + 'static,
    {
        self.hooks.add_callback(callback);
        self
    }

    /// Registers a hook callback for event type `E` with an explicit order.
    pub fn hook_with_order<E, F>(self, callback: F, order: i32) -> Self
    where
        E: HookEvent + 'static,
        F: Fn(&mut E) -> Result<(), StrandsError> + Send + Sync + 'static,
    {
        self.hooks.add_callback_with_order(callback, order);
        self
    }

    /// Builds the agent.
    ///
    /// # Panics
    /// Panics if no model was set. Use [`AgentBuilder::try_build`] for a
    /// non-panicking variant.
    pub fn build(self) -> Agent {
        self.try_build()
            .expect("Agent requires a model; call .model(...) before .build()")
    }

    /// Builds the agent, then fires [`InitializedEvent`] to registered hooks.
    ///
    /// Returns an error if no model was set or an initialized-hook callback
    /// fails.
    pub fn try_build(self) -> Result<Agent, StrandsError> {
        let model = self.model.ok_or_else(|| {
            StrandsError::model("Agent requires a model; call .model(...) before building")
        })?;
        let agent = Agent::new(
            model,
            self.system_prompt,
            self.messages,
            self.tool_registry,
            self.hooks,
        );
        agent.hooks().invoke_callbacks(&mut InitializedEvent {})?;
        Ok(agent)
    }
}
