//! Builder for [`Agent`]. Ports the constructor-config surface of `agent/agent.ts`
//! into the idiomatic Rust builder pattern.

use std::sync::Arc;

use crate::agent::{Agent, AgentState};
use crate::conversation_manager::ConversationManager;
use crate::errors::StrandsError;
use crate::hooks::{HookEvent, HookFuture, HookProvider, HookRegistry, InitializedEvent};
use crate::models::Model;
use crate::session::SessionManager;
use crate::tools::{Tool, ToolProvider, ToolRegistry};
use crate::types::messages::{Message, SystemPrompt};

/// Builder for constructing an [`Agent`].
#[derive(Default)]
pub struct AgentBuilder {
    model: Option<Arc<dyn Model>>,
    name: Option<String>,
    system_prompt: Option<SystemPrompt>,
    messages: Vec<Message>,
    tool_registry: ToolRegistry,
    tool_providers: Vec<Arc<dyn ToolProvider>>,
    hooks: HookRegistry,
    state: Option<AgentState>,
    conversation_manager: Option<Arc<dyn ConversationManager>>,
    session_manager: Option<Arc<dyn SessionManager>>,
    structured_output_schema: Option<serde_json::Value>,
}

/// Default agent name used in telemetry when none is set.
const DEFAULT_AGENT_NAME: &str = "Strands Agents";

impl AgentBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        AgentBuilder::default()
    }

    /// Sets the model provider that drives the agent loop.
    pub fn model(mut self, model: impl Model + 'static) -> Self {
        self.model = Some(Arc::new(model));
        self
    }

    /// Sets the model provider from an already-boxed trait object. Useful when
    /// the concrete model type is chosen at runtime.
    pub fn model_boxed(mut self, model: Box<dyn Model>) -> Self {
        self.model = Some(Arc::from(model));
        self
    }

    /// Sets the system prompt.
    pub fn system_prompt(mut self, system_prompt: impl Into<SystemPrompt>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Sets the agent name used in telemetry (`gen_ai.agent.name`).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Seeds the agent's persisted [`AgentState`]. Defaults to empty.
    pub fn state(mut self, state: AgentState) -> Self {
        self.state = Some(state);
        self
    }

    /// Sets the conversation manager that reduces history on context-window
    /// overflow. Defaults to none (an overflow propagates).
    pub fn conversation_manager(mut self, manager: impl ConversationManager + 'static) -> Self {
        self.conversation_manager = Some(Arc::new(manager));
        self
    }

    /// Sets the session manager, exposed to hooks via `event.agent.session_manager()`
    /// so an async hook can persist the session. Defaults to none.
    pub fn session_manager(mut self, manager: impl SessionManager + 'static) -> Self {
        self.session_manager = Some(Arc::new(manager));
        self
    }

    /// Configures structured output: a JSON Schema for the desired result. When
    /// set, the agent offers a `strands_structured_output` tool with this schema,
    /// forces it if the model replies with plain text, and returns the captured
    /// value in [`crate::AgentResult::structured_output`].
    pub fn structured_output_schema(mut self, schema: serde_json::Value) -> Self {
        self.structured_output_schema = Some(schema);
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

    /// Registers a [`ToolProvider`] whose tools are loaded lazily at the start of
    /// the first invocation.
    pub fn tool_provider(mut self, provider: impl ToolProvider + 'static) -> Self {
        self.tool_providers.push(Arc::new(provider));
        self
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

    /// Registers an asynchronous hook callback for event type `E`, fired at the
    /// default order. The callback returns a boxed future that may borrow the
    /// event across `await` points.
    pub fn hook_async<E, F>(self, callback: F) -> Self
    where
        E: HookEvent + 'static,
        F: for<'a> Fn(&'a mut E) -> HookFuture<'a> + Send + Sync + 'static,
    {
        self.hooks.add_callback_async(callback);
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

    /// Registers all of a [`HookProvider`]'s callbacks as one unit.
    pub fn hook_provider(self, provider: impl HookProvider) -> Self {
        provider.register_hooks(&self.hooks);
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
            self.name.unwrap_or_else(|| DEFAULT_AGENT_NAME.to_string()),
            self.system_prompt,
            self.messages,
            self.tool_registry,
            self.tool_providers,
            self.hooks,
            self.state.unwrap_or_default(),
            self.conversation_manager,
            self.session_manager,
            self.structured_output_schema,
        );
        let mut initialized = InitializedEvent {
            agent: agent.agent_handle(),
        };
        // `try_build` is synchronous; sync init hooks resolve as ready futures so
        // this needs no runtime. A genuinely-async init hook (unusual) runs to
        // completion here on the current thread.
        futures::executor::block_on(agent.hooks().invoke_callbacks(&mut initialized))?;
        Ok(agent)
    }
}
