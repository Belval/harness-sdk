//! A [`Tool`] that wraps a boxed async closure. Ports `tools/function-tool.ts`.

use std::future::Future;
use std::pin::Pin;

use async_trait::async_trait;

use crate::errors::StrandsError;
use crate::tools::{Tool, ToolContext};
use crate::types::tools::ToolSpec;

type ToolFuture = Pin<Box<dyn Future<Output = Result<serde_json::Value, StrandsError>> + Send>>;
type ToolCallback = Box<dyn Fn(ToolContext) -> ToolFuture + Send + Sync>;

/// A [`Tool`] implementation backed by a callback.
///
/// Lets a tool be created from an existing async function without hand-writing a
/// `Tool` impl, mirroring the TypeScript `FunctionTool`. The `#[tool]` macro
/// generates a dedicated `Tool` struct instead; `FunctionTool` is the
/// programmatic, macro-free path.
pub struct FunctionTool {
    name: String,
    description: String,
    input_schema: Option<serde_json::Value>,
    output_schema: Option<serde_json::Value>,
    callback: ToolCallback,
}

impl FunctionTool {
    /// Creates a tool from a name, description, input schema, and callback.
    ///
    /// When `input_schema` is `None`, an empty-object schema is used, matching
    /// the TypeScript default.
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Option<serde_json::Value>,
        callback: F,
    ) -> Self
    where
        F: Fn(ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<serde_json::Value, StrandsError>> + Send + 'static,
    {
        FunctionTool {
            name: name.into(),
            description: description.into(),
            input_schema,
            output_schema: None,
            callback: Box::new(move |context| Box::pin(callback(context))),
        }
    }

    /// Creates a tool at runtime from a full [`ToolSpec`] and an async callback.
    ///
    /// The runtime counterpart to the compile-time `#[tool]` macro (which derives
    /// a spec from a function signature): here the caller supplies the spec
    /// explicitly. Ports the TypeScript `DecoratedFunctionTool(name, spec, callable, …)`
    /// runtime constructor.
    pub fn from_spec<F, Fut>(spec: ToolSpec, callback: F) -> Self
    where
        F: Fn(ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<serde_json::Value, StrandsError>> + Send + 'static,
    {
        FunctionTool {
            name: spec.name,
            description: spec.description,
            input_schema: spec.input_schema,
            output_schema: spec.output_schema,
            callback: Box::new(move |context| Box::pin(callback(context))),
        }
    }
}

#[async_trait]
impl Tool for FunctionTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn tool_spec(&self) -> ToolSpec {
        let input_schema = self.input_schema.clone().unwrap_or_else(|| {
            serde_json::json!({ "type": "object", "properties": {}, "additionalProperties": false })
        });
        ToolSpec {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: Some(input_schema),
            output_schema: self.output_schema.clone(),
        }
    }

    async fn invoke(&self, context: ToolContext) -> Result<serde_json::Value, StrandsError> {
        (self.callback)(context).await
    }
}
