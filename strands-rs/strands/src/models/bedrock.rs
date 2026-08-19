//! AWS Bedrock model provider. Ports `models/bedrock.ts`.
//!
//! Uses the Bedrock Converse Stream API via the AWS Rust SDK. This slice ports
//! the request formatting (messages, system prompt, tools, inference config),
//! the streamed-event mapping, and prompt caching; guardrails, citations, and
//! native token counting from the TypeScript provider are out of scope.

use async_stream::stream;
use aws_sdk_bedrockruntime::types as brt;
use aws_sdk_bedrockruntime::Client;

use crate::errors::StrandsError;
use crate::models::{CacheStrategy, Model, ModelEventStream, StreamOptions};
use crate::types::messages::{
    ContentBlock, Message, StopReason, SystemContentBlock, ToolResultContent,
};
use crate::types::streaming::{ContentBlockDelta, Metrics, ModelStreamEvent, ToolUseStart, Usage};
use crate::types::tools::{ToolChoice, ToolSpec};

/// Default model ID used when none is configured. Mirrors the TypeScript SDK's
/// Bedrock default.
const DEFAULT_MODEL_ID: &str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0";

/// Model-ID substrings that support Anthropic-style prompt caching, used to
/// auto-detect when `cacheConfig.strategy` is `Auto`. Ports
/// `MODELS_SUPPORTING_ANTHROPIC_CACHING`.
const MODELS_SUPPORTING_ANTHROPIC_CACHING: &[&str] = &["anthropic", "claude"];

/// Substrings that identify a Bedrock context-window-overflow error, mapped to
/// [`StrandsError::ContextWindowOverflow`]. Mirrors `BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES`.
const CONTEXT_WINDOW_OVERFLOW_MESSAGES: &[&str] = &[
    "Input is too long for requested model",
    "input length and `max_tokens` exceed context limit",
    "too many total text bytes",
];

/// Prompt-caching configuration for the Bedrock provider. Ports
/// `BedrockCacheConfig`.
///
/// TTLs are provider strings (`"5m"`, `"1h"`, or any value Bedrock accepts) and
/// must be non-increasing across tools → system → messages, per the Converse API.
#[derive(Debug, Clone)]
pub struct BedrockCacheConfig {
    /// Whether to auto-detect caching support or force it on.
    pub strategy: CacheStrategy,
    /// TTL for the cache point appended after the tool definitions.
    pub tools_ttl: Option<String>,
    /// TTL for the cache point injected into the last user message.
    pub messages_ttl: Option<String>,
}

impl BedrockCacheConfig {
    /// Creates a config with the given strategy and no explicit TTLs.
    pub fn new(strategy: CacheStrategy) -> Self {
        BedrockCacheConfig {
            strategy,
            tools_ttl: None,
            messages_ttl: None,
        }
    }

    /// Sets the tools cache-point TTL.
    pub fn with_tools_ttl(mut self, ttl: impl Into<String>) -> Self {
        self.tools_ttl = Some(ttl.into());
        self
    }

    /// Sets the messages cache-point TTL.
    pub fn with_messages_ttl(mut self, ttl: impl Into<String>) -> Self {
        self.messages_ttl = Some(ttl.into());
        self
    }
}

/// AWS Bedrock implementation of [`Model`], using the Converse Stream API.
pub struct BedrockModel {
    client: Client,
    model_id: String,
    max_tokens: Option<i32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    cache_config: Option<BedrockCacheConfig>,
}

impl BedrockModel {
    /// Creates a Bedrock model with the given model ID, loading AWS config from
    /// the environment (the standard credential/region chain).
    pub async fn new(model_id: impl Into<String>) -> Self {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        BedrockModel {
            client: Client::new(&config),
            model_id: model_id.into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            cache_config: None,
        }
    }

    /// Creates a Bedrock model with the default model ID.
    pub async fn default_model() -> Self {
        BedrockModel::new(DEFAULT_MODEL_ID).await
    }

    /// Creates a Bedrock model from an existing client, without touching the
    /// environment. Useful for tests and custom credential setups.
    pub fn from_client(client: Client, model_id: impl Into<String>) -> Self {
        BedrockModel {
            client,
            model_id: model_id.into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            cache_config: None,
        }
    }

    /// Sets the maximum number of tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: i32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Enables prompt caching with the given configuration. Ports
    /// `BedrockModelConfig.cacheConfig`.
    pub fn with_cache_config(mut self, cache_config: BedrockCacheConfig) -> Self {
        self.cache_config = Some(cache_config);
        self
    }

    /// Whether prompt caching should be applied for this request. Ports
    /// `_shouldEnableCaching`, additionally warning when `Auto` is configured for
    /// a model that does not support automatic caching.
    fn should_enable_caching(&self) -> bool {
        let enabled = caching_enabled(self.cache_config.as_ref(), &self.model_id);
        if !enabled
            && matches!(
                self.cache_config.as_ref().map(|config| config.strategy),
                Some(CacheStrategy::Auto)
            )
        {
            tracing::warn!(
                model_id = %self.model_id,
                "cache config is enabled but this model does not support automatic caching"
            );
        }
        enabled
    }

    /// Sets the sampling temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the nucleus-sampling `top_p`.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    fn build_tool_config(
        &self,
        tool_specs: &[ToolSpec],
        tool_choice: Option<&ToolChoice>,
    ) -> Option<brt::ToolConfiguration> {
        if tool_specs.is_empty() {
            return None;
        }
        let mut tools = Vec::new();
        for spec in tool_specs {
            let schema = spec
                .input_schema
                .clone()
                .unwrap_or_else(|| serde_json::json!({ "type": "object" }));
            let tool_spec = brt::ToolSpecification::builder()
                .name(&spec.name)
                .description(&spec.description)
                .input_schema(brt::ToolInputSchema::Json(json_to_document(&schema)))
                .build()
                .expect("tool name and input schema are always set");
            tools.push(brt::Tool::ToolSpec(tool_spec));
        }

        // Cache point after the tool definitions, so the tools prefix is cached.
        if self.should_enable_caching() {
            let ttl = self
                .cache_config
                .as_ref()
                .and_then(|config| config.tools_ttl.as_deref());
            tools.push(brt::Tool::CachePoint(bedrock_cache_point("default", ttl)));
        }

        let mut builder = brt::ToolConfiguration::builder().set_tools(Some(tools));
        if let Some(choice) = tool_choice {
            builder = builder.tool_choice(match choice {
                ToolChoice::Auto => brt::ToolChoice::Auto(brt::AutoToolChoice::builder().build()),
                ToolChoice::Any => brt::ToolChoice::Any(brt::AnyToolChoice::builder().build()),
                ToolChoice::Tool { name } => brt::ToolChoice::Tool(
                    brt::SpecificToolChoice::builder()
                        .name(name)
                        .build()
                        .expect("tool name is set"),
                ),
            });
        }
        builder.build().ok()
    }
}

#[async_trait::async_trait]
impl Model for BedrockModel {
    fn model_id(&self) -> Option<&str> {
        Some(&self.model_id)
    }

    fn get_config(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut config = serde_json::Map::new();
        config.insert(
            "model_id".to_string(),
            serde_json::Value::String(self.model_id.clone()),
        );
        if let Some(max_tokens) = self.max_tokens {
            config.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
        }
        if let Some(temperature) = self.temperature {
            config.insert("temperature".to_string(), serde_json::json!(temperature));
        }
        if let Some(top_p) = self.top_p {
            config.insert("top_p".to_string(), serde_json::json!(top_p));
        }
        config
    }

    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        options: &'a StreamOptions,
    ) -> ModelEventStream<'a> {
        Box::pin(stream! {
            let mut message_contents = match format_message_contents(messages) {
                Ok(contents) => contents,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };

            // Auto mode manages cache points itself: strip any manual ones and
            // inject after the tools and into the last user message.
            if self.should_enable_caching() {
                let ttl = self.cache_config.as_ref().and_then(|config| config.messages_ttl.as_deref());
                inject_cache_point(&mut message_contents, ttl);
            }

            let bedrock_messages = match build_messages(message_contents) {
                Ok(messages) => messages,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };

            let mut request = self.client
                .converse_stream()
                .model_id(&self.model_id)
                .set_messages(Some(bedrock_messages));

            if let Some(prompt) = &options.system_prompt {
                let system_blocks: Vec<brt::SystemContentBlock> = prompt
                    .blocks()
                    .into_iter()
                    .map(|block| match block {
                        SystemContentBlock::Text(text) => brt::SystemContentBlock::Text(text),
                        SystemContentBlock::CachePoint(cache_point) => brt::SystemContentBlock::CachePoint(
                            bedrock_cache_point(&cache_point.cache_type, cache_point.ttl.as_deref()),
                        ),
                    })
                    .collect();
                request = request.set_system(Some(system_blocks));
            }

            if let Some(tool_config) = self.build_tool_config(&options.tool_specs, options.tool_choice.as_ref()) {
                request = request.tool_config(tool_config);
            }

            let mut inference = brt::InferenceConfiguration::builder();
            if let Some(max_tokens) = self.max_tokens {
                inference = inference.max_tokens(max_tokens);
            }
            if let Some(temperature) = self.temperature {
                inference = inference.temperature(temperature);
            }
            if let Some(top_p) = self.top_p {
                inference = inference.top_p(top_p);
            }
            request = request.inference_config(inference.build());

            let mut response = match request.send().await {
                Ok(response) => response,
                Err(error) => {
                    let is_throttling = error.as_service_error().is_some_and(|service| service.is_throttling_exception());
                    yield Err(map_bedrock_error(&error, is_throttling, "bedrock converse stream request failed"));
                    return;
                }
            };

            loop {
                match response.stream.recv().await {
                    Ok(Some(output)) => {
                        for event in map_stream_output(output) {
                            yield Ok(event);
                        }
                    }
                    Ok(None) => break,
                    Err(error) => {
                        let is_throttling = error.as_service_error().is_some_and(|service| service.is_throttling_exception());
                        yield Err(map_bedrock_error(&error, is_throttling, "error receiving bedrock stream event"));
                        return;
                    }
                }
            }
        })
    }
}

/// Maps a Bedrock error into a typed [`StrandsError`].
///
/// Ports the vendor-error translation from `bedrock.ts`: throttling →
/// [`StrandsError::ModelThrottled`], context-window-overflow messages →
/// [`StrandsError::ContextWindowOverflow`], everything else → a generic model
/// error wrapping the cause. Applied on both the request and stream-receive
/// paths, matching the TypeScript provider's single outer catch that covers
/// both request-time and mid-stream failures.
fn map_bedrock_error<E>(error: &E, is_throttling: bool, context: &str) -> StrandsError
where
    E: std::error::Error,
{
    let message = format!(
        "{}",
        aws_smithy_types::error::display::DisplayErrorContext(error)
    );
    if is_throttling {
        return StrandsError::ModelThrottled {
            message,
            source: None,
        };
    }
    if CONTEXT_WINDOW_OVERFLOW_MESSAGES
        .iter()
        .any(|needle| message.contains(needle))
    {
        return StrandsError::ContextWindowOverflow(message);
    }
    StrandsError::model(format!("{context}: {message}"))
}

/// Maps one Bedrock `ConverseStreamOutput` chunk to zero or more SDK events.
///
/// Ports `_mapStreamedBedrockEventToSDKEvent`. Unknown or unhandled event types
/// yield nothing, matching the TypeScript default branch's warn-and-skip.
fn map_stream_output(output: brt::ConverseStreamOutput) -> Vec<ModelStreamEvent> {
    match output {
        brt::ConverseStreamOutput::MessageStart(event) => {
            let role = match event.role {
                brt::ConversationRole::Assistant => crate::types::messages::Role::Assistant,
                _ => crate::types::messages::Role::User,
            };
            vec![ModelStreamEvent::MessageStart { role }]
        }
        brt::ConverseStreamOutput::ContentBlockStart(event) => {
            let start = match event.start {
                Some(brt::ContentBlockStart::ToolUse(tool_use)) => Some(ToolUseStart {
                    name: tool_use.name,
                    tool_use_id: tool_use.tool_use_id,
                    reasoning_signature: None,
                }),
                _ => None,
            };
            vec![ModelStreamEvent::ContentBlockStart { start }]
        }
        brt::ConverseStreamOutput::ContentBlockDelta(event) => {
            let Some(delta) = event.delta else {
                return Vec::new();
            };
            match delta {
                brt::ContentBlockDelta::Text(text) => {
                    vec![ModelStreamEvent::ContentBlockDelta {
                        delta: ContentBlockDelta::Text(text),
                    }]
                }
                brt::ContentBlockDelta::ToolUse(tool_use) => {
                    vec![ModelStreamEvent::ContentBlockDelta {
                        delta: ContentBlockDelta::ToolUseInput(tool_use.input),
                    }]
                }
                brt::ContentBlockDelta::ReasoningContent(reasoning) => match reasoning {
                    brt::ReasoningContentBlockDelta::Text(text) => {
                        vec![ModelStreamEvent::ContentBlockDelta {
                            delta: ContentBlockDelta::Reasoning {
                                text: Some(text),
                                signature: None,
                                redacted_content: None,
                            },
                        }]
                    }
                    brt::ReasoningContentBlockDelta::Signature(signature) => {
                        vec![ModelStreamEvent::ContentBlockDelta {
                            delta: ContentBlockDelta::Reasoning {
                                text: None,
                                signature: Some(signature),
                                redacted_content: None,
                            },
                        }]
                    }
                    _ => Vec::new(),
                },
                _ => Vec::new(),
            }
        }
        brt::ConverseStreamOutput::ContentBlockStop(_) => vec![ModelStreamEvent::ContentBlockStop],
        brt::ConverseStreamOutput::MessageStop(event) => {
            vec![ModelStreamEvent::MessageStop {
                stop_reason: map_stop_reason(&event.stop_reason),
            }]
        }
        brt::ConverseStreamOutput::Metadata(event) => {
            let usage = event.usage.map(|usage| Usage {
                input_tokens: usage.input_tokens.max(0) as u64,
                output_tokens: usage.output_tokens.max(0) as u64,
                total_tokens: usage.total_tokens.max(0) as u64,
                cache_read_input_tokens: usage
                    .cache_read_input_tokens
                    .map(|value| value.max(0) as u64),
                cache_write_input_tokens: usage
                    .cache_write_input_tokens
                    .map(|value| value.max(0) as u64),
            });
            let metrics = event.metrics.map(|metrics| Metrics {
                latency_ms: metrics.latency_ms.max(0) as u64,
                time_to_first_byte_ms: None,
            });
            vec![ModelStreamEvent::Metadata { usage, metrics }]
        }
        _ => Vec::new(),
    }
}

/// Maps a Bedrock stop reason to the SDK's [`StopReason`]. Ports `STOP_REASON_MAP`
/// and the `_transformStopReason` fallback (unknown values pass through as-is).
fn map_stop_reason(reason: &brt::StopReason) -> StopReason {
    match reason {
        brt::StopReason::EndTurn => StopReason::EndTurn,
        brt::StopReason::ToolUse => StopReason::ToolUse,
        brt::StopReason::MaxTokens => StopReason::MaxTokens,
        brt::StopReason::StopSequence => StopReason::StopSequence,
        brt::StopReason::ContentFiltered => StopReason::ContentFiltered,
        brt::StopReason::GuardrailIntervened => StopReason::GuardrailIntervened,
        other => StopReason::from_wire(other.as_str()),
    }
}

/// Formats SDK messages into `(role, content)` pairs. Ports `_formatMessages` /
/// `_formatContentBlock`. Empty messages are dropped, matching the TypeScript
/// `content.length > 0` guard. Kept separate from [`build_messages`] so cache
/// points can be injected into the content vecs before the messages are built.
fn format_message_contents(
    messages: &[Message],
) -> Result<Vec<(brt::ConversationRole, Vec<brt::ContentBlock>)>, StrandsError> {
    let mut formatted = Vec::new();
    for message in messages {
        let mut content = Vec::new();
        for block in &message.content {
            if let Some(bedrock_block) = format_content_block(block)? {
                content.push(bedrock_block);
            }
        }
        if content.is_empty() {
            continue;
        }
        let role = match message.role {
            crate::types::messages::Role::User => brt::ConversationRole::User,
            crate::types::messages::Role::Assistant => brt::ConversationRole::Assistant,
        };
        formatted.push((role, content));
    }
    Ok(formatted)
}

/// Builds Bedrock `Message`s from `(role, content)` pairs.
fn build_messages(
    pairs: Vec<(brt::ConversationRole, Vec<brt::ContentBlock>)>,
) -> Result<Vec<brt::Message>, StrandsError> {
    let mut messages = Vec::with_capacity(pairs.len());
    for (role, content) in pairs {
        let message = brt::Message::builder()
            .role(role)
            .set_content(Some(content))
            .build()
            .map_err(|error| {
                StrandsError::model_with_source("failed to build bedrock message", error)
            })?;
        messages.push(message);
    }
    Ok(messages)
}

/// Whether prompt caching should be applied. Ports `_shouldEnableCaching`'s pure
/// decision: `Anthropic` forces it on; `Auto` enables it only when the model id
/// is known to support Anthropic-style caching.
fn caching_enabled(cache_config: Option<&BedrockCacheConfig>, model_id: &str) -> bool {
    match cache_config {
        None => false,
        Some(config) => match config.strategy {
            CacheStrategy::Anthropic => true,
            CacheStrategy::Auto => MODELS_SUPPORTING_ANTHROPIC_CACHING
                .iter()
                .any(|pattern| model_id.contains(pattern)),
        },
    }
}

/// Builds a Bedrock `CachePointBlock` of type `cache_type` with an optional TTL.
fn bedrock_cache_point(cache_type: &str, ttl: Option<&str>) -> brt::CachePointBlock {
    let mut builder = brt::CachePointBlock::builder().r#type(brt::CachePointType::from(cache_type));
    if let Some(ttl) = ttl {
        // Bedrock validates TTL values server-side, so any string is accepted.
        builder = builder.ttl(brt::CacheTtl::from(ttl));
    }
    builder.build().expect("cache point type is always set")
}

/// Strips any existing cache points and injects one into the last user message.
/// Ports `_injectCachePoint`: auto mode manages cache points itself.
///
/// The cache point is placed before the first non-PDF document block (which
/// Bedrock rejects as the block directly preceding a cache point); if such a
/// block leads the message there is no cacheable prefix, so injection is skipped.
fn inject_cache_point(
    messages: &mut [(brt::ConversationRole, Vec<brt::ContentBlock>)],
    ttl: Option<&str>,
) {
    let mut last_user_idx: Option<usize> = None;
    for (index, (role, content)) in messages.iter_mut().enumerate() {
        content.retain(|block| !matches!(block, brt::ContentBlock::CachePoint(_)));
        if *role == brt::ConversationRole::User {
            last_user_idx = Some(index);
        }
    }

    let Some(index) = last_user_idx else {
        return;
    };
    let content = &mut messages[index].1;
    let cache_point = brt::ContentBlock::CachePoint(bedrock_cache_point("default", ttl));

    let first_non_pdf_document = content.iter().position(|block| {
        matches!(block, brt::ContentBlock::Document(document) if *document.format() != brt::DocumentFormat::Pdf)
    });

    match first_non_pdf_document {
        None => content.push(cache_point),
        Some(0) => {
            tracing::debug!(
                msg_idx = index,
                "skipped cache point for leading non-pdf document"
            );
        }
        Some(position) => content.insert(position, cache_point),
    }
}

fn format_content_block(block: &ContentBlock) -> Result<Option<brt::ContentBlock>, StrandsError> {
    match block {
        ContentBlock::Text(text) => Ok(Some(brt::ContentBlock::Text(text.clone()))),
        ContentBlock::ToolUse(tool_use) => {
            let bedrock_tool_use = brt::ToolUseBlock::builder()
                .tool_use_id(&tool_use.tool_use_id)
                .name(&tool_use.name)
                .input(json_to_document(&tool_use.input))
                .build()
                .map_err(|error| {
                    StrandsError::model_with_source("failed to build tool use block", error)
                })?;
            Ok(Some(brt::ContentBlock::ToolUse(bedrock_tool_use)))
        }
        ContentBlock::ToolResult(tool_result) => {
            let mut result_content = Vec::new();
            for item in &tool_result.content {
                match item {
                    ToolResultContent::Text(text) => {
                        result_content.push(brt::ToolResultContentBlock::Text(text.clone()));
                    }
                    ToolResultContent::Json(value) => {
                        result_content
                            .push(brt::ToolResultContentBlock::Json(json_to_document(value)));
                    }
                    // Media tool-result content is out of scope for the slice.
                    _ => {}
                }
            }
            let status = match tool_result.status {
                crate::types::messages::ToolResultStatus::Success => brt::ToolResultStatus::Success,
                crate::types::messages::ToolResultStatus::Error => brt::ToolResultStatus::Error,
            };
            let bedrock_result = brt::ToolResultBlock::builder()
                .tool_use_id(&tool_result.tool_use_id)
                .set_content(Some(result_content))
                .status(status)
                .build()
                .map_err(|error| {
                    StrandsError::model_with_source("failed to build tool result block", error)
                })?;
            Ok(Some(brt::ContentBlock::ToolResult(bedrock_result)))
        }
        // A manually-placed cache point passes through; `type` comes from `cache_type`.
        ContentBlock::CachePoint(cache_point) => Ok(Some(brt::ContentBlock::CachePoint(
            bedrock_cache_point(&cache_point.cache_type, cache_point.ttl.as_deref()),
        ))),
        // Reasoning and media blocks are not sent back in the slice.
        _ => Ok(None),
    }
}

/// Converts a `serde_json::Value` into an AWS Smithy `Document`.
fn json_to_document(value: &serde_json::Value) -> aws_smithy_types::Document {
    use aws_smithy_types::{Document, Number};
    match value {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(boolean) => Document::Bool(*boolean),
        serde_json::Value::Number(number) => {
            if let Some(unsigned) = number.as_u64() {
                Document::Number(Number::PosInt(unsigned))
            } else if let Some(signed) = number.as_i64() {
                Document::Number(Number::NegInt(signed))
            } else {
                Document::Number(Number::Float(number.as_f64().unwrap_or(0.0)))
            }
        }
        serde_json::Value::String(text) => Document::String(text.clone()),
        serde_json::Value::Array(items) => {
            Document::Array(items.iter().map(json_to_document).collect())
        }
        serde_json::Value::Object(map) => Document::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), json_to_document(value)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    //! Ports the Bedrock prompt-caching specs from `models/__tests__/bedrock*`:
    //! the caching-enabled decision, cache-point wire mapping, and auto-mode
    //! cache-point injection into the last user message.

    use super::*;

    fn user(content: Vec<brt::ContentBlock>) -> (brt::ConversationRole, Vec<brt::ContentBlock>) {
        (brt::ConversationRole::User, content)
    }

    fn assistant(
        content: Vec<brt::ContentBlock>,
    ) -> (brt::ConversationRole, Vec<brt::ContentBlock>) {
        (brt::ConversationRole::Assistant, content)
    }

    fn is_cache_point(block: &brt::ContentBlock) -> bool {
        matches!(block, brt::ContentBlock::CachePoint(_))
    }

    // "_shouldEnableCaching": auto enables only for supported models; anthropic forces on
    #[test]
    fn caching_enabled_decision() {
        assert!(!caching_enabled(None, "anthropic.claude-3"));

        let auto = BedrockCacheConfig::new(CacheStrategy::Auto);
        assert!(caching_enabled(
            Some(&auto),
            "us.anthropic.claude-sonnet-4-5"
        ));
        assert!(caching_enabled(Some(&auto), "some-claude-model"));
        assert!(!caching_enabled(Some(&auto), "amazon.titan-text"));

        // Explicit anthropic strategy forces caching on even for an unrecognized id.
        let forced = BedrockCacheConfig::new(CacheStrategy::Anthropic);
        assert!(caching_enabled(Some(&forced), "custom.inference-profile"));
    }

    // cache point maps cacheType -> type and TTL strings to CacheTtl
    #[test]
    fn cache_point_wire_mapping() {
        let default_point = bedrock_cache_point("default", None);
        assert_eq!(default_point.r#type(), &brt::CachePointType::Default);
        assert!(default_point.ttl().is_none());

        assert_eq!(
            bedrock_cache_point("default", Some("5m"))
                .ttl()
                .unwrap()
                .as_str(),
            "5m"
        );
        assert_eq!(
            bedrock_cache_point("default", Some("1h"))
                .ttl()
                .unwrap()
                .as_str(),
            "1h"
        );
        // Bedrock validates server-side, so any TTL string is preserved.
        assert_eq!(
            bedrock_cache_point("default", Some("2h"))
                .ttl()
                .unwrap()
                .as_str(),
            "2h"
        );
    }

    // "_injectCachePoint": appends a cache point to the last user message
    #[test]
    fn injects_cache_point_into_last_user_message() {
        let mut messages = vec![
            user(vec![brt::ContentBlock::Text("first".to_string())]),
            assistant(vec![brt::ContentBlock::Text("reply".to_string())]),
            user(vec![brt::ContentBlock::Text("second".to_string())]),
        ];
        inject_cache_point(&mut messages, None);

        // Only the last user message gains a trailing cache point.
        assert!(!messages[0].1.iter().any(is_cache_point));
        assert!(is_cache_point(messages[2].1.last().unwrap()));
        assert_eq!(messages[2].1.len(), 2);
    }

    // "_injectCachePoint": strips any manually-placed cache points first (auto mode manages them)
    #[test]
    fn strips_existing_cache_points_before_injecting() {
        let mut messages = vec![user(vec![
            brt::ContentBlock::Text("hi".to_string()),
            brt::ContentBlock::CachePoint(bedrock_cache_point("default", None)),
        ])];
        inject_cache_point(&mut messages, Some("5m"));

        // Exactly one cache point remains — the freshly injected one, carrying the TTL.
        let cache_points: Vec<_> = messages[0]
            .1
            .iter()
            .filter(|block| is_cache_point(block))
            .collect();
        assert_eq!(cache_points.len(), 1);
        let brt::ContentBlock::CachePoint(point) = cache_points[0] else {
            unreachable!();
        };
        assert_eq!(point.ttl().unwrap().as_str(), "5m");
    }

    // "_injectCachePoint": no user message means nothing to cache
    #[test]
    fn injects_nothing_without_a_user_message() {
        let mut messages = vec![assistant(vec![brt::ContentBlock::Text("only".to_string())])];
        inject_cache_point(&mut messages, None);
        assert!(!messages[0].1.iter().any(is_cache_point));
    }

    // manual cache point passes through _formatContentBlock with type from cacheType
    #[test]
    fn format_content_block_passes_through_cache_point() {
        let block =
            ContentBlock::CachePoint(crate::types::messages::CachePointBlock::with_ttl("1h"));
        let formatted = format_content_block(&block).unwrap().unwrap();
        let brt::ContentBlock::CachePoint(point) = formatted else {
            panic!("expected a cache point block");
        };
        assert_eq!(point.r#type(), &brt::CachePointType::Default);
        assert_eq!(point.ttl().unwrap().as_str(), "1h");
    }
}
