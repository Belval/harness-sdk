//! Bedrock Mantle: an OpenAI-compatible Bedrock endpoint (`strands-ts/src/models/openai/mantle.ts`).

use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_sigv4::http_request::{sign, SignableBody, SignableRequest, SignatureLocation, SigningSettings};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use base64::Engine;

use crate::error::{Error, Result};

/// Model id prefixes served under the `/openai/v1` base path.
const OPENAI_BASE_PATH_PREFIXES: &[&str] = &[
    "openai.gpt-5.",
    "openai.gpt-6-",
    "openai.gpt-6.",
    "xai.grok-4.",
    "google.gemma-4-",
];

/// Configuration for talking to Bedrock Mantle instead of api.openai.com.
#[derive(Clone, Debug, Default)]
pub struct BedrockMantleConfig {
    /// AWS region; falls back to `AWS_REGION` / `AWS_DEFAULT_REGION`, and with
    /// [`OpenAIModel::connect`](super::OpenAIModel::connect) to the profile's and the AWS chain's.
    pub region: Option<String>,
    /// Lifetime requested for the minted bearer token (default 12 hours).
    pub expires_in_seconds: Option<u64>,
    /// Profile for the credentials and, without `region`, the region, in place of `AWS_PROFILE`'s;
    /// neither access keys nor `AWS_REGION` in the environment override it. Used by
    /// [`OpenAIModel::connect`](super::OpenAIModel::connect).
    pub profile: Option<String>,
    /// Config file in place of `AWS_CONFIG_FILE`'s. Used by `connect`.
    pub config_file: Option<PathBuf>,
    /// Shared credentials file in place of `AWS_SHARED_CREDENTIALS_FILE`'s. Used by `connect`.
    pub credentials_file: Option<PathBuf>,
}

/// How long a minted token lives when the config names no lifetime.
const DEFAULT_TOKEN_LIFETIME: Duration = Duration::from_secs(12 * 60 * 60);

/// What mints a Mantle bearer token for each request: the AWS credentials of the chain (or
/// profile) the model was connected with, which refresh themselves.
#[derive(Clone, Debug)]
pub(crate) struct MantleAuth {
    credentials: SharedCredentialsProvider,
    region: String,
    lifetime: Duration,
}

impl MantleAuth {
    /// Mint a token: a SigV4-presigned `CallWithBearerToken` URL, as `aws-bedrock-token-generator`.
    pub(crate) async fn token(&self) -> Result<String> {
        let fail = |e: &dyn std::fmt::Display| {
            Error::model(format!("failed to mint Bedrock Mantle bearer token | {e}"))
        };
        let credentials = self
            .credentials
            .provide_credentials()
            .await
            .map_err(|e| fail(&e))?;
        let identity = Identity::from(credentials);
        let mut settings = SigningSettings::default();
        settings.signature_location = SignatureLocation::QueryParams;
        settings.expires_in = Some(self.lifetime);
        let params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name("bedrock")
            .time(SystemTime::now())
            .settings(settings)
            .build()
            .map_err(|e| fail(&e))?
            .into();
        const HOST: &str = "bedrock.amazonaws.com";
        let url = format!("https://{HOST}/?Action=CallWithBearerToken");
        let request = SignableRequest::new(
            "POST",
            url.as_str(),
            std::iter::once(("host", HOST)),
            SignableBody::Bytes(&[]),
        )
        .map_err(|e| fail(&e))?;
        let (instructions, _) = sign(request, &params).map_err(|e| fail(&e))?.into_parts();
        let mut signed = http::Request::builder()
            .method("POST")
            .uri(url.as_str())
            .body(())
            .map_err(|e| fail(&e))?;
        instructions.apply_to_request_http1x(&mut signed);
        let signed = signed.uri().to_string();
        let signed = signed.strip_prefix("https://").unwrap_or(&signed);
        let encoded = base64::engine::general_purpose::STANDARD.encode(format!("{signed}&Version=1"));
        Ok(format!("bedrock-api-key-{encoded}"))
    }
}

/// Resolve what `config` leaves to the AWS chain, loaded with its profile and files in place of
/// the environment's: the region when unset (the profile's, else the environment's, else the
/// chain's), and the credentials to mint tokens from. `None` when a token comes from elsewhere:
/// `has_api_key`, or `AWS_BEARER_TOKEN_BEDROCK` with no profile named.
pub(crate) async fn connect(
    config: &mut BedrockMantleConfig,
    has_api_key: bool,
) -> Result<Option<MantleAuth>> {
    let env_token = config.profile.is_none() && std::env::var_os("AWS_BEARER_TOKEN_BEDROCK").is_some();
    if has_api_key || env_token && config.region.is_some() {
        return Ok(None);
    }
    let sdk = super::super::aws::aws_loader(
        config.profile.as_deref(),
        config.config_file.as_deref(),
        config.credentials_file.as_deref(),
    )
    .load()
    .await;
    if config.region.is_none() {
        config.region = sdk.region().map(|region| region.to_string());
    }
    if env_token {
        return Ok(None);
    }
    let region = resolve_region(config)?;
    let credentials = sdk.credentials_provider().ok_or_else(|| {
        Error::model("failed to mint Bedrock Mantle bearer token | no AWS credentials provider found")
    })?;
    let auth = MantleAuth {
        credentials,
        region,
        lifetime: config
            .expires_in_seconds
            .map_or(DEFAULT_TOKEN_LIFETIME, Duration::from_secs),
    };
    // Fail now, not on the first request, when there are no credentials.
    auth.token().await?;
    Ok(Some(auth))
}

/// `/openai/v1` for the OpenAI-shaped model families, `/v1` otherwise.
pub fn resolve_mantle_base_path(model_id: &str) -> &'static str {
    if OPENAI_BASE_PATH_PREFIXES
        .iter()
        .any(|prefix| model_id.starts_with(prefix))
    {
        "/openai/v1"
    } else {
        "/v1"
    }
}

/// `https://bedrock-mantle.<region>.api.aws<base path>`, validating the region.
pub fn resolve_mantle_base_url(config: &BedrockMantleConfig, model_id: &str) -> Result<String> {
    let region = resolve_region(config)?;
    Ok(format!(
        "https://bedrock-mantle.{region}.api.aws{}",
        resolve_mantle_base_path(model_id)
    ))
}

fn resolve_region(config: &BedrockMantleConfig) -> Result<String> {
    let region = config
        .region
        .clone()
        .or_else(|| std::env::var("AWS_REGION").ok())
        .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok())
        .ok_or_else(|| {
            Error::model(
                "could not resolve an AWS region for Bedrock Mantle. Set 'region' in bedrockMantleConfig or the AWS_REGION environment variable.",
            )
        })?;
    if !is_valid_region(&region) {
        return Err(Error::model(format!("invalid AWS region: '{region}'")));
    }
    Ok(region)
}

/// Anchored equivalent of the TS `/^[a-z]{2}(-[a-z]+)+-[0-9]+$/` region check.
fn is_valid_region(region: &str) -> bool {
    let mut parts = region.split('-');
    let Some(first) = parts.next() else {
        return false;
    };
    if first.len() != 2 || !first.chars().all(|c| c.is_ascii_lowercase()) {
        return false;
    }
    let rest: Vec<&str> = parts.collect();
    if rest.len() < 2 {
        return false;
    }
    let (last, middle) = rest.split_last().expect("rest is non-empty");
    if last.is_empty() || !last.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    middle
        .iter()
        .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_lowercase()))
}

/// Mint a Mantle bearer token.
///
/// Gap vs TS: the TS provider lazily imports `@aws/bedrock-token-generator` and mints a fresh
/// short-lived token per request. No Rust equivalent exists yet, so the token must be supplied
/// out of band via `api_key` or `AWS_BEARER_TOKEN_BEDROCK`, unless the model is built with
/// [`OpenAIModel::connect`](super::OpenAIModel::connect), which mints from AWS credentials.
pub(crate) fn mint_bearer_token() -> Result<Option<String>> {
    std::env::var("AWS_BEARER_TOKEN_BEDROCK").map(Some).map_err(|_| {
        Error::model(
            "failed to mint Bedrock Mantle bearer token | OpenAIModel::new does not mint one: set AWS_BEARER_TOKEN_BEDROCK, pass api_key, or build the model with OpenAIModel::connect",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_openai_base_path_for_known_prefixes() {
        for id in [
            "openai.gpt-5.4",
            "openai.gpt-6-preview",
            "openai.gpt-6.1",
            "xai.grok-4.1",
            "google.gemma-4-12b",
        ] {
            assert_eq!(resolve_mantle_base_path(id), "/openai/v1", "{id}");
        }
    }

    #[test]
    fn falls_back_to_v1_base_path() {
        for id in ["anthropic.claude-sonnet-4-6", "openai.gpt-4.1", "meta.llama4"] {
            assert_eq!(resolve_mantle_base_path(id), "/v1", "{id}");
        }
    }

    #[test]
    fn builds_base_url_from_region_and_model() {
        let config = BedrockMantleConfig {
            region: Some("us-west-2".into()),
            ..Default::default()
        };
        assert_eq!(
            resolve_mantle_base_url(&config, "openai.gpt-5.4").unwrap(),
            "https://bedrock-mantle.us-west-2.api.aws/openai/v1"
        );
        assert_eq!(
            resolve_mantle_base_url(&config, "anthropic.claude-sonnet-4-6").unwrap(),
            "https://bedrock-mantle.us-west-2.api.aws/v1"
        );
    }

    #[test]
    fn rejects_invalid_regions() {
        for region in [
            "US-WEST-2",
            "us-west",
            "uswest2",
            "us-west-2/../evil",
            "us-west-2.example.com",
            "u-west-2",
        ] {
            let config = BedrockMantleConfig {
                region: Some(region.into()),
                ..Default::default()
            };
            let err = resolve_mantle_base_url(&config, "openai.gpt-5.4").unwrap_err();
            assert_eq!(err.to_string(), format!("invalid AWS region: '{region}'"));
        }
    }

    #[test]
    fn accepts_multi_segment_regions() {
        assert!(is_valid_region("ap-southeast-1"));
        assert!(is_valid_region("us-gov-west-1"));
    }
}
