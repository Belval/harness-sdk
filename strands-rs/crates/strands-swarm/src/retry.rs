//! Which failed model calls an agent retries, and for how long. The SDK retries only throttling,
//! six attempts in all (about a minute of backoff); a swarm runs unattended for hours, so its
//! agents also retry server errors and dropped connections, and keep at it for several minutes
//! before a turn fails.

use std::sync::Arc;

use strands_agents::retry::DefaultModelRetryStrategyOptions;
use strands_agents::Error;

/// Model attempts per call: with the SDK's backoff (4 s doubling, jittered, capped at 4 min)
/// about eight minutes of retrying.
pub const MAX_ATTEMPTS: u32 = 10;

/// Error texts of a failure worth retrying: Bedrock's 5xx exceptions (their type name is in the
/// source error's text), HTTP 5xx and overload statuses, and transport failures.
const TRANSIENT: &[&str] = &[
    "internalserverexception",
    "serviceunavailableexception",
    "modelstreamerrorexception",
    "modelnotreadyexception",
    "try your request again",
    "status 500",
    "status 502",
    "status 503",
    "status 504",
    "status 529",
    "overloaded",
    "dispatch failure",
    "response error",
    "timeout",
    "timed out",
    "connection",
    "broken pipe",
];

/// Error texts that make a failure permanent even if it also reads as transient: a request the
/// service will always reject, or missing credentials (the provider's hint names them).
const PERMANENT: &[&str] = &["validationexception", "accessdenied", "credentials"];

/// The retry policy for every swarm agent's model calls.
pub fn policy() -> DefaultModelRetryStrategyOptions {
    DefaultModelRetryStrategyOptions {
        max_attempts: MAX_ATTEMPTS,
        retryable: Arc::new(is_transient),
        ..Default::default()
    }
}

/// Whether `error` is likely to go away on its own: throttling, or a model error that reads as a
/// server or transport failure.
pub fn is_transient(error: &Error) -> bool {
    match error {
        Error::ModelThrottled { .. } => true,
        Error::Model { .. } => {
            let text = chain(error).to_lowercase();
            !PERMANENT.iter().any(|m| text.contains(m)) && TRANSIENT.iter().any(|m| text.contains(m))
        }
        _ => false,
    }
}

/// `error` and every source under it, joined.
pub(crate) fn chain(error: &(dyn std::error::Error + 'static)) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Cause(&'static str);

    impl std::fmt::Display for Cause {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.0)
        }
    }

    impl std::error::Error for Cause {}

    fn with_cause(message: &str, cause: &'static str) -> Error {
        Error::Model {
            message: message.into(),
            source: Some(Box::new(Cause(cause))),
        }
    }

    #[test]
    fn server_and_transport_failures_are_transient() {
        assert!(is_transient(&Error::throttled("slow down")));
        assert!(is_transient(&with_cause(
            "Bedrock is unable to process your request.",
            "ServiceUnavailableException: Bedrock is unable to process your request."
        )));
        assert!(is_transient(&with_cause(
            "unhandled error",
            "response error: connection reset"
        )));
        assert!(is_transient(&Error::model(
            "The system encountered an unexpected error during processing. Try your request again."
        )));
        assert!(is_transient(&Error::model(
            "anthropic request failed with status 529"
        )));
    }

    #[test]
    fn rejected_requests_and_other_errors_are_not() {
        assert!(!is_transient(&with_cause(
            "A conversation must alternate between user and assistant roles.",
            "ValidationException: A conversation must alternate between user and assistant roles."
        )));
        assert!(!is_transient(&Error::model(
            "No AWS credentials found (dispatch failure). Run `aws configure`."
        )));
        assert!(!is_transient(&Error::model(
            "The provided model identifier is invalid."
        )));
        assert!(!is_transient(&Error::ContextWindowOverflow("too long".into())));
        assert!(!is_transient(&Error::other("timeout")));
    }
}
