//! Client-side errors.
//!
//! The HTTP layer returns one of three classes:
//! - `Network`: no response (connection refused, DNS, timeout).
//! - `Http { status, body }`: server replied with a non-2xx; the
//!   plain-text body is preserved so callers can match on the
//!   honest reason ("missing Authorization", "schema evolution
//!   rejected: …", etc).
//! - `Decode`: 2xx but the body didn't deserialize to the expected
//!   type — wire-shape mismatch between client and server (the
//!   contract test in `tests/integration.rs` is what catches these
//!   in CI).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("http {status}: {body}")]
    Http {
        status: reqwest::StatusCode,
        body: String,
    },

    #[error("decode error: {0}")]
    Decode(#[from] serde_json::Error),
}

impl ClientError {
    /// HTTP status when this error was produced from a non-2xx
    /// response. `Some(s)` if the server responded with `s`; `None`
    /// for `Network` and `Decode`.
    pub fn status(&self) -> Option<reqwest::StatusCode> {
        match self {
            Self::Http { status, .. } => Some(*status),
            _ => None,
        }
    }
}
