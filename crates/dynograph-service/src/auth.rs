//! Pluggable request authentication.

use std::sync::Arc;

use axum::http::HeaderMap;

/// Caller identity extracted from request headers. Cheap to clone —
/// the inner `Arc<str>` is shared across requests for `NoAuth`.
#[derive(Debug, Clone)]
pub struct Identity(pub Arc<str>);

#[derive(Debug, thiserror::Error)]
#[error("authentication failed: {message}")]
pub struct AuthError {
    message: String,
}

impl AuthError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

pub trait AuthProvider: Send + Sync + 'static {
    fn authenticate(&self, headers: &HeaderMap) -> Result<Identity, AuthError>;
}

/// Open access — every request is authenticated as `"anonymous"`.
/// Holds a shared `Arc<str>` so per-request authentication doesn't
/// allocate.
#[derive(Debug, Clone)]
pub struct NoAuth {
    id: Arc<str>,
}

impl NoAuth {
    pub fn new() -> Self {
        Self {
            id: Arc::from("anonymous"),
        }
    }
}

impl Default for NoAuth {
    fn default() -> Self {
        Self::new()
    }
}

impl AuthProvider for NoAuth {
    fn authenticate(&self, _headers: &HeaderMap) -> Result<Identity, AuthError> {
        Ok(Identity(self.id.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_auth_always_authenticates_as_anonymous() {
        let provider = NoAuth::new();
        let id = provider.authenticate(&HeaderMap::new()).unwrap();
        assert_eq!(&*id.0, "anonymous");
    }

    #[test]
    fn no_auth_shares_identity_arc_across_calls() {
        let provider = NoAuth::new();
        let a = provider.authenticate(&HeaderMap::new()).unwrap();
        let b = provider.authenticate(&HeaderMap::new()).unwrap();
        assert!(
            Arc::ptr_eq(&a.0, &b.0),
            "Arc should be shared, not re-allocated"
        );
    }
}
