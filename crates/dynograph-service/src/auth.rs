//! Pluggable authentication.
//!
//! v0.3.0 ships only `NoAuth`. `BearerJwt` lands in a follow-up slice
//! and will populate `Identity` with a real user_id from the validated
//! token. `Identity` stays a thin wrapper for now — when JWT extends it,
//! the trait shape doesn't break.

use axum::http::HeaderMap;

/// Caller identity extracted from request headers.
#[derive(Debug, Clone)]
pub struct Identity(pub String);

/// Authentication failure (bad token, missing header, etc.).
#[derive(Debug, thiserror::Error)]
#[error("authentication failed: {0}")]
pub struct AuthError(pub String);

/// Authenticates an incoming request. Object-safe so `AppState` can hold
/// `Arc<dyn AuthProvider>` and swap impls at construction.
pub trait AuthProvider: Send + Sync + 'static {
    fn authenticate(&self, headers: &HeaderMap) -> Result<Identity, AuthError>;
}

/// Open access — every request is authenticated as `"anonymous"`.
/// For dev / private-network deployments.
#[derive(Debug, Default, Clone)]
pub struct NoAuth;

impl AuthProvider for NoAuth {
    fn authenticate(&self, _headers: &HeaderMap) -> Result<Identity, AuthError> {
        Ok(Identity("anonymous".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_auth_always_authenticates_as_anonymous() {
        let provider = NoAuth;
        let headers = HeaderMap::new();
        let id = provider.authenticate(&headers).unwrap();
        assert_eq!(id.0, "anonymous");
    }
}
