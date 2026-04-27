//! Pluggable request authentication.

use std::sync::Arc;

use axum::http::{HeaderMap, header::AUTHORIZATION};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode};
use serde::Deserialize;

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

/// HS256 bearer-token validator. The `Authorization: Bearer <jwt>`
/// header is required; tokens are validated against a symmetric
/// secret with `exp` mandatory, `iss`/`aud` enforced when configured
/// at construction. The token's `sub` claim becomes the `Identity`.
///
/// Foundation only authenticates — it identifies the caller. Per-
/// user authorization (which graphs/data they can touch) is consumer
/// business logic that wraps these handlers downstream.
///
/// JWKS / asymmetric (RS256/ES256) deferred — adds an async key
/// fetcher and an HTTP client dep with no caller asking yet. When a
/// real consumer needs it, layer it as a separate AuthProvider impl.
pub struct BearerJwt {
    decoding_key: DecodingKey,
    validation: Validation,
}

#[derive(Debug, Deserialize)]
struct Claims {
    /// Subject — the user identifier the service surfaces as
    /// `Identity`. `jsonwebtoken`'s default `required_spec_claims`
    /// includes `exp`, so missing-expiry tokens are rejected before
    /// we get here.
    sub: String,
}

impl BearerJwt {
    pub fn new(secret: &[u8]) -> Self {
        Self {
            decoding_key: DecodingKey::from_secret(secret),
            validation: Validation::new(Algorithm::HS256),
        }
    }

    pub fn with_issuer(mut self, issuer: &str) -> Self {
        self.validation.set_issuer(&[issuer]);
        self
    }

    pub fn with_audience(mut self, audience: &str) -> Self {
        self.validation.set_audience(&[audience]);
        self
    }
}

impl AuthProvider for BearerJwt {
    fn authenticate(&self, headers: &HeaderMap) -> Result<Identity, AuthError> {
        let header = headers
            .get(AUTHORIZATION)
            .ok_or_else(|| AuthError::new("missing Authorization header"))?;
        let header_str = header
            .to_str()
            .map_err(|_| AuthError::new("Authorization header is not valid UTF-8"))?;
        // RFC 6750 §2.1: scheme matching is case-insensitive
        // ("Bearer" / "bearer" / "BEARER" all valid).
        let (scheme, token) = header_str
            .split_once(' ')
            .ok_or_else(|| AuthError::new("Authorization header must use Bearer scheme"))?;
        if !scheme.eq_ignore_ascii_case("Bearer") {
            return Err(AuthError::new(
                "Authorization header must use Bearer scheme",
            ));
        }
        let data = decode::<Claims>(token, &self.decoding_key, &self.validation)
            .map_err(|e| AuthError::new(format!("invalid token: {e}")))?;
        // Move the decoded `String` into the `Arc<str>`: `From<String>`
        // reuses the heap buffer where possible, avoiding the alloc +
        // memcpy that `Arc::from(s.as_str())` would do.
        Ok(Identity(Arc::from(data.claims.sub)))
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

    use axum::http::HeaderValue;
    use jsonwebtoken::{EncodingKey, Header, encode};
    use serde::Serialize;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Test-side claims shape — matches `Claims` plus the optional
    /// fields jsonwebtoken's `Validation` checks (iss/aud/exp). We
    /// keep this distinct from the production `Claims` so tests can
    /// mint tokens with arbitrary extras without those leaking onto
    /// the production decode struct.
    #[derive(Debug, Serialize)]
    struct TestClaims {
        sub: String,
        exp: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        iss: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        aud: Option<String>,
    }

    fn now_secs() -> usize {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as usize
    }

    fn mint_token(secret: &[u8], claims: &TestClaims) -> String {
        encode(
            &Header::new(Algorithm::HS256),
            claims,
            &EncodingKey::from_secret(secret),
        )
        .unwrap()
    }

    fn headers_with_bearer(token: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        h
    }

    #[test]
    fn bearer_jwt_accepts_valid_token_and_returns_sub_as_identity() {
        let secret = b"correct-horse-battery-staple";
        let token = mint_token(
            secret,
            &TestClaims {
                sub: "alice".into(),
                exp: now_secs() + 60,
                iss: None,
                aud: None,
            },
        );
        let provider = BearerJwt::new(secret);
        let id = provider.authenticate(&headers_with_bearer(&token)).unwrap();
        assert_eq!(&*id.0, "alice");
    }

    #[test]
    fn bearer_jwt_rejects_expired_token() {
        let secret = b"s";
        // jsonwebtoken's default `Validation::leeway` is 60 seconds —
        // push expiry well past that so the test is definitively
        // past the clock-skew window, not on the edge.
        let token = mint_token(
            secret,
            &TestClaims {
                sub: "alice".into(),
                exp: now_secs() - 600,
                iss: None,
                aud: None,
            },
        );
        let provider = BearerJwt::new(secret);
        let err = provider
            .authenticate(&headers_with_bearer(&token))
            .unwrap_err();
        assert!(err.message().contains("invalid token"), "{err:?}");
    }

    #[test]
    fn bearer_jwt_rejects_wrong_signature() {
        let token = mint_token(
            b"signing-secret",
            &TestClaims {
                sub: "alice".into(),
                exp: now_secs() + 60,
                iss: None,
                aud: None,
            },
        );
        let provider = BearerJwt::new(b"different-secret");
        let err = provider
            .authenticate(&headers_with_bearer(&token))
            .unwrap_err();
        assert!(err.message().contains("invalid token"), "{err:?}");
    }

    #[test]
    fn bearer_jwt_rejects_missing_header() {
        let provider = BearerJwt::new(b"s");
        let err = provider.authenticate(&HeaderMap::new()).unwrap_err();
        assert!(err.message().contains("missing Authorization"), "{err:?}");
    }

    #[test]
    fn bearer_jwt_rejects_non_bearer_scheme() {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_static("Basic dXNlcjpwYXNz"),
        );
        let provider = BearerJwt::new(b"s");
        let err = provider.authenticate(&headers).unwrap_err();
        assert!(err.message().contains("Bearer scheme"), "{err:?}");
    }

    #[test]
    fn bearer_jwt_scheme_match_is_case_insensitive() {
        // RFC 6750 §2.1 — clients may send `bearer`, `BEARER`, etc.
        let secret = b"s";
        let token = mint_token(
            secret,
            &TestClaims {
                sub: "alice".into(),
                exp: now_secs() + 60,
                iss: None,
                aud: None,
            },
        );
        let provider = BearerJwt::new(secret);
        for prefix in ["Bearer ", "bearer ", "BEARER ", "BeArEr "] {
            let mut headers = HeaderMap::new();
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("{prefix}{token}")).unwrap(),
            );
            let id = provider.authenticate(&headers).unwrap();
            assert_eq!(&*id.0, "alice", "prefix {prefix:?} should be accepted");
        }
    }

    #[test]
    fn bearer_jwt_rejects_garbage_token() {
        let provider = BearerJwt::new(b"s");
        let err = provider
            .authenticate(&headers_with_bearer("not.a.jwt"))
            .unwrap_err();
        assert!(err.message().contains("invalid token"), "{err:?}");
    }

    #[test]
    fn bearer_jwt_enforces_configured_issuer() {
        let secret = b"s";
        let token = mint_token(
            secret,
            &TestClaims {
                sub: "alice".into(),
                exp: now_secs() + 60,
                iss: Some("wrong".into()),
                aud: None,
            },
        );
        let provider = BearerJwt::new(secret).with_issuer("expected");
        let err = provider
            .authenticate(&headers_with_bearer(&token))
            .unwrap_err();
        assert!(err.message().contains("invalid token"), "{err:?}");

        // Same provider accepts a matching iss.
        let good = mint_token(
            secret,
            &TestClaims {
                sub: "alice".into(),
                exp: now_secs() + 60,
                iss: Some("expected".into()),
                aud: None,
            },
        );
        let id = provider.authenticate(&headers_with_bearer(&good)).unwrap();
        assert_eq!(&*id.0, "alice");
    }

    #[test]
    fn bearer_jwt_enforces_configured_audience() {
        let secret = b"s";
        let token = mint_token(
            secret,
            &TestClaims {
                sub: "alice".into(),
                exp: now_secs() + 60,
                iss: None,
                aud: Some("wrong".into()),
            },
        );
        let provider = BearerJwt::new(secret).with_audience("expected");
        let err = provider
            .authenticate(&headers_with_bearer(&token))
            .unwrap_err();
        assert!(err.message().contains("invalid token"), "{err:?}");
    }
}
