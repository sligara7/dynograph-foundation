//! Service configuration — TOML on disk, env-var overrides on top.
//!
//! Layered precedence (later wins):
//! 1. Hard-coded defaults (`bind = 127.0.0.1:8080`, no storage root)
//! 2. TOML file at the path passed to `Config::load`, when supplied
//! 3. Env vars: `DYNOGRAPH_BIND`, `DYNOGRAPH_STORAGE_ROOT`
//!
//! Absent `storage.root` means in-memory mode. Setting it (via TOML
//! or env) flips the registry to on-disk and `rehydrate()` runs at
//! startup.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;

use crate::auth::{AuthProvider, BearerJwt, NoAuth};

// Defaults for the ingress hardening limits. Single source of truth so
// the serde field defaults and `ServerLimits::default` can't drift.
//
// `max_body_bytes`: a binary-vector util request carries two vectors of
// up to `MAX_VECTOR_LEN` (100_000) f64 each — ~4 MB serialized — so the
// cap must clear that with headroom for large `/batch` payloads. Bodies
// above it get 413 before the handler runs.
const DEFAULT_MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
// Wall-clock cap per request; long handlers shed the client wait with
// 408. (A `spawn_blocking` storage op already in flight still runs to
// completion in the background — this bounds the client's wait, not the
// scan.)
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 30;
// In-flight ceiling for `/v1` routes; beyond it new requests are shed
// with 503 rather than piling up against the blocking pool.
const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 1024;

/// Runtime form of the ingress hardening limits (parsed `ServerConfig`
/// with `request_timeout` as a `Duration`). Consumed by `app()` to
/// build the body-limit / timeout / concurrency layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerLimits {
    pub max_body_bytes: usize,
    pub request_timeout: Duration,
    pub max_concurrent_requests: usize,
}

impl Default for ServerLimits {
    fn default() -> Self {
        Self {
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            request_timeout: Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS),
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("parse {path}: {source}")]
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
    #[error("auth config: {0}")]
    Auth(String),
}

/// Top-level config denies unknown fields so a typo'd section name
/// (e.g. `[srever]`) fails loud at startup. Inner sections allow
/// unknown fields so adding a new key in a future release doesn't
/// break old binaries reading newer configs.
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub auth: AuthConfig,
}

/// Auth provider selection. `noauth` (the default) accepts every
/// request as `"anonymous"`; `bearer_jwt` requires a valid HS256
/// JWT in `Authorization: Bearer …`.
///
/// `bearer_jwt.secret` may be a literal string (convenient for dev)
/// or `secret_env = "VAR_NAME"` (resolves the secret from the
/// process environment at startup — the recommended path for
/// production so secrets aren't committed to TOML). Exactly one of
/// `secret` / `secret_env` must be set; both or neither is a loud
/// error from `resolve_secret`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case")]
pub enum AuthConfig {
    #[default]
    NoAuth,
    BearerJwt {
        #[serde(default)]
        secret: Option<String>,
        #[serde(default)]
        secret_env: Option<String>,
        #[serde(default)]
        issuer: Option<String>,
        #[serde(default)]
        audience: Option<String>,
    },
}

impl AuthConfig {
    /// Construct the configured `AuthProvider`. Reading the JWT
    /// secret from env happens here (at startup) so a missing env
    /// var fails before the listener binds — better to crash with a
    /// clear message than to accept connections and 401 every
    /// request.
    pub fn build_provider(&self) -> Result<Arc<dyn AuthProvider>, ConfigError> {
        match self {
            AuthConfig::NoAuth => Ok(Arc::new(NoAuth::new())),
            AuthConfig::BearerJwt {
                issuer, audience, ..
            } => {
                let secret = self.resolve_bearer_secret()?;
                let mut bj = BearerJwt::new(&secret);
                if let Some(iss) = issuer {
                    bj = bj.with_issuer(iss);
                }
                if let Some(aud) = audience {
                    bj = bj.with_audience(aud);
                }
                Ok(Arc::new(bj))
            }
        }
    }

    /// Resolve the symmetric signing secret. Private — `build_provider`
    /// is the public entry point. Only valid on the BearerJwt arm;
    /// callers via `build_provider` only reach this through that
    /// branch.
    fn resolve_bearer_secret(&self) -> Result<Vec<u8>, ConfigError> {
        let AuthConfig::BearerJwt {
            secret, secret_env, ..
        } = self
        else {
            unreachable!("resolve_bearer_secret called on non-BearerJwt variant");
        };
        match (secret, secret_env) {
            (Some(s), None) => Ok(s.as_bytes().to_vec()),
            (None, Some(var)) => std::env::var(var).map(String::into_bytes).map_err(|_| {
                ConfigError::Auth(format!(
                    "bearer_jwt.secret_env=\"{var}\" but env var not set"
                ))
            }),
            (Some(_), Some(_)) => Err(ConfigError::Auth(
                "bearer_jwt: set exactly one of `secret` / `secret_env`, not both".into(),
            )),
            (None, None) => Err(ConfigError::Auth(
                "bearer_jwt: must set one of `secret` / `secret_env`".into(),
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ServerConfig {
    /// Field-level default for the `[server]` table being present
    /// but missing `bind`. The `Default` impl below covers the
    /// case where `[server]` is absent entirely (driven by
    /// `Config`'s `#[serde(default)]` on the `server` field).
    #[serde(default = "default_bind")]
    pub bind: String,
    /// Max request body size in bytes (413 above it).
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    /// Per-request wall-clock cap in seconds (408 past it).
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    /// Max concurrently-executing `/v1` requests (503 beyond it).
    #[serde(default = "default_max_concurrent_requests")]
    pub max_concurrent_requests: usize,
}

impl ServerConfig {
    /// Project the raw TOML fields into the runtime `ServerLimits`
    /// (converting `request_timeout_secs` into a `Duration`).
    pub fn limits(&self) -> ServerLimits {
        ServerLimits {
            max_body_bytes: self.max_body_bytes,
            request_timeout: Duration::from_secs(self.request_timeout_secs),
            max_concurrent_requests: self.max_concurrent_requests,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
pub struct StorageConfig {
    /// `Some(path)` → on-disk RocksDB rooted at `path`. `None` →
    /// in-memory (HashMap-backed) storage.
    #[serde(default)]
    pub root: Option<PathBuf>,
}

fn default_bind() -> String {
    "127.0.0.1:8080".to_string()
}

fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

fn default_request_timeout_secs() -> u64 {
    DEFAULT_REQUEST_TIMEOUT_SECS
}

fn default_max_concurrent_requests() -> usize {
    DEFAULT_MAX_CONCURRENT_REQUESTS
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: default_bind(),
            max_body_bytes: default_max_body_bytes(),
            request_timeout_secs: default_request_timeout_secs(),
            max_concurrent_requests: default_max_concurrent_requests(),
        }
    }
}

impl Config {
    /// Build the full config: optional TOML file, then env-var
    /// overrides. A missing TOML file at an explicit path is an
    /// error; a missing default-path TOML (when the caller passes
    /// `None`) is fine.
    pub fn load(path: Option<&std::path::Path>) -> Result<Self, ConfigError> {
        let mut cfg = match path {
            Some(p) => Self::from_toml_file(p)?,
            None => Self::default(),
        };
        cfg.apply_env_overrides();
        Ok(cfg)
    }

    pub fn from_toml_file(path: &std::path::Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|e| ConfigError::Read {
            path: path.to_path_buf(),
            source: e,
        })?;
        toml::from_str(&text).map_err(|e| ConfigError::Parse {
            path: path.to_path_buf(),
            source: e,
        })
    }

    pub fn apply_env_overrides(&mut self) {
        if let Ok(bind) = std::env::var("DYNOGRAPH_BIND") {
            self.server.bind = bind;
        }
        if let Ok(root) = std::env::var("DYNOGRAPH_STORAGE_ROOT") {
            self.storage.root = Some(PathBuf::from(root));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_toml_uses_defaults() {
        let cfg: Config = toml::from_str("").unwrap();
        assert_eq!(cfg.server.bind, "127.0.0.1:8080");
        assert!(cfg.storage.root.is_none());
    }

    #[test]
    fn full_toml_round_trips() {
        let cfg: Config = toml::from_str(
            r#"
[server]
bind = "0.0.0.0:9090"

[storage]
root = "/var/lib/dynograph"
"#,
        )
        .unwrap();
        assert_eq!(cfg.server.bind, "0.0.0.0:9090");
        assert_eq!(cfg.storage.root, Some(PathBuf::from("/var/lib/dynograph")));
    }

    #[test]
    fn unknown_top_level_section_is_loud_error() {
        // A typo'd section name — `[srever]` instead of `[server]` —
        // is the failure mode that bites; without `deny_unknown_fields`
        // at the top level the typo'd section is silently dropped and
        // the binary boots with the wrong config. Inner sections
        // deliberately allow unknown keys for forward-compat with
        // newer config schemas.
        let res: Result<Config, _> = toml::from_str(
            r#"
[srever]
bind = "x"
"#,
        );
        assert!(res.is_err(), "top-level deny_unknown_fields should reject");
    }

    #[test]
    fn auth_defaults_to_noauth() {
        let cfg: Config = toml::from_str("").unwrap();
        assert_eq!(cfg.auth, AuthConfig::NoAuth);
        assert!(cfg.auth.build_provider().is_ok());
    }

    #[test]
    fn auth_bearer_jwt_with_literal_secret() {
        let cfg: Config = toml::from_str(
            r#"
[auth]
provider = "bearer_jwt"
secret = "dev-secret"
"#,
        )
        .unwrap();
        assert!(matches!(cfg.auth, AuthConfig::BearerJwt { .. }));
        cfg.auth.build_provider().expect("provider builds");
    }

    #[test]
    fn auth_bearer_jwt_secret_env_resolves_at_load() {
        let var = "DYNOGRAPH_TEST_SECRET_4f29";
        // SAFETY: only this test reads or writes this env var name.
        unsafe { std::env::set_var(var, "from-env") };
        let cfg: Config = toml::from_str(&format!(
            r#"
[auth]
provider = "bearer_jwt"
secret_env = "{var}"
"#,
        ))
        .unwrap();
        cfg.auth
            .build_provider()
            .expect("provider builds with resolved env-var secret");
        // SAFETY: same scope as the set_var above; isolated to this test.
        unsafe { std::env::remove_var(var) };
    }

    #[test]
    fn auth_bearer_jwt_missing_secret_is_loud_error() {
        let cfg: Config = toml::from_str(
            r#"
[auth]
provider = "bearer_jwt"
"#,
        )
        .unwrap();
        let err = cfg.auth.build_provider().err().expect("expected error");
        assert!(err.to_string().contains("must set one of"), "{err}");
    }

    #[test]
    fn auth_bearer_jwt_both_secrets_is_loud_error() {
        let cfg: Config = toml::from_str(
            r#"
[auth]
provider = "bearer_jwt"
secret = "literal"
secret_env = "VAR"
"#,
        )
        .unwrap();
        let err = cfg.auth.build_provider().err().expect("expected error");
        assert!(err.to_string().contains("not both"), "{err}");
    }

    #[test]
    fn auth_bearer_jwt_secret_env_unset_is_loud_error() {
        let cfg: Config = toml::from_str(
            r#"
[auth]
provider = "bearer_jwt"
secret_env = "DYNOGRAPH_DEFINITELY_UNSET_zzz"
"#,
        )
        .unwrap();
        let err = cfg.auth.build_provider().err().expect("expected error");
        assert!(err.to_string().contains("env var not set"), "{err}");
    }

    #[test]
    fn unknown_field_in_inner_section_is_tolerated() {
        // Forward-compat: a future foundation release may add new
        // keys; old binaries reading newer configs must not refuse
        // to boot.
        let cfg: Config = toml::from_str(
            r#"
[server]
bind = "0.0.0.0:9090"
future_unknown_key = "ignored"
"#,
        )
        .unwrap();
        assert_eq!(cfg.server.bind, "0.0.0.0:9090");
    }

    #[test]
    fn from_toml_file_reports_path_on_io_error() {
        let err = Config::from_toml_file(std::path::Path::new("/nonexistent/dynograph.toml"))
            .unwrap_err();
        assert!(matches!(err, ConfigError::Read { .. }));
        assert!(err.to_string().contains("/nonexistent/dynograph.toml"));
    }
}
