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

use serde::Deserialize;

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
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub storage: StorageConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_bind")]
    pub bind: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageConfig {
    /// `Some(path)` → on-disk RocksDB rooted at `path`. `None` →
    /// in-memory (HashMap-backed) storage.
    #[serde(default)]
    pub root: Option<PathBuf>,
}

fn default_bind() -> String {
    "127.0.0.1:8080".to_string()
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: default_bind(),
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
    fn unknown_field_is_loud_error() {
        let res: Result<Config, _> = toml::from_str(
            r#"
[server]
bind = "x"
nonexistent = "y"
"#,
        );
        assert!(res.is_err(), "deny_unknown_fields should reject");
    }

    #[test]
    fn from_toml_file_reports_path_on_io_error() {
        let err = Config::from_toml_file(std::path::Path::new("/nonexistent/dynograph.toml"))
            .unwrap_err();
        assert!(matches!(err, ConfigError::Read { .. }));
        assert!(err.to_string().contains("/nonexistent/dynograph.toml"));
    }
}
