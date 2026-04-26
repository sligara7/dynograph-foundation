//! Wire shape for schema endpoints + content-hash drift detection.
//!
//! Mirrors the contract C-partial established on the storyflow side:
//! every schema response carries a `wire_version` (foundation crate
//! version) and a `content_hash` (SHA256 of the schema's canonical JSON).
//! Consumers (e.g. generation_plus's Pydantic codegen) fail-fast on
//! mismatch — see `services/generation_plus/src/schemas/startup_guard.py`
//! in the storyflow repo.

use dynograph_core::Schema;
use serde::Serialize;
use sha2::{Digest, Sha256};

/// Foundation crate version this service was compiled against.
pub const WIRE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// JSON shape returned by graph lifecycle endpoints. Owned (not borrowed)
/// because axum's Json extractor requires the payload to be `'static`.
#[derive(Debug, Serialize)]
pub struct SchemaResponse {
    pub id: String,
    pub wire_version: &'static str,
    pub content_hash: String,
    pub schema: Schema,
}

impl SchemaResponse {
    pub fn new(id: String, schema: Schema) -> Self {
        let content_hash = content_hash(&schema);
        Self {
            id,
            wire_version: WIRE_VERSION,
            content_hash,
            schema,
        }
    }
}

/// SHA256 of the schema's canonical JSON serialization, hex-encoded.
///
/// Determinism trick: serde_json's `to_value` constructs maps using the
/// `serde_json::Map` type, which without the `preserve_order` feature is
/// backed by `BTreeMap` — sorted keys. So `to_value` first turns every
/// `HashMap` in the schema into a sorted Map, and the subsequent
/// `to_string` emits keys alphabetically. The hash is stable across
/// runs even though `Schema` itself uses `HashMap` for node/edge types.
pub fn content_hash(schema: &Schema) -> String {
    let value = serde_json::to_value(schema).expect("schema → json value");
    let canonical = serde_json::to_string(&value).expect("value → string");
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    // sha2 0.11 returns `hybrid_array::Array` from `finalize()`, which
    // dropped the `LowerHex` impl that 0.10's `GenericArray` had —
    // hex-encode by hand instead.
    hasher
        .finalize()
        .iter()
        .fold(String::with_capacity(64), |mut s, b| {
            use std::fmt::Write;
            write!(&mut s, "{:02x}", b).unwrap();
            s
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_schema() -> Schema {
        Schema::from_yaml(
            r#"
schema:
  name: demo
  version: 1
  node_types:
    Item:
      properties:
        name: { type: string, required: true }
  edge_types: {}
"#,
        )
        .unwrap()
    }

    #[test]
    fn content_hash_is_deterministic_across_runs() {
        let s = tiny_schema();
        let h1 = content_hash(&s);
        let h2 = content_hash(&s);
        assert_eq!(h1, h2);
        // SHA256 hex is 64 chars
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn content_hash_changes_with_schema_change() {
        let mut s = tiny_schema();
        let h_before = content_hash(&s);
        s.version = 2;
        let h_after = content_hash(&s);
        assert_ne!(h_before, h_after);
    }

    #[test]
    fn wire_version_matches_crate_version() {
        assert_eq!(WIRE_VERSION, env!("CARGO_PKG_VERSION"));
    }
}
