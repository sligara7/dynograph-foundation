//! Wire shape for schema endpoints + content-hash drift detection.
//!
//! Every schema response carries a `wire_version` (foundation crate
//! version) and a `content_hash` (SHA256 of the schema's canonical
//! JSON). Consumers compare these against compiled-in constants and
//! fail-fast on mismatch.

use dynograph_core::Schema;
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const WIRE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Serialize)]
pub struct SchemaResponse {
    pub id: String,
    pub wire_version: &'static str,
    pub content_hash: String,
    pub schema: Schema,
}

impl SchemaResponse {
    /// Compute the hash from the schema. Use `with_cached_hash` instead
    /// when the registry already holds a cached hash for this schema.
    pub fn new(id: String, schema: Schema) -> Self {
        let content_hash = content_hash(&schema);
        Self::with_cached_hash(id, schema, content_hash)
    }

    pub fn with_cached_hash(id: String, schema: Schema, content_hash: String) -> Self {
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
/// Determinism: serde_json's default `Map` is backed by `BTreeMap`, so
/// `to_value` first turns every `HashMap` in the schema into a sorted
/// Map, and `to_writer` emits keys alphabetically. Hash is stable
/// across runs even though `Schema` itself uses `HashMap` internally.
pub fn content_hash(schema: &Schema) -> String {
    let value = serde_json::to_value(schema).expect("schema → json value");
    let mut hasher = Sha256::new();
    serde_json::to_writer(HashWriter(&mut hasher), &value).expect("value → hasher");
    let bytes = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write;
        write!(&mut hex, "{:02x}", b).unwrap();
    }
    hex
}

/// `std::io::Write` adapter that streams bytes directly into a SHA256
/// hasher. Lets `serde_json::to_writer` skip the intermediate `String`
/// the previous `to_string` + `update` shape allocated.
struct HashWriter<'a>(&'a mut Sha256);

impl std::io::Write for HashWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
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
