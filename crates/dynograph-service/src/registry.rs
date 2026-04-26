//! In-process registry of named graphs.
//!
//! Each graph is a `GraphEntry` — a `StorageEngine` plus a
//! pre-computed `content_hash` of its schema. The hash is computed
//! once at create time and reused on every schema-endpoint read,
//! avoiding the O(N) JSON+SHA256 walk per GET.
//!
//! Lock topology: outer `RwLock` on the registry HashMap (rare writes,
//! frequent reads), inner per-graph `RwLock` on the engine (matches
//! `StorageEngine`'s `&self` reads vs `&mut self` writes).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use dynograph_core::{DynoError, Schema};
use dynograph_storage::StorageEngine;

use crate::schema_response::content_hash;

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("graph already exists: {0}")]
    AlreadyExists(String),
    #[error("graph not found: {0}")]
    NotFound(String),
    #[error(transparent)]
    Storage(#[from] DynoError),
}

impl IntoResponse for RegistryError {
    fn into_response(self) -> Response {
        let status = match self {
            Self::AlreadyExists(_) => StatusCode::CONFLICT,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Storage(_) => StatusCode::BAD_REQUEST,
        };
        (status, self.to_string()).into_response()
    }
}

/// A graph hosted by the registry.
pub struct GraphEntry {
    engine: RwLock<StorageEngine>,
    content_hash: Arc<str>,
}

impl std::fmt::Debug for GraphEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphEntry")
            .field("content_hash", &self.content_hash)
            .field("engine", &"<StorageEngine>")
            .finish()
    }
}

impl GraphEntry {
    /// Cached SHA256 of the schema's canonical JSON. Computed once at
    /// create time; refreshed by `replace_schema` (when that lands).
    pub fn content_hash(&self) -> &Arc<str> {
        &self.content_hash
    }

    /// Run a closure with a read-lock on the engine. Hides the
    /// per-engine locking strategy from handlers.
    pub fn with_engine_read<R>(&self, f: impl FnOnce(&StorageEngine) -> R) -> R {
        let guard = self.engine.read().expect("engine read lock poisoned");
        f(&guard)
    }
}

pub struct GraphRegistry {
    graphs: RwLock<HashMap<String, Arc<GraphEntry>>>,
}

impl GraphRegistry {
    pub fn new() -> Self {
        Self {
            graphs: RwLock::new(HashMap::new()),
        }
    }

    pub fn create_graph(&self, id: &str, schema: Schema) -> Result<Arc<GraphEntry>, RegistryError> {
        let mut graphs = self.graphs.write().expect("registry write lock poisoned");
        if graphs.contains_key(id) {
            return Err(RegistryError::AlreadyExists(id.to_string()));
        }
        let hash = Arc::from(content_hash(&schema));
        let entry = Arc::new(GraphEntry {
            engine: RwLock::new(StorageEngine::new_in_memory(schema)),
            content_hash: hash,
        });
        graphs.insert(id.to_string(), entry.clone());
        Ok(entry)
    }

    pub fn get(&self, id: &str) -> Option<Arc<GraphEntry>> {
        self.graphs
            .read()
            .expect("registry read lock poisoned")
            .get(id)
            .cloned()
    }
}

impl Default for GraphRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl GraphRegistry {
    pub(crate) fn len(&self) -> usize {
        self.graphs
            .read()
            .expect("registry read lock poisoned")
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_schema() -> Schema {
        Schema::from_yaml(
            r#"
schema:
  name: t
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
    fn create_graph_inserts_with_cached_hash() {
        let r = GraphRegistry::new();
        let entry = r.create_graph("g1", tiny_schema()).unwrap();
        assert_eq!(r.len(), 1);
        assert!(r.get("g1").is_some());
        assert!(r.get("missing").is_none());
        assert_eq!(entry.content_hash().len(), 64);
    }

    #[test]
    fn create_graph_rejects_duplicate() {
        let r = GraphRegistry::new();
        r.create_graph("g1", tiny_schema()).unwrap();
        let err = r.create_graph("g1", tiny_schema()).unwrap_err();
        assert!(matches!(err, RegistryError::AlreadyExists(ref id) if id == "g1"));
    }

    #[test]
    fn cached_hash_is_stable_across_get_calls() {
        // Proves the hash returned by GET == GET == … without recompute.
        let r = GraphRegistry::new();
        r.create_graph("g1", tiny_schema()).unwrap();
        let h1 = r.get("g1").unwrap().content_hash().clone();
        let h2 = r.get("g1").unwrap().content_hash().clone();
        assert!(
            Arc::ptr_eq(&h1, &h2),
            "Arc<str> should be shared, not re-allocated"
        );
    }

    #[test]
    fn distinct_graphs_get_distinct_hashes() {
        let r = GraphRegistry::new();
        let a = r.create_graph("a", tiny_schema()).unwrap();
        // Same schema → same hash, but separate Arcs.
        let b = r.create_graph("b", tiny_schema()).unwrap();
        assert_eq!(a.content_hash(), b.content_hash());
        assert!(!Arc::ptr_eq(a.content_hash(), b.content_hash()));
    }
}
