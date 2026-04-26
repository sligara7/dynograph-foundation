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
    #[error("node not found: {node_type}/{node_id}")]
    NodeNotFound { node_type: String, node_id: String },
    #[error("edge not found: {edge_type} {from_id}->{to_id}")]
    EdgeNotFound {
        edge_type: String,
        from_id: String,
        to_id: String,
    },
    #[error(transparent)]
    Storage(#[from] DynoError),
}

impl IntoResponse for RegistryError {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::AlreadyExists(_) => StatusCode::CONFLICT,
            Self::NotFound(_) | Self::NodeNotFound { .. } | Self::EdgeNotFound { .. } => {
                StatusCode::NOT_FOUND
            }
            Self::Storage(inner) => status_for_dyno_error(inner),
        };
        (status, self.to_string()).into_response()
    }
}

/// Map `DynoError` variants to HTTP status. Validation / unknown-type /
/// schema / invalid-edge are caller errors (400). NodeNotFound /
/// EdgeNotFound surface as 404 — though node CRUD handlers prefer to
/// detect missing resources via `Option` returns from the engine and
/// emit their own 404s, this catch-all keeps any deeper layer that
/// raises `*NotFound` mapped honestly. The remaining variants
/// (storage, serialization, query, resolution, extraction) are
/// internal failures (500).
fn status_for_dyno_error(e: &DynoError) -> StatusCode {
    match e {
        DynoError::Validation { .. }
        | DynoError::InvalidEdge { .. }
        | DynoError::UnknownNodeType(_)
        | DynoError::UnknownEdgeType(_)
        | DynoError::Schema(_) => StatusCode::BAD_REQUEST,
        DynoError::NodeNotFound { .. } | DynoError::EdgeNotFound { .. } => StatusCode::NOT_FOUND,
        DynoError::Storage(_)
        | DynoError::Serialization(_)
        | DynoError::Query(_)
        | DynoError::Resolution(_)
        | DynoError::Extraction(_) => StatusCode::INTERNAL_SERVER_ERROR,
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

    /// Run a closure with a write-lock on the engine.
    pub fn with_engine_write<R>(&self, f: impl FnOnce(&mut StorageEngine) -> R) -> R {
        let mut guard = self.engine.write().expect("engine write lock poisoned");
        f(&mut guard)
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

    /// Drop a graph from the registry. Returns `Err(NotFound)` if the
    /// id isn't registered. The dropped `Arc<GraphEntry>` may outlive
    /// the registry entry while concurrent handlers finish their work
    /// on already-cloned `Arc`s; once those drop, `StorageEngine` is
    /// reclaimed.
    pub fn delete_graph(&self, id: &str) -> Result<(), RegistryError> {
        let mut graphs = self.graphs.write().expect("registry write lock poisoned");
        if graphs.remove(id).is_none() {
            return Err(RegistryError::NotFound(id.to_string()));
        }
        Ok(())
    }

    /// Snapshot of registered graph ids. Sorted for stable output.
    pub fn list_ids(&self) -> Vec<String> {
        let graphs = self.graphs.read().expect("registry read lock poisoned");
        let mut ids: Vec<String> = graphs.keys().cloned().collect();
        ids.sort();
        ids
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
    fn delete_graph_removes_entry() {
        let r = GraphRegistry::new();
        r.create_graph("g1", tiny_schema()).unwrap();
        assert!(r.get("g1").is_some());
        r.delete_graph("g1").unwrap();
        assert!(r.get("g1").is_none());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn delete_unknown_graph_errors_not_found() {
        let r = GraphRegistry::new();
        let err = r.delete_graph("missing").unwrap_err();
        assert!(matches!(err, RegistryError::NotFound(ref id) if id == "missing"));
    }

    #[test]
    fn list_ids_is_sorted() {
        let r = GraphRegistry::new();
        r.create_graph("zeta", tiny_schema()).unwrap();
        r.create_graph("alpha", tiny_schema()).unwrap();
        r.create_graph("mike", tiny_schema()).unwrap();
        assert_eq!(r.list_ids(), vec!["alpha", "mike", "zeta"]);
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
