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
//!
//! Storage backend: the registry can run in-memory (default; tests,
//! ephemeral dev) or against a `root` directory on disk. On-disk
//! mode persists each graph at `{root}/{id}/schema.json` (canonical
//! schema for rehydration) + `{root}/{id}/db/` (RocksDB column-family
//! store). On startup, `rehydrate()` walks `root` and registers each
//! valid graph dir; the existing in-memory create/get/delete surface
//! is unchanged from the caller's view.

use std::collections::HashMap;
use std::path::PathBuf;
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
    #[error("embedding not found: {node_type}/{node_id}")]
    EmbeddingNotFound { node_type: String, node_id: String },
    #[error("invalid graph id: {0}")]
    InvalidId(String),
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("filesystem error: {0}")]
    Filesystem(String),
    #[error("rehydration failed: {0}")]
    Rehydration(String),
    #[error(transparent)]
    Storage(#[from] DynoError),
}

impl IntoResponse for RegistryError {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::AlreadyExists(_) => StatusCode::CONFLICT,
            Self::NotFound(_)
            | Self::NodeNotFound { .. }
            | Self::EdgeNotFound { .. }
            | Self::EmbeddingNotFound { .. } => StatusCode::NOT_FOUND,
            Self::InvalidId(_) | Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Filesystem(_) | Self::Rehydration(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Storage(inner) => status_for_dyno_error(inner),
        };
        (status, self.to_string()).into_response()
    }
}

/// Constrain graph ids to characters that are safe as a directory
/// name across platforms. ASCII alphanumeric + `_-` is a deliberate
/// floor — no `.` (rules out `..`, leading-dot hidden files), no `/`
/// (rules out path traversal), no whitespace. Limit length to 100
/// chars; that's well above any realistic graph identity scheme and
/// well below per-component path limits (255 on most filesystems).
pub fn validate_graph_id(id: &str) -> Result<(), RegistryError> {
    if id.is_empty() {
        return Err(RegistryError::InvalidId("empty".to_string()));
    }
    if id.len() > 100 {
        return Err(RegistryError::InvalidId(format!(
            "too long ({} chars; max 100)",
            id.len()
        )));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(RegistryError::InvalidId(format!(
            "must match [A-Za-z0-9_-]+, got {id:?}"
        )));
    }
    Ok(())
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

/// A graph hosted by the registry. Engine + cached content_hash live
/// together under one lock so a schema replacement (slice 7's PUT
/// `/v1/graphs/{id}/schema`) can swap both atomically — concurrent
/// readers never observe a torn (schema, hash) pair.
pub struct GraphEntry {
    state: RwLock<GraphState>,
}

struct GraphState {
    engine: StorageEngine,
    content_hash: Arc<str>,
}

impl std::fmt::Debug for GraphEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphEntry")
            .field("content_hash", &self.content_hash())
            .field("engine", &"<StorageEngine>")
            .finish()
    }
}

impl GraphEntry {
    fn new(engine: StorageEngine, content_hash: Arc<str>) -> Self {
        Self {
            state: RwLock::new(GraphState {
                engine,
                content_hash,
            }),
        }
    }

    /// Cached SHA256 of the schema's canonical JSON. Returns an owned
    /// `Arc<str>` (cheap refcount bump) rather than a borrow because
    /// the underlying value lives behind a `RwLock` and a borrow
    /// would tie the caller to the lock guard.
    pub fn content_hash(&self) -> Arc<str> {
        self.state
            .read()
            .expect("graph state read lock poisoned")
            .content_hash
            .clone()
    }

    /// Run a closure with a read-lock on the engine.
    pub fn with_engine_read<R>(&self, f: impl FnOnce(&StorageEngine) -> R) -> R {
        let guard = self.state.read().expect("graph state read lock poisoned");
        f(&guard.engine)
    }

    /// Run a closure with a write-lock on the engine.
    pub fn with_engine_write<R>(&self, f: impl FnOnce(&mut StorageEngine) -> R) -> R {
        let mut guard = self.state.write().expect("graph state write lock poisoned");
        f(&mut guard.engine)
    }

    /// Validate, persist, then swap. The whole sequence runs under
    /// one write lock so concurrent readers never see a torn (schema,
    /// content_hash) pair, and a `persist` failure aborts before any
    /// in-memory mutation — no in-memory-vs-disk skew across a
    /// process restart. Returns the new content hash so the caller
    /// can construct the response without re-acquiring the lock.
    pub(crate) fn replace_schema_with(
        &self,
        new_schema: Schema,
        validator: impl FnOnce(&Schema, &Schema) -> Result<(), RegistryError>,
        persist: impl FnOnce(&Schema) -> Result<(), RegistryError>,
    ) -> Result<Arc<str>, RegistryError> {
        let mut guard = self.state.write().expect("graph state write lock poisoned");
        validator(guard.engine.schema(), &new_schema)?;
        persist(&new_schema)?;
        let new_hash: Arc<str> = Arc::from(content_hash(&new_schema).as_str());
        guard.engine.replace_schema(new_schema);
        guard.content_hash = new_hash.clone();
        Ok(new_hash)
    }
}

/// Where graphs live. `InMemory` is ephemeral (HashMap-backed
/// storage); `OnDisk { root }` writes each graph at `{root}/{id}/`
/// — `schema.json` is the canonical source-of-truth read on
/// rehydrate, and `db/` is the per-graph RocksDB column-family
/// store.
#[derive(Debug, Clone)]
pub enum StorageBackend {
    InMemory,
    OnDisk { root: PathBuf },
}

pub struct GraphRegistry {
    graphs: RwLock<HashMap<String, Arc<GraphEntry>>>,
    backend: StorageBackend,
}

impl GraphRegistry {
    /// Default: in-memory backend. Equivalent to `in_memory()`.
    /// Kept for back-compat with slice 1–3 call sites.
    pub fn new() -> Self {
        Self::in_memory()
    }

    pub fn in_memory() -> Self {
        Self {
            graphs: RwLock::new(HashMap::new()),
            backend: StorageBackend::InMemory,
        }
    }

    /// Persistent backend rooted at `root`. The directory is created
    /// lazily by `create_graph` and `rehydrate`; nothing happens on
    /// construction.
    pub fn on_disk(root: impl Into<PathBuf>) -> Self {
        Self {
            graphs: RwLock::new(HashMap::new()),
            backend: StorageBackend::OnDisk { root: root.into() },
        }
    }

    pub fn create_graph(&self, id: &str, schema: Schema) -> Result<Arc<GraphEntry>, RegistryError> {
        validate_graph_id(id)?;
        let mut graphs = self.graphs.write().expect("registry write lock poisoned");
        if graphs.contains_key(id) {
            return Err(RegistryError::AlreadyExists(id.to_string()));
        }
        let hash = Arc::from(content_hash(&schema));
        let engine = build_engine(&self.backend, id, schema)?;
        let entry = Arc::new(GraphEntry::new(engine, hash));
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

    /// Drop a graph from the registry. On `OnDisk` backends, also
    /// remove the per-graph directory — DELETE is destructive and
    /// symmetric with create. Concurrent handlers holding a cloned
    /// `Arc<GraphEntry>` will continue to operate on their copy
    /// against the now-removed RocksDB; expect a `Storage` error
    /// from any subsequent op rather than corruption (RocksDB
    /// surfaces "DB is closed / file not found" loudly). For dev
    /// simplicity we accept that race; production deployments that
    /// need orderly drain should serialize DELETE with their own
    /// quiescence step.
    pub fn delete_graph(&self, id: &str) -> Result<(), RegistryError> {
        let mut graphs = self.graphs.write().expect("registry write lock poisoned");
        if graphs.remove(id).is_none() {
            return Err(RegistryError::NotFound(id.to_string()));
        }
        if let StorageBackend::OnDisk { root } = &self.backend {
            let graph_dir = root.join(id);
            // Tolerate "already gone" — the registry entry was the
            // source of truth that the dir should exist; if it
            // doesn't, our job is done. Any other I/O error is real.
            match std::fs::remove_dir_all(&graph_dir) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    return Err(RegistryError::Filesystem(format!(
                        "remove dir {}: {e}",
                        graph_dir.display()
                    )));
                }
            }
        }
        Ok(())
    }

    /// Replace the graph's schema after the additive-evolution
    /// check passes. Disk-write (OnDisk backends) and in-memory swap
    /// run together under one lock so a disk-write failure leaves
    /// the in-memory state untouched.
    pub fn replace_schema(&self, id: &str, new_schema: Schema) -> Result<Arc<str>, RegistryError> {
        let entry = self
            .get(id)
            .ok_or_else(|| RegistryError::NotFound(id.to_string()))?;
        let backend = &self.backend;
        entry.replace_schema_with(
            new_schema,
            |old, new| {
                crate::schema_evolution::validate_compatible(old, new).map_err(|errors| {
                    let joined = errors
                        .iter()
                        .map(|e| e.to_string())
                        .collect::<Vec<_>>()
                        .join("; ");
                    RegistryError::BadRequest(format!("schema evolution rejected: {joined}"))
                })
            },
            |new| match backend {
                StorageBackend::InMemory => Ok(()),
                StorageBackend::OnDisk { root } => {
                    let graph_dir = root.join(id);
                    write_schema_file(&graph_dir, new)
                }
            },
        )
    }

    /// Snapshot of registered graph ids. Sorted for stable output.
    pub fn list_ids(&self) -> Vec<String> {
        let graphs = self.graphs.read().expect("registry read lock poisoned");
        let mut ids: Vec<String> = graphs.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Walk the on-disk root, registering every valid graph dir into
    /// this registry. Idempotent: ids already loaded in memory are
    /// skipped. Fails on the first malformed dir — unreadable
    /// `schema.json`, parse error, RocksDB open failure all bubble
    /// up as `RegistryError::Rehydration`. The fail-loud policy is
    /// deliberate (project-wide rule against silent fallbacks); a
    /// future opt-in `partial: true` mode can layer on top once a
    /// real use case demands skipping over corrupt graphs.
    ///
    /// On the `InMemory` backend this is a no-op returning an empty
    /// list.
    ///
    /// **Intended for startup, not steady-state.** Holds the registry
    /// write-lock for the whole walk + per-graph RocksDB open
    /// (~50ms each on disk), which would block concurrent reads. A
    /// missing root dir is treated as "fresh start" (empty result),
    /// not an error.
    ///
    /// Returns the ids that were freshly rehydrated (excluding ones
    /// already present), in sorted order.
    pub fn rehydrate(&self) -> Result<Vec<String>, RegistryError> {
        let StorageBackend::OnDisk { root } = &self.backend else {
            return Ok(Vec::new());
        };
        let entries = match std::fs::read_dir(root) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => {
                return Err(RegistryError::Rehydration(format!(
                    "read_dir {}: {e}",
                    root.display()
                )));
            }
        };
        let mut graphs = self.graphs.write().expect("registry write lock poisoned");
        let mut rehydrated = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| RegistryError::Rehydration(e.to_string()))?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let id = entry.file_name().to_string_lossy().into_owned();
            validate_graph_id(&id).map_err(|e| {
                RegistryError::Rehydration(format!("malformed dir name {id:?}: {e}"))
            })?;
            if graphs.contains_key(&id) {
                continue;
            }
            let schema = read_schema_file(&path)?;
            let db_path = path.join("db");
            let db_path_str = db_path
                .to_str()
                .ok_or_else(|| RegistryError::Rehydration(format!("non-utf8 path {db_path:?}")))?;
            let hash = Arc::from(content_hash(&schema));
            let engine = StorageEngine::new_rocksdb(schema, db_path_str).map_err(|e| {
                RegistryError::Rehydration(format!("open rocksdb at {db_path_str}: {e}"))
            })?;
            let entry_arc = Arc::new(GraphEntry::new(engine, hash));
            graphs.insert(id.clone(), entry_arc);
            rehydrated.push(id);
        }
        rehydrated.sort();
        Ok(rehydrated)
    }
}

/// Allocate a fresh `StorageEngine` for `id` against `backend`. On
/// `OnDisk` this also writes the canonical `schema.json` (used by
/// `rehydrate`), creates the graph directory, and opens RocksDB
/// inside it. If the RocksDB open fails after the dir + schema.json
/// have been written, we leave the partial state in place — the
/// caller can retry, manually clean up, or accept that the next
/// `rehydrate` will fail loudly on the broken dir (preferred to
/// silent cleanup that swallows the actual root cause).
///
/// Schema is consumed by value: `new_in_memory` and `new_rocksdb`
/// take ownership, and the on-disk arm only borrows once for the
/// schema.json write. Caller is expected to compute `content_hash`
/// before passing the schema in.
fn build_engine(
    backend: &StorageBackend,
    id: &str,
    schema: Schema,
) -> Result<StorageEngine, RegistryError> {
    match backend {
        StorageBackend::InMemory => Ok(StorageEngine::new_in_memory(schema)),
        StorageBackend::OnDisk { root } => {
            let graph_dir = root.join(id);
            std::fs::create_dir_all(&graph_dir).map_err(|e| {
                RegistryError::Filesystem(format!("create dir {}: {e}", graph_dir.display()))
            })?;
            write_schema_file(&graph_dir, &schema)?;
            let db_path = graph_dir.join("db");
            let db_path_str = db_path.to_str().ok_or_else(|| {
                RegistryError::Filesystem(format!("non-utf8 path {}", db_path.display()))
            })?;
            StorageEngine::new_rocksdb(schema, db_path_str).map_err(RegistryError::Storage)
        }
    }
}

fn write_schema_file(graph_dir: &std::path::Path, schema: &Schema) -> Result<(), RegistryError> {
    let schema_path = graph_dir.join("schema.json");
    let schema_json = serde_json::to_string_pretty(schema).map_err(|e| {
        RegistryError::Storage(DynoError::Serialization(format!("serialize schema: {e}")))
    })?;
    std::fs::write(&schema_path, schema_json)
        .map_err(|e| RegistryError::Filesystem(format!("write {}: {e}", schema_path.display())))
}

fn read_schema_file(graph_dir: &std::path::Path) -> Result<Schema, RegistryError> {
    let schema_path = graph_dir.join("schema.json");
    let text = std::fs::read_to_string(&schema_path)
        .map_err(|e| RegistryError::Rehydration(format!("read {}: {e}", schema_path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| RegistryError::Rehydration(format!("parse {}: {e}", schema_path.display())))
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
        let h1 = r.get("g1").unwrap().content_hash();
        let h2 = r.get("g1").unwrap().content_hash();
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
        let ha = a.content_hash();
        let hb = b.content_hash();
        assert_eq!(ha, hb);
        assert!(!Arc::ptr_eq(&ha, &hb));
    }

    #[test]
    fn validate_graph_id_rejects_path_traversal() {
        let too_long = "a".repeat(200);
        let bad: &[&str] = &[
            "",
            ".",
            "..",
            "../etc/passwd",
            "foo/bar",
            "foo bar",
            ".hidden",
            "a/b",
            "café",
            &too_long,
            "with\0null",
        ];
        for id in bad {
            assert!(
                matches!(validate_graph_id(id), Err(RegistryError::InvalidId(_))),
                "expected reject for {id:?}"
            );
        }
        for ok in ["g1", "alpha", "STORY_42", "abc-def", "X"] {
            validate_graph_id(ok).unwrap_or_else(|e| panic!("expected accept for {ok:?}: {e}"));
        }
    }

    #[test]
    fn create_graph_rejects_invalid_id() {
        let r = GraphRegistry::new();
        let err = r.create_graph("../escape", tiny_schema()).unwrap_err();
        assert!(matches!(err, RegistryError::InvalidId(_)));
    }

    #[test]
    fn rehydrate_in_memory_is_no_op() {
        let r = GraphRegistry::in_memory();
        assert_eq!(r.rehydrate().unwrap(), Vec::<String>::new());
    }
}
