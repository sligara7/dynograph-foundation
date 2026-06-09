//! Storage engine — supports in-memory (testing) and RocksDB (production).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use dynograph_core::{DynoError, Schema, Value};

use crate::backend::{
    ALL_CFS, BufferedEffect, BufferedOp, CF_ADJ_IN, CF_ADJ_OUT, CF_EDGES, CF_EMBEDDINGS,
    CF_NODE_IDX, CF_NODES, CfId, KvBackend, MemoryBackend, RocksBackend,
};
use crate::cache::{CacheConfig, ReadCache};

/// A node stored in the graph.
#[derive(Debug, Clone)]
pub struct StoredNode {
    pub graph_id: String,
    pub node_type: String,
    pub node_id: String,
    pub properties: HashMap<String, Value>,
}

/// An edge stored in the graph.
#[derive(Debug, Clone)]
pub struct StoredEdge {
    pub graph_id: String,
    pub edge_type: String,
    pub from_id: String,
    pub to_id: String,
    pub properties: HashMap<String, Value>,
}

/// The storage engine — schema-validated graph storage.
pub struct StorageEngine {
    /// `Arc<Schema>` so the schema can be reference-shared without a
    /// deep `Schema::clone`. Public `schema()` derefs through the arc;
    /// constructors and `replace_schema` take `Schema` and wrap.
    schema: Arc<Schema>,
    /// The byte-level store. `Box<dyn KvBackend>` keeps `StorageEngine` a
    /// single concrete type regardless of which backend is in use, so both
    /// constructors return the same type and the registry holds them
    /// uniformly. The vtable hop is negligible next to the actual KV work,
    /// and every read goes through the cache / write-buffer overlay first.
    backend: Box<dyn KvBackend>,
    /// LRU read cache for node lookups and adjacency scans.
    /// Mutex allows cache updates through &self (get path is immutable at API level).
    read_cache: Mutex<ReadCache>,
    /// When `Some`, all writes (put / delete / prefix-delete) buffer
    /// here instead of hitting the backend. `commit_batch` flushes
    /// atomically; `discard_batch` drops them.
    write_buffer: Option<Vec<BufferedOp>>,
    /// Optional sidecar full-text index (only with the `fulltext` feature).
    /// `Some` only when the schema declares at least one `fulltext` property —
    /// otherwise there's nothing to mirror and we skip the writer arena. RocksDB
    /// stays the source of truth; this index is derived and rebuildable via
    /// `reindex_fulltext`.
    #[cfg(feature = "fulltext")]
    text_index: Option<dynograph_text::TextIndex>,
}

impl StorageEngine {
    /// Create an in-memory storage engine (for testing).
    pub fn new_in_memory(schema: Schema) -> Self {
        // RAM-backed full-text index when the schema uses `fulltext`. Creation
        // can only fail on OOM, so `expect` here keeps the infallible
        // constructor signature; the ephemeral backend is test-oriented anyway.
        #[cfg(feature = "fulltext")]
        let text_index = schema.has_any_fulltext_properties().then(|| {
            dynograph_text::TextIndex::open_in_ram()
                .expect("in-memory full-text index creation should not fail")
        });
        Self {
            schema: Arc::new(schema),
            backend: Box::new(MemoryBackend::new()),
            read_cache: Mutex::new(ReadCache::new(CacheConfig::default())),
            write_buffer: None,
            #[cfg(feature = "fulltext")]
            text_index,
        }
    }

    /// Create a RocksDB-backed storage engine (for production).
    pub fn new_rocksdb(schema: Schema, path: &str) -> Result<Self, DynoError> {
        let backend = RocksBackend::open(path)?;

        // The full-text index lives in a sibling subdir of the RocksDB store, so
        // it travels with the data dir. Built only when the schema uses it.
        #[cfg(feature = "fulltext")]
        let text_index = if schema.has_any_fulltext_properties() {
            let ft_dir = std::path::Path::new(path).join("fulltext");
            Some(
                dynograph_text::TextIndex::open(&ft_dir)
                    .map_err(|e| DynoError::Storage(format!("full-text index open failed: {e}")))?,
            )
        } else {
            None
        };

        Ok(Self {
            schema: Arc::new(schema),
            backend: Box::new(backend),
            read_cache: Mutex::new(ReadCache::new(CacheConfig::default())),
            write_buffer: None,
            #[cfg(feature = "fulltext")]
            text_index,
        })
    }

    /// Get the schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Replace the in-memory schema. Caller is responsible for any
    /// schema-evolution compatibility checks — this is a pure field
    /// swap. No re-indexing happens: indexed-property names are
    /// derived from `schema` on each scan, so adding a new indexed
    /// property starts indexing forward; existing rows won't be
    /// back-indexed but they don't have the new property anyway. A
    /// previously-indexed property losing its `indexed: true` flag
    /// leaves stale entries in `CF_NODE_IDX` — those are unreachable
    /// (the property may not exist on the new schema) and tolerable
    /// garbage; cleaning them up is a future-slice concern.
    pub fn replace_schema(&mut self, new_schema: Schema) {
        self.schema = Arc::new(new_schema);
    }

    // =========================================================================
    // Full-text index sidecar (optional `fulltext` feature)
    // =========================================================================
    //
    // RocksDB is authoritative; the Tantivy index is a derived, rebuildable
    // view. Node writes mirror into it (`fulltext_upsert` / `fulltext_delete`),
    // and visibility follows the same boundary as RocksDB durability: a
    // non-batched write commits the index immediately, a batched write becomes
    // visible at `commit_batch` (and is rolled back by `discard_batch`).
    //
    // Error contract: the authoritative RocksDB write lands BEFORE the index is
    // mirrored, so a node-write method can return `Err` (an index commit failed)
    // *after* the node is already durable in RocksDB. RocksDB stays correct; the
    // index may lag and is recoverable with `reindex_fulltext`. Likewise
    // `commit_batch` can leave RocksDB ahead of the index if the index commit
    // fails after the batch write — the next `begin_batch` re-commits any pending
    // index ops to re-establish a clean baseline (see `begin_batch`).
    //
    // The index is built once, when the engine opens (it needs the schema and,
    // for RocksDB, the data path). Enabling `fulltext` on an existing graph via
    // `replace_schema` therefore does NOT build an index for the live engine;
    // `search_fulltext` / `reindex_fulltext` fail loud in that state rather than
    // silently returning empty / `Ok(0)`. Reopen the engine (restart) to build it.

    /// `Some(err)` when the schema declares full-text but no index exists — i.e.
    /// full-text was enabled on a live engine (via `replace_schema`) that opened
    /// without one. Lets `search_fulltext` / `reindex_fulltext` fail loud instead
    /// of silently returning empty results or a bogus `Ok(0)`. `None` when the
    /// index is present, or when the schema genuinely declares no full-text (a
    /// real "nothing to do").
    #[cfg(feature = "fulltext")]
    fn fulltext_unavailable(&self) -> Option<DynoError> {
        if self.text_index.is_none() && self.schema.has_any_fulltext_properties() {
            Some(DynoError::Storage(
                "full-text index unavailable: it is built when the engine opens, so enabling \
                 full-text on an existing graph requires reopening the engine (restart) before \
                 search or reindex"
                    .to_string(),
            ))
        } else {
            None
        }
    }

    /// Extract `(name, value)` pairs for this node type's `fulltext` string
    /// properties. Non-string values are skipped — schema validation already
    /// rejects `fulltext` on non-string types, so this is belt-and-braces.
    #[cfg(feature = "fulltext")]
    fn fulltext_fields(
        &self,
        node_type: &str,
        properties: &HashMap<String, Value>,
    ) -> Vec<(String, String)> {
        self.schema
            .fulltext_properties(node_type)
            .into_iter()
            .filter_map(|name| match properties.get(name) {
                Some(Value::String(s)) => Some((name.to_string(), s.clone())),
                _ => None,
            })
            .collect()
    }

    /// Mirror a node write into the full-text index. No-op when the index is
    /// absent or the type declares no `fulltext` properties. Commits outside a
    /// batch; buffers inside one.
    #[cfg(feature = "fulltext")]
    fn fulltext_upsert(
        &self,
        graph_id: &str,
        node_type: &str,
        node_id: &str,
        properties: &HashMap<String, Value>,
    ) -> Result<(), DynoError> {
        let Some(ti) = &self.text_index else {
            return Ok(());
        };
        if !self.schema.has_fulltext_properties(node_type) {
            return Ok(());
        }
        let fields = self.fulltext_fields(node_type, properties);
        ti.upsert(graph_id, node_type, node_id, &fields)
            .map_err(|e| DynoError::Storage(format!("full-text upsert failed: {e}")))?;
        if self.write_buffer.is_none() {
            ti.commit()
                .map_err(|e| DynoError::Storage(format!("full-text commit failed: {e}")))?;
        }
        Ok(())
    }

    /// Mirror a node delete into the full-text index. See `fulltext_upsert` for
    /// commit cadence. A type that lost its `fulltext` flag via schema evolution
    /// won't be cleaned up here — tolerable derived-index garbage, same posture
    /// as `replace_schema`; a `reindex_fulltext` rebuild clears it.
    #[cfg(feature = "fulltext")]
    fn fulltext_delete(
        &self,
        graph_id: &str,
        node_type: &str,
        node_id: &str,
    ) -> Result<(), DynoError> {
        let Some(ti) = &self.text_index else {
            return Ok(());
        };
        if !self.schema.has_fulltext_properties(node_type) {
            return Ok(());
        }
        ti.delete(graph_id, node_id)
            .map_err(|e| DynoError::Storage(format!("full-text delete failed: {e}")))?;
        if self.write_buffer.is_none() {
            ti.commit()
                .map_err(|e| DynoError::Storage(format!("full-text commit failed: {e}")))?;
        }
        Ok(())
    }

    /// BM25 keyword search over the full-text index, scoped to `graph_id` and
    /// optionally one `node_type`. Returns up to `limit` hits, highest score
    /// first. Empty when the schema declares no `fulltext` property. Fails loud
    /// if full-text was enabled on a live engine that opened without an index
    /// (see `fulltext_unavailable`).
    #[cfg(feature = "fulltext")]
    pub fn search_fulltext(
        &self,
        graph_id: &str,
        query: &str,
        node_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<dynograph_text::TextHit>, DynoError> {
        if let Some(ti) = &self.text_index {
            return ti
                .search(graph_id, query, node_type, limit)
                .map_err(|e| DynoError::Storage(format!("full-text search failed: {e}")));
        }
        match self.fulltext_unavailable() {
            Some(err) => Err(err),
            None => Ok(Vec::new()),
        }
    }

    /// Rebuild the full-text index for `graph_id` from the authoritative node
    /// store: drop the graph's existing documents, then re-index every
    /// `fulltext` node. Use to recover from drift. Returns the number of nodes
    /// indexed; `Ok(0)` only when the schema genuinely declares no full-text.
    ///
    /// Fails loud (not `Ok(0)`) if full-text was enabled on a live engine that
    /// opened without an index — reopen the engine to build it first.
    ///
    /// Cost: materializes every full-text node and rebuilds in a single pass
    /// under the caller's lock (the service holds the per-graph write lock), so
    /// a reindex of a large graph blocks its reads and writes for the duration.
    /// An incremental / double-buffered rebuild is tracked as a follow-up.
    #[cfg(feature = "fulltext")]
    pub fn reindex_fulltext(&self, graph_id: &str) -> Result<usize, DynoError> {
        // A rebuild can't run inside an open batch: `scan_nodes` would see
        // uncommitted node state, and the final `ti.commit()` would flush the
        // batch's buffered text ops, breaking its all-or-nothing semantics.
        if self.is_batching() {
            return Err(DynoError::Storage(
                "reindex_fulltext cannot run inside an open batch".to_string(),
            ));
        }
        let ti = match &self.text_index {
            Some(ti) => ti,
            None => {
                return match self.fulltext_unavailable() {
                    Some(err) => Err(err),
                    None => Ok(0),
                };
            }
        };
        // Clear-then-rebuild buffers a `delete_graph` plus a series of
        // `upsert`s into the Tantivy writer before the final `commit()`. If any
        // step fails partway through, the half-built batch must be rolled back
        // before returning — otherwise those ops stay queued in the shared
        // writer and a later unrelated `commit()` (e.g. from a node write)
        // would flush a *partial* rebuild, silently dropping the graph's prior
        // index contents and leaving hard-to-debug drift.
        let rebuild = || -> Result<usize, DynoError> {
            ti.delete_graph(graph_id)
                .map_err(|e| DynoError::Storage(format!("full-text reindex clear failed: {e}")))?;
            // Snapshot the fulltext node types first — `scan_nodes` borrows &self.
            let node_types: Vec<String> = self
                .schema
                .node_types
                .keys()
                .filter(|nt| self.schema.has_fulltext_properties(nt))
                .cloned()
                .collect();
            let mut count = 0usize;
            for node_type in node_types {
                for node in self.scan_nodes(graph_id, &node_type)? {
                    let fields = self.fulltext_fields(&node_type, &node.properties);
                    ti.upsert(graph_id, &node_type, &node.node_id, &fields)
                        .map_err(|e| {
                            DynoError::Storage(format!("full-text reindex upsert failed: {e}"))
                        })?;
                    count += 1;
                }
            }
            ti.commit()
                .map_err(|e| DynoError::Storage(format!("full-text reindex commit failed: {e}")))?;
            Ok(count)
        };
        // Best-effort rollback to drop the uncommitted batch on failure;
        // surface the original rebuild error regardless of the rollback result.
        rebuild().inspect_err(|_| {
            let _ = ti.rollback();
        })
    }

    // =========================================================================
    // Embedding sidecar (CF_EMBEDDINGS)
    // =========================================================================

    /// Set the embedding for an existing node. Fails loud (`NodeNotFound`)
    /// if the node doesn't exist — silently creating an embedding for a
    /// non-existent node would orphan it forever (no `delete_node`
    /// cascade target). Empty embeddings are rejected: zero-dim vectors
    /// are meaningless and would foot-gun slice 8b's HNSW config. Re-set
    /// overwrites in place.
    pub fn set_embedding(
        &mut self,
        graph_id: &str,
        node_type: &str,
        node_id: &str,
        embedding: &[f32],
    ) -> Result<(), DynoError> {
        if embedding.is_empty() {
            return Err(DynoError::Validation {
                node_type: node_type.to_string(),
                property: "embedding".to_string(),
                message: "embedding must be non-empty".to_string(),
            });
        }
        // Reject non-finite components (NaN / ±∞) before the existence
        // read or any write — a non-finite float is corrupt data, not a
        // measurement, and it poisons every distance computation that
        // later reads it back. Checked here (cheaper than the DB read
        // below) and again on decode, so neither the write nor the read
        // path can admit it. (Zero-magnitude is a *similarity* concern,
        // screened at the service boundary, not here.)
        if let Some(i) = embedding.iter().position(|f| !f.is_finite()) {
            return Err(DynoError::Validation {
                node_type: node_type.to_string(),
                property: "embedding".to_string(),
                message: format!("embedding component at index {i} is non-finite (NaN or ±∞)"),
            });
        }
        let key = crate::keys::node_key(graph_id, node_type, node_id);
        if self.get(CF_NODES, &key)?.is_none() {
            return Err(DynoError::NodeNotFound {
                node_type: node_type.to_string(),
                node_id: node_id.to_string(),
            });
        }
        let mut bytes = Vec::with_capacity(embedding.len() * 4);
        for f in embedding {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        self.put(CF_EMBEDDINGS, key, bytes)?;
        Ok(())
    }

    /// Returns the embedding for a node, or `None` if no embedding has
    /// been set. Doesn't distinguish "node doesn't exist" from "node
    /// exists but has no embedding"; the caller is expected to have
    /// asserted node existence separately when that distinction matters.
    pub fn get_embedding(
        &self,
        graph_id: &str,
        node_type: &str,
        node_id: &str,
    ) -> Result<Option<Vec<f32>>, DynoError> {
        let key = crate::keys::node_key(graph_id, node_type, node_id);
        match self.get(CF_EMBEDDINGS, &key)? {
            Some(bytes) => Ok(Some(decode_embedding(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Drop an embedding. Returns `true` if one existed. Idempotent
    /// (a missing embedding is `Ok(false)`, not an error) — `delete_node`
    /// calls this unconditionally for cascade cleanup.
    pub fn delete_embedding(
        &mut self,
        graph_id: &str,
        node_type: &str,
        node_id: &str,
    ) -> Result<bool, DynoError> {
        let key = crate::keys::node_key(graph_id, node_type, node_id);
        let existed = self.get(CF_EMBEDDINGS, &key)?.is_some();
        self.delete(CF_EMBEDDINGS, &key)?;
        Ok(existed)
    }

    /// Walk every embedding of a given node type. Slice 8b will use
    /// this on rehydrate to populate the in-memory HNSW per-type
    /// indexes; not on the hot search path.
    pub fn scan_embeddings_by_type(
        &self,
        graph_id: &str,
        node_type: &str,
    ) -> Result<Vec<(String, Vec<f32>)>, DynoError> {
        let prefix = crate::keys::node_type_prefix(graph_id, node_type);
        let entries = self.prefix_scan(CF_EMBEDDINGS, &prefix)?;
        let mut results = Vec::with_capacity(entries.len());
        for (key, bytes) in entries {
            let after_prefix = &key[prefix.len()..];
            let node_id = decode_key_segment(after_prefix, "embedding key node_id suffix")?;
            results.push((node_id, decode_embedding(&bytes)?));
        }
        Ok(results)
    }

    // =========================================================================
    // Internal backend operations
    // =========================================================================

    fn put(&mut self, cf: &str, key: Vec<u8>, value: Vec<u8>) -> Result<(), DynoError> {
        // If batching, buffer the write — don't invalidate cache yet because
        // the data isn't on disk. Cache invalidation happens in commit_batch().
        if let Some(ref mut buffer) = self.write_buffer {
            let cf_id = CfId::from_str(cf)
                .ok_or_else(|| DynoError::Storage(format!("Unknown CF: {}", cf)))?;
            buffer.push(BufferedOp::Put {
                cf: cf_id,
                key,
                value,
            });
            return Ok(());
        }

        self.read_cache
            .lock()
            .expect("read_cache lock poisoned")
            .invalidate(&key);

        self.backend.put(cf, key, value)
    }

    fn get(&self, cf: &str, key: &[u8]) -> Result<Option<Arc<[u8]>>, DynoError> {
        // Buffer wins over backend. Reverse-walk so a late Put resurrects
        // a key tombstoned by an earlier PrefixDelete in the same batch.
        // The cache is bypassed for buffer-served reads — the value isn't
        // on disk yet, so caching it would risk a stale view on discard.
        if let (Some(buffer), Some(cf_id)) = (self.write_buffer.as_ref(), CfId::from_str(cf))
            && !buffer.is_empty()
            && let Some(effect) = buffer.iter().rev().find_map(|op| op.affecting(cf_id, key))
        {
            return Ok(match effect {
                BufferedEffect::Put(v) => Some(Arc::<[u8]>::from(v)),
                BufferedEffect::Tombstoned => None,
            });
        }

        // For node lookups, use the read cache (single lock acquisition)
        if cf == CF_NODES {
            let mut cache = self.read_cache.lock().expect("read_cache lock poisoned");
            if let Some(data) = cache.get(key) {
                return Ok(Some(data));
            }
            drop(cache); // Release lock before backend read

            let result = self.backend_get(cf, key)?;
            if let Some(ref data) = result {
                self.read_cache
                    .lock()
                    .unwrap()
                    .put(key.to_vec(), Arc::clone(data));
            }
            return Ok(result);
        }

        self.backend_get(cf, key)
    }

    fn backend_get(&self, cf: &str, key: &[u8]) -> Result<Option<Arc<[u8]>>, DynoError> {
        self.backend.get(cf, key)
    }

    /// Delete a key. Idempotent — deleting a missing key is a no-op,
    /// not an error. Public callers that need an existence-bool should
    /// `get` first; embedding the bool here cost a disk read per delete
    /// and only two of nine internal callers used it.
    fn delete(&mut self, cf: &str, key: &[u8]) -> Result<(), DynoError> {
        if let Some(ref mut buffer) = self.write_buffer {
            let cf_id = CfId::from_str(cf)
                .ok_or_else(|| DynoError::Storage(format!("Unknown CF: {}", cf)))?;
            buffer.push(BufferedOp::Delete {
                cf: cf_id,
                key: key.to_vec(),
            });
            return Ok(());
        }

        self.read_cache
            .lock()
            .expect("read_cache lock poisoned")
            .invalidate(key);
        self.backend.delete(cf, key)
    }

    /// Scan all keys with a given prefix in a column family.
    #[allow(
        clippy::type_complexity,
        reason = "raw KV pairs straight out of RocksDB; an alias would only obscure"
    )]
    fn prefix_scan(&self, cf: &str, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, DynoError> {
        let backend_results = self.backend.prefix_scan(cf, prefix)?;

        // Buffer wins over backend on scans. Skip the overlay alloc
        // entirely if no batch is active or no buffered op touches this
        // CF + prefix range — the common case for reads outside a batch.
        if let (Some(buffer), Some(cf_id)) = (self.write_buffer.as_ref(), CfId::from_str(cf))
            && !buffer.is_empty()
            && Self::buffer_touches_scan(buffer, cf_id, prefix)
        {
            return Ok(Self::overlay_buffer_on_scan(
                backend_results,
                buffer,
                cf_id,
                prefix,
            ));
        }

        Ok(backend_results)
    }

    /// Cheap pre-flight: does any buffered op affect this scan range?
    /// A `PrefixDelete` matches if its prefix overlaps `prefix` in either
    /// direction (sub-range delete OR superset clear); `Put`/`Delete`
    /// match if their key starts with `prefix`.
    fn buffer_touches_scan(buffer: &[BufferedOp], cf_id: CfId, prefix: &[u8]) -> bool {
        buffer.iter().any(|op| {
            if op.cf() != cf_id {
                return false;
            }
            match op {
                BufferedOp::Put { key, .. } | BufferedOp::Delete { key, .. } => {
                    key.starts_with(prefix)
                }
                BufferedOp::PrefixDelete { prefix: p, .. } => {
                    p.starts_with(prefix) || prefix.starts_with(p)
                }
            }
        })
    }

    /// Apply buffered ops to a backend scan result in insertion order.
    /// Late puts can resurrect a key that an earlier `PrefixDelete` in
    /// the same batch tombstoned — ordered application required.
    /// `HashMap` (not `BTreeMap`): callers don't depend on key order.
    fn overlay_buffer_on_scan(
        backend_results: Vec<(Vec<u8>, Vec<u8>)>,
        buffer: &[BufferedOp],
        cf_id: CfId,
        prefix: &[u8],
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        use std::collections::HashMap;

        let mut by_key: HashMap<Vec<u8>, Vec<u8>> = backend_results.into_iter().collect();

        for op in buffer {
            if op.cf() != cf_id {
                continue;
            }
            match op {
                BufferedOp::Put { key, value, .. } if key.starts_with(prefix) => {
                    by_key.insert(key.clone(), value.clone());
                }
                BufferedOp::Delete { key, .. } if key.starts_with(prefix) => {
                    by_key.remove(key);
                }
                BufferedOp::PrefixDelete {
                    prefix: del_prefix, ..
                } => {
                    // del_prefix may extend past `prefix` (sub-range delete)
                    // or may BE `prefix` (clears whole scan) — both correct
                    // under starts_with.
                    by_key.retain(|k, _| !k.starts_with(del_prefix));
                }
                _ => {}
            }
        }

        by_key.into_iter().collect()
    }

    /// Delete all keys with a given prefix in a column family.
    fn prefix_delete(&mut self, cf: &str, prefix: &[u8]) -> Result<(), DynoError> {
        if let Some(ref mut buffer) = self.write_buffer {
            let cf_id = CfId::from_str(cf)
                .ok_or_else(|| DynoError::Storage(format!("Unknown CF: {}", cf)))?;
            buffer.push(BufferedOp::PrefixDelete {
                cf: cf_id,
                prefix: prefix.to_vec(),
            });
            return Ok(());
        }
        // Invalidate cached entries under this prefix before deleting —
        // the non-batch path must mirror what `commit_batch` does for
        // buffered PrefixDeletes. Without this, `clear_graph` (which
        // prefix-deletes CF_NODES) could keep serving cached nodes of a
        // cleared graph until their TTL. The cache holds only CF_NODES
        // bodies, so an adjacency-prefix delete here is a harmless no-op.
        self.read_cache
            .lock()
            .expect("read_cache lock poisoned")
            .invalidate_prefix(prefix);
        self.backend.prefix_delete(cf, prefix)
    }

    // =========================================================================
    // Public API (unchanged from before)
    // =========================================================================

    /// Indexed property names for a node type, owned so callers can free the
    /// schema borrow before doing `&mut self` writes.
    fn indexed_property_names(&self, node_type: &str) -> Vec<String> {
        self.schema
            .indexed_properties(node_type)
            .into_iter()
            .map(String::from)
            .collect()
    }

    /// Write CF_NODE_IDX entries for every indexed property present in
    /// `properties`. Skips properties whose value type isn't supported by
    /// `value_to_index_bytes` (floats, lists, maps, null).
    fn write_index_entries(
        &mut self,
        graph_id: &str,
        node_type: &str,
        node_id: &str,
        properties: &HashMap<String, Value>,
    ) -> Result<(), DynoError> {
        let indexed = self.indexed_property_names(node_type);
        for prop_name in indexed {
            let Some(value) = properties.get(&prop_name) else {
                continue;
            };
            let Some(bytes) = crate::keys::value_to_index_bytes(value) else {
                continue;
            };
            let key = crate::keys::node_idx_key(graph_id, node_type, &prop_name, &bytes, node_id);
            self.put(CF_NODE_IDX, key, Vec::new())?;
        }
        Ok(())
    }

    /// Delete CF_NODE_IDX entries matching the indexed property values in
    /// `properties`. Used during node delete and as the first half of update.
    fn delete_index_entries(
        &mut self,
        graph_id: &str,
        node_type: &str,
        node_id: &str,
        properties: &HashMap<String, Value>,
    ) -> Result<(), DynoError> {
        let indexed = self.indexed_property_names(node_type);
        for prop_name in indexed {
            let Some(value) = properties.get(&prop_name) else {
                continue;
            };
            let Some(bytes) = crate::keys::value_to_index_bytes(value) else {
                continue;
            };
            let key = crate::keys::node_idx_key(graph_id, node_type, &prop_name, &bytes, node_id);
            self.delete(CF_NODE_IDX, &key)?;
        }
        Ok(())
    }

    /// Reject node-write key segments that contain the NUL separator
    /// *before* anything is persisted. A NUL in `graph_id`, `node_id`,
    /// or an indexed string property value would make the encoded key
    /// ambiguous on decode (scans split at the wrong position, forged
    /// index boundaries), silently corrupting later reads — so we fail
    /// loud at write time.
    /// `node_type` is already constrained by `validate_node` (it must be
    /// a declared type), and non-indexed property values live in the
    /// msgpack body, not in keys, so neither needs checking here.
    fn validate_node_key_segments(
        &self,
        graph_id: &str,
        node_type: &str,
        node_id: &str,
        properties: &HashMap<String, Value>,
    ) -> Result<(), DynoError> {
        crate::keys::validate_key_segment("graph_id", graph_id)?;
        crate::keys::validate_key_segment("node_id", node_id)?;
        // Validation is `&self`, so iterate the borrowed `&str` names
        // directly — no need for the owned-`String` copy or the
        // has-indexed fast-path that `write_index_entries` needs on the
        // `&mut self` write path. Empty for non-indexed types.
        for prop_name in self.schema.indexed_properties(node_type) {
            if let Some(Value::String(s)) = properties.get(prop_name) {
                crate::keys::validate_key_segment(prop_name, s)?;
            }
        }
        Ok(())
    }

    /// Create a node with schema validation.
    pub fn create_node(
        &mut self,
        graph_id: &str,
        node_type: &str,
        node_id: &str,
        mut properties: HashMap<String, Value>,
    ) -> Result<StoredNode, DynoError> {
        // validate_node mutates `properties` to apply schema defaults.
        self.schema.validate_node(node_type, &mut properties)?;
        // Reject NUL-bearing key segments before any put (an orphaned
        // CF_NODES body would otherwise outlive a later-rejected index
        // write).
        self.validate_node_key_segments(graph_id, node_type, node_id, &properties)?;

        let key = crate::keys::node_key(graph_id, node_type, node_id);
        let has_indexed = self.schema.has_indexed_properties(node_type);

        // Create-or-replace semantics: if a node already exists at this id,
        // reconcile its reverse-index entries before overwriting. A bare
        // create would leave the prior (value -> node_id) entries dangling
        // under stale values, corrupting `scan_nodes_by_property`. Mirrors
        // the diff in `replace_node_properties`.
        if has_indexed && let Some(old_bytes) = self.get(CF_NODES, &key)? {
            let old: HashMap<String, Value> = rmp_serde::from_slice(&old_bytes)
                .map_err(|e| DynoError::Serialization(e.to_string()))?;
            self.delete_index_entries(graph_id, node_type, node_id, &old)?;
        }

        let value =
            rmp_serde::to_vec(&properties).map_err(|e| DynoError::Serialization(e.to_string()))?;

        self.put(CF_NODES, key, value)?;
        if has_indexed {
            self.write_index_entries(graph_id, node_type, node_id, &properties)?;
        }
        #[cfg(feature = "fulltext")]
        self.fulltext_upsert(graph_id, node_type, node_id, &properties)?;

        Ok(StoredNode {
            graph_id: graph_id.to_string(),
            node_type: node_type.to_string(),
            node_id: node_id.to_string(),
            properties,
        })
    }

    /// Get a node by ID.
    pub fn get_node(
        &self,
        graph_id: &str,
        node_type: &str,
        node_id: &str,
    ) -> Result<Option<StoredNode>, DynoError> {
        let key = crate::keys::node_key(graph_id, node_type, node_id);
        match self.get(CF_NODES, &key)? {
            Some(bytes) => {
                let properties: HashMap<String, Value> = rmp_serde::from_slice(&bytes)
                    .map_err(|e| DynoError::Serialization(e.to_string()))?;
                Ok(Some(StoredNode {
                    graph_id: graph_id.to_string(),
                    node_type: node_type.to_string(),
                    node_id: node_id.to_string(),
                    properties,
                }))
            }
            None => Ok(None),
        }
    }

    /// Delete a node and every edge attached to it, including the
    /// peer-side adjacency entries on neighbor nodes.
    ///
    /// To update a node's properties in place, use
    /// `replace_node_properties` — delete-and-recreate-with-the-same-id
    /// drops every edge attached to the node.
    ///
    /// Cleanup steps (in order, with rationale):
    /// 1. Scan `node_id`'s outgoing and incoming adjacency *before*
    ///    touching anything — once we delete this node's own adjacency
    ///    prefix in step 4, we lose the information needed to find the
    ///    peer-side keys that need cleaning up.
    /// 2. Delete the node from CF_NODES + reconcile any reverse-index
    ///    entries.
    /// 3. Prefix-delete this node's own outgoing + incoming adjacency.
    /// 4. NEW: for every edge involving this node, also delete (a) the
    ///    edge from CF_EDGES and (b) the symmetric adjacency entry on
    ///    the peer node. Without this step the storage was leaving
    ///    dangling edges behind delete (tech-debt C1 — `get_edge` would
    ///    still resolve, `scan_incoming_edges` on a peer would still
    ///    return the deleted endpoint).
    pub fn delete_node(
        &mut self,
        graph_id: &str,
        node_type: &str,
        node_id: &str,
    ) -> Result<bool, DynoError> {
        // Step 1: capture peer-side cleanup info before any deletes.
        let outgoing = self.scan_outgoing_edges(graph_id, node_id, None)?;
        let incoming = self.scan_incoming_edges(graph_id, node_id, None)?;

        // Step 2: own-node cleanup. We need the existence answer for the
        // public-API bool, and (when this type has indexed properties) the
        // stored properties to reconcile reverse-index entries — fold both
        // into a single `get` and decode the value lazily.
        let key = crate::keys::node_key(graph_id, node_type, node_id);
        let raw = self.get(CF_NODES, &key)?;
        let existed = raw.is_some();
        let old_properties = if self.schema.has_indexed_properties(node_type) {
            raw.map(|bytes| {
                rmp_serde::from_slice::<HashMap<String, Value>>(&bytes)
                    .map_err(|e| DynoError::Serialization(e.to_string()))
            })
            .transpose()?
        } else {
            None
        };
        self.delete(CF_NODES, &key)?;
        if let Some(props) = old_properties {
            self.delete_index_entries(graph_id, node_type, node_id, &props)?;
        }

        // Step 3: own adjacency.
        let out_prefix = crate::keys::adj_out_prefix(graph_id, node_id);
        let in_prefix = crate::keys::adj_in_prefix(graph_id, node_id);
        self.prefix_delete(CF_ADJ_OUT, &out_prefix)?;
        self.prefix_delete(CF_ADJ_IN, &in_prefix)?;

        // Step 4: edge + peer-adjacency cleanup.
        for edge in outgoing {
            let edge_key = crate::keys::edge_key(graph_id, &edge.edge_type, node_id, &edge.to_id);
            self.delete(CF_EDGES, &edge_key)?;
            let peer_in = crate::keys::adj_in_key(graph_id, &edge.to_id, &edge.edge_type, node_id);
            self.delete(CF_ADJ_IN, &peer_in)?;
        }
        for edge in incoming {
            let edge_key = crate::keys::edge_key(graph_id, &edge.edge_type, &edge.from_id, node_id);
            self.delete(CF_EDGES, &edge_key)?;
            let peer_out =
                crate::keys::adj_out_key(graph_id, &edge.from_id, &edge.edge_type, node_id);
            self.delete(CF_ADJ_OUT, &peer_out)?;
        }

        // Step 5: drop the sidecar embedding if any. Idempotent — most
        // nodes won't have one.
        let emb_key = crate::keys::node_key(graph_id, node_type, node_id);
        self.delete(CF_EMBEDDINGS, &emb_key)?;

        // Step 6: drop the full-text document if the node existed (skipping the
        // commit cost when there was nothing to delete).
        #[cfg(feature = "fulltext")]
        if existed {
            self.fulltext_delete(graph_id, node_type, node_id)?;
        }

        Ok(existed)
    }

    /// REPLACE a node's properties — the new map is the complete new state;
    /// any property not in `properties` is dropped from the stored node
    /// (subject to schema defaults being re-applied by `validate_node`).
    /// Use `merge_edge_properties` as the analogous shape for partial-update
    /// semantics on edges; if you need merge semantics for nodes, do a
    /// `get_node` + caller-side merge + `replace_node_properties` round-trip.
    /// Edges + adjacency entries are left untouched. Returns `Ok(None)`
    /// when the node doesn't exist.
    pub fn replace_node_properties(
        &mut self,
        graph_id: &str,
        node_type: &str,
        node_id: &str,
        mut properties: HashMap<String, Value>,
    ) -> Result<Option<StoredNode>, DynoError> {
        let key = crate::keys::node_key(graph_id, node_type, node_id);
        let has_indexed = self.schema.has_indexed_properties(node_type);

        // When the type has indexed properties we need old values to drive
        // `delete_index_entries`. Otherwise just confirm existence — skipping
        // the msgpack decode that the pre-index path avoided.
        let old_properties: Option<HashMap<String, Value>> = if has_indexed {
            match self.get(CF_NODES, &key)? {
                Some(bytes) => Some(
                    rmp_serde::from_slice(&bytes)
                        .map_err(|e| DynoError::Serialization(e.to_string()))?,
                ),
                None => return Ok(None),
            }
        } else {
            if self.get(CF_NODES, &key)?.is_none() {
                return Ok(None);
            }
            None
        };

        // validate_node mutates `properties` to apply schema defaults.
        self.schema.validate_node(node_type, &mut properties)?;
        // Reject NUL-bearing key segments before the put — the new
        // indexed string values are the corruption vector on replace.
        self.validate_node_key_segments(graph_id, node_type, node_id, &properties)?;

        let value =
            rmp_serde::to_vec(&properties).map_err(|e| DynoError::Serialization(e.to_string()))?;
        self.put(CF_NODES, key, value)?;

        // Diff indexed properties: drop entries whose old value no longer
        // matches, add entries for new values. Unchanged values are a wash —
        // simplest is drop-all-old + write-all-new, since each put/delete is
        // a single KV operation and RocksDB tombstones collapse at compaction.
        if let Some(old) = old_properties {
            self.delete_index_entries(graph_id, node_type, node_id, &old)?;
            self.write_index_entries(graph_id, node_type, node_id, &properties)?;
        }
        // Re-mirror full-text: `upsert` has replace semantics, so the prior
        // document (if any) is overwritten with the new property values.
        #[cfg(feature = "fulltext")]
        self.fulltext_upsert(graph_id, node_type, node_id, &properties)?;

        Ok(Some(StoredNode {
            graph_id: graph_id.to_string(),
            node_type: node_type.to_string(),
            node_id: node_id.to_string(),
            properties,
        }))
    }

    /// Create an edge with schema validation.
    #[allow(
        clippy::too_many_arguments,
        reason = "edges are inherently 4-endpoint values; a builder would only push the count out of one signature into another"
    )]
    pub fn create_edge(
        &mut self,
        graph_id: &str,
        edge_type: &str,
        from_type: &str,
        from_id: &str,
        to_type: &str,
        to_id: &str,
        mut properties: HashMap<String, Value>,
    ) -> Result<StoredEdge, DynoError> {
        self.schema.validate_edge(edge_type, from_type, to_type)?;
        self.schema
            .validate_edge_properties(edge_type, &mut properties)?;
        // Reject NUL-bearing key segments before any put. `edge_type`
        // is already constrained by `validate_edge`; `graph_id` and the
        // endpoint ids are the unguarded segments. (create_edge does
        // not require the endpoints to pre-exist, so a NUL id here
        // would otherwise persist a corrupt adjacency key.)
        crate::keys::validate_key_segment("graph_id", graph_id)?;
        crate::keys::validate_key_segment("from_id", from_id)?;
        crate::keys::validate_key_segment("to_id", to_id)?;

        let edge_key = crate::keys::edge_key(graph_id, edge_type, from_id, to_id);
        let adj_out = crate::keys::adj_out_key(graph_id, from_id, edge_type, to_id);
        let adj_in = crate::keys::adj_in_key(graph_id, to_id, edge_type, from_id);

        let value =
            rmp_serde::to_vec(&properties).map_err(|e| DynoError::Serialization(e.to_string()))?;

        self.put(CF_EDGES, edge_key, value.clone())?;
        self.put(CF_ADJ_OUT, adj_out, value.clone())?;
        self.put(CF_ADJ_IN, adj_in, value)?;

        Ok(StoredEdge {
            graph_id: graph_id.to_string(),
            edge_type: edge_type.to_string(),
            from_id: from_id.to_string(),
            to_id: to_id.to_string(),
            properties,
        })
    }

    /// Delete an edge and its adjacency entries.
    pub fn delete_edge(
        &mut self,
        graph_id: &str,
        edge_type: &str,
        from_id: &str,
        to_id: &str,
    ) -> Result<bool, DynoError> {
        let edge_key = crate::keys::edge_key(graph_id, edge_type, from_id, to_id);
        let existed = self.get(CF_EDGES, &edge_key)?.is_some();
        self.delete(CF_EDGES, &edge_key)?;

        let adj_out = crate::keys::adj_out_key(graph_id, from_id, edge_type, to_id);
        let adj_in = crate::keys::adj_in_key(graph_id, to_id, edge_type, from_id);
        self.delete(CF_ADJ_OUT, &adj_out)?;
        self.delete(CF_ADJ_IN, &adj_in)?;

        Ok(existed)
    }

    /// MERGE properties into an existing edge — `updates` overlays the
    /// existing properties, missing keys are preserved. Read-merge-write
    /// across all 3 CFs (CF_EDGES + adj_out + adj_in). Counterpart of
    /// `replace_node_properties` (which is REPLACE, not merge — see that
    /// method's doc for why the asymmetry exists at the storage layer).
    /// Returns `Ok(None)` when the edge doesn't exist.
    pub fn merge_edge_properties(
        &mut self,
        graph_id: &str,
        edge_type: &str,
        from_id: &str,
        to_id: &str,
        updates: HashMap<String, Value>,
    ) -> Result<Option<StoredEdge>, DynoError> {
        let edge_key = crate::keys::edge_key(graph_id, edge_type, from_id, to_id);
        let existing = match self.get(CF_EDGES, &edge_key)? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };

        let mut properties: HashMap<String, Value> = rmp_serde::from_slice(&existing)
            .map_err(|e| DynoError::Serialization(e.to_string()))?;

        for (k, v) in updates {
            properties.insert(k, v);
        }

        let value =
            rmp_serde::to_vec(&properties).map_err(|e| DynoError::Serialization(e.to_string()))?;

        let adj_out = crate::keys::adj_out_key(graph_id, from_id, edge_type, to_id);
        let adj_in = crate::keys::adj_in_key(graph_id, to_id, edge_type, from_id);

        self.put(CF_EDGES, edge_key, value.clone())?;
        self.put(CF_ADJ_OUT, adj_out, value.clone())?;
        self.put(CF_ADJ_IN, adj_in, value)?;

        Ok(Some(StoredEdge {
            graph_id: graph_id.to_string(),
            edge_type: edge_type.to_string(),
            from_id: from_id.to_string(),
            to_id: to_id.to_string(),
            properties,
        }))
    }

    /// Get an edge.
    pub fn get_edge(
        &self,
        graph_id: &str,
        edge_type: &str,
        from_id: &str,
        to_id: &str,
    ) -> Result<Option<StoredEdge>, DynoError> {
        let key = crate::keys::edge_key(graph_id, edge_type, from_id, to_id);
        match self.get(CF_EDGES, &key)? {
            Some(bytes) => {
                let properties: HashMap<String, Value> = rmp_serde::from_slice(&bytes)
                    .map_err(|e| DynoError::Serialization(e.to_string()))?;
                Ok(Some(StoredEdge {
                    graph_id: graph_id.to_string(),
                    edge_type: edge_type.to_string(),
                    from_id: from_id.to_string(),
                    to_id: to_id.to_string(),
                    properties,
                }))
            }
            None => Ok(None),
        }
    }

    /// Scan nodes of a type filtered by a schema-declared indexed property.
    ///
    /// Prefix-scans `CF_NODE_IDX` + point-looks-up each matching node.
    /// Complexity is O(matching_nodes) regardless of total graph size.
    /// Assumes the index has been kept consistent by write-path hooks since
    /// the first write — no fallback for pre-existing un-indexed data.
    ///
    /// Returns `Ok(vec![])` for unsupported value types (`Float`/`List`/
    /// `Map`/`Null`) — those are never stored in the index by design.
    pub fn scan_nodes_by_property(
        &self,
        graph_id: &str,
        node_type: &str,
        prop_name: &str,
        prop_value: &Value,
    ) -> Result<Vec<StoredNode>, DynoError> {
        // Read-side mirror of the write guard: a NUL in a string query
        // value can't match any stored key (writes reject it), so an
        // unguarded scan would silently return empty — masking a
        // structurally-invalid query as "no matches". Fail loud instead.
        if let Value::String(s) = prop_value {
            crate::keys::validate_key_segment(prop_name, s)?;
        }
        let Some(value_bytes) = crate::keys::value_to_index_bytes(prop_value) else {
            return Ok(Vec::new());
        };

        let value_prefix =
            crate::keys::node_idx_value_prefix(graph_id, node_type, prop_name, &value_bytes);
        let entries = self.prefix_scan(CF_NODE_IDX, &value_prefix)?;

        let mut results = Vec::with_capacity(entries.len());
        for (key, _) in entries {
            let Some(node_id_bytes) = crate::keys::node_idx_key_node_id(&key, &value_prefix) else {
                continue;
            };
            let node_id = decode_key_segment(node_id_bytes, "node_idx key node_id suffix")?;
            if let Some(node) = self.get_node(graph_id, node_type, &node_id)? {
                results.push(node);
            }
        }
        Ok(results)
    }

    /// Count all nodes of a given type in a graph.
    pub fn count_nodes(&self, graph_id: &str, node_type: &str) -> Result<usize, DynoError> {
        let prefix = crate::keys::node_type_prefix(graph_id, node_type);
        // Propagate scan failures rather than collapsing them to 0: a
        // transient I/O error, a missing CF, or a buffer-overlay error
        // would otherwise be indistinguishable from a genuinely empty
        // graph, silently misleading the caller.
        Ok(self.prefix_scan(CF_NODES, &prefix)?.len())
    }

    /// Scan all nodes of a given type in a graph.
    pub fn scan_nodes(
        &self,
        graph_id: &str,
        node_type: &str,
    ) -> Result<Vec<StoredNode>, DynoError> {
        let prefix = crate::keys::node_type_prefix(graph_id, node_type);
        let entries = self.prefix_scan(CF_NODES, &prefix)?;
        let mut results = Vec::new();

        for (key, bytes) in entries {
            let properties: HashMap<String, Value> = rmp_serde::from_slice(&bytes)
                .map_err(|e| DynoError::Serialization(e.to_string()))?;
            let after_prefix = &key[prefix.len()..];
            let node_id = decode_key_segment(after_prefix, "node key node_id suffix")?;
            results.push(StoredNode {
                graph_id: graph_id.to_string(),
                node_type: node_type.to_string(),
                node_id,
                properties,
            });
        }

        Ok(results)
    }

    /// Scan outgoing edges from a node, optionally filtered by edge type.
    pub fn scan_outgoing_edges(
        &self,
        graph_id: &str,
        from_id: &str,
        edge_type_filter: Option<&str>,
    ) -> Result<Vec<StoredEdge>, DynoError> {
        let prefix = crate::keys::adj_out_prefix(graph_id, from_id);
        let entries = self.prefix_scan(CF_ADJ_OUT, &prefix)?;
        let mut results = Vec::new();

        for (key, bytes) in entries {
            let after_prefix = &key[prefix.len()..];
            let parts: Vec<&[u8]> = after_prefix.splitn(2, |&b| b == 0x00).collect();
            if parts.len() != 2 {
                continue;
            }
            let edge_type = decode_key_segment(parts[0], "adj_out key edge_type")?;
            let to_id = decode_key_segment(parts[1], "adj_out key to_id")?;

            if let Some(filter) = edge_type_filter
                && edge_type != filter
            {
                continue;
            }

            let properties: HashMap<String, Value> = rmp_serde::from_slice(&bytes)
                .map_err(|e| DynoError::Serialization(e.to_string()))?;

            results.push(StoredEdge {
                graph_id: graph_id.to_string(),
                edge_type,
                from_id: from_id.to_string(),
                to_id,
                properties,
            });
        }

        Ok(results)
    }

    /// Scan incoming edges to a node (reverse adjacency).
    /// Key format in CF_ADJ_IN: `{graph_id}\x00{to_id}\x00{edge_type}\x00{from_id}`
    pub fn scan_incoming_edges(
        &self,
        graph_id: &str,
        to_id: &str,
        edge_type_filter: Option<&str>,
    ) -> Result<Vec<StoredEdge>, DynoError> {
        let prefix = crate::keys::adj_in_prefix(graph_id, to_id);
        let entries = self.prefix_scan(CF_ADJ_IN, &prefix)?;
        let mut results = Vec::new();

        for (key, bytes) in entries {
            let after_prefix = &key[prefix.len()..];
            let parts: Vec<&[u8]> = after_prefix.splitn(2, |&b| b == 0x00).collect();
            if parts.len() != 2 {
                continue;
            }
            let edge_type = decode_key_segment(parts[0], "adj_in key edge_type")?;
            let from_id = decode_key_segment(parts[1], "adj_in key from_id")?;

            if let Some(filter) = edge_type_filter
                && edge_type != filter
            {
                continue;
            }

            let properties: HashMap<String, Value> = rmp_serde::from_slice(&bytes)
                .map_err(|e| DynoError::Serialization(e.to_string()))?;

            results.push(StoredEdge {
                graph_id: graph_id.to_string(),
                edge_type,
                from_id,
                to_id: to_id.to_string(),
                properties,
            });
        }

        Ok(results)
    }

    // =========================================================================
    // Write batching
    // =========================================================================

    /// Begin buffering writes. All subsequent `put()` calls will be buffered
    /// instead of committed immediately. Call `commit_batch()` to write all
    /// buffered operations atomically.
    pub fn begin_batch(&mut self) {
        if self.write_buffer.is_some() {
            tracing::warn!(
                "begin_batch() called while batch already active — committing previous batch"
            );
            let _ = self.commit_batch();
        }
        // Establish a clean full-text baseline before the batch buffers its own
        // ops. `discard_batch` reverts via a writer-global `rollback()`, which is
        // only correct if the writer holds *exactly* this batch's ops. A prior
        // non-batched write whose index commit failed (the node is in RocksDB,
        // its index op left uncommitted) would otherwise be dropped by that
        // rollback — silent, permanent drift. Committing here flushes any such
        // stranded op (it matches a durable RocksDB write, so committing is the
        // correct heal) and guarantees the writer is clean when buffering starts.
        #[cfg(feature = "fulltext")]
        if let Some(ti) = &self.text_index
            && let Err(e) = ti.commit()
        {
            tracing::error!("full-text commit at begin_batch failed: {e}");
        }
        self.write_buffer = Some(Vec::new());
    }

    /// Returns true if write batching is currently active.
    pub fn is_batching(&self) -> bool {
        self.write_buffer.is_some()
    }

    /// Commit all buffered writes as a single atomic operation.
    /// For RocksDB, this uses `WriteBatch` for atomic multi-CF writes.
    /// For in-memory backend, applies writes directly.
    pub fn commit_batch(&mut self) -> Result<usize, DynoError> {
        let buffer = match self.write_buffer.take() {
            Some(b) => b,
            None => return Ok(0),
        };

        let count = buffer.len();
        if count == 0 {
            return Ok(0);
        }

        // Invalidate cache before applying the batch so a concurrent
        // reader either sees pre-batch + cache-miss (re-fetches) or
        // post-batch + cache-miss — never stale data.
        {
            let mut cache = self.read_cache.lock().expect("read_cache lock poisoned");
            for op in &buffer {
                match op {
                    BufferedOp::Put { key, .. } | BufferedOp::Delete { key, .. } => {
                        cache.invalidate(key);
                    }
                    BufferedOp::PrefixDelete { prefix, .. } => {
                        cache.invalidate_prefix(prefix);
                    }
                }
            }
        }

        // Apply the buffered ops atomically. The backend owns the
        // all-or-nothing semantics (RocksDB `WriteBatch`; the in-memory
        // backend an in-order loop) and the `PrefixDelete`-supersedes-
        // earlier-puts ordering.
        self.backend.commit_batch(buffer)?;

        // Make this batch's buffered full-text writes visible now that the
        // authoritative backend write has landed. Reached only when count > 0
        // (the empty-buffer case returns early above), and any full-text op in
        // the batch rode alongside a node put/delete that's part of `count`.
        #[cfg(feature = "fulltext")]
        if let Some(ti) = &self.text_index {
            ti.commit()
                .map_err(|e| DynoError::Storage(format!("full-text batch commit failed: {e}")))?;
        }

        Ok(count)
    }

    /// Discard all buffered writes without committing.
    pub fn discard_batch(&mut self) {
        self.write_buffer = None;
        // Revert the batch's buffered full-text writes too. Best-effort: a
        // rollback failure can't be surfaced through this infallible signature,
        // and the index is rebuildable via `reindex_fulltext` regardless.
        #[cfg(feature = "fulltext")]
        if let Some(ti) = &self.text_index
            && let Err(e) = ti.rollback()
        {
            tracing::error!("full-text rollback after discard_batch failed: {e}");
        }
    }

    // =========================================================================
    // Graph lifecycle
    // =========================================================================

    /// Drop every key belonging to `graph_id` across all column
    /// families. Idempotent — clearing an unknown graph returns `Ok`
    /// since the post-condition holds either way. Routes through
    /// `prefix_delete` per CF, so write batching / cache invalidation
    /// / snapshot-interaction caveats match every other write path.
    pub fn clear_graph(&mut self, graph_id: &str) -> Result<(), DynoError> {
        let prefix = crate::keys::graph_prefix(graph_id);
        for cf in ALL_CFS {
            self.prefix_delete(cf, &prefix)?;
        }
        // Drop the graph's full-text documents too, so they don't outlive the
        // nodes. Commit cadence follows the batch state, as for node writes.
        #[cfg(feature = "fulltext")]
        if let Some(ti) = &self.text_index {
            ti.delete_graph(graph_id)
                .map_err(|e| DynoError::Storage(format!("full-text clear failed: {e}")))?;
            if self.write_buffer.is_none() {
                ti.commit()
                    .map_err(|e| DynoError::Storage(format!("full-text commit failed: {e}")))?;
            }
        }
        Ok(())
    }

    // =========================================================================
    // Cache management
    // =========================================================================

    /// Get cache statistics: (hits, misses, current_size).
    pub fn cache_stats(&self) -> (u64, u64, usize) {
        self.read_cache
            .lock()
            .expect("read_cache lock poisoned")
            .stats()
    }

    /// Clear the entire read cache.
    pub fn clear_cache(&self) {
        self.read_cache
            .lock()
            .expect("read_cache lock poisoned")
            .clear();
    }
}

/// Decode a key segment as UTF-8. Every key the storage layer writes
/// is built from `&str` arguments, so a non-UTF-8 byte sequence on
/// readback can only mean on-disk corruption — fail loud with the
/// surrounding context rather than mangling silently into a
/// believable-looking but fictional node/edge id.
fn decode_key_segment(bytes: &[u8], context: &str) -> Result<String, DynoError> {
    std::str::from_utf8(bytes)
        .map(|s| s.to_string())
        .map_err(|e| {
            DynoError::Storage(format!(
                "corrupt key segment ({context}): {e}; raw bytes = {bytes:?}"
            ))
        })
}

/// Decode raw f32-LE bytes to a `Vec<f32>`. Length must be a multiple
/// of 4; anything else means the on-disk value is corrupt and we fail
/// loud rather than truncating.
fn decode_embedding(bytes: &[u8]) -> Result<Vec<f32>, DynoError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(DynoError::Storage(format!(
            "embedding byte length {} is not a multiple of 4 (corrupt sidecar?)",
            bytes.len()
        )));
    }
    let mut floats = Vec::with_capacity(bytes.len() / 4);
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        let arr: [u8; 4] = chunk.try_into().expect("chunks_exact yields 4-byte slices");
        let f = f32::from_le_bytes(arr);
        // The write path rejects non-finite embeddings; a NaN/±∞ read
        // back here means the on-disk bytes are corrupt (or were written
        // before that guard existed). Fail loud rather than hand a
        // poison value to the distance functions.
        if !f.is_finite() {
            return Err(DynoError::Storage(format!(
                "embedding component at index {i} is non-finite (NaN or ±∞) — corrupt sidecar?"
            )));
        }
        floats.push(f);
    }
    Ok(floats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynograph_core::{Schema, props};

    fn test_schema() -> Schema {
        Schema::from_yaml(
            r#"
schema:
  name: test
  version: 1
  node_types:
    Character:
      properties:
        name:
          type: string
          required: true
        role:
          type: enum
          values: [protagonist, antagonist, supporting]
    Location:
      properties:
        name:
          type: string
          required: true
  edge_types:
    KNOWS:
      from: Character
      to: Character
    VISITS:
      from: Character
      to: Location
"#,
        )
        .unwrap()
    }

    /// Build a `StorageEngine` for a backend-agnostic engine test,
    /// honoring the `DYNOGRAPH_TEST_BACKEND` env var: unset (or
    /// `"memory"`) gives the in-memory backend, `"rocksdb"` a fresh
    /// temp-dir RocksDB store. CI runs the storage suite once per
    /// backend (the "backend matrix") so engine logic is exercised
    /// against both — the in-memory string map and the array-indexed
    /// column-family handles that diverge (the C1 batch panic only
    /// reproduced on RocksDB, because every batch test was in-memory).
    ///
    /// In rocksdb mode the temp dir is intentionally leaked via
    /// `keep()`: the engine holds it open for the whole test and there's
    /// no place to park the `TempDir` guard when returning only the
    /// engine. CI runners are ephemeral; local rocksdb runs leave the
    /// dirs under the system temp dir for the OS to reclaim. Tests that
    /// are specifically about one backend keep calling
    /// `new_in_memory` / `new_rocksdb` directly.
    fn test_engine(schema: Schema) -> StorageEngine {
        match std::env::var("DYNOGRAPH_TEST_BACKEND").as_deref() {
            Ok("rocksdb") => {
                let dir = tempfile::tempdir()
                    .expect("create temp dir for rocksdb test")
                    .keep();
                let path = dir.to_str().expect("temp path is valid utf-8");
                StorageEngine::new_rocksdb(schema, path).expect("open rocksdb test engine")
            }
            _ => StorageEngine::new_in_memory(schema),
        }
    }

    // Backend-agnostic engine tests (run against both backends via the
    // matrix — see `test_engine`).

    #[test]
    fn create_and_get_node() {
        let mut engine = test_engine(test_schema());
        let props = props! { "name" => "Alice", "role" => "protagonist" };
        let node = engine.create_node("g1", "Character", "c1", props).unwrap();
        assert_eq!(node.node_id, "c1");
        let fetched = engine.get_node("g1", "Character", "c1").unwrap().unwrap();
        assert_eq!(fetched.properties["name"].as_str().unwrap(), "Alice");
    }

    #[test]
    fn create_node_rejects_nul_in_node_id() {
        let mut engine = test_engine(test_schema());
        let err = engine
            .create_node("g1", "Character", "c\x001", props! { "name" => "Alice" })
            .unwrap_err();
        assert!(
            matches!(err, DynoError::InvalidKeySegment { ref field, .. } if field == "node_id"),
            "{err:?}"
        );
        // The reject must happen before any put — no orphaned body left.
        assert_eq!(engine.count_nodes("g1", "Character").unwrap(), 0);
    }

    #[test]
    fn create_node_rejects_nul_in_graph_id() {
        let mut engine = test_engine(test_schema());
        let err = engine
            .create_node("g\x001", "Character", "c1", props! { "name" => "Alice" })
            .unwrap_err();
        assert!(
            matches!(err, DynoError::InvalidKeySegment { ref field, .. } if field == "graph_id"),
            "{err:?}"
        );
    }

    #[test]
    fn create_node_allows_nul_in_non_indexed_property_value() {
        // `name` is not indexed, so its value lives in the msgpack body,
        // never in a key — a NUL there is harmless and must be allowed.
        let mut engine = test_engine(test_schema());
        let node = engine
            .create_node("g1", "Character", "c1", props! { "name" => "a\x00b" })
            .unwrap();
        assert_eq!(node.properties["name"].as_str().unwrap(), "a\x00b");
        let fetched = engine.get_node("g1", "Character", "c1").unwrap().unwrap();
        assert_eq!(fetched.properties["name"].as_str().unwrap(), "a\x00b");
    }

    #[test]
    fn create_node_rejects_nul_in_indexed_property_value() {
        // `story_id` IS indexed → its value becomes a key segment.
        let mut engine = test_engine(indexed_schema());
        let err = engine
            .create_node(
                "g1",
                "Fragment",
                "f1",
                props! { "name" => "A", "story_id" => "s\x00A" },
            )
            .unwrap_err();
        assert!(
            matches!(err, DynoError::InvalidKeySegment { ref field, .. } if field == "story_id"),
            "{err:?}"
        );
        assert_eq!(engine.count_nodes("g1", "Fragment").unwrap(), 0);
    }

    #[test]
    fn replace_node_properties_rejects_nul_in_indexed_value() {
        let mut engine = test_engine(indexed_schema());
        engine
            .create_node(
                "g1",
                "Fragment",
                "f1",
                props! { "name" => "A", "story_id" => "sA" },
            )
            .unwrap();
        let err = engine
            .replace_node_properties(
                "g1",
                "Fragment",
                "f1",
                props! { "name" => "A", "story_id" => "s\x00B" },
            )
            .unwrap_err();
        assert!(
            matches!(err, DynoError::InvalidKeySegment { ref field, .. } if field == "story_id"),
            "{err:?}"
        );
    }

    #[test]
    fn scan_nodes_by_property_rejects_nul_in_query_value() {
        // A NUL query value can't match any stored key, so without the
        // guard the scan would silently return empty rather than fail.
        let engine = test_engine(indexed_schema());
        let err = engine
            .scan_nodes_by_property(
                "g1",
                "Fragment",
                "story_id",
                &Value::String("s\x00A".into()),
            )
            .unwrap_err();
        assert!(
            matches!(err, DynoError::InvalidKeySegment { ref field, .. } if field == "story_id"),
            "{err:?}"
        );
    }

    #[test]
    fn create_edge_rejects_nul_in_endpoint_id() {
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "alice", props! { "name" => "A" })
            .unwrap();
        let err = engine
            .create_edge(
                "g1",
                "KNOWS",
                "Character",
                "alice",
                "Character",
                "b\x00ob",
                HashMap::new(),
            )
            .unwrap_err();
        assert!(
            matches!(err, DynoError::InvalidKeySegment { ref field, .. } if field == "to_id"),
            "{err:?}"
        );
    }

    #[test]
    fn create_node_validates_schema() {
        let mut engine = test_engine(test_schema());
        let result = engine.create_node("g1", "Character", "c1", HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn create_node_validates_enum() {
        let mut engine = test_engine(test_schema());
        let props = props! { "name" => "Bob", "role" => "villain" };
        assert!(engine.create_node("g1", "Character", "c1", props).is_err());
    }

    #[test]
    fn get_nonexistent_node_returns_none() {
        let engine = test_engine(test_schema());
        assert!(
            engine
                .get_node("g1", "Character", "missing")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn delete_node() {
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
            .unwrap();
        assert!(engine.delete_node("g1", "Character", "c1").unwrap());
        assert!(engine.get_node("g1", "Character", "c1").unwrap().is_none());
        assert!(!engine.delete_node("g1", "Character", "c1").unwrap());
    }

    #[test]
    fn delete_node_removes_outgoing_edges_and_peer_inverse_adjacency() {
        // Tech-debt C1 regression: delete_node used to leave dangling
        // CF_EDGES entries and inverse adjacency on neighbor nodes.
        // After deleting alice, bob's incoming-edge scan should be empty
        // and the alice→bob edge should be unresolvable.
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "alice", props! { "name" => "Alice" })
            .unwrap();
        engine
            .create_node("g1", "Character", "bob", props! { "name" => "Bob" })
            .unwrap();
        engine
            .create_edge(
                "g1",
                "KNOWS",
                "Character",
                "alice",
                "Character",
                "bob",
                HashMap::new(),
            )
            .unwrap();

        // Sanity: edge exists pre-delete.
        assert!(
            engine
                .get_edge("g1", "KNOWS", "alice", "bob")
                .unwrap()
                .is_some()
        );
        assert_eq!(
            engine.scan_incoming_edges("g1", "bob", None).unwrap().len(),
            1
        );

        engine.delete_node("g1", "Character", "alice").unwrap();

        // Edge no longer resolves from CF_EDGES.
        assert!(
            engine
                .get_edge("g1", "KNOWS", "alice", "bob")
                .unwrap()
                .is_none(),
            "edge should be gone after endpoint delete"
        );
        // bob's incoming-edge scan should not return the alice→bob entry.
        assert_eq!(
            engine.scan_incoming_edges("g1", "bob", None).unwrap().len(),
            0,
            "peer inverse adjacency should be cleaned up"
        );
    }

    #[test]
    fn delete_node_removes_incoming_edges_and_peer_outgoing_adjacency() {
        // Symmetric case: delete the destination of an edge; the source's
        // outgoing-edge scan should no longer include the deleted endpoint.
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "alice", props! { "name" => "Alice" })
            .unwrap();
        engine
            .create_node("g1", "Character", "bob", props! { "name" => "Bob" })
            .unwrap();
        engine
            .create_edge(
                "g1",
                "KNOWS",
                "Character",
                "alice",
                "Character",
                "bob",
                HashMap::new(),
            )
            .unwrap();

        engine.delete_node("g1", "Character", "bob").unwrap();

        assert!(
            engine
                .get_edge("g1", "KNOWS", "alice", "bob")
                .unwrap()
                .is_none(),
        );
        assert_eq!(
            engine
                .scan_outgoing_edges("g1", "alice", None)
                .unwrap()
                .len(),
            0,
            "alice's outgoing-edge scan must not reference the deleted bob"
        );
    }

    #[test]
    fn delete_node_with_mixed_incoming_and_outgoing_edges() {
        // alice has both an outgoing edge (alice → bob) and an incoming
        // edge (carol → alice via VISITS — Character VISITS Location;
        // we'll use a Location-typed `loc1` with a back-link via KNOWS
        // — but the test_schema only allows VISITS Character→Location.
        // So: alice → loc1 (VISITS), bob → alice (KNOWS).
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "alice", props! { "name" => "A" })
            .unwrap();
        engine
            .create_node("g1", "Character", "bob", props! { "name" => "B" })
            .unwrap();
        engine
            .create_node("g1", "Location", "loc1", props! { "name" => "Tavern" })
            .unwrap();
        engine
            .create_edge(
                "g1",
                "VISITS",
                "Character",
                "alice",
                "Location",
                "loc1",
                HashMap::new(),
            )
            .unwrap();
        engine
            .create_edge(
                "g1",
                "KNOWS",
                "Character",
                "bob",
                "Character",
                "alice",
                HashMap::new(),
            )
            .unwrap();

        engine.delete_node("g1", "Character", "alice").unwrap();

        // Both edges gone from CF_EDGES.
        assert!(
            engine
                .get_edge("g1", "VISITS", "alice", "loc1")
                .unwrap()
                .is_none()
        );
        assert!(
            engine
                .get_edge("g1", "KNOWS", "bob", "alice")
                .unwrap()
                .is_none()
        );
        // Loc1 has no incoming visits anymore.
        assert_eq!(
            engine
                .scan_incoming_edges("g1", "loc1", None)
                .unwrap()
                .len(),
            0
        );
        // Bob has no outgoing knows anymore.
        assert_eq!(
            engine.scan_outgoing_edges("g1", "bob", None).unwrap().len(),
            0
        );
    }

    #[test]
    fn create_and_get_edge() {
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
            .unwrap();
        engine
            .create_node("g1", "Character", "c2", props! { "name" => "Bob" })
            .unwrap();
        let edge = engine
            .create_edge(
                "g1",
                "KNOWS",
                "Character",
                "c1",
                "Character",
                "c2",
                HashMap::new(),
            )
            .unwrap();
        assert_eq!(edge.edge_type, "KNOWS");
        let fetched = engine.get_edge("g1", "KNOWS", "c1", "c2").unwrap().unwrap();
        assert_eq!(fetched.from_id, "c1");
    }

    #[test]
    fn edge_validates_types() {
        let mut engine = test_engine(test_schema());
        assert!(
            engine
                .create_edge(
                    "g1",
                    "KNOWS",
                    "Location",
                    "l1",
                    "Character",
                    "c1",
                    HashMap::new()
                )
                .is_err()
        );
    }

    #[test]
    fn cross_type_edge() {
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
            .unwrap();
        engine
            .create_node("g1", "Location", "loc1", props! { "name" => "Tavern" })
            .unwrap();
        let edge = engine
            .create_edge(
                "g1",
                "VISITS",
                "Character",
                "c1",
                "Location",
                "loc1",
                HashMap::new(),
            )
            .unwrap();
        assert_eq!(edge.edge_type, "VISITS");
    }

    #[test]
    fn count_nodes_by_type() {
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
            .unwrap();
        engine
            .create_node("g1", "Character", "c2", props! { "name" => "Bob" })
            .unwrap();
        engine
            .create_node("g1", "Location", "loc1", props! { "name" => "Tavern" })
            .unwrap();
        assert_eq!(engine.count_nodes("g1", "Character").unwrap(), 2);
        assert_eq!(engine.count_nodes("g1", "Location").unwrap(), 1);
    }

    #[test]
    fn graph_isolation() {
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
            .unwrap();
        engine
            .create_node("g2", "Character", "c1", props! { "name" => "Bob" })
            .unwrap();
        assert_eq!(
            engine
                .get_node("g1", "Character", "c1")
                .unwrap()
                .unwrap()
                .properties["name"]
                .as_str()
                .unwrap(),
            "Alice"
        );
        assert_eq!(
            engine
                .get_node("g2", "Character", "c1")
                .unwrap()
                .unwrap()
                .properties["name"]
                .as_str()
                .unwrap(),
            "Bob"
        );
    }

    // Reverse-index tests (CF_NODE_IDX)

    fn indexed_schema() -> Schema {
        Schema::from_yaml(
            r#"
schema:
  name: test_indexed
  version: 1
  node_types:
    Fragment:
      properties:
        name: { type: string, required: true }
        story_id: { type: string, required: true, indexed: true }
    Character:
      properties:
        name: { type: string, required: true }
        story_id: { type: string, indexed: true }
  edge_types: {}
"#,
        )
        .unwrap()
    }

    #[test]
    fn create_populates_index_and_scan_filters_by_value() {
        let mut engine = test_engine(indexed_schema());
        engine
            .create_node(
                "g1",
                "Fragment",
                "f1",
                props! { "name" => "A", "story_id" => "sA" },
            )
            .unwrap();
        engine
            .create_node(
                "g1",
                "Fragment",
                "f2",
                props! { "name" => "B", "story_id" => "sA" },
            )
            .unwrap();
        engine
            .create_node(
                "g1",
                "Fragment",
                "f3",
                props! { "name" => "C", "story_id" => "sB" },
            )
            .unwrap();

        let sid_a = Value::from("sA");
        let got_a = engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &sid_a)
            .unwrap();
        let mut ids: Vec<_> = got_a.iter().map(|n| n.node_id.clone()).collect();
        ids.sort();
        assert_eq!(ids, vec!["f1", "f2"]);

        let sid_b = Value::from("sB");
        let got_b = engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &sid_b)
            .unwrap();
        assert_eq!(got_b.len(), 1);
        assert_eq!(got_b[0].node_id, "f3");
    }

    #[test]
    fn scan_filters_by_node_type() {
        // Same story_id across Fragment and Character — scan must not bleed types.
        let mut engine = test_engine(indexed_schema());
        engine
            .create_node(
                "g1",
                "Fragment",
                "f1",
                props! { "name" => "F", "story_id" => "sA" },
            )
            .unwrap();
        engine
            .create_node(
                "g1",
                "Character",
                "c1",
                props! { "name" => "C", "story_id" => "sA" },
            )
            .unwrap();

        let sid = Value::from("sA");
        let frags = engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &sid)
            .unwrap();
        assert_eq!(frags.len(), 1);
        assert_eq!(frags[0].node_id, "f1");
        assert_eq!(frags[0].node_type, "Fragment");

        let chars = engine
            .scan_nodes_by_property("g1", "Character", "story_id", &sid)
            .unwrap();
        assert_eq!(chars.len(), 1);
        assert_eq!(chars[0].node_id, "c1");
        assert_eq!(chars[0].node_type, "Character");
    }

    #[test]
    fn update_moves_index_entry() {
        let mut engine = test_engine(indexed_schema());
        engine
            .create_node(
                "g1",
                "Fragment",
                "f1",
                props! { "name" => "A", "story_id" => "sA" },
            )
            .unwrap();

        // Reparent f1 from sA to sB.
        engine
            .replace_node_properties(
                "g1",
                "Fragment",
                "f1",
                props! { "name" => "A", "story_id" => "sB" },
            )
            .unwrap();

        let sid_a = Value::from("sA");
        let sid_b = Value::from("sB");
        assert_eq!(
            engine
                .scan_nodes_by_property("g1", "Fragment", "story_id", &sid_a)
                .unwrap()
                .len(),
            0,
            "old story_id should no longer match"
        );
        let hits_b = engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &sid_b)
            .unwrap();
        assert_eq!(hits_b.len(), 1);
        assert_eq!(hits_b[0].node_id, "f1");
    }

    #[test]
    fn create_node_overwrite_reconciles_index() {
        // create-or-replace: re-creating an existing id with a different
        // indexed value must drop the old index entry, not leave it
        // dangling (which would make scan_nodes_by_property return the
        // node under its stale value).
        let mut engine = test_engine(indexed_schema());
        engine
            .create_node(
                "g1",
                "Fragment",
                "f1",
                props! { "name" => "A", "story_id" => "sA" },
            )
            .unwrap();
        // Overwrite the SAME id with a new story_id (bare create, not
        // replace_node_properties).
        engine
            .create_node(
                "g1",
                "Fragment",
                "f1",
                props! { "name" => "A", "story_id" => "sB" },
            )
            .unwrap();

        assert_eq!(
            engine
                .scan_nodes_by_property("g1", "Fragment", "story_id", &Value::from("sA"))
                .unwrap()
                .len(),
            0,
            "stale index entry under the old value must be gone"
        );
        let hits_b = engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &Value::from("sB"))
            .unwrap();
        assert_eq!(hits_b.len(), 1);
        assert_eq!(hits_b[0].node_id, "f1");
    }

    #[test]
    fn delete_cleans_up_index_entries() {
        let mut engine = test_engine(indexed_schema());
        engine
            .create_node(
                "g1",
                "Fragment",
                "f1",
                props! { "name" => "A", "story_id" => "sA" },
            )
            .unwrap();

        assert!(engine.delete_node("g1", "Fragment", "f1").unwrap());

        let sid = Value::from("sA");
        let hits = engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &sid)
            .unwrap();
        assert_eq!(hits.len(), 0);
    }

    #[test]
    fn non_indexed_property_returns_empty() {
        // `name` isn't declared indexed, so no CF_NODE_IDX entries are written
        // and scans against it see nothing.
        let mut engine = test_engine(indexed_schema());
        engine
            .create_node(
                "g1",
                "Fragment",
                "f1",
                props! { "name" => "Alice", "story_id" => "sA" },
            )
            .unwrap();

        let name = Value::from("Alice");
        let hits = engine
            .scan_nodes_by_property("g1", "Fragment", "name", &name)
            .unwrap();
        assert_eq!(hits.len(), 0);
    }

    #[test]
    fn unsupported_value_types_return_empty() {
        let engine = test_engine(indexed_schema());
        assert_eq!(
            engine
                .scan_nodes_by_property("g1", "Fragment", "story_id", &Value::Float(1.0))
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            engine
                .scan_nodes_by_property("g1", "Fragment", "story_id", &Value::Null)
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            engine
                .scan_nodes_by_property("g1", "Fragment", "story_id", &Value::List(vec![]))
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn index_survives_through_rocksdb_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap().to_string();

        {
            let mut engine = StorageEngine::new_rocksdb(indexed_schema(), &path).unwrap();
            engine
                .create_node(
                    "g1",
                    "Fragment",
                    "f1",
                    props! { "name" => "A", "story_id" => "sA" },
                )
                .unwrap();
            engine
                .create_node(
                    "g1",
                    "Fragment",
                    "f2",
                    props! { "name" => "B", "story_id" => "sB" },
                )
                .unwrap();
            engine
                .replace_node_properties(
                    "g1",
                    "Fragment",
                    "f2",
                    props! { "name" => "B", "story_id" => "sA" },
                )
                .unwrap();
            engine
                .create_node(
                    "g1",
                    "Fragment",
                    "f3",
                    props! { "name" => "C", "story_id" => "sA" },
                )
                .unwrap();
            engine.delete_node("g1", "Fragment", "f3").unwrap();
        }

        // Reopen. f1 and f2 should both be under sA; f3 should be gone.
        let engine = StorageEngine::new_rocksdb(indexed_schema(), &path).unwrap();
        let sid_a = Value::from("sA");
        let hits = engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &sid_a)
            .unwrap();
        let mut ids: Vec<_> = hits.iter().map(|n| n.node_id.clone()).collect();
        ids.sort();
        assert_eq!(ids, vec!["f1", "f2"]);

        let sid_b = Value::from("sB");
        let empty = engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &sid_b)
            .unwrap();
        assert_eq!(empty.len(), 0, "update should have removed sB entry for f2");
    }

    // RocksDB tests

    #[test]
    fn rocksdb_create_and_get_node() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine =
            StorageEngine::new_rocksdb(test_schema(), dir.path().to_str().unwrap()).unwrap();

        engine
            .create_node(
                "g1",
                "Character",
                "c1",
                props! { "name" => "Alice", "role" => "protagonist" },
            )
            .unwrap();
        let node = engine.get_node("g1", "Character", "c1").unwrap().unwrap();
        assert_eq!(node.properties["name"].as_str().unwrap(), "Alice");
    }

    #[test]
    fn rocksdb_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap().to_string();

        // Write
        {
            let mut engine = StorageEngine::new_rocksdb(test_schema(), &path).unwrap();
            engine
                .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
                .unwrap();
            engine
                .create_node("g1", "Character", "c2", props! { "name" => "Bob" })
                .unwrap();
            engine
                .create_edge(
                    "g1",
                    "KNOWS",
                    "Character",
                    "c1",
                    "Character",
                    "c2",
                    HashMap::new(),
                )
                .unwrap();
            // engine drops here, RocksDB flushes
        }

        // Re-open and verify data survived
        {
            let engine = StorageEngine::new_rocksdb(test_schema(), &path).unwrap();
            let alice = engine.get_node("g1", "Character", "c1").unwrap().unwrap();
            assert_eq!(alice.properties["name"].as_str().unwrap(), "Alice");
            let bob = engine.get_node("g1", "Character", "c2").unwrap().unwrap();
            assert_eq!(bob.properties["name"].as_str().unwrap(), "Bob");
            let edge = engine.get_edge("g1", "KNOWS", "c1", "c2").unwrap().unwrap();
            assert_eq!(edge.from_id, "c1");
            assert_eq!(engine.count_nodes("g1", "Character").unwrap(), 2);
        }
    }

    #[test]
    fn rocksdb_scan_and_count() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine =
            StorageEngine::new_rocksdb(test_schema(), dir.path().to_str().unwrap()).unwrap();

        engine
            .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
            .unwrap();
        engine
            .create_node("g1", "Character", "c2", props! { "name" => "Bob" })
            .unwrap();
        engine
            .create_node("g1", "Location", "loc1", props! { "name" => "Tavern" })
            .unwrap();

        assert_eq!(engine.count_nodes("g1", "Character").unwrap(), 2);
        assert_eq!(engine.count_nodes("g1", "Location").unwrap(), 1);

        let chars = engine.scan_nodes("g1", "Character").unwrap();
        assert_eq!(chars.len(), 2);
    }

    #[test]
    fn rocksdb_delete_node() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine =
            StorageEngine::new_rocksdb(test_schema(), dir.path().to_str().unwrap()).unwrap();

        engine
            .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
            .unwrap();
        assert!(engine.delete_node("g1", "Character", "c1").unwrap());
        assert!(engine.get_node("g1", "Character", "c1").unwrap().is_none());
    }

    #[test]
    fn rocksdb_outgoing_edges() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine =
            StorageEngine::new_rocksdb(test_schema(), dir.path().to_str().unwrap()).unwrap();

        engine
            .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
            .unwrap();
        engine
            .create_node("g1", "Character", "c2", props! { "name" => "Bob" })
            .unwrap();
        engine
            .create_node("g1", "Location", "loc1", props! { "name" => "Tavern" })
            .unwrap();
        engine
            .create_edge(
                "g1",
                "KNOWS",
                "Character",
                "c1",
                "Character",
                "c2",
                HashMap::new(),
            )
            .unwrap();
        engine
            .create_edge(
                "g1",
                "VISITS",
                "Character",
                "c1",
                "Location",
                "loc1",
                HashMap::new(),
            )
            .unwrap();

        let all = engine.scan_outgoing_edges("g1", "c1", None).unwrap();
        assert_eq!(all.len(), 2);

        let knows = engine
            .scan_outgoing_edges("g1", "c1", Some("KNOWS"))
            .unwrap();
        assert_eq!(knows.len(), 1);
        assert_eq!(knows[0].to_id, "c2");
    }

    // -- Tech-debt C4 regression tests: batch atomicity for mixed put + delete

    #[test]
    fn batch_buffers_mixed_put_and_delete_visible_within_batch() {
        let mut engine = test_engine(test_schema());
        // Pre-existing node we'll delete inside the batch.
        engine
            .create_node("g1", "Character", "to_delete", props! { "name" => "Goner" })
            .unwrap();

        engine.begin_batch();
        engine
            .create_node("g1", "Character", "to_create", props! { "name" => "New" })
            .unwrap();
        engine.delete_node("g1", "Character", "to_delete").unwrap();

        // Within-batch reads see buffered ops — read-your-own-writes.
        assert!(
            engine
                .get_node("g1", "Character", "to_create")
                .unwrap()
                .is_some(),
            "buffered create should be visible within batch (read-your-own-writes)"
        );
        assert!(
            engine
                .get_node("g1", "Character", "to_delete")
                .unwrap()
                .is_none(),
            "buffered delete should be visible within batch (read-your-own-writes)"
        );

        engine.commit_batch().unwrap();

        // Post-commit: both ops applied atomically — same view as within-batch.
        assert!(
            engine
                .get_node("g1", "Character", "to_create")
                .unwrap()
                .is_some()
        );
        assert!(
            engine
                .get_node("g1", "Character", "to_delete")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn batch_discard_with_mixed_ops_leaves_pre_batch_state() {
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "keep_me", props! { "name" => "Stay" })
            .unwrap();

        engine.begin_batch();
        engine
            .create_node("g1", "Character", "ghost", props! { "name" => "Phantom" })
            .unwrap();
        engine.delete_node("g1", "Character", "keep_me").unwrap();
        engine.discard_batch();

        // Discarded ops must not affect disk.
        assert!(
            engine
                .get_node("g1", "Character", "ghost")
                .unwrap()
                .is_none()
        );
        assert!(
            engine
                .get_node("g1", "Character", "keep_me")
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn batched_replace_on_indexed_node_visible_to_index_scans() {
        // Replace on an indexed node applies an index delete + put
        // atomically. v0.5.5+ buffer-aware scans surface the swapped
        // index state mid-batch; the within-batch view matches the
        // post-commit view (with the only difference being whether
        // the change has hit disk).
        let mut engine = test_engine(indexed_schema());
        engine
            .create_node(
                "g1",
                "Fragment",
                "f1",
                props! { "name" => "A", "story_id" => "sA" },
            )
            .unwrap();

        engine.begin_batch();
        engine
            .replace_node_properties(
                "g1",
                "Fragment",
                "f1",
                props! { "name" => "A", "story_id" => "sB" },
            )
            .unwrap();

        // Within-batch: index reflects the swap — f1 lives under sB,
        // sA index entry is gone (read-your-own-writes).
        let sa = Value::from("sA");
        let sb = Value::from("sB");
        assert_eq!(
            engine
                .scan_nodes_by_property("g1", "Fragment", "story_id", &sa)
                .unwrap()
                .len(),
            0,
            "old story_id index entry should be tombstoned within batch"
        );
        let hits_b_in_batch = engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &sb)
            .unwrap();
        assert_eq!(
            hits_b_in_batch.len(),
            1,
            "new story_id index entry should be visible within batch"
        );
        assert_eq!(hits_b_in_batch[0].node_id, "f1");

        engine.commit_batch().unwrap();

        // Post-commit: same view as within-batch.
        assert_eq!(
            engine
                .scan_nodes_by_property("g1", "Fragment", "story_id", &sa)
                .unwrap()
                .len(),
            0
        );
        let hits_b = engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &sb)
            .unwrap();
        assert_eq!(hits_b.len(), 1);
        assert_eq!(hits_b[0].node_id, "f1");
    }

    #[test]
    fn batch_prefix_delete_visible_within_batch() {
        // delete_node uses prefix_delete on adj_out/adj_in. Pre-v0.5.5
        // these tombstones were invisible to scans until commit; v0.5.5+
        // overlays the buffer on scans, so the deletion is observable
        // mid-batch.
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "alice", props! { "name" => "A" })
            .unwrap();
        engine
            .create_node("g1", "Character", "bob", props! { "name" => "B" })
            .unwrap();
        engine
            .create_edge(
                "g1",
                "KNOWS",
                "Character",
                "alice",
                "Character",
                "bob",
                HashMap::new(),
            )
            .unwrap();

        engine.begin_batch();
        engine.delete_node("g1", "Character", "alice").unwrap();

        // Within-batch reads see the buffered prefix-delete: alice is
        // gone, her outgoing adjacency is empty, bob's incoming is empty.
        assert!(
            engine
                .get_node("g1", "Character", "alice")
                .unwrap()
                .is_none(),
            "buffered delete_node should be visible within batch"
        );
        assert_eq!(
            engine
                .scan_outgoing_edges("g1", "alice", None)
                .unwrap()
                .len(),
            0,
            "buffered prefix_delete on adj_out should be visible within batch"
        );
        assert_eq!(
            engine.scan_incoming_edges("g1", "bob", None).unwrap().len(),
            0,
            "buffered prefix_delete on adj_in should be visible within batch"
        );

        engine.commit_batch().unwrap();

        // Post-commit: same view as within-batch.
        assert!(
            engine
                .get_node("g1", "Character", "alice")
                .unwrap()
                .is_none()
        );
        assert!(
            engine
                .get_edge("g1", "KNOWS", "alice", "bob")
                .unwrap()
                .is_none()
        );
        assert_eq!(
            engine.scan_incoming_edges("g1", "bob", None).unwrap().len(),
            0
        );
    }

    /// New v0.5.5 positive tests for buffer-aware reads. Each exercises
    /// one of the read paths against an active batch: get + scan + the
    /// PrefixDelete-then-Put resurrection corner.

    #[test]
    fn buffer_aware_get_sees_in_batch_create() {
        let mut engine = test_engine(test_schema());
        engine.begin_batch();
        engine
            .create_node("g1", "Character", "alice", props! { "name" => "Alice" })
            .unwrap();
        let n = engine
            .get_node("g1", "Character", "alice")
            .unwrap()
            .expect("buffered create must be readable mid-batch");
        assert_eq!(
            n.properties.get("name").and_then(|v| v.as_str()),
            Some("Alice")
        );
    }

    #[test]
    fn buffer_aware_get_sees_in_batch_delete_of_existing_node() {
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "alice", props! { "name" => "Alice" })
            .unwrap();
        engine.begin_batch();
        engine.delete_node("g1", "Character", "alice").unwrap();
        assert!(
            engine
                .get_node("g1", "Character", "alice")
                .unwrap()
                .is_none(),
            "buffered delete must shadow the backend node"
        );
    }

    #[test]
    fn buffer_aware_scan_overlays_in_batch_creates_and_deletes() {
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "pre", props! { "name" => "Pre" })
            .unwrap();
        engine.begin_batch();
        engine
            .create_node("g1", "Character", "new", props! { "name" => "New" })
            .unwrap();
        engine.delete_node("g1", "Character", "pre").unwrap();

        let mut nodes = engine.scan_nodes("g1", "Character").unwrap();
        nodes.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        assert_eq!(nodes.len(), 1, "expected only the in-batch new node");
        assert_eq!(nodes[0].node_id, "new");
    }

    #[test]
    fn buffer_aware_get_late_put_resurrects_after_prefix_delete() {
        // [delete_node X (prefix-delete on adj), Put X again] — the
        // late Put wins. Ordering is preserved end-to-end at commit;
        // mid-batch reads must see the same final state.
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "phoenix", props! { "name" => "v1" })
            .unwrap();
        engine.begin_batch();
        engine.delete_node("g1", "Character", "phoenix").unwrap();
        engine
            .create_node("g1", "Character", "phoenix", props! { "name" => "v2" })
            .unwrap();
        let n = engine
            .get_node("g1", "Character", "phoenix")
            .unwrap()
            .expect("late put after delete must resurrect the key");
        assert_eq!(
            n.properties.get("name").and_then(|v| v.as_str()),
            Some("v2"),
            "late put's value must win over earlier delete"
        );
    }

    #[test]
    fn buffer_aware_discard_keeps_backend_unchanged() {
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "keep", props! { "name" => "Keep" })
            .unwrap();
        engine.begin_batch();
        engine
            .create_node("g1", "Character", "ghost", props! { "name" => "Ghost" })
            .unwrap();
        engine.delete_node("g1", "Character", "keep").unwrap();
        // Within-batch view: ghost present, keep gone.
        assert!(
            engine
                .get_node("g1", "Character", "ghost")
                .unwrap()
                .is_some()
        );
        assert!(
            engine
                .get_node("g1", "Character", "keep")
                .unwrap()
                .is_none()
        );
        engine.discard_batch();
        // Post-discard view: backend unchanged. Ghost never existed,
        // keep is still there.
        assert!(
            engine
                .get_node("g1", "Character", "ghost")
                .unwrap()
                .is_none()
        );
        assert!(
            engine
                .get_node("g1", "Character", "keep")
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn replace_schema_swaps_validation_rules() {
        // Old schema: Person has `name: string, required`. New schema:
        // adds `age: int, required` with a default. Verify (a) the
        // schema accessor returns the new shape and (b) writes
        // post-swap apply the new validation rules (default applied
        // for `age`).
        let old = Schema::from_yaml(
            r#"
schema:
  name: t
  version: 1
  node_types:
    Person:
      properties:
        name: { type: string, required: true }
  edge_types: {}
"#,
        )
        .unwrap();
        let new = Schema::from_yaml(
            r#"
schema:
  name: t
  version: 2
  node_types:
    Person:
      properties:
        name: { type: string, required: true }
        age:  { type: int,    required: true, default: 0 }
  edge_types: {}
"#,
        )
        .unwrap();

        let mut engine = test_engine(old);
        engine
            .create_node("g1", "Person", "alice", HashMap::new())
            .unwrap_err(); // missing required `name` — sanity check old schema in force

        engine.replace_schema(new);
        assert_eq!(engine.schema().version, 2);

        // New schema's `age` default applies on create — `name` still
        // required.
        let mut props = HashMap::new();
        props.insert("name".to_string(), Value::String("alice".into()));
        engine.create_node("g1", "Person", "alice", props).unwrap();
        let stored = engine.get_node("g1", "Person", "alice").unwrap().unwrap();
        assert_eq!(stored.properties.get("age"), Some(&Value::Int(0)));
    }

    fn embedding_schema() -> Schema {
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

    fn make_item(engine: &mut StorageEngine, id: &str) {
        let mut props = HashMap::new();
        props.insert("name".to_string(), Value::String(id.into()));
        engine.create_node("g1", "Item", id, props).unwrap();
    }

    #[test]
    fn embedding_round_trip() {
        let mut engine = test_engine(embedding_schema());
        make_item(&mut engine, "n1");
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        engine.set_embedding("g1", "Item", "n1", &v).unwrap();
        let got = engine
            .get_embedding("g1", "Item", "n1")
            .unwrap()
            .expect("embedding present");
        assert_eq!(got, v);
    }

    #[test]
    fn embedding_overwrites_in_place() {
        let mut engine = test_engine(embedding_schema());
        make_item(&mut engine, "n1");
        engine
            .set_embedding("g1", "Item", "n1", &[1.0, 2.0])
            .unwrap();
        engine
            .set_embedding("g1", "Item", "n1", &[9.0, 8.0, 7.0])
            .unwrap();
        let got = engine.get_embedding("g1", "Item", "n1").unwrap().unwrap();
        assert_eq!(got, vec![9.0, 8.0, 7.0]);
    }

    #[test]
    fn set_embedding_on_missing_node_errors_loudly() {
        let mut engine = test_engine(embedding_schema());
        let err = engine
            .set_embedding("g1", "Item", "ghost", &[1.0, 2.0])
            .unwrap_err();
        assert!(matches!(
            err,
            DynoError::NodeNotFound { ref node_type, ref node_id }
                if node_type == "Item" && node_id == "ghost"
        ));
    }

    #[test]
    fn set_embedding_rejects_empty_vector() {
        let mut engine = test_engine(embedding_schema());
        make_item(&mut engine, "n1");
        let err = engine.set_embedding("g1", "Item", "n1", &[]).unwrap_err();
        assert!(matches!(err, DynoError::Validation { .. }));
    }

    #[test]
    fn set_embedding_rejects_non_finite() {
        let mut engine = test_engine(embedding_schema());
        make_item(&mut engine, "n1");
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = engine
                .set_embedding("g1", "Item", "n1", &[1.0, bad, 3.0])
                .unwrap_err();
            assert!(
                matches!(err, DynoError::Validation { ref message, .. } if message.contains("non-finite")),
                "{bad} should be rejected loudly, got {err:?}"
            );
        }
        // Nothing was persisted — the reject ran before the write.
        assert!(engine.get_embedding("g1", "Item", "n1").unwrap().is_none());
    }

    #[test]
    fn decode_embedding_rejects_non_finite() {
        // NaN bytes (exponent all ones, non-zero mantissa) on the read
        // path must fail loud rather than yield a poison float.
        let nan_bytes = f32::NAN.to_le_bytes();
        let err = decode_embedding(&nan_bytes).unwrap_err();
        assert!(
            matches!(err, DynoError::Storage(ref m) if m.contains("non-finite")),
            "{err:?}"
        );
    }

    #[test]
    fn delete_embedding_returns_existed_bool() {
        let mut engine = test_engine(embedding_schema());
        make_item(&mut engine, "n1");
        engine.set_embedding("g1", "Item", "n1", &[1.0]).unwrap();
        assert!(engine.delete_embedding("g1", "Item", "n1").unwrap());
        // Idempotent — second delete returns false but is not an error.
        assert!(!engine.delete_embedding("g1", "Item", "n1").unwrap());
        // And the embedding is gone for real.
        assert!(engine.get_embedding("g1", "Item", "n1").unwrap().is_none());
    }

    #[test]
    fn delete_node_cascades_to_embedding() {
        let mut engine = test_engine(embedding_schema());
        make_item(&mut engine, "n1");
        engine
            .set_embedding("g1", "Item", "n1", &[0.5, 0.5])
            .unwrap();
        engine.delete_node("g1", "Item", "n1").unwrap();
        assert!(engine.get_embedding("g1", "Item", "n1").unwrap().is_none());
    }

    #[test]
    fn scan_embeddings_by_type_returns_all_and_only_that_type() {
        let mut engine = test_engine(embedding_schema());
        for id in ["a", "b", "c"] {
            make_item(&mut engine, id);
        }
        engine
            .set_embedding("g1", "Item", "a", &[1.0, 0.0])
            .unwrap();
        engine
            .set_embedding("g1", "Item", "b", &[0.0, 1.0])
            .unwrap();
        // c has no embedding.
        let mut scanned = engine.scan_embeddings_by_type("g1", "Item").unwrap();
        scanned.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(scanned.len(), 2);
        assert_eq!(scanned[0].0, "a");
        assert_eq!(scanned[0].1, vec![1.0, 0.0]);
        assert_eq!(scanned[1].0, "b");
        assert_eq!(scanned[1].1, vec![0.0, 1.0]);
    }

    #[test]
    fn rocksdb_embedding_persistence() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        let v = vec![0.25_f32, 0.5, 0.75, 1.0];
        {
            let mut engine = StorageEngine::new_rocksdb(embedding_schema(), path).unwrap();
            make_item(&mut engine, "n1");
            engine.set_embedding("g1", "Item", "n1", &v).unwrap();
        }
        let engine = StorageEngine::new_rocksdb(embedding_schema(), path).unwrap();
        let got = engine.get_embedding("g1", "Item", "n1").unwrap().unwrap();
        assert_eq!(got, v);
    }

    /// Regression for the C1 panic: a batched `delete_node` buffers a
    /// `Delete` on `CF_EMBEDDINGS` (the embedding cascade), and on the
    /// RocksDB backend `commit_batch` used to index a 5-element handle
    /// array with `CfId::Embeddings as usize == 5`, panicking out of
    /// bounds. Reachable in production via the `/batch` endpoint. The
    /// in-memory backend keys CFs by string so it never hit this; the
    /// bug only surfaces on RocksDB, hence the explicit backend here.
    #[test]
    fn rocksdb_batched_delete_node_with_embedding_commits() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut engine =
            StorageEngine::new_rocksdb(embedding_schema(), tmp.path().to_str().unwrap()).unwrap();

        make_item(&mut engine, "n1");
        engine
            .set_embedding("g1", "Item", "n1", &[0.1_f32, 0.2, 0.3, 0.4])
            .unwrap();

        engine.begin_batch();
        engine.delete_node("g1", "Item", "n1").unwrap();
        let committed = engine.commit_batch().unwrap();

        assert!(committed > 0, "batch should have committed ops");
        assert!(engine.get_node("g1", "Item", "n1").unwrap().is_none());
        assert!(
            engine.get_embedding("g1", "Item", "n1").unwrap().is_none(),
            "embedding cascade should have removed the sidecar embedding"
        );
    }

    /// Companion to the above: the batched `set_embedding` *put* path
    /// also resolves the `CF_EMBEDDINGS` handle on commit, the same
    /// handle that was missing from the old array.
    #[test]
    fn rocksdb_batched_set_embedding_commits() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut engine =
            StorageEngine::new_rocksdb(embedding_schema(), tmp.path().to_str().unwrap()).unwrap();

        make_item(&mut engine, "n1");

        let v = vec![0.5_f32, 0.25, 0.125, 0.0625];
        engine.begin_batch();
        engine.set_embedding("g1", "Item", "n1", &v).unwrap();
        engine.commit_batch().unwrap();

        assert_eq!(
            engine.get_embedding("g1", "Item", "n1").unwrap().unwrap(),
            v
        );
    }

    #[test]
    fn decode_embedding_rejects_non_multiple_of_4() {
        let err = decode_embedding(&[1, 2, 3, 4, 5]).unwrap_err();
        assert!(
            matches!(err, DynoError::Storage(ref m) if m.contains("not a multiple of 4")),
            "{err:?}"
        );
    }

    /// alice→bob KNOWS in each named graph. Optionally also sets an
    /// embedding on alice. Used by clear_graph + S7 adjacency tests.
    fn seed_alice_bob(engine: &mut StorageEngine, graphs: &[&str], with_embedding: bool) {
        for graph in graphs {
            engine
                .create_node(graph, "Character", "alice", props! { "name" => "A" })
                .unwrap();
            engine
                .create_node(graph, "Character", "bob", props! { "name" => "B" })
                .unwrap();
            engine
                .create_edge(
                    graph,
                    "KNOWS",
                    "Character",
                    "alice",
                    "Character",
                    "bob",
                    HashMap::new(),
                )
                .unwrap();
            if with_embedding {
                engine
                    .set_embedding(graph, "Character", "alice", &[0.1, 0.2, 0.3])
                    .unwrap();
            }
        }
    }

    #[test]
    fn clear_graph_drops_every_cf_for_graph_only() {
        // Two graphs share the schema and overlap in node/edge ids.
        // After clear_graph("g1"), every g1 key — nodes, edges, both
        // adjacency CFs, and embeddings — must be gone, but g2's data
        // must survive untouched. Idempotent: a second clear is Ok.
        let mut engine = test_engine(test_schema());
        seed_alice_bob(&mut engine, &["g1", "g2"], true);

        engine.clear_graph("g1").unwrap();

        // g1 is gone everywhere.
        assert!(
            engine
                .get_node("g1", "Character", "alice")
                .unwrap()
                .is_none()
        );
        assert!(
            engine
                .get_edge("g1", "KNOWS", "alice", "bob")
                .unwrap()
                .is_none()
        );
        assert_eq!(
            engine
                .scan_outgoing_edges("g1", "alice", None)
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            engine.scan_incoming_edges("g1", "bob", None).unwrap().len(),
            0
        );
        assert!(
            engine
                .get_embedding("g1", "Character", "alice")
                .unwrap()
                .is_none()
        );

        // g2 is untouched.
        assert!(
            engine
                .get_node("g2", "Character", "alice")
                .unwrap()
                .is_some()
        );
        assert!(
            engine
                .get_edge("g2", "KNOWS", "alice", "bob")
                .unwrap()
                .is_some()
        );
        assert_eq!(
            engine
                .get_embedding("g2", "Character", "alice")
                .unwrap()
                .unwrap(),
            vec![0.1, 0.2, 0.3]
        );

        // Idempotent re-clear.
        engine.clear_graph("g1").unwrap();
        engine.clear_graph("never_existed").unwrap();
    }

    #[test]
    fn clear_graph_invalidates_read_cache() {
        // get_node populates the CF_NODES read cache; clear_graph
        // prefix-deletes CF_NODES, which must also invalidate the cache
        // or a subsequent read would serve the cleared node from cache.
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
            .unwrap();
        // Prime the cache.
        assert!(engine.get_node("g1", "Character", "c1").unwrap().is_some());

        engine.clear_graph("g1").unwrap();

        assert!(
            engine.get_node("g1", "Character", "c1").unwrap().is_none(),
            "cleared node must not be served from a stale read cache"
        );
    }

    #[test]
    fn clear_graph_does_not_sweep_prefix_neighbors() {
        // Regression: a naive prefix without the trailing separator
        // would let `clear_graph("g1")` also wipe "g10", "g1_test", etc.
        // The graph_prefix helper appends \x00, so the boundary is exact.
        let mut engine = test_engine(test_schema());
        engine
            .create_node("g1", "Character", "a", props! { "name" => "A" })
            .unwrap();
        engine
            .create_node("g10", "Character", "a", props! { "name" => "A10" })
            .unwrap();
        engine
            .create_node("g1_test", "Character", "a", props! { "name" => "AT" })
            .unwrap();

        engine.clear_graph("g1").unwrap();

        assert!(engine.get_node("g1", "Character", "a").unwrap().is_none());
        assert!(engine.get_node("g10", "Character", "a").unwrap().is_some());
        assert!(
            engine
                .get_node("g1_test", "Character", "a")
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn rocksdb_short_graph_id_adjacency_is_correct() {
        // S7 regression: with `fixed_prefix(48)` on adjacency CFs, a
        // short graph_id like "g1" let RocksDB group keys by a prefix
        // that crossed the graph_id/node_id/edge_type boundaries —
        // potentially returning incorrect adjacency results once the
        // CF spilled to disk. Without the extractor, plain seek + range
        // iteration is correct on any key shape. This test exercises
        // the same shape with multiple graphs sharing node_ids.
        let dir = tempfile::tempdir().unwrap();
        let mut engine =
            StorageEngine::new_rocksdb(test_schema(), dir.path().to_str().unwrap()).unwrap();
        seed_alice_bob(&mut engine, &["g1", "g2"], false);
        // Per-graph adjacency must not bleed across graphs.
        let g1_out = engine.scan_outgoing_edges("g1", "alice", None).unwrap();
        let g2_out = engine.scan_outgoing_edges("g2", "alice", None).unwrap();
        assert_eq!(g1_out.len(), 1);
        assert_eq!(g2_out.len(), 1);
        assert_eq!(g1_out[0].to_id, "bob");
        assert_eq!(g2_out[0].to_id, "bob");
    }
}

#[cfg(all(test, feature = "fulltext"))]
mod fulltext_tests {
    use super::*;
    use dynograph_core::{Schema, props};

    /// A schema with one full-text node type (`Document` with `title`/`body`
    /// fulltext) and one without (`Tag`).
    fn ft_schema() -> Schema {
        Schema::from_yaml(
            r#"
schema:
  name: ft
  version: 1
  node_types:
    Document:
      properties:
        title: { type: string, fulltext: true }
        body:  { type: string, fulltext: true }
        author: { type: string, indexed: true }
    Tag:
      properties:
        name: { type: string, indexed: true }
  edge_types: {}
"#,
        )
        .unwrap()
    }

    /// RocksDB engine over a fresh temp dir (leaked — the engine holds it open
    /// for the test, mirroring `test_engine`'s rocksdb arm).
    fn rocks_engine(schema: Schema) -> StorageEngine {
        let dir = tempfile::tempdir().expect("temp dir").keep();
        let path = dir.to_str().expect("utf-8 temp path");
        StorageEngine::new_rocksdb(schema, path).expect("open rocksdb engine")
    }

    #[test]
    fn no_index_built_when_schema_has_no_fulltext() {
        // Schema with no fulltext property → search is a clean empty, never an
        // error, and no index is constructed.
        let schema = Schema::from_yaml(
            r#"
schema:
  name: plain
  version: 1
  node_types:
    Tag:
      properties:
        name: { type: string, indexed: true }
  edge_types: {}
"#,
        )
        .unwrap();
        let engine = StorageEngine::new_in_memory(schema);
        assert!(
            engine
                .search_fulltext("g1", "anything", None, 10)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn create_indexes_and_delete_clears_in_memory() {
        let mut engine = StorageEngine::new_in_memory(ft_schema());
        engine
            .create_node(
                "g1",
                "Document",
                "n1",
                props! { "title" => "Rust Graphs", "body" => "embedded full text search" },
            )
            .unwrap();

        // Findable by a token from either fulltext field.
        let hits = engine.search_fulltext("g1", "graphs", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node_id, "n1");
        assert_eq!(hits[0].node_type, "Document");
        assert_eq!(
            engine
                .search_fulltext("g1", "search", None, 10)
                .unwrap()
                .len(),
            1
        );

        // Delete clears the document.
        engine.delete_node("g1", "Document", "n1").unwrap();
        assert!(
            engine
                .search_fulltext("g1", "graphs", None, 10)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn replace_properties_reindexes() {
        let mut engine = StorageEngine::new_in_memory(ft_schema());
        engine
            .create_node(
                "g1",
                "Document",
                "n1",
                props! { "title" => "alpha", "body" => "first" },
            )
            .unwrap();
        assert_eq!(
            engine
                .search_fulltext("g1", "alpha", None, 10)
                .unwrap()
                .len(),
            1
        );

        engine
            .replace_node_properties(
                "g1",
                "Document",
                "n1",
                props! { "title" => "beta", "body" => "second" },
            )
            .unwrap();
        // Old token gone, new token present — replace semantics held.
        assert!(
            engine
                .search_fulltext("g1", "alpha", None, 10)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            engine
                .search_fulltext("g1", "beta", None, 10)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn node_type_with_no_fulltext_is_not_searchable() {
        let mut engine = StorageEngine::new_in_memory(ft_schema());
        engine
            .create_node("g1", "Tag", "t1", props! { "name" => "important" })
            .unwrap();
        // Tag has no fulltext property → never indexed.
        assert!(
            engine
                .search_fulltext("g1", "important", None, 10)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn batch_commit_makes_writes_visible_and_discard_rolls_back() {
        let mut engine = StorageEngine::new_in_memory(ft_schema());

        // Committed batch → searchable.
        engine.begin_batch();
        engine
            .create_node(
                "g1",
                "Document",
                "n1",
                props! { "title" => "committed", "body" => "x" },
            )
            .unwrap();
        // Buffered: not yet visible.
        assert!(
            engine
                .search_fulltext("g1", "committed", None, 10)
                .unwrap()
                .is_empty()
        );
        engine.commit_batch().unwrap();
        assert_eq!(
            engine
                .search_fulltext("g1", "committed", None, 10)
                .unwrap()
                .len(),
            1
        );

        // Discarded batch → rolled back, never visible.
        engine.begin_batch();
        engine
            .create_node(
                "g1",
                "Document",
                "n2",
                props! { "title" => "transient", "body" => "y" },
            )
            .unwrap();
        engine.discard_batch();
        assert!(
            engine
                .search_fulltext("g1", "transient", None, 10)
                .unwrap()
                .is_empty()
        );
        // The earlier committed doc is untouched by the rollback.
        assert_eq!(
            engine
                .search_fulltext("g1", "committed", None, 10)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn reindex_rebuilds_and_survives_rocksdb_reopen() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().to_str().expect("utf-8 temp path").to_string();

        {
            let mut engine =
                StorageEngine::new_rocksdb(ft_schema(), &path).expect("open rocksdb engine");
            engine
                .create_node(
                    "g1",
                    "Document",
                    "n1",
                    props! { "title" => "persistent", "body" => "z" },
                )
                .unwrap();
            assert_eq!(
                engine
                    .search_fulltext("g1", "persistent", None, 10)
                    .unwrap()
                    .len(),
                1
            );
        }

        // Reopen the same dir: the on-disk Tantivy index reloads and the doc is
        // still searchable without re-indexing.
        let engine = StorageEngine::new_rocksdb(ft_schema(), &path).expect("reopen rocksdb engine");
        assert_eq!(
            engine
                .search_fulltext("g1", "persistent", None, 10)
                .unwrap()
                .len(),
            1
        );

        // reindex_fulltext is idempotent: rebuild from RocksDB, still one hit.
        let n = engine.reindex_fulltext("g1").unwrap();
        assert_eq!(n, 1);
        assert_eq!(
            engine
                .search_fulltext("g1", "persistent", None, 10)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn scoped_by_graph_and_node_type() {
        let mut engine = rocks_engine(ft_schema());
        engine
            .create_node(
                "g1",
                "Document",
                "d1",
                props! { "title" => "common", "body" => "a" },
            )
            .unwrap();
        engine
            .create_node(
                "g2",
                "Document",
                "d2",
                props! { "title" => "common", "body" => "b" },
            )
            .unwrap();
        engine
            .create_node("g1", "Tag", "t1", props! { "name" => "common" })
            .unwrap();

        // Graph scoping.
        let g1 = engine.search_fulltext("g1", "common", None, 10).unwrap();
        assert_eq!(g1.len(), 1);
        assert_eq!(g1[0].node_id, "d1");
        // node_type filter (Tag isn't indexed anyway, so only Document matches).
        let docs = engine
            .search_fulltext("g1", "common", Some("Document"), 10)
            .unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].node_id, "d1");
    }

    #[test]
    fn clear_graph_drops_fulltext_documents() {
        let mut engine = StorageEngine::new_in_memory(ft_schema());
        engine
            .create_node(
                "g1",
                "Document",
                "n1",
                props! { "title" => "scrubme", "body" => "a" },
            )
            .unwrap();
        assert_eq!(
            engine
                .search_fulltext("g1", "scrubme", None, 10)
                .unwrap()
                .len(),
            1
        );
        engine.clear_graph("g1").unwrap();
        assert!(
            engine
                .search_fulltext("g1", "scrubme", None, 10)
                .unwrap()
                .is_empty()
        );
    }

    /// #1 regression guard: a discarded batch must not drop a prior committed
    /// full-text document (the writer is clean at begin_batch, so the rollback
    /// only reverts the batch's own ops).
    #[test]
    fn discard_batch_preserves_prior_committed_fulltext() {
        let mut engine = StorageEngine::new_in_memory(ft_schema());
        engine
            .create_node(
                "g1",
                "Document",
                "n1",
                props! { "title" => "keepme", "body" => "a" },
            )
            .unwrap();
        assert_eq!(
            engine
                .search_fulltext("g1", "keepme", None, 10)
                .unwrap()
                .len(),
            1
        );

        engine.begin_batch();
        engine
            .create_node(
                "g1",
                "Document",
                "n2",
                props! { "title" => "dropme", "body" => "b" },
            )
            .unwrap();
        engine.discard_batch();

        // Prior committed doc survives; the discarded batch's doc never appears.
        assert_eq!(
            engine
                .search_fulltext("g1", "keepme", None, 10)
                .unwrap()
                .len(),
            1
        );
        assert!(
            engine
                .search_fulltext("g1", "dropme", None, 10)
                .unwrap()
                .is_empty()
        );
    }

    /// #2: enabling full-text on a live engine (via replace_schema) doesn't
    /// build an index, so search/reindex fail loud instead of silently empty /
    /// Ok(0).
    #[test]
    fn fulltext_enabled_at_runtime_fails_loud_until_reopen() {
        let plain = Schema::from_yaml(
            r#"
schema:
  name: p
  version: 1
  node_types:
    Document:
      properties:
        title: { type: string }
  edge_types: {}
"#,
        )
        .unwrap();
        let mut engine = StorageEngine::new_in_memory(plain);
        // Genuinely no full-text → clean empty, no error.
        assert!(
            engine
                .search_fulltext("g1", "x", None, 10)
                .unwrap()
                .is_empty()
        );
        assert_eq!(engine.reindex_fulltext("g1").unwrap(), 0);

        // Enable full-text at runtime; no index is built for the live engine.
        engine.replace_schema(ft_schema());
        assert!(engine.search_fulltext("g1", "x", None, 10).is_err());
        assert!(engine.reindex_fulltext("g1").is_err());
    }

    /// #6: a rebuild can't run inside an open batch.
    #[test]
    fn reindex_inside_batch_errors() {
        let mut engine = StorageEngine::new_in_memory(ft_schema());
        engine.begin_batch();
        let err = engine.reindex_fulltext("g1").unwrap_err();
        engine.discard_batch();
        match err {
            DynoError::Storage(msg) => assert!(msg.contains("batch"), "got: {msg}"),
            other => panic!("expected Storage error, got {other:?}"),
        }
    }
}
