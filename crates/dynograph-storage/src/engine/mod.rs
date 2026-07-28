//! Storage engine — supports in-memory (testing) and RocksDB (production).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use dynograph_core::{DynoError, Schema, Value};

#[cfg(feature = "rocksdb")]
use crate::backend::RocksBackend;
use crate::backend::{
    ALL_CFS, BufferedEffect, BufferedOp, CF_ADJ_IN, CF_ADJ_OUT, CF_EDGES, CF_EMBEDDINGS,
    CF_NODE_IDX, CF_NODES, CfId, KvBackend, MemoryBackend,
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

mod batch;
mod edges;
mod embeddings;
mod fulltext;
mod nodes;
mod scan;

#[cfg(feature = "fulltext")]
pub use fulltext::FulltextHit;

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
    ///
    /// Requires the `rocksdb` feature (on by default). A crate built with
    /// `--no-default-features` keeps this method but fails loud (see the
    /// `cfg(not(feature = "rocksdb"))` variant below) rather than silently
    /// degrading to an in-memory store — on-disk mode is selected explicitly,
    /// so quietly dropping it would lose the caller's data.
    #[cfg(feature = "rocksdb")]
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

    /// Fail-loud stub when the crate is built without the `rocksdb` feature.
    /// The method stays in the API so callers (e.g. the service registry) need
    /// no `cfg` gating, but on-disk storage genuinely isn't compiled into this
    /// build, so we return an actionable error instead of a silent fallback.
    #[cfg(not(feature = "rocksdb"))]
    pub fn new_rocksdb(_schema: Schema, _path: &str) -> Result<Self, DynoError> {
        Err(DynoError::Storage(
            "on-disk RocksDB storage is unavailable: this build was compiled without the \
             `rocksdb` feature. Rebuild with default features (or `--features rocksdb`), or \
             use the in-memory backend."
                .to_string(),
        ))
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

#[cfg(all(test, feature = "fulltext"))]
mod fulltext_tests;
#[cfg(test)]
mod tests;
