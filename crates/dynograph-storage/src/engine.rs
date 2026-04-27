//! Storage engine — supports in-memory (testing) and RocksDB (production).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use dynograph_core::{DynoError, Schema, Value};
use rocksdb::{
    BlockBasedOptions, ColumnFamilyDescriptor, DB, IteratorMode, Options, SliceTransform,
    WriteBatch,
};

use crate::cache::{CacheConfig, ReadCache};

/// Column family names.
pub const CF_NODES: &str = "nodes";
pub const CF_EDGES: &str = "edges";
pub const CF_ADJ_OUT: &str = "adj_out";
pub const CF_ADJ_IN: &str = "adj_in";
/// Reverse index for schema-declared indexed properties. Keys are
/// `{graph_id}\x00{node_type}\x00{prop_name}\x00{prop_value}\x00{node_id}`
/// with empty values; the payload is the node_id suffix. Populated on
/// create/update and cleaned up on delete, driven by `Schema::indexed_properties`.
pub const CF_NODE_IDX: &str = "node_idx";

const ALL_CFS: &[&str] = &[CF_NODES, CF_EDGES, CF_ADJ_OUT, CF_ADJ_IN, CF_NODE_IDX];

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

/// Backend storage — either in-memory HashMap or RocksDB on disk.
enum Backend {
    Memory {
        nodes: HashMap<Vec<u8>, Vec<u8>>,
        edges: HashMap<Vec<u8>, Vec<u8>>,
        adj_out: HashMap<Vec<u8>, Vec<u8>>,
        adj_in: HashMap<Vec<u8>, Vec<u8>>,
        node_idx: HashMap<Vec<u8>, Vec<u8>>,
    },
    Rocks {
        db: DB,
    },
}

impl Backend {
    /// Pick the in-memory store for `cf`. Errors when called against a
    /// `Rocks` backend or an unknown CF. Used by every Memory-path
    /// op (get / put / delete / prefix_scan / prefix_delete) so the
    /// five-way `match cf` shape isn't repeated.
    fn memory_store_mut(&mut self, cf: &str) -> Result<&mut HashMap<Vec<u8>, Vec<u8>>, DynoError> {
        match self {
            Backend::Memory {
                nodes,
                edges,
                adj_out,
                adj_in,
                node_idx,
            } => match cf {
                CF_NODES => Ok(nodes),
                CF_EDGES => Ok(edges),
                CF_ADJ_OUT => Ok(adj_out),
                CF_ADJ_IN => Ok(adj_in),
                CF_NODE_IDX => Ok(node_idx),
                _ => Err(DynoError::Storage(format!("Unknown CF: {}", cf))),
            },
            Backend::Rocks { .. } => Err(DynoError::Storage(
                "memory_store_mut called on Rocks backend".to_string(),
            )),
        }
    }

    fn memory_store(&self, cf: &str) -> Result<&HashMap<Vec<u8>, Vec<u8>>, DynoError> {
        match self {
            Backend::Memory {
                nodes,
                edges,
                adj_out,
                adj_in,
                node_idx,
            } => match cf {
                CF_NODES => Ok(nodes),
                CF_EDGES => Ok(edges),
                CF_ADJ_OUT => Ok(adj_out),
                CF_ADJ_IN => Ok(adj_in),
                CF_NODE_IDX => Ok(node_idx),
                _ => Err(DynoError::Storage(format!("Unknown CF: {}", cf))),
            },
            Backend::Rocks { .. } => Err(DynoError::Storage(
                "memory_store called on Rocks backend".to_string(),
            )),
        }
    }
}

/// Per-column-family RocksDB options tuned for access patterns.
fn cf_options(cf_name: &str) -> Options {
    let mut opts = Options::default();
    match cf_name {
        CF_NODES | CF_EDGES => {
            // Point lookups — bloom filter reduces unnecessary disk reads
            let mut block_opts = BlockBasedOptions::default();
            block_opts.set_bloom_filter(10.0, false);
            opts.set_block_based_table_factory(&block_opts);
        }
        CF_ADJ_OUT | CF_ADJ_IN => {
            // Sequential prefix scans — prefix extractor enables efficient iteration.
            // 48 bytes covers typical graph_id (36 UUID) + separator + start of node_id.
            opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(48));
        }
        CF_NODE_IDX => {
            // Prefix scans on `{graph_id}\x00{node_type}\x00{prop_name}\x00{value}\x00`.
            // Variable-length — no fixed prefix extractor. Seek-to-prefix still benefits
            // from SST block ordering.
        }
        _ => {}
    }
    opts
}

/// Column family identifier — avoids String allocations in the write buffer.
#[derive(Debug, Clone, Copy)]
enum CfId {
    Nodes,
    Edges,
    AdjOut,
    AdjIn,
    NodeIdx,
}

impl CfId {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            CF_NODES => Some(Self::Nodes),
            CF_EDGES => Some(Self::Edges),
            CF_ADJ_OUT => Some(Self::AdjOut),
            CF_ADJ_IN => Some(Self::AdjIn),
            CF_NODE_IDX => Some(Self::NodeIdx),
            _ => None,
        }
    }
    fn as_str(&self) -> &'static str {
        match self {
            Self::Nodes => CF_NODES,
            Self::Edges => CF_EDGES,
            Self::AdjOut => CF_ADJ_OUT,
            Self::AdjIn => CF_ADJ_IN,
            Self::NodeIdx => CF_NODE_IDX,
        }
    }
}

/// One operation queued in a batch. Applied in insertion order at
/// `commit_batch` time, with `PrefixDelete` shadowing any earlier
/// `Put` whose key matches the prefix. (Memory: in-order loop. Rocks:
/// `delete_range_cf` adds a range tombstone that supersedes earlier
/// puts in the same `WriteBatch` by sequence number.)
enum BufferedOp {
    Put {
        cf: CfId,
        key: Vec<u8>,
        value: Vec<u8>,
    },
    Delete {
        cf: CfId,
        key: Vec<u8>,
    },
    PrefixDelete {
        cf: CfId,
        prefix: Vec<u8>,
    },
}

impl BufferedOp {
    fn cf(&self) -> CfId {
        match self {
            Self::Put { cf, .. } | Self::Delete { cf, .. } | Self::PrefixDelete { cf, .. } => *cf,
        }
    }
}

/// The storage engine — schema-validated graph storage.
pub struct StorageEngine {
    schema: Schema,
    backend: Backend,
    /// LRU read cache for node lookups and adjacency scans.
    /// Mutex allows cache updates through &self (get path is immutable at API level).
    read_cache: Mutex<ReadCache>,
    /// When `Some`, all writes (put / delete / prefix-delete) buffer
    /// here instead of hitting the backend. `commit_batch` flushes
    /// atomically; `discard_batch` drops them.
    write_buffer: Option<Vec<BufferedOp>>,
}

impl StorageEngine {
    /// Create an in-memory storage engine (for testing).
    pub fn new_in_memory(schema: Schema) -> Self {
        Self {
            schema,
            backend: Backend::Memory {
                nodes: HashMap::new(),
                edges: HashMap::new(),
                adj_out: HashMap::new(),
                adj_in: HashMap::new(),
                node_idx: HashMap::new(),
            },
            read_cache: Mutex::new(ReadCache::new(CacheConfig::default())),
            write_buffer: None,
        }
    }

    /// Create a RocksDB-backed storage engine (for production).
    pub fn new_rocksdb(schema: Schema, path: &str) -> Result<Self, DynoError> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cf_descriptors: Vec<ColumnFamilyDescriptor> = ALL_CFS
            .iter()
            .map(|name| ColumnFamilyDescriptor::new(*name, cf_options(name)))
            .collect();

        let db = DB::open_cf_descriptors(&opts, Path::new(path), cf_descriptors).map_err(|e| {
            DynoError::Storage(format!("Failed to open RocksDB at {}: {}", path, e))
        })?;

        Ok(Self {
            schema,
            backend: Backend::Rocks { db },
            read_cache: Mutex::new(ReadCache::new(CacheConfig::default())),
            write_buffer: None,
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
        self.schema = new_schema;
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

        match &mut self.backend {
            Backend::Memory { .. } => {
                self.backend.memory_store_mut(cf)?.insert(key, value);
                Ok(())
            }
            Backend::Rocks { db } => {
                let cf_handle = db
                    .cf_handle(cf)
                    .ok_or_else(|| DynoError::Storage(format!("CF not found: {}", cf)))?;
                db.put_cf(&cf_handle, &key, &value)
                    .map_err(|e| DynoError::Storage(e.to_string()))
            }
        }
    }

    fn get(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>, DynoError> {
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
                    .put(key.to_vec(), data.clone());
            }
            return Ok(result);
        }

        self.backend_get(cf, key)
    }

    fn backend_get(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>, DynoError> {
        match &self.backend {
            Backend::Memory { .. } => Ok(self.backend.memory_store(cf)?.get(key).cloned()),
            Backend::Rocks { db } => {
                let cf_handle = db
                    .cf_handle(cf)
                    .ok_or_else(|| DynoError::Storage(format!("CF not found: {}", cf)))?;
                db.get_cf(&cf_handle, key)
                    .map_err(|e| DynoError::Storage(e.to_string()))
            }
        }
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
        match &mut self.backend {
            Backend::Memory { .. } => {
                self.backend.memory_store_mut(cf)?.remove(key);
                Ok(())
            }
            Backend::Rocks { db } => {
                let cf_handle = db
                    .cf_handle(cf)
                    .ok_or_else(|| DynoError::Storage(format!("CF not found: {}", cf)))?;
                db.delete_cf(&cf_handle, key)
                    .map_err(|e| DynoError::Storage(e.to_string()))
            }
        }
    }

    /// Scan all keys with a given prefix in a column family.
    #[allow(
        clippy::type_complexity,
        reason = "raw KV pairs straight out of RocksDB; an alias would only obscure"
    )]
    fn prefix_scan(&self, cf: &str, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, DynoError> {
        match &self.backend {
            Backend::Memory { .. } => Ok(self
                .backend
                .memory_store(cf)?
                .iter()
                .filter(|(k, _)| k.starts_with(prefix))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()),
            Backend::Rocks { db } => {
                let cf_handle = db
                    .cf_handle(cf)
                    .ok_or_else(|| DynoError::Storage(format!("CF not found: {}", cf)))?;
                let iter = db.iterator_cf(
                    &cf_handle,
                    IteratorMode::From(prefix, rocksdb::Direction::Forward),
                );
                let mut results = Vec::new();
                for item in iter {
                    let (key, value) = item.map_err(|e| DynoError::Storage(e.to_string()))?;
                    if !key.starts_with(prefix) {
                        break; // Past our prefix
                    }
                    results.push((key.to_vec(), value.to_vec()));
                }
                Ok(results)
            }
        }
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
        match &mut self.backend {
            Backend::Memory { .. } => {
                self.backend
                    .memory_store_mut(cf)?
                    .retain(|k, _| !k.starts_with(prefix));
                Ok(())
            }
            Backend::Rocks { db } => {
                let cf_handle = db
                    .cf_handle(cf)
                    .ok_or_else(|| DynoError::Storage(format!("CF not found: {}", cf)))?;
                // Single range tombstone via `delete_range_cf` when an
                // exclusive upper bound exists (almost always — the only
                // miss is an all-`0xFF` prefix). Fall back to per-key
                // deletes otherwise. NOTE: do not call while a snapshot
                // taken before this point is held — RocksDB's range
                // tombstones interact badly with older snapshots.
                if let Some(end) = crate::keys::next_prefix(prefix) {
                    db.delete_range_cf(&cf_handle, prefix, &end)
                        .map_err(|e| DynoError::Storage(e.to_string()))?;
                } else {
                    let keys: Vec<Vec<u8>> = db
                        .iterator_cf(
                            &cf_handle,
                            IteratorMode::From(prefix, rocksdb::Direction::Forward),
                        )
                        .take_while(|item| item.as_ref().is_ok_and(|(k, _)| k.starts_with(prefix)))
                        .filter_map(|item| item.ok().map(|(k, _)| k.to_vec()))
                        .collect();
                    for key in keys {
                        db.delete_cf(&cf_handle, &key)
                            .map_err(|e| DynoError::Storage(e.to_string()))?;
                    }
                }
                Ok(())
            }
        }
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

        let key = crate::keys::node_key(graph_id, node_type, node_id);
        let value =
            rmp_serde::to_vec(&properties).map_err(|e| DynoError::Serialization(e.to_string()))?;

        self.put(CF_NODES, key, value)?;
        if self.schema.has_indexed_properties(node_type) {
            self.write_index_entries(graph_id, node_type, node_id, &properties)?;
        }

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
        properties: HashMap<String, Value>,
    ) -> Result<StoredEdge, DynoError> {
        self.schema.validate_edge(edge_type, from_type, to_type)?;

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
            let node_id = String::from_utf8_lossy(node_id_bytes).to_string();
            if let Some(node) = self.get_node(graph_id, node_type, &node_id)? {
                results.push(node);
            }
        }
        Ok(results)
    }

    /// Count all nodes of a given type in a graph.
    pub fn count_nodes(&self, graph_id: &str, node_type: &str) -> usize {
        let prefix = crate::keys::node_type_prefix(graph_id, node_type);
        self.prefix_scan(CF_NODES, &prefix)
            .map(|entries| entries.len())
            .unwrap_or(0)
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
            let node_id = String::from_utf8_lossy(after_prefix).to_string();
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
            let edge_type = String::from_utf8_lossy(parts[0]).to_string();
            let to_id = String::from_utf8_lossy(parts[1]).to_string();

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
            let edge_type = String::from_utf8_lossy(parts[0]).to_string();
            let from_id = String::from_utf8_lossy(parts[1]).to_string();

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

        match &mut self.backend {
            Backend::Memory { .. } => {
                for op in buffer {
                    // memory_store_mut errors only on Rocks/unknown-cf;
                    // CfId::as_str produces only known CFs, so unwrap is
                    // unreachable in practice.
                    let store = self
                        .backend
                        .memory_store_mut(op.cf().as_str())
                        .expect("CfId is always a known CF");
                    match op {
                        BufferedOp::Put { key, value, .. } => {
                            store.insert(key, value);
                        }
                        BufferedOp::Delete { key, .. } => {
                            store.remove(&key);
                        }
                        BufferedOp::PrefixDelete { prefix, .. } => {
                            store.retain(|k, _| !k.starts_with(&prefix));
                        }
                    }
                }
            }
            Backend::Rocks { db } => {
                // Resolve all 5 cf_handles up front so the per-op loop
                // doesn't re-do the string-keyed lookup against the same
                // five names — that's N HashMap lookups under an
                // internal lock for an N-op batch, all redundant.
                let handles = [
                    db.cf_handle(CfId::Nodes.as_str()),
                    db.cf_handle(CfId::Edges.as_str()),
                    db.cf_handle(CfId::AdjOut.as_str()),
                    db.cf_handle(CfId::AdjIn.as_str()),
                    db.cf_handle(CfId::NodeIdx.as_str()),
                ];
                let handle_for = |cf: CfId| -> Result<_, DynoError> {
                    handles[cf as usize]
                        .ok_or_else(|| DynoError::Storage(format!("CF not found: {}", cf.as_str())))
                };

                let mut batch = WriteBatch::default();
                for op in &buffer {
                    let cf_handle = handle_for(op.cf())?;
                    match op {
                        BufferedOp::Put { key, value, .. } => {
                            batch.put_cf(&cf_handle, key, value);
                        }
                        BufferedOp::Delete { key, .. } => {
                            batch.delete_cf(&cf_handle, key);
                        }
                        BufferedOp::PrefixDelete { prefix, .. } => {
                            // One range tombstone instead of N
                            // per-key deletes; falls back to iterate-
                            // and-delete only for the all-`0xFF` prefix
                            // corner case where no exclusive upper
                            // bound exists.
                            if let Some(end) = crate::keys::next_prefix(prefix) {
                                batch.delete_range_cf(&cf_handle, prefix, &end);
                            } else {
                                for item in db.iterator_cf(
                                    &cf_handle,
                                    IteratorMode::From(prefix, rocksdb::Direction::Forward),
                                ) {
                                    let (key, _) =
                                        item.map_err(|e| DynoError::Storage(e.to_string()))?;
                                    if !key.starts_with(prefix) {
                                        break;
                                    }
                                    batch.delete_cf(&cf_handle, &key);
                                }
                            }
                        }
                    }
                }
                db.write(batch)
                    .map_err(|e| DynoError::Storage(format!("Batch write failed: {}", e)))?;
            }
        }

        Ok(count)
    }

    /// Discard all buffered writes without committing.
    pub fn discard_batch(&mut self) {
        self.write_buffer = None;
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

    // In-memory tests (existing)

    #[test]
    fn create_and_get_node() {
        let mut engine = StorageEngine::new_in_memory(test_schema());
        let props = props! { "name" => "Alice", "role" => "protagonist" };
        let node = engine.create_node("g1", "Character", "c1", props).unwrap();
        assert_eq!(node.node_id, "c1");
        let fetched = engine.get_node("g1", "Character", "c1").unwrap().unwrap();
        assert_eq!(fetched.properties["name"].as_str().unwrap(), "Alice");
    }

    #[test]
    fn create_node_validates_schema() {
        let mut engine = StorageEngine::new_in_memory(test_schema());
        let result = engine.create_node("g1", "Character", "c1", HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn create_node_validates_enum() {
        let mut engine = StorageEngine::new_in_memory(test_schema());
        let props = props! { "name" => "Bob", "role" => "villain" };
        assert!(engine.create_node("g1", "Character", "c1", props).is_err());
    }

    #[test]
    fn get_nonexistent_node_returns_none() {
        let engine = StorageEngine::new_in_memory(test_schema());
        assert!(
            engine
                .get_node("g1", "Character", "missing")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn delete_node() {
        let mut engine = StorageEngine::new_in_memory(test_schema());
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
        let mut engine = StorageEngine::new_in_memory(test_schema());
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
        let mut engine = StorageEngine::new_in_memory(test_schema());
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
        let mut engine = StorageEngine::new_in_memory(test_schema());
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
        let mut engine = StorageEngine::new_in_memory(test_schema());
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
        let mut engine = StorageEngine::new_in_memory(test_schema());
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
        let mut engine = StorageEngine::new_in_memory(test_schema());
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
        let mut engine = StorageEngine::new_in_memory(test_schema());
        engine
            .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
            .unwrap();
        engine
            .create_node("g1", "Character", "c2", props! { "name" => "Bob" })
            .unwrap();
        engine
            .create_node("g1", "Location", "loc1", props! { "name" => "Tavern" })
            .unwrap();
        assert_eq!(engine.count_nodes("g1", "Character"), 2);
        assert_eq!(engine.count_nodes("g1", "Location"), 1);
    }

    #[test]
    fn graph_isolation() {
        let mut engine = StorageEngine::new_in_memory(test_schema());
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
        let mut engine = StorageEngine::new_in_memory(indexed_schema());
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
        let mut engine = StorageEngine::new_in_memory(indexed_schema());
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
        let mut engine = StorageEngine::new_in_memory(indexed_schema());
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
    fn delete_cleans_up_index_entries() {
        let mut engine = StorageEngine::new_in_memory(indexed_schema());
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
        let mut engine = StorageEngine::new_in_memory(indexed_schema());
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
        let engine = StorageEngine::new_in_memory(indexed_schema());
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
            assert_eq!(engine.count_nodes("g1", "Character"), 2);
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

        assert_eq!(engine.count_nodes("g1", "Character"), 2);
        assert_eq!(engine.count_nodes("g1", "Location"), 1);

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
    fn batch_buffers_mixed_put_and_delete_until_commit() {
        let mut engine = StorageEngine::new_in_memory(test_schema());
        // Pre-existing node we'll delete inside the batch.
        engine
            .create_node("g1", "Character", "to_delete", props! { "name" => "Goner" })
            .unwrap();

        engine.begin_batch();
        engine
            .create_node("g1", "Character", "to_create", props! { "name" => "New" })
            .unwrap();
        engine.delete_node("g1", "Character", "to_delete").unwrap();

        // Pre-commit: neither op is visible.
        assert!(
            engine
                .get_node("g1", "Character", "to_create")
                .unwrap()
                .is_none(),
            "buffered create should not be visible until commit"
        );
        assert!(
            engine
                .get_node("g1", "Character", "to_delete")
                .unwrap()
                .is_some(),
            "buffered delete should not be visible until commit"
        );

        engine.commit_batch().unwrap();

        // Post-commit: both ops applied atomically.
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
        let mut engine = StorageEngine::new_in_memory(test_schema());
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
    fn batched_replace_on_indexed_node_is_atomic_against_index_scans() {
        // Replace on an indexed node must apply the index delete + put
        // atomically — index scans may not see a "deleted-but-not-yet-
        // re-written" state mid-batch.
        let mut engine = StorageEngine::new_in_memory(indexed_schema());
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

        // Pre-commit: index unchanged — f1 still under sA, no sB entry.
        let sa = Value::from("sA");
        let sb = Value::from("sB");
        assert_eq!(
            engine
                .scan_nodes_by_property("g1", "Fragment", "story_id", &sa)
                .unwrap()
                .len(),
            1,
            "old story_id should still be indexed pre-commit"
        );
        assert_eq!(
            engine
                .scan_nodes_by_property("g1", "Fragment", "story_id", &sb)
                .unwrap()
                .len(),
            0,
            "new story_id should not be visible pre-commit"
        );

        engine.commit_batch().unwrap();

        // Post-commit: index swapped atomically.
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
    fn batch_buffers_prefix_delete_until_commit() {
        // delete_node uses prefix_delete on adj_out/adj_in; verify that
        // running it inside a batch doesn't leak adjacency removal mid-batch.
        let mut engine = StorageEngine::new_in_memory(test_schema());
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

        // Pre-commit: alice still resolves; her outgoing edge still scans.
        assert!(
            engine
                .get_node("g1", "Character", "alice")
                .unwrap()
                .is_some()
        );
        assert_eq!(
            engine
                .scan_outgoing_edges("g1", "alice", None)
                .unwrap()
                .len(),
            1
        );

        engine.commit_batch().unwrap();

        // Post-commit: alice and her adjacency + the edge + bob's
        // inverse adjacency are all gone.
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

        let mut engine = StorageEngine::new_in_memory(old);
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
}
