//! Pluggable storage backend — the narrow byte-level KV seam underneath
//! [`crate::engine::StorageEngine`].
//!
//! `StorageEngine` owns all the schema-aware behaviour (validation, the
//! reverse index, the read cache, the write-buffer overlay, the optional
//! full-text sidecar). Everything that actually *touches bytes on a
//! store* is funneled through the [`KvBackend`] trait defined here: six
//! primitives over six named column families. Adding a new backend
//! (Postgres, sled, …) means implementing this trait — nothing in
//! `StorageEngine` changes.
//!
//! The trait deliberately stays at the raw-KV level. Column families are
//! addressed by `&str` name (the `CF_*` consts), keys/values are opaque
//! bytes, and ordered prefix iteration is the one structural assumption
//! every backend must honour — it is what the adjacency and index scans
//! are built on.

use std::collections::HashMap;
#[cfg(feature = "rocksdb")]
use std::path::Path;
use std::sync::Arc;

use dynograph_core::DynoError;
#[cfg(feature = "rocksdb")]
use rocksdb::{BlockBasedOptions, ColumnFamilyDescriptor, DB, IteratorMode, Options, WriteBatch};

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
/// Sidecar embedding store. Keys are the same `{graph_id}\x00{node_type}\x00{node_id}`
/// shape as `CF_NODES` (1-to-1 with nodes); values are raw f32
/// little-endian bytes (no length prefix — the value's byte length /
/// 4 *is* the dimension). Embeddings are managed through dedicated
/// API methods (`set_embedding` / `get_embedding` / `delete_embedding`)
/// rather than riding on `properties`, since `PropertyType` has no
/// vector-of-floats variant. `delete_node` cascades to drop the
/// associated embedding so we don't accumulate orphans.
pub const CF_EMBEDDINGS: &str = "embeddings";

pub const ALL_CFS: &[&str] = &[
    CF_NODES,
    CF_EDGES,
    CF_ADJ_OUT,
    CF_ADJ_IN,
    CF_NODE_IDX,
    CF_EMBEDDINGS,
];

/// Per-column-family RocksDB options tuned for access patterns.
#[cfg(feature = "rocksdb")]
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
            // No fixed-prefix extractor: keys are
            // `{graph_id}\x00{node_id}\x00{edge_type}\x00{peer_id}` and
            // graph_id/node_id are user-supplied with no length bound.
            // The previous `fixed_prefix(48)` assumed UUID-shaped graph
            // ids and bled into the edge_type field on shorter ones
            // (e.g. graph_id="g1"), which is incorrect — RocksDB would
            // group keys by a prefix that crosses key boundaries.
            // Without an extractor, prefix scans use plain seek + range
            // iteration, which is correct on any key shape; SST block
            // ordering still keeps a node's adjacency keys close on
            // disk so seek-and-iterate cost is bounded.
        }
        CF_NODE_IDX => {
            // Prefix scans on `{graph_id}\x00{node_type}\x00{prop_name}\x00{value}\x00`.
            // Variable-length — no fixed prefix extractor. Seek-to-prefix still benefits
            // from SST block ordering.
        }
        CF_EMBEDDINGS => {
            // Mixed: point lookups by (graph_id, node_type, node_id) for
            // GET/DELETE; prefix scans by (graph_id, node_type) for the
            // future `scan_embeddings_by_type` path slice 8b will use to
            // rebuild HNSW state on rehydrate.
            let mut block_opts = BlockBasedOptions::default();
            block_opts.set_bloom_filter(10.0, false);
            opts.set_block_based_table_factory(&block_opts);
        }
        _ => {}
    }
    opts
}

/// Column family identifier — avoids String allocations in the write buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CfId {
    Nodes,
    Edges,
    AdjOut,
    AdjIn,
    NodeIdx,
    Embeddings,
}

impl CfId {
    pub(crate) fn from_str(s: &str) -> Option<Self> {
        match s {
            CF_NODES => Some(Self::Nodes),
            CF_EDGES => Some(Self::Edges),
            CF_ADJ_OUT => Some(Self::AdjOut),
            CF_ADJ_IN => Some(Self::AdjIn),
            CF_NODE_IDX => Some(Self::NodeIdx),
            CF_EMBEDDINGS => Some(Self::Embeddings),
            _ => None,
        }
    }
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Nodes => CF_NODES,
            Self::Edges => CF_EDGES,
            Self::AdjOut => CF_ADJ_OUT,
            Self::AdjIn => CF_ADJ_IN,
            Self::NodeIdx => CF_NODE_IDX,
            Self::Embeddings => CF_EMBEDDINGS,
        }
    }
}

/// One operation queued in a batch. Applied in insertion order at
/// `commit_batch` time, with `PrefixDelete` shadowing any earlier
/// `Put` whose key matches the prefix. (Memory: in-order loop. Rocks:
/// `delete_range_cf` adds a range tombstone that supersedes earlier
/// puts in the same `WriteBatch` by sequence number.)
pub(crate) enum BufferedOp {
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

/// Result of asking a buffered op "do you affect this `(cf, key)`?".
/// `Put(value)` = key has this value buffered; `Tombstoned` = key is
/// shadowed by a buffered `Delete` or covering `PrefixDelete`; the
/// outer `None` (returned by `BufferedOp::affecting`) = this op doesn't
/// match.
pub(crate) enum BufferedEffect<'a> {
    Put(&'a [u8]),
    Tombstoned,
}

impl BufferedOp {
    pub(crate) fn cf(&self) -> CfId {
        match self {
            Self::Put { cf, .. } | Self::Delete { cf, .. } | Self::PrefixDelete { cf, .. } => *cf,
        }
    }

    /// Returns `Some` if this op affects `(cf_id, key)`. Used by both
    /// `get` (reverse-walk: first match wins) and `overlay_buffer_on_scan`
    /// (forward-walk: each match overwrites the prior overlay state).
    pub(crate) fn affecting(&self, cf_id: CfId, key: &[u8]) -> Option<BufferedEffect<'_>> {
        if self.cf() != cf_id {
            return None;
        }
        match self {
            Self::Put { key: k, value, .. } if k.as_slice() == key => {
                Some(BufferedEffect::Put(value.as_slice()))
            }
            Self::Delete { key: k, .. } if k.as_slice() == key => Some(BufferedEffect::Tombstoned),
            Self::PrefixDelete { prefix, .. } if key.starts_with(prefix) => {
                Some(BufferedEffect::Tombstoned)
            }
            _ => None,
        }
    }
}

/// The byte-level storage seam. Six primitives over the six named
/// column families ([`ALL_CFS`]); everything schema-aware lives above
/// this in [`crate::engine::StorageEngine`].
///
/// Contract notes that every implementor must honour:
/// - **Ordered prefix iteration.** `prefix_scan` must return every
///   `(key, value)` whose key starts with `prefix`. The adjacency and
///   reverse-index scans depend on this being correct for arbitrary
///   (variable-length, NUL-separated) key shapes.
/// - **Atomic batch.** `commit_batch` applies the whole `ops` vector as
///   one all-or-nothing unit, in insertion order, with `PrefixDelete`
///   superseding earlier puts it covers.
/// - **Idempotent delete.** Deleting a missing key is a no-op, not an
///   error.
/// - **Unknown CF is an error**, not a silent no-op (fail loud).
///
/// `Send + Sync` so a `Box<dyn KvBackend>` keeps `StorageEngine`
/// shareable behind the registry's `Arc<RwLock<_>>`.
pub(crate) trait KvBackend: Send + Sync {
    /// Point lookup. `None` when the key is absent.
    fn get(&self, cf: &str, key: &[u8]) -> Result<Option<Arc<[u8]>>, DynoError>;

    /// Insert or overwrite. Takes ownership of `key`/`value` so the
    /// in-memory backend can move them straight into its map without a
    /// copy; the RocksDB backend borrows them for the WAL write.
    fn put(&mut self, cf: &str, key: Vec<u8>, value: Vec<u8>) -> Result<(), DynoError>;

    /// Remove a single key. Idempotent.
    fn delete(&mut self, cf: &str, key: &[u8]) -> Result<(), DynoError>;

    /// Every `(key, value)` whose key starts with `prefix`, in key order.
    #[allow(
        clippy::type_complexity,
        reason = "raw KV pairs straight out of the store; an alias would only obscure"
    )]
    fn prefix_scan(&self, cf: &str, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, DynoError>;

    /// Remove every key starting with `prefix`. Idempotent.
    fn prefix_delete(&mut self, cf: &str, prefix: &[u8]) -> Result<(), DynoError>;

    /// Apply `ops` atomically, in order. Takes the vector by value so the
    /// in-memory backend can move buffered keys/values into its maps; the
    /// RocksDB backend borrows them into a `WriteBatch`.
    fn commit_batch(&mut self, ops: Vec<BufferedOp>) -> Result<(), DynoError>;
}

// =============================================================================
// In-memory backend (testing)
// =============================================================================

/// Six `HashMap`s, one per column family. Test-oriented: no durability,
/// no cross-key atomicity guarantees beyond the single-threaded
/// `commit_batch` loop (which is enough for the engine's own tests).
pub(crate) struct MemoryBackend {
    nodes: HashMap<Vec<u8>, Vec<u8>>,
    edges: HashMap<Vec<u8>, Vec<u8>>,
    adj_out: HashMap<Vec<u8>, Vec<u8>>,
    adj_in: HashMap<Vec<u8>, Vec<u8>>,
    node_idx: HashMap<Vec<u8>, Vec<u8>>,
    embeddings: HashMap<Vec<u8>, Vec<u8>>,
}

impl MemoryBackend {
    pub(crate) fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            adj_out: HashMap::new(),
            adj_in: HashMap::new(),
            node_idx: HashMap::new(),
            embeddings: HashMap::new(),
        }
    }

    /// Pick the map for `cf`, erroring on an unknown CF. Shared by every
    /// op so the six-way `match cf` shape isn't repeated.
    fn store(&self, cf: &str) -> Result<&HashMap<Vec<u8>, Vec<u8>>, DynoError> {
        match cf {
            CF_NODES => Ok(&self.nodes),
            CF_EDGES => Ok(&self.edges),
            CF_ADJ_OUT => Ok(&self.adj_out),
            CF_ADJ_IN => Ok(&self.adj_in),
            CF_NODE_IDX => Ok(&self.node_idx),
            CF_EMBEDDINGS => Ok(&self.embeddings),
            _ => Err(DynoError::Storage(format!("Unknown CF: {}", cf))),
        }
    }

    fn store_mut(&mut self, cf: &str) -> Result<&mut HashMap<Vec<u8>, Vec<u8>>, DynoError> {
        match cf {
            CF_NODES => Ok(&mut self.nodes),
            CF_EDGES => Ok(&mut self.edges),
            CF_ADJ_OUT => Ok(&mut self.adj_out),
            CF_ADJ_IN => Ok(&mut self.adj_in),
            CF_NODE_IDX => Ok(&mut self.node_idx),
            CF_EMBEDDINGS => Ok(&mut self.embeddings),
            _ => Err(DynoError::Storage(format!("Unknown CF: {}", cf))),
        }
    }
}

impl KvBackend for MemoryBackend {
    fn get(&self, cf: &str, key: &[u8]) -> Result<Option<Arc<[u8]>>, DynoError> {
        Ok(self
            .store(cf)?
            .get(key)
            .map(|v| Arc::<[u8]>::from(v.as_slice())))
    }

    fn put(&mut self, cf: &str, key: Vec<u8>, value: Vec<u8>) -> Result<(), DynoError> {
        self.store_mut(cf)?.insert(key, value);
        Ok(())
    }

    fn delete(&mut self, cf: &str, key: &[u8]) -> Result<(), DynoError> {
        self.store_mut(cf)?.remove(key);
        Ok(())
    }

    #[allow(
        clippy::type_complexity,
        reason = "raw KV pairs straight out of the store; an alias would only obscure"
    )]
    fn prefix_scan(&self, cf: &str, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, DynoError> {
        let mut results: Vec<(Vec<u8>, Vec<u8>)> = self
            .store(cf)?
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        // Sort by key so the in-memory backend honours the trait's
        // ordered-prefix-scan contract. `HashMap` iteration is otherwise
        // arbitrary, which would diverge from RocksDB's lexicographic SST
        // ordering — and the in-memory backend is the default *test*
        // backend, so it must be a faithful stand-in for production order.
        // `Vec<u8>`'s `Ord` is lexicographic, matching RocksDB's default
        // byte-wise comparator.
        results.sort_by(|(a, _), (b, _)| a.cmp(b));
        Ok(results)
    }

    fn prefix_delete(&mut self, cf: &str, prefix: &[u8]) -> Result<(), DynoError> {
        self.store_mut(cf)?.retain(|k, _| !k.starts_with(prefix));
        Ok(())
    }

    fn commit_batch(&mut self, ops: Vec<BufferedOp>) -> Result<(), DynoError> {
        for op in ops {
            // store_mut errors only on an unknown CF; CfId::as_str
            // produces only known CFs, so this unwrap is unreachable in
            // practice.
            let store = self
                .store_mut(op.cf().as_str())
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
        Ok(())
    }
}

// =============================================================================
// RocksDB backend (production)
// =============================================================================

/// RocksDB on disk, one column family per [`ALL_CFS`] entry.
#[cfg(feature = "rocksdb")]
pub(crate) struct RocksBackend {
    db: DB,
}

#[cfg(feature = "rocksdb")]
impl RocksBackend {
    /// Open (creating if missing) a RocksDB store with all column
    /// families at `path`.
    pub(crate) fn open(path: &str) -> Result<Self, DynoError> {
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

        Ok(Self { db })
    }

    /// Collect every key in `cf_handle` starting with `prefix`, via seek +
    /// range iteration. The fallback for a prefix delete when no exclusive
    /// upper bound exists (an all-`0xFF` prefix, so `next_prefix` is `None`)
    /// and a single `delete_range_cf` tombstone can't be used. Shared by the
    /// immediate (`prefix_delete`) and batched (`commit_batch`) delete paths
    /// so the seek-and-take loop isn't open-coded twice.
    fn keys_with_prefix(
        db: &DB,
        cf_handle: &impl rocksdb::AsColumnFamilyRef,
        prefix: &[u8],
    ) -> Result<Vec<Vec<u8>>, DynoError> {
        let mut keys = Vec::new();
        for item in db.iterator_cf(
            cf_handle,
            IteratorMode::From(prefix, rocksdb::Direction::Forward),
        ) {
            let (key, _) = item.map_err(|e| DynoError::Storage(e.to_string()))?;
            if !key.starts_with(prefix) {
                break;
            }
            keys.push(key.to_vec());
        }
        Ok(keys)
    }
}

#[cfg(feature = "rocksdb")]
impl KvBackend for RocksBackend {
    fn get(&self, cf: &str, key: &[u8]) -> Result<Option<Arc<[u8]>>, DynoError> {
        let cf_handle = self
            .db
            .cf_handle(cf)
            .ok_or_else(|| DynoError::Storage(format!("CF not found: {}", cf)))?;
        self.db
            .get_cf(&cf_handle, key)
            .map(|opt| opt.map(Arc::<[u8]>::from))
            .map_err(|e| DynoError::Storage(e.to_string()))
    }

    fn put(&mut self, cf: &str, key: Vec<u8>, value: Vec<u8>) -> Result<(), DynoError> {
        let cf_handle = self
            .db
            .cf_handle(cf)
            .ok_or_else(|| DynoError::Storage(format!("CF not found: {}", cf)))?;
        self.db
            .put_cf(&cf_handle, &key, &value)
            .map_err(|e| DynoError::Storage(e.to_string()))
    }

    fn delete(&mut self, cf: &str, key: &[u8]) -> Result<(), DynoError> {
        let cf_handle = self
            .db
            .cf_handle(cf)
            .ok_or_else(|| DynoError::Storage(format!("CF not found: {}", cf)))?;
        self.db
            .delete_cf(&cf_handle, key)
            .map_err(|e| DynoError::Storage(e.to_string()))
    }

    #[allow(
        clippy::type_complexity,
        reason = "raw KV pairs straight out of the store; an alias would only obscure"
    )]
    fn prefix_scan(&self, cf: &str, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, DynoError> {
        let cf_handle = self
            .db
            .cf_handle(cf)
            .ok_or_else(|| DynoError::Storage(format!("CF not found: {}", cf)))?;
        let iter = self.db.iterator_cf(
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

    fn prefix_delete(&mut self, cf: &str, prefix: &[u8]) -> Result<(), DynoError> {
        let cf_handle = self
            .db
            .cf_handle(cf)
            .ok_or_else(|| DynoError::Storage(format!("CF not found: {}", cf)))?;
        // Single range tombstone via `delete_range_cf` when an
        // exclusive upper bound exists (almost always — the only
        // miss is an all-`0xFF` prefix). Fall back to per-key
        // deletes otherwise. NOTE: do not call while a snapshot
        // taken before this point is held — RocksDB's range
        // tombstones interact badly with older snapshots.
        if let Some(end) = crate::keys::next_prefix(prefix) {
            self.db
                .delete_range_cf(&cf_handle, prefix, &end)
                .map_err(|e| DynoError::Storage(e.to_string()))?;
        } else {
            for key in Self::keys_with_prefix(&self.db, &cf_handle, prefix)? {
                self.db
                    .delete_cf(&cf_handle, &key)
                    .map_err(|e| DynoError::Storage(e.to_string()))?;
            }
        }
        Ok(())
    }

    fn commit_batch(&mut self, ops: Vec<BufferedOp>) -> Result<(), DynoError> {
        // Resolve every cf_handle once up front so the per-op loop
        // doesn't re-do the string-keyed lookup against the same names —
        // N redundant HashMap lookups under an internal lock for an N-op
        // batch. `handle_for` then dispatches with an exhaustive match
        // over `CfId` rather than indexing a fixed-size array by `cf as
        // usize`: the array form desynced from the enum and panicked out
        // of bounds on `CfId::Embeddings` (index 5 into 5 entries). The
        // match makes adding a column family a compile error here instead.
        let nodes = self.db.cf_handle(CfId::Nodes.as_str());
        let edges = self.db.cf_handle(CfId::Edges.as_str());
        let adj_out = self.db.cf_handle(CfId::AdjOut.as_str());
        let adj_in = self.db.cf_handle(CfId::AdjIn.as_str());
        let node_idx = self.db.cf_handle(CfId::NodeIdx.as_str());
        let embeddings = self.db.cf_handle(CfId::Embeddings.as_str());
        let handle_for = |cf: CfId| -> Result<_, DynoError> {
            // `cf_handle` returns `Option<&ColumnFamily>` (Copy), so the
            // arms copy the handle out by value — no clone.
            match cf {
                CfId::Nodes => nodes,
                CfId::Edges => edges,
                CfId::AdjOut => adj_out,
                CfId::AdjIn => adj_in,
                CfId::NodeIdx => node_idx,
                CfId::Embeddings => embeddings,
            }
            .ok_or_else(|| DynoError::Storage(format!("CF not found: {}", cf.as_str())))
        };

        let mut batch = WriteBatch::default();
        for op in &ops {
            let cf_handle = handle_for(op.cf())?;
            match op {
                BufferedOp::Put { key, value, .. } => {
                    batch.put_cf(&cf_handle, key, value);
                }
                BufferedOp::Delete { key, .. } => {
                    batch.delete_cf(&cf_handle, key);
                }
                BufferedOp::PrefixDelete { prefix, .. } => {
                    // One range tombstone instead of N per-key deletes;
                    // falls back to iterate-and-delete only for the
                    // all-`0xFF` prefix corner case where no exclusive
                    // upper bound exists.
                    if let Some(end) = crate::keys::next_prefix(prefix) {
                        batch.delete_range_cf(&cf_handle, prefix, &end);
                    } else {
                        for key in Self::keys_with_prefix(&self.db, &cf_handle, prefix)? {
                            batch.delete_cf(&cf_handle, &key);
                        }
                    }
                }
            }
        }
        self.db
            .write(batch)
            .map_err(|e| DynoError::Storage(format!("Batch write failed: {}", e)))
    }
}
