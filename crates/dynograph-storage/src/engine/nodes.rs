//! `StorageEngine` — node CRUD. Split out of `engine.rs`; `use super::*`
//! inherits the shared imports and types from the parent `engine` module.
//! Private helper methods live in `engine/mod.rs` (a parent module, so
//! these methods reach them as descendants).

use super::*;

impl StorageEngine {
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
}
