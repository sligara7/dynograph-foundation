//! `StorageEngine` — node/edge scans + counts. Split out of `engine.rs`; `use super::*`
//! inherits the shared imports and types from the parent `engine` module.
//! Private helper methods live in `engine/mod.rs` (a parent module, so
//! these methods reach them as descendants).

use super::*;

impl StorageEngine {
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
}
