//! `StorageEngine` — edge CRUD. Split out of `engine.rs`; `use super::*`
//! inherits the shared imports and types from the parent `engine` module.
//! Private helper methods live in `engine/mod.rs` (a parent module, so
//! these methods reach them as descendants).

use super::*;

impl StorageEngine {
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
}
