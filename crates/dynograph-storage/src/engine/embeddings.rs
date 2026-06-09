//! `StorageEngine` — embedding storage. Split out of `engine.rs`; `use super::*`
//! inherits the shared imports and types from the parent `engine` module.
//! Private helper methods live in `engine/mod.rs` (a parent module, so
//! these methods reach them as descendants).

use super::*;

impl StorageEngine {
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
}
