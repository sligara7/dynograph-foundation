//! Wire shape for edge endpoints.
//!
//! Mirrors `NodeResponse` — drops `graph_id` (already in the URL).
//! `from_type` and `to_type` aren't part of the storage layer's
//! `StoredEdge` (only needed for `validate_edge` at create time, not
//! a key component) so they don't appear here either; if a caller
//! wants the type info they look it up via the schema.

use std::collections::HashMap;

use dynograph_core::Value;
use dynograph_storage::StoredEdge;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct EdgeResponse {
    pub edge_type: String,
    pub from_id: String,
    pub to_id: String,
    pub properties: HashMap<String, Value>,
}

impl From<StoredEdge> for EdgeResponse {
    fn from(e: StoredEdge) -> Self {
        Self {
            edge_type: e.edge_type,
            from_id: e.from_id,
            to_id: e.to_id,
            properties: e.properties,
        }
    }
}
