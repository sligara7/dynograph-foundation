//! Wire shape for node endpoints.
//!
//! `StoredNode` includes `graph_id` because the storage layer is
//! agnostic to which graph it's serving; on the HTTP surface the
//! graph id is already in the URL, so the response body drops it.

use std::collections::HashMap;

use dynograph_core::Value;
use dynograph_storage::StoredNode;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct NodeResponse {
    pub node_type: String,
    pub node_id: String,
    pub properties: HashMap<String, Value>,
}

impl From<StoredNode> for NodeResponse {
    fn from(n: StoredNode) -> Self {
        Self {
            node_type: n.node_type,
            node_id: n.node_id,
            properties: n.properties,
        }
    }
}
