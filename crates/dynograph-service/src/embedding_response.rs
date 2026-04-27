//! Wire shape for embedding endpoints (`*/nodes/{node_type}/{node_id}/embedding`).
//!
//! Embeddings ride a sidecar storage path (`CF_EMBEDDINGS`) rather
//! than node properties, since `PropertyType` has no vector-of-floats
//! variant. The wire shape keeps `node_type`/`node_id` for symmetry
//! with `NodeResponse`; `graph_id` stays in the URL.

use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub node_type: String,
    pub node_id: String,
    pub embedding: Vec<f32>,
}
