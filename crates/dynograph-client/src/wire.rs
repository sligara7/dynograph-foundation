//! Wire types — mirror of `dynograph-service`'s response shapes.
//!
//! Duplicated rather than re-exported because making `dynograph-client`
//! depend on `dynograph-service` would drag axum + tokio + rocksdb
//! into a thin HTTP client. The trade-off: the two crates can drift
//! if a developer changes one without the other; integration tests
//! (`tests/integration.rs`) exercise the round-trip against a real
//! in-process service to pin the contract.
//!
//! When a real consumer needs both crates and the duplication starts
//! to bite, extract these types to a `dynograph-wire` crate that
//! both depend on (a future-slice refactor).

use std::collections::HashMap;

use dynograph_core::{Schema, Value};
use serde::Deserialize;

/// Returned by `POST /v1/graphs` (creation), `GET /v1/graphs/{id}/schema`
/// (full read), and `PUT /v1/graphs/{id}/schema` (replacement).
#[derive(Debug, Clone, Deserialize)]
pub struct SchemaResponse {
    pub id: String,
    pub wire_version: String,
    pub content_hash: String,
    pub schema: Schema,
}

/// Returned by `GET /v1/graphs/{id}` — schema-less view for cheap
/// existence checks and content-hash drift comparisons.
#[derive(Debug, Clone, Deserialize)]
pub struct GraphMetadataResponse {
    pub id: String,
    pub wire_version: String,
    pub content_hash: String,
}

/// Returned by `POST /v1/graphs/{id}/nodes`, `GET /…/{type}/{id}`,
/// `PUT /…/{type}/{id}`. The `graph_id` lives in the URL, not the
/// body.
#[derive(Debug, Clone, Deserialize)]
pub struct NodeResponse {
    pub node_type: String,
    pub node_id: String,
    pub properties: HashMap<String, Value>,
}

/// Returned by `GET /v1/graphs/{id}/nodes?…`. Envelope keeps room for
/// pagination cursors without a wire shape break.
#[derive(Debug, Clone, Deserialize)]
pub struct NodeListResponse {
    pub nodes: Vec<NodeResponse>,
}

/// Returned by `POST /v1/graphs/{id}/edges`, `GET /…/{type}/{from}/{to}`,
/// `PATCH /…/{type}/{from}/{to}`.
#[derive(Debug, Clone, Deserialize)]
pub struct EdgeResponse {
    pub edge_type: String,
    pub from_id: String,
    pub to_id: String,
    pub properties: HashMap<String, Value>,
}

/// Returned by `PUT` and `GET` on `/v1/graphs/{id}/nodes/{type}/{id}/embedding`.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingResponse {
    pub node_type: String,
    pub node_id: String,
    pub embedding: Vec<f32>,
}

/// One hit in a `/similar` response.
#[derive(Debug, Clone, Deserialize)]
pub struct SimilarHit {
    pub node_id: String,
    pub score: f32,
}

/// Returned by `POST /v1/graphs/{id}/similar`.
#[derive(Debug, Clone, Deserialize)]
pub struct SimilarResponse {
    pub results: Vec<SimilarHit>,
}
