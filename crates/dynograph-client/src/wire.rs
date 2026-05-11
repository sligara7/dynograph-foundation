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

// =========================================================================
// Tier-2 + new-in-v0.5.6 routes. Complex routes (batch, edges:collect,
// traverse, nodes:scan, resolve-or-create) take/return `serde_json::Value`
// — same untyped-body pattern create_node already uses for properties.
// Future PRs can replace any of these with typed shells per-endpoint as
// real consumers grow IDE-autocomplete pressure.
// =========================================================================

/// Returned by `POST /v1/graphs/{id}/resolve-or-create`. `match_kind`
/// is the wire-string form of the service-side enum
/// (`auto_merge` / `vector_merge` / `created_new`).
#[derive(Debug, Clone, Deserialize)]
pub struct ResolveOrCreateResponse {
    pub id: String,
    pub was_created: bool,
    pub match_kind: String,
}

/// One entry in `POST /v1/graphs/{id}/nodes:exists`. `id` is `None`
/// when `exists = false`; otherwise carries the node_id of the first
/// match.
#[derive(Debug, Clone, Deserialize)]
pub struct NodeExistence {
    #[serde(rename = "type")]
    pub node_type: String,
    pub name: String,
    pub exists: bool,
    pub id: Option<String>,
}

/// Returned by `POST /v1/graphs/{id}/nodes:exists`. Result order
/// mirrors the request's `queries` order.
#[derive(Debug, Clone, Deserialize)]
pub struct NodesExistsResponse {
    pub results: Vec<NodeExistence>,
}

/// Returned by `POST /v1/graphs/{id}/edges/.../welford_update`.
#[derive(Debug, Clone, Deserialize)]
pub struct WelfordUpdateResponse {
    pub score: f64,
    pub score_m2: f64,
    pub score_stddev: f64,
    pub score_min: f64,
    pub score_max: f64,
    pub score_count: i64,
}
