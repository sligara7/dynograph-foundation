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
    /// Provenance of the winning match pair (v0.9.3): `name_to_name`,
    /// `name_to_stored_alias`, `incoming_alias_to_name`, or
    /// `incoming_alias_to_stored_alias`. `None` on `created_new` (and
    /// when talking to a pre-0.9.3 server).
    #[serde(default)]
    pub match_source: Option<String>,
    /// Incoming aliases excluded as non-identifying — they matched two
    /// or more distinct in-scope candidates (v0.9.3).
    #[serde(default)]
    pub ambiguous_aliases: Vec<String>,
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

/// Single-field `{"result": T}` envelope used by every
/// `POST /v1/util/*` math endpoint. Aliases below pin the concrete
/// `T` per route: scalar (f64), vector (Vec<f64>), or ratio (u32).
#[derive(Debug, Clone, Deserialize)]
pub struct UtilResult<T> {
    pub result: T,
}

/// `T = f64` — the scalar-returning util routes: cosine_similarity,
/// dot_product, (squared_)euclidean_distance, manhattan_distance,
/// l2_norm, pearson_correlation, spearman_correlation,
/// linear_regression_slope, mean, variance, std_dev, median, percentile.
pub type UtilScalarResponse = UtilResult<f64>;

/// `T = Vec<f64>` — the vector-returning util routes: hadamard,
/// hadamard_division, add, subtract, scale, negate, elementwise_power,
/// l2_normalize, centroid, softmax.
pub type UtilVectorResponse = UtilResult<Vec<f64>>;

/// `T = u32` (0..=100) — jaro_winkler, token_sort_ratio.
pub type UtilRatioResponse = UtilResult<u32>;

/// Returned by `POST /v1/util/pairwise_cosine` and
/// `POST /v1/util/pairwise_distance` — the N×N result matrix
/// (row-major) over the input `vectors`.
#[derive(Debug, Clone, Deserialize)]
pub struct MatrixResponse {
    pub matrix: Vec<Vec<f64>>,
}

/// Distance metric for [`util_pairwise_distance`](crate::DynographClient::util_pairwise_distance).
/// Wire form is the snake_case variant name. Mirrors the service-side
/// `DistanceMetric` (duplicated, not imported — see the module note).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    #[default]
    Euclidean,
    SquaredEuclidean,
    Manhattan,
    /// Cosine distance, `1 - cosine_similarity`.
    Cosine,
}

// =========================================================================
// search:text / search:hybrid (v0.7.0 service surface). The keyword leg of
// each is behind the server's `fulltext` build feature: without it the route
// returns 501, surfaced as `ClientError::Http { status: 501, .. }`.
// =========================================================================

/// One hit in a `POST /v1/graphs/{id}/search:text` response.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchTextHit {
    pub node_id: String,
    pub node_type: String,
    pub score: f32,
}

/// Returned by `POST /v1/graphs/{id}/search:text`. Hits are
/// highest-score-first.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchTextResponse {
    pub results: Vec<SearchTextHit>,
}

/// Returned by `POST /v1/graphs/{id}/search:reindex` — the number of
/// nodes (re)indexed into the full-text index.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchReindexResponse {
    pub indexed: usize,
}

/// One leg's contribution to a fused `search:hybrid` hit: its 1-based
/// rank within that leg and that leg's native score (BM25 for keyword,
/// similarity for vector).
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct HybridLegInfo {
    pub rank: usize,
    pub score: f32,
}

/// Per-leg breakdown attached to each hybrid hit. Only legs that ranked
/// the node are present.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct HybridLegBreakdown {
    pub vector: Option<HybridLegInfo>,
    pub keyword: Option<HybridLegInfo>,
}

/// One fused result from `search:hybrid`, ordered by RRF `score`
/// (highest first).
#[derive(Debug, Clone, Deserialize)]
pub struct HybridHit {
    pub node_id: String,
    pub node_type: String,
    /// Fused Reciprocal-Rank-Fusion score.
    pub score: f32,
    pub legs: HybridLegBreakdown,
}

/// Returned by `POST /v1/graphs/{id}/search:hybrid`.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchHybridResponse {
    pub hits: Vec<HybridHit>,
}

/// Numeric precision for `/v1/util/*` vector ops. Wire form is the
/// lowercase variant name (`"f32"` / `"f64"`). Duplicated rather
/// than re-exported from `dynograph-service` for the same reason the
/// other wire types are duplicated (preserves the lean HTTP-client
/// dependency tree — service would drag axum/tokio/rocksdb in). The
/// integration tests pin the two definitions against each other via
/// round-trip serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    F32,
    #[default]
    F64,
}

// =========================================================================
// algo/* (v0.7.0 service surface). Behind the server's `graph` build feature:
// without it every algo route returns 501 (surfaced as `ClientError::Http`).
// Requests are passed through untyped (the `scope`/`where`/`weight` grammar
// would otherwise be duplicated, as with batch/traverse); responses are typed.
// =========================================================================

/// One node's score. Shared by every score-producing algorithm (degree,
/// pagerank, eigenvector, closeness, betweenness, personalized_pagerank).
#[derive(Debug, Clone, Deserialize)]
pub struct NodeScore {
    pub node: String,
    pub score: f64,
}

/// Per-node centrality scores, highest first. Returned by `algo/degree`,
/// `pagerank`, `eigenvector`, `closeness`, `betweenness`,
/// `personalized_pagerank`.
#[derive(Debug, Clone, Deserialize)]
pub struct ScoresResponse {
    pub scores: Vec<NodeScore>,
}

/// Returned by `algo/components` (weakly-connected) and `algo/scc` (strongly-
/// connected): the partition as lists of node ids.
#[derive(Debug, Clone, Deserialize)]
pub struct ComponentsResponse {
    pub count: usize,
    pub components: Vec<Vec<String>>,
}

/// One bridge edge (unordered pair, `a < b`) from `algo/cuts`.
#[derive(Debug, Clone, Deserialize)]
pub struct CutEdge {
    pub a: String,
    pub b: String,
}

/// Returned by `algo/cuts`: articulation points and bridges.
#[derive(Debug, Clone, Deserialize)]
pub struct CutsResponse {
    pub articulation_points: Vec<String>,
    pub bridges: Vec<CutEdge>,
}

/// Returned by `algo/cycles`: `acyclic` for a DAG, else a witness `cycle`.
#[derive(Debug, Clone, Deserialize)]
pub struct CyclesResponse {
    pub acyclic: bool,
    pub cycle: Vec<String>,
}

/// Returned by `algo/toposort`: `acyclic` with a topological `order`, else
/// `order` empty.
#[derive(Debug, Clone, Deserialize)]
pub struct ToposortResponse {
    pub acyclic: bool,
    pub order: Vec<String>,
}

/// Returned by `algo/shortest_path`. `found` is false (empty `path`) when the
/// target is unreachable.
#[derive(Debug, Clone, Deserialize)]
pub struct ShortestPathResponse {
    pub found: bool,
    pub path: Vec<String>,
    pub distance: f64,
}

/// One predicted (currently-absent) link from `algo/link_prediction`.
#[derive(Debug, Clone, Deserialize)]
pub struct PredictedLink {
    pub a: String,
    pub b: String,
    pub score: f64,
}

/// Returned by `algo/link_prediction`, highest score first.
#[derive(Debug, Clone, Deserialize)]
pub struct LinkPredictionResponse {
    pub links: Vec<PredictedLink>,
}

/// Returned by `algo/clustering`: per-node local clustering plus the two global
/// summaries.
#[derive(Debug, Clone, Deserialize)]
pub struct ClusteringResponse {
    pub scores: Vec<NodeScore>,
    pub transitivity: f64,
    pub average_clustering: f64,
}

/// One min-cut edge from `algo/max_flow`.
#[derive(Debug, Clone, Deserialize)]
pub struct FlowEdge {
    pub from: String,
    pub to: String,
}

/// Returned by `algo/max_flow`: the max-flow value, the source-side partition,
/// and the crossing cut edges.
#[derive(Debug, Clone, Deserialize)]
pub struct MaxFlowResponse {
    pub max_flow: f64,
    pub source_side: Vec<String>,
    pub cut_edges: Vec<FlowEdge>,
}

/// Returned by `algo/communities` (Louvain): the partition plus its modularity.
#[derive(Debug, Clone, Deserialize)]
pub struct CommunitiesResponse {
    pub count: usize,
    pub communities: Vec<Vec<String>>,
    pub modularity: f64,
}

/// One op's outcome in a `/batch` `dry_run` report.
#[derive(Debug, Clone, Deserialize)]
pub struct BatchOpResult {
    pub index: usize,
    pub op: String,
    pub ok: bool,
    #[serde(default)]
    pub error: Option<String>,
}

/// Returned by `POST /v1/graphs/{id}/batch` with `dry_run: true`: a per-op
/// pass/fail report (covering ops up to and including the first failure) and an
/// overall `valid` flag. The graph is never mutated.
#[derive(Debug, Clone, Deserialize)]
pub struct BatchValidation {
    pub valid: bool,
    pub results: Vec<BatchOpResult>,
}
