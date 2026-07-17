//! Wire request/response types for the `algo/*` endpoints.
//!
//! These are **always compiled** (independent of the `graph` feature) because
//! they *are* the published OpenAPI contract — the routes and their schemas
//! exist whether or not the algorithm implementations are built in.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::nodes_scan::WhereClause;

/// Subgraph selection shared by every algorithm. Omitted fields mean "all":
/// no `node_types` => every node type in the schema; no `edge_types` => every
/// edge type; an empty `where` => no property filter.
#[derive(Debug, Deserialize, ToSchema)]
#[cfg_attr(not(feature = "graph"), allow(dead_code))]
pub(crate) struct AlgoScope {
    /// Node types to include. Omit for all types; an explicit empty list is a
    /// 400. Note this interacts with `where`: omitting it means a `where`
    /// clause's property must be indexed on *every* node type in the schema, so
    /// to predicate on a property only some types carry, name those types here.
    #[serde(default)]
    pub node_types: Option<Vec<String>>,
    /// Edge types to include. Omit for all types.
    #[serde(default)]
    pub edge_types: Option<Vec<String>>,
    /// Optional property predicate (same clause grammar as `nodes:scan` /
    /// `search:hybrid`): only nodes matching **all** clauses enter the
    /// projected subgraph, and only edges between two surviving nodes are
    /// kept. Lets one logical subgraph partitioned by a node property (not a
    /// type) be analyzed in isolation. Each clause's property must be declared
    /// and `indexed` on **every** in-scope node type, else 400 — narrow
    /// `node_types` to disambiguate.
    #[serde(default, rename = "where")]
    pub where_clauses: Vec<WhereClause>,
}

/// Edge-weight projection. With neither field set, edges are unweighted (every
/// weight is `1.0`). `property` takes precedence over `edge_type_weights` if
/// both are given.
#[derive(Debug, Deserialize, ToSchema)]
#[cfg_attr(not(feature = "graph"), allow(dead_code))]
pub(crate) struct WeightSpec {
    /// Read this numeric edge property as the weight. Missing or non-numeric on
    /// any in-scope edge is a 400 (fail loud).
    #[serde(default)]
    pub property: Option<String>,
    /// Per-edge-type constant weights. An in-scope edge whose type is absent
    /// from this map is a 400.
    #[serde(default)]
    pub edge_type_weights: Option<HashMap<String, f64>>,
}

/// Whether edges are treated as one-way or symmetric.
#[derive(Debug, Deserialize, Default, Clone, Copy, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AlgoDirection {
    /// Directed: `out`/`in` adjacency are distinct.
    #[default]
    Directed,
    /// Undirected: each edge counts in both directions.
    Undirected,
}

/// Which incident edges degree centrality counts. For an undirected graph all
/// three read the same symmetric adjacency.
#[derive(Debug, Deserialize, Default, Clone, Copy, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DegreeModeWire {
    In,
    Out,
    #[default]
    Total,
}

/// A request that carries only a subgraph `scope` — shared by the purely
/// structural algorithms (`/algo/components`, `/algo/cuts`, `/algo/scc`), which
/// take no weights or direction.
#[derive(Debug, Deserialize, ToSchema)]
#[cfg_attr(not(feature = "graph"), allow(dead_code))]
pub(crate) struct ScopedRequest {
    #[serde(default)]
    pub scope: Option<AlgoScope>,
}

/// Request for `POST /v1/graphs/{id}/algo/degree`.
#[derive(Debug, Deserialize, ToSchema)]
#[cfg_attr(not(feature = "graph"), allow(dead_code))]
pub(crate) struct DegreeRequest {
    #[serde(default)]
    pub scope: Option<AlgoScope>,
    /// Edge-weight projection. When present, scores are weighted degree
    /// (strength); when absent, scores are plain incident-edge counts.
    #[serde(default)]
    pub weight: Option<WeightSpec>,
    #[serde(default)]
    pub direction: AlgoDirection,
    #[serde(default)]
    pub mode: DegreeModeWire,
}

/// Request for `POST /v1/graphs/{id}/algo/pagerank`. Edge weights are
/// **strengths** (a node splits its rank among out-edges in proportion to
/// weight); omit `weight` for an unweighted (equal-split) run.
#[derive(Debug, Deserialize, ToSchema)]
#[cfg_attr(not(feature = "graph"), allow(dead_code))]
pub(crate) struct PageRankRequest {
    #[serde(default)]
    pub scope: Option<AlgoScope>,
    #[serde(default)]
    pub weight: Option<WeightSpec>,
    #[serde(default)]
    pub direction: AlgoDirection,
    /// Damping factor in `[0, 1]`. Defaults to 0.85.
    #[serde(default)]
    pub damping: Option<f64>,
    /// L1 convergence threshold. Defaults to 1e-6.
    #[serde(default)]
    pub tolerance: Option<f64>,
    /// Iteration budget before a non-convergence error. Defaults to 100.
    #[serde(default)]
    pub max_iterations: Option<usize>,
}

/// Request for `POST /v1/graphs/{id}/algo/eigenvector`. Edge weights are
/// **strengths**; omit `weight` for an unweighted run.
///
/// Eigenvector centrality is computed on the **undirected** graph (it is only
/// well-defined there). `direction` therefore accepts only `undirected` (the
/// default); `directed` is rejected with a 400 pointing to `/algo/pagerank`,
/// which is the directed importance measure.
#[derive(Debug, Deserialize, ToSchema)]
#[cfg_attr(not(feature = "graph"), allow(dead_code))]
pub(crate) struct EigenvectorRequest {
    #[serde(default)]
    pub scope: Option<AlgoScope>,
    #[serde(default)]
    pub weight: Option<WeightSpec>,
    /// Only `undirected` is accepted (the default). `directed` => 400.
    #[serde(default)]
    pub direction: Option<AlgoDirection>,
    /// L1 convergence threshold. Defaults to 1e-6.
    #[serde(default)]
    pub tolerance: Option<f64>,
    /// Iteration budget before a non-convergence error. Defaults to 100.
    #[serde(default)]
    pub max_iterations: Option<usize>,
}

/// Request for `POST /v1/graphs/{id}/algo/communities`. Edge weights are
/// **strengths** (tighter tie); omit `weight` for an unweighted run.
///
/// Community detection (Leiden) is defined on the **undirected** graph, so
/// `direction` accepts only `undirected` (the default); `directed` is rejected
/// with a 400, the same posture as `/algo/eigenvector`.
#[derive(Debug, Deserialize, ToSchema)]
#[cfg_attr(not(feature = "graph"), allow(dead_code))]
pub(crate) struct CommunitiesRequest {
    #[serde(default)]
    pub scope: Option<AlgoScope>,
    #[serde(default)]
    pub weight: Option<WeightSpec>,
    /// Only `undirected` is accepted (the default). `directed` => 400.
    #[serde(default)]
    pub direction: Option<AlgoDirection>,
    /// Resolution γ (strictly positive, finite). Higher => more, smaller
    /// communities; lower => fewer, larger. Defaults to 1.0 (classic modularity).
    #[serde(default)]
    pub resolution: Option<f64>,
}

/// Request for `POST /v1/graphs/{id}/algo/closeness`. Edge weights are path
/// **costs** (higher = farther) and must be strictly positive; omit `weight`
/// for unit-cost (hop-count) distances. On a `directed` graph this is *outward*
/// closeness (distances from each node to the rest, following edge direction).
#[derive(Debug, Deserialize, ToSchema)]
#[cfg_attr(not(feature = "graph"), allow(dead_code))]
pub(crate) struct ClosenessRequest {
    #[serde(default)]
    pub scope: Option<AlgoScope>,
    #[serde(default)]
    pub weight: Option<WeightSpec>,
    #[serde(default)]
    pub direction: AlgoDirection,
}

/// Request for `POST /v1/graphs/{id}/algo/betweenness`. Edge weights are path
/// **costs** and must be strictly positive; omit `weight` for unit-cost
/// distances.
#[derive(Debug, Deserialize, ToSchema)]
#[cfg_attr(not(feature = "graph"), allow(dead_code))]
pub(crate) struct BetweennessRequest {
    #[serde(default)]
    pub scope: Option<AlgoScope>,
    #[serde(default)]
    pub weight: Option<WeightSpec>,
    #[serde(default)]
    pub direction: AlgoDirection,
    /// Normalize scores by the number of node pairs. Defaults to true.
    #[serde(default)]
    pub normalized: Option<bool>,
}

/// One bridge (cut edge), as the unordered pair of node ids it connects
/// (`a < b`).
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct CutEdge {
    pub a: String,
    pub b: String,
}

/// Articulation points and bridges of the (undirected) subgraph.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct CutsResponse {
    /// Node ids whose removal increases the number of connected components.
    pub articulation_points: Vec<String>,
    /// Edges whose removal increases the number of connected components.
    pub bridges: Vec<CutEdge>,
}

/// One (weakly-)connected component: the node ids it contains. Also the response
/// shape for strongly-connected components (`/algo/scc`).
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ComponentsResponse {
    /// Number of components.
    pub count: usize,
    /// Components as lists of node ids; each list sorted, and the lists ordered
    /// deterministically by their smallest id.
    pub components: Vec<Vec<String>>,
}

/// Response for `POST /v1/graphs/{id}/algo/communities`. The Leiden partition
/// plus its modularity under the requested resolution.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct CommunitiesResponse {
    /// Number of communities.
    pub count: usize,
    /// Communities as lists of node ids; each list sorted, and the lists
    /// ordered deterministically by their smallest id (as `/algo/components`).
    pub communities: Vec<Vec<String>>,
    /// Modularity of the partition (0.0 for an edgeless or empty subgraph).
    pub modularity: f64,
}

/// A single node's score.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct NodeScore {
    pub node: String,
    pub score: f64,
}

/// Per-node centrality scores, highest first (ties broken by node id). Shared by
/// every score-producing algorithm (degree, PageRank, eigenvector, closeness,
/// betweenness, personalized PageRank).
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ScoresResponse {
    pub scores: Vec<NodeScore>,
}

/// Request for `POST /v1/graphs/{id}/algo/personalized_pagerank` (random walk
/// with restart). Like PageRank, but teleport mass returns to `seeds`, so scores
/// measure relevance to that seed set. Edge weights are **strengths**.
#[derive(Debug, Deserialize, ToSchema)]
#[cfg_attr(not(feature = "graph"), allow(dead_code))]
pub(crate) struct PersonalizedPageRankRequest {
    #[serde(default)]
    pub scope: Option<AlgoScope>,
    #[serde(default)]
    pub weight: Option<WeightSpec>,
    #[serde(default)]
    pub direction: AlgoDirection,
    /// Seed node ids the walk restarts to. Required (non-empty).
    #[serde(default)]
    pub seeds: Vec<String>,
    /// Damping factor in `[0, 1]`. Defaults to 0.85.
    #[serde(default)]
    pub damping: Option<f64>,
    /// L1 convergence threshold. Defaults to 1e-6.
    #[serde(default)]
    pub tolerance: Option<f64>,
    /// Iteration budget before a non-convergence error. Defaults to 100.
    #[serde(default)]
    pub max_iterations: Option<usize>,
}

/// A request carrying a `source`/`target` node pair plus the usual scope, weight,
/// and direction — shared by `/algo/shortest_path` (weight = path cost) and
/// `/algo/max_flow` (weight = capacity). The two `source`/`target` ids are
/// required; the per-endpoint weight semantics are documented on each route.
#[derive(Debug, Deserialize, ToSchema)]
#[cfg_attr(not(feature = "graph"), allow(dead_code))]
pub(crate) struct SourceTargetRequest {
    #[serde(default)]
    pub scope: Option<AlgoScope>,
    /// Start node id (required).
    #[serde(default)]
    pub source: Option<String>,
    /// End node id (required).
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub weight: Option<WeightSpec>,
    #[serde(default)]
    pub direction: AlgoDirection,
}

/// A shortest path between two nodes. `found` is false (with an empty `path`)
/// when the target is unreachable from the source.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ShortestPathResponse {
    pub found: bool,
    /// Node ids in order, source first and target last; empty when not found.
    pub path: Vec<String>,
    /// Total path cost (hop count when unweighted); 0 when not found.
    pub distance: f64,
}

/// Which neighborhood-overlap score link prediction uses.
#[derive(Debug, Deserialize, Default, Clone, Copy, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LinkPredictionMethodWire {
    #[default]
    CommonNeighbors,
    Jaccard,
    AdamicAdar,
}

/// Request for `POST /v1/graphs/{id}/algo/link_prediction`. Treats the subgraph
/// as undirected. With `source`, predicts links from that node; without it,
/// across all non-adjacent pairs. Results are ranked by score and capped.
#[derive(Debug, Deserialize, ToSchema)]
#[cfg_attr(not(feature = "graph"), allow(dead_code))]
pub(crate) struct LinkPredictionRequest {
    #[serde(default)]
    pub scope: Option<AlgoScope>,
    #[serde(default)]
    pub method: LinkPredictionMethodWire,
    /// Predict links from this node only; omit for all non-adjacent pairs.
    #[serde(default)]
    pub source: Option<String>,
    /// Max ranked results. Defaults to 100, capped at the standard result limit.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// One predicted (currently-absent) link and its score. `a`/`b` are the endpoint
/// node ids (for a single-source request, `a` is the source).
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct PredictedLink {
    pub a: String,
    pub b: String,
    pub score: f64,
}

/// Predicted links, highest score first.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct LinkPredictionResponse {
    pub links: Vec<PredictedLink>,
}

/// Response for `POST /v1/graphs/{id}/algo/cycles`. `acyclic` is true for a DAG;
/// otherwise `cycle` is one witness directed cycle (node ids, closed by the edge
/// from the last back to the first).
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct CyclesResponse {
    pub acyclic: bool,
    pub cycle: Vec<String>,
}

/// Response for `POST /v1/graphs/{id}/algo/clustering` (undirected). Per-node
/// local clustering scores plus the two global summaries.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ClusteringResponse {
    /// Local clustering coefficient per node, highest first.
    pub scores: Vec<NodeScore>,
    /// Global transitivity: 3·triangles / connected-triples.
    pub transitivity: f64,
    /// Mean local clustering coefficient over all nodes.
    pub average_clustering: f64,
}

/// Response for `POST /v1/graphs/{id}/algo/toposort` (directed). `acyclic` is
/// true for a DAG, with `order` a topological ordering; otherwise `order` is
/// empty (the graph has a cycle).
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ToposortResponse {
    pub acyclic: bool,
    pub order: Vec<String>,
}

/// One min-cut edge crossing from the source side to the sink side. For an
/// undirected graph `from` is always the source-side endpoint (there is no
/// stored edge direction).
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct FlowEdge {
    pub from: String,
    pub to: String,
}

/// Response for `POST /v1/graphs/{id}/algo/max_flow`. The max flow value equals
/// the min-cut capacity; `source_side` is the node partition on the source side
/// of that cut, and `cut_edges` the crossing edges.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct MaxFlowResponse {
    pub max_flow: f64,
    pub source_side: Vec<String>,
    pub cut_edges: Vec<FlowEdge>,
}
