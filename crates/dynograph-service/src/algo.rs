//! `POST /v1/graphs/{id}/algo/*` — classic graph-theory algorithms over the
//! generic node/edge graph, backed by the `dynograph-graph` crate.
//!
//! ## Feature gating
//!
//! The algorithms live behind the optional `graph` build feature (mirroring
//! `fulltext`): the routes and their OpenAPI contract always exist, but without
//! the feature the handlers return `501 Not Implemented`. This keeps the
//! published wire contract feature-independent while letting consumers that
//! don't need topology algorithms drop the dependency.
//!
//! ## Domain neutrality
//!
//! Algorithms run on the generic graph only. The caller supplies what's
//! domain-specific via the request: a **scope** (which node/edge types form the
//! subgraph), an **edge-weight projection** (read a numeric edge property, or a
//! per-edge-type constant; default unweighted), and a **direction** flag. No
//! domain vocabulary lives here.
//!
//! ## How the in-memory graph is built (the storage ↔ algorithm seam)
//!
//! `dynograph-graph` is pure and storage-agnostic. The service is responsible
//! for reading storage and handing it a finished `Graph`:
//! 1. Resolve the in-scope node types (request `scope.node_types`, else every
//!    type in the schema); `scan_nodes` each and intern its ids.
//! 2. Per node, `scan_outgoing_edges` and keep edges whose type is in scope and
//!    whose target is an in-scope node (edges leaving the subgraph are dropped —
//!    that's the defined scope boundary, not a silent failure).
//! 3. Project each kept edge to a finite `f64` weight, **failing loud** (400) on
//!    a missing/non-numeric weight property rather than defaulting silently.
//!
//! Node identity in the in-memory graph is the bare `node_id`, and edges (which
//! store only ids, not endpoint types) are matched to nodes by that bare id. If
//! the same id appears under two different node types **both in scope**, the
//! build fails loud (400) rather than conflate two distinct nodes — narrow
//! `scope.node_types` to disambiguate. The guard only sees in-scope types, so an
//! id reused across an in-scope and an out-of-scope type can't be detected;
//! callers that rely on id uniqueness should keep ids globally unique per graph.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::nodes_scan::WhereClause;

// ---- Wire types (always compiled — they ARE the OpenAPI contract) ----

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
/// Community detection (Louvain) is defined on the **undirected** graph, so
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

/// Response for `POST /v1/graphs/{id}/algo/communities`. The Louvain partition
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

// ---- Algorithm entry points ----
//
// Every `run_*` shares one signature `(&StorageEngine, &str, Req) ->
// Result<Resp, RegistryError>` regardless of the feature, so the app-layer
// handlers are uniform (no `cfg` in app.rs). With the `graph` feature they run
// the algorithm; without it they return 501. A new algo endpoint adds a `run_*`
// in `imp`, a no-feature stub line below, and a thin handler — no per-endpoint
// feature plumbing.

#[cfg(feature = "graph")]
pub(crate) use imp::{
    run_betweenness, run_closeness, run_clustering, run_communities, run_components, run_cuts,
    run_cycles, run_degree, run_eigenvector, run_link_prediction, run_max_flow, run_pagerank,
    run_personalized_pagerank, run_scc, run_shortest_path, run_toposort,
};

#[cfg(not(feature = "graph"))]
fn not_enabled() -> crate::registry::RegistryError {
    crate::registry::RegistryError::NotImplemented(
        "graph algorithms are not enabled in this build (compile with --features graph)"
            .to_string(),
    )
}

/// Declares a no-feature `run_*` stub returning 501, matching the real
/// signature so the handlers stay feature-agnostic.
#[cfg(not(feature = "graph"))]
macro_rules! not_enabled_stub {
    ($name:ident, $req:ty, $resp:ty) => {
        pub(crate) fn $name(
            _engine: &dynograph_storage::StorageEngine,
            _graph_id: &str,
            _req: $req,
        ) -> Result<$resp, crate::registry::RegistryError> {
            Err(not_enabled())
        }
    };
}

#[cfg(not(feature = "graph"))]
not_enabled_stub!(run_components, ScopedRequest, ComponentsResponse);
#[cfg(not(feature = "graph"))]
not_enabled_stub!(run_degree, DegreeRequest, ScoresResponse);
#[cfg(not(feature = "graph"))]
not_enabled_stub!(run_pagerank, PageRankRequest, ScoresResponse);
#[cfg(not(feature = "graph"))]
not_enabled_stub!(run_eigenvector, EigenvectorRequest, ScoresResponse);
#[cfg(not(feature = "graph"))]
not_enabled_stub!(run_closeness, ClosenessRequest, ScoresResponse);
#[cfg(not(feature = "graph"))]
not_enabled_stub!(run_betweenness, BetweennessRequest, ScoresResponse);
#[cfg(not(feature = "graph"))]
not_enabled_stub!(run_cuts, ScopedRequest, CutsResponse);
#[cfg(not(feature = "graph"))]
not_enabled_stub!(run_scc, ScopedRequest, ComponentsResponse);
#[cfg(not(feature = "graph"))]
not_enabled_stub!(run_cycles, ScopedRequest, CyclesResponse);
#[cfg(not(feature = "graph"))]
not_enabled_stub!(
    run_personalized_pagerank,
    PersonalizedPageRankRequest,
    ScoresResponse
);
#[cfg(not(feature = "graph"))]
not_enabled_stub!(run_shortest_path, SourceTargetRequest, ShortestPathResponse);
#[cfg(not(feature = "graph"))]
not_enabled_stub!(
    run_link_prediction,
    LinkPredictionRequest,
    LinkPredictionResponse
);
#[cfg(not(feature = "graph"))]
not_enabled_stub!(run_clustering, ScopedRequest, ClusteringResponse);
#[cfg(not(feature = "graph"))]
not_enabled_stub!(run_communities, CommunitiesRequest, CommunitiesResponse);
#[cfg(not(feature = "graph"))]
not_enabled_stub!(run_toposort, ScopedRequest, ToposortResponse);
#[cfg(not(feature = "graph"))]
not_enabled_stub!(run_max_flow, SourceTargetRequest, MaxFlowResponse);

#[cfg(feature = "graph")]
mod imp {
    use super::*;

    use dynograph_graph::{
        Communities, Components, DegreeMode, EigenvectorConfig, Graph, GraphBuilder, GraphError,
        LinkPredictionMethod, PageRankConfig, betweenness_centrality, closeness_centrality,
        clustering, connected_components, cut_structure, degree_centrality, eigenvector_centrality,
        find_cycle, link_prediction_all, link_prediction_from, louvain, max_flow_min_cut, pagerank,
        personalized_pagerank, shortest_path, strongly_connected_components, topological_sort,
    };
    use dynograph_storage::{StorageEngine, StoredEdge};

    use crate::registry::RegistryError;
    use crate::validation::validate_limit;

    /// Default cap on link-prediction results when the request omits `limit`.
    const DEFAULT_LINK_LIMIT: usize = 100;

    impl From<LinkPredictionMethodWire> for LinkPredictionMethod {
        fn from(w: LinkPredictionMethodWire) -> Self {
            match w {
                LinkPredictionMethodWire::CommonNeighbors => LinkPredictionMethod::CommonNeighbors,
                LinkPredictionMethodWire::Jaccard => LinkPredictionMethod::Jaccard,
                LinkPredictionMethodWire::AdamicAdar => LinkPredictionMethod::AdamicAdar,
            }
        }
    }

    /// Safety cap on subgraph size. Consumer graphs are small (10^2-10^3 nodes);
    /// a scope that pulls far more is almost certainly a wrongly-scoped request, so
    /// fail loud rather than risk an OOM building the in-memory graph.
    const MAX_ALGO_NODES: usize = 100_000;
    /// Companion cap on edge count: a subgraph can stay under the node cap yet be
    /// dense (a hub/near-complete graph), so the node cap alone doesn't bound the
    /// in-memory adjacency. Cap edges too, for the same fail-loud-not-OOM reason.
    const MAX_ALGO_EDGES: usize = 2_000_000;

    impl From<DegreeModeWire> for DegreeMode {
        fn from(w: DegreeModeWire) -> Self {
            match w {
                DegreeModeWire::In => DegreeMode::In,
                DegreeModeWire::Out => DegreeMode::Out,
                DegreeModeWire::Total => DegreeMode::Total,
            }
        }
    }

    /// `algo/components` — (weakly-)connected components over the scoped graph.
    /// Direction-independent, so always built directed (cheapest) and walked
    /// over both adjacency directions by `connected_components`.
    pub(crate) fn run_components(
        engine: &StorageEngine,
        graph_id: &str,
        req: ScopedRequest,
    ) -> Result<ComponentsResponse, RegistryError> {
        let graph = build_graph(engine, graph_id, req.scope.as_ref(), None, true)?;
        Ok(components_response(&graph, connected_components(&graph)))
    }

    /// `algo/scc` — strongly-connected components over the scoped directed graph.
    pub(crate) fn run_scc(
        engine: &StorageEngine,
        graph_id: &str,
        req: ScopedRequest,
    ) -> Result<ComponentsResponse, RegistryError> {
        let graph = build_graph(engine, graph_id, req.scope.as_ref(), None, true)?;
        Ok(components_response(
            &graph,
            strongly_connected_components(&graph),
        ))
    }

    /// `algo/cuts` — articulation points and bridges over the scoped undirected
    /// graph. Purely structural: built undirected, weights ignored.
    pub(crate) fn run_cuts(
        engine: &StorageEngine,
        graph_id: &str,
        req: ScopedRequest,
    ) -> Result<CutsResponse, RegistryError> {
        let graph = build_graph(engine, graph_id, req.scope.as_ref(), None, false)?;
        let cuts = cut_structure(&graph);
        Ok(CutsResponse {
            articulation_points: cuts
                .articulation_points
                .into_iter()
                .map(|idx| graph.id_of(idx).to_string())
                .collect(),
            bridges: cuts
                .bridges
                .into_iter()
                .map(|(a, b)| CutEdge {
                    a: graph.id_of(a).to_string(),
                    b: graph.id_of(b).to_string(),
                })
                .collect(),
        })
    }

    /// `algo/personalized_pagerank` — PageRank with restart to `seeds`.
    pub(crate) fn run_personalized_pagerank(
        engine: &StorageEngine,
        graph_id: &str,
        req: PersonalizedPageRankRequest,
    ) -> Result<ScoresResponse, RegistryError> {
        if req.seeds.is_empty() {
            return Err(RegistryError::BadRequest(
                "personalized_pagerank requires at least one seed".to_string(),
            ));
        }
        let config = pagerank_config_from(req.damping, req.tolerance, req.max_iterations)?;
        let directed = req.direction == AlgoDirection::Directed;
        let graph = build_graph(
            engine,
            graph_id,
            req.scope.as_ref(),
            req.weight.as_ref(),
            directed,
        )?;
        let mut seeds = Vec::with_capacity(req.seeds.len());
        for id in &req.seeds {
            seeds.push(graph.idx_of(id).ok_or_else(|| {
                RegistryError::BadRequest(format!("seed node {id:?} is not in the scoped graph"))
            })?);
        }
        let raw = personalized_pagerank(&graph, &seeds, &config).map_err(map_graph_err)?;
        Ok(ScoresResponse {
            scores: sorted_scores(&graph, raw),
        })
    }

    /// `algo/shortest_path` — one shortest route between two nodes.
    pub(crate) fn run_shortest_path(
        engine: &StorageEngine,
        graph_id: &str,
        req: SourceTargetRequest,
    ) -> Result<ShortestPathResponse, RegistryError> {
        let source = required_node(&req.source, "source")?;
        let target = required_node(&req.target, "target")?;
        let directed = req.direction == AlgoDirection::Directed;
        let weighted = req.weight.is_some();
        let graph = build_graph(
            engine,
            graph_id,
            req.scope.as_ref(),
            req.weight.as_ref(),
            directed,
        )?;
        let s = node_index(&graph, source, "source")?;
        let t = node_index(&graph, target, "target")?;
        match shortest_path(&graph, s, t, weighted).map_err(map_graph_err)? {
            Some(path) => Ok(ShortestPathResponse {
                found: true,
                path: path
                    .nodes
                    .into_iter()
                    .map(|idx| graph.id_of(idx).to_string())
                    .collect(),
                distance: path.distance,
            }),
            None => Ok(ShortestPathResponse {
                found: false,
                path: Vec::new(),
                distance: 0.0,
            }),
        }
    }

    /// `algo/link_prediction` — score candidate (absent) edges by neighborhood
    /// overlap over the undirected subgraph.
    pub(crate) fn run_link_prediction(
        engine: &StorageEngine,
        graph_id: &str,
        req: LinkPredictionRequest,
    ) -> Result<LinkPredictionResponse, RegistryError> {
        let limit = req.limit.unwrap_or(DEFAULT_LINK_LIMIT);
        validate_limit(limit, "limit")?;
        let method = req.method.into();
        // Link prediction is an undirected, structural measure (weights ignored).
        let graph = build_graph(engine, graph_id, req.scope.as_ref(), None, false)?;

        let mut links: Vec<PredictedLink> = match &req.source {
            Some(src) => {
                let s = node_index(&graph, src, "source")?;
                link_prediction_from(&graph, s, method)
                    .into_iter()
                    .map(|(t, score)| PredictedLink {
                        a: src.clone(),
                        b: graph.id_of(t).to_string(),
                        score,
                    })
                    .collect()
            }
            None => link_prediction_all(&graph, method)
                .into_iter()
                .map(|(u, v, score)| PredictedLink {
                    a: graph.id_of(u).to_string(),
                    b: graph.id_of(v).to_string(),
                    score,
                })
                .collect(),
        };
        // Highest score first; deterministic tie-break by endpoint ids.
        links.sort_by(|x, y| {
            y.score
                .partial_cmp(&x.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| (&x.a, &x.b).cmp(&(&y.a, &y.b)))
        });
        links.truncate(limit);
        Ok(LinkPredictionResponse { links })
    }

    /// `algo/cycles` — is the directed subgraph acyclic, with a witness cycle.
    pub(crate) fn run_cycles(
        engine: &StorageEngine,
        graph_id: &str,
        req: ScopedRequest,
    ) -> Result<CyclesResponse, RegistryError> {
        let graph = build_graph(engine, graph_id, req.scope.as_ref(), None, true)?;
        match find_cycle(&graph) {
            Some(cycle) => Ok(CyclesResponse {
                acyclic: false,
                cycle: cycle
                    .into_iter()
                    .map(|idx| graph.id_of(idx).to_string())
                    .collect(),
            }),
            None => Ok(CyclesResponse {
                acyclic: true,
                cycle: Vec::new(),
            }),
        }
    }

    /// `algo/clustering` — local clustering + global transitivity (undirected).
    pub(crate) fn run_clustering(
        engine: &StorageEngine,
        graph_id: &str,
        req: ScopedRequest,
    ) -> Result<ClusteringResponse, RegistryError> {
        let graph = build_graph(engine, graph_id, req.scope.as_ref(), None, false)?;
        let c = clustering(&graph);
        Ok(ClusteringResponse {
            scores: sorted_scores(&graph, c.local),
            transitivity: c.transitivity,
            average_clustering: c.average,
        })
    }

    /// `algo/communities` — Louvain community detection over the scoped
    /// undirected graph (weights = strengths).
    pub(crate) fn run_communities(
        engine: &StorageEngine,
        graph_id: &str,
        req: CommunitiesRequest,
    ) -> Result<CommunitiesResponse, RegistryError> {
        // Modularity is defined on an undirected graph; reject an explicit
        // `directed` and always build undirected (mirrors `/algo/eigenvector`).
        if req.direction == Some(AlgoDirection::Directed) {
            return Err(RegistryError::BadRequest(
                "community detection is only defined for undirected graphs; pass \
                 direction=undirected (or omit it)"
                    .to_string(),
            ));
        }
        let resolution = match req.resolution {
            None => 1.0,
            Some(r) => validate_positive("resolution", r)?,
        };
        let graph = build_graph(
            engine,
            graph_id,
            req.scope.as_ref(),
            req.weight.as_ref(),
            false,
        )?;
        let result = louvain(&graph, resolution);
        Ok(communities_response(&graph, result))
    }

    /// Map a Louvain partition to the wire response. Invert the per-node labels
    /// into index groups, then share the sort/id-mapping with components.
    fn communities_response(graph: &Graph, c: Communities) -> CommunitiesResponse {
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); c.num_communities];
        for (idx, &label) in c.labels.iter().enumerate() {
            groups[label].push(idx);
        }
        CommunitiesResponse {
            count: c.num_communities,
            communities: id_groups_sorted(graph, groups),
            modularity: c.modularity,
        }
    }

    /// `algo/toposort` — topological order of the directed subgraph, or a
    /// not-acyclic flag if it has a cycle.
    pub(crate) fn run_toposort(
        engine: &StorageEngine,
        graph_id: &str,
        req: ScopedRequest,
    ) -> Result<ToposortResponse, RegistryError> {
        let graph = build_graph(engine, graph_id, req.scope.as_ref(), None, true)?;
        match topological_sort(&graph) {
            Some(order) => Ok(ToposortResponse {
                acyclic: true,
                order: order
                    .into_iter()
                    .map(|idx| graph.id_of(idx).to_string())
                    .collect(),
            }),
            None => Ok(ToposortResponse {
                acyclic: false,
                order: Vec::new(),
            }),
        }
    }

    /// `algo/max_flow` — max flow and min s-t cut (weights = capacities).
    pub(crate) fn run_max_flow(
        engine: &StorageEngine,
        graph_id: &str,
        req: SourceTargetRequest,
    ) -> Result<MaxFlowResponse, RegistryError> {
        let source = required_node(&req.source, "source")?;
        let target = required_node(&req.target, "target")?;
        let directed = req.direction == AlgoDirection::Directed;
        let graph = build_graph(
            engine,
            graph_id,
            req.scope.as_ref(),
            req.weight.as_ref(),
            directed,
        )?;
        let s = node_index(&graph, source, "source")?;
        let t = node_index(&graph, target, "target")?;
        let cut = max_flow_min_cut(&graph, s, t).map_err(map_graph_err)?;
        Ok(MaxFlowResponse {
            max_flow: cut.max_flow,
            source_side: cut
                .source_side
                .into_iter()
                .map(|idx| graph.id_of(idx).to_string())
                .collect(),
            cut_edges: cut
                .cut_edges
                .into_iter()
                .map(|(u, v)| FlowEdge {
                    from: graph.id_of(u).to_string(),
                    to: graph.id_of(v).to_string(),
                })
                .collect(),
        })
    }

    /// Unwrap a required node-id field, failing loud if absent or blank.
    fn required_node<'a>(field: &'a Option<String>, name: &str) -> Result<&'a str, RegistryError> {
        match field.as_deref() {
            Some(id) if !id.is_empty() => Ok(id),
            _ => Err(RegistryError::BadRequest(format!("{name} is required"))),
        }
    }

    /// Resolve a node id to its dense index in the scoped graph, failing loud if
    /// it isn't present.
    fn node_index(graph: &Graph, id: &str, name: &str) -> Result<usize, RegistryError> {
        graph.idx_of(id).ok_or_else(|| {
            RegistryError::BadRequest(format!("{name} node {id:?} is not in the scoped graph"))
        })
    }

    /// Map a [`Components`] partition to the wire response.
    fn components_response(graph: &Graph, comps: Components) -> ComponentsResponse {
        ComponentsResponse {
            count: comps.count,
            components: id_groups_sorted(graph, comps.groups()),
        }
    }

    /// Map index-keyed groups to wire form: each group a sorted list of node
    /// ids, the lists ordered by their smallest id — deterministic regardless
    /// of internal label assignment. Shared by `/algo/components`, `/algo/scc`,
    /// and `/algo/communities`.
    fn id_groups_sorted(graph: &Graph, groups: Vec<Vec<usize>>) -> Vec<Vec<String>> {
        let mut out: Vec<Vec<String>> = groups
            .into_iter()
            .map(|group| {
                let mut ids: Vec<String> = group
                    .into_iter()
                    .map(|idx| graph.id_of(idx).to_string())
                    .collect();
                ids.sort();
                ids
            })
            .collect();
        out.sort();
        out
    }

    /// `algo/degree` — degree centrality over the scoped graph.
    pub(crate) fn run_degree(
        engine: &StorageEngine,
        graph_id: &str,
        req: DegreeRequest,
    ) -> Result<ScoresResponse, RegistryError> {
        let directed = req.direction == AlgoDirection::Directed;
        let weighted = req.weight.is_some();
        let graph = build_graph(
            engine,
            graph_id,
            req.scope.as_ref(),
            req.weight.as_ref(),
            directed,
        )?;
        let raw = degree_centrality(&graph, req.mode.into(), weighted);
        Ok(ScoresResponse {
            scores: sorted_scores(&graph, raw),
        })
    }

    /// `algo/pagerank` — PageRank over the scoped graph (weights = strength).
    pub(crate) fn run_pagerank(
        engine: &StorageEngine,
        graph_id: &str,
        req: PageRankRequest,
    ) -> Result<ScoresResponse, RegistryError> {
        let config = pagerank_config_from(req.damping, req.tolerance, req.max_iterations)?;
        let directed = req.direction == AlgoDirection::Directed;
        let graph = build_graph(
            engine,
            graph_id,
            req.scope.as_ref(),
            req.weight.as_ref(),
            directed,
        )?;
        let raw = pagerank(&graph, &config).map_err(map_graph_err)?;
        Ok(ScoresResponse {
            scores: sorted_scores(&graph, raw),
        })
    }

    /// Build a [`PageRankConfig`] from optional request fields, validating
    /// damping/tolerance/max_iterations. Shared by `run_pagerank` and
    /// `run_personalized_pagerank`.
    fn pagerank_config_from(
        damping: Option<f64>,
        tolerance: Option<f64>,
        max_iterations: Option<usize>,
    ) -> Result<PageRankConfig, RegistryError> {
        let mut config = PageRankConfig::default();
        if let Some(d) = damping {
            if !(0.0..=1.0).contains(&d) {
                return Err(RegistryError::BadRequest(format!(
                    "damping must be in [0, 1], got {d}"
                )));
            }
            config.damping = d;
        }
        if let Some(t) = tolerance {
            config.tolerance = validate_tolerance(t)?;
        }
        if let Some(m) = max_iterations {
            config.max_iterations = validate_max_iterations(m)?;
        }
        Ok(config)
    }

    /// `algo/eigenvector` — eigenvector centrality (weights = strength).
    pub(crate) fn run_eigenvector(
        engine: &StorageEngine,
        graph_id: &str,
        req: EigenvectorRequest,
    ) -> Result<ScoresResponse, RegistryError> {
        let mut config = EigenvectorConfig::default();
        if let Some(t) = req.tolerance {
            config.tolerance = validate_tolerance(t)?;
        }
        if let Some(m) = req.max_iterations {
            config.max_iterations = validate_max_iterations(m)?;
        }
        // Eigenvector centrality is only sound on an undirected (symmetric)
        // graph; a directed graph can yield a power-iteration artifact. Reject
        // an explicit `directed` and always build undirected.
        if req.direction == Some(AlgoDirection::Directed) {
            return Err(RegistryError::BadRequest(
                "eigenvector centrality is only defined for undirected graphs; pass \
                 direction=undirected, or use /algo/pagerank for directed importance"
                    .to_string(),
            ));
        }
        let graph = build_graph(
            engine,
            graph_id,
            req.scope.as_ref(),
            req.weight.as_ref(),
            false,
        )?;
        let raw = eigenvector_centrality(&graph, &config).map_err(map_graph_err)?;
        Ok(ScoresResponse {
            scores: sorted_scores(&graph, raw),
        })
    }

    /// `algo/closeness` — closeness centrality (weights = path cost).
    pub(crate) fn run_closeness(
        engine: &StorageEngine,
        graph_id: &str,
        req: ClosenessRequest,
    ) -> Result<ScoresResponse, RegistryError> {
        let directed = req.direction == AlgoDirection::Directed;
        let weighted = req.weight.is_some();
        let graph = build_graph(
            engine,
            graph_id,
            req.scope.as_ref(),
            req.weight.as_ref(),
            directed,
        )?;
        let raw = closeness_centrality(&graph, weighted).map_err(map_graph_err)?;
        Ok(ScoresResponse {
            scores: sorted_scores(&graph, raw),
        })
    }

    /// `algo/betweenness` — betweenness centrality (weights = path cost).
    pub(crate) fn run_betweenness(
        engine: &StorageEngine,
        graph_id: &str,
        req: BetweennessRequest,
    ) -> Result<ScoresResponse, RegistryError> {
        let directed = req.direction == AlgoDirection::Directed;
        let weighted = req.weight.is_some();
        let normalized = req.normalized.unwrap_or(true);
        let graph = build_graph(
            engine,
            graph_id,
            req.scope.as_ref(),
            req.weight.as_ref(),
            directed,
        )?;
        let raw = betweenness_centrality(&graph, weighted, normalized).map_err(map_graph_err)?;
        Ok(ScoresResponse {
            scores: sorted_scores(&graph, raw),
        })
    }

    /// Map dense per-index scores to `NodeScore`s, highest score first with a
    /// deterministic tie-break by node id.
    fn sorted_scores(graph: &Graph, raw: Vec<f64>) -> Vec<NodeScore> {
        let mut scores: Vec<NodeScore> = raw
            .into_iter()
            .enumerate()
            .map(|(idx, score)| NodeScore {
                node: graph.id_of(idx).to_string(),
                score,
            })
            .collect();
        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.node.cmp(&b.node))
        });
        scores
    }

    /// Algorithm-domain errors (bad weights, non-convergence) all stem from the
    /// caller's scope/weights/config, so surface them as 400s with the crate's
    /// explanatory message.
    fn map_graph_err(e: GraphError) -> RegistryError {
        RegistryError::BadRequest(e.to_string())
    }

    /// Validate a strictly-positive finite parameter, naming it in the 400.
    /// Shared by `tolerance` (power-iteration configs) and `resolution`
    /// (communities).
    fn validate_positive(name: &str, x: f64) -> Result<f64, RegistryError> {
        if !x.is_finite() || x <= 0.0 {
            return Err(RegistryError::BadRequest(format!(
                "{name} must be a positive number, got {x}"
            )));
        }
        Ok(x)
    }

    fn validate_tolerance(t: f64) -> Result<f64, RegistryError> {
        validate_positive("tolerance", t)
    }

    /// Upper bound on `max_iterations`. The power-iteration loops are CPU-bound
    /// and run under the per-graph read lock on the blocking pool, where the
    /// HTTP timeout layer can't interrupt them — an unbounded budget lets one
    /// request pin a worker and starve writers. Cap it (same fail-loud posture
    /// as MAX_ALGO_NODES/EDGES); 10k iterations is far past any real convergence.
    const MAX_ITERATIONS: usize = 10_000;

    fn validate_max_iterations(m: usize) -> Result<usize, RegistryError> {
        if m == 0 || m > MAX_ITERATIONS {
            return Err(RegistryError::BadRequest(format!(
                "max_iterations must be in 1..={MAX_ITERATIONS}, got {m}"
            )));
        }
        Ok(m)
    }

    /// Build the in-memory graph from storage under the caller's read lock.
    fn build_graph(
        engine: &StorageEngine,
        graph_id: &str,
        scope: Option<&AlgoScope>,
        weight: Option<&WeightSpec>,
        directed: bool,
    ) -> Result<Graph, RegistryError> {
        // A weight projection that specifies neither source is a no-op that would
        // silently score every edge 1.0 — i.e. return counts when the caller
        // asked for strengths. Reject it rather than mislabel the result.
        if let Some(w) = weight
            && w.property.is_none()
            && w.edge_type_weights.is_none()
        {
            return Err(RegistryError::BadRequest(
                "weight requires either 'property' or 'edge_type_weights'".to_string(),
            ));
        }

        let schema = engine.schema();

        // Resolve in-scope node types (validate any explicitly requested). An
        // explicitly empty list is rejected rather than silently treated as the
        // empty graph: it would scope every algorithm to zero nodes (an empty
        // result that reads like a real answer) and would also skip the `where`
        // validation below, which runs per in-scope type. Omit the field for
        // "all types".
        let node_types: Vec<String> = match scope.and_then(|s| s.node_types.as_ref()) {
            Some(types) => {
                if types.is_empty() {
                    return Err(RegistryError::BadRequest(
                        "scope.node_types must be non-empty (omit it to scope to all node types)"
                            .to_string(),
                    ));
                }
                for t in types {
                    if !schema.node_types.contains_key(t) {
                        return Err(RegistryError::BadRequest(format!(
                            "scope.node_types references unknown node type: {t}"
                        )));
                    }
                }
                types.clone()
            }
            None => schema.node_types.keys().cloned().collect(),
        };

        // Resolve in-scope edge types into a membership set.
        let edge_types: std::collections::HashSet<String> =
            match scope.and_then(|s| s.edge_types.as_ref()) {
                Some(types) => {
                    for t in types {
                        if !schema.edge_types.contains_key(t) {
                            return Err(RegistryError::BadRequest(format!(
                                "scope.edge_types references unknown edge type: {t}"
                            )));
                        }
                    }
                    types.iter().cloned().collect()
                }
                None => schema.edge_types.keys().cloned().collect(),
            };

        // Optional property predicate (`scope.where`): only nodes matching all
        // clauses enter the projected subgraph. Validate every clause against
        // every in-scope node type up front (indexed-property + value-shape,
        // all 400) — a clause whose property isn't indexed on some in-scope
        // type fails loud, pushing the caller to narrow scope.node_types,
        // matching the id-conflation guard's posture. An empty predicate is a
        // no-op in both helpers (`validate_clauses` iterates zero clauses;
        // `clauses_match` is a vacuous `all` => true), so it isn't special-cased.
        let where_clauses: &[WhereClause] =
            scope.map(|s| s.where_clauses.as_slice()).unwrap_or(&[]);
        for nt in &node_types {
            crate::nodes_scan::validate_clauses(schema, nt, where_clauses)?;
        }

        // Collect in-scope nodes; detect id conflation across types (fail loud).
        let mut id_type: HashMap<String, String> = HashMap::new();
        for nt in &node_types {
            for node in engine.scan_nodes(graph_id, nt)? {
                // Drop nodes failing the property predicate before they (or any
                // edge touching them) enter the graph.
                if !crate::nodes_scan::clauses_match(where_clauses, &node) {
                    continue;
                }
                if let Some(prev) = id_type.insert(node.node_id.clone(), nt.clone())
                    && &prev != nt
                {
                    return Err(RegistryError::BadRequest(format!(
                        "node id {:?} exists under multiple node types ({prev}, {nt}) within the \
                         algorithm scope; restrict scope.node_types to disambiguate",
                        node.node_id
                    )));
                }
                if id_type.len() > MAX_ALGO_NODES {
                    return Err(RegistryError::BadRequest(format!(
                        "algorithm scope exceeds the {MAX_ALGO_NODES}-node limit; narrow scope.node_types"
                    )));
                }
            }
        }

        // Sort node ids so dense indices (and therefore component ordering and
        // any index-derived output) are deterministic rather than dependent on
        // HashMap iteration order.
        let mut ids: Vec<&String> = id_type.keys().collect();
        ids.sort();

        let mut builder = GraphBuilder::new();
        for id in &ids {
            builder.add_node(id.as_str());
        }

        // Add in-scope edges (both endpoints in scope), projecting weights.
        let mut edge_count = 0usize;
        for from_id in &ids {
            for edge in engine.scan_outgoing_edges(graph_id, from_id.as_str(), None)? {
                if !edge_types.contains(&edge.edge_type) || !id_type.contains_key(&edge.to_id) {
                    continue;
                }
                let w = project_weight(&edge, weight)?;
                builder
                    .add_edge(from_id.as_str(), &edge.to_id, w)
                    .map_err(|e| RegistryError::BadRequest(e.to_string()))?;
                edge_count += 1;
                if edge_count > MAX_ALGO_EDGES {
                    return Err(RegistryError::BadRequest(format!(
                        "algorithm scope exceeds the {MAX_ALGO_EDGES}-edge limit; narrow scope"
                    )));
                }
            }
        }

        Ok(builder.build(directed))
    }

    /// Project an edge to its `f64` weight per the [`WeightSpec`], failing loud
    /// on a missing/non-numeric property or an unmapped edge type.
    fn project_weight(
        edge: &StoredEdge,
        weight: Option<&WeightSpec>,
    ) -> Result<f64, RegistryError> {
        let Some(spec) = weight else {
            return Ok(1.0);
        };
        if let Some(prop) = &spec.property {
            let value = edge.properties.get(prop).ok_or_else(|| {
                RegistryError::BadRequest(format!(
                    "edge ({} -> {}) of type {} is missing weight property {prop:?}",
                    edge.from_id, edge.to_id, edge.edge_type
                ))
            })?;
            return value.as_f64().ok_or_else(|| {
                RegistryError::BadRequest(format!(
                    "edge ({} -> {}) weight property {prop:?} is not numeric",
                    edge.from_id, edge.to_id
                ))
            });
        }
        if let Some(map) = &spec.edge_type_weights {
            return map.get(&edge.edge_type).copied().ok_or_else(|| {
                RegistryError::BadRequest(format!(
                    "edge type {} has no entry in weight.edge_type_weights",
                    edge.edge_type
                ))
            });
        }
        Ok(1.0)
    }
}
