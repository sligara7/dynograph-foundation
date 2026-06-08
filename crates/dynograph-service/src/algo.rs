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

// ---- Wire types (always compiled — they ARE the OpenAPI contract) ----

/// Subgraph selection shared by every algorithm. Omitted fields mean "all":
/// no `node_types` => every node type in the schema; no `edge_types` => every
/// edge type.
#[derive(Debug, Deserialize, ToSchema)]
#[cfg_attr(not(feature = "graph"), allow(dead_code))]
pub(crate) struct AlgoScope {
    /// Node types to include. Omit for all types.
    #[serde(default)]
    pub node_types: Option<Vec<String>>,
    /// Edge types to include. Omit for all types.
    #[serde(default)]
    pub edge_types: Option<Vec<String>>,
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

/// Request for `POST /v1/graphs/{id}/algo/components`.
#[derive(Debug, Deserialize, ToSchema)]
#[cfg_attr(not(feature = "graph"), allow(dead_code))]
pub(crate) struct ComponentsRequest {
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

/// One (weakly-)connected component: the node ids it contains.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ComponentsResponse {
    /// Number of components.
    pub count: usize,
    /// Components as lists of node ids. Order is deterministic (by first node
    /// visited); within a component, by node index.
    pub components: Vec<Vec<String>>,
}

/// A single node's score.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct NodeScore {
    pub node: String,
    pub score: f64,
}

/// Per-node centrality scores, highest first (ties broken by node id). Shared by
/// every score-producing algorithm (degree, PageRank, eigenvector, closeness,
/// betweenness).
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ScoresResponse {
    pub scores: Vec<NodeScore>,
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
    run_betweenness, run_closeness, run_components, run_degree, run_eigenvector, run_pagerank,
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
not_enabled_stub!(run_components, ComponentsRequest, ComponentsResponse);
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

#[cfg(feature = "graph")]
mod imp {
    use super::*;

    use dynograph_graph::{
        DegreeMode, EigenvectorConfig, Graph, GraphBuilder, GraphError, PageRankConfig,
        betweenness_centrality, closeness_centrality, connected_components, degree_centrality,
        eigenvector_centrality, pagerank,
    };
    use dynograph_storage::{StorageEngine, StoredEdge};

    use crate::registry::RegistryError;

    /// Safety cap on subgraph size. Consumer graphs are small (10^2-10^3 nodes);
    /// a scope that pulls far more is almost certainly a mis-scoped request, so
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
        req: ComponentsRequest,
    ) -> Result<ComponentsResponse, RegistryError> {
        let graph = build_graph(engine, graph_id, req.scope.as_ref(), None, true)?;
        let comps = connected_components(&graph);
        let components = comps
            .groups()
            .into_iter()
            .map(|group| {
                group
                    .into_iter()
                    .map(|idx| graph.id_of(idx).to_string())
                    .collect()
            })
            .collect();
        Ok(ComponentsResponse {
            count: comps.count,
            components,
        })
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
        let mut config = PageRankConfig::default();
        if let Some(d) = req.damping {
            if !(0.0..=1.0).contains(&d) {
                return Err(RegistryError::BadRequest(format!(
                    "damping must be in [0, 1], got {d}"
                )));
            }
            config.damping = d;
        }
        if let Some(t) = req.tolerance {
            config.tolerance = validate_tolerance(t)?;
        }
        if let Some(m) = req.max_iterations {
            config.max_iterations = validate_max_iterations(m)?;
        }
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

    fn validate_tolerance(t: f64) -> Result<f64, RegistryError> {
        if !t.is_finite() || t <= 0.0 {
            return Err(RegistryError::BadRequest(format!(
                "tolerance must be a positive number, got {t}"
            )));
        }
        Ok(t)
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

        // Resolve in-scope node types (validate any explicitly requested).
        let node_types: Vec<String> = match scope.and_then(|s| s.node_types.as_ref()) {
            Some(types) => {
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

        // Collect in-scope nodes; detect id conflation across types (fail loud).
        let mut id_type: HashMap<String, String> = HashMap::new();
        for nt in &node_types {
            for node in engine.scan_nodes(graph_id, nt)? {
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
