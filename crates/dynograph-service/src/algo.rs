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
//! Node identity in the in-memory graph is the bare `node_id`. If the same id
//! appears under two different node types within scope, the build fails loud
//! (400) rather than conflate two distinct nodes — narrow `scope.node_types` to
//! disambiguate.

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

/// Degree-centrality scores, highest first (ties broken by node id).
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct DegreeResponse {
    pub scores: Vec<NodeScore>,
}

// ---- Algorithm implementation (only when the `graph` feature is on) ----

#[cfg(feature = "graph")]
pub(crate) use imp::{run_components, run_degree};

#[cfg(feature = "graph")]
mod imp {
    use super::*;

    use dynograph_graph::{
        DegreeMode, Graph, GraphBuilder, connected_components, degree_centrality,
    };
    use dynograph_storage::{StorageEngine, StoredEdge};

    use crate::registry::RegistryError;

    /// Safety cap on subgraph size. Consumer graphs are small (10^2-10^3 nodes);
    /// a scope that pulls far more is almost certainly a mis-scoped request, so
    /// fail loud rather than risk an OOM building the in-memory graph.
    const MAX_ALGO_NODES: usize = 100_000;

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
    ) -> Result<DegreeResponse, RegistryError> {
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
        let mut scores: Vec<NodeScore> = raw
            .into_iter()
            .enumerate()
            .map(|(idx, score)| NodeScore {
                node: graph.id_of(idx).to_string(),
                score,
            })
            .collect();
        // Highest score first; deterministic tie-break by node id.
        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.node.cmp(&b.node))
        });
        Ok(DegreeResponse { scores })
    }

    /// Build the in-memory graph from storage under the caller's read lock.
    fn build_graph(
        engine: &StorageEngine,
        graph_id: &str,
        scope: Option<&AlgoScope>,
        weight: Option<&WeightSpec>,
        directed: bool,
    ) -> Result<Graph, RegistryError> {
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
        for from_id in &ids {
            for edge in engine.scan_outgoing_edges(graph_id, from_id.as_str(), None)? {
                if !edge_types.contains(&edge.edge_type) || !id_type.contains_key(&edge.to_id) {
                    continue;
                }
                let w = project_weight(&edge, weight)?;
                builder
                    .add_edge(from_id.as_str(), &edge.to_id, w)
                    .map_err(|e| RegistryError::BadRequest(e.to_string()))?;
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
