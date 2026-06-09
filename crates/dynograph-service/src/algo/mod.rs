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

mod types;
pub(crate) use types::*;

#[cfg(feature = "graph")]
mod imp;

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
