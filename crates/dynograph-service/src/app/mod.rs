//! axum app + route handlers.

use std::collections::HashMap;
use std::sync::Arc;

use std::time::Instant;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, MatchedPath, Path, Query, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use utoipa::{IntoParams, ToSchema};

use dynograph_core::{PropertyType, Schema, Value};
use dynograph_vector::HnswIndex;

use crate::{
    algo::{
        AlgoDirection, AlgoScope, BetweennessRequest, ClosenessRequest, ClusteringResponse,
        CommunitiesRequest, CommunitiesResponse, ComponentsResponse, CutEdge, CutsResponse,
        CyclesResponse, DegreeModeWire, DegreeRequest, EigenvectorRequest, FlowEdge,
        LinkPredictionMethodWire, LinkPredictionRequest, LinkPredictionResponse, MaxFlowResponse,
        NodeScore, PageRankRequest, PersonalizedPageRankRequest, PredictedLink, ScopedRequest,
        ScoresResponse, ShortestPathResponse, SourceTargetRequest, ToposortResponse, WeightSpec,
    },
    auth::{AuthProvider, NoAuth},
    batch::{
        BatchOk, BatchOp, BatchOpError, BatchOpResult, BatchRequest, BatchResponse,
        BatchValidation, MAX_BATCH_OPS, dry_run_ops, run_ops,
    },
    buildinfo_response::{BuildInfoResponse, GIT_DIRTY, GIT_SHA},
    config::ServerLimits,
    dbscan::{DbscanRequest, DbscanResponse, run as run_dbscan},
    edge_response::EdgeResponse,
    edges_adjacent::{
        AdjacentEdge, Direction as AdjacentDirection, EdgesAdjacentRequest, EdgesAdjacentResponse,
        run as run_edges_adjacent,
    },
    edges_collect::{
        AdjacencyEntry, CollectedEdge, EdgesCollectRequest, EdgesCollectResponse,
        PropertyFilter as CollectPropertyFilter, ResponseFormat, SourceSpec, SourceTypeFilter,
        TargetNode, run as run_edges_collect,
    },
    embedding_response::EmbeddingResponse,
    error_body::error_response,
    game::{
        CellSpec, DominantInfo, GameAnalyzeRequest, GameAnalyzeResponse, Mixed2x2Info,
        NashDominationInfo, PlayerSpec, run_analyze as run_game_analyze,
    },
    metadata_response::GraphMetadataResponse,
    metrics_state::MetricsState,
    node_response::{NodeListResponse, NodeResponse},
    nodes_exists::{
        NodeExistence, NodeQuery, NodesExistsRequest, NodesExistsResponse, run as run_nodes_exists,
    },
    nodes_scan::{
        NodesScanRequest, NodesScanResponse, Op, ReturnShape, WhereClause, run as run_nodes_scan,
    },
    readiness::Readiness,
    registry::{GraphEntry, GraphRegistry, RegistryError, validate_graph_id},
    resolve_or_create::{
        MatchKind, ResolveOrCreateRequest, ResolveOrCreateResponse, ScopeFilter,
        run as run_resolve_or_create,
    },
    schema_response::{SchemaResponse, WIRE_VERSION},
    search_hybrid::{
        HybridHit, HybridLegBreakdown, HybridLegInfo, LegName, LegWeights, SearchHybridBody,
        SearchHybridResponse, run as run_search_hybrid,
    },
    similar_response::{SimilarHit, SimilarResponse},
    traverse::{
        Direction as TraverseDirection, PropertyFilter as TraversePropertyFilter, ReturnFormat,
        StartSpec, TraverseRequest, TraverseResponse, TraverseStep, TraversedNode,
        run as run_traverse,
    },
    util::{
        BinaryStringRequest, BinaryVectorRequest, DistanceMetric, MatrixResponse,
        PairwiseDistanceRequest, PercentileRequest, PointsRequest, PowerRequest, Precision,
        SampleRequest, ScalarResponse, ScaleRequest, TwoVectorF64Request, UnaryVectorRequest,
        VectorResponse, VectorsRequest, run_add, run_centroid, run_cosine_similarity,
        run_dot_product, run_elementwise_power, run_euclidean_distance, run_hadamard,
        run_hadamard_division, run_jaro_winkler, run_l2_norm, run_l2_normalize,
        run_linear_regression_slope, run_manhattan_distance, run_mean, run_median, run_negate,
        run_pairwise_cosine, run_pairwise_distance, run_pearson_correlation, run_percentile,
        run_scale, run_softmax, run_spearman_correlation, run_squared_euclidean_distance,
        run_std_dev, run_subtract, run_token_sort_ratio, run_variance, validate_embedding_values,
    },
    validation::validate_limit,
    welford_update::{WelfordUpdateRequest, WelfordUpdateResponse, run as run_welford_update},
};

mod algo;
mod apidoc;
mod edges;
mod embeddings;
mod graphs;
mod nodes;
mod primitives;
mod search;
mod util;

pub use apidoc::ApiDoc;

// Bring every handler fn (and its utoipa `__path_*` item) + handler-local wire
// body struct into scope by bare name, so the `app()` router below — and, via
// `use super::*`, the `paths(...)`/`schemas(...)` lists in `apidoc` — resolve
// exactly as they did when everything lived in one file.
use algo::*;
use edges::*;
use embeddings::*;
use graphs::*;
use nodes::*;
use primitives::*;
use search::*;
use util::*;

#[derive(Clone)]
pub struct AppState {
    pub(crate) registry: Arc<GraphRegistry>,
    pub(crate) auth: Arc<dyn AuthProvider>,
    pub(crate) readiness: Arc<Readiness>,
    pub(crate) metrics: Arc<MetricsState>,
    /// Ingress hardening limits (body size, request timeout) read by
    /// `app()` when building the middleware layers.
    pub(crate) limits: ServerLimits,
    /// Shared permit pool backing the `/v1` concurrency limit. Sized
    /// from `limits.max_concurrent_requests`; one permit is held for
    /// the duration of each in-flight `/v1` request.
    pub(crate) limiter: Arc<Semaphore>,
}

impl AppState {
    pub fn new(
        registry: Arc<GraphRegistry>,
        auth: Arc<dyn AuthProvider>,
        readiness: Arc<Readiness>,
    ) -> Self {
        let limits = ServerLimits::default();
        Self {
            registry,
            auth,
            readiness,
            metrics: Arc::new(MetricsState::new()),
            limiter: Arc::new(Semaphore::new(limits.max_concurrent_requests)),
            limits,
        }
    }

    /// Override the ingress hardening limits (the `dynograph` binary
    /// feeds these from `[server]` config; tests inject tiny values to
    /// exercise the 413/408/503 paths). Rebuilds the concurrency permit
    /// pool to match `max_concurrent_requests`.
    pub fn with_limits(mut self, limits: ServerLimits) -> Self {
        self.limiter = Arc::new(Semaphore::new(limits.max_concurrent_requests));
        self.limits = limits;
        self
    }

    /// Convenience for the dev / private-network default. Picks
    /// `NoAuth` and `Readiness::ready` — the right defaults for
    /// in-memory test code and embedded use, neither of which has
    /// startup work that would warrant a not-ready window. The
    /// `dynograph` binary uses the lower-level `AppState::new`
    /// with an explicit not-ready `Readiness` because it does have
    /// startup work (`rehydrate`) and flips ready only after.
    pub fn with_no_auth(registry: Arc<GraphRegistry>) -> Self {
        Self::new(
            registry,
            Arc::new(NoAuth::new()),
            Arc::new(Readiness::ready()),
        )
    }

    pub fn readiness(&self) -> &Arc<Readiness> {
        &self.readiness
    }
}

pub fn app(state: AppState) -> Router {
    // /v1/* routes go through both the metrics middleware (outer)
    // and the auth middleware (inner): incoming → record start time
    // → authenticate → handler → record latency. /metrics itself is
    // public AND skips the metrics middleware (no self-recording on
    // every scrape). /health and /ready are public but ARE recorded
    // — useful for "is anyone hitting the probes" debugging.
    let v1: Router<AppState> = Router::new()
        .route("/v1/graphs", get(list_graphs).post(create_graph))
        .route("/v1/graphs/{id}", get(get_graph).delete(delete_graph))
        .route(
            "/v1/graphs/{id}/schema",
            get(get_schema).put(replace_schema),
        )
        .route("/v1/graphs/{id}/nodes", get(list_nodes).post(create_node))
        .route(
            "/v1/graphs/{id}/nodes/{node_type}/{node_id}",
            get(get_node).put(replace_node).delete(delete_node),
        )
        .route("/v1/graphs/{id}/edges", post(create_edge))
        .route(
            "/v1/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}",
            get(get_edge).patch(merge_edge).delete(delete_edge),
        )
        .route(
            "/v1/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}/welford_update",
            post(welford_update),
        )
        .route("/v1/graphs/{id}/batch", post(batch))
        .route("/v1/graphs/{id}/resolve-or-create", post(resolve_or_create))
        .route("/v1/graphs/{id}/edges:collect", post(edges_collect))
        .route("/v1/graphs/{id}/edges:adjacent", post(edges_adjacent))
        .route("/v1/graphs/{id}/nodes:exists", post(nodes_exists))
        .route("/v1/graphs/{id}/nodes:scan", post(nodes_scan))
        .route("/v1/graphs/{id}/traverse", post(traverse))
        .route("/v1/graphs/{id}/algo/components", post(algo_components))
        .route("/v1/graphs/{id}/algo/degree", post(algo_degree))
        .route("/v1/graphs/{id}/algo/pagerank", post(algo_pagerank))
        .route("/v1/graphs/{id}/algo/eigenvector", post(algo_eigenvector))
        .route("/v1/graphs/{id}/algo/closeness", post(algo_closeness))
        .route("/v1/graphs/{id}/algo/betweenness", post(algo_betweenness))
        .route("/v1/graphs/{id}/algo/cuts", post(algo_cuts))
        .route("/v1/graphs/{id}/algo/scc", post(algo_scc))
        .route("/v1/graphs/{id}/algo/cycles", post(algo_cycles))
        .route(
            "/v1/graphs/{id}/algo/personalized_pagerank",
            post(algo_personalized_pagerank),
        )
        .route(
            "/v1/graphs/{id}/algo/shortest_path",
            post(algo_shortest_path),
        )
        .route(
            "/v1/graphs/{id}/algo/link_prediction",
            post(algo_link_prediction),
        )
        .route("/v1/graphs/{id}/algo/clustering", post(algo_clustering))
        .route("/v1/graphs/{id}/algo/communities", post(algo_communities))
        .route("/v1/graphs/{id}/algo/toposort", post(algo_toposort))
        .route("/v1/graphs/{id}/algo/max_flow", post(algo_max_flow))
        .route(
            "/v1/graphs/{id}/nodes/{node_type}/{node_id}/embedding",
            get(get_embedding)
                .put(set_embedding)
                .delete(delete_embedding),
        )
        .route("/v1/graphs/{id}/similar", post(similar))
        .route("/v1/graphs/{id}/search:text", post(search_text))
        .route("/v1/graphs/{id}/search:hybrid", post(search_hybrid))
        .route("/v1/graphs/{id}/search:reindex", post(search_reindex))
        .route("/v1/util/cosine_similarity", post(util_cosine_similarity))
        .route("/v1/util/dot_product", post(util_dot_product))
        .route("/v1/util/euclidean_distance", post(util_euclidean_distance))
        .route("/v1/util/l2_norm", post(util_l2_norm))
        .route("/v1/util/hadamard", post(util_hadamard))
        .route(
            "/v1/util/pearson_correlation",
            post(util_pearson_correlation),
        )
        .route(
            "/v1/util/linear_regression_slope",
            post(util_linear_regression_slope),
        )
        .route("/v1/util/jaro_winkler", post(util_jaro_winkler))
        .route("/v1/util/token_sort_ratio", post(util_token_sort_ratio))
        .route(
            "/v1/util/squared_euclidean_distance",
            post(util_squared_euclidean_distance),
        )
        .route("/v1/util/manhattan_distance", post(util_manhattan_distance))
        .route("/v1/util/add", post(util_add))
        .route("/v1/util/subtract", post(util_subtract))
        .route("/v1/util/scale", post(util_scale))
        .route("/v1/util/negate", post(util_negate))
        .route("/v1/util/hadamard_division", post(util_hadamard_division))
        .route("/v1/util/elementwise_power", post(util_elementwise_power))
        .route("/v1/util/l2_normalize", post(util_l2_normalize))
        .route("/v1/util/centroid", post(util_centroid))
        .route("/v1/util/pairwise_cosine", post(util_pairwise_cosine))
        .route("/v1/util/pairwise_distance", post(util_pairwise_distance))
        .route("/v1/util/game/analyze", post(util_game_analyze))
        .route("/v1/util/dbscan", post(util_dbscan))
        .route("/v1/util/mean", post(util_mean))
        .route("/v1/util/variance", post(util_variance))
        .route("/v1/util/std_dev", post(util_std_dev))
        .route("/v1/util/median", post(util_median))
        .route("/v1/util/percentile", post(util_percentile))
        .route("/v1/util/softmax", post(util_softmax))
        .route(
            "/v1/util/spearman_correlation",
            post(util_spearman_correlation),
        )
        // route_layer order is inner→outer in source order, so the
        // concurrency limit (added last) wraps auth — load is shed at
        // the door before any auth/handler work. Deliberately scoped to
        // /v1 only: probes (/health,/ready,/metrics) below are NOT
        // behind it, so liveness stays observable when /v1 is saturated.
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            concurrency_middleware,
        ));

    let observed_public: Router<AppState> = Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .merge(v1)
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            metrics_middleware,
        ));

    let body_limit = state.limits.max_body_bytes;
    Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/buildinfo", get(buildinfo_handler))
        .route("/openapi.json", get(openapi_json))
        .merge(observed_public)
        // Global: body-size cap (413) + per-request timeout (408) apply
        // to every route, probes included — probes are cheap so neither
        // ever trips for them.
        .layer(middleware::from_fn_with_state(
            state.clone(),
            timeout_middleware,
        ))
        .layer(DefaultBodyLimit::max(body_limit))
        .with_state(state)
}

/// Per-request wall-clock cap → 408 when a handler runs past
/// `limits.request_timeout`. A `spawn_blocking` storage op already in
/// flight runs to completion in the background; this sheds the client's
/// wait, it does not cancel the scan.
async fn timeout_middleware(State(state): State<AppState>, req: Request, next: Next) -> Response {
    match tokio::time::timeout(state.limits.request_timeout, next.run(req)).await {
        Ok(resp) => resp,
        Err(_) => error_response(
            StatusCode::REQUEST_TIMEOUT,
            "request exceeded the server time limit",
        ),
    }
}

/// In-flight cap for `/v1` → 503 (load-shed, not queue) once
/// `limits.max_concurrent_requests` requests are already executing, so
/// a burst of heavy scans can't exhaust the blocking pool or memory.
/// The permit borrows `state.limiter` and is held across the handler;
/// `state` outlives it, so a borrowed (not owned) permit suffices.
async fn concurrency_middleware(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    match state.limiter.try_acquire() {
        Ok(_permit) => next.run(req).await,
        Err(_) => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "server at capacity, retry later",
        ),
    }
}

/// Axum middleware: runs `state.auth.authenticate(headers)` on every
/// protected request. On success, inserts the resolved `Identity`
/// into request extensions so downstream handlers can read the
/// caller's user_id via `Extension<Identity>` if they need it. On
/// failure, short-circuits with 401 + the auth error's message.
///
/// NOTE: no handler reads `Extension<Identity>` yet — the principal is
/// authenticated but does not affect behavior (a valid token gets
/// registry-wide access). This insert is deliberate forward plumbing
/// for per-graph ACLs; until those land, identity is decoded-but-unused
/// by design, NOT an authorization layer. (Tracked: per-graph ACL TD.)
async fn auth_middleware(State(state): State<AppState>, mut req: Request, next: Next) -> Response {
    match state.auth.authenticate(req.headers()) {
        Ok(identity) => {
            req.extensions_mut().insert(identity);
            next.run(req).await
        }
        Err(e) => error_response(StatusCode::UNAUTHORIZED, e.message().to_string()),
    }
}

/// Axum middleware: records (method, matched-path, status) +
/// latency into `MetricsState`. The matched-path label uses axum's
/// `MatchedPath` extension so cardinality stays bounded by static
/// route count (e.g. `/v1/graphs/{id}` is one label, regardless of
/// how many distinct ids were hit). Requests that miss every route
/// (404 from the router itself) won't have a `MatchedPath` set —
/// those are intentionally skipped to keep the label set finite.
async fn metrics_middleware(State(state): State<AppState>, req: Request, next: Next) -> Response {
    // Skip recording (and the per-request String allocs) when the
    // router didn't match anything — those 404s have no MatchedPath
    // and would inflate label cardinality if we made up a `__none__`
    // bucket. Allocate the owned method+path strings only after we
    // know we're going to insert.
    let matched_path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string());
    let Some(path) = matched_path else {
        return next.run(req).await;
    };
    let method = req.method().as_str().to_string();
    let start = Instant::now();
    let response = next.run(req).await;
    let elapsed_micros = start.elapsed().as_micros() as u64;
    state
        .metrics
        .record(&method, &path, response.status().as_u16(), elapsed_micros);
    response
}

/// JSON build provenance: `version` + `git_sha` + `git_dirty` +
/// `uptime_seconds`. Same provenance triple the `dynograph_build_info`
/// gauge surfaces in `/metrics`, JSON-shaped for callers that don't
/// want to parse Prometheus text format. Public; same auth/middleware
/// posture as `/metrics`.
#[utoipa::path(
    get,
    path = "/buildinfo",
    responses((status = 200, description = "build provenance", body = BuildInfoResponse)),
    tag = "ops",
)]
async fn buildinfo_handler(State(state): State<AppState>) -> impl IntoResponse {
    Json(BuildInfoResponse::new(state.metrics.uptime_secs()))
}

/// Prometheus text-format scrape endpoint. Public — sit alongside
/// `/health` and `/ready`; the assumption is that the network/
/// ingress layer gates Prometheus scrape access to the metrics
/// endpoint when needed (k8s NetworkPolicy / Caddy IP allowlist /
/// etc). `/metrics` itself bypasses the metrics middleware to avoid
/// recording every scrape into the request-counter series, which
/// would inflate cardinality and mostly measure Prometheus's own
/// scrape interval.
#[utoipa::path(
    get,
    path = "/metrics",
    responses((status = 200, description = "Prometheus text-format metrics", body = String)),
    tag = "ops",
)]
async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    use std::fmt::Write;
    let mut out = String::new();

    let _ = writeln!(out, "# HELP dynograph_build_info Build information");
    let _ = writeln!(out, "# TYPE dynograph_build_info gauge");
    let _ = writeln!(
        out,
        "dynograph_build_info{{version=\"{WIRE_VERSION}\",git_sha=\"{GIT_SHA}\",git_dirty=\"{GIT_DIRTY}\"}} 1",
    );

    let _ = writeln!(
        out,
        "# HELP dynograph_uptime_seconds Process uptime since start"
    );
    let _ = writeln!(out, "# TYPE dynograph_uptime_seconds gauge");
    let _ = writeln!(
        out,
        "dynograph_uptime_seconds {:.3}",
        state.metrics.uptime_secs()
    );

    let snap = state.metrics.snapshot();
    let _ = writeln!(
        out,
        "# HELP dynograph_http_requests_total Requests handled, by route + status"
    );
    let _ = writeln!(out, "# TYPE dynograph_http_requests_total counter");
    for (key, count, _sum) in &snap {
        let _ = writeln!(
            out,
            "dynograph_http_requests_total{{method=\"{}\",path=\"{}\",status=\"{}\"}} {}",
            key.method, key.path, key.status, count
        );
    }
    let _ = writeln!(
        out,
        "# HELP dynograph_http_request_duration_microseconds_sum Cumulative request latency"
    );
    let _ = writeln!(
        out,
        "# TYPE dynograph_http_request_duration_microseconds_sum counter"
    );
    for (key, _count, sum) in &snap {
        let _ = writeln!(
            out,
            "dynograph_http_request_duration_microseconds_sum{{method=\"{}\",path=\"{}\",status=\"{}\"}} {}",
            key.method, key.path, key.status, sum
        );
    }

    // Per-(graph, node_type) HNSW stats. Walks the registry under
    // its read lock; per-graph stats acquire the per-graph state
    // read lock briefly. Scrape cost scales with the number of
    // graphs × node_types-with-embeddings — for the foreseeable
    // range (dozens of graphs, single-digit indexed types each)
    // this is microseconds per scrape.
    let mut hnsw_snap: Vec<(String, String, dynograph_vector::HnswStats)> = Vec::new();
    for graph_id in state.registry.list_ids() {
        if let Some(entry) = state.registry.get(&graph_id) {
            for (node_type, stats) in entry.hnsw_stats_snapshot() {
                hnsw_snap.push((graph_id.clone(), node_type, stats));
            }
        }
    }
    emit_hnsw_metric(
        &mut out,
        "dynograph_hnsw_index_size",
        "gauge",
        "Live (non-tombstoned) embeddings per index",
        &hnsw_snap,
        |s| s.index_size as u64,
    );
    emit_hnsw_metric(
        &mut out,
        "dynograph_hnsw_searches_total",
        "counter",
        "HNSW search calls per index",
        &hnsw_snap,
        |s| s.searches_total,
    );
    emit_hnsw_metric(
        &mut out,
        "dynograph_hnsw_inserts_total",
        "counter",
        "HNSW insert calls per index",
        &hnsw_snap,
        |s| s.inserts_total,
    );
    emit_hnsw_metric(
        &mut out,
        "dynograph_hnsw_removes_total",
        "counter",
        "HNSW remove calls per index",
        &hnsw_snap,
        |s| s.removes_total,
    );

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        out,
    )
}

/// Emit one HNSW metric block (HELP + TYPE header + per-(graph,
/// node_type) line). The `field` projector picks the relevant stat
/// — collapses the four near-identical emission blocks (one per
/// counter) to one helper.
fn emit_hnsw_metric(
    out: &mut String,
    metric: &str,
    metric_type: &str,
    help: &str,
    snap: &[(String, String, dynograph_vector::HnswStats)],
    field: impl Fn(&dynograph_vector::HnswStats) -> u64,
) {
    use std::fmt::Write;
    let _ = writeln!(out, "# HELP {metric} {help}");
    let _ = writeln!(out, "# TYPE {metric} {metric_type}");
    for (graph, node_type, stats) in snap {
        let _ = writeln!(
            out,
            "{metric}{{graph=\"{graph}\",node_type=\"{node_type}\"}} {}",
            field(stats)
        );
    }
}

/// Serve the generated OpenAPI 3.1 document. Public — same posture as
/// `/metrics` and `/buildinfo` (no auth, no metrics-recording layer).
async fn openapi_json() -> impl IntoResponse {
    Json(<ApiDoc as utoipa::OpenApi>::openapi())
}

#[utoipa::path(
    get,
    path = "/health",
    responses((status = 200, description = "process is alive", body = String)),
    tag = "ops",
)]
async fn health() -> &'static str {
    "ok"
}

/// Readiness probe — distinct from `/health`, which only confirms
/// the process is running. `/ready` returns 200 once the service
/// has finished startup work (notably `rehydrate()` on the on-disk
/// backend); 503 before that.
#[utoipa::path(
    get,
    path = "/ready",
    responses(
        (status = 200, description = "service is ready", body = String),
        (status = 503, description = "still starting up"),
    ),
    tag = "ops",
)]
async fn ready(State(state): State<AppState>) -> (StatusCode, &'static str) {
    if state.readiness.is_ready() {
        (StatusCode::OK, "ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "starting")
    }
}

/// Look up a graph by id or surface a 404. Folds the
/// `state.registry.get(...).ok_or_else(...)` boilerplate that every
/// graph-id-bearing handler shares.
fn graph_entry(state: &AppState, id: &str) -> Result<Arc<GraphEntry>, RegistryError> {
    // Validate the id on every read path, not just on create. Today
    // `registry.get` is a plain map lookup so a malformed id merely
    // misses (404), but the OnDisk backend joins the id into a
    // filesystem path elsewhere — validating here keeps an unvalidated
    // id from ever reaching a path join (a future-proofed 400 instead of
    // a latent traversal footgun), and matches the create-time contract.
    validate_graph_id(id)?;
    state
        .registry
        .get(id)
        .ok_or_else(|| RegistryError::NotFound(id.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::GraphRegistry;
    use axum::body::Body;
    use std::time::Duration;
    use tower::ServiceExt;

    /// `timeout_middleware` returns 408 when the wrapped handler runs
    /// past the configured `request_timeout`. Driven through a synthetic
    /// always-slow route so the elapse is deterministic (a real endpoint
    /// is too fast to race a timer against). The slow future is dropped
    /// when the timeout fires, so the test finishes in ~the timeout, not
    /// the 30 s sleep.
    #[tokio::test]
    async fn timeout_middleware_returns_408_for_slow_handler() {
        async fn slow() -> &'static str {
            tokio::time::sleep(Duration::from_secs(30)).await;
            "unreachable"
        }
        let state =
            AppState::with_no_auth(Arc::new(GraphRegistry::new())).with_limits(ServerLimits {
                request_timeout: Duration::from_millis(10),
                ..Default::default()
            });
        let app = Router::new()
            .route("/slow", get(slow))
            .layer(middleware::from_fn_with_state(
                state.clone(),
                timeout_middleware,
            ))
            .with_state(state);

        let res = app
            .oneshot(Request::builder().uri("/slow").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::REQUEST_TIMEOUT);
    }

    /// Contract gate: the committed `docs/openapi.json` must match the
    /// spec generated from the code. This makes the OpenAPI document a
    /// reviewed artifact — any change to a route or wire type that
    /// alters the contract fails this test (and the existing CI
    /// `cargo test`) until the spec is regenerated and committed.
    ///
    /// Regenerate after an intended contract change with:
    ///   `UPDATE_OPENAPI=1 cargo test -p dynograph-service openapi_spec`
    #[test]
    fn openapi_spec_matches_committed_docs() {
        use utoipa::OpenApi;
        let generated =
            serde_json::to_string_pretty(&ApiDoc::openapi()).expect("serialize openapi") + "\n";
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/openapi.json");

        if std::env::var_os("UPDATE_OPENAPI").is_some() {
            std::fs::write(path, &generated).expect("write docs/openapi.json");
            return;
        }

        let committed = std::fs::read_to_string(path).unwrap_or_else(|e| {
            panic!(
                "docs/openapi.json unreadable ({e}); regenerate with \
                 `UPDATE_OPENAPI=1 cargo test -p dynograph-service openapi_spec`"
            )
        });
        assert_eq!(
            generated, committed,
            "OpenAPI spec drifted from docs/openapi.json — the wire contract changed. \
             Review the diff, then regenerate with \
             `UPDATE_OPENAPI=1 cargo test -p dynograph-service openapi_spec`."
        );
    }
}
