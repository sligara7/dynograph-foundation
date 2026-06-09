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

/// OpenAPI 3.1 document aggregator. Lists every annotated handler and
/// every `ToSchema` wire type so `<ApiDoc as OpenApi>::openapi()` (served
/// at `GET /openapi.json`) describes the whole `/v1` surface plus the ops
/// endpoints.
#[derive(utoipa::OpenApi)]
#[openapi(
    info(title = "dynograph-foundation", version = env!("CARGO_PKG_VERSION")),
    modifiers(&AuthResponses),
    paths(
        // graphs
        list_graphs,
        create_graph,
        get_graph,
        get_schema,
        replace_schema,
        delete_graph,
        // nodes
        create_node,
        list_nodes,
        get_node,
        replace_node,
        delete_node,
        // edges
        create_edge,
        get_edge,
        merge_edge,
        delete_edge,
        // primitives
        batch,
        resolve_or_create,
        edges_collect,
        edges_adjacent,
        nodes_exists,
        nodes_scan,
        welford_update,
        traverse,
        // graph-theory algorithms (behind the `graph` feature)
        algo_components,
        algo_degree,
        algo_pagerank,
        algo_eigenvector,
        algo_closeness,
        algo_betweenness,
        algo_cuts,
        algo_scc,
        algo_cycles,
        algo_personalized_pagerank,
        algo_shortest_path,
        algo_link_prediction,
        algo_clustering,
        algo_communities,
        algo_toposort,
        algo_max_flow,
        // embeddings + search
        set_embedding,
        get_embedding,
        delete_embedding,
        similar,
        search_text,
        search_hybrid,
        search_reindex,
        // util
        util_cosine_similarity,
        util_dot_product,
        util_euclidean_distance,
        util_l2_norm,
        util_hadamard,
        util_pearson_correlation,
        util_linear_regression_slope,
        util_jaro_winkler,
        util_token_sort_ratio,
        util_squared_euclidean_distance,
        util_manhattan_distance,
        util_add,
        util_subtract,
        util_scale,
        util_negate,
        util_hadamard_division,
        util_elementwise_power,
        util_l2_normalize,
        util_centroid,
        util_pairwise_cosine,
        util_pairwise_distance,
        util_mean,
        util_variance,
        util_std_dev,
        util_median,
        util_percentile,
        util_softmax,
        util_spearman_correlation,
        // ops
        health,
        ready,
        buildinfo_handler,
        metrics_handler,
    ),
    components(schemas(
        // graph / node / edge / schema wire types
        GraphListResponse,
        CreateGraphBody,
        SchemaResponse,
        GraphMetadataResponse,
        CreateNodeBody,
        ReplaceNodeBody,
        NodeResponse,
        NodeListResponse,
        CreateEdgeBody,
        MergeEdgeBody,
        EdgeResponse,
        // embeddings + search
        SetEmbeddingBody,
        EmbeddingResponse,
        SimilarBody,
        SimilarHit,
        SimilarResponse,
        SearchTextBody,
        SearchTextHit,
        SearchTextResponse,
        SearchReindexResponse,
        SearchHybridBody,
        LegName,
        LegWeights,
        SearchHybridResponse,
        HybridHit,
        HybridLegBreakdown,
        HybridLegInfo,
        // batch
        BatchRequest,
        BatchOp,
        BatchResponse,
        BatchOpError,
        BatchOpResult,
        BatchValidation,
        BatchOk,
        // resolve-or-create
        ResolveOrCreateRequest,
        ScopeFilter,
        MatchKind,
        ResolveOrCreateResponse,
        // edges:collect
        EdgesCollectRequest,
        SourceSpec,
        SourceTypeFilter,
        CollectPropertyFilter,
        ResponseFormat,
        CollectedEdge,
        AdjacencyEntry,
        TargetNode,
        EdgesCollectResponse,
        // edges:adjacent
        EdgesAdjacentRequest,
        AdjacentDirection,
        AdjacentEdge,
        EdgesAdjacentResponse,
        // nodes:exists
        NodesExistsRequest,
        NodeQuery,
        NodeExistence,
        NodesExistsResponse,
        // nodes:scan
        NodesScanRequest,
        WhereClause,
        Op,
        ReturnShape,
        NodesScanResponse,
        // traverse
        TraverseRequest,
        StartSpec,
        TraverseStep,
        TraverseDirection,
        TraversePropertyFilter,
        ReturnFormat,
        TraversedNode,
        TraverseResponse,
        // graph-theory algorithms
        AlgoScope,
        WeightSpec,
        AlgoDirection,
        DegreeModeWire,
        ScopedRequest,
        DegreeRequest,
        PageRankRequest,
        EigenvectorRequest,
        ClosenessRequest,
        BetweennessRequest,
        PersonalizedPageRankRequest,
        SourceTargetRequest,
        LinkPredictionRequest,
        LinkPredictionMethodWire,
        ComponentsResponse,
        ScoresResponse,
        NodeScore,
        CutsResponse,
        CutEdge,
        CyclesResponse,
        ShortestPathResponse,
        PredictedLink,
        LinkPredictionResponse,
        ClusteringResponse,
        CommunitiesRequest,
        CommunitiesResponse,
        ToposortResponse,
        MaxFlowResponse,
        FlowEdge,
        // welford
        WelfordUpdateRequest,
        WelfordUpdateResponse,
        // util
        Precision,
        BinaryVectorRequest,
        UnaryVectorRequest,
        TwoVectorF64Request,
        PointsRequest,
        BinaryStringRequest,
        VectorsRequest,
        ScaleRequest,
        PowerRequest,
        SampleRequest,
        PercentileRequest,
        ScalarResponse<f64>,
        ScalarResponse<u32>,
        VectorResponse,
        DistanceMetric,
        PairwiseDistanceRequest,
        MatrixResponse,
        // ops
        BuildInfoResponse,
    )),
    tags(
        (name = "graphs", description = "Graph lifecycle + schema"),
        (name = "nodes", description = "Node CRUD"),
        (name = "edges", description = "Edge CRUD"),
        (name = "embeddings", description = "Per-node embeddings"),
        (name = "search", description = "Vector similarity search"),
        (name = "primitives", description = "Composite graph primitives (batch, resolve, edges:*, nodes:*, traverse, welford)"),
        (name = "algo", description = "Graph-theory algorithms (components, centrality, ...). Requires the `graph` build feature; otherwise 501."),
        (name = "util", description = "Stateless pure-math utilities"),
        (name = "ops", description = "Health, readiness, metrics, build info"),
    )
)]
pub struct ApiDoc;

/// Documents the cross-cutting 401 that `auth_middleware` can return on
/// every `/v1` route when bearer auth is configured. Modeled once as a
/// modifier rather than repeated in 30 `#[utoipa::path]` blocks — the
/// 401 comes from middleware, not the handlers. (Default config is
/// `noauth`, hence "when bearer auth is enabled".)
struct AuthResponses;

impl utoipa::Modify for AuthResponses {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        use utoipa::openapi::{RefOr, Response};
        for (path, item) in openapi.paths.paths.iter_mut() {
            if !path.starts_with("/v1/") {
                continue;
            }
            let ops = [
                item.get.as_mut(),
                item.post.as_mut(),
                item.put.as_mut(),
                item.delete.as_mut(),
                item.patch.as_mut(),
            ];
            for op in ops.into_iter().flatten() {
                op.responses
                    .responses
                    .entry("401".to_string())
                    .or_insert_with(|| {
                        RefOr::T(Response::new(
                            "missing or invalid credentials (when bearer auth is enabled)",
                        ))
                    });
            }
        }
    }
}

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

#[derive(Debug, Deserialize, ToSchema)]
struct CreateGraphBody {
    id: String,
    #[schema(value_type = Object)]
    schema: Schema,
}

#[derive(Debug, Serialize, ToSchema)]
struct GraphListResponse {
    graphs: Vec<String>,
}

#[utoipa::path(
    get,
    path = "/v1/graphs",
    responses((status = 200, description = "graph ids", body = GraphListResponse)),
    tag = "graphs",
)]
async fn list_graphs(State(state): State<AppState>) -> Json<GraphListResponse> {
    Json(GraphListResponse {
        graphs: state.registry.list_ids(),
    })
}

#[utoipa::path(
    post,
    path = "/v1/graphs",
    request_body = CreateGraphBody,
    responses(
        (status = 201, description = "graph created", body = SchemaResponse),
        (status = 400, description = "invalid request"),
        (status = 409, description = "graph already exists"),
    ),
    tag = "graphs",
)]
async fn create_graph(
    State(state): State<AppState>,
    Json(body): Json<CreateGraphBody>,
) -> Result<Response, RegistryError> {
    let CreateGraphBody { id, schema } = body;
    let entry = state.registry.create_graph(&id, schema.clone())?;
    let response = SchemaResponse::with_cached_hash(id, schema, entry.content_hash().to_string());
    Ok((StatusCode::CREATED, Json(response)).into_response())
}

/// Metadata-only — see `GET /v1/graphs/{id}/schema` for the full schema.
#[utoipa::path(
    get,
    path = "/v1/graphs/{id}",
    params(("id" = String, Path, description = "graph id")),
    responses(
        (status = 200, description = "graph metadata", body = GraphMetadataResponse),
        (status = 404, description = "graph not found"),
    ),
    tag = "graphs",
)]
async fn get_graph(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = GraphMetadataResponse::new(id, entry.content_hash().to_string());
    Ok(Json(response).into_response())
}

/// Full schema view: same shape consumed by generation_plus codegen
/// (matches the C-partial `build_schema_contract` output).
#[utoipa::path(
    get,
    path = "/v1/graphs/{id}/schema",
    params(("id" = String, Path, description = "graph id")),
    responses(
        (status = 200, description = "full schema", body = SchemaResponse),
        (status = 404, description = "graph not found"),
    ),
    tag = "graphs",
)]
async fn get_schema(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let schema = entry
        .with_engine_read(|engine| engine.schema().clone())
        .await;
    let response = SchemaResponse::with_cached_hash(id, schema, entry.content_hash().to_string());
    Ok(Json(response).into_response())
}

/// Replace a graph's schema. Compat rules + atomicity guarantees
/// live on `GraphRegistry::replace_schema`; this is a thin wrapper.
#[utoipa::path(
    put,
    path = "/v1/graphs/{id}/schema",
    params(("id" = String, Path, description = "graph id")),
    request_body = Object,
    responses(
        (status = 200, description = "schema replaced", body = SchemaResponse),
        (status = 400, description = "incompatible schema change"),
        (status = 404, description = "graph not found"),
    ),
    tag = "graphs",
)]
async fn replace_schema(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(new_schema): Json<Schema>,
) -> Result<Response, RegistryError> {
    let new_hash = state
        .registry
        .replace_schema(&id, new_schema.clone())
        .await?;
    let response = SchemaResponse::with_cached_hash(id, new_schema, new_hash.to_string());
    Ok(Json(response).into_response())
}

#[utoipa::path(
    delete,
    path = "/v1/graphs/{id}",
    params(("id" = String, Path, description = "graph id")),
    responses(
        (status = 204, description = "graph deleted"),
        (status = 404, description = "graph not found"),
    ),
    tag = "graphs",
)]
async fn delete_graph(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, RegistryError> {
    state.registry.delete_graph(&id)?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Deserialize, ToSchema)]
struct CreateNodeBody {
    node_type: String,
    node_id: String,
    #[serde(default)]
    #[schema(value_type = Object)]
    properties: HashMap<String, Value>,
}

#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/nodes",
    params(("id" = String, Path, description = "graph id")),
    request_body = CreateNodeBody,
    responses(
        (status = 201, description = "node created", body = NodeResponse),
        (status = 400, description = "invalid request / schema violation"),
        (status = 404, description = "graph not found"),
    ),
    tag = "nodes",
)]
async fn create_node(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<CreateNodeBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let CreateNodeBody {
        node_type,
        node_id,
        properties,
    } = body;
    let stored = entry
        .with_engine_write(move |engine| engine.create_node(&id, &node_type, &node_id, properties))
        .await?;
    Ok((StatusCode::CREATED, Json(NodeResponse::from(stored))).into_response())
}

#[derive(Debug, Deserialize, IntoParams)]
#[into_params(parameter_in = Query)]
struct ListNodesQuery {
    #[serde(rename = "type")]
    node_type: String,
    prop: Option<String>,
    value: Option<String>,
}

/// List nodes of a given type, optionally filtered by a single
/// (`prop`, `value`) pair. The pair must be supplied together — half
/// of it is a 400. `value` arrives as a URL string and is coerced to
/// the schema-declared `PropertyType` for the property; coerce
/// failures are 400, not silent zero-result.
#[utoipa::path(
    get,
    path = "/v1/graphs/{id}/nodes",
    params(
        ("id" = String, Path, description = "graph id"),
        ListNodesQuery,
    ),
    responses(
        (status = 200, description = "matching nodes", body = NodeListResponse),
        (status = 400, description = "invalid filter"),
        (status = 404, description = "graph not found"),
    ),
    tag = "nodes",
)]
async fn list_nodes(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(q): Query<ListNodesQuery>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let ListNodesQuery {
        node_type,
        prop,
        value,
    } = q;

    let nodes = entry
        .with_engine_read(move |engine| -> Result<Vec<_>, RegistryError> {
            match (prop, value) {
                (None, None) => engine
                    .scan_nodes(&id, &node_type)
                    .map_err(RegistryError::Storage),
                (Some(prop), Some(value)) => {
                    let coerced = coerce_query_value(engine.schema(), &node_type, &prop, &value)?;
                    engine
                        .scan_nodes_by_property(&id, &node_type, &prop, &coerced)
                        .map_err(RegistryError::Storage)
                }
                (Some(_), None) | (None, Some(_)) => Err(RegistryError::BadRequest(
                    "prop and value must be supplied together".to_string(),
                )),
            }
        })
        .await?;

    let response = NodeListResponse::new(nodes.into_iter().map(NodeResponse::from).collect());
    Ok(Json(response).into_response())
}

/// Coerce a URL-string `value` into a `Value` typed per the schema's
/// declaration of `node_type.prop`. Mirrors the indexable subset of
/// `PropertyType`s — `Float`/`ListString` aren't indexed by storage's
/// `scan_nodes_by_property`, so filtering by them is rejected up
/// front (400) rather than silently returning empty. `Enum` accepts
/// any string; storage validates it against `values` only on writes,
/// so a non-member enum filter cleanly returns no matches.
fn coerce_query_value(
    schema: &Schema,
    node_type: &str,
    prop: &str,
    value: &str,
) -> Result<Value, RegistryError> {
    let nt = schema
        .node_types
        .get(node_type)
        .ok_or_else(|| RegistryError::BadRequest(format!("unknown node type: {node_type}")))?;
    let pd = nt.properties.get(prop).ok_or_else(|| {
        RegistryError::BadRequest(format!("unknown property: {node_type}.{prop}"))
    })?;
    match pd.prop_type {
        PropertyType::String | PropertyType::Enum => Ok(Value::String(value.to_string())),
        PropertyType::Datetime => Ok(Value::String(value.to_string())),
        PropertyType::Int => value.parse::<i64>().map(Value::Int).map_err(|e| {
            RegistryError::BadRequest(format!("value {value:?} is not a valid int: {e}"))
        }),
        PropertyType::Bool => value.parse::<bool>().map(Value::Bool).map_err(|e| {
            RegistryError::BadRequest(format!("value {value:?} is not a valid bool: {e}"))
        }),
        // Float / ListString aren't equality-indexed; future PropertyType
        // variants without explicit coercion above also fall through here.
        _ => Err(RegistryError::BadRequest(format!(
            "filtering by {node_type}.{prop} is not supported (property type {:?} is not indexed)",
            pd.prop_type
        ))),
    }
}

#[utoipa::path(
    get,
    path = "/v1/graphs/{id}/nodes/{node_type}/{node_id}",
    params(
        ("id" = String, Path, description = "graph id"),
        ("node_type" = String, Path, description = "node type"),
        ("node_id" = String, Path, description = "node id"),
    ),
    responses(
        (status = 200, description = "the node", body = NodeResponse),
        (status = 404, description = "graph or node not found"),
    ),
    tag = "nodes",
)]
async fn get_node(
    State(state): State<AppState>,
    Path((id, node_type, node_id)): Path<(String, String, String)>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let stored = entry
        .with_engine_read({
            let node_type = node_type.clone();
            let node_id = node_id.clone();
            move |engine| engine.get_node(&id, &node_type, &node_id)
        })
        .await?
        .ok_or(RegistryError::NodeNotFound { node_type, node_id })?;
    Ok(Json(NodeResponse::from(stored)).into_response())
}

#[derive(Debug, Deserialize, ToSchema)]
struct ReplaceNodeBody {
    #[serde(default)]
    #[schema(value_type = Object)]
    properties: HashMap<String, Value>,
}

/// PUT semantics — full replacement of the node's property map (the
/// underlying storage call is `replace_node_properties`). PATCH is
/// not exposed because there is no merge primitive on nodes; if a
/// caller needs partial-update semantics they GET, mutate, PUT.
///
/// An empty or omitted `properties` (`{}`) is a deliberate full wipe —
/// PUT replaces the whole map, so `{}` clears every property. This is
/// the REST-correct reading of PUT and is intentionally NOT rejected;
/// callers that don't mean to clear should send the properties they
/// want to keep (or use the GET-mutate-PUT cycle above).
#[utoipa::path(
    put,
    path = "/v1/graphs/{id}/nodes/{node_type}/{node_id}",
    params(
        ("id" = String, Path, description = "graph id"),
        ("node_type" = String, Path, description = "node type"),
        ("node_id" = String, Path, description = "node id"),
    ),
    request_body = ReplaceNodeBody,
    responses(
        (status = 200, description = "node replaced", body = NodeResponse),
        (status = 400, description = "schema violation"),
        (status = 404, description = "graph or node not found"),
    ),
    tag = "nodes",
)]
async fn replace_node(
    State(state): State<AppState>,
    Path((id, node_type, node_id)): Path<(String, String, String)>,
    Json(body): Json<ReplaceNodeBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let stored = entry
        .with_engine_write({
            let node_type = node_type.clone();
            let node_id = node_id.clone();
            move |engine| engine.replace_node_properties(&id, &node_type, &node_id, body.properties)
        })
        .await?
        .ok_or(RegistryError::NodeNotFound { node_type, node_id })?;
    Ok(Json(NodeResponse::from(stored)).into_response())
}

#[utoipa::path(
    delete,
    path = "/v1/graphs/{id}/nodes/{node_type}/{node_id}",
    params(
        ("id" = String, Path, description = "graph id"),
        ("node_type" = String, Path, description = "node type"),
        ("node_id" = String, Path, description = "node id"),
    ),
    responses(
        (status = 204, description = "node deleted"),
        (status = 404, description = "graph or node not found"),
    ),
    tag = "nodes",
)]
async fn delete_node(
    State(state): State<AppState>,
    Path((id, node_type, node_id)): Path<(String, String, String)>,
) -> Result<StatusCode, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    // Storage's delete_node already cascades to drop the sidecar
    // embedding (slice 8a). The HNSW index is service-side state, so
    // we mirror the cascade here: if an index exists for this type,
    // remove the node from it. The whole cycle runs under one lock.
    let existed = entry
        .with_state_write({
            let node_type = node_type.clone();
            let node_id = node_id.clone();
            move |engine, indexes| -> Result<bool, RegistryError> {
                let existed = engine.delete_node(&id, &node_type, &node_id)?;
                if existed && let Some(index) = indexes.get_mut(&node_type) {
                    index.remove(&node_id);
                }
                Ok(existed)
            }
        })
        .await?;
    if existed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(RegistryError::NodeNotFound { node_type, node_id })
    }
}

#[derive(Debug, Deserialize, ToSchema)]
struct CreateEdgeBody {
    edge_type: String,
    from_type: String,
    from_id: String,
    to_type: String,
    to_id: String,
    #[serde(default)]
    #[schema(value_type = Object)]
    properties: HashMap<String, Value>,
}

#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/edges",
    params(("id" = String, Path, description = "graph id")),
    request_body = CreateEdgeBody,
    responses(
        (status = 201, description = "edge created", body = EdgeResponse),
        (status = 400, description = "invalid request / schema violation"),
        (status = 404, description = "graph not found"),
    ),
    tag = "edges",
)]
async fn create_edge(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<CreateEdgeBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let CreateEdgeBody {
        edge_type,
        from_type,
        from_id,
        to_type,
        to_id,
        properties,
    } = body;
    let stored = entry
        .with_engine_write(move |engine| {
            engine.create_edge(
                &id, &edge_type, &from_type, &from_id, &to_type, &to_id, properties,
            )
        })
        .await?;
    Ok((StatusCode::CREATED, Json(EdgeResponse::from(stored))).into_response())
}

#[utoipa::path(
    get,
    path = "/v1/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}",
    params(
        ("id" = String, Path, description = "graph id"),
        ("edge_type" = String, Path, description = "edge type"),
        ("from_id" = String, Path, description = "source node id"),
        ("to_id" = String, Path, description = "target node id"),
    ),
    responses(
        (status = 200, description = "the edge", body = EdgeResponse),
        (status = 404, description = "graph or edge not found"),
    ),
    tag = "edges",
)]
async fn get_edge(
    State(state): State<AppState>,
    Path((id, edge_type, from_id, to_id)): Path<(String, String, String, String)>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let stored = entry
        .with_engine_read({
            let edge_type = edge_type.clone();
            let from_id = from_id.clone();
            let to_id = to_id.clone();
            move |engine| engine.get_edge(&id, &edge_type, &from_id, &to_id)
        })
        .await?
        .ok_or(RegistryError::EdgeNotFound {
            edge_type,
            from_id,
            to_id,
        })?;
    Ok(Json(EdgeResponse::from(stored)).into_response())
}

#[derive(Debug, Deserialize, ToSchema)]
struct MergeEdgeBody {
    #[serde(default)]
    #[schema(value_type = Object)]
    properties: HashMap<String, Value>,
}

/// PATCH semantics — partial-update of the edge's property map. The
/// underlying storage call is `merge_edge_properties`. This mirrors
/// node CRUD's PUT (REPLACE) shape but with the verb flipped to match
/// the storage primitive's asymmetry — see `replace_node_properties`
/// docs for why nodes don't have a merge primitive.
#[utoipa::path(
    patch,
    path = "/v1/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}",
    params(
        ("id" = String, Path, description = "graph id"),
        ("edge_type" = String, Path, description = "edge type"),
        ("from_id" = String, Path, description = "source node id"),
        ("to_id" = String, Path, description = "target node id"),
    ),
    request_body = MergeEdgeBody,
    responses(
        (status = 200, description = "edge merged", body = EdgeResponse),
        (status = 400, description = "schema violation"),
        (status = 404, description = "graph or edge not found"),
    ),
    tag = "edges",
)]
async fn merge_edge(
    State(state): State<AppState>,
    Path((id, edge_type, from_id, to_id)): Path<(String, String, String, String)>,
    Json(body): Json<MergeEdgeBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let stored = entry
        .with_engine_write({
            let edge_type = edge_type.clone();
            let from_id = from_id.clone();
            let to_id = to_id.clone();
            move |engine| {
                engine.merge_edge_properties(&id, &edge_type, &from_id, &to_id, body.properties)
            }
        })
        .await?
        .ok_or(RegistryError::EdgeNotFound {
            edge_type,
            from_id,
            to_id,
        })?;
    Ok(Json(EdgeResponse::from(stored)).into_response())
}

#[utoipa::path(
    delete,
    path = "/v1/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}",
    params(
        ("id" = String, Path, description = "graph id"),
        ("edge_type" = String, Path, description = "edge type"),
        ("from_id" = String, Path, description = "source node id"),
        ("to_id" = String, Path, description = "target node id"),
    ),
    responses(
        (status = 204, description = "edge deleted"),
        (status = 404, description = "graph or edge not found"),
    ),
    tag = "edges",
)]
async fn delete_edge(
    State(state): State<AppState>,
    Path((id, edge_type, from_id, to_id)): Path<(String, String, String, String)>,
) -> Result<StatusCode, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let existed = entry
        .with_engine_write({
            let edge_type = edge_type.clone();
            let from_id = from_id.clone();
            let to_id = to_id.clone();
            move |engine| engine.delete_edge(&id, &edge_type, &from_id, &to_id)
        })
        .await?;
    if existed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(RegistryError::EdgeNotFound {
            edge_type,
            from_id,
            to_id,
        })
    }
}

/// Atomic multi-op transaction. See `crate::batch` for the wire shape
/// and design rationale. Whole batch runs under one `with_state_write`
/// lock so (a) ops and HNSW maintenance for any `delete_node` happen
/// in lockstep, and (b) concurrent readers either see pre-batch or
/// post-batch state, never a torn intermediate.
///
/// With `dry_run: true` the ops run against the batch buffer for a per-op
/// validation report and are then discarded — the graph is never mutated and
/// the response is a `BatchValidation` (HTTP 200 regardless of `valid`).
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/batch",
    params(("id" = String, Path, description = "graph id")),
    request_body = BatchRequest,
    responses(
        (status = 200, description = "commit summary, or a dry_run per-op report (incl. dry_run failures)", body = BatchOk),
        (status = 400, description = "request-shape error, or a commit-path per-op failure (dry_run failures are reported via 200)", body = BatchOpError),
        (status = 404, description = "graph not found"),
    ),
    tag = "primitives",
)]
async fn batch(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<BatchRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;

    if req.ops.is_empty() {
        return Err(RegistryError::BadRequest(
            "ops must be non-empty".to_string(),
        ));
    }
    if req.ops.len() > MAX_BATCH_OPS {
        return Err(RegistryError::BadRequest(format!(
            "ops length {} exceeds maximum {MAX_BATCH_OPS}",
            req.ops.len()
        )));
    }

    // Validate-only: run ops against the buffer, then discard. Always 200 —
    // the dry run itself succeeded; `valid` reports whether the ops would.
    if req.dry_run {
        let validation = entry
            .with_state_write(move |engine, _indexes| {
                engine.begin_batch();
                let validation = dry_run_ops(engine, &id, req.ops);
                engine.discard_batch();
                validation
            })
            .await;
        return Ok(Json(BatchOk::DryRun(validation)).into_response());
    }

    enum Outcome {
        Success(BatchResponse),
        OpFailed(BatchOpError),
        CommitFailed(dynograph_core::DynoError),
    }

    let outcome = entry
        .with_state_write(move |engine, indexes| -> Outcome {
            engine.begin_batch();
            match run_ops(engine, &id, req.ops) {
                Ok((response, deleted_nodes)) => match engine.commit_batch() {
                    Ok(_) => {
                        // HNSW maintenance for delete_node ops happens
                        // post-commit so a commit failure leaves the
                        // index untouched (matches the storage rollback).
                        for (node_type, node_id) in deleted_nodes {
                            if let Some(index) = indexes.get_mut(&node_type) {
                                index.remove(&node_id);
                            }
                        }
                        Outcome::Success(response)
                    }
                    Err(e) => Outcome::CommitFailed(e),
                },
                Err(per_op_err) => {
                    engine.discard_batch();
                    Outcome::OpFailed(per_op_err)
                }
            }
        })
        .await;

    match outcome {
        Outcome::Success(resp) => Ok(Json(BatchOk::Commit(resp)).into_response()),
        Outcome::OpFailed(err) => Ok((StatusCode::BAD_REQUEST, Json(err)).into_response()),
        Outcome::CommitFailed(e) => Err(RegistryError::Storage(e)),
    }
}

/// Fuzzy/vector entity resolution with create-on-miss. See
/// `crate::resolve_or_create` for wire shape, validation order, and
/// atomicity model. Whole call runs under one `with_state_write`
/// lock so candidate scan + resolve + (create + set_embedding +
/// HNSW insert) compose atomically.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/resolve-or-create",
    params(("id" = String, Path, description = "graph id")),
    request_body = ResolveOrCreateRequest,
    responses(
        (status = 200, description = "resolved or created", body = ResolveOrCreateResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
    ),
    tag = "primitives",
)]
async fn resolve_or_create(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<ResolveOrCreateRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_state_write(move |engine, indexes| run_resolve_or_create(engine, indexes, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Fan-out edge collection. See `crate::edges_collect` for wire
/// shape, validation, and the per-node-iteration cost model.
/// Read-only — single `with_engine_read` lock for the whole scan.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/edges:collect",
    params(("id" = String, Path, description = "graph id")),
    request_body = EdgesCollectRequest,
    responses(
        (status = 200, description = "collected edges (edges or adjacency form)", body = EdgesCollectResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
    ),
    tag = "primitives",
)]
async fn edges_collect(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<EdgesCollectRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| run_edges_collect(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Single-node 1-hop adjacency (the per-node `outgoing_edges(id)` /
/// `incoming_edges(id)` that `edges:collect` — fan-out by source type — does
/// not cover). See `crate::edges_adjacent`. Read-only.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/edges:adjacent",
    params(("id" = String, Path, description = "graph id")),
    request_body = EdgesAdjacentRequest,
    responses(
        (status = 200, description = "incident edges of one node", body = EdgesAdjacentResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
    ),
    tag = "primitives",
)]
async fn edges_adjacent(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<EdgesAdjacentRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| run_edges_adjacent(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Batch (type, name) existence check. See `crate::nodes_exists` for
/// wire shape and the indexed-`name` requirement. Read-only — one
/// `with_engine_read` lock for the whole batch so the per-query
/// scans see one consistent snapshot.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/nodes:exists",
    params(("id" = String, Path, description = "graph id")),
    request_body = NodesExistsRequest,
    responses(
        (status = 200, description = "per-query existence results", body = NodesExistsResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
    ),
    tag = "primitives",
)]
async fn nodes_exists(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<NodesExistsRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| run_nodes_exists(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Predicate-filtered scan over a single node type. See
/// `crate::nodes_scan` for wire shape, the seven supported ops, and
/// the seed-strategy/in-memory-filter design. Read-only — one
/// `with_engine_read` lock for the whole scan.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/nodes:scan",
    params(("id" = String, Path, description = "graph id")),
    request_body = NodesScanRequest,
    responses(
        (status = 200, description = "matching nodes or ids", body = NodesScanResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
    ),
    tag = "primitives",
)]
async fn nodes_scan(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<NodesScanRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| run_nodes_scan(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Atomic Welford-style EMA update of the score property family on
/// an existing edge. See `crate::welford_update` for the math, the
/// six-property convention, and pre-flight validation. Whole
/// read-modify-write runs under one `with_engine_write` lock —
/// concurrent updates serialize.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}/welford_update",
    params(
        ("id" = String, Path, description = "graph id"),
        ("edge_type" = String, Path, description = "edge type"),
        ("from_id" = String, Path, description = "source node id"),
        ("to_id" = String, Path, description = "target node id"),
    ),
    request_body = WelfordUpdateRequest,
    responses(
        (status = 200, description = "updated score statistics", body = WelfordUpdateResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph or edge not found"),
    ),
    tag = "primitives",
)]
async fn welford_update(
    State(state): State<AppState>,
    Path((id, edge_type, from_id, to_id)): Path<(String, String, String, String)>,
    Json(req): Json<WelfordUpdateRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_write(move |engine| {
            run_welford_update(engine, &id, &edge_type, &from_id, &to_id, req)
        })
        .await?;
    Ok(Json(response).into_response())
}

/// Typed BFS traversal from a start node. See `crate::traverse` for
/// wire shape, validation, and the multi-step / transitive
/// semantics. Read-only — single `with_engine_read` lock for the
/// whole BFS so the candidate scans + per-node edge walks see one
/// consistent snapshot.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/traverse",
    params(("id" = String, Path, description = "graph id")),
    request_body = TraverseRequest,
    responses(
        (status = 200, description = "traversed nodes", body = TraverseResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph or start node not found"),
    ),
    tag = "primitives",
)]
async fn traverse(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<TraverseRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| run_traverse(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// (Weakly-)connected components over the scoped subgraph. Read-only; the whole
/// build + analysis runs under one `with_engine_read` lock. Requires the `graph`
/// build feature; otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/components",
    params(("id" = String, Path, description = "graph id")),
    request_body = ScopedRequest,
    responses(
        (status = 200, description = "connected components", body = ComponentsResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_components(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<ScopedRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_components(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Degree centrality over the scoped subgraph. Read-only; runs under one
/// `with_engine_read` lock. Requires the `graph` build feature; otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/degree",
    params(("id" = String, Path, description = "graph id")),
    request_body = DegreeRequest,
    responses(
        (status = 200, description = "degree centrality scores", body = ScoresResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_degree(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<DegreeRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_degree(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// PageRank over the scoped subgraph (edge weights = strength). Read-only; runs
/// under one `with_engine_read` lock. Requires the `graph` build feature;
/// otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/pagerank",
    params(("id" = String, Path, description = "graph id")),
    request_body = PageRankRequest,
    responses(
        (status = 200, description = "PageRank scores", body = ScoresResponse),
        (status = 400, description = "validation error / did not converge"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_pagerank(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<PageRankRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_pagerank(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Eigenvector centrality over the scoped subgraph (edge weights = strength).
/// Read-only; runs under one `with_engine_read` lock. Requires the `graph` build
/// feature; otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/eigenvector",
    params(("id" = String, Path, description = "graph id")),
    request_body = EigenvectorRequest,
    responses(
        (status = 200, description = "eigenvector centrality scores", body = ScoresResponse),
        (status = 400, description = "validation error / undefined or did not converge"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_eigenvector(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<EigenvectorRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_eigenvector(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Closeness centrality over the scoped subgraph (edge weights = positive path
/// cost). Read-only; runs under one `with_engine_read` lock. Requires the
/// `graph` build feature; otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/closeness",
    params(("id" = String, Path, description = "graph id")),
    request_body = ClosenessRequest,
    responses(
        (status = 200, description = "closeness centrality scores", body = ScoresResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_closeness(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<ClosenessRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_closeness(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Betweenness centrality over the scoped subgraph (edge weights = positive path
/// cost). Read-only; runs under one `with_engine_read` lock. Requires the
/// `graph` build feature; otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/betweenness",
    params(("id" = String, Path, description = "graph id")),
    request_body = BetweennessRequest,
    responses(
        (status = 200, description = "betweenness centrality scores", body = ScoresResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_betweenness(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<BetweennessRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_betweenness(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Articulation points and bridges of the scoped subgraph (treated undirected).
/// Read-only; runs under one `with_engine_read` lock. Requires the `graph` build
/// feature; otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/cuts",
    params(("id" = String, Path, description = "graph id")),
    request_body = ScopedRequest,
    responses(
        (status = 200, description = "articulation points + bridges", body = CutsResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_cuts(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<ScopedRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_cuts(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Strongly-connected components of the scoped directed subgraph. Read-only;
/// runs under one `with_engine_read` lock. Requires the `graph` build feature;
/// otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/scc",
    params(("id" = String, Path, description = "graph id")),
    request_body = ScopedRequest,
    responses(
        (status = 200, description = "strongly-connected components", body = ComponentsResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_scc(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<ScopedRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_scc(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Directed cycle detection: is the scoped subgraph acyclic, plus one witness
/// cycle if not. Read-only; one `with_engine_read` lock. Requires the `graph`
/// build feature; otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/cycles",
    params(("id" = String, Path, description = "graph id")),
    request_body = ScopedRequest,
    responses(
        (status = 200, description = "acyclic flag + witness cycle", body = CyclesResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_cycles(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<ScopedRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_cycles(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Personalized PageRank (random walk with restart) over the scoped subgraph,
/// seeded by node ids. Read-only; one `with_engine_read` lock. Requires the
/// `graph` build feature; otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/personalized_pagerank",
    params(("id" = String, Path, description = "graph id")),
    request_body = PersonalizedPageRankRequest,
    responses(
        (status = 200, description = "relevance scores", body = ScoresResponse),
        (status = 400, description = "validation error / did not converge"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_personalized_pagerank(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<PersonalizedPageRankRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_personalized_pagerank(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// One shortest path between two nodes of the scoped subgraph. Read-only; one
/// `with_engine_read` lock. Requires the `graph` build feature; otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/shortest_path",
    params(("id" = String, Path, description = "graph id")),
    request_body = SourceTargetRequest,
    responses(
        (status = 200, description = "shortest path + distance", body = ShortestPathResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_shortest_path(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<SourceTargetRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_shortest_path(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Link prediction over the scoped (undirected) subgraph — ranked candidate
/// edges by neighborhood overlap. Read-only; one `with_engine_read` lock.
/// Requires the `graph` build feature; otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/link_prediction",
    params(("id" = String, Path, description = "graph id")),
    request_body = LinkPredictionRequest,
    responses(
        (status = 200, description = "ranked predicted links", body = LinkPredictionResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_link_prediction(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<LinkPredictionRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_link_prediction(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Local clustering coefficients + global transitivity over the scoped
/// (undirected) subgraph. Read-only; one `with_engine_read` lock. Requires the
/// `graph` build feature; otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/clustering",
    params(("id" = String, Path, description = "graph id")),
    request_body = ScopedRequest,
    responses(
        (status = 200, description = "clustering coefficients + transitivity", body = ClusteringResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_clustering(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<ScopedRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_clustering(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Louvain community detection over the scoped undirected subgraph: returns the
/// partition (communities as id lists) plus its modularity. Read-only; one
/// `with_engine_read` lock. Requires the `graph` build feature; otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/communities",
    params(("id" = String, Path, description = "graph id")),
    request_body = CommunitiesRequest,
    responses(
        (status = 200, description = "communities + modularity", body = CommunitiesResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_communities(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<CommunitiesRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_communities(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Topological order of the scoped directed subgraph (or a not-acyclic flag).
/// Read-only; one `with_engine_read` lock. Requires the `graph` build feature;
/// otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/toposort",
    params(("id" = String, Path, description = "graph id")),
    request_body = ScopedRequest,
    responses(
        (status = 200, description = "topological order", body = ToposortResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_toposort(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<ScopedRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_toposort(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

/// Maximum flow and minimum s-t cut over the scoped subgraph (edge weights =
/// capacities). Read-only; one `with_engine_read` lock. Requires the `graph`
/// build feature; otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/algo/max_flow",
    params(("id" = String, Path, description = "graph id")),
    request_body = SourceTargetRequest,
    responses(
        (status = 200, description = "max flow value + min cut", body = MaxFlowResponse),
        (status = 400, description = "validation error"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "graph feature not enabled in this build"),
    ),
    tag = "algo",
)]
async fn algo_max_flow(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<SourceTargetRequest>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_engine_read(move |engine| crate::algo::run_max_flow(engine, &id, req))
        .await?;
    Ok(Json(response).into_response())
}

#[derive(Debug, Deserialize, ToSchema)]
struct SetEmbeddingBody {
    embedding: Vec<f32>,
}

/// Set an embedding and update the per-type HNSW index in lockstep.
/// Preflight order matters: dim check against any existing index
/// runs *before* the storage write, so a mismatch rejects without
/// on-disk rollback. Per-type dim is locked at first insert.
#[utoipa::path(
    put,
    path = "/v1/graphs/{id}/nodes/{node_type}/{node_id}/embedding",
    params(
        ("id" = String, Path, description = "graph id"),
        ("node_type" = String, Path, description = "node type"),
        ("node_id" = String, Path, description = "node id"),
    ),
    request_body = SetEmbeddingBody,
    responses(
        (status = 200, description = "embedding set", body = EmbeddingResponse),
        (status = 400, description = "dimension mismatch / invalid request"),
        (status = 404, description = "graph or node not found"),
    ),
    tag = "embeddings",
)]
async fn set_embedding(
    State(state): State<AppState>,
    Path((id, node_type, node_id)): Path<(String, String, String)>,
    Json(body): Json<SetEmbeddingBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let SetEmbeddingBody { embedding } = body;
    // Reject degenerate embeddings (non-finite / zero magnitude) before
    // any storage write or index insert — a silent 0.0-against-everything
    // vector must never enter the index. 400, no on-disk rollback needed.
    validate_embedding_values(&embedding)?;
    // Build the response inside the closure and hand it back: this
    // moves `node_type`/`node_id`/`embedding` into the blocking task
    // (required by `Send + 'static`) without cloning the embedding
    // vector back out afterward.
    let response = entry
        .with_state_write(
            move |engine, indexes| -> Result<EmbeddingResponse, RegistryError> {
                if let Some(index) = indexes.get(&node_type)
                    && index.dim() != embedding.len()
                {
                    return Err(RegistryError::EmbeddingDimMismatch {
                        node_type: node_type.clone(),
                        expected: index.dim(),
                        actual: embedding.len(),
                    });
                }
                engine.set_embedding(&id, &node_type, &node_id, &embedding)?;
                // Avoid the `entry()` clone on the hot post-first-insert
                // path: only allocate the key when we actually need to
                // insert. After the first insert per type, this is a single
                // get_mut.
                let index = match indexes.get_mut(&node_type) {
                    Some(i) => i,
                    None => indexes
                        .entry(node_type.clone())
                        .or_insert_with(|| HnswIndex::with_dim(embedding.len())),
                };
                index.insert(&node_id, &embedding);
                Ok(EmbeddingResponse {
                    node_type,
                    node_id,
                    embedding,
                })
            },
        )
        .await?;
    Ok(Json(response).into_response())
}

#[utoipa::path(
    get,
    path = "/v1/graphs/{id}/nodes/{node_type}/{node_id}/embedding",
    params(
        ("id" = String, Path, description = "graph id"),
        ("node_type" = String, Path, description = "node type"),
        ("node_id" = String, Path, description = "node id"),
    ),
    responses(
        (status = 200, description = "the embedding", body = EmbeddingResponse),
        (status = 404, description = "graph or embedding not found"),
    ),
    tag = "embeddings",
)]
async fn get_embedding(
    State(state): State<AppState>,
    Path((id, node_type, node_id)): Path<(String, String, String)>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let embedding = entry
        .with_engine_read({
            let node_type = node_type.clone();
            let node_id = node_id.clone();
            move |engine| engine.get_embedding(&id, &node_type, &node_id)
        })
        .await?
        .ok_or_else(|| RegistryError::EmbeddingNotFound {
            node_type: node_type.clone(),
            node_id: node_id.clone(),
        })?;
    let response = EmbeddingResponse {
        node_type,
        node_id,
        embedding,
    };
    Ok(Json(response).into_response())
}

#[utoipa::path(
    delete,
    path = "/v1/graphs/{id}/nodes/{node_type}/{node_id}/embedding",
    params(
        ("id" = String, Path, description = "graph id"),
        ("node_type" = String, Path, description = "node type"),
        ("node_id" = String, Path, description = "node id"),
    ),
    responses(
        (status = 204, description = "embedding deleted"),
        (status = 404, description = "graph or embedding not found"),
    ),
    tag = "embeddings",
)]
async fn delete_embedding(
    State(state): State<AppState>,
    Path((id, node_type, node_id)): Path<(String, String, String)>,
) -> Result<StatusCode, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let existed = entry
        .with_state_write({
            let node_type = node_type.clone();
            let node_id = node_id.clone();
            move |engine, indexes| -> Result<bool, RegistryError> {
                let existed = engine.delete_embedding(&id, &node_type, &node_id)?;
                if existed && let Some(index) = indexes.get_mut(&node_type) {
                    index.remove(&node_id);
                }
                Ok(existed)
            }
        })
        .await?;
    if existed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(RegistryError::EmbeddingNotFound { node_type, node_id })
    }
}

#[derive(Debug, Deserialize, ToSchema)]
struct SimilarBody {
    embedding: Vec<f32>,
    top_k: usize,
    node_type: String,
}

/// HNSW vector search over the per-type index. `node_type` is
/// required: per-type indexes can have different dimensions (set
/// independently by the first `set_embedding` for each type), so a
/// merged "search all types" answer is ambiguous about score
/// comparability. If a real consumer needs cross-type search later,
/// add it as an explicit second route.
///
/// If no index exists for `node_type` (no embedding has ever been
/// set for any node of that type), returns an empty result list —
/// the type-name is honest, just no data to search yet.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/similar",
    params(("id" = String, Path, description = "graph id")),
    request_body = SimilarBody,
    responses(
        (status = 200, description = "nearest neighbors", body = SimilarResponse),
        (status = 400, description = "dimension mismatch / invalid request"),
        (status = 404, description = "graph not found"),
    ),
    tag = "search",
)]
async fn similar(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<SimilarBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let SimilarBody {
        embedding,
        top_k,
        node_type,
    } = body;
    if embedding.is_empty() {
        return Err(RegistryError::BadRequest(
            "embedding must be non-empty".to_string(),
        ));
    }
    // A degenerate query (non-finite / zero magnitude) scores 0.0
    // against every node — a silent "nothing is similar" that hides bad
    // upstream data. Reject loudly.
    validate_embedding_values(&embedding)?;
    // Cap `top_k` at MAX_LIMIT (and reject 0) like every other
    // result-bearing route — an unbounded `top_k` is a DoS/OOM vector
    // (the HNSW search collects up to that many hits).
    validate_limit(top_k, "top_k")?;
    let response = entry
        .with_state_read(
            move |engine, indexes| -> Result<SimilarResponse, RegistryError> {
                if !engine.schema().node_types.contains_key(&node_type) {
                    return Err(RegistryError::BadRequest(format!(
                        "unknown node type: {node_type}"
                    )));
                }
                let Some(index) = indexes.get(&node_type) else {
                    return Ok(SimilarResponse {
                        results: Vec::new(),
                    });
                };
                if index.dim() != embedding.len() {
                    return Err(RegistryError::EmbeddingDimMismatch {
                        node_type: node_type.clone(),
                        expected: index.dim(),
                        actual: embedding.len(),
                    });
                }
                let results = index
                    .search(&embedding, top_k)
                    .into_iter()
                    .map(|sr| SimilarHit {
                        node_id: sr.id.to_string(),
                        score: sr.score,
                    })
                    .collect();
                Ok(SimilarResponse { results })
            },
        )
        .await?;
    Ok(Json(response).into_response())
}

// =========================================================================
// /v1/graphs/{id}/search:* — full-text (BM25) search.
//
// Behind the `fulltext` cargo feature. The routes are always registered; when
// the feature is off the handlers return 501 so the API surface is stable and
// the OpenAPI spec is identical across builds.
// =========================================================================

fn default_search_limit() -> usize {
    10
}

#[derive(Debug, Deserialize, ToSchema)]
// In a build without the `fulltext` feature the handler returns 501 and never
// reads these fields — but they're still deserialized from the request and are
// part of the published wire/OpenAPI contract, so the "unread" lint is a false
// positive there.
#[cfg_attr(not(feature = "fulltext"), allow(dead_code))]
struct SearchTextBody {
    /// Raw keyword query. Tokenized with the index analyzer and matched as a
    /// conjunction (every token must occur). Punctuation and `field:value`
    /// input are treated as plain text, not query syntax.
    query: String,
    /// Optional node-type filter; omit to search all types in the graph.
    #[serde(default)]
    node_type: Option<String>,
    /// Max hits to return (1..=MAX_LIMIT). Defaults to 10.
    // `usize` defaults to `minimum: 0` in the generated schema, which
    // contradicts the `validate_limit` 1..=MAX_LIMIT runtime bound. Pin the
    // advertised bounds so the OpenAPI contract matches the endpoint. The
    // literal `10_000` mirrors `validation::MAX_LIMIT` — utoipa's `#[schema]`
    // can't reference a const, so keep the two in sync by hand.
    #[schema(minimum = 1, maximum = 10_000)]
    #[serde(default = "default_search_limit")]
    limit: usize,
}

#[derive(Debug, Serialize, ToSchema)]
struct SearchTextHit {
    node_id: String,
    node_type: String,
    score: f32,
}

#[derive(Debug, Serialize, ToSchema)]
struct SearchTextResponse {
    results: Vec<SearchTextHit>,
}

#[derive(Debug, Serialize, ToSchema)]
struct SearchReindexResponse {
    indexed: usize,
}

/// BM25 keyword search over the graph's full-text index. Scoped to the graph,
/// optionally to one `node_type`. Returns hits highest-score-first. Requires
/// the `fulltext` build feature; otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/search:text",
    params(("id" = String, Path, description = "graph id")),
    request_body = SearchTextBody,
    responses(
        (status = 200, description = "BM25 keyword hits", body = SearchTextResponse),
        (status = 400, description = "invalid request"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "full-text feature not enabled in this build"),
    ),
    tag = "search",
)]
async fn search_text(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<SearchTextBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    search_text_impl(entry, id, body).await
}

#[cfg(feature = "fulltext")]
async fn search_text_impl(
    entry: Arc<GraphEntry>,
    id: String,
    body: SearchTextBody,
) -> Result<Response, RegistryError> {
    // Cap `limit` like every other result-bearing route — an unbounded limit
    // is a DoS/OOM vector (the search collects up to that many hits).
    validate_limit(body.limit, "limit")?;
    let SearchTextBody {
        query,
        node_type,
        limit,
    } = body;
    let response = entry
        .with_engine_read(move |engine| -> Result<SearchTextResponse, RegistryError> {
            // Fail loud on a node_type filter that can never match — an unknown
            // type, or a known type with no `fulltext` property. Mirrors
            // `similar` / `nodes_scan` rather than masking a client typo as an
            // empty result.
            if let Some(nt) = &node_type {
                crate::validation::validate_fulltext_searchable(engine.schema(), nt)?;
            }
            let hits = engine.search_fulltext(&id, &query, node_type.as_deref(), limit)?;
            Ok(SearchTextResponse {
                results: hits
                    .into_iter()
                    .map(|h| SearchTextHit {
                        node_id: h.node_id,
                        node_type: h.node_type,
                        score: h.score,
                    })
                    .collect(),
            })
        })
        .await?;
    Ok(Json(response).into_response())
}

#[cfg(not(feature = "fulltext"))]
async fn search_text_impl(
    _entry: Arc<GraphEntry>,
    _id: String,
    _body: SearchTextBody,
) -> Result<Response, RegistryError> {
    Err(RegistryError::NotImplemented(
        "full-text search is not enabled in this build (compile with --features fulltext)"
            .to_string(),
    ))
}

/// Hybrid search: Reciprocal-Rank-Fusion over the vector (HNSW) and
/// keyword (BM25) legs, with an optional structured `where` prefilter. See
/// `crate::search_hybrid` for the wire shape, fusion math, and per-leg
/// design decisions. Read-only — one `with_state_read` lock exposes both
/// the engine and the per-type HNSW indexes the two legs need. At least one
/// ranked leg is required (`query` and/or `query_vector`) — a pure `where`
/// filter with no ranked leg is a 400 (that's what `nodes:scan` is for). The
/// keyword leg requires the `fulltext` build feature (otherwise 501); a
/// vector leg (optionally with a `where` prefilter) succeeds in any build.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/search:hybrid",
    params(("id" = String, Path, description = "graph id")),
    request_body = SearchHybridBody,
    responses(
        (status = 200, description = "fused, ranked hits", body = SearchHybridResponse),
        (status = 400, description = "invalid request"),
        (status = 404, description = "graph not found"),
        (status = 501, description = "keyword leg requested but full-text feature not enabled"),
    ),
    tag = "search",
)]
async fn search_hybrid(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<SearchHybridBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = entry
        .with_state_read(move |engine, indexes| run_search_hybrid(engine, indexes, &id, body))
        .await?;
    Ok(Json(response).into_response())
}

/// Rebuild the graph's full-text index from the authoritative node store. Admin
/// op — clears then re-indexes every `fulltext` node. Requires the `fulltext`
/// build feature; otherwise 501.
#[utoipa::path(
    post,
    path = "/v1/graphs/{id}/search:reindex",
    params(("id" = String, Path, description = "graph id")),
    responses(
        (status = 200, description = "nodes reindexed", body = SearchReindexResponse),
        (status = 404, description = "graph not found"),
        (status = 501, description = "full-text feature not enabled in this build"),
    ),
    tag = "search",
)]
async fn search_reindex(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    search_reindex_impl(entry, id).await
}

#[cfg(feature = "fulltext")]
async fn search_reindex_impl(
    entry: Arc<GraphEntry>,
    id: String,
) -> Result<Response, RegistryError> {
    // Write lock: serializes the rebuild against node writes and any concurrent
    // reindex so the clear-then-rebuild can't interleave.
    let response = entry
        .with_engine_write(
            move |engine| -> Result<SearchReindexResponse, RegistryError> {
                let indexed = engine.reindex_fulltext(&id)?;
                Ok(SearchReindexResponse { indexed })
            },
        )
        .await?;
    Ok(Json(response).into_response())
}

#[cfg(not(feature = "fulltext"))]
async fn search_reindex_impl(
    _entry: Arc<GraphEntry>,
    _id: String,
) -> Result<Response, RegistryError> {
    Err(RegistryError::NotImplemented(
        "full-text reindex is not enabled in this build (compile with --features fulltext)"
            .to_string(),
    ))
}

// =========================================================================
// /v1/util/* — pure-math utility endpoints.
//
// Stateless (no graph_id, no registry access). Each handler is a thin
// adapter over `crate::util::run_*`; the math lives there and stays
// trivially unit-testable.
// =========================================================================

#[utoipa::path(
    post,
    path = "/v1/util/cosine_similarity",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "cosine similarity", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_cosine_similarity(
    Json(req): Json<BinaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_cosine_similarity(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/dot_product",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "dot product", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_dot_product(Json(req): Json<BinaryVectorRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_dot_product(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/euclidean_distance",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "euclidean distance", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_euclidean_distance(
    Json(req): Json<BinaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_euclidean_distance(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/l2_norm",
    request_body = UnaryVectorRequest,
    responses(
        (status = 200, description = "L2 norm", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_l2_norm(Json(req): Json<UnaryVectorRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_l2_norm(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/hadamard",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "elementwise product", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_hadamard(Json(req): Json<BinaryVectorRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_hadamard(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/pearson_correlation",
    request_body = TwoVectorF64Request,
    responses(
        (status = 200, description = "Pearson correlation", body = ScalarResponse<f64>),
        (status = 400, description = "validation error / undefined"),
    ),
    tag = "util",
)]
async fn util_pearson_correlation(
    Json(req): Json<TwoVectorF64Request>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_pearson_correlation(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/linear_regression_slope",
    request_body = PointsRequest,
    responses(
        (status = 200, description = "regression slope", body = ScalarResponse<f64>),
        (status = 400, description = "validation error / undefined"),
    ),
    tag = "util",
)]
async fn util_linear_regression_slope(
    Json(req): Json<PointsRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_linear_regression_slope(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/jaro_winkler",
    request_body = BinaryStringRequest,
    responses(
        (status = 200, description = "Jaro-Winkler similarity (0..=100)", body = ScalarResponse<u32>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_jaro_winkler(
    Json(req): Json<BinaryStringRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_jaro_winkler(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/token_sort_ratio",
    request_body = BinaryStringRequest,
    responses(
        (status = 200, description = "token-sort ratio (0..=100)", body = ScalarResponse<u32>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_token_sort_ratio(
    Json(req): Json<BinaryStringRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_token_sort_ratio(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/squared_euclidean_distance",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "squared euclidean distance", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_squared_euclidean_distance(
    Json(req): Json<BinaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_squared_euclidean_distance(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/manhattan_distance",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "manhattan (L1) distance", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_manhattan_distance(
    Json(req): Json<BinaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_manhattan_distance(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/add",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "element-wise sum", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_add(Json(req): Json<BinaryVectorRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_add(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/subtract",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "element-wise difference", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_subtract(Json(req): Json<BinaryVectorRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_subtract(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/scale",
    request_body = ScaleRequest,
    responses(
        (status = 200, description = "scalar multiple", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_scale(Json(req): Json<ScaleRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_scale(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/negate",
    request_body = UnaryVectorRequest,
    responses(
        (status = 200, description = "negated vector", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_negate(Json(req): Json<UnaryVectorRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_negate(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/hadamard_division",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "element-wise quotient", body = VectorResponse),
        (status = 400, description = "validation error / zero divisor"),
    ),
    tag = "util",
)]
async fn util_hadamard_division(
    Json(req): Json<BinaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_hadamard_division(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/elementwise_power",
    request_body = PowerRequest,
    responses(
        (status = 200, description = "element-wise power", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_elementwise_power(Json(req): Json<PowerRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_elementwise_power(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/l2_normalize",
    request_body = UnaryVectorRequest,
    responses(
        (status = 200, description = "unit-length vector", body = VectorResponse),
        (status = 400, description = "validation error / zero magnitude"),
    ),
    tag = "util",
)]
async fn util_l2_normalize(Json(req): Json<UnaryVectorRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_l2_normalize(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/centroid",
    request_body = VectorsRequest,
    responses(
        (status = 200, description = "component-wise mean vector", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_centroid(Json(req): Json<VectorsRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_centroid(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/pairwise_cosine",
    request_body = VectorsRequest,
    responses(
        (status = 200, description = "N×N cosine-similarity matrix", body = MatrixResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_pairwise_cosine(Json(req): Json<VectorsRequest>) -> Result<Response, RegistryError> {
    // O(N²·dim) — offload to the blocking pool so a large matrix can't pin a
    // tokio worker (the timeout layer can't preempt CPU-bound work). Mirrors how
    // the storage/engine wrappers run their synchronous work; a panic in the
    // task is re-raised here, a validation error becomes a 400.
    let resp = tokio::task::spawn_blocking(move || run_pairwise_cosine(req))
        .await
        .unwrap_or_else(|e| std::panic::resume_unwind(e.into_panic()))?;
    Ok(Json(resp).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/pairwise_distance",
    request_body = PairwiseDistanceRequest,
    responses(
        (status = 200, description = "N×N distance matrix under the chosen metric", body = MatrixResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_pairwise_distance(
    Json(req): Json<PairwiseDistanceRequest>,
) -> Result<Response, RegistryError> {
    // O(N²·dim) — offload to the blocking pool (see `util_pairwise_cosine`).
    let resp = tokio::task::spawn_blocking(move || run_pairwise_distance(req))
        .await
        .unwrap_or_else(|e| std::panic::resume_unwind(e.into_panic()))?;
    Ok(Json(resp).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/mean",
    request_body = SampleRequest,
    responses(
        (status = 200, description = "arithmetic mean", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_mean(Json(req): Json<SampleRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_mean(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/variance",
    request_body = SampleRequest,
    responses(
        (status = 200, description = "sample variance (n-1)", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_variance(Json(req): Json<SampleRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_variance(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/std_dev",
    request_body = SampleRequest,
    responses(
        (status = 200, description = "sample standard deviation", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_std_dev(Json(req): Json<SampleRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_std_dev(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/median",
    request_body = SampleRequest,
    responses(
        (status = 200, description = "median (50th percentile)", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_median(Json(req): Json<SampleRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_median(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/percentile",
    request_body = PercentileRequest,
    responses(
        (status = 200, description = "linear-interpolated percentile", body = ScalarResponse<f64>),
        (status = 400, description = "validation error / p out of range"),
    ),
    tag = "util",
)]
async fn util_percentile(Json(req): Json<PercentileRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_percentile(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/softmax",
    request_body = SampleRequest,
    responses(
        (status = 200, description = "softmax distribution (sums to 1)", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
async fn util_softmax(Json(req): Json<SampleRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_softmax(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/spearman_correlation",
    request_body = TwoVectorF64Request,
    responses(
        (status = 200, description = "Spearman rank correlation", body = ScalarResponse<f64>),
        (status = 400, description = "validation error / undefined"),
    ),
    tag = "util",
)]
async fn util_spearman_correlation(
    Json(req): Json<TwoVectorF64Request>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_spearman_correlation(req)?).into_response())
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
