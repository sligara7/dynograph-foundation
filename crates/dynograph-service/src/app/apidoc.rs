//! The OpenAPI 3.1 document aggregator (`ApiDoc`) and the cross-cutting
//! `AuthResponses` modifier. Split out of `app.rs`. `use super::*` pulls in
//! the handler `__path` items and every `ToSchema` wire type (handler-local
//! bodies via the parent's handler-module globs; imported types directly),
//! so the `paths(...)`/`schemas(...)` lists resolve by bare name as before.

use super::*;

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
        util_game_analyze,
        util_dbscan,
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
        // game theory
        GameAnalyzeRequest,
        PlayerSpec,
        CellSpec,
        GameAnalyzeResponse,
        DominantInfo,
        NashDominationInfo,
        Mixed2x2Info,
        // clustering
        DbscanRequest,
        DbscanResponse,
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
