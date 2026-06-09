//! `algo — graph-theory algorithms` route handlers — split out of `app.rs`. `use super::*`
//! inherits the shared imports and helpers (`AppState`, `graph_entry`,
//! the `crate::*` wire types) from the parent `app` module.

use super::*;

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
pub(crate) async fn algo_components(
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
pub(crate) async fn algo_degree(
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
pub(crate) async fn algo_pagerank(
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
pub(crate) async fn algo_eigenvector(
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
pub(crate) async fn algo_closeness(
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
pub(crate) async fn algo_betweenness(
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
pub(crate) async fn algo_cuts(
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
pub(crate) async fn algo_scc(
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
pub(crate) async fn algo_cycles(
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
pub(crate) async fn algo_personalized_pagerank(
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
pub(crate) async fn algo_shortest_path(
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
pub(crate) async fn algo_link_prediction(
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
pub(crate) async fn algo_clustering(
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
pub(crate) async fn algo_communities(
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
pub(crate) async fn algo_toposort(
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
pub(crate) async fn algo_max_flow(
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
