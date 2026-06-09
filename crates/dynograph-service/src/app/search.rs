//! `search — vector similarity + full-text` route handlers — split out of `app.rs`. `use super::*`
//! inherits the shared imports and helpers (`AppState`, `graph_entry`,
//! the `crate::*` wire types) from the parent `app` module.

use super::*;

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct SimilarBody {
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
pub(crate) async fn similar(
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
pub(crate) struct SearchTextBody {
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
pub(crate) struct SearchTextHit {
    node_id: String,
    node_type: String,
    score: f32,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct SearchTextResponse {
    results: Vec<SearchTextHit>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct SearchReindexResponse {
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
pub(crate) async fn search_text(
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
pub(crate) async fn search_hybrid(
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
pub(crate) async fn search_reindex(
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
