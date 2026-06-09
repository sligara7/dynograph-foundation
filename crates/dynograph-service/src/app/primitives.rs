//! `primitives — batch / resolve / edges:* / nodes:* / traverse / welford` route handlers — split out of `app.rs`. `use super::*`
//! inherits the shared imports and helpers (`AppState`, `graph_entry`,
//! the `crate::*` wire types) from the parent `app` module.

use super::*;

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
pub(crate) async fn batch(
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
pub(crate) async fn resolve_or_create(
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
pub(crate) async fn edges_collect(
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
pub(crate) async fn edges_adjacent(
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
pub(crate) async fn nodes_exists(
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
pub(crate) async fn nodes_scan(
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
pub(crate) async fn welford_update(
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
pub(crate) async fn traverse(
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
