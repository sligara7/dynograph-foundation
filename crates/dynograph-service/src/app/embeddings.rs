//! `embeddings — per-node embeddings` route handlers — split out of `app.rs`. `use super::*`
//! inherits the shared imports and helpers (`AppState`, `graph_entry`,
//! the `crate::*` wire types) from the parent `app` module.

use super::*;

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct SetEmbeddingBody {
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
pub(crate) async fn set_embedding(
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
pub(crate) async fn get_embedding(
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
pub(crate) async fn delete_embedding(
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
