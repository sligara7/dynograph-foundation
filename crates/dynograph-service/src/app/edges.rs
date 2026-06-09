//! `edges — edge CRUD` route handlers — split out of `app.rs`. `use super::*`
//! inherits the shared imports and helpers (`AppState`, `graph_entry`,
//! the `crate::*` wire types) from the parent `app` module.

use super::*;

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct CreateEdgeBody {
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
pub(crate) async fn create_edge(
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
pub(crate) async fn get_edge(
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
pub(crate) struct MergeEdgeBody {
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
pub(crate) async fn merge_edge(
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
pub(crate) async fn delete_edge(
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
