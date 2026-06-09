//! `graphs — graph lifecycle + schema` route handlers — split out of `app.rs`. `use super::*`
//! inherits the shared imports and helpers (`AppState`, `graph_entry`,
//! the `crate::*` wire types) from the parent `app` module.

use super::*;

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct CreateGraphBody {
    id: String,
    #[schema(value_type = Object)]
    schema: Schema,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct GraphListResponse {
    graphs: Vec<String>,
}

#[utoipa::path(
    get,
    path = "/v1/graphs",
    responses((status = 200, description = "graph ids", body = GraphListResponse)),
    tag = "graphs",
)]
pub(crate) async fn list_graphs(State(state): State<AppState>) -> Json<GraphListResponse> {
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
pub(crate) async fn create_graph(
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
pub(crate) async fn get_graph(
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
pub(crate) async fn get_schema(
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
pub(crate) async fn replace_schema(
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
pub(crate) async fn delete_graph(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, RegistryError> {
    state.registry.delete_graph(&id)?;
    Ok(StatusCode::NO_CONTENT)
}
