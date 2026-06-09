//! `nodes — node CRUD` route handlers — split out of `app.rs`. `use super::*`
//! inherits the shared imports and helpers (`AppState`, `graph_entry`,
//! the `crate::*` wire types) from the parent `app` module.

use super::*;

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct CreateNodeBody {
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
pub(crate) async fn create_node(
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
pub(crate) struct ListNodesQuery {
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
pub(crate) async fn list_nodes(
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
pub(crate) async fn get_node(
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
pub(crate) struct ReplaceNodeBody {
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
pub(crate) async fn replace_node(
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
pub(crate) async fn delete_node(
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
