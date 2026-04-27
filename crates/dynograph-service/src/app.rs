//! axum app + route handlers.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use dynograph_core::{PropertyType, Schema, Value};

use crate::{
    auth::{AuthProvider, NoAuth},
    edge_response::EdgeResponse,
    metadata_response::GraphMetadataResponse,
    node_response::{NodeListResponse, NodeResponse},
    readiness::Readiness,
    registry::{GraphEntry, GraphRegistry, RegistryError},
    schema_response::SchemaResponse,
};

#[derive(Clone)]
pub struct AppState {
    pub(crate) registry: Arc<GraphRegistry>,
    /// Plumbed through now; first handler that needs an `Identity`
    /// (any non-public route) will call `auth.authenticate(&headers)`.
    #[allow(dead_code)]
    pub(crate) auth: Arc<dyn AuthProvider>,
    pub(crate) readiness: Arc<Readiness>,
}

impl AppState {
    pub fn new(
        registry: Arc<GraphRegistry>,
        auth: Arc<dyn AuthProvider>,
        readiness: Arc<Readiness>,
    ) -> Self {
        Self {
            registry,
            auth,
            readiness,
        }
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
    Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
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
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

/// Readiness probe — distinct from `/health`, which only confirms
/// the process is running. `/ready` returns 200 once the service
/// has finished startup work (notably `rehydrate()` on the on-disk
/// backend); 503 before that.
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
    state
        .registry
        .get(id)
        .ok_or_else(|| RegistryError::NotFound(id.to_string()))
}

#[derive(Debug, Deserialize)]
struct CreateGraphBody {
    id: String,
    schema: Schema,
}

#[derive(Debug, Serialize)]
struct GraphListResponse {
    graphs: Vec<String>,
}

async fn list_graphs(State(state): State<AppState>) -> Json<GraphListResponse> {
    Json(GraphListResponse {
        graphs: state.registry.list_ids(),
    })
}

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
async fn get_graph(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = GraphMetadataResponse::new(id, entry.content_hash().to_string());
    Ok(Json(response).into_response())
}

/// Full schema view: same shape consumed by generation_plus codegen
/// (matches storyflow's C-partial `build_schema_contract` output).
async fn get_schema(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let schema = entry.with_engine_read(|engine| engine.schema().clone());
    let response = SchemaResponse::with_cached_hash(id, schema, entry.content_hash().to_string());
    Ok(Json(response).into_response())
}

/// Replace a graph's schema. Compat rules + atomicity guarantees
/// live on `GraphRegistry::replace_schema`; this is a thin wrapper.
async fn replace_schema(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(new_schema): Json<Schema>,
) -> Result<Response, RegistryError> {
    let new_hash = state.registry.replace_schema(&id, new_schema.clone())?;
    let response = SchemaResponse::with_cached_hash(id, new_schema, new_hash.to_string());
    Ok(Json(response).into_response())
}

async fn delete_graph(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, RegistryError> {
    state.registry.delete_graph(&id)?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Deserialize)]
struct CreateNodeBody {
    node_type: String,
    node_id: String,
    #[serde(default)]
    properties: HashMap<String, Value>,
}

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
        .with_engine_write(|engine| engine.create_node(&id, &node_type, &node_id, properties))?;
    Ok((StatusCode::CREATED, Json(NodeResponse::from(stored))).into_response())
}

#[derive(Debug, Deserialize)]
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

    let nodes = entry.with_engine_read(|engine| -> Result<Vec<_>, RegistryError> {
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
    })?;

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
        PropertyType::Float | PropertyType::ListString => Err(RegistryError::BadRequest(format!(
            "filtering by {node_type}.{prop} is not supported (property type {:?} is not indexed)",
            pd.prop_type
        ))),
    }
}

async fn get_node(
    State(state): State<AppState>,
    Path((id, node_type, node_id)): Path<(String, String, String)>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let stored = entry
        .with_engine_read(|engine| engine.get_node(&id, &node_type, &node_id))?
        .ok_or(RegistryError::NodeNotFound { node_type, node_id })?;
    Ok(Json(NodeResponse::from(stored)).into_response())
}

#[derive(Debug, Deserialize)]
struct ReplaceNodeBody {
    #[serde(default)]
    properties: HashMap<String, Value>,
}

/// PUT semantics — full replacement of the node's property map (the
/// underlying storage call is `replace_node_properties`). PATCH is
/// not exposed because there is no merge primitive on nodes; if a
/// caller needs partial-update semantics they GET, mutate, PUT.
async fn replace_node(
    State(state): State<AppState>,
    Path((id, node_type, node_id)): Path<(String, String, String)>,
    Json(body): Json<ReplaceNodeBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let stored = entry
        .with_engine_write(|engine| {
            engine.replace_node_properties(&id, &node_type, &node_id, body.properties)
        })?
        .ok_or(RegistryError::NodeNotFound { node_type, node_id })?;
    Ok(Json(NodeResponse::from(stored)).into_response())
}

async fn delete_node(
    State(state): State<AppState>,
    Path((id, node_type, node_id)): Path<(String, String, String)>,
) -> Result<StatusCode, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let existed =
        entry.with_engine_write(|engine| engine.delete_node(&id, &node_type, &node_id))?;
    if existed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(RegistryError::NodeNotFound { node_type, node_id })
    }
}

#[derive(Debug, Deserialize)]
struct CreateEdgeBody {
    edge_type: String,
    from_type: String,
    from_id: String,
    to_type: String,
    to_id: String,
    #[serde(default)]
    properties: HashMap<String, Value>,
}

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
    let stored = entry.with_engine_write(|engine| {
        engine.create_edge(
            &id, &edge_type, &from_type, &from_id, &to_type, &to_id, properties,
        )
    })?;
    Ok((StatusCode::CREATED, Json(EdgeResponse::from(stored))).into_response())
}

async fn get_edge(
    State(state): State<AppState>,
    Path((id, edge_type, from_id, to_id)): Path<(String, String, String, String)>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let stored = entry
        .with_engine_read(|engine| engine.get_edge(&id, &edge_type, &from_id, &to_id))?
        .ok_or(RegistryError::EdgeNotFound {
            edge_type,
            from_id,
            to_id,
        })?;
    Ok(Json(EdgeResponse::from(stored)).into_response())
}

#[derive(Debug, Deserialize)]
struct MergeEdgeBody {
    #[serde(default)]
    properties: HashMap<String, Value>,
}

/// PATCH semantics — partial-update of the edge's property map. The
/// underlying storage call is `merge_edge_properties`. This mirrors
/// node CRUD's PUT (REPLACE) shape but with the verb flipped to match
/// the storage primitive's asymmetry — see `replace_node_properties`
/// docs for why nodes don't have a merge primitive.
async fn merge_edge(
    State(state): State<AppState>,
    Path((id, edge_type, from_id, to_id)): Path<(String, String, String, String)>,
    Json(body): Json<MergeEdgeBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let stored = entry
        .with_engine_write(|engine| {
            engine.merge_edge_properties(&id, &edge_type, &from_id, &to_id, body.properties)
        })?
        .ok_or(RegistryError::EdgeNotFound {
            edge_type,
            from_id,
            to_id,
        })?;
    Ok(Json(EdgeResponse::from(stored)).into_response())
}

async fn delete_edge(
    State(state): State<AppState>,
    Path((id, edge_type, from_id, to_id)): Path<(String, String, String, String)>,
) -> Result<StatusCode, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let existed =
        entry.with_engine_write(|engine| engine.delete_edge(&id, &edge_type, &from_id, &to_id))?;
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
