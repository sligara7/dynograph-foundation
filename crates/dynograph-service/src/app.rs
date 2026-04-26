//! axum app + route handlers.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use dynograph_core::{Schema, Value};

use crate::{
    auth::{AuthProvider, NoAuth},
    node_response::NodeResponse,
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
}

impl AppState {
    pub fn new(registry: Arc<GraphRegistry>, auth: Arc<dyn AuthProvider>) -> Self {
        Self { registry, auth }
    }

    /// Convenience for the dev / private-network default.
    pub fn with_no_auth(registry: Arc<GraphRegistry>) -> Self {
        Self::new(registry, Arc::new(NoAuth::new()))
    }
}

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/graphs", get(list_graphs).post(create_graph))
        .route("/v1/graphs/{id}", get(get_graph).delete(delete_graph))
        .route("/v1/graphs/{id}/nodes", post(create_node))
        .route(
            "/v1/graphs/{id}/nodes/{node_type}/{node_id}",
            get(get_node).put(replace_node).delete(delete_node),
        )
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
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

async fn get_graph(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let schema = entry.with_engine_read(|engine| engine.schema().clone());
    let response = SchemaResponse::with_cached_hash(id, schema, entry.content_hash().to_string());
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
