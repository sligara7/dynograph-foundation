//! axum app + route handlers.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;

use dynograph_core::Schema;

use crate::{
    auth::{AuthProvider, NoAuth},
    registry::{GraphRegistry, RegistryError},
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
        .route("/v1/graphs", post(create_graph))
        .route("/v1/graphs/{id}", get(get_graph))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

#[derive(Debug, Deserialize)]
struct CreateGraphBody {
    id: String,
    schema: Schema,
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
    let entry = state
        .registry
        .get(&id)
        .ok_or(RegistryError::NotFound(id.clone()))?;
    let schema = entry.with_engine_read(|engine| engine.schema().clone());
    let response = SchemaResponse::with_cached_hash(id, schema, entry.content_hash().to_string());
    Ok(Json(response).into_response())
}
