//! axum app + route handlers.
//!
//! Slice 1 routes:
//! - `GET  /health`            — liveness ("ok")
//! - `POST /v1/graphs`         — create graph; body `{id, schema}`
//! - `GET  /v1/graphs/{id}`    — fetch schema + content_hash
//!
//! Node/edge/query/similar routes follow in subsequent slices.

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
    pub registry: Arc<GraphRegistry>,
    pub auth: Arc<dyn AuthProvider>,
}

impl AppState {
    /// Convenience for `NoAuth` setups (slice-1 default + tests).
    pub fn with_no_auth(registry: Arc<GraphRegistry>) -> Self {
        Self {
            registry,
            auth: Arc::new(NoAuth),
        }
    }
}

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/graphs", post(create_graph))
        .route("/v1/graphs/:id", get(get_graph))
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
) -> Response {
    match state.registry.create_graph(&body.id, body.schema.clone()) {
        Ok(()) => {
            let response = SchemaResponse::new(body.id, body.schema);
            (StatusCode::CREATED, Json(response)).into_response()
        }
        Err(e @ RegistryError::AlreadyExists(_)) => {
            (StatusCode::CONFLICT, e.to_string()).into_response()
        }
        Err(e) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

async fn get_graph(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let Some(arc) = state.registry.get(&id) else {
        return (StatusCode::NOT_FOUND, format!("graph not found: {}", id)).into_response();
    };
    let schema = {
        let guard = arc.read().expect("graph read lock poisoned");
        guard.schema().clone()
    };
    Json(SchemaResponse::new(id, schema)).into_response()
}
