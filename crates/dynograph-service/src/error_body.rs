//! The single JSON error envelope every endpoint returns on failure:
//! `{ "error": "<message>" }`.
//!
//! Before this, single handlers returned a bare `text/plain` reason
//! while `/batch` returned a structured per-op body — so a client had to
//! special-case the body shape per route. Now every error response is
//! JSON with an `error` field (the same field `/batch`'s per-op errors
//! carry, plus their op metadata), so one parse path works everywhere.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

/// JSON body for any error response: `{ "error": "<message>" }`.
#[derive(Debug, Serialize)]
pub(crate) struct ErrorBody {
    pub error: String,
}

/// Build a `(status, Json({ "error": <message> }))` response — the
/// canonical error shape. Used by `RegistryError::into_response` and the
/// timeout / concurrency / auth middlewares so every failure path is
/// consistent.
pub(crate) fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(ErrorBody {
            error: message.into(),
        }),
    )
        .into_response()
}
