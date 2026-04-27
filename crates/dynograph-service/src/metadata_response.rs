//! Wire shape for graph metadata (schema-less).
//!
//! `GET /v1/graphs/{id}` returns this — id + version + content_hash, no
//! schema body. Callers that need the schema fetch
//! `GET /v1/graphs/{id}/schema` (returns `SchemaResponse`). Splitting
//! avoids shipping the full schema on every metadata-style query
//! (existence checks, drift comparisons by hash alone).

use serde::Serialize;

use crate::schema_response::WIRE_VERSION;

#[derive(Debug, Serialize)]
pub struct GraphMetadataResponse {
    pub id: String,
    pub wire_version: &'static str,
    pub content_hash: String,
}

impl GraphMetadataResponse {
    pub fn new(id: String, content_hash: String) -> Self {
        Self {
            id,
            wire_version: WIRE_VERSION,
            content_hash,
        }
    }
}
