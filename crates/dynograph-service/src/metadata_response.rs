//! Wire shape for `GET /v1/graphs/{id}`. Full schema lives at
//! `GET /v1/graphs/{id}/schema` (returns `SchemaResponse`).

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
