//! Wire shape for `POST /v1/graphs/{id}/similar`.

use serde::Serialize;
use utoipa::ToSchema;

#[derive(Debug, Serialize, ToSchema)]
pub struct SimilarHit {
    pub node_id: String,
    pub score: f32,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct SimilarResponse {
    pub results: Vec<SimilarHit>,
}
