//! Wire shape for `POST /v1/graphs/{id}/similar`.

use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct SimilarHit {
    pub node_id: String,
    pub score: f32,
}

#[derive(Debug, Serialize)]
pub struct SimilarResponse {
    pub results: Vec<SimilarHit>,
}
