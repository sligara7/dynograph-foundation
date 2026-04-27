//! Async HTTP client for the dynograph-service `/v1/` API.
//!
//! Construct with [`DynographClient::new`] (a base URL like
//! `http://localhost:8080`); attach a JWT via [`DynographClient::with_bearer`]
//! when the server is configured with `provider = "bearer_jwt"`. Each
//! method wraps one HTTP route, returning the corresponding wire
//! type from [`wire`] or `()` for routes whose response carries no
//! body. Errors surface as [`ClientError`] with three explicit
//! shapes — see the error module's doc.

mod error;
mod wire;

use std::sync::Arc;
use std::time::Duration;

use dynograph_core::Schema;
use reqwest::{Method, RequestBuilder, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;

pub use error::ClientError;
pub use wire::{
    EdgeResponse, EmbeddingResponse, GraphMetadataResponse, NodeListResponse, NodeResponse,
    SchemaResponse, SimilarHit, SimilarResponse,
};

#[derive(Clone)]
pub struct DynographClient {
    http: reqwest::Client,
    base_url: Arc<str>,
    bearer: Option<Arc<str>>,
}

impl DynographClient {
    /// Build a client targeting `base_url` (e.g. `http://localhost:8080`).
    /// The trailing slash is normalized away.
    pub fn new(base_url: impl Into<String>) -> Self {
        let mut url = base_url.into();
        while url.ends_with('/') {
            url.pop();
        }
        Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("default reqwest client"),
            base_url: Arc::from(url),
            bearer: None,
        }
    }

    /// Attach a bearer token. Sent as `Authorization: Bearer <token>`
    /// on every request — matching the slice 9 BearerJwt provider's
    /// case-insensitive scheme check.
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.bearer = Some(Arc::from(token.into()));
        self
    }

    /// Service base URL (with no trailing slash).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn url(&self, path: &str) -> String {
        debug_assert!(path.starts_with('/'), "path must start with /");
        format!("{}{}", self.base_url, path)
    }

    /// Build a `RequestBuilder` for `(method, path)` with the bearer
    /// header attached if configured. Centralizing this keeps every
    /// method honest about auth.
    fn request(&self, method: Method, path: &str) -> RequestBuilder {
        let mut req = self.http.request(method, self.url(path));
        if let Some(token) = &self.bearer {
            req = req.bearer_auth(token.as_ref());
        }
        req
    }

    /// Fire the request and surface a `ClientError::Http { status,
    /// body }` for any non-2xx — preserving the server's plain-text
    /// reason. 2xx responses pass through.
    async fn send_raw(&self, req: RequestBuilder) -> Result<Response, ClientError> {
        let response = req.send().await?;
        if response.status().is_success() {
            return Ok(response);
        }
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        Err(ClientError::Http { status, body })
    }

    async fn send_json<T: for<'de> Deserialize<'de>>(
        &self,
        req: RequestBuilder,
    ) -> Result<T, ClientError> {
        let response = self.send_raw(req).await?;
        let bytes = response.bytes().await?;
        serde_json::from_slice(&bytes).map_err(ClientError::from)
    }

    /// For DELETE-style endpoints whose success status is 204.
    async fn send_unit(&self, req: RequestBuilder) -> Result<(), ClientError> {
        self.send_raw(req).await?;
        Ok(())
    }

    /// For `/metrics` (Prometheus text) and `/health` / `/ready`.
    async fn send_text(&self, req: RequestBuilder) -> Result<String, ClientError> {
        let response = self.send_raw(req).await?;
        Ok(response.text().await?)
    }

    // =========================================================================
    // Operational endpoints (public on the server — no auth required)
    // =========================================================================

    pub async fn health(&self) -> Result<String, ClientError> {
        self.send_text(self.request(Method::GET, "/health")).await
    }

    pub async fn ready(&self) -> Result<String, ClientError> {
        self.send_text(self.request(Method::GET, "/ready")).await
    }

    /// `GET /metrics` — Prometheus text-format scrape body.
    pub async fn metrics(&self) -> Result<String, ClientError> {
        self.send_text(self.request(Method::GET, "/metrics")).await
    }

    // =========================================================================
    // Graph lifecycle
    // =========================================================================

    pub async fn create_graph(
        &self,
        id: &str,
        schema: &Schema,
    ) -> Result<SchemaResponse, ClientError> {
        let body = json!({ "id": id, "schema": schema });
        self.send_json(self.request(Method::POST, "/v1/graphs").json(&body))
            .await
    }

    pub async fn list_graphs(&self) -> Result<Vec<String>, ClientError> {
        #[derive(Deserialize)]
        struct ListBody {
            graphs: Vec<String>,
        }
        let body: ListBody = self
            .send_json(self.request(Method::GET, "/v1/graphs"))
            .await?;
        Ok(body.graphs)
    }

    pub async fn get_graph(&self, id: &str) -> Result<GraphMetadataResponse, ClientError> {
        self.send_json(self.request(Method::GET, &format!("/v1/graphs/{id}")))
            .await
    }

    pub async fn delete_graph(&self, id: &str) -> Result<(), ClientError> {
        self.send_unit(self.request(Method::DELETE, &format!("/v1/graphs/{id}")))
            .await
    }

    // =========================================================================
    // Schema
    // =========================================================================

    pub async fn get_schema(&self, id: &str) -> Result<SchemaResponse, ClientError> {
        self.send_json(self.request(Method::GET, &format!("/v1/graphs/{id}/schema")))
            .await
    }

    /// `PUT /v1/graphs/{id}/schema` — replace after additive-evolution
    /// check. The server returns the new content_hash on success.
    pub async fn replace_schema(
        &self,
        id: &str,
        schema: &Schema,
    ) -> Result<SchemaResponse, ClientError> {
        self.send_json(
            self.request(Method::PUT, &format!("/v1/graphs/{id}/schema"))
                .json(schema),
        )
        .await
    }

    // =========================================================================
    // Nodes
    // =========================================================================

    pub async fn create_node(
        &self,
        id: &str,
        node_type: &str,
        node_id: &str,
        properties: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<NodeResponse, ClientError> {
        let body = json!({
            "node_type": node_type,
            "node_id": node_id,
            "properties": properties,
        });
        self.send_json(
            self.request(Method::POST, &format!("/v1/graphs/{id}/nodes"))
                .json(&body),
        )
        .await
    }

    /// `GET /v1/graphs/{id}/nodes?type=X[&prop=Y&value=Z]`.
    pub async fn list_nodes(
        &self,
        id: &str,
        node_type: &str,
        prop_filter: Option<(&str, &str)>,
    ) -> Result<NodeListResponse, ClientError> {
        let mut query: Vec<(&str, &str)> = vec![("type", node_type)];
        if let Some((p, v)) = prop_filter {
            query.push(("prop", p));
            query.push(("value", v));
        }
        self.send_json(
            self.request(Method::GET, &format!("/v1/graphs/{id}/nodes"))
                .query(&query),
        )
        .await
    }

    pub async fn get_node(
        &self,
        id: &str,
        node_type: &str,
        node_id: &str,
    ) -> Result<NodeResponse, ClientError> {
        self.send_json(self.request(
            Method::GET,
            &format!("/v1/graphs/{id}/nodes/{node_type}/{node_id}"),
        ))
        .await
    }

    /// PUT REPLACES the node's property map; defaults re-apply.
    pub async fn replace_node(
        &self,
        id: &str,
        node_type: &str,
        node_id: &str,
        properties: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<NodeResponse, ClientError> {
        let body = json!({ "properties": properties });
        self.send_json(
            self.request(
                Method::PUT,
                &format!("/v1/graphs/{id}/nodes/{node_type}/{node_id}"),
            )
            .json(&body),
        )
        .await
    }

    /// Cascades server-side to drop the node's edges + embedding.
    pub async fn delete_node(
        &self,
        id: &str,
        node_type: &str,
        node_id: &str,
    ) -> Result<(), ClientError> {
        self.send_unit(self.request(
            Method::DELETE,
            &format!("/v1/graphs/{id}/nodes/{node_type}/{node_id}"),
        ))
        .await
    }

    // =========================================================================
    // Edges
    // =========================================================================

    #[allow(clippy::too_many_arguments)]
    pub async fn create_edge(
        &self,
        id: &str,
        edge_type: &str,
        from_type: &str,
        from_id: &str,
        to_type: &str,
        to_id: &str,
        properties: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<EdgeResponse, ClientError> {
        let body = json!({
            "edge_type": edge_type,
            "from_type": from_type,
            "from_id": from_id,
            "to_type": to_type,
            "to_id": to_id,
            "properties": properties,
        });
        self.send_json(
            self.request(Method::POST, &format!("/v1/graphs/{id}/edges"))
                .json(&body),
        )
        .await
    }

    pub async fn get_edge(
        &self,
        id: &str,
        edge_type: &str,
        from_id: &str,
        to_id: &str,
    ) -> Result<EdgeResponse, ClientError> {
        self.send_json(self.request(
            Method::GET,
            &format!("/v1/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}"),
        ))
        .await
    }

    /// PATCH MERGES the edge's property map (asymmetric to nodes'
    /// PUT semantics; reflects storage's `merge_edge_properties`
    /// primitive).
    pub async fn merge_edge(
        &self,
        id: &str,
        edge_type: &str,
        from_id: &str,
        to_id: &str,
        properties: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<EdgeResponse, ClientError> {
        let body = json!({ "properties": properties });
        self.send_json(
            self.request(
                Method::PATCH,
                &format!("/v1/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}"),
            )
            .json(&body),
        )
        .await
    }

    pub async fn delete_edge(
        &self,
        id: &str,
        edge_type: &str,
        from_id: &str,
        to_id: &str,
    ) -> Result<(), ClientError> {
        self.send_unit(self.request(
            Method::DELETE,
            &format!("/v1/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}"),
        ))
        .await
    }

    // =========================================================================
    // Embeddings
    // =========================================================================

    /// PUT sets + indexes. Per-(graph, node_type) HNSW dim is locked
    /// at the first insert; subsequent dim mismatch returns 400.
    pub async fn set_embedding(
        &self,
        id: &str,
        node_type: &str,
        node_id: &str,
        embedding: &[f32],
    ) -> Result<EmbeddingResponse, ClientError> {
        let body = json!({ "embedding": embedding });
        self.send_json(
            self.request(
                Method::PUT,
                &format!("/v1/graphs/{id}/nodes/{node_type}/{node_id}/embedding"),
            )
            .json(&body),
        )
        .await
    }

    pub async fn get_embedding(
        &self,
        id: &str,
        node_type: &str,
        node_id: &str,
    ) -> Result<EmbeddingResponse, ClientError> {
        self.send_json(self.request(
            Method::GET,
            &format!("/v1/graphs/{id}/nodes/{node_type}/{node_id}/embedding"),
        ))
        .await
    }

    pub async fn delete_embedding(
        &self,
        id: &str,
        node_type: &str,
        node_id: &str,
    ) -> Result<(), ClientError> {
        self.send_unit(self.request(
            Method::DELETE,
            &format!("/v1/graphs/{id}/nodes/{node_type}/{node_id}/embedding"),
        ))
        .await
    }

    // =========================================================================
    // Similarity
    // =========================================================================

    /// `node_type` is required; the per-type index dim must match
    /// `embedding.len()`.
    pub async fn similar(
        &self,
        id: &str,
        node_type: &str,
        embedding: &[f32],
        top_k: usize,
    ) -> Result<SimilarResponse, ClientError> {
        #[derive(Serialize)]
        struct Body<'a> {
            embedding: &'a [f32],
            top_k: usize,
            node_type: &'a str,
        }
        self.send_json(
            self.request(Method::POST, &format!("/v1/graphs/{id}/similar"))
                .json(&Body {
                    embedding,
                    top_k,
                    node_type,
                }),
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_strips_trailing_slashes() {
        let c = DynographClient::new("http://example.com:8080/");
        assert_eq!(c.base_url(), "http://example.com:8080");
        let c = DynographClient::new("http://example.com:8080///");
        assert_eq!(c.base_url(), "http://example.com:8080");
    }

    #[test]
    fn with_bearer_carries_token() {
        let c = DynographClient::new("http://x").with_bearer("abc.def.ghi");
        assert_eq!(c.bearer.as_deref(), Some("abc.def.ghi"));
    }

    #[test]
    fn url_joins_path_under_base() {
        let c = DynographClient::new("http://example.com");
        assert_eq!(c.url("/v1/graphs"), "http://example.com/v1/graphs");
    }
}
