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
    EdgeResponse, EmbeddingResponse, GraphMetadataResponse, NodeExistence, NodeListResponse,
    NodeResponse, NodesExistsResponse, ResolveOrCreateResponse, SchemaResponse, SimilarHit,
    SimilarResponse, UtilRatioResponse, UtilScalarResponse, UtilVectorResponse,
    WelfordUpdateResponse,
};

/// Request body for `create_edge`. Carries the same fields as the
/// service's `CreateEdgeBody`; promoted to a struct so the client's
/// signature stays under the `clippy::too_many_arguments` floor and
/// reads like a declaration of intent at call sites.
#[derive(Debug, Clone, Serialize)]
pub struct CreateEdge<'a> {
    pub edge_type: &'a str,
    pub from_type: &'a str,
    pub from_id: &'a str,
    pub to_type: &'a str,
    pub to_id: &'a str,
    #[serde(borrow)]
    pub properties: &'a serde_json::Map<String, serde_json::Value>,
}

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

    /// Attach a bearer token. Sent on every request via reqwest's
    /// `bearer_auth`.
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
    ///
    /// To update a node's properties in place, use `replace_node` —
    /// delete-and-recreate-with-the-same-id drops every edge
    /// attached to the node.
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

    pub async fn create_edge(
        &self,
        id: &str,
        edge: &CreateEdge<'_>,
    ) -> Result<EdgeResponse, ClientError> {
        self.send_json(
            self.request(Method::POST, &format!("/v1/graphs/{id}/edges"))
                .json(edge),
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

    // =========================================================================
    // Audit-promoted primitives (v0.5.1 → v0.5.4) and v0.5.6 additions.
    //
    // The complex-body endpoints take `&serde_json::Value` for the
    // request and return `serde_json::Value` for the response — same
    // untyped pattern create_node uses for properties. See wire.rs for
    // the rationale and the few endpoints that do carry typed
    // responses (resolve_or_create, nodes:exists, welford_update).
    // =========================================================================

    /// `POST /v1/graphs/{id}/batch` — atomic multi-op transaction.
    /// `body` carries `{"ops": [{"op": "create_node", ...}, ...]}`.
    /// See service-side `crate::batch` for the wire shape and per-op
    /// error contract; foundation responds 400 with an
    /// `{op_index, op_type, error}` body if any op fails.
    pub async fn batch(
        &self,
        id: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, ClientError> {
        self.send_json(
            self.request(Method::POST, &format!("/v1/graphs/{id}/batch"))
                .json(body),
        )
        .await
    }

    /// `POST /v1/graphs/{id}/resolve-or-create` — fuzzy/vector entity
    /// resolution with create-on-miss. `body` carries
    /// `{node_type, properties, embedding?, scope?}`.
    pub async fn resolve_or_create(
        &self,
        id: &str,
        body: &serde_json::Value,
    ) -> Result<ResolveOrCreateResponse, ClientError> {
        self.send_json(
            self.request(Method::POST, &format!("/v1/graphs/{id}/resolve-or-create"))
                .json(body),
        )
        .await
    }

    /// `POST /v1/graphs/{id}/edges:collect` — fan-out edge collection
    /// across a typed source set. Response shape is untagged
    /// (`{"edges": [...]}` vs `{"adjacency": {...}}`) so it's returned
    /// as `serde_json::Value`; consumers branch on the field they
    /// requested via `format`.
    pub async fn edges_collect(
        &self,
        id: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, ClientError> {
        self.send_json(
            self.request(Method::POST, &format!("/v1/graphs/{id}/edges:collect"))
                .json(body),
        )
        .await
    }

    /// `POST /v1/graphs/{id}/traverse` — typed BFS over edge-type
    /// steps. Response carries `{nodes: [...], truncated: bool}` where
    /// each node has `node_type`, `node_id`, and (when `return=nodes`)
    /// `properties`.
    pub async fn traverse(
        &self,
        id: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, ClientError> {
        self.send_json(
            self.request(Method::POST, &format!("/v1/graphs/{id}/traverse"))
                .json(body),
        )
        .await
    }

    /// `POST /v1/graphs/{id}/nodes:exists` — batch (type, name)
    /// existence check. Returns typed results in request order.
    pub async fn nodes_exists(
        &self,
        id: &str,
        body: &serde_json::Value,
    ) -> Result<NodesExistsResponse, ClientError> {
        self.send_json(
            self.request(Method::POST, &format!("/v1/graphs/{id}/nodes:exists"))
                .json(body),
        )
        .await
    }

    /// `POST /v1/graphs/{id}/nodes:scan` — predicate-filtered scan.
    /// Response shape varies by `return` (`ids` vs `nodes`) so it's
    /// returned as `serde_json::Value`; consumers branch on the
    /// requested return shape.
    pub async fn nodes_scan(
        &self,
        id: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, ClientError> {
        self.send_json(
            self.request(Method::POST, &format!("/v1/graphs/{id}/nodes:scan"))
                .json(body),
        )
        .await
    }

    /// `POST /v1/graphs/{id}/edges/{type}/{from}/{to}/welford_update`
    /// — atomic EMA + Welford increment of the score property family
    /// on an existing edge. Returns the new (score, m2, stddev, min,
    /// max, count) sextuple.
    pub async fn welford_update(
        &self,
        id: &str,
        edge_type: &str,
        from_id: &str,
        to_id: &str,
        observation: f64,
        alpha: f64,
    ) -> Result<WelfordUpdateResponse, ClientError> {
        let body = json!({ "observation": observation, "alpha": alpha });
        self.send_json(
            self.request(
                Method::POST,
                &format!("/v1/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}/welford_update"),
            )
            .json(&body),
        )
        .await
    }

    // =========================================================================
    // /v1/util/* — pure-math utility endpoints. Stateless; no graph_id.
    // `precision` is `"f32" | "f64"` on the wire, default `"f64"`. For
    // f64-only stats (pearson_correlation, linear_regression_slope) the
    // field is omitted.
    // =========================================================================

    pub async fn util_cosine_similarity(
        &self,
        a: &[f64],
        b: &[f64],
        precision: Option<&str>,
    ) -> Result<UtilScalarResponse, ClientError> {
        let body = match precision {
            Some(p) => json!({ "a": a, "b": b, "precision": p }),
            None => json!({ "a": a, "b": b }),
        };
        self.send_json(
            self.request(Method::POST, "/v1/util/cosine_similarity")
                .json(&body),
        )
        .await
    }

    pub async fn util_dot_product(
        &self,
        a: &[f64],
        b: &[f64],
        precision: Option<&str>,
    ) -> Result<UtilScalarResponse, ClientError> {
        let body = match precision {
            Some(p) => json!({ "a": a, "b": b, "precision": p }),
            None => json!({ "a": a, "b": b }),
        };
        self.send_json(
            self.request(Method::POST, "/v1/util/dot_product")
                .json(&body),
        )
        .await
    }

    pub async fn util_euclidean_distance(
        &self,
        a: &[f64],
        b: &[f64],
        precision: Option<&str>,
    ) -> Result<UtilScalarResponse, ClientError> {
        let body = match precision {
            Some(p) => json!({ "a": a, "b": b, "precision": p }),
            None => json!({ "a": a, "b": b }),
        };
        self.send_json(
            self.request(Method::POST, "/v1/util/euclidean_distance")
                .json(&body),
        )
        .await
    }

    pub async fn util_l2_norm(
        &self,
        v: &[f64],
        precision: Option<&str>,
    ) -> Result<UtilScalarResponse, ClientError> {
        let body = match precision {
            Some(p) => json!({ "v": v, "precision": p }),
            None => json!({ "v": v }),
        };
        self.send_json(self.request(Method::POST, "/v1/util/l2_norm").json(&body))
            .await
    }

    pub async fn util_hadamard(
        &self,
        a: &[f64],
        b: &[f64],
        precision: Option<&str>,
    ) -> Result<UtilVectorResponse, ClientError> {
        let body = match precision {
            Some(p) => json!({ "a": a, "b": b, "precision": p }),
            None => json!({ "a": a, "b": b }),
        };
        self.send_json(self.request(Method::POST, "/v1/util/hadamard").json(&body))
            .await
    }

    /// f64-only — `dynograph-vector` ships no f32 variant.
    pub async fn util_pearson_correlation(
        &self,
        a: &[f64],
        b: &[f64],
    ) -> Result<UtilScalarResponse, ClientError> {
        self.send_json(
            self.request(Method::POST, "/v1/util/pearson_correlation")
                .json(&json!({ "a": a, "b": b })),
        )
        .await
    }

    pub async fn util_linear_regression_slope(
        &self,
        points: &[(f64, f64)],
    ) -> Result<UtilScalarResponse, ClientError> {
        self.send_json(
            self.request(Method::POST, "/v1/util/linear_regression_slope")
                .json(&json!({ "points": points })),
        )
        .await
    }

    pub async fn util_jaro_winkler(
        &self,
        a: &str,
        b: &str,
    ) -> Result<UtilRatioResponse, ClientError> {
        self.send_json(
            self.request(Method::POST, "/v1/util/jaro_winkler")
                .json(&json!({ "a": a, "b": b })),
        )
        .await
    }

    pub async fn util_token_sort_ratio(
        &self,
        a: &str,
        b: &str,
    ) -> Result<UtilRatioResponse, ClientError> {
        self.send_json(
            self.request(Method::POST, "/v1/util/token_sort_ratio")
                .json(&json!({ "a": a, "b": b })),
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
