//! Async HTTP client for the dynograph-service `/v1/` API.
//!
//! Construct with [`DynographClient::new`] (a base URL like
//! `http://localhost:8080`) for TCP, or [`DynographClient::connect_unix`]
//! (a socket path) to use foundation's faster same-host Unix-socket
//! transport — both speak the identical JSON `/v1` API, so the methods
//! are transport-agnostic. Attach a JWT via
//! [`DynographClient::with_bearer`] when the server is configured with
//! `provider = "bearer_jwt"`. Each method wraps one HTTP route, returning
//! the corresponding wire type from [`wire`] or `()` for routes whose
//! response carries no body. Errors surface as [`ClientError`] — see the
//! error module's doc.

mod error;
mod wire;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use dynograph_core::Schema;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Per-request wall-clock cap. Matches the reqwest TCP client's timeout
/// so both transports shed a hung request the same way.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

pub use error::ClientError;
pub use wire::{
    EdgeResponse, EmbeddingResponse, GraphMetadataResponse, NodeExistence, NodeListResponse,
    NodeResponse, NodesExistsResponse, Precision, ResolveOrCreateResponse, SchemaResponse,
    SimilarHit, SimilarResponse, UtilRatioResponse, UtilResult, UtilScalarResponse,
    UtilVectorResponse, WelfordUpdateResponse,
};

/// The service's JSON error envelope, `{ "error": "<message>" }`,
/// decoded to surface the message in [`ClientError::Http`].
#[derive(Deserialize)]
struct ErrorBody {
    error: String,
}

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
    transport: Transport,
    /// Human-readable endpoint for [`DynographClient::base_url`] and
    /// logging: the base URL (TCP) or the socket path (UDS).
    endpoint: Arc<str>,
    bearer: Option<Arc<str>>,
}

/// How the client reaches the service. `Tcp` keeps the original reqwest
/// path (unchanged behavior). `Unix` adds a pooled hyper client over a
/// Unix domain socket — the faster same-host transport foundation serves
/// when `[server].uds_path` is set. Both speak the identical HTTP/1.1 +
/// JSON `/v1` API, so the public methods don't care which is active.
#[derive(Clone)]
enum Transport {
    Tcp {
        http: reqwest::Client,
        base_url: Arc<str>,
    },
    Unix {
        client: hyper_util::client::legacy::Client<hyperlocal::UnixConnector, Full<Bytes>>,
        socket: Arc<Path>,
    },
}

impl DynographClient {
    /// Build a client targeting `base_url` (e.g. `http://localhost:8080`)
    /// over TCP. The trailing slash is normalized away.
    pub fn new(base_url: impl Into<String>) -> Self {
        let mut url = base_url.into();
        while url.ends_with('/') {
            url.pop();
        }
        let base_url: Arc<str> = Arc::from(url);
        Self {
            transport: Transport::Tcp {
                http: reqwest::Client::builder()
                    .timeout(REQUEST_TIMEOUT)
                    .build()
                    .expect("default reqwest client"),
                base_url: base_url.clone(),
            },
            endpoint: base_url,
            bearer: None,
        }
    }

    /// Build a client that reaches the service over a Unix domain socket
    /// at `socket_path` — the faster same-host transport foundation
    /// serves when `[server].uds_path` is set. Connections are pooled and
    /// reused (keep-alive), which is where the UDS win over TCP is
    /// largest. The wire protocol is identical HTTP/1.1 + JSON, so every
    /// method behaves exactly as it does over TCP.
    pub fn connect_unix(socket_path: impl AsRef<Path>) -> Self {
        let socket: Arc<Path> = Arc::from(socket_path.as_ref());
        let client =
            hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
                .build(hyperlocal::UnixConnector);
        Self {
            endpoint: Arc::from(socket.to_string_lossy().as_ref()),
            transport: Transport::Unix { client, socket },
            bearer: None,
        }
    }

    /// Attach a bearer token. Sent on every request as
    /// `Authorization: Bearer <token>`.
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.bearer = Some(Arc::from(token.into()));
        self
    }

    /// The endpoint this client targets: the base URL (TCP, no trailing
    /// slash) or the socket path (UDS).
    pub fn base_url(&self) -> &str {
        &self.endpoint
    }

    fn request(&self, method: Method, path: &str) -> Pending {
        debug_assert!(path.starts_with('/'), "path must start with /");
        Pending {
            method,
            path: path.to_string(),
            query: Vec::new(),
            body: Ok(None),
        }
    }

    /// Execute a built request over the active transport, returning the
    /// raw status + body bytes. This is the only transport-specific
    /// method; everything above and below it is transport-agnostic.
    async fn execute(&self, pending: Pending) -> Result<(reqwest::StatusCode, Bytes), ClientError> {
        let Pending {
            method,
            path,
            query,
            body,
        } = pending;
        // A request-body encode failure surfaces here rather than being
        // swallowed at `.json()` time.
        let body = body?;

        match &self.transport {
            Transport::Tcp { http, base_url } => {
                let mut req = http.request(method, format!("{base_url}{path}"));
                if !query.is_empty() {
                    req = req.query(&query);
                }
                if let Some(bytes) = body {
                    req = req
                        .header(reqwest::header::CONTENT_TYPE, "application/json")
                        .body(bytes);
                }
                if let Some(token) = &self.bearer {
                    req = req.bearer_auth(token.as_ref());
                }
                let response = req.send().await?;
                let status = response.status();
                let bytes = response.bytes().await?;
                Ok((status, bytes))
            }
            Transport::Unix { client, socket } => {
                let uri: hyper::Uri =
                    hyperlocal::Uri::new(socket.as_ref(), &path_and_query(&path, &query)).into();
                let mut builder = hyper::Request::builder().method(method).uri(uri);
                if body.is_some() {
                    builder = builder.header(hyper::header::CONTENT_TYPE, "application/json");
                }
                if let Some(token) = &self.bearer {
                    builder =
                        builder.header(hyper::header::AUTHORIZATION, format!("Bearer {token}"));
                }
                let request = builder
                    .body(Full::new(Bytes::from(body.unwrap_or_default())))
                    .map_err(|e| ClientError::Unix(e.to_string()))?;
                // One timeout over the whole exchange — header wait *and*
                // body read — so a stalled response body is shed the same
                // way the reqwest TCP client's wall-clock timeout sheds it.
                let exchange = async {
                    let response = client
                        .request(request)
                        .await
                        .map_err(|e| ClientError::Unix(e.to_string()))?;
                    let status = response.status();
                    let bytes = response
                        .into_body()
                        .collect()
                        .await
                        .map_err(|e| ClientError::Unix(e.to_string()))?
                        .to_bytes();
                    Ok::<_, ClientError>((status, bytes))
                };
                tokio::time::timeout(REQUEST_TIMEOUT, exchange)
                    .await
                    .map_err(|_| ClientError::Unix("request timed out".into()))?
            }
        }
    }

    /// Send and surface a `ClientError::Http { status, body }` for any
    /// non-2xx — `body` is the server's error message. 2xx returns the
    /// raw body bytes.
    ///
    /// The service returns errors as JSON `{ "error": "<message>" }`, so
    /// we extract the `error` field; a body not in that shape (e.g. from
    /// an upstream proxy) is passed through verbatim.
    async fn send_raw(&self, pending: Pending) -> Result<Bytes, ClientError> {
        let (status, bytes) = self.execute(pending).await?;
        if status.is_success() {
            return Ok(bytes);
        }
        let body = serde_json::from_slice::<ErrorBody>(&bytes)
            .map(|e| e.error)
            .unwrap_or_else(|_| String::from_utf8_lossy(&bytes).into_owned());
        Err(ClientError::Http { status, body })
    }

    async fn send_json<T: for<'de> Deserialize<'de>>(
        &self,
        pending: Pending,
    ) -> Result<T, ClientError> {
        let bytes = self.send_raw(pending).await?;
        serde_json::from_slice(&bytes).map_err(ClientError::from)
    }

    /// For DELETE-style endpoints whose success status is 204.
    async fn send_unit(&self, pending: Pending) -> Result<(), ClientError> {
        self.send_raw(pending).await?;
        Ok(())
    }

    /// Shorthand for `POST <path>` + JSON body + JSON response, the
    /// shape that dominates the v0.5.6 surface.
    async fn post_json<B: Serialize, T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, ClientError> {
        self.send_json(self.request(Method::POST, path).json(body))
            .await
    }

    /// For `/metrics` (Prometheus text) and `/health` / `/ready`.
    async fn send_text(&self, pending: Pending) -> Result<String, ClientError> {
        let bytes = self.send_raw(pending).await?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
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
        self.post_json(&format!("/v1/graphs/{id}/batch"), body)
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
        self.post_json(&format!("/v1/graphs/{id}/resolve-or-create"), body)
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
        self.post_json(&format!("/v1/graphs/{id}/edges:collect"), body)
            .await
    }

    /// `POST /v1/graphs/{id}/edges:adjacent` — single-node 1-hop adjacency
    /// (the per-node `outgoing_edges(id)` / `incoming_edges(id)` that
    /// `edges:collect` does not cover). Body:
    /// `{node_id, direction?: "outgoing"|"incoming"|"both", edge_type?,
    /// limit?: 1..=10_000 (default 10_000)}`.
    /// Response: `{"edges": [{edge_type, from_id, to_id, properties}],
    /// "truncated": bool}`.
    pub async fn edges_adjacent(
        &self,
        id: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, ClientError> {
        self.post_json(&format!("/v1/graphs/{id}/edges:adjacent"), body)
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
        self.post_json(&format!("/v1/graphs/{id}/traverse"), body)
            .await
    }

    /// `POST /v1/graphs/{id}/nodes:exists` — batch (type, name)
    /// existence check. Returns typed results in request order.
    pub async fn nodes_exists(
        &self,
        id: &str,
        body: &serde_json::Value,
    ) -> Result<NodesExistsResponse, ClientError> {
        self.post_json(&format!("/v1/graphs/{id}/nodes:exists"), body)
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
        self.post_json(&format!("/v1/graphs/{id}/nodes:scan"), body)
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
        self.post_json(
            &format!("/v1/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}/welford_update"),
            &body,
        )
        .await
    }

    // =========================================================================
    // /v1/util/* — pure-math utility endpoints. Stateless; no graph_id.
    // `precision` is typed (`Precision::F32` / `Precision::F64`, default
    // `F64`). Omitted from the request body when `None`. F64-only stats
    // (pearson_correlation, linear_regression_slope) don't take it.
    // =========================================================================

    /// `POST /v1/util/cosine_similarity` — clamped to `[-1, 1]`;
    /// returns 0.0 when either vector has zero magnitude.
    pub async fn util_cosine_similarity(
        &self,
        a: &[f64],
        b: &[f64],
        precision: Option<Precision>,
    ) -> Result<UtilScalarResponse, ClientError> {
        self.post_json(
            "/v1/util/cosine_similarity",
            &binary_vec_body(a, b, precision),
        )
        .await
    }

    /// `POST /v1/util/dot_product` — inner product of two vectors.
    pub async fn util_dot_product(
        &self,
        a: &[f64],
        b: &[f64],
        precision: Option<Precision>,
    ) -> Result<UtilScalarResponse, ClientError> {
        self.post_json("/v1/util/dot_product", &binary_vec_body(a, b, precision))
            .await
    }

    /// `POST /v1/util/euclidean_distance` — L2 distance between two
    /// vectors of equal length.
    pub async fn util_euclidean_distance(
        &self,
        a: &[f64],
        b: &[f64],
        precision: Option<Precision>,
    ) -> Result<UtilScalarResponse, ClientError> {
        self.post_json(
            "/v1/util/euclidean_distance",
            &binary_vec_body(a, b, precision),
        )
        .await
    }

    /// `POST /v1/util/l2_norm` — magnitude (Euclidean norm) of a
    /// vector.
    pub async fn util_l2_norm(
        &self,
        v: &[f64],
        precision: Option<Precision>,
    ) -> Result<UtilScalarResponse, ClientError> {
        self.post_json("/v1/util/l2_norm", &unary_vec_body(v, precision))
            .await
    }

    /// `POST /v1/util/hadamard` — element-wise product. Result length
    /// equals input length.
    pub async fn util_hadamard(
        &self,
        a: &[f64],
        b: &[f64],
        precision: Option<Precision>,
    ) -> Result<UtilVectorResponse, ClientError> {
        self.post_json("/v1/util/hadamard", &binary_vec_body(a, b, precision))
            .await
    }

    /// `POST /v1/util/pearson_correlation` — sample correlation
    /// coefficient. f64-only; `dynograph-vector` ships no f32 variant.
    /// Returns 400 for n<3 or zero-variance inputs (loud-fail).
    pub async fn util_pearson_correlation(
        &self,
        a: &[f64],
        b: &[f64],
    ) -> Result<UtilScalarResponse, ClientError> {
        self.post_json("/v1/util/pearson_correlation", &json!({ "a": a, "b": b }))
            .await
    }

    /// `POST /v1/util/linear_regression_slope` — OLS slope of `y` on
    /// `x`. f64-only. Returns 400 when input has fewer than 2 points
    /// or zero variance in x.
    pub async fn util_linear_regression_slope(
        &self,
        points: &[(f64, f64)],
    ) -> Result<UtilScalarResponse, ClientError> {
        self.post_json(
            "/v1/util/linear_regression_slope",
            &json!({ "points": points }),
        )
        .await
    }

    /// `POST /v1/util/jaro_winkler` — Jaro-Winkler ratio scaled to
    /// `0..=100`. Strings are capped at 64 KB.
    pub async fn util_jaro_winkler(
        &self,
        a: &str,
        b: &str,
    ) -> Result<UtilRatioResponse, ClientError> {
        self.post_json("/v1/util/jaro_winkler", &json!({ "a": a, "b": b }))
            .await
    }

    /// `POST /v1/util/token_sort_ratio` — token-aware fuzzy ratio
    /// (permutation-invariant), scaled to `0..=100`. Strings are
    /// capped at 64 KB.
    pub async fn util_token_sort_ratio(
        &self,
        a: &str,
        b: &str,
    ) -> Result<UtilRatioResponse, ClientError> {
        self.post_json("/v1/util/token_sort_ratio", &json!({ "a": a, "b": b }))
            .await
    }

    // ---------------------------------------------------------------------
    // v0.6.x util additions — distances, element-wise algebra, stats.
    // ---------------------------------------------------------------------

    /// `POST /v1/util/squared_euclidean_distance` — Euclidean distance
    /// without the `sqrt` (order-preserving, cheaper for ranking).
    pub async fn util_squared_euclidean_distance(
        &self,
        a: &[f64],
        b: &[f64],
        precision: Option<Precision>,
    ) -> Result<UtilScalarResponse, ClientError> {
        self.post_json(
            "/v1/util/squared_euclidean_distance",
            &binary_vec_body(a, b, precision),
        )
        .await
    }

    /// `POST /v1/util/manhattan_distance` — L1 / taxicab distance.
    pub async fn util_manhattan_distance(
        &self,
        a: &[f64],
        b: &[f64],
        precision: Option<Precision>,
    ) -> Result<UtilScalarResponse, ClientError> {
        self.post_json(
            "/v1/util/manhattan_distance",
            &binary_vec_body(a, b, precision),
        )
        .await
    }

    /// `POST /v1/util/add` — element-wise sum `a + b`.
    pub async fn util_add(
        &self,
        a: &[f64],
        b: &[f64],
        precision: Option<Precision>,
    ) -> Result<UtilVectorResponse, ClientError> {
        self.post_json("/v1/util/add", &binary_vec_body(a, b, precision))
            .await
    }

    /// `POST /v1/util/subtract` — element-wise difference `a - b`.
    pub async fn util_subtract(
        &self,
        a: &[f64],
        b: &[f64],
        precision: Option<Precision>,
    ) -> Result<UtilVectorResponse, ClientError> {
        self.post_json("/v1/util/subtract", &binary_vec_body(a, b, precision))
            .await
    }

    /// `POST /v1/util/hadamard_division` — element-wise `a / b`. The
    /// server returns 400 (`ClientError::Http`) if any component of `b`
    /// is zero.
    pub async fn util_hadamard_division(
        &self,
        a: &[f64],
        b: &[f64],
        precision: Option<Precision>,
    ) -> Result<UtilVectorResponse, ClientError> {
        self.post_json(
            "/v1/util/hadamard_division",
            &binary_vec_body(a, b, precision),
        )
        .await
    }

    /// `POST /v1/util/scale` — scalar multiple `c * v`.
    pub async fn util_scale(
        &self,
        v: &[f64],
        c: f64,
        precision: Option<Precision>,
    ) -> Result<UtilVectorResponse, ClientError> {
        let body = match precision {
            Some(p) => json!({ "v": v, "c": c, "precision": p }),
            None => json!({ "v": v, "c": c }),
        };
        self.post_json("/v1/util/scale", &body).await
    }

    /// `POST /v1/util/negate` — reverse direction (`-v`).
    pub async fn util_negate(
        &self,
        v: &[f64],
        precision: Option<Precision>,
    ) -> Result<UtilVectorResponse, ClientError> {
        self.post_json("/v1/util/negate", &unary_vec_body(v, precision))
            .await
    }

    /// `POST /v1/util/elementwise_power` — each component raised to `p`.
    pub async fn util_elementwise_power(
        &self,
        v: &[f64],
        p: f64,
        precision: Option<Precision>,
    ) -> Result<UtilVectorResponse, ClientError> {
        let body = match precision {
            Some(prec) => json!({ "v": v, "p": p, "precision": prec }),
            None => json!({ "v": v, "p": p }),
        };
        self.post_json("/v1/util/elementwise_power", &body).await
    }

    /// `POST /v1/util/l2_normalize` — unit-length copy of `v`. The server
    /// returns 400 if `v` has zero magnitude.
    pub async fn util_l2_normalize(
        &self,
        v: &[f64],
        precision: Option<Precision>,
    ) -> Result<UtilVectorResponse, ClientError> {
        self.post_json("/v1/util/l2_normalize", &unary_vec_body(v, precision))
            .await
    }

    /// `POST /v1/util/centroid` — component-wise mean of equal-length
    /// `vectors`. 400 if empty or ragged.
    pub async fn util_centroid(
        &self,
        vectors: &[Vec<f64>],
        precision: Option<Precision>,
    ) -> Result<UtilVectorResponse, ClientError> {
        let body = match precision {
            Some(p) => json!({ "vectors": vectors, "precision": p }),
            None => json!({ "vectors": vectors }),
        };
        self.post_json("/v1/util/centroid", &body).await
    }

    /// `POST /v1/util/mean` — arithmetic mean (f64-only). 400 on empty.
    pub async fn util_mean(&self, values: &[f64]) -> Result<UtilScalarResponse, ClientError> {
        self.post_json("/v1/util/mean", &json!({ "values": values }))
            .await
    }

    /// `POST /v1/util/variance` — sample variance (`n-1`). 400 for n<2.
    pub async fn util_variance(&self, values: &[f64]) -> Result<UtilScalarResponse, ClientError> {
        self.post_json("/v1/util/variance", &json!({ "values": values }))
            .await
    }

    /// `POST /v1/util/std_dev` — sample standard deviation. 400 for n<2.
    pub async fn util_std_dev(&self, values: &[f64]) -> Result<UtilScalarResponse, ClientError> {
        self.post_json("/v1/util/std_dev", &json!({ "values": values }))
            .await
    }

    /// `POST /v1/util/median` — 50th percentile. 400 on empty.
    pub async fn util_median(&self, values: &[f64]) -> Result<UtilScalarResponse, ClientError> {
        self.post_json("/v1/util/median", &json!({ "values": values }))
            .await
    }

    /// `POST /v1/util/percentile` — linear-interpolated percentile `p`
    /// (`0..=100`). 400 on empty or out-of-range `p`.
    pub async fn util_percentile(
        &self,
        values: &[f64],
        p: f64,
    ) -> Result<UtilScalarResponse, ClientError> {
        self.post_json("/v1/util/percentile", &json!({ "values": values, "p": p }))
            .await
    }

    /// `POST /v1/util/softmax` — numerically-stable softmax distribution.
    pub async fn util_softmax(&self, values: &[f64]) -> Result<UtilVectorResponse, ClientError> {
        self.post_json("/v1/util/softmax", &json!({ "values": values }))
            .await
    }

    /// `POST /v1/util/spearman_correlation` — rank correlation (f64-only,
    /// n>=3). 400 on a constant input.
    pub async fn util_spearman_correlation(
        &self,
        a: &[f64],
        b: &[f64],
    ) -> Result<UtilScalarResponse, ClientError> {
        self.post_json("/v1/util/spearman_correlation", &json!({ "a": a, "b": b }))
            .await
    }
}

/// A transport-agnostic request under construction. Mirrors the small
/// slice of reqwest's builder the client actually uses (`.json()` /
/// `.query()`) so `execute` can dispatch the same description to either
/// transport.
struct Pending {
    method: Method,
    path: String,
    query: Vec<(String, String)>,
    /// Serialized request body, or the encode error to surface at send
    /// time (mirrors reqwest's eager-serialize-on-`.json()` behavior).
    body: Result<Option<Vec<u8>>, serde_json::Error>,
}

impl Pending {
    fn json<B: Serialize>(mut self, body: &B) -> Self {
        self.body = serde_json::to_vec(body).map(Some);
        self
    }

    fn query(mut self, query: &[(&str, &str)]) -> Self {
        self.query = query
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        self
    }
}

/// Build the `path[?query]` string for the UDS transport (the TCP path
/// uses reqwest's own `.query()`). Both encode via `serde_urlencoded`, so
/// the wire form matches across transports. String key/value pairs always
/// encode, so the `expect` is an invariant check, not a runtime failure
/// mode.
fn path_and_query(path: &str, query: &[(String, String)]) -> String {
    if query.is_empty() {
        return path.to_string();
    }
    let encoded = serde_urlencoded::to_string(query).expect("string pairs always url-encode");
    format!("{path}?{encoded}")
}

/// Body builder for the binary-vector `/v1/util/*` endpoints.
/// `precision` is omitted from the JSON object when `None` (server
/// defaults to f64) so the wire shape matches the
/// `#[serde(default)]` deserializer.
fn binary_vec_body(a: &[f64], b: &[f64], precision: Option<Precision>) -> serde_json::Value {
    match precision {
        Some(p) => json!({ "a": a, "b": b, "precision": p }),
        None => json!({ "a": a, "b": b }),
    }
}

fn unary_vec_body(v: &[f64], precision: Option<Precision>) -> serde_json::Value {
    match precision {
        Some(p) => json!({ "v": v, "precision": p }),
        None => json!({ "v": v }),
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
    fn connect_unix_reports_socket_path_as_endpoint() {
        let c = DynographClient::connect_unix("/run/dynograph/dynograph.sock");
        assert_eq!(c.base_url(), "/run/dynograph/dynograph.sock");
    }

    #[test]
    fn path_and_query_appends_encoded_pairs() {
        assert_eq!(
            path_and_query("/v1/graphs/g/nodes", &[]),
            "/v1/graphs/g/nodes"
        );
        let q = vec![
            ("type".to_string(), "Item".to_string()),
            ("value".to_string(), "a b".to_string()),
        ];
        assert_eq!(
            path_and_query("/v1/graphs/g/nodes", &q),
            "/v1/graphs/g/nodes?type=Item&value=a+b"
        );
    }
}
