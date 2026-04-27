//! axum app + route handlers.

use std::collections::HashMap;
use std::sync::Arc;

use std::time::Instant;

use axum::{
    Json, Router,
    extract::{MatchedPath, Path, Query, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use dynograph_core::{PropertyType, Schema, Value};
use dynograph_vector::HnswIndex;

use crate::{
    auth::{AuthProvider, NoAuth},
    edge_response::EdgeResponse,
    embedding_response::EmbeddingResponse,
    metadata_response::GraphMetadataResponse,
    metrics_state::MetricsState,
    node_response::{NodeListResponse, NodeResponse},
    readiness::Readiness,
    registry::{GraphEntry, GraphRegistry, RegistryError},
    schema_response::SchemaResponse,
    similar_response::{SimilarHit, SimilarResponse},
};

#[derive(Clone)]
pub struct AppState {
    pub(crate) registry: Arc<GraphRegistry>,
    pub(crate) auth: Arc<dyn AuthProvider>,
    pub(crate) readiness: Arc<Readiness>,
    pub(crate) metrics: Arc<MetricsState>,
}

impl AppState {
    pub fn new(
        registry: Arc<GraphRegistry>,
        auth: Arc<dyn AuthProvider>,
        readiness: Arc<Readiness>,
    ) -> Self {
        Self {
            registry,
            auth,
            readiness,
            metrics: Arc::new(MetricsState::new()),
        }
    }

    /// Convenience for the dev / private-network default. Picks
    /// `NoAuth` and `Readiness::ready` — the right defaults for
    /// in-memory test code and embedded use, neither of which has
    /// startup work that would warrant a not-ready window. The
    /// `dynograph` binary uses the lower-level `AppState::new`
    /// with an explicit not-ready `Readiness` because it does have
    /// startup work (`rehydrate`) and flips ready only after.
    pub fn with_no_auth(registry: Arc<GraphRegistry>) -> Self {
        Self::new(
            registry,
            Arc::new(NoAuth::new()),
            Arc::new(Readiness::ready()),
        )
    }

    pub fn readiness(&self) -> &Arc<Readiness> {
        &self.readiness
    }
}

pub fn app(state: AppState) -> Router {
    // /v1/* routes go through both the metrics middleware (outer)
    // and the auth middleware (inner): incoming → record start time
    // → authenticate → handler → record latency. /metrics itself is
    // public AND skips the metrics middleware (no self-recording on
    // every scrape). /health and /ready are public but ARE recorded
    // — useful for "is anyone hitting the probes" debugging.
    let v1: Router<AppState> = Router::new()
        .route("/v1/graphs", get(list_graphs).post(create_graph))
        .route("/v1/graphs/{id}", get(get_graph).delete(delete_graph))
        .route(
            "/v1/graphs/{id}/schema",
            get(get_schema).put(replace_schema),
        )
        .route("/v1/graphs/{id}/nodes", get(list_nodes).post(create_node))
        .route(
            "/v1/graphs/{id}/nodes/{node_type}/{node_id}",
            get(get_node).put(replace_node).delete(delete_node),
        )
        .route("/v1/graphs/{id}/edges", post(create_edge))
        .route(
            "/v1/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}",
            get(get_edge).patch(merge_edge).delete(delete_edge),
        )
        .route(
            "/v1/graphs/{id}/nodes/{node_type}/{node_id}/embedding",
            get(get_embedding)
                .put(set_embedding)
                .delete(delete_embedding),
        )
        .route("/v1/graphs/{id}/similar", post(similar))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    let observed_public: Router<AppState> = Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .merge(v1)
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            metrics_middleware,
        ));

    Router::new()
        .route("/metrics", get(metrics_handler))
        .merge(observed_public)
        .with_state(state)
}

/// Axum middleware: runs `state.auth.authenticate(headers)` on every
/// protected request. On success, inserts the resolved `Identity`
/// into request extensions so downstream handlers can read the
/// caller's user_id via `Extension<Identity>` if they need it. On
/// failure, short-circuits with 401 + the auth error's message.
async fn auth_middleware(State(state): State<AppState>, mut req: Request, next: Next) -> Response {
    match state.auth.authenticate(req.headers()) {
        Ok(identity) => {
            req.extensions_mut().insert(identity);
            next.run(req).await
        }
        Err(e) => (StatusCode::UNAUTHORIZED, e.message().to_string()).into_response(),
    }
}

/// Axum middleware: records (method, matched-path, status) +
/// latency into `MetricsState`. The matched-path label uses axum's
/// `MatchedPath` extension so cardinality stays bounded by static
/// route count (e.g. `/v1/graphs/{id}` is one label, regardless of
/// how many distinct ids were hit). Requests that miss every route
/// (404 from the router itself) won't have a `MatchedPath` set —
/// those are intentionally skipped to keep the label set finite.
async fn metrics_middleware(State(state): State<AppState>, req: Request, next: Next) -> Response {
    // Skip recording (and the per-request String allocs) when the
    // router didn't match anything — those 404s have no MatchedPath
    // and would inflate label cardinality if we made up a `__none__`
    // bucket. Allocate the owned method+path strings only after we
    // know we're going to insert.
    let matched_path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string());
    let Some(path) = matched_path else {
        return next.run(req).await;
    };
    let method = req.method().as_str().to_string();
    let start = Instant::now();
    let response = next.run(req).await;
    let elapsed_micros = start.elapsed().as_micros() as u64;
    state
        .metrics
        .record(&method, &path, response.status().as_u16(), elapsed_micros);
    response
}

/// Prometheus text-format scrape endpoint. Public — sit alongside
/// `/health` and `/ready`; the assumption is that the network/
/// ingress layer gates Prometheus scrape access to the metrics
/// endpoint when needed (k8s NetworkPolicy / Caddy IP allowlist /
/// etc). `/metrics` itself bypasses the metrics middleware to avoid
/// recording every scrape into the request-counter series, which
/// would inflate cardinality and mostly measure Prometheus's own
/// scrape interval.
async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    use std::fmt::Write;
    let mut out = String::new();

    let _ = writeln!(out, "# HELP dynograph_build_info Build information");
    let _ = writeln!(out, "# TYPE dynograph_build_info gauge");
    let _ = writeln!(
        out,
        "dynograph_build_info{{version=\"{}\"}} 1",
        env!("CARGO_PKG_VERSION")
    );

    let _ = writeln!(
        out,
        "# HELP dynograph_uptime_seconds Process uptime since start"
    );
    let _ = writeln!(out, "# TYPE dynograph_uptime_seconds gauge");
    let _ = writeln!(
        out,
        "dynograph_uptime_seconds {:.3}",
        state.metrics.uptime_secs()
    );

    let snap = state.metrics.snapshot();
    let _ = writeln!(
        out,
        "# HELP dynograph_http_requests_total Requests handled, by route + status"
    );
    let _ = writeln!(out, "# TYPE dynograph_http_requests_total counter");
    for (key, count, _sum) in &snap {
        let _ = writeln!(
            out,
            "dynograph_http_requests_total{{method=\"{}\",path=\"{}\",status=\"{}\"}} {}",
            key.method, key.path, key.status, count
        );
    }
    let _ = writeln!(
        out,
        "# HELP dynograph_http_request_duration_microseconds_sum Cumulative request latency"
    );
    let _ = writeln!(
        out,
        "# TYPE dynograph_http_request_duration_microseconds_sum counter"
    );
    for (key, _count, sum) in &snap {
        let _ = writeln!(
            out,
            "dynograph_http_request_duration_microseconds_sum{{method=\"{}\",path=\"{}\",status=\"{}\"}} {}",
            key.method, key.path, key.status, sum
        );
    }

    // Per-(graph, node_type) HNSW stats. Walks the registry under
    // its read lock; per-graph stats acquire the per-graph state
    // read lock briefly. Scrape cost scales with the number of
    // graphs × node_types-with-embeddings — for the foreseeable
    // range (dozens of graphs, single-digit indexed types each)
    // this is microseconds per scrape.
    let mut hnsw_snap: Vec<(String, String, dynograph_vector::HnswStats)> = Vec::new();
    for graph_id in state.registry.list_ids() {
        if let Some(entry) = state.registry.get(&graph_id) {
            for (node_type, stats) in entry.hnsw_stats_snapshot() {
                hnsw_snap.push((graph_id.clone(), node_type, stats));
            }
        }
    }
    emit_hnsw_metric(
        &mut out,
        "dynograph_hnsw_index_size",
        "gauge",
        "Live (non-tombstoned) embeddings per index",
        &hnsw_snap,
        |s| s.index_size as u64,
    );
    emit_hnsw_metric(
        &mut out,
        "dynograph_hnsw_searches_total",
        "counter",
        "HNSW search calls per index",
        &hnsw_snap,
        |s| s.searches_total,
    );
    emit_hnsw_metric(
        &mut out,
        "dynograph_hnsw_inserts_total",
        "counter",
        "HNSW insert calls per index",
        &hnsw_snap,
        |s| s.inserts_total,
    );
    emit_hnsw_metric(
        &mut out,
        "dynograph_hnsw_removes_total",
        "counter",
        "HNSW remove calls per index",
        &hnsw_snap,
        |s| s.removes_total,
    );

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        out,
    )
}

/// Emit one HNSW metric block (HELP + TYPE header + per-(graph,
/// node_type) line). The `field` projector picks the relevant stat
/// — collapses the four near-identical emission blocks (one per
/// counter) to one helper.
fn emit_hnsw_metric(
    out: &mut String,
    metric: &str,
    metric_type: &str,
    help: &str,
    snap: &[(String, String, dynograph_vector::HnswStats)],
    field: impl Fn(&dynograph_vector::HnswStats) -> u64,
) {
    use std::fmt::Write;
    let _ = writeln!(out, "# HELP {metric} {help}");
    let _ = writeln!(out, "# TYPE {metric} {metric_type}");
    for (graph, node_type, stats) in snap {
        let _ = writeln!(
            out,
            "{metric}{{graph=\"{graph}\",node_type=\"{node_type}\"}} {}",
            field(stats)
        );
    }
}

async fn health() -> &'static str {
    "ok"
}

/// Readiness probe — distinct from `/health`, which only confirms
/// the process is running. `/ready` returns 200 once the service
/// has finished startup work (notably `rehydrate()` on the on-disk
/// backend); 503 before that.
async fn ready(State(state): State<AppState>) -> (StatusCode, &'static str) {
    if state.readiness.is_ready() {
        (StatusCode::OK, "ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "starting")
    }
}

/// Look up a graph by id or surface a 404. Folds the
/// `state.registry.get(...).ok_or_else(...)` boilerplate that every
/// graph-id-bearing handler shares.
fn graph_entry(state: &AppState, id: &str) -> Result<Arc<GraphEntry>, RegistryError> {
    state
        .registry
        .get(id)
        .ok_or_else(|| RegistryError::NotFound(id.to_string()))
}

#[derive(Debug, Deserialize)]
struct CreateGraphBody {
    id: String,
    schema: Schema,
}

#[derive(Debug, Serialize)]
struct GraphListResponse {
    graphs: Vec<String>,
}

async fn list_graphs(State(state): State<AppState>) -> Json<GraphListResponse> {
    Json(GraphListResponse {
        graphs: state.registry.list_ids(),
    })
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

/// Metadata-only — see `GET /v1/graphs/{id}/schema` for the full schema.
async fn get_graph(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let response = GraphMetadataResponse::new(id, entry.content_hash().to_string());
    Ok(Json(response).into_response())
}

/// Full schema view: same shape consumed by generation_plus codegen
/// (matches storyflow's C-partial `build_schema_contract` output).
async fn get_schema(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let schema = entry.with_engine_read(|engine| engine.schema().clone());
    let response = SchemaResponse::with_cached_hash(id, schema, entry.content_hash().to_string());
    Ok(Json(response).into_response())
}

/// Replace a graph's schema. Compat rules + atomicity guarantees
/// live on `GraphRegistry::replace_schema`; this is a thin wrapper.
async fn replace_schema(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(new_schema): Json<Schema>,
) -> Result<Response, RegistryError> {
    let new_hash = state.registry.replace_schema(&id, new_schema.clone())?;
    let response = SchemaResponse::with_cached_hash(id, new_schema, new_hash.to_string());
    Ok(Json(response).into_response())
}

async fn delete_graph(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, RegistryError> {
    state.registry.delete_graph(&id)?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Deserialize)]
struct CreateNodeBody {
    node_type: String,
    node_id: String,
    #[serde(default)]
    properties: HashMap<String, Value>,
}

async fn create_node(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<CreateNodeBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let CreateNodeBody {
        node_type,
        node_id,
        properties,
    } = body;
    let stored = entry
        .with_engine_write(|engine| engine.create_node(&id, &node_type, &node_id, properties))?;
    Ok((StatusCode::CREATED, Json(NodeResponse::from(stored))).into_response())
}

#[derive(Debug, Deserialize)]
struct ListNodesQuery {
    #[serde(rename = "type")]
    node_type: String,
    prop: Option<String>,
    value: Option<String>,
}

/// List nodes of a given type, optionally filtered by a single
/// (`prop`, `value`) pair. The pair must be supplied together — half
/// of it is a 400. `value` arrives as a URL string and is coerced to
/// the schema-declared `PropertyType` for the property; coerce
/// failures are 400, not silent zero-result.
async fn list_nodes(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(q): Query<ListNodesQuery>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let ListNodesQuery {
        node_type,
        prop,
        value,
    } = q;

    let nodes = entry.with_engine_read(|engine| -> Result<Vec<_>, RegistryError> {
        match (prop, value) {
            (None, None) => engine
                .scan_nodes(&id, &node_type)
                .map_err(RegistryError::Storage),
            (Some(prop), Some(value)) => {
                let coerced = coerce_query_value(engine.schema(), &node_type, &prop, &value)?;
                engine
                    .scan_nodes_by_property(&id, &node_type, &prop, &coerced)
                    .map_err(RegistryError::Storage)
            }
            (Some(_), None) | (None, Some(_)) => Err(RegistryError::BadRequest(
                "prop and value must be supplied together".to_string(),
            )),
        }
    })?;

    let response = NodeListResponse::new(nodes.into_iter().map(NodeResponse::from).collect());
    Ok(Json(response).into_response())
}

/// Coerce a URL-string `value` into a `Value` typed per the schema's
/// declaration of `node_type.prop`. Mirrors the indexable subset of
/// `PropertyType`s — `Float`/`ListString` aren't indexed by storage's
/// `scan_nodes_by_property`, so filtering by them is rejected up
/// front (400) rather than silently returning empty. `Enum` accepts
/// any string; storage validates it against `values` only on writes,
/// so a non-member enum filter cleanly returns no matches.
fn coerce_query_value(
    schema: &Schema,
    node_type: &str,
    prop: &str,
    value: &str,
) -> Result<Value, RegistryError> {
    let nt = schema
        .node_types
        .get(node_type)
        .ok_or_else(|| RegistryError::BadRequest(format!("unknown node type: {node_type}")))?;
    let pd = nt.properties.get(prop).ok_or_else(|| {
        RegistryError::BadRequest(format!("unknown property: {node_type}.{prop}"))
    })?;
    match pd.prop_type {
        PropertyType::String | PropertyType::Enum => Ok(Value::String(value.to_string())),
        PropertyType::Datetime => Ok(Value::String(value.to_string())),
        PropertyType::Int => value.parse::<i64>().map(Value::Int).map_err(|e| {
            RegistryError::BadRequest(format!("value {value:?} is not a valid int: {e}"))
        }),
        PropertyType::Bool => value.parse::<bool>().map(Value::Bool).map_err(|e| {
            RegistryError::BadRequest(format!("value {value:?} is not a valid bool: {e}"))
        }),
        PropertyType::Float | PropertyType::ListString => Err(RegistryError::BadRequest(format!(
            "filtering by {node_type}.{prop} is not supported (property type {:?} is not indexed)",
            pd.prop_type
        ))),
    }
}

async fn get_node(
    State(state): State<AppState>,
    Path((id, node_type, node_id)): Path<(String, String, String)>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let stored = entry
        .with_engine_read(|engine| engine.get_node(&id, &node_type, &node_id))?
        .ok_or(RegistryError::NodeNotFound { node_type, node_id })?;
    Ok(Json(NodeResponse::from(stored)).into_response())
}

#[derive(Debug, Deserialize)]
struct ReplaceNodeBody {
    #[serde(default)]
    properties: HashMap<String, Value>,
}

/// PUT semantics — full replacement of the node's property map (the
/// underlying storage call is `replace_node_properties`). PATCH is
/// not exposed because there is no merge primitive on nodes; if a
/// caller needs partial-update semantics they GET, mutate, PUT.
async fn replace_node(
    State(state): State<AppState>,
    Path((id, node_type, node_id)): Path<(String, String, String)>,
    Json(body): Json<ReplaceNodeBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let stored = entry
        .with_engine_write(|engine| {
            engine.replace_node_properties(&id, &node_type, &node_id, body.properties)
        })?
        .ok_or(RegistryError::NodeNotFound { node_type, node_id })?;
    Ok(Json(NodeResponse::from(stored)).into_response())
}

async fn delete_node(
    State(state): State<AppState>,
    Path((id, node_type, node_id)): Path<(String, String, String)>,
) -> Result<StatusCode, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    // Storage's delete_node already cascades to drop the sidecar
    // embedding (slice 8a). The HNSW index is service-side state, so
    // we mirror the cascade here: if an index exists for this type,
    // remove the node from it. The whole cycle runs under one lock.
    let existed = entry.with_state_write(|engine, indexes| -> Result<bool, RegistryError> {
        let existed = engine.delete_node(&id, &node_type, &node_id)?;
        if existed && let Some(index) = indexes.get_mut(&node_type) {
            index.remove(&node_id);
        }
        Ok(existed)
    })?;
    if existed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(RegistryError::NodeNotFound { node_type, node_id })
    }
}

#[derive(Debug, Deserialize)]
struct CreateEdgeBody {
    edge_type: String,
    from_type: String,
    from_id: String,
    to_type: String,
    to_id: String,
    #[serde(default)]
    properties: HashMap<String, Value>,
}

async fn create_edge(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<CreateEdgeBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let CreateEdgeBody {
        edge_type,
        from_type,
        from_id,
        to_type,
        to_id,
        properties,
    } = body;
    let stored = entry.with_engine_write(|engine| {
        engine.create_edge(
            &id, &edge_type, &from_type, &from_id, &to_type, &to_id, properties,
        )
    })?;
    Ok((StatusCode::CREATED, Json(EdgeResponse::from(stored))).into_response())
}

async fn get_edge(
    State(state): State<AppState>,
    Path((id, edge_type, from_id, to_id)): Path<(String, String, String, String)>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let stored = entry
        .with_engine_read(|engine| engine.get_edge(&id, &edge_type, &from_id, &to_id))?
        .ok_or(RegistryError::EdgeNotFound {
            edge_type,
            from_id,
            to_id,
        })?;
    Ok(Json(EdgeResponse::from(stored)).into_response())
}

#[derive(Debug, Deserialize)]
struct MergeEdgeBody {
    #[serde(default)]
    properties: HashMap<String, Value>,
}

/// PATCH semantics — partial-update of the edge's property map. The
/// underlying storage call is `merge_edge_properties`. This mirrors
/// node CRUD's PUT (REPLACE) shape but with the verb flipped to match
/// the storage primitive's asymmetry — see `replace_node_properties`
/// docs for why nodes don't have a merge primitive.
async fn merge_edge(
    State(state): State<AppState>,
    Path((id, edge_type, from_id, to_id)): Path<(String, String, String, String)>,
    Json(body): Json<MergeEdgeBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let stored = entry
        .with_engine_write(|engine| {
            engine.merge_edge_properties(&id, &edge_type, &from_id, &to_id, body.properties)
        })?
        .ok_or(RegistryError::EdgeNotFound {
            edge_type,
            from_id,
            to_id,
        })?;
    Ok(Json(EdgeResponse::from(stored)).into_response())
}

async fn delete_edge(
    State(state): State<AppState>,
    Path((id, edge_type, from_id, to_id)): Path<(String, String, String, String)>,
) -> Result<StatusCode, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let existed =
        entry.with_engine_write(|engine| engine.delete_edge(&id, &edge_type, &from_id, &to_id))?;
    if existed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(RegistryError::EdgeNotFound {
            edge_type,
            from_id,
            to_id,
        })
    }
}

#[derive(Debug, Deserialize)]
struct SetEmbeddingBody {
    embedding: Vec<f32>,
}

/// Set an embedding and update the per-type HNSW index in lockstep.
/// Preflight order matters: dim check against any existing index
/// runs *before* the storage write, so a mismatch rejects without
/// on-disk rollback. Per-type dim is locked at first insert.
async fn set_embedding(
    State(state): State<AppState>,
    Path((id, node_type, node_id)): Path<(String, String, String)>,
    Json(body): Json<SetEmbeddingBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let SetEmbeddingBody { embedding } = body;
    entry.with_state_write(|engine, indexes| -> Result<(), RegistryError> {
        if let Some(index) = indexes.get(&node_type)
            && index.dim() != embedding.len()
        {
            return Err(RegistryError::EmbeddingDimMismatch {
                node_type: node_type.clone(),
                expected: index.dim(),
                actual: embedding.len(),
            });
        }
        engine.set_embedding(&id, &node_type, &node_id, &embedding)?;
        // Avoid the `entry()` clone on the hot post-first-insert
        // path: only allocate the key when we actually need to
        // insert. After the first insert per type, this is a single
        // get_mut.
        let index = match indexes.get_mut(&node_type) {
            Some(i) => i,
            None => indexes
                .entry(node_type.clone())
                .or_insert_with(|| HnswIndex::with_dim(embedding.len())),
        };
        index.insert(&node_id, &embedding);
        Ok(())
    })?;
    let response = EmbeddingResponse {
        node_type,
        node_id,
        embedding,
    };
    Ok(Json(response).into_response())
}

async fn get_embedding(
    State(state): State<AppState>,
    Path((id, node_type, node_id)): Path<(String, String, String)>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let embedding = entry
        .with_engine_read(|engine| engine.get_embedding(&id, &node_type, &node_id))?
        .ok_or_else(|| RegistryError::EmbeddingNotFound {
            node_type: node_type.clone(),
            node_id: node_id.clone(),
        })?;
    let response = EmbeddingResponse {
        node_type,
        node_id,
        embedding,
    };
    Ok(Json(response).into_response())
}

async fn delete_embedding(
    State(state): State<AppState>,
    Path((id, node_type, node_id)): Path<(String, String, String)>,
) -> Result<StatusCode, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let existed = entry.with_state_write(|engine, indexes| -> Result<bool, RegistryError> {
        let existed = engine.delete_embedding(&id, &node_type, &node_id)?;
        if existed && let Some(index) = indexes.get_mut(&node_type) {
            index.remove(&node_id);
        }
        Ok(existed)
    })?;
    if existed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(RegistryError::EmbeddingNotFound { node_type, node_id })
    }
}

#[derive(Debug, Deserialize)]
struct SimilarBody {
    embedding: Vec<f32>,
    top_k: usize,
    node_type: String,
}

/// HNSW vector search over the per-type index. `node_type` is
/// required: per-type indexes can have different dimensions (set
/// independently by the first `set_embedding` for each type), so a
/// merged "search all types" answer is ambiguous about score
/// comparability. If a real consumer needs cross-type search later,
/// add it as an explicit second route.
///
/// If no index exists for `node_type` (no embedding has ever been
/// set for any node of that type), returns an empty result list —
/// the type-name is honest, just no data to search yet.
async fn similar(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<SimilarBody>,
) -> Result<Response, RegistryError> {
    let entry = graph_entry(&state, &id)?;
    let SimilarBody {
        embedding,
        top_k,
        node_type,
    } = body;
    if embedding.is_empty() {
        return Err(RegistryError::BadRequest(
            "embedding must be non-empty".to_string(),
        ));
    }
    if top_k == 0 {
        return Err(RegistryError::BadRequest("top_k must be > 0".to_string()));
    }
    let response = entry.with_state_read(
        |engine, indexes| -> Result<SimilarResponse, RegistryError> {
            if !engine.schema().node_types.contains_key(&node_type) {
                return Err(RegistryError::BadRequest(format!(
                    "unknown node type: {node_type}"
                )));
            }
            let Some(index) = indexes.get(&node_type) else {
                return Ok(SimilarResponse {
                    results: Vec::new(),
                });
            };
            if index.dim() != embedding.len() {
                return Err(RegistryError::EmbeddingDimMismatch {
                    node_type: node_type.clone(),
                    expected: index.dim(),
                    actual: embedding.len(),
                });
            }
            let results = index
                .search(&embedding, top_k)
                .into_iter()
                .map(|sr| SimilarHit {
                    node_id: sr.id,
                    score: sr.score,
                })
                .collect();
            Ok(SimilarResponse { results })
        },
    )?;
    Ok(Json(response).into_response())
}
