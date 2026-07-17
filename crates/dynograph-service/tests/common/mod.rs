//! Shared test fixtures, app builders, request helpers, and assertion
//! utilities for the integration-test suite. Split out of the monolithic
//! `integration.rs`; each `tests/it_*.rs` file does `mod common; use common::*;`.
//!
//! `#![allow(dead_code, unused_imports)]` because each test binary is a separate
//! crate that uses only a subset of these helpers and re-exported imports, so the
//! unused ones are dead / unused *in that binary* (but live in others).
#![allow(dead_code, unused_imports)]

pub use std::sync::Arc;

pub use axum::{
    body::Body,
    http::{Request, StatusCode},
};
pub use http_body_util::BodyExt;
pub use serde_json::{Value, json};
pub use tower::ServiceExt;

pub use dynograph_service::{
    AppState, AuthProvider, BearerJwt, GraphRegistry, NoAuth, Readiness, ServerLimits, app,
};

pub fn build_app() -> axum::Router {
    let registry = Arc::new(GraphRegistry::new());
    app(AppState::with_no_auth(registry))
}

/// The message from a parsed error body. Every error response is the
/// JSON envelope `{ "error": "<message>" }`, so the assertions read the
/// `error` field rather than the whole `Value`.
pub fn err_msg(resp: &Value) -> &str {
    resp["error"].as_str().unwrap_or("")
}

/// Schema with one node type carrying a default-valued property
/// (covers C3 at the HTTP level) and one edge type carrying a
/// default-valued property (covers the same default-application path
/// at the edge layer for slice 3).
pub fn item_schema_body() -> Value {
    json!({
        "id": "g1",
        "schema": {
            "name": "demo",
            "version": 1,
            "node_types": {
                "Item": {
                    "properties": {
                        "name": { "type": "string", "required": true },
                        "tier": { "type": "string", "default": "standard" }
                    }
                }
            },
            "edge_types": {
                "Likes": {
                    "from": "Item",
                    "to": "Item",
                    "properties": {
                        "weight": { "type": "float" },
                        "source": { "type": "string" }
                    }
                }
            }
        }
    })
}

pub async fn create_item(app: &axum::Router, node_id: &str) {
    let body = json!({
        "node_type": "Item",
        "node_id": node_id,
        "properties": { "name": node_id }
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED, "create_item({node_id})");
}

pub async fn build_app_with_item_graph() -> axum::Router {
    let app = build_app();
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .body(Body::from(item_schema_body().to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    app
}

/// Schema for slice 6 node-list tests — `Item` carries one indexed
/// string property (`tag`), one indexed int property (`level`), one
/// non-indexed float property (`score`, used to verify the unsupported-
/// type 400 path). `name` stays required to keep create-call ergonomics.
pub fn indexed_item_schema_body() -> Value {
    json!({
        "id": "g1",
        "schema": {
            "name": "demo",
            "version": 1,
            "node_types": {
                "Item": {
                    "properties": {
                        "name":  { "type": "string", "required": true },
                        "tag":   { "type": "string", "indexed": true },
                        "level": { "type": "int",    "indexed": true },
                        "score": { "type": "float",  "indexed": true }
                    }
                }
            },
            "edge_types": {}
        }
    })
}

pub async fn build_app_with_indexed_graph() -> axum::Router {
    let app = build_app();
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .body(Body::from(indexed_item_schema_body().to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    app
}

pub async fn create_indexed_item(app: &axum::Router, id: &str, tag: &str, level: i64) {
    let body = json!({
        "node_type": "Item",
        "node_id": id,
        "properties": { "name": id, "tag": tag, "level": level }
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        res.status(),
        StatusCode::CREATED,
        "create_indexed_item({id})"
    );
}

pub async fn get_node_list(app: &axum::Router, query: &str) -> (StatusCode, Value) {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/graphs/g1/nodes?{query}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, body)
}

/// Body shape: PUT takes a bare `Schema` (id is in URL; wire_version
/// and content_hash are server-derived). Helper folds the construction
/// of the inner schema (no `{schema: ...}` envelope).
pub fn put_schema(node_types: Value, edge_types: Value) -> Value {
    json!({
        "name": "demo",
        "version": 2,
        "node_types": node_types,
        "edge_types": edge_types,
    })
}

pub async fn put_g1_schema(app: &axum::Router, schema: Value) -> (StatusCode, String) {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/graphs/g1/schema")
                .header("content-type", "application/json")
                .body(Body::from(schema.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8_lossy(&bytes).into_owned();
    (status, body)
}

pub async fn put_embedding(
    app: &axum::Router,
    node_type: &str,
    node_id: &str,
    embedding: &[f32],
) -> StatusCode {
    let body = json!({ "embedding": embedding });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri(format!(
                    "/v1/graphs/g1/nodes/{node_type}/{node_id}/embedding"
                ))
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    res.status()
}

pub async fn get_embedding(
    app: &axum::Router,
    node_type: &str,
    node_id: &str,
) -> (StatusCode, Value) {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!(
                    "/v1/graphs/g1/nodes/{node_type}/{node_id}/embedding"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, body)
}

pub async fn delete_embedding(app: &axum::Router, node_type: &str, node_id: &str) -> StatusCode {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!(
                    "/v1/graphs/g1/nodes/{node_type}/{node_id}/embedding"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    res.status()
}

pub async fn post_similar(app: &axum::Router, body: Value) -> (StatusCode, Value) {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/similar")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let parsed = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, parsed)
}

pub const TEST_JWT_SECRET: &[u8] = b"slice-9-test-signing-secret";

pub fn build_app_with_bearer_jwt() -> axum::Router {
    let registry = Arc::new(GraphRegistry::new());
    let auth: Arc<dyn AuthProvider> = Arc::new(BearerJwt::new(TEST_JWT_SECRET));
    let state = AppState::new(registry, auth, Arc::new(Readiness::ready()));
    app(state)
}

pub fn mint_jwt(sub: &str) -> String {
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use serde::Serialize;
    use std::time::{SystemTime, UNIX_EPOCH};
    #[derive(Serialize)]
    struct C<'a> {
        sub: &'a str,
        exp: usize,
    }
    let exp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as usize
        + 60;
    encode(
        &Header::new(Algorithm::HS256),
        &C { sub, exp },
        &EncodingKey::from_secret(TEST_JWT_SECRET),
    )
    .unwrap()
}

pub async fn fetch_metrics(app: &axum::Router) -> (StatusCode, String) {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

pub async fn post_batch(app: &axum::Router, body: Value) -> (StatusCode, Value) {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/batch")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    // Empty body (e.g. plain-text RegistryError) is still useful to surface
    // for debugging — wrap as a string Value so callers can assert on it.
    let parsed: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, parsed)
}

pub async fn node_exists(app: &axum::Router, node_type: &str, node_id: &str) -> bool {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/graphs/g1/nodes/{node_type}/{node_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    res.status() == StatusCode::OK
}

pub async fn edge_exists(app: &axum::Router, edge_type: &str, from_id: &str, to_id: &str) -> bool {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/graphs/g1/edges/{edge_type}/{from_id}/{to_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    res.status() == StatusCode::OK
}

/// Schema with a resolvable Character type:
/// - `name`: required string (the resolution query lives here)
/// - `story_id`: indexed string (so `scope: {prop: story_id, value: X}` works)
/// - `resolution` config with `fuzzy_then_vector` strategy + audit-cited
///   thresholds (95 auto-merge / 70 fuzzy floor / 0.85 vector cutoff)
pub fn character_schema_body() -> Value {
    json!({
        "id": "g1",
        "schema": {
            "name": "demo",
            "version": 1,
            "node_types": {
                "Character": {
                    "properties": {
                        "name":     {"type": "string", "required": true},
                        "story_id": {"type": "string", "indexed": true},
                        // JSON-array-encoded alternate names — exercises the
                        // stored-alias side of resolve-or-create.
                        "aliases":  {"type": "string"}
                    },
                    "resolution": {
                        "strategy": "fuzzy_then_vector",
                        "fuzzy_threshold": 70,
                        "vector_threshold": 0.85,
                        // High auto-merge so the fuzzy-zone tests have
                        // a wide [70, 99) window without near-misses
                        // bypassing into auto_merge. Tests behavior of
                        // the route, not realistic threshold tuning.
                        "auto_merge_threshold": 99
                    }
                },
                // No `resolution` block — used for the "type without
                // resolution config" rejection test.
                "Tag": {
                    "properties": {
                        "name": {"type": "string", "required": true}
                    }
                }
            },
            "edge_types": {}
        }
    })
}

pub async fn build_app_with_character_graph() -> axum::Router {
    let app = build_app();
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .body(Body::from(character_schema_body().to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    app
}

pub async fn create_character(app: &axum::Router, node_id: &str, name: &str, story_id: &str) {
    let body = json!({
        "node_type": "Character",
        "node_id": node_id,
        "properties": {"name": name, "story_id": story_id}
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        res.status(),
        StatusCode::CREATED,
        "create_character({node_id})"
    );
}

pub async fn post_resolve(app: &axum::Router, body: Value) -> (StatusCode, Value) {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/resolve-or-create")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let parsed: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, parsed)
}

/// Multi-type schema for edges:collect tests. Three node types
/// (Character, Event, Location) all carry an indexed `story_id` for
/// scope-filter tests; three edge types with varied endpoint shapes:
/// - MENTIONS: Character → (Character | Event | Location)  (Multiple)
/// - VISITS:   Character → Location                         (Single)
/// - INVOLVES: Event → *                                    (wildcard)
pub fn knowledge_graph_schema_body() -> Value {
    json!({
        "id": "g1",
        "schema": {
            "name": "demo",
            "version": 1,
            "node_types": {
                "Character": {
                    "properties": {
                        "name":     {"type": "string", "required": true},
                        "story_id": {"type": "string", "indexed": true}
                    }
                },
                "Event": {
                    "properties": {
                        "name":     {"type": "string", "required": true},
                        "story_id": {"type": "string", "indexed": true}
                    }
                },
                "Location": {
                    "properties": {
                        "name":     {"type": "string", "required": true},
                        "story_id": {"type": "string", "indexed": true}
                    }
                }
            },
            "edge_types": {
                "MENTIONS": {
                    "from": "Character",
                    "to":   ["Character", "Event", "Location"]
                },
                "VISITS": {
                    "from": "Character",
                    "to":   "Location"
                },
                "INVOLVES": {
                    "from": "Event",
                    "to":   "*"
                }
            }
        }
    })
}

pub async fn build_app_with_knowledge_graph() -> axum::Router {
    let app = build_app();
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .body(Body::from(knowledge_graph_schema_body().to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    app
}

pub async fn create_typed_node(
    app: &axum::Router,
    node_type: &str,
    node_id: &str,
    name: &str,
    story_id: &str,
) {
    let body = json!({
        "node_type": node_type,
        "node_id": node_id,
        "properties": {"name": name, "story_id": story_id}
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        res.status(),
        StatusCode::CREATED,
        "create_typed_node({node_type}/{node_id})"
    );
}

pub async fn create_typed_edge(
    app: &axum::Router,
    edge_type: &str,
    from_type: &str,
    from_id: &str,
    to_type: &str,
    to_id: &str,
) {
    let body = json!({
        "edge_type": edge_type,
        "from_type": from_type, "from_id": from_id,
        "to_type":   to_type,   "to_id":   to_id,
        "properties": {}
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/edges")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        res.status(),
        StatusCode::CREATED,
        "create_typed_edge({edge_type} {from_id}->{to_id})"
    );
}

pub async fn post_collect(app: &axum::Router, body: Value) -> (StatusCode, Value) {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/edges:collect")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let parsed: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, parsed)
}

/// Build a small two-story knowledge graph used by several tests.
/// Story A: char-A1 MENTIONS char-A2; char-A1 VISITS loc-A1; ev-A1
/// INVOLVES char-A1. Story B: char-B1 MENTIONS char-B2.
pub async fn seed_two_story_graph(app: &axum::Router) {
    create_typed_node(app, "Character", "char-A1", "Alice", "story-A").await;
    create_typed_node(app, "Character", "char-A2", "Bob", "story-A").await;
    create_typed_node(app, "Location", "loc-A1", "Tower", "story-A").await;
    create_typed_node(app, "Event", "ev-A1", "Duel", "story-A").await;
    create_typed_node(app, "Character", "char-B1", "Carol", "story-B").await;
    create_typed_node(app, "Character", "char-B2", "Dave", "story-B").await;

    create_typed_edge(
        app,
        "MENTIONS",
        "Character",
        "char-A1",
        "Character",
        "char-A2",
    )
    .await;
    create_typed_edge(app, "VISITS", "Character", "char-A1", "Location", "loc-A1").await;
    create_typed_edge(app, "INVOLVES", "Event", "ev-A1", "Character", "char-A1").await;
    create_typed_edge(
        app,
        "MENTIONS",
        "Character",
        "char-B1",
        "Character",
        "char-B2",
    )
    .await;
}

pub async fn post_adjacent(app: &axum::Router, body: Value) -> (StatusCode, Value) {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/edges:adjacent")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let parsed: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, parsed)
}

/// Collect (edge_type, from_id, to_id) triples from an edges:adjacent response.
pub fn adjacent_triples(resp: &Value) -> std::collections::HashSet<(String, String, String)> {
    resp["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            (
                e["edge_type"].as_str().unwrap().to_string(),
                e["from_id"].as_str().unwrap().to_string(),
                e["to_id"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

/// Schema mirroring a temporal use case: NarrativeEpoch
/// nodes joined by PRECEDES edges, scoped by `story_id`. Plus a Tag
/// node type with a TAGS edge so the multi-step / multi-type tests
/// have somewhere to chain to.
pub fn temporal_schema_body() -> Value {
    json!({
        "id": "g1",
        "schema": {
            "name": "temporal-demo",
            "version": 1,
            "node_types": {
                "NarrativeEpoch": {
                    "properties": {
                        "name":     {"type": "string", "required": true},
                        "story_id": {"type": "string", "indexed": true}
                    }
                },
                "Tag": {
                    "properties": {
                        "name":     {"type": "string", "required": true},
                        "story_id": {"type": "string", "indexed": true}
                    }
                }
            },
            "edge_types": {
                "PRECEDES": {
                    "from": "NarrativeEpoch",
                    "to":   "NarrativeEpoch"
                },
                "TAGS": {
                    "from": "NarrativeEpoch",
                    "to":   "Tag"
                }
            }
        }
    })
}

pub async fn build_app_with_temporal_schema() -> axum::Router {
    let app = build_app();
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .body(Body::from(temporal_schema_body().to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    app
}

pub async fn post_traverse(app: &axum::Router, body: Value) -> (StatusCode, Value) {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/traverse")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let parsed: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, parsed)
}

/// Seed three linear-chained PRECEDES epochs in story-A
/// (e1 → e2 → e3) plus an unrelated story-B epoch (e4 → e5), and a
/// TAGS edge from e2 → tag-A1 for the multi-step / multi-type tests.
pub async fn seed_temporal_graph(app: &axum::Router) {
    create_typed_node(app, "NarrativeEpoch", "e1", "Beginning", "story-A").await;
    create_typed_node(app, "NarrativeEpoch", "e2", "Middle", "story-A").await;
    create_typed_node(app, "NarrativeEpoch", "e3", "End", "story-A").await;
    create_typed_node(app, "NarrativeEpoch", "e4", "B-Beginning", "story-B").await;
    create_typed_node(app, "NarrativeEpoch", "e5", "B-End", "story-B").await;
    create_typed_node(app, "Tag", "tag-A1", "important", "story-A").await;

    create_typed_edge(
        app,
        "PRECEDES",
        "NarrativeEpoch",
        "e1",
        "NarrativeEpoch",
        "e2",
    )
    .await;
    create_typed_edge(
        app,
        "PRECEDES",
        "NarrativeEpoch",
        "e2",
        "NarrativeEpoch",
        "e3",
    )
    .await;
    create_typed_edge(
        app,
        "PRECEDES",
        "NarrativeEpoch",
        "e4",
        "NarrativeEpoch",
        "e5",
    )
    .await;
    create_typed_edge(app, "TAGS", "NarrativeEpoch", "e2", "Tag", "tag-A1").await;
}

/// Schema where `name` is `indexed: true` on every type — the
/// precondition for nodes:exists. Distinct from `knowledge_graph_schema_body`
/// (where `name` is intentionally un-indexed to feed the
/// rejection-path tests).
pub fn indexed_name_schema_body() -> Value {
    json!({
        "id": "g1",
        "schema": {
            "name": "indexed-name-demo",
            "version": 1,
            "node_types": {
                "Geography": {
                    "properties": {
                        "name": {"type": "string", "required": true, "indexed": true}
                    }
                },
                "Commodity": {
                    "properties": {
                        "name": {"type": "string", "required": true, "indexed": true}
                    }
                }
            },
            "edge_types": {}
        }
    })
}

pub async fn build_app_with_indexed_name_schema() -> axum::Router {
    let app = build_app();
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .body(Body::from(indexed_name_schema_body().to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    app
}

pub async fn post_exists(app: &axum::Router, body: Value) -> (StatusCode, Value) {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/nodes:exists")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let parsed: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, parsed)
}

pub async fn create_named_node(app: &axum::Router, node_type: &str, node_id: &str, name: &str) {
    let body = json!({
        "node_type": node_type,
        "node_id": node_id,
        "properties": {"name": name}
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        res.status(),
        StatusCode::CREATED,
        "create_named_node({node_type}/{node_id})"
    );
}

/// Schema for nodes:scan tests: a Person type with four indexed
/// properties spanning the indexable type variants (string, int,
/// bool) so range / eq / in / neq can all be exercised. `bio` is
/// declared but un-indexed so the "reject un-indexed property" path
/// has a target.
pub fn person_scan_schema_body() -> Value {
    json!({
        "id": "g1",
        "schema": {
            "name": "person-scan-demo",
            "version": 1,
            "node_types": {
                "Person": {
                    "properties": {
                        "name":            {"type": "string", "required": true, "indexed": true},
                        "age":             {"type": "int",                       "indexed": true},
                        "influence_level": {"type": "string",                    "indexed": true},
                        "verified":        {"type": "bool",                      "indexed": true},
                        "bio":             {"type": "string"}
                    }
                }
            },
            "edge_types": {}
        }
    })
}

pub async fn build_app_with_person_scan_schema() -> axum::Router {
    let app = build_app();
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .body(Body::from(person_scan_schema_body().to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    app
}

pub async fn post_scan(app: &axum::Router, body: Value) -> (StatusCode, Value) {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/nodes:scan")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let parsed: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, parsed)
}

pub async fn create_person(
    app: &axum::Router,
    node_id: &str,
    name: &str,
    age: i64,
    influence: &str,
    verified: bool,
) {
    let body = json!({
        "node_type": "Person",
        "node_id": node_id,
        "properties": {
            "name": name,
            "age": age,
            "influence_level": influence,
            "verified": verified,
            "bio": format!("{} the {}", name, influence)
        }
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        res.status(),
        StatusCode::CREATED,
        "create_person({node_id})"
    );
}

pub async fn seed_people(app: &axum::Router) {
    create_person(app, "p1", "Alice", 30, "market_moving", true).await;
    create_person(app, "p2", "Bob", 45, "background", true).await;
    create_person(app, "p3", "Carol", 25, "market_moving", false).await;
    create_person(app, "p4", "Dave", 60, "leading", true).await;
    create_person(app, "p5", "Eve", 50, "background", false).await;
}

pub fn id_set(resp: &Value) -> std::collections::HashSet<String> {
    resp["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.get("node_id")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| r.as_str().unwrap().to_string())
        })
        .collect()
}

pub fn ids_of(strs: &[&str]) -> std::collections::HashSet<String> {
    strs.iter().map(|s| s.to_string()).collect()
}

pub fn welford_schema_body() -> Value {
    json!({
        "id": "g1",
        "schema": {
            "name": "welford-demo",
            "version": 1,
            "node_types": {
                "Indicator": {
                    "properties": {
                        "name": {"type": "string", "required": true}
                    }
                }
            },
            "edge_types": {
                "CAUSES": {
                    "from": "Indicator",
                    "to":   "Indicator"
                }
            }
        }
    })
}

pub async fn build_app_with_welford_schema() -> axum::Router {
    let app = build_app();
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .body(Body::from(welford_schema_body().to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    app
}

pub async fn create_indicator(app: &axum::Router, id: &str, name: &str) {
    let body = json!({
        "node_type": "Indicator",
        "node_id": id,
        "properties": {"name": name}
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
}

pub async fn create_causes_edge(app: &axum::Router, from: &str, to: &str, extra_props: Value) {
    let body = json!({
        "edge_type": "CAUSES",
        "from_type": "Indicator", "from_id": from,
        "to_type":   "Indicator", "to_id":   to,
        "properties": extra_props
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/edges")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
}

pub async fn post_welford(
    app: &axum::Router,
    edge_type: &str,
    from: &str,
    to: &str,
    body: Value,
) -> (StatusCode, Value) {
    let uri = format!("/v1/graphs/g1/edges/{edge_type}/{from}/{to}/welford_update");
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let parsed: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, parsed)
}

pub async fn get_edge_props(app: &axum::Router, edge_type: &str, from: &str, to: &str) -> Value {
    let uri = format!("/v1/graphs/g1/edges/{edge_type}/{from}/{to}");
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

pub fn approx_f64(v: &Value, want: f64) {
    let got = v
        .as_f64()
        .unwrap_or_else(|| panic!("expected number, got {v}"));
    assert!((got - want).abs() < 1e-9, "approx: got {got}, want {want}");
}

pub fn build_app_with_limits(limits: ServerLimits) -> axum::Router {
    let registry = Arc::new(GraphRegistry::new());
    app(AppState::with_no_auth(registry).with_limits(limits))
}

pub fn json_post(uri: &str, body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// Schema with two full-text string properties on `Document`.
pub fn fulltext_graph_body() -> Value {
    json!({
        "id": "g1",
        "schema": {
            "name": "demo",
            "version": 1,
            "node_types": {
                "Document": { "properties": {
                    "title": { "type": "string", "fulltext": true },
                    "body":  { "type": "string", "fulltext": true }
                }}
            },
            "edge_types": {}
        }
    })
}

#[cfg(feature = "graph")]
pub async fn post_algo(app: &axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let res = app
        .clone()
        .oneshot(json_post(&format!("/v1/graphs/g1/algo/{path}"), &body))
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let parsed: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, parsed)
}

// algo/communities (#24) — Leiden. Build a connected two-faction graph (two
// Character triangles joined by a single bridge MENTIONS edge) so the split is
// a genuine within-component community partition, not just two components.
#[cfg(feature = "graph")]
pub async fn seed_two_faction_graph(app: &axum::Router) {
    for id in ["a1", "a2", "a3", "b1", "b2", "b3"] {
        create_typed_node(app, "Character", id, id, "s").await;
    }
    for (x, y) in [
        ("a1", "a2"),
        ("a2", "a3"),
        ("a3", "a1"),
        ("b1", "b2"),
        ("b2", "b3"),
        ("b3", "b1"),
        ("a1", "b1"), // the single bridge
    ] {
        create_typed_edge(app, "MENTIONS", "Character", x, "Character", y).await;
    }
}

// Helper: pull a single node's score out of a ScoresResponse body.
#[cfg(feature = "graph")]
pub fn score_of(resp: &Value, node: &str) -> f64 {
    resp["scores"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["node"] == node)
        .unwrap_or_else(|| panic!("node {node} not in scores: {resp}"))["score"]
        .as_f64()
        .unwrap()
}

pub async fn post_util(app: &axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let res = app
        .clone()
        .oneshot(json_post(&format!("/v1/util/{path}"), &body))
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let parsed: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, parsed)
}

/// A Document type that is full-text searchable (`title`), has an indexed
/// scalar for the structured prefilter (`act`), and carries embeddings —
/// so all three legs/inputs can be exercised against one graph.
#[cfg(feature = "fulltext")]
pub fn hybrid_graph_body() -> Value {
    json!({
        "id": "g1",
        "schema": {
            "name": "demo",
            "version": 1,
            "node_types": {
                "Document": { "properties": {
                    "title": { "type": "string", "fulltext": true },
                    "act":   { "type": "int", "indexed": true }
                }}
            },
            "edge_types": {}
        }
    })
}

#[cfg(feature = "fulltext")]
pub async fn post_hybrid(app: &axum::Router, body: Value) -> (StatusCode, Value) {
    let res = app
        .clone()
        .oneshot(json_post("/v1/graphs/g1/search:hybrid", &body))
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let parsed = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, parsed)
}

#[cfg(feature = "fulltext")]
pub async fn create_doc(
    app: &axum::Router,
    id: &str,
    title: &str,
    act: i64,
    embedding: Option<&[f32]>,
) {
    let node = json!({
        "node_type": "Document",
        "node_id": id,
        "properties": { "title": title, "act": act }
    });
    let res = app
        .clone()
        .oneshot(json_post("/v1/graphs/g1/nodes", &node))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED, "create_doc({id})");
    if let Some(e) = embedding {
        assert_eq!(
            put_embedding(app, "Document", id, e).await,
            StatusCode::OK,
            "embed({id})"
        );
    }
}

/// Set up `g1` with three Documents that isolate each leg. The HNSW leg
/// returns *every* embedded node (an orthogonal vector still comes back at
/// score 0), so "vector-only" means "has an embedding but no keyword match"
/// and "keyword-only" means "matches the query but has no embedding":
/// - `d1` — keyword match **and** the query vector → both legs.
/// - `d2` — embedding but no `alpha` token → vector leg only.
/// - `d3` — `alpha` token but no embedding → keyword leg only.
#[cfg(feature = "fulltext")]
pub async fn hybrid_fixture() -> axum::Router {
    let app = build_app();
    let res = app
        .clone()
        .oneshot(json_post("/v1/graphs", &hybrid_graph_body()))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    create_doc(&app, "d1", "alpha keyword", 1, Some(&[1.0, 0.0, 0.0])).await;
    create_doc(&app, "d2", "gamma delta", 2, Some(&[0.0, 1.0, 0.0])).await;
    create_doc(&app, "d3", "alpha solo", 1, None).await;
    app
}

pub fn pd_body() -> Value {
    // Prisoner's dilemma. Strategy 0 = cooperate, 1 = defect.
    // (C,C)=3,3  (C,D)=0,5  (D,C)=5,0  (D,D)=1,1
    json!({
        "players": [
            {"strategies": ["cooperate", "defect"]},
            {"strategies": ["cooperate", "defect"]}
        ],
        "payoffs": [
            {"profile": [0, 0], "utilities": [3.0, 3.0]},
            {"profile": [0, 1], "utilities": [0.0, 5.0]},
            {"profile": [1, 0], "utilities": [5.0, 0.0]},
            {"profile": [1, 1], "utilities": [1.0, 1.0]}
        ]
    })
}
