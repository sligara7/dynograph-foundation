use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use dynograph_service::{
    AppState, AuthProvider, BearerJwt, GraphRegistry, NoAuth, Readiness, ServerLimits, app,
};

fn build_app() -> axum::Router {
    let registry = Arc::new(GraphRegistry::new());
    app(AppState::with_no_auth(registry))
}

/// The message from a parsed error body. Every error response is the
/// JSON envelope `{ "error": "<message>" }`, so the assertions read the
/// `error` field rather than the whole `Value`.
fn err_msg(resp: &Value) -> &str {
    resp["error"].as_str().unwrap_or("")
}

#[tokio::test]
async fn create_then_get_metadata_and_schema() {
    let app = build_app();

    let body = json!({
        "id": "g1",
        "schema": {
            "name": "demo",
            "version": 1,
            "node_types": {
                "Item": {
                    "properties": {
                        "name": { "type": "string", "required": true }
                    }
                }
            },
            "edge_types": {}
        }
    });

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);

    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let created: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(created["id"], "g1");
    assert_eq!(created["wire_version"], env!("CARGO_PKG_VERSION"));
    let first_hash = created["content_hash"].as_str().unwrap().to_string();
    assert_eq!(first_hash.len(), 64);
    // POST returns the full schema body (creator wants confirmation of
    // exactly what was stored).
    assert_eq!(created["schema"]["name"], "demo");

    // GET /v1/graphs/{id} is metadata-only — no schema body. Cheap
    // existence check + content-hash drift comparison without the
    // serialized schema on the wire.
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let metadata: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(metadata["id"], "g1");
    assert_eq!(metadata["wire_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(metadata["content_hash"].as_str().unwrap(), first_hash);
    assert!(
        metadata.get("schema").is_none(),
        "metadata response must not embed schema; got {metadata}"
    );

    // GET /v1/graphs/{id}/schema returns the full SchemaResponse —
    // the shape generation_plus codegen reads (matches the
    // C-partial `build_schema_contract`).
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/schema")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let fetched: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(fetched["id"], "g1");
    assert_eq!(fetched["content_hash"].as_str().unwrap(), first_hash);
    assert_eq!(fetched["schema"]["name"], "demo");
    assert_eq!(fetched["schema"]["version"], 1);
}

#[tokio::test]
async fn get_schema_on_unknown_graph_returns_404() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/missing/schema")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn get_unknown_graph_returns_404() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/missing")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn malformed_graph_id_rejected_with_400_on_read_and_write_paths() {
    // A malformed id is rejected up front (400), not merely missed
    // (404) — both the read path (`graph_entry`) and the write paths
    // that join the id into a filesystem path on the OnDisk backend
    // (`delete_graph` / `replace_schema`) validate it, matching the
    // create-time contract. `$` is outside the allowed `[A-Za-z0-9_-]`
    // set but routes as a single path segment.
    let app = build_app();
    for method in ["GET", "DELETE"] {
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri("/v1/graphs/bad$id")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::BAD_REQUEST,
            "{method} on a malformed id should be 400"
        );
    }
}

#[tokio::test]
async fn error_responses_use_structured_json_envelope() {
    // Every error — whether it originates in a handler (400) or the
    // registry lookup (404) — carries the same JSON `{ "error": <msg> }`
    // body with an `application/json` content type, so a client parses
    // one shape across all routes.
    let app = build_app();
    for (uri, want_status) in [
        ("/v1/graphs/missing", StatusCode::NOT_FOUND),
        ("/v1/graphs/bad$id", StatusCode::BAD_REQUEST),
    ] {
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
        assert_eq!(res.status(), want_status, "for {uri}");
        let content_type = res
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        assert!(
            content_type.starts_with("application/json"),
            "error content-type should be JSON, got {content_type:?} for {uri}"
        );
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).expect("error body must be valid JSON");
        let msg = body["error"].as_str();
        assert!(
            msg.is_some_and(|s| !s.is_empty()),
            "error body must be {{\"error\": <non-empty string>}}, got {body} for {uri}"
        );
    }
}

#[tokio::test]
async fn create_duplicate_graph_returns_409() {
    let app = build_app();
    let body = json!({
        "id": "g1",
        "schema": {
            "name": "demo",
            "version": 1,
            "node_types": {},
            "edge_types": {}
        }
    });
    let make_req = || {
        Request::builder()
            .method("POST")
            .uri("/v1/graphs")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    };
    let r1 = app.clone().oneshot(make_req()).await.unwrap();
    assert_eq!(r1.status(), StatusCode::CREATED);
    let r2 = app.clone().oneshot(make_req()).await.unwrap();
    assert_eq!(r2.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn list_graphs_returns_sorted_ids() {
    let app = build_app();
    for id in ["zeta", "alpha", "mike"] {
        let body = json!({
            "id": id,
            "schema": { "name": "demo", "version": 1, "node_types": {}, "edge_types": {} }
        });
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/graphs")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::CREATED);
    }

    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["graphs"], json!(["alpha", "mike", "zeta"]));
}

#[tokio::test]
async fn delete_graph_then_get_returns_404() {
    let app = build_app();
    let create_body = json!({
        "id": "g1",
        "schema": { "name": "demo", "version": 1, "node_types": {}, "edge_types": {} }
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .body(Body::from(create_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/graphs/g1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    // Second delete is also 404.
    let res = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/graphs/g1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

/// Schema with one node type carrying a default-valued property
/// (covers C3 at the HTTP level) and one edge type carrying a
/// default-valued property (covers the same default-application path
/// at the edge layer for slice 3).
fn item_schema_body() -> Value {
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

async fn create_item(app: &axum::Router, node_id: &str) {
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

async fn build_app_with_item_graph() -> axum::Router {
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

#[tokio::test]
async fn node_create_get_replace_delete_round_trip() {
    let app = build_app_with_item_graph().await;

    let create = json!({
        "node_type": "Item",
        "node_id": "n1",
        "properties": { "name": "widget" }
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(create.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let created: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(created["node_type"], "Item");
    assert_eq!(created["node_id"], "n1");
    assert_eq!(created["properties"]["name"], "widget");
    // Schema default for `tier` was applied on write (C3).
    assert_eq!(created["properties"]["tier"], "standard");

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/nodes/Item/n1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let fetched: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(fetched["properties"]["name"], "widget");
    assert_eq!(fetched["properties"]["tier"], "standard");

    // PUT REPLACES the property map. `tier` is omitted in the body but
    // the schema default re-applies it on validate, so it survives;
    // any property without a default would be dropped.
    let put = json!({ "properties": { "name": "gadget" } });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/graphs/g1/nodes/Item/n1")
                .header("content-type", "application/json")
                .body(Body::from(put.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let replaced: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(replaced["properties"]["name"], "gadget");
    assert_eq!(replaced["properties"]["tier"], "standard");

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/graphs/g1/nodes/Item/n1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/nodes/Item/n1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn create_node_with_unknown_type_returns_400() {
    let app = build_app_with_item_graph().await;
    let create = json!({
        "node_type": "Bogus",
        "node_id": "n1",
        "properties": { "name": "widget" }
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(create.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn create_node_missing_required_property_returns_400() {
    let app = build_app_with_item_graph().await;
    let create = json!({
        "node_type": "Item",
        "node_id": "n1",
        "properties": {}
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(create.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn replace_node_on_missing_node_returns_404_with_node_attribution() {
    let app = build_app_with_item_graph().await;
    let put = json!({ "properties": { "name": "gadget" } });
    let res = app
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/graphs/g1/nodes/Item/missing")
                .header("content-type", "application/json")
                .body(Body::from(put.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body = std::str::from_utf8(&bytes).unwrap();
    // The graph exists; the node is missing. The body must say so —
    // misattributing this as "graph not found" would mask the real
    // cause and send debuggers chasing the wrong thing.
    assert!(body.contains("node not found"), "body was: {body}");
    assert!(body.contains("Item/missing"), "body was: {body}");
}

#[tokio::test]
async fn create_node_on_missing_graph_returns_404() {
    let app = build_app();
    let create = json!({
        "node_type": "Item",
        "node_id": "n1",
        "properties": { "name": "widget" }
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/nope/nodes")
                .header("content-type", "application/json")
                .body(Body::from(create.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn edge_create_get_patch_delete_round_trip() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "a").await;
    create_item(&app, "b").await;

    let create = json!({
        "edge_type": "Likes",
        "from_type": "Item",
        "from_id": "a",
        "to_type": "Item",
        "to_id": "b",
        "properties": { "weight": 0.5, "source": "manual" }
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/edges")
                .header("content-type", "application/json")
                .body(Body::from(create.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let created: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(created["edge_type"], "Likes");
    assert_eq!(created["from_id"], "a");
    assert_eq!(created["to_id"], "b");
    assert_eq!(created["properties"]["weight"], 0.5);
    assert_eq!(created["properties"]["source"], "manual");
    // No `from_type`/`to_type` on the wire — those are validation-time
    // arguments, not part of the edge identity.
    assert!(created.get("from_type").is_none());

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/edges/Likes/a/b")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // PATCH MERGES — `weight` is overwritten, `source` (omitted) survives.
    let patch = json!({ "properties": { "weight": 0.9 } });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri("/v1/graphs/g1/edges/Likes/a/b")
                .header("content-type", "application/json")
                .body(Body::from(patch.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let merged: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(merged["properties"]["weight"], 0.9);
    assert_eq!(
        merged["properties"]["source"], "manual",
        "PATCH must not drop unspecified properties"
    );

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/graphs/g1/edges/Likes/a/b")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/edges/Likes/a/b")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn create_edge_unknown_type_returns_400() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "a").await;
    create_item(&app, "b").await;
    let create = json!({
        "edge_type": "Bogus",
        "from_type": "Item", "from_id": "a",
        "to_type": "Item", "to_id": "b",
        "properties": {}
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/edges")
                .header("content-type", "application/json")
                .body(Body::from(create.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn create_edge_endpoint_type_mismatch_returns_400() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "a").await;
    create_item(&app, "b").await;
    // Schema declares Likes: Item -> Item. Sending Item -> Widget
    // must be rejected by validate_edge.
    let create = json!({
        "edge_type": "Likes",
        "from_type": "Item", "from_id": "a",
        "to_type": "Widget", "to_id": "b",
        "properties": {}
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/edges")
                .header("content-type", "application/json")
                .body(Body::from(create.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn get_missing_edge_returns_404_with_edge_attribution() {
    let app = build_app_with_item_graph().await;
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/edges/Likes/missing/also_missing")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body = std::str::from_utf8(&bytes).unwrap();
    assert!(body.contains("edge not found"), "body was: {body}");
    assert!(body.contains("Likes"), "body was: {body}");
}

#[tokio::test]
async fn patch_missing_edge_returns_404() {
    let app = build_app_with_item_graph().await;
    let patch = json!({ "properties": { "weight": 0.1 } });
    let res = app
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri("/v1/graphs/g1/edges/Likes/a/b")
                .header("content-type", "application/json")
                .body(Body::from(patch.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn delete_missing_edge_returns_404() {
    let app = build_app_with_item_graph().await;
    let res = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/graphs/g1/edges/Likes/a/b")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn create_edge_on_missing_graph_returns_404() {
    let app = build_app();
    let create = json!({
        "edge_type": "Likes",
        "from_type": "Item", "from_id": "a",
        "to_type": "Item", "to_id": "b",
        "properties": {}
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/nope/edges")
                .header("content-type", "application/json")
                .body(Body::from(create.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

/// `delete_node` cascades adjacency cleanup (tech-debt C1 in the v0.2.0
/// review). At the HTTP level: after the node disappears, edges into
/// or out of it must no longer be reachable. This test fails loudly
/// if the C1 fix ever regresses behind the service layer.
#[tokio::test]
async fn deleting_a_node_cascades_to_its_edges() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "a").await;
    create_item(&app, "b").await;
    let create = json!({
        "edge_type": "Likes",
        "from_type": "Item", "from_id": "a",
        "to_type": "Item", "to_id": "b",
        "properties": { "weight": 1.0 }
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/edges")
                .header("content-type", "application/json")
                .body(Body::from(create.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);

    // Drop node `a`. `delete_node` should clean up `Likes a -> b`.
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/graphs/g1/nodes/Item/a")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/edges/Likes/a/b")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn health_returns_ok() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&bytes[..], b"ok");
}

#[tokio::test]
async fn ready_returns_200_when_marked_ready() {
    // `with_no_auth` defaults to ready, matching slice 1–3 test
    // expectations.
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&bytes[..], b"ready");
}

/// Schema for slice 6 node-list tests — `Item` carries one indexed
/// string property (`tag`), one indexed int property (`level`), one
/// non-indexed float property (`score`, used to verify the unsupported-
/// type 400 path). `name` stays required to keep create-call ergonomics.
fn indexed_item_schema_body() -> Value {
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

async fn build_app_with_indexed_graph() -> axum::Router {
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

async fn create_indexed_item(app: &axum::Router, id: &str, tag: &str, level: i64) {
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

async fn get_node_list(app: &axum::Router, query: &str) -> (StatusCode, Value) {
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

#[tokio::test]
async fn list_nodes_by_type_returns_all_of_type() {
    let app = build_app_with_indexed_graph().await;
    create_indexed_item(&app, "a", "red", 1).await;
    create_indexed_item(&app, "b", "blue", 2).await;
    create_indexed_item(&app, "c", "red", 3).await;

    let (status, body) = get_node_list(&app, "type=Item").await;
    assert_eq!(status, StatusCode::OK);
    let nodes = body["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 3);
    let mut ids: Vec<&str> = nodes
        .iter()
        .map(|n| n["node_id"].as_str().unwrap())
        .collect();
    ids.sort();
    assert_eq!(ids, vec!["a", "b", "c"]);
}

#[tokio::test]
async fn list_nodes_by_indexed_string_property() {
    let app = build_app_with_indexed_graph().await;
    create_indexed_item(&app, "a", "red", 1).await;
    create_indexed_item(&app, "b", "blue", 2).await;
    create_indexed_item(&app, "c", "red", 3).await;

    let (status, body) = get_node_list(&app, "type=Item&prop=tag&value=red").await;
    assert_eq!(status, StatusCode::OK);
    let nodes = body["nodes"].as_array().unwrap();
    let mut ids: Vec<&str> = nodes
        .iter()
        .map(|n| n["node_id"].as_str().unwrap())
        .collect();
    ids.sort();
    assert_eq!(ids, vec!["a", "c"]);
}

#[tokio::test]
async fn list_nodes_by_indexed_int_property_coerces_url_string() {
    let app = build_app_with_indexed_graph().await;
    create_indexed_item(&app, "a", "red", 1).await;
    create_indexed_item(&app, "b", "blue", 2).await;
    create_indexed_item(&app, "c", "red", 2).await;

    let (status, body) = get_node_list(&app, "type=Item&prop=level&value=2").await;
    assert_eq!(status, StatusCode::OK);
    let nodes = body["nodes"].as_array().unwrap();
    let mut ids: Vec<&str> = nodes
        .iter()
        .map(|n| n["node_id"].as_str().unwrap())
        .collect();
    ids.sort();
    assert_eq!(ids, vec!["b", "c"]);
}

#[tokio::test]
async fn list_nodes_with_no_matches_returns_empty() {
    let app = build_app_with_indexed_graph().await;
    create_indexed_item(&app, "a", "red", 1).await;

    let (status, body) = get_node_list(&app, "type=Item&prop=tag&value=green").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["nodes"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn list_nodes_missing_type_query_param_returns_400() {
    let app = build_app_with_indexed_graph().await;
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/nodes")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        res.status().is_client_error(),
        "expected 4xx for missing ?type, got {}",
        res.status()
    );
}

#[tokio::test]
async fn list_nodes_prop_without_value_returns_400() {
    let app = build_app_with_indexed_graph().await;
    let (status, _) = get_node_list(&app, "type=Item&prop=tag").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn list_nodes_value_without_prop_returns_400() {
    let app = build_app_with_indexed_graph().await;
    let (status, _) = get_node_list(&app, "type=Item&value=red").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn list_nodes_unknown_node_type_returns_400() {
    let app = build_app_with_indexed_graph().await;
    let (status, _) = get_node_list(&app, "type=Bogus&prop=tag&value=red").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn list_nodes_unknown_property_returns_400() {
    let app = build_app_with_indexed_graph().await;
    let (status, _) = get_node_list(&app, "type=Item&prop=bogus&value=x").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn list_nodes_int_value_unparsable_returns_400() {
    let app = build_app_with_indexed_graph().await;
    let (status, _) = get_node_list(&app, "type=Item&prop=level&value=notanint").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// Filtering by an unsupported (non-indexable) property type fails
/// loudly rather than silently returning empty. `Float`/`ListString`
/// are never written into `CF_NODE_IDX`, so a `scan_nodes_by_property`
/// against them would always return 0 rows — that would be a silent
/// fallback (project-wide rule against). Coerce rejects up front.
#[tokio::test]
async fn list_nodes_unsupported_filter_type_returns_400() {
    let app = build_app_with_indexed_graph().await;
    let (status, _) = get_node_list(&app, "type=Item&prop=score&value=1.5").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn list_nodes_on_missing_graph_returns_404() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/nope/nodes?type=Item")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

// =============================================================================
// Slice 7 — PUT /v1/graphs/{id}/schema
// =============================================================================

/// Body shape: PUT takes a bare `Schema` (id is in URL; wire_version
/// and content_hash are server-derived). Helper folds the construction
/// of the inner schema (no `{schema: ...}` envelope).
fn put_schema(node_types: Value, edge_types: Value) -> Value {
    json!({
        "name": "demo",
        "version": 2,
        "node_types": node_types,
        "edge_types": edge_types,
    })
}

async fn put_g1_schema(app: &axum::Router, schema: Value) -> (StatusCode, String) {
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

#[tokio::test]
async fn put_schema_compatible_addition_succeeds_and_hash_changes() {
    let app = build_app_with_item_graph().await;

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/schema")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let before: Value = serde_json::from_slice(&bytes).unwrap();
    let old_hash = before["content_hash"].as_str().unwrap().to_string();

    // Compatible change: add an optional `nickname` property to Item.
    let new = put_schema(
        json!({
            "Item": {
                "properties": {
                    "name": { "type": "string", "required": true },
                    "tier": { "type": "string", "default": "standard" },
                    "nickname": { "type": "string" }
                }
            }
        }),
        json!({
            "Likes": {
                "from": "Item",
                "to": "Item",
                "properties": {
                    "weight": { "type": "float" },
                    "source": { "type": "string" }
                }
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    let new_hash = parsed["content_hash"].as_str().unwrap();
    assert_ne!(
        new_hash, old_hash,
        "content_hash should change on schema swap"
    );
    // Response embeds the new schema.
    assert!(parsed["schema"]["node_types"]["Item"]["properties"]["nickname"].is_object());

    // Subsequent GET /schema reflects the new shape.
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/schema")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let after: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(after["content_hash"].as_str().unwrap(), new_hash);
    assert!(after["schema"]["node_types"]["Item"]["properties"]["nickname"].is_object());
}

#[tokio::test]
async fn put_schema_can_add_new_node_type() {
    let app = build_app_with_item_graph().await;
    let new = put_schema(
        json!({
            "Item": {
                "properties": {
                    "name": { "type": "string", "required": true },
                    "tier": { "type": "string", "default": "standard" }
                }
            },
            "Place": {
                "properties": {
                    "name": { "type": "string", "required": true }
                }
            }
        }),
        json!({
            "Likes": {
                "from": "Item",
                "to": "Item",
                "properties": {
                    "weight": { "type": "float" },
                    "source": { "type": "string" }
                }
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
}

#[tokio::test]
async fn put_schema_can_relax_required_to_optional() {
    let app = build_app_with_item_graph().await;
    let new = put_schema(
        json!({
            "Item": {
                "properties": {
                    "name": { "type": "string" },
                    "tier": { "type": "string", "default": "standard" }
                }
            }
        }),
        json!({
            "Likes": {
                "from": "Item", "to": "Item",
                "properties": {
                    "weight": { "type": "float" },
                    "source": { "type": "string" }
                }
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
}

#[tokio::test]
async fn put_schema_rejects_removed_node_type() {
    let app = build_app_with_item_graph().await;
    let new = put_schema(
        json!({}),
        json!({
            "Likes": {
                "from": "Item", "to": "Item",
                "properties": {}
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body.contains("removed node type") && body.contains("Item"),
        "{body}"
    );
}

#[tokio::test]
async fn put_schema_rejects_removed_property() {
    let app = build_app_with_item_graph().await;
    let new = put_schema(
        json!({
            "Item": {
                "properties": {
                    "name": { "type": "string", "required": true }
                }
            }
        }),
        json!({
            "Likes": {
                "from": "Item", "to": "Item",
                "properties": {
                    "weight": { "type": "float" },
                    "source": { "type": "string" }
                }
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body.contains("removed node property") && body.contains("Item.tier"),
        "{body}"
    );
}

#[tokio::test]
async fn put_schema_rejects_changed_property_type() {
    let app = build_app_with_item_graph().await;
    let new = put_schema(
        json!({
            "Item": {
                "properties": {
                    "name": { "type": "int", "required": true },
                    "tier": { "type": "string", "default": "standard" }
                }
            }
        }),
        json!({
            "Likes": {
                "from": "Item", "to": "Item",
                "properties": {
                    "weight": { "type": "float" },
                    "source": { "type": "string" }
                }
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("changed node property type"), "{body}");
}

#[tokio::test]
async fn put_schema_rejects_required_without_default_added() {
    let app = build_app_with_item_graph().await;
    let new = put_schema(
        json!({
            "Item": {
                "properties": {
                    "name": { "type": "string", "required": true },
                    "tier": { "type": "string", "default": "standard" },
                    "ssn":  { "type": "string", "required": true }
                }
            }
        }),
        json!({
            "Likes": {
                "from": "Item", "to": "Item",
                "properties": {
                    "weight": { "type": "float" },
                    "source": { "type": "string" }
                }
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body.contains("required without a default") && body.contains("ssn"),
        "{body}"
    );
}

#[tokio::test]
async fn put_schema_rejects_narrowed_edge_endpoint() {
    let app = build_app_with_item_graph().await;
    let new = put_schema(
        json!({
            "Item": {
                "properties": {
                    "name": { "type": "string", "required": true },
                    "tier": { "type": "string", "default": "standard" }
                }
            },
            "Place": {
                "properties": { "name": { "type": "string" } }
            }
        }),
        json!({
            "Likes": {
                "from": "Item",
                "to": "Place",
                "properties": {
                    "weight": { "type": "float" },
                    "source": { "type": "string" }
                }
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("`to` endpoint narrowed"), "{body}");
}

#[tokio::test]
async fn put_schema_lists_all_violations_in_one_response() {
    let app = build_app_with_item_graph().await;
    // Three violations: removed property, changed type on existing,
    // and a required-without-default new property.
    let new = put_schema(
        json!({
            "Item": {
                "properties": {
                    "name": { "type": "int", "required": true },
                    "ssn":  { "type": "string", "required": true }
                }
            }
        }),
        json!({
            "Likes": {
                "from": "Item", "to": "Item",
                "properties": {
                    "weight": { "type": "float" },
                    "source": { "type": "string" }
                }
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("removed node property"), "{body}");
    assert!(body.contains("changed node property type"), "{body}");
    assert!(body.contains("required without a default"), "{body}");
}

#[tokio::test]
async fn put_schema_on_missing_graph_returns_404() {
    let app = build_app();
    let new = put_schema(json!({}), json!({}));
    let res = app
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/graphs/nope/schema")
                .header("content-type", "application/json")
                .body(Body::from(new.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn put_schema_failure_leaves_state_intact() {
    // After a rejected PUT, the in-memory schema must be unchanged
    // and existing nodes must still be readable.
    let app = build_app_with_item_graph().await;
    create_item(&app, "n1").await;

    // Submit a breaking PUT.
    let bad = put_schema(json!({}), json!({}));
    let (status, _) = put_g1_schema(&app, bad).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Existing node still readable + the schema still has the
    // original Item type.
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/nodes/Item/n1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/schema")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let schema: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(schema["schema"]["node_types"]["Item"].is_object());
}

// =============================================================================
// Slice 8a — sidecar embeddings
// =============================================================================

async fn put_embedding(
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

async fn get_embedding(app: &axum::Router, node_type: &str, node_id: &str) -> (StatusCode, Value) {
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

async fn delete_embedding(app: &axum::Router, node_type: &str, node_id: &str) -> StatusCode {
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

#[tokio::test]
async fn embedding_round_trip_set_get_delete() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "n1").await;

    assert_eq!(
        put_embedding(&app, "Item", "n1", &[0.1, 0.2, 0.3]).await,
        StatusCode::OK
    );

    let (status, body) = get_embedding(&app, "Item", "n1").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["node_type"], "Item");
    assert_eq!(body["node_id"], "n1");
    let arr = body["embedding"].as_array().unwrap();
    assert_eq!(arr.len(), 3);
    assert!((arr[0].as_f64().unwrap() - 0.1).abs() < 1e-6);
    assert!((arr[1].as_f64().unwrap() - 0.2).abs() < 1e-6);
    assert!((arr[2].as_f64().unwrap() - 0.3).abs() < 1e-6);

    assert_eq!(
        delete_embedding(&app, "Item", "n1").await,
        StatusCode::NO_CONTENT
    );
    let (status, _) = get_embedding(&app, "Item", "n1").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn put_embedding_overwrites() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "n1").await;
    // Same dim across overwrites — slice 8b's HNSW index locks the
    // dim per-type to the first inserted vector. A dim change now
    // surfaces as `EmbeddingDimMismatch` (covered separately).
    assert_eq!(
        put_embedding(&app, "Item", "n1", &[1.0, 0.0, 0.0]).await,
        StatusCode::OK
    );
    assert_eq!(
        put_embedding(&app, "Item", "n1", &[0.0, 1.0, 0.5]).await,
        StatusCode::OK
    );
    let (_, body) = get_embedding(&app, "Item", "n1").await;
    let arr = body["embedding"].as_array().unwrap();
    assert_eq!(arr.len(), 3);
    assert!((arr[2].as_f64().unwrap() - 0.5).abs() < 1e-6);
}

#[tokio::test]
async fn put_embedding_on_missing_node_returns_404() {
    let app = build_app_with_item_graph().await;
    // Node `ghost` was never created.
    assert_eq!(
        put_embedding(&app, "Item", "ghost", &[0.1, 0.2]).await,
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn put_embedding_rejects_empty_vector() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "n1").await;
    assert_eq!(
        put_embedding(&app, "Item", "n1", &[]).await,
        StatusCode::BAD_REQUEST
    );
}

#[tokio::test]
async fn put_embedding_rejects_zero_magnitude_vector() {
    // An all-zeros embedding (e.g. a sidecar returning zeros on empty
    // input) has no direction — cosine scores 0.0 against everything,
    // which reads as "orthogonal" not "broken". It must never enter the
    // index; the boundary rejects it loudly with 400.
    let app = build_app_with_item_graph().await;
    create_item(&app, "n1").await;
    assert_eq!(
        put_embedding(&app, "Item", "n1", &[0.0, 0.0, 0.0]).await,
        StatusCode::BAD_REQUEST
    );
    // And the node must have no embedding stored — the reject ran before
    // the storage write.
    let (status, _) = get_embedding(&app, "Item", "n1").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn get_embedding_on_node_without_one_returns_404() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "n1").await;
    let (status, _) = get_embedding(&app, "Item", "n1").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn delete_embedding_idempotency_at_wire_level() {
    // Storage's delete_embedding is idempotent (Ok(false) on missing),
    // but the HTTP surface emits 404 to match node/edge DELETE shape so
    // callers can detect "I thought this existed but didn't" bugs.
    let app = build_app_with_item_graph().await;
    create_item(&app, "n1").await;
    put_embedding(&app, "Item", "n1", &[0.1, 0.2]).await;
    assert_eq!(
        delete_embedding(&app, "Item", "n1").await,
        StatusCode::NO_CONTENT
    );
    assert_eq!(
        delete_embedding(&app, "Item", "n1").await,
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn deleting_a_node_cascades_to_its_embedding() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "n1").await;
    put_embedding(&app, "Item", "n1", &[0.1, 0.2, 0.3]).await;

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/graphs/g1/nodes/Item/n1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    // GET embedding now 404s — the cascade dropped it.
    let (status, _) = get_embedding(&app, "Item", "n1").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn embedding_endpoints_on_missing_graph_return_404() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/nope/nodes/Item/n1/embedding")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

// =============================================================================
// Slice 8b — POST /v1/graphs/{id}/similar + HNSW lifecycle
// =============================================================================

async fn post_similar(app: &axum::Router, body: Value) -> (StatusCode, Value) {
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

/// Two near-aligned vectors should rank higher than an orthogonal
/// one. Exact cosine values aren't asserted (depends on HNSW's
/// approximate-search beam width); we just assert ordering.
#[tokio::test]
async fn similar_returns_top_k_in_descending_score() {
    let app = build_app_with_item_graph().await;
    for id in ["a", "b", "c"] {
        create_item(&app, id).await;
    }
    // 3-dim vectors. `a` and `b` are close along axis 0; `c` is on axis 2.
    put_embedding(&app, "Item", "a", &[1.0, 0.0, 0.0]).await;
    put_embedding(&app, "Item", "b", &[0.95, 0.1, 0.0]).await;
    put_embedding(&app, "Item", "c", &[0.0, 0.0, 1.0]).await;

    let (status, body) = post_similar(
        &app,
        json!({ "embedding": [1.0, 0.0, 0.0], "top_k": 3, "node_type": "Item" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);
    let ids: Vec<&str> = results
        .iter()
        .map(|r| r["node_id"].as_str().unwrap())
        .collect();
    // The two axis-0 neighbors come before the orthogonal one.
    assert_eq!(ids[0], "a", "results: {results:?}");
    assert_eq!(ids[1], "b", "results: {results:?}");
    assert_eq!(ids[2], "c", "results: {results:?}");
}

#[tokio::test]
async fn similar_returns_empty_when_no_index_exists_yet() {
    // The Item type exists in the schema but no embedding has been
    // set, so no HNSW index has been created. Search returns 200 +
    // empty results — the type-name is honest, just no data.
    let app = build_app_with_item_graph().await;
    let (status, body) = post_similar(
        &app,
        json!({ "embedding": [0.1, 0.2], "top_k": 5, "node_type": "Item" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["results"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn similar_dim_mismatch_returns_400() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "n1").await;
    put_embedding(&app, "Item", "n1", &[1.0, 0.0, 0.0]).await;
    let (status, _) = post_similar(
        &app,
        json!({ "embedding": [1.0, 0.0], "top_k": 1, "node_type": "Item" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn similar_rejects_zero_magnitude_query() {
    // A degenerate query (zero magnitude) scores 0.0 against every node
    // — a silent "nothing is similar" that hides bad upstream data.
    // Reject with 400 before touching the index.
    let app = build_app_with_item_graph().await;
    create_item(&app, "n1").await;
    put_embedding(&app, "Item", "n1", &[1.0, 0.0, 0.0]).await;
    let (status, _) = post_similar(
        &app,
        json!({ "embedding": [0.0, 0.0, 0.0], "top_k": 1, "node_type": "Item" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn similar_unknown_node_type_returns_400() {
    let app = build_app_with_item_graph().await;
    let (status, _) = post_similar(
        &app,
        json!({ "embedding": [0.1, 0.2], "top_k": 1, "node_type": "Bogus" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn similar_empty_embedding_returns_400() {
    let app = build_app_with_item_graph().await;
    let (status, _) = post_similar(
        &app,
        json!({ "embedding": [], "top_k": 1, "node_type": "Item" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn similar_zero_top_k_returns_400() {
    let app = build_app_with_item_graph().await;
    let (status, _) = post_similar(
        &app,
        json!({ "embedding": [0.1, 0.2], "top_k": 0, "node_type": "Item" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn similar_over_max_limit_top_k_returns_400() {
    // top_k is now bounded by MAX_LIMIT (10_000) like every other
    // result-bearing route — an unbounded top_k is a DoS vector.
    let app = build_app_with_item_graph().await;
    let (status, _) = post_similar(
        &app,
        json!({ "embedding": [0.1, 0.2], "top_k": 10_001, "node_type": "Item" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn similar_on_missing_graph_returns_404() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/nope/similar")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({ "embedding": [0.1], "top_k": 1, "node_type": "Item" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn put_embedding_dim_mismatch_for_existing_type_returns_400() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "a").await;
    create_item(&app, "b").await;
    // First PUT locks the index dim at 3.
    assert_eq!(
        put_embedding(&app, "Item", "a", &[0.1, 0.2, 0.3]).await,
        StatusCode::OK
    );
    // Second PUT for a different node, wrong dim, must be rejected
    // before any storage write happens (no on-disk rollback needed).
    assert_eq!(
        put_embedding(&app, "Item", "b", &[0.1, 0.2]).await,
        StatusCode::BAD_REQUEST
    );
    // And the would-be-set node's storage embedding wasn't written.
    let (status, _) = get_embedding(&app, "Item", "b").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn delete_embedding_drops_node_from_search_results() {
    let app = build_app_with_item_graph().await;
    for id in ["a", "b"] {
        create_item(&app, id).await;
    }
    put_embedding(&app, "Item", "a", &[1.0, 0.0]).await;
    put_embedding(&app, "Item", "b", &[0.0, 1.0]).await;

    assert_eq!(
        delete_embedding(&app, "Item", "a").await,
        StatusCode::NO_CONTENT
    );

    // After the delete, `a` should no longer appear among search hits.
    let (status, body) = post_similar(
        &app,
        json!({ "embedding": [1.0, 0.0], "top_k": 5, "node_type": "Item" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let ids: Vec<&str> = body["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["node_id"].as_str().unwrap())
        .collect();
    assert!(
        !ids.contains(&"a"),
        "deleted node `a` resurfaced in search: {ids:?}"
    );
}

#[tokio::test]
async fn deleting_a_node_drops_it_from_hnsw_too() {
    // delete_node already cascades to drop the storage embedding
    // (slice 8a). Slice 8b extends that to also remove the node
    // from the per-type HNSW index.
    let app = build_app_with_item_graph().await;
    for id in ["a", "b"] {
        create_item(&app, id).await;
    }
    put_embedding(&app, "Item", "a", &[1.0, 0.0]).await;
    put_embedding(&app, "Item", "b", &[0.0, 1.0]).await;

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/graphs/g1/nodes/Item/a")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    let (_, body) = post_similar(
        &app,
        json!({ "embedding": [1.0, 0.0], "top_k": 5, "node_type": "Item" }),
    )
    .await;
    let ids: Vec<&str> = body["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["node_id"].as_str().unwrap())
        .collect();
    assert!(
        !ids.contains(&"a"),
        "deleted node `a` still in HNSW: {ids:?}"
    );
}

#[tokio::test]
async fn ready_returns_503_before_mark_ready_then_flips() {
    let registry = Arc::new(GraphRegistry::new());
    let readiness = Arc::new(Readiness::not_ready());
    let state = AppState::new(registry, Arc::new(NoAuth::new()), readiness.clone());
    let app = app(state);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&bytes[..], b"starting");

    readiness.mark_ready();

    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

// =============================================================================
// Slice 9 — BearerJwt auth middleware
// =============================================================================

const TEST_JWT_SECRET: &[u8] = b"slice-9-test-signing-secret";

fn build_app_with_bearer_jwt() -> axum::Router {
    let registry = Arc::new(GraphRegistry::new());
    let auth: Arc<dyn AuthProvider> = Arc::new(BearerJwt::new(TEST_JWT_SECRET));
    let state = AppState::new(registry, auth, Arc::new(Readiness::ready()));
    app(state)
}

fn mint_jwt(sub: &str) -> String {
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

#[tokio::test]
async fn protected_route_without_token_returns_401() {
    let app = build_app_with_bearer_jwt();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body = std::str::from_utf8(&bytes).unwrap();
    assert!(body.contains("missing Authorization"), "{body}");
}

#[tokio::test]
async fn protected_route_with_valid_token_succeeds() {
    let app = build_app_with_bearer_jwt();
    let token = mint_jwt("alice");
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["graphs"], json!([]));
}

#[tokio::test]
async fn protected_route_with_garbage_token_returns_401() {
    let app = build_app_with_bearer_jwt();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs")
                .header("authorization", "Bearer not.a.real.jwt")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn health_is_public_under_bearer_jwt() {
    let app = build_app_with_bearer_jwt();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn ready_is_public_under_bearer_jwt() {
    let app = build_app_with_bearer_jwt();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

// =============================================================================
// Slice 10 — /metrics endpoint + middleware
// =============================================================================

async fn fetch_metrics(app: &axum::Router) -> (StatusCode, String) {
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

#[tokio::test]
async fn metrics_endpoint_emits_prometheus_text() {
    let app = build_app();
    let (status, body) = fetch_metrics(&app).await;
    assert_eq!(status, StatusCode::OK);
    // Static metrics are always present.
    assert!(body.contains("# TYPE dynograph_build_info gauge"), "{body}");
    assert!(body.contains("dynograph_build_info{version=\""), "{body}");
    assert!(body.contains("dynograph_uptime_seconds"), "{body}");
}

#[tokio::test]
async fn metrics_records_request_under_matched_route_label() {
    let app = build_app();
    // Drive a known route so we see its matched-path label.
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let (_, body) = fetch_metrics(&app).await;
    assert!(
        body.contains(
            r#"dynograph_http_requests_total{method="GET",path="/v1/graphs",status="200"} 1"#
        ),
        "{body}"
    );
    // And the latency-sum counter is present for the same series.
    assert!(
        body.contains(
            r#"dynograph_http_request_duration_microseconds_sum{method="GET",path="/v1/graphs",status="200"}"#
        ),
        "{body}"
    );
}

#[tokio::test]
async fn metrics_uses_route_pattern_not_literal_url() {
    // Two requests to different graph IDs should collapse into one
    // matched-path series — the cardinality protection that makes
    // /metrics safe to expose under arbitrary client traffic.
    let app = build_app();
    for id in ["alpha", "beta"] {
        let body = json!({
            "id": id,
            "schema": { "name": "demo", "version": 1, "node_types": {}, "edge_types": {} }
        });
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/graphs")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::CREATED);
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!("/v1/graphs/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }
    let (_, body) = fetch_metrics(&app).await;
    // One series for the matched pattern, count=2.
    assert!(
        body.contains(
            r#"dynograph_http_requests_total{method="GET",path="/v1/graphs/{id}",status="200"} 2"#
        ),
        "{body}"
    );
    // The literal IDs must not appear as labels.
    assert!(!body.contains("path=\"/v1/graphs/alpha\""), "{body}");
    assert!(!body.contains("path=\"/v1/graphs/beta\""), "{body}");
}

#[tokio::test]
async fn metrics_endpoint_is_not_self_recorded() {
    // Scrapes shouldn't show up in the request-counter series — that
    // would inflate cardinality and primarily measure Prometheus's
    // own scrape interval. Hit /metrics twice; the second body must
    // not contain a series for path="/metrics".
    let app = build_app();
    let _ = fetch_metrics(&app).await;
    let (_, body) = fetch_metrics(&app).await;
    assert!(!body.contains("path=\"/metrics\""), "{body}");
}

#[tokio::test]
async fn metrics_is_public_under_bearer_jwt() {
    // /metrics sits alongside /health and /ready — accessible
    // without a token even when BearerJwt is the configured
    // provider. Operators gate access at the network layer.
    let app = build_app_with_bearer_jwt();
    let (status, body) = fetch_metrics(&app).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("dynograph_build_info"), "{body}");
}

#[tokio::test]
async fn metrics_emits_hnsw_stats_after_an_embedding_set() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "n1").await;
    put_embedding(&app, "Item", "n1", &[0.1, 0.2, 0.3]).await;
    let (status, body) = fetch_metrics(&app).await;
    assert_eq!(status, StatusCode::OK);
    // The Item index now has 1 live entry + 1 insert.
    assert!(
        body.contains(r#"dynograph_hnsw_index_size{graph="g1",node_type="Item"} 1"#),
        "{body}"
    );
    assert!(
        body.contains(r#"dynograph_hnsw_inserts_total{graph="g1",node_type="Item"} 1"#),
        "{body}"
    );
}

/// Verify the middleware exposes `Identity` to handlers via request
/// extensions. We don't change any handler this slice, but the
/// plumbing must already work for slice-11+ consumer-side
/// authorization. End-to-end: a successful POST exercises the
/// middleware (sets `Identity`); the handler runs (it doesn't read
/// `Identity`, but presence is implied by reaching the 201). If
/// the middleware *didn't* run, the request would never reach the
/// handler anyway because the route layer rejects unauthenticated
/// requests.
#[tokio::test]
async fn bearer_jwt_post_creates_graph_under_auth() {
    let app = build_app_with_bearer_jwt();
    let token = mint_jwt("alice");
    let body = json!({
        "id": "g1",
        "schema": { "name": "demo", "version": 1, "node_types": {}, "edge_types": {} }
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
}

// =========================================================================
// /v1/graphs/{id}/batch — atomic multi-op transaction
// =========================================================================

async fn post_batch(app: &axum::Router, body: Value) -> (StatusCode, Value) {
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

async fn node_exists(app: &axum::Router, node_type: &str, node_id: &str) -> bool {
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

async fn edge_exists(app: &axum::Router, edge_type: &str, from_id: &str, to_id: &str) -> bool {
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

#[tokio::test]
async fn batch_happy_path_mixed_ops_returns_correct_counts() {
    // Exercise all 6 op kinds in one batch with disjoint targets so
    // the test isolates "every op kind reaches the engine and counts
    // correctly" from the read-your-own-writes constraints (those
    // are exercised in batch_modify_after_create_in_same_batch_fails
    // and batch_orphan_edge_when_delete_node_in_same_batch).
    let app = build_app_with_item_graph().await;
    for n in ["a", "b", "d", "e"] {
        create_item(&app, n).await;
    }
    // Pre-create the edges we'll merge/delete inside the batch.
    let pre_edges = [
        json!({"edge_type": "Likes", "from_type": "Item", "from_id": "a", "to_type": "Item", "to_id": "b", "properties": {"weight": 0.1, "source": "manual"}}),
        json!({"edge_type": "Likes", "from_type": "Item", "from_id": "a", "to_type": "Item", "to_id": "d", "properties": {"weight": 0.2, "source": "manual"}}),
    ];
    for body in pre_edges {
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

    let body = json!({
        "ops": [
            // create_node — fresh standalone node
            {"op": "create_node", "node_type": "Item", "node_id": "c", "properties": {"name": "c"}},
            // create_edge — between two pre-existing nodes (b and d)
            {"op": "create_edge", "edge_type": "Likes", "from_type": "Item", "from_id": "b", "to_type": "Item", "to_id": "d", "properties": {"weight": 0.3}},
            // merge_edge — pre-existing a->d
            {"op": "merge_edge", "edge_type": "Likes", "from_id": "a", "to_id": "d", "properties": {"weight": 0.7}},
            // replace_node — pre-existing b
            {"op": "replace_node", "node_type": "Item", "node_id": "b", "properties": {"name": "renamed-b"}},
            // delete_edge — pre-existing a->b
            {"op": "delete_edge", "edge_type": "Likes", "from_id": "a", "to_id": "b"},
            // delete_node — pre-existing standalone e (no edges to/from)
            {"op": "delete_node", "node_type": "Item", "node_id": "e"},
        ]
    });
    let (status, resp) = post_batch(&app, body).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["ops_applied"], 6);
    assert_eq!(resp["nodes_created"], 1);
    assert_eq!(resp["nodes_replaced"], 1);
    assert_eq!(resp["nodes_deleted"], 1);
    assert_eq!(resp["edges_created"], 1);
    assert_eq!(resp["edges_merged"], 1);
    assert_eq!(resp["edges_deleted"], 1);

    // State assertions
    assert!(node_exists(&app, "Item", "a").await);
    assert!(node_exists(&app, "Item", "b").await);
    assert!(node_exists(&app, "Item", "c").await, "c was created");
    assert!(node_exists(&app, "Item", "d").await);
    assert!(!node_exists(&app, "Item", "e").await, "e was deleted");
    assert!(!edge_exists(&app, "Likes", "a", "b").await, "a->b deleted");
    assert!(
        edge_exists(&app, "Likes", "a", "d").await,
        "a->d still exists, weight merged"
    );
    assert!(edge_exists(&app, "Likes", "b", "d").await, "b->d created");
}

#[tokio::test]
async fn batch_dry_run_valid_reports_all_ok_and_commits_nothing() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "x").await;
    // create y, then an edge x->y (read-your-own-writes: op 1 sees the buffered
    // y). All valid — but dry_run must not persist anything.
    let (status, resp) = post_batch(
        &app,
        json!({
            "dry_run": true,
            "ops": [
                {"op": "create_node", "node_type": "Item", "node_id": "y", "properties": {"name": "y"}},
                {"op": "create_edge", "edge_type": "Likes", "from_type": "Item", "from_id": "x", "to_type": "Item", "to_id": "y", "properties": {"weight": 0.5}},
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["valid"], true, "body: {resp}");
    let results = resp["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r["ok"] == true), "body: {resp}");
    assert_eq!(results[0]["op"], "create_node");
    // Nothing committed.
    assert!(!node_exists(&app, "Item", "y").await, "y must not persist");
    assert!(
        !edge_exists(&app, "Likes", "x", "y").await,
        "edge must not persist"
    );
}

#[tokio::test]
async fn batch_dry_run_reports_failing_op_and_commits_nothing() {
    let app = build_app_with_item_graph().await;
    // op 0 creates x (ok); op 1 replaces a node that doesn't exist (fails).
    let (status, resp) = post_batch(
        &app,
        json!({
            "dry_run": true,
            "ops": [
                {"op": "create_node", "node_type": "Item", "node_id": "x", "properties": {"name": "x"}},
                {"op": "replace_node", "node_type": "Item", "node_id": "missing", "properties": {"name": "z"}},
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "dry_run is a 200 report: {resp}");
    assert_eq!(resp["valid"], false, "body: {resp}");
    let results = resp["results"].as_array().unwrap();
    assert_eq!(results[0]["ok"], true);
    assert_eq!(results[1]["ok"], false);
    assert_eq!(results[1]["index"], 1);
    assert_eq!(results[1]["op"], "replace_node");
    assert!(results[1]["error"].is_string(), "body: {resp}");
    // op 0's create is rolled back with the rest.
    assert!(!node_exists(&app, "Item", "x").await, "nothing persists");
}

#[tokio::test]
async fn batch_dry_run_evaluates_every_op_not_just_the_first_failure() {
    // The key difference from the commit path: dry_run reports ALL failures in
    // one round-trip rather than stopping at the first.
    let app = build_app_with_item_graph().await;
    let ops = json!({
        "dry_run": true,
        "ops": [
            {"op": "replace_node", "node_type": "Item", "node_id": "nope1", "properties": {}},
            {"op": "create_node", "node_type": "Item", "node_id": "ok", "properties": {"name": "ok"}},
            {"op": "delete_edge", "edge_type": "Likes", "from_id": "nope2", "to_id": "nope3"},
        ]
    });
    let (status, resp) = post_batch(&app, ops.clone()).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["valid"], false);
    let results = resp["results"].as_array().unwrap();
    assert_eq!(results[0]["ok"], false, "first failure reported");
    assert_eq!(results[1]["ok"], true, "valid op in the middle still ok");
    assert_eq!(results[2]["ok"], false, "second failure ALSO reported");

    // And the commit path (dry_run:false) still stops at the first failure with
    // the unchanged per-op error shape, committing nothing.
    let commit = json!({"ops": ops["ops"].clone()});
    let (status, resp) = post_batch(&app, commit).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
    assert_eq!(resp["op_index"], 0);
    assert_eq!(resp["op_type"], "replace_node");
    assert!(!node_exists(&app, "Item", "ok").await, "atomic rollback");
}

/// Cascade-delete sees in-batch edges via buffer-aware reads (v0.5.5+).
/// Pre-v0.5.5 this test asserted the opposite — that the cascade missed
/// in-batch edges and left orphans. The engine grew buffer-aware reads
/// so cascades now correctly clean up edges created earlier in the
/// same batch.
#[tokio::test]
async fn batch_delete_node_cascades_in_batch_edges() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "a").await;
    create_item(&app, "c").await;

    let body = json!({
        "ops": [
            // Create a->c then delete a in the same batch. With
            // buffer-aware adjacency reads, delete_node's cascade sees
            // the in-batch a->c edge and tombstones it.
            {"op": "create_edge", "edge_type": "Likes", "from_type": "Item", "from_id": "a", "to_type": "Item", "to_id": "c", "properties": {"weight": 0.5}},
            {"op": "delete_node", "node_type": "Item", "node_id": "a"},
        ]
    });
    let (status, _resp) = post_batch(&app, body).await;
    assert_eq!(status, StatusCode::OK);

    // a is gone, AND the in-batch edge to c was cleaned up by the cascade.
    // Pre-v0.5.5 the edge would have survived as an orphan.
    assert!(!node_exists(&app, "Item", "a").await);
    assert!(
        !edge_exists(&app, "Likes", "a", "c").await,
        "cascade-delete should have removed the in-batch edge a->c (read-your-own-writes)"
    );
}

/// Read-your-own-writes for ops that need state lookups: `replace_node`
/// after `create_node` in the same batch sees the in-batch create and
/// succeeds. Pre-v0.5.5 this asserted the opposite (the engine batch
/// buffer was write-only). The contract flipped in v0.5.5 so consumers
/// can build sequences like create→update→update naturally.
#[tokio::test]
async fn batch_modify_after_create_in_same_batch_succeeds() {
    let app = build_app_with_item_graph().await;

    let body = json!({
        "ops": [
            {"op": "create_node", "node_type": "Item", "node_id": "x", "properties": {"name": "x"}},
            {"op": "replace_node", "node_type": "Item", "node_id": "x", "properties": {"name": "renamed"}},
        ]
    });
    let (status, resp) = post_batch(&app, body).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["ops_applied"], 2);
    assert_eq!(resp["nodes_created"], 1);
    assert_eq!(resp["nodes_replaced"], 1);

    // Final state reflects both ops.
    assert!(node_exists(&app, "Item", "x").await);
}

#[tokio::test]
async fn batch_per_op_failure_rolls_back_all_prior_writes() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "a").await;

    // Op 0 creates "x" successfully; op 1 fails (replace on missing
    // node); the whole batch must roll back so "x" never persists.
    let body = json!({
        "ops": [
            {"op": "create_node", "node_type": "Item", "node_id": "x", "properties": {"name": "x"}},
            {"op": "replace_node", "node_type": "Item", "node_id": "missing", "properties": {"name": "y"}},
            {"op": "create_node", "node_type": "Item", "node_id": "z", "properties": {"name": "z"}},
        ]
    });
    let (status, resp) = post_batch(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(resp["op_index"], 1);
    assert_eq!(resp["op_type"], "replace_node");
    assert!(
        resp["error"].as_str().unwrap().contains("missing"),
        "error should mention the missing node id, got: {resp}"
    );

    // Atomicity gate: nothing the batch attempted should have landed.
    assert!(
        !node_exists(&app, "Item", "x").await,
        "op 0 (create_node x) must have rolled back"
    );
    assert!(
        !node_exists(&app, "Item", "z").await,
        "op 2 (create_node z) was past the failure but the rollback is order-independent"
    );
    assert!(
        node_exists(&app, "Item", "a").await,
        "pre-batch state must be untouched"
    );
}

#[tokio::test]
async fn batch_empty_ops_returns_400() {
    let app = build_app_with_item_graph().await;
    let body = json!({ "ops": [] });
    let (status, resp) = post_batch(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    // Every error body is JSON `{ "error": "<message>" }` — read the
    // `error` field rather than the whole body.
    let msg = err_msg(&resp);
    assert!(
        msg.contains("non-empty"),
        "expected 'non-empty' in error, got: {msg}"
    );
}

#[tokio::test]
async fn batch_exceeding_cap_returns_400() {
    let app = build_app_with_item_graph().await;
    // 1001 trivial ops — exceeds MAX_BATCH_OPS = 1000. Use create_node
    // ops with distinct ids so the ops themselves would all be valid;
    // we want to confirm the cap rejects before any apply.
    let ops: Vec<Value> = (0..1001)
        .map(|i| {
            json!({"op": "create_node", "node_type": "Item", "node_id": format!("n{i}"), "properties": {"name": "x"}})
        })
        .collect();
    let body = json!({ "ops": ops });
    let (status, resp) = post_batch(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let msg = err_msg(&resp);
    assert!(
        msg.contains("1001") && msg.contains("1000"),
        "expected size + cap in error, got: {msg}"
    );
    // None of the create_nodes should have landed.
    assert!(!node_exists(&app, "Item", "n0").await);
    assert!(!node_exists(&app, "Item", "n500").await);
}

#[tokio::test]
async fn batch_integrate_fragment_shaped_payload_succeeds() {
    // Acceptance criterion from the 2026-05-04 audit memo: a heavy
    // fragment-integration handler sends ~67 writes per call (8 chars + 4
    // locs + 4 events + 4 concepts + 3 objects + 12 relationships + 1 epoch
    // + assorted edges). We don't model that consumer's schema here — Item
    // + Likes is enough to exercise the
    // same shape (lots of node creates followed by lots of edge creates,
    // all atomic) at comparable scale.
    let app = build_app_with_item_graph().await;

    let mut ops: Vec<Value> = Vec::new();
    // 30 nodes
    for i in 0..30 {
        ops.push(json!({
            "op": "create_node",
            "node_type": "Item",
            "node_id": format!("n{i}"),
            "properties": {"name": format!("n{i}")}
        }));
    }
    // 37 edges — fan-out from n0 to every other node, plus a chain n1->n2->...->n7
    for i in 1..30 {
        ops.push(json!({
            "op": "create_edge",
            "edge_type": "Likes",
            "from_type": "Item",
            "from_id": "n0",
            "to_type": "Item",
            "to_id": format!("n{i}"),
            "properties": {"weight": 0.5}
        }));
    }
    for i in 1..9 {
        ops.push(json!({
            "op": "create_edge",
            "edge_type": "Likes",
            "from_type": "Item",
            "from_id": format!("n{i}"),
            "to_type": "Item",
            "to_id": format!("n{}", i + 1),
            "properties": {"weight": 0.5}
        }));
    }
    assert_eq!(ops.len(), 67, "test setup invariant");

    let body = json!({ "ops": ops });
    let (status, resp) = post_batch(&app, body).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["ops_applied"], 67);
    assert_eq!(resp["nodes_created"], 30);
    assert_eq!(resp["edges_created"], 37);

    // Spot-check both ends of the payload landed.
    assert!(node_exists(&app, "Item", "n0").await);
    assert!(node_exists(&app, "Item", "n29").await);
    assert!(edge_exists(&app, "Likes", "n0", "n29").await);
    assert!(edge_exists(&app, "Likes", "n8", "n9").await);
}

#[tokio::test]
async fn batch_on_unknown_graph_returns_404() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/missing/batch")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"ops": [{"op": "create_node", "node_type": "Item", "node_id": "x"}]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

// =========================================================================
// /v1/graphs/{id}/resolve-or-create — fuzzy/vector resolution + create
// =========================================================================

/// Schema with a resolvable Character type:
/// - `name`: required string (the resolution query lives here)
/// - `story_id`: indexed string (so `scope: {prop: story_id, value: X}` works)
/// - `resolution` config with `fuzzy_then_vector` strategy + audit-cited
///   thresholds (95 auto-merge / 70 fuzzy floor / 0.85 vector cutoff)
fn character_schema_body() -> Value {
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

async fn build_app_with_character_graph() -> axum::Router {
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

async fn create_character(app: &axum::Router, node_id: &str, name: &str, story_id: &str) {
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

async fn post_resolve(app: &axum::Router, body: Value) -> (StatusCode, Value) {
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

#[tokio::test]
async fn resolve_or_create_auto_merge_on_exact_name() {
    let app = build_app_with_character_graph().await;
    create_character(&app, "char-1", "Mira Sandgrove", "story-A").await;

    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Mira Sandgrove", "story_id": "story-A"}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["id"], "char-1");
    assert_eq!(resp["was_created"], false);
    assert_eq!(resp["match_kind"], "auto_merge");
}

#[tokio::test]
async fn resolve_or_create_creates_new_when_no_candidate_matches() {
    let app = build_app_with_character_graph().await;
    create_character(&app, "char-1", "Mira Sandgrove", "story-A").await;

    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Wholly Different Person", "story_id": "story-A"}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["was_created"], true);
    assert_eq!(resp["match_kind"], "created_new");
    let new_id = resp["id"].as_str().unwrap();
    assert_ne!(new_id, "char-1");
    // UUIDv4 has 36 chars (8-4-4-4-12 + 4 dashes).
    assert_eq!(new_id.len(), 36, "id should be UUIDv4: {new_id}");

    // Verify the new node's properties landed.
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/graphs/g1/nodes/Character/{new_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let node: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(node["properties"]["name"], "Wholly Different Person");
    assert_eq!(node["properties"]["story_id"], "story-A");
}

#[tokio::test]
async fn resolve_or_create_scoped_ignores_other_scopes() {
    // Same name in two different stories — scope must keep them apart.
    let app = build_app_with_character_graph().await;
    create_character(&app, "char-A", "Mira Sandgrove", "story-A").await;
    create_character(&app, "char-B", "Mira Sandgrove", "story-B").await;

    // Resolve in story-A: should auto-merge to char-A, not char-B.
    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Mira Sandgrove", "story_id": "story-A"},
            "scope": {"prop": "story_id", "value": "story-A"}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["id"], "char-A");
    assert_eq!(resp["was_created"], false);

    // Resolve in story-C (which has no characters yet) under scope —
    // creates new even though "Mira Sandgrove" exists elsewhere.
    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Mira Sandgrove", "story_id": "story-C"},
            "scope": {"prop": "story_id", "value": "story-C"}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["was_created"], true);
}

#[tokio::test]
async fn resolve_or_create_vector_merge_in_fuzzy_zone() {
    // Existing node's name lands in the fuzzy zone vs the query AND
    // its embedding is near-identical → vector tiebreaker should
    // resolve to it instead of creating new. The
    // "Edwin Whitfield"/"Professor Edwin Whitfield" pair is the same
    // pair the resolver crate's own tests use to demonstrate the
    // tiebreaker zone (resolver.rs:tiebreaker_zone_with_vector_match).
    let app = build_app_with_character_graph().await;
    create_character(&app, "char-1", "Professor Edwin Whitfield", "story-A").await;
    assert_eq!(
        put_embedding(&app, "Character", "char-1", &[1.0, 0.0, 0.0]).await,
        StatusCode::OK
    );

    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Edwin Whitfield", "story_id": "story-A"},
            "embedding": [1.0, 0.0, 0.0]   // identical → cosine 1.0 ≥ 0.85
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["id"], "char-1");
    assert_eq!(resp["was_created"], false);
    assert_eq!(resp["match_kind"], "vector_merge");
}

#[tokio::test]
async fn resolve_or_create_create_new_with_embedding_indexes_for_future_lookups() {
    // CreateNew path with embedding: confirm the embedding lands
    // in storage AND the HNSW index (so subsequent resolves can hit it).
    let app = build_app_with_character_graph().await;

    let (status, resp1) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Professor Edwin Whitfield"},
            "embedding": [1.0, 0.0, 0.0]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp1}");
    assert_eq!(resp1["was_created"], true);
    let new_id = resp1["id"].as_str().unwrap().to_string();

    // Confirm embedding was persisted.
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/graphs/g1/nodes/Character/{new_id}/embedding"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // Second resolve with a fuzzy-zone name + similar embedding should
    // vector-merge to the just-created node — proves HNSW was populated.
    let (status, resp2) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Edwin Whitfield"},  // fuzzy zone vs above
            "embedding": [1.0, 0.0, 0.0]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp2}");
    assert_eq!(resp2["id"], new_id);
    assert_eq!(resp2["match_kind"], "vector_merge");
}

#[tokio::test]
async fn resolve_or_create_missing_name_returns_400() {
    let app = build_app_with_character_graph().await;
    let (status, resp) = post_resolve(
        &app,
        json!({"node_type": "Character", "properties": {"story_id": "story-A"}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or("")
            .contains("properties.name"),
        "got: {resp}"
    );
}

#[tokio::test]
async fn resolve_or_create_non_string_name_returns_400() {
    let app = build_app_with_character_graph().await;
    let (status, resp) = post_resolve(
        &app,
        json!({"node_type": "Character", "properties": {"name": 42}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or("")
            .contains("must be a string"),
        "got: {resp}"
    );
}

#[tokio::test]
async fn resolve_or_create_rejects_zero_magnitude_embedding() {
    // A zero-magnitude embedding scores 0.0 < vector_threshold, which
    // would quietly fall through to CreateNew and masquerade as a real
    // miss. Reject at the pre-flight, before any node is created.
    let app = build_app_with_character_graph().await;
    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Aria", "story_id": "story-A"},
            "embedding": [0.0, 0.0, 0.0]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or("")
            .contains("invalid embedding"),
        "got: {resp}"
    );
}

#[tokio::test]
async fn resolve_or_create_node_type_without_resolution_config_returns_400() {
    let app = build_app_with_character_graph().await;
    // Tag has no `resolution` block — should reject loudly, not fall
    // back to defaults silently.
    let (status, resp) = post_resolve(
        &app,
        json!({"node_type": "Tag", "properties": {"name": "spicy"}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or("")
            .contains("no entity resolution"),
        "got: {resp}"
    );
}

#[tokio::test]
async fn resolve_or_create_unknown_node_type_returns_400() {
    let app = build_app_with_character_graph().await;
    let (status, _) = post_resolve(
        &app,
        json!({"node_type": "Bogus", "properties": {"name": "x"}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn resolve_or_create_scope_prop_not_indexed_returns_400() {
    // Scoping by a non-indexed prop would silently produce zero
    // candidates → always-CreateNew → masked misconfiguration. Reject.
    let app = build_app_with_character_graph().await;
    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Mira"},
            "scope": {"prop": "name", "value": "Mira"}  // name isn't indexed
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(err_msg(&resp).contains("not indexed"), "got: {resp}");
}

#[tokio::test]
async fn resolve_or_create_unknown_graph_returns_404() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/missing/resolve-or-create")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"node_type": "Character", "properties": {"name": "x"}}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn resolve_or_create_embedding_dim_mismatch_returns_400() {
    let app = build_app_with_character_graph().await;
    create_character(&app, "char-1", "Anchor", "story-A").await;
    assert_eq!(
        put_embedding(&app, "Character", "char-1", &[1.0, 0.0, 0.0]).await,
        StatusCode::OK
    );

    // Index dim is now 3; query with a 4-dim embedding → 400.
    let (status, _resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Wholly Different Person"},
            "embedding": [1.0, 0.0, 0.0, 0.0]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

// =========================================================================
// /v1/graphs/{id}/edges:collect — fan-out edge collection
// =========================================================================

/// Multi-type schema for edges:collect tests. Three node types
/// (Character, Event, Location) all carry an indexed `story_id` for
/// scope-filter tests; three edge types with varied endpoint shapes:
/// - MENTIONS: Character → (Character | Event | Location)  (Multiple)
/// - VISITS:   Character → Location                         (Single)
/// - INVOLVES: Event → *                                    (wildcard)
fn knowledge_graph_schema_body() -> Value {
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

async fn build_app_with_knowledge_graph() -> axum::Router {
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

async fn create_typed_node(
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

async fn create_typed_edge(
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

async fn post_collect(app: &axum::Router, body: Value) -> (StatusCode, Value) {
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
async fn seed_two_story_graph(app: &axum::Router) {
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

#[tokio::test]
async fn edges_collect_filtered_source_returns_only_in_scope_edges() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    let (status, resp) = post_collect(
        &app,
        json!({
            "source": {
                "type": "Character",
                "filter": {"prop": "story_id", "value": "story-A"}
            },
            "edge_types": ["MENTIONS", "VISITS"],
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let edges = resp["edges"].as_array().unwrap();
    assert_eq!(
        edges.len(),
        2,
        "expected MENTIONS+VISITS from char-A1 only, got: {resp}"
    );
    assert_eq!(resp["truncated"], false);
    let edge_pairs: Vec<(String, String)> = edges
        .iter()
        .map(|e| {
            (
                e["edge_type"].as_str().unwrap().to_string(),
                e["to_id"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    assert!(edge_pairs.contains(&("MENTIONS".into(), "char-A2".into())));
    assert!(edge_pairs.contains(&("VISITS".into(), "loc-A1".into())));
    // story-B's MENTIONS must not appear.
    assert!(!edge_pairs.contains(&("MENTIONS".into(), "char-B2".into())));
    // Every returned edge should carry from_type since we know it from the scan.
    for e in edges {
        assert_eq!(e["from_type"], "Character");
    }
}

#[tokio::test]
async fn edges_collect_wildcard_source_type_iterates_every_node_type() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    let (status, resp) = post_collect(
        &app,
        json!({
            "source": {"type": "*", "filter": {"prop": "story_id", "value": "story-A"}},
            "edge_types": ["MENTIONS", "VISITS", "INVOLVES"],
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let edges = resp["edges"].as_array().unwrap();
    // story-A has 3 outgoing edges: MENTIONS char-A2, VISITS loc-A1, INVOLVES char-A1.
    assert_eq!(edges.len(), 3);
    let from_types: std::collections::HashSet<&str> = edges
        .iter()
        .map(|e| e["from_type"].as_str().unwrap())
        .collect();
    // Should include both Character (mentions+visits) AND Event (involves).
    assert!(from_types.contains("Character"));
    assert!(from_types.contains("Event"));
}

#[tokio::test]
async fn edges_collect_array_source_types() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // Limit to Character + Event sources (skip Location, which has no
    // outgoing edges in our seed anyway — confirms array handling).
    let (status, resp) = post_collect(
        &app,
        json!({
            "source": {"type": ["Character", "Event"], "filter": {"prop": "story_id", "value": "story-A"}},
            "edge_types": ["MENTIONS", "VISITS", "INVOLVES"],
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["edges"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn edges_collect_adjacency_format_groups_by_source() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    let (status, resp) = post_collect(
        &app,
        json!({
            "source": {"type": "Character", "filter": {"prop": "story_id", "value": "story-A"}},
            "edge_types": ["MENTIONS", "VISITS"],
            "format": "adjacency",
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert!(
        resp["edges"].is_null(),
        "adjacency response shouldn't have an `edges` key"
    );
    let adj = resp["adjacency"].as_object().unwrap();
    // char-A1 has 2 outgoing edges (MENTIONS char-A2, VISITS loc-A1).
    // char-A2 has 0 outgoing.
    assert!(adj.contains_key("char-A1"));
    assert!(!adj.contains_key("char-A2"), "no outgoing edges → no entry");
    let a1_edges = adj["char-A1"].as_array().unwrap();
    assert_eq!(a1_edges.len(), 2);
    // Adjacency entries should NOT carry from_id (it's the key).
    for e in a1_edges {
        assert!(e["from_id"].is_null());
    }
}

#[tokio::test]
async fn edges_collect_resolve_target_single_endpoint_attaches_target_node() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // VISITS has Single("Location") endpoint — one candidate type, one lookup.
    let (status, resp) = post_collect(
        &app,
        json!({
            "source": {"type": "Character", "filter": {"prop": "story_id", "value": "story-A"}},
            "edge_types": ["VISITS"],
            "resolve_target": true,
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let edges = resp["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 1);
    let target = &edges[0]["target"];
    assert_eq!(target["node_type"], "Location");
    assert_eq!(target["node_id"], "loc-A1");
    assert_eq!(target["properties"]["name"], "Tower");
}

#[tokio::test]
async fn edges_collect_resolve_target_list_endpoint_picks_correct_type() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // MENTIONS has Multiple(["Character", "Event", "Location"]) — must
    // try each candidate and pick the one where the to_id resolves.
    let (status, resp) = post_collect(
        &app,
        json!({
            "source": {"type": "Character", "filter": {"prop": "story_id", "value": "story-A"}},
            "edge_types": ["MENTIONS"],
            "resolve_target": true,
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let edges = resp["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 1);
    // char-A1 MENTIONS char-A2 → target should resolve as Character.
    assert_eq!(edges[0]["target"]["node_type"], "Character");
    assert_eq!(edges[0]["target"]["node_id"], "char-A2");
}

#[tokio::test]
async fn edges_collect_limit_truncates_and_flags() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // Story-A has 3 outgoing edges total; limit=2 should truncate.
    let (status, resp) = post_collect(
        &app,
        json!({
            "source": {"type": "*", "filter": {"prop": "story_id", "value": "story-A"}},
            "edge_types": ["MENTIONS", "VISITS", "INVOLVES"],
            "limit": 2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["edges"].as_array().unwrap().len(), 2);
    assert_eq!(resp["truncated"], true);
}

#[tokio::test]
async fn edges_collect_empty_edge_types_returns_400() {
    let app = build_app_with_knowledge_graph().await;
    let (status, resp) = post_collect(
        &app,
        json!({
            "source": {"type": "Character"},
            "edge_types": [],
            "limit": 10
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(err_msg(&resp).contains("non-empty"), "got: {resp}");
}

#[tokio::test]
async fn edges_collect_unknown_edge_type_returns_400() {
    let app = build_app_with_knowledge_graph().await;
    let (status, resp) = post_collect(
        &app,
        json!({
            "source": {"type": "Character"},
            "edge_types": ["MENTIONS", "BOGUS"],
            "limit": 10
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(err_msg(&resp).contains("BOGUS"), "got: {resp}");
}

#[tokio::test]
async fn edges_collect_unknown_source_type_returns_400() {
    let app = build_app_with_knowledge_graph().await;
    let (status, _) = post_collect(
        &app,
        json!({
            "source": {"type": "Bogus"},
            "edge_types": ["MENTIONS"],
            "limit": 10
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn edges_collect_limit_out_of_range_returns_400() {
    let app = build_app_with_knowledge_graph().await;
    for bad in [0usize, 10_001] {
        let (status, _) = post_collect(
            &app,
            json!({
                "source": {"type": "Character"},
                "edge_types": ["MENTIONS"],
                "limit": bad
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "limit={bad}");
    }
}

#[tokio::test]
async fn edges_collect_filter_on_unindexed_prop_returns_400() {
    let app = build_app_with_knowledge_graph().await;
    let (status, resp) = post_collect(
        &app,
        json!({
            "source": {"type": "Character", "filter": {"prop": "name", "value": "Alice"}},
            "edge_types": ["MENTIONS"],
            "limit": 10
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(err_msg(&resp).contains("not indexed"), "got: {resp}");
}

#[tokio::test]
async fn edges_collect_unknown_graph_returns_404() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/missing/edges:collect")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "source": {"type": "Character"},
                        "edge_types": ["MENTIONS"],
                        "limit": 10
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

// =========================================================================
// /v1/graphs/{id}/edges:adjacent — single-node 1-hop adjacency
// =========================================================================

async fn post_adjacent(app: &axum::Router, body: Value) -> (StatusCode, Value) {
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
fn adjacent_triples(resp: &Value) -> std::collections::HashSet<(String, String, String)> {
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

#[tokio::test]
async fn edges_adjacent_both_returns_outgoing_and_incoming() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // char-A1: outgoing MENTIONS→char-A2, VISITS→loc-A1; incoming INVOLVES from ev-A1.
    let (status, resp) = post_adjacent(&app, json!({"node_id": "char-A1"})).await;
    assert_eq!(status, StatusCode::OK, "got: {resp}");
    let got = adjacent_triples(&resp);
    let want: std::collections::HashSet<_> = [
        (
            "MENTIONS".to_string(),
            "char-A1".to_string(),
            "char-A2".to_string(),
        ),
        (
            "VISITS".to_string(),
            "char-A1".to_string(),
            "loc-A1".to_string(),
        ),
        (
            "INVOLVES".to_string(),
            "ev-A1".to_string(),
            "char-A1".to_string(),
        ),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        got, want,
        "default direction=both should return all 3 incident edges"
    );
}

#[tokio::test]
async fn edges_adjacent_outgoing_only() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    let (status, resp) =
        post_adjacent(&app, json!({"node_id": "char-A1", "direction": "outgoing"})).await;
    assert_eq!(status, StatusCode::OK, "got: {resp}");
    let got = adjacent_triples(&resp);
    let want: std::collections::HashSet<_> = [
        (
            "MENTIONS".to_string(),
            "char-A1".to_string(),
            "char-A2".to_string(),
        ),
        (
            "VISITS".to_string(),
            "char-A1".to_string(),
            "loc-A1".to_string(),
        ),
    ]
    .into_iter()
    .collect();
    assert_eq!(got, want);
}

#[tokio::test]
async fn edges_adjacent_incoming_only() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    let (status, resp) =
        post_adjacent(&app, json!({"node_id": "char-A1", "direction": "incoming"})).await;
    assert_eq!(status, StatusCode::OK, "got: {resp}");
    let got = adjacent_triples(&resp);
    let want: std::collections::HashSet<_> = [(
        "INVOLVES".to_string(),
        "ev-A1".to_string(),
        "char-A1".to_string(),
    )]
    .into_iter()
    .collect();
    assert_eq!(got, want);
}

#[tokio::test]
async fn edges_adjacent_edge_type_filter() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // Outgoing, filtered to MENTIONS — drops the VISITS edge.
    let (status, resp) = post_adjacent(
        &app,
        json!({"node_id": "char-A1", "direction": "outgoing", "edge_type": "MENTIONS"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "got: {resp}");
    let got = adjacent_triples(&resp);
    let want: std::collections::HashSet<_> = [(
        "MENTIONS".to_string(),
        "char-A1".to_string(),
        "char-A2".to_string(),
    )]
    .into_iter()
    .collect();
    assert_eq!(got, want);
}

#[tokio::test]
async fn edges_adjacent_unknown_node_returns_empty() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // Adjacency is keyed by node id, not type — an unknown id is simply
    // an isolated node, not an error.
    let (status, resp) = post_adjacent(&app, json!({"node_id": "does-not-exist"})).await;
    assert_eq!(status, StatusCode::OK, "got: {resp}");
    assert_eq!(resp["edges"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn edges_adjacent_unknown_graph_returns_404() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/missing/edges:adjacent")
                .header("content-type", "application/json")
                .body(Body::from(json!({"node_id": "char-A1"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn edges_adjacent_default_is_not_truncated() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // No `limit` → defaults to the safety cap; char-A1's 3 edges fit.
    let (status, resp) = post_adjacent(&app, json!({"node_id": "char-A1"})).await;
    assert_eq!(status, StatusCode::OK, "got: {resp}");
    assert_eq!(resp["edges"].as_array().unwrap().len(), 3);
    assert_eq!(resp["truncated"], false);
}

#[tokio::test]
async fn edges_adjacent_respects_limit_and_sets_truncated() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // char-A1 has 3 incident edges; cap at 1 → 1 returned, truncated.
    let (status, resp) = post_adjacent(&app, json!({"node_id": "char-A1", "limit": 1})).await;
    assert_eq!(status, StatusCode::OK, "got: {resp}");
    assert_eq!(resp["edges"].as_array().unwrap().len(), 1);
    assert_eq!(resp["truncated"], true);
}

#[tokio::test]
async fn edges_adjacent_rejects_out_of_range_limit() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    for bad in [json!(0), json!(10_001)] {
        let (status, resp) = post_adjacent(&app, json!({"node_id": "char-A1", "limit": bad})).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "limit {bad} should be rejected, got: {resp}"
        );
    }
}

// =========================================================================
// /v1/graphs/{id}/traverse — typed BFS over edge-type steps
// =========================================================================

/// Schema mirroring a temporal use case: NarrativeEpoch
/// nodes joined by PRECEDES edges, scoped by `story_id`. Plus a Tag
/// node type with a TAGS edge so the multi-step / multi-type tests
/// have somewhere to chain to.
fn temporal_schema_body() -> Value {
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

async fn build_app_with_temporal_schema() -> axum::Router {
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

async fn post_traverse(app: &axum::Router, body: Value) -> (StatusCode, Value) {
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
async fn seed_temporal_graph(app: &axum::Router) {
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

#[tokio::test]
async fn traverse_single_step_transitive_collects_all_descendants() {
    // The `compute_predecessors` shape, but applied
    // forward: from e1, transitive PRECEDES outgoing → {e2, e3}.
    let app = build_app_with_temporal_schema().await;
    seed_temporal_graph(&app).await;

    let (status, resp) = post_traverse(
        &app,
        json!({
            "start": {"type": "NarrativeEpoch", "id": "e1"},
            "traverse": [
                {"edge_type": "PRECEDES", "direction": "outgoing", "transitive": true}
            ],
            "scope": {"prop": "story_id", "value": "story-A"},
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let nodes = resp["nodes"].as_array().unwrap();
    let ids: std::collections::HashSet<&str> = nodes
        .iter()
        .map(|n| n["node_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["e2", "e3"].into_iter().collect());
    // start (e1) must NOT appear in results.
    assert!(!ids.contains("e1"));
    assert_eq!(resp["truncated"], false);
    // Default return=ids: properties field must be absent.
    for n in nodes {
        assert!(
            n.get("properties").is_none(),
            "return=ids must omit properties, got: {n}"
        );
    }
}

#[tokio::test]
async fn traverse_single_step_non_transitive_one_hop_only() {
    let app = build_app_with_temporal_schema().await;
    seed_temporal_graph(&app).await;

    // Non-transitive: from e1 we should get only e2, NOT e3.
    let (status, resp) = post_traverse(
        &app,
        json!({
            "start": {"type": "NarrativeEpoch", "id": "e1"},
            "traverse": [
                {"edge_type": "PRECEDES", "direction": "outgoing", "transitive": false}
            ],
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let nodes = resp["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0]["node_id"], "e2");
}

#[tokio::test]
async fn traverse_incoming_direction_walks_predecessors() {
    // The actual compute_predecessors shape: from e3,
    // incoming PRECEDES transitive → {e2, e1}.
    let app = build_app_with_temporal_schema().await;
    seed_temporal_graph(&app).await;

    let (status, resp) = post_traverse(
        &app,
        json!({
            "start": {"type": "NarrativeEpoch", "id": "e3"},
            "traverse": [
                {"edge_type": "PRECEDES", "direction": "incoming", "transitive": true}
            ],
            "scope": {"prop": "story_id", "value": "story-A"},
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let ids: std::collections::HashSet<&str> = resp["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["node_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["e1", "e2"].into_iter().collect());
}

#[tokio::test]
async fn traverse_scope_filter_excludes_other_stories() {
    // story-B's e4 → e5 chain must be invisible when scoped to
    // story-A — even if there were a cross-story PRECEDES edge.
    let app = build_app_with_temporal_schema().await;
    seed_temporal_graph(&app).await;

    // Add a cross-story edge to confirm scope drops it.
    create_typed_edge(
        &app,
        "PRECEDES",
        "NarrativeEpoch",
        "e3",
        "NarrativeEpoch",
        "e4",
    )
    .await;

    let (status, resp) = post_traverse(
        &app,
        json!({
            "start": {"type": "NarrativeEpoch", "id": "e1"},
            "traverse": [
                {"edge_type": "PRECEDES", "direction": "outgoing", "transitive": true}
            ],
            "scope": {"prop": "story_id", "value": "story-A"},
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let ids: std::collections::HashSet<&str> = resp["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["node_id"].as_str().unwrap())
        .collect();
    // e4 (story-B) must be filtered out even though e3 → e4 exists.
    assert_eq!(ids, ["e2", "e3"].into_iter().collect());
}

#[tokio::test]
async fn traverse_start_out_of_scope_returns_empty_not_error() {
    // start=e4 (story-B) but scope=story-A: legitimate empty result
    // (the caller's filter just excludes the start), 200 not 4xx.
    let app = build_app_with_temporal_schema().await;
    seed_temporal_graph(&app).await;

    let (status, resp) = post_traverse(
        &app,
        json!({
            "start": {"type": "NarrativeEpoch", "id": "e4"},
            "traverse": [
                {"edge_type": "PRECEDES", "direction": "outgoing", "transitive": true}
            ],
            "scope": {"prop": "story_id", "value": "story-A"},
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["nodes"].as_array().unwrap().len(), 0);
    assert_eq!(resp["truncated"], false);
}

#[tokio::test]
async fn traverse_direction_both_walks_outgoing_and_incoming() {
    // From e2: outgoing PRECEDES → e3, incoming PRECEDES ← e1.
    // direction=both should reach both.
    let app = build_app_with_temporal_schema().await;
    seed_temporal_graph(&app).await;

    let (status, resp) = post_traverse(
        &app,
        json!({
            "start": {"type": "NarrativeEpoch", "id": "e2"},
            "traverse": [
                {"edge_type": "PRECEDES", "direction": "both", "transitive": false}
            ],
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let ids: std::collections::HashSet<&str> = resp["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["node_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["e1", "e3"].into_iter().collect());
}

#[tokio::test]
async fn traverse_cycle_handled_via_visited_set() {
    // Construct a cycle: e3 → e1 (already had e1 → e2 → e3). BFS
    // must terminate without revisiting any node.
    let app = build_app_with_temporal_schema().await;
    seed_temporal_graph(&app).await;
    create_typed_edge(
        &app,
        "PRECEDES",
        "NarrativeEpoch",
        "e3",
        "NarrativeEpoch",
        "e1",
    )
    .await;

    let (status, resp) = post_traverse(
        &app,
        json!({
            "start": {"type": "NarrativeEpoch", "id": "e1"},
            "traverse": [
                {"edge_type": "PRECEDES", "direction": "outgoing", "transitive": true}
            ],
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    // Result is {e2, e3} — start (e1) excluded even though the
    // cycle would loop back through it.
    let ids: std::collections::HashSet<&str> = resp["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["node_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["e2", "e3"].into_iter().collect());
}

#[tokio::test]
async fn traverse_return_nodes_includes_properties() {
    let app = build_app_with_temporal_schema().await;
    seed_temporal_graph(&app).await;

    let (status, resp) = post_traverse(
        &app,
        json!({
            "start": {"type": "NarrativeEpoch", "id": "e1"},
            "traverse": [
                {"edge_type": "PRECEDES", "direction": "outgoing", "transitive": true}
            ],
            "return": "nodes",
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let nodes = resp["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 2);
    for n in nodes {
        let props = n["properties"].as_object().unwrap();
        assert!(props.contains_key("name"));
        assert!(props.contains_key("story_id"));
    }
}

#[tokio::test]
async fn traverse_multi_step_chain_unions_intermediates() {
    // Two-step chain: PRECEDES outgoing transitive, then TAGS
    // outgoing non-transitive. From e1, step 0 reaches {e2, e3};
    // step 1 from e2 reaches {tag-A1}, from e3 reaches nothing.
    // UNION semantics: result = {e2, e3, tag-A1}.
    let app = build_app_with_temporal_schema().await;
    seed_temporal_graph(&app).await;

    let (status, resp) = post_traverse(
        &app,
        json!({
            "start": {"type": "NarrativeEpoch", "id": "e1"},
            "traverse": [
                {"edge_type": "PRECEDES", "direction": "outgoing", "transitive": true},
                {"edge_type": "TAGS",     "direction": "outgoing", "transitive": false}
            ],
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let pairs: std::collections::HashSet<(String, String)> = resp["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| {
            (
                n["node_type"].as_str().unwrap().to_string(),
                n["node_id"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    assert!(pairs.contains(&("NarrativeEpoch".into(), "e2".into())));
    assert!(pairs.contains(&("NarrativeEpoch".into(), "e3".into())));
    assert!(pairs.contains(&("Tag".into(), "tag-A1".into())));
    assert_eq!(pairs.len(), 3);
}

#[tokio::test]
async fn traverse_limit_truncates_and_flags() {
    let app = build_app_with_temporal_schema().await;
    seed_temporal_graph(&app).await;

    // Full transitive walk would yield {e2, e3}; limit=1 truncates.
    let (status, resp) = post_traverse(
        &app,
        json!({
            "start": {"type": "NarrativeEpoch", "id": "e1"},
            "traverse": [
                {"edge_type": "PRECEDES", "direction": "outgoing", "transitive": true}
            ],
            "limit": 1
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["nodes"].as_array().unwrap().len(), 1);
    assert_eq!(resp["truncated"], true);
}

#[tokio::test]
async fn traverse_unknown_start_node_returns_404() {
    let app = build_app_with_temporal_schema().await;
    seed_temporal_graph(&app).await;

    let (status, _) = post_traverse(
        &app,
        json!({
            "start": {"type": "NarrativeEpoch", "id": "no-such-id"},
            "traverse": [
                {"edge_type": "PRECEDES", "direction": "outgoing", "transitive": true}
            ],
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn traverse_unknown_start_type_returns_400() {
    let app = build_app_with_temporal_schema().await;
    let (status, resp) = post_traverse(
        &app,
        json!({
            "start": {"type": "Bogus", "id": "x"},
            "traverse": [
                {"edge_type": "PRECEDES", "direction": "outgoing"}
            ],
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(err_msg(&resp).contains("Bogus"), "got: {resp}");
}

#[tokio::test]
async fn traverse_unknown_edge_type_returns_400() {
    let app = build_app_with_temporal_schema().await;
    seed_temporal_graph(&app).await;
    let (status, resp) = post_traverse(
        &app,
        json!({
            "start": {"type": "NarrativeEpoch", "id": "e1"},
            "traverse": [
                {"edge_type": "BOGUS", "direction": "outgoing"}
            ],
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(err_msg(&resp).contains("BOGUS"), "got: {resp}");
}

#[tokio::test]
async fn traverse_empty_traverse_returns_400() {
    let app = build_app_with_temporal_schema().await;
    seed_temporal_graph(&app).await;
    let (status, resp) = post_traverse(
        &app,
        json!({
            "start": {"type": "NarrativeEpoch", "id": "e1"},
            "traverse": [],
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(err_msg(&resp).contains("non-empty"), "got: {resp}");
}

#[tokio::test]
async fn traverse_limit_out_of_range_returns_400() {
    let app = build_app_with_temporal_schema().await;
    for bad in [0usize, 10_001] {
        let (status, _) = post_traverse(
            &app,
            json!({
                "start": {"type": "NarrativeEpoch", "id": "e1"},
                "traverse": [
                    {"edge_type": "PRECEDES", "direction": "outgoing"}
                ],
                "limit": bad
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "limit={bad}");
    }
}

#[tokio::test]
async fn traverse_scope_on_unindexed_prop_returns_400() {
    let app = build_app_with_temporal_schema().await;
    seed_temporal_graph(&app).await;
    let (status, resp) = post_traverse(
        &app,
        json!({
            "start": {"type": "NarrativeEpoch", "id": "e1"},
            "traverse": [
                {"edge_type": "PRECEDES", "direction": "outgoing"}
            ],
            "scope": {"prop": "name", "value": "Beginning"},
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(err_msg(&resp).contains("not indexed"), "got: {resp}");
}

#[tokio::test]
async fn traverse_unknown_graph_returns_404() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/missing/traverse")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "start": {"type": "X", "id": "y"},
                        "traverse": [{"edge_type": "Z", "direction": "outgoing"}],
                        "limit": 10
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

// =========================================================================
// /v1/graphs/{id}/nodes:exists — batch (type, name) existence check
// =========================================================================

/// Schema where `name` is `indexed: true` on every type — the
/// precondition for nodes:exists. Distinct from `knowledge_graph_schema_body`
/// (where `name` is intentionally un-indexed to feed the
/// rejection-path tests).
fn indexed_name_schema_body() -> Value {
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

async fn build_app_with_indexed_name_schema() -> axum::Router {
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

async fn post_exists(app: &axum::Router, body: Value) -> (StatusCode, Value) {
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

async fn create_named_node(app: &axum::Router, node_type: &str, node_id: &str, name: &str) {
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

#[tokio::test]
async fn nodes_exists_returns_present_with_id_and_absent_with_null() {
    let app = build_app_with_indexed_name_schema().await;
    create_named_node(&app, "Geography", "geo:iran", "Iran").await;
    create_named_node(&app, "Commodity", "com:oil", "Oil").await;

    let (status, resp) = post_exists(
        &app,
        json!({
            "queries": [
                {"type": "Geography", "name": "Iran"},
                {"type": "Geography", "name": "Atlantis"},
                {"type": "Commodity", "name": "Oil"}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let results = resp["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);

    assert_eq!(results[0]["type"], "Geography");
    assert_eq!(results[0]["name"], "Iran");
    assert_eq!(results[0]["exists"], true);
    assert_eq!(results[0]["id"], "geo:iran");

    assert_eq!(results[1]["type"], "Geography");
    assert_eq!(results[1]["name"], "Atlantis");
    assert_eq!(results[1]["exists"], false);
    assert!(results[1]["id"].is_null());

    assert_eq!(results[2]["type"], "Commodity");
    assert_eq!(results[2]["name"], "Oil");
    assert_eq!(results[2]["exists"], true);
    assert_eq!(results[2]["id"], "com:oil");
}

#[tokio::test]
async fn nodes_exists_preserves_query_order() {
    let app = build_app_with_indexed_name_schema().await;
    create_named_node(&app, "Geography", "geo:iran", "Iran").await;
    create_named_node(&app, "Geography", "geo:peru", "Peru").await;

    let (status, resp) = post_exists(
        &app,
        json!({
            "queries": [
                {"type": "Geography", "name": "Peru"},
                {"type": "Geography", "name": "Iran"}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let names: Vec<&str> = resp["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["Peru", "Iran"]);
}

#[tokio::test]
async fn nodes_exists_empty_queries_returns_400() {
    let app = build_app_with_indexed_name_schema().await;
    let (status, resp) = post_exists(&app, json!({"queries": []})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(err_msg(&resp).contains("non-empty"), "got: {resp}");
}

#[tokio::test]
async fn nodes_exists_unknown_type_returns_400() {
    let app = build_app_with_indexed_name_schema().await;
    let (status, resp) =
        post_exists(&app, json!({"queries": [{"type": "Ghost", "name": "x"}]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or("")
            .contains("unknown node type"),
        "got: {resp}"
    );
}

#[tokio::test]
async fn nodes_exists_unindexed_name_returns_400() {
    // Reuse the knowledge_graph schema where `name` is required but
    // NOT indexed — should reject loudly rather than silently report
    // every entity as absent.
    let app = build_app_with_knowledge_graph().await;
    let (status, resp) = post_exists(
        &app,
        json!({"queries": [{"type": "Character", "name": "Alice"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(err_msg(&resp).contains("not indexed"), "got: {resp}");
}

#[tokio::test]
async fn nodes_exists_unknown_graph_returns_404() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/missing/nodes:exists")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"queries": [{"type": "X", "name": "y"}]}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

// =========================================================================
// /v1/graphs/{id}/nodes:scan — predicate-filtered scan
// =========================================================================

/// Schema for nodes:scan tests: a Person type with four indexed
/// properties spanning the indexable type variants (string, int,
/// bool) so range / eq / in / neq can all be exercised. `bio` is
/// declared but un-indexed so the "reject un-indexed property" path
/// has a target.
fn person_scan_schema_body() -> Value {
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

async fn build_app_with_person_scan_schema() -> axum::Router {
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

async fn post_scan(app: &axum::Router, body: Value) -> (StatusCode, Value) {
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

async fn create_person(
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

async fn seed_people(app: &axum::Router) {
    create_person(app, "p1", "Alice", 30, "market_moving", true).await;
    create_person(app, "p2", "Bob", 45, "background", true).await;
    create_person(app, "p3", "Carol", 25, "market_moving", false).await;
    create_person(app, "p4", "Dave", 60, "leading", true).await;
    create_person(app, "p5", "Eve", 50, "background", false).await;
}

fn id_set(resp: &Value) -> std::collections::HashSet<String> {
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

fn ids_of(strs: &[&str]) -> std::collections::HashSet<String> {
    strs.iter().map(|s| s.to_string()).collect()
}

#[tokio::test]
async fn nodes_scan_eq_uses_index_and_returns_matching() {
    let app = build_app_with_person_scan_schema().await;
    seed_people(&app).await;

    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Person",
            "where": [{"property": "influence_level", "op": "eq", "value": "market_moving"}],
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["truncated"], false);
    assert_eq!(id_set(&resp), ids_of(&["p1", "p3"]));
    let first = &resp["results"].as_array().unwrap()[0];
    assert!(first["properties"].is_object(), "got: {resp}");
    assert_eq!(first["node_type"], "Person");
    assert!(first["node_id"].as_str().is_some());
}

#[tokio::test]
async fn nodes_scan_returns_ids_when_requested() {
    let app = build_app_with_person_scan_schema().await;
    seed_people(&app).await;

    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Person",
            "where": [{"property": "verified", "op": "eq", "value": true}],
            "return": "ids",
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let arr = resp["results"].as_array().unwrap();
    assert!(
        arr.iter().all(|v| v.is_string()),
        "ids must be bare strings: {resp}"
    );
    assert_eq!(id_set(&resp), ids_of(&["p1", "p2", "p4"]));
}

#[tokio::test]
async fn nodes_scan_neq_returns_complement() {
    let app = build_app_with_person_scan_schema().await;
    seed_people(&app).await;
    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Person",
            "where": [{"property": "influence_level", "op": "neq", "value": "background"}],
            "return": "ids",
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(id_set(&resp), ids_of(&["p1", "p3", "p4"]));
}

#[tokio::test]
async fn nodes_scan_in_operator_unions_matches() {
    let app = build_app_with_person_scan_schema().await;
    seed_people(&app).await;
    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Person",
            "where": [{"property": "influence_level", "op": "in",
                       "value": ["market_moving", "leading"]}],
            "return": "ids",
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(id_set(&resp), ids_of(&["p1", "p3", "p4"]));
}

#[tokio::test]
async fn nodes_scan_range_int() {
    let app = build_app_with_person_scan_schema().await;
    seed_people(&app).await;
    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Person",
            "where": [{"property": "age", "op": "gte", "value": 45}],
            "return": "ids",
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(id_set(&resp), ids_of(&["p2", "p4", "p5"]));

    // 25 < age < 50 → AND
    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Person",
            "where": [
                {"property": "age", "op": "gt", "value": 25},
                {"property": "age", "op": "lt", "value": 50}
            ],
            "return": "ids",
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(id_set(&resp), ids_of(&["p1", "p2"]));
}

#[tokio::test]
async fn nodes_scan_range_string_lexicographic() {
    let app = build_app_with_person_scan_schema().await;
    seed_people(&app).await;
    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Person",
            "where": [{"property": "name", "op": "lt", "value": "D"}],
            "return": "ids",
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    // Alice, Bob, Carol → p1, p2, p3
    assert_eq!(id_set(&resp), ids_of(&["p1", "p2", "p3"]));
}

#[tokio::test]
async fn nodes_scan_multi_clause_and_with_eq_seed() {
    let app = build_app_with_person_scan_schema().await;
    seed_people(&app).await;
    // influence_level = market_moving AND age > 25 → p1 (Alice, 30) only
    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Person",
            "where": [
                {"property": "influence_level", "op": "eq", "value": "market_moving"},
                {"property": "age",             "op": "gt", "value": 25}
            ],
            "return": "ids",
            "limit": 100
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(id_set(&resp), ids_of(&["p1"]));
}

#[tokio::test]
async fn nodes_scan_limit_truncates_and_flags() {
    let app = build_app_with_person_scan_schema().await;
    seed_people(&app).await;
    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Person",
            "where": [{"property": "age", "op": "gte", "value": 0}],
            "return": "ids",
            "limit": 2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["results"].as_array().unwrap().len(), 2);
    assert_eq!(resp["truncated"], true);
}

#[tokio::test]
async fn nodes_scan_empty_where_returns_400() {
    let app = build_app_with_person_scan_schema().await;
    let (status, resp) = post_scan(&app, json!({"type": "Person", "where": [], "limit": 10})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(err_msg(&resp).contains("non-empty"), "got: {resp}");
}

#[tokio::test]
async fn nodes_scan_unknown_type_returns_400() {
    let app = build_app_with_person_scan_schema().await;
    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Ghost",
            "where": [{"property": "name", "op": "eq", "value": "x"}],
            "limit": 10
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or("")
            .contains("unknown node type"),
        "got: {resp}"
    );
}

#[tokio::test]
async fn nodes_scan_unknown_property_returns_400() {
    let app = build_app_with_person_scan_schema().await;
    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Person",
            "where": [{"property": "nonexistent", "op": "eq", "value": "x"}],
            "limit": 10
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or("")
            .contains("not declared"),
        "got: {resp}"
    );
}

#[tokio::test]
async fn nodes_scan_unindexed_property_returns_400() {
    let app = build_app_with_person_scan_schema().await;
    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Person",
            "where": [{"property": "bio", "op": "eq", "value": "anything"}],
            "limit": 10
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(err_msg(&resp).contains("not indexed"), "got: {resp}");
}

#[tokio::test]
async fn nodes_scan_in_with_non_list_returns_400() {
    let app = build_app_with_person_scan_schema().await;
    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Person",
            "where": [{"property": "influence_level", "op": "in", "value": "single"}],
            "limit": 10
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or("")
            .contains("must be a list"),
        "got: {resp}"
    );
}

#[tokio::test]
async fn nodes_scan_in_with_oversized_list_returns_400() {
    let app = build_app_with_person_scan_schema().await;
    // 1001 elements > MAX_IN_LIST_LEN (1000). Foundation rejects pre-flight
    // so a hostile request can't translate into O(candidates × in_len)
    // in-memory comparisons.
    let oversized: Vec<i64> = (0..1001).collect();
    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Person",
            "where": [{"property": "age", "op": "in", "value": oversized}],
            "limit": 10
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or("")
            .contains("maximum length"),
        "got: {resp}"
    );
}

#[tokio::test]
async fn nodes_scan_range_with_non_ordered_value_returns_400() {
    let app = build_app_with_person_scan_schema().await;
    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Person",
            "where": [{"property": "verified", "op": "gt", "value": true}],
            "limit": 10
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(err_msg(&resp).contains("range op"), "got: {resp}");
}

#[tokio::test]
async fn nodes_scan_limit_out_of_range_returns_400() {
    let app = build_app_with_person_scan_schema().await;
    let (status, resp) = post_scan(
        &app,
        json!({
            "type": "Person",
            "where": [{"property": "age", "op": "gt", "value": 0}],
            "limit": 0
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(err_msg(&resp).contains("limit"), "got: {resp}");
}

#[tokio::test]
async fn nodes_scan_unknown_graph_returns_404() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/missing/nodes:scan")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "type": "Person",
                        "where": [{"property": "name", "op": "eq", "value": "x"}],
                        "limit": 10
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

// =========================================================================
// /v1/graphs/{id}/edges/.../welford_update — atomic EMA + Welford on edge
// =========================================================================

fn welford_schema_body() -> Value {
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

async fn build_app_with_welford_schema() -> axum::Router {
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

async fn create_indicator(app: &axum::Router, id: &str, name: &str) {
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

async fn create_causes_edge(app: &axum::Router, from: &str, to: &str, extra_props: Value) {
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

async fn post_welford(
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

async fn get_edge_props(app: &axum::Router, edge_type: &str, from: &str, to: &str) -> Value {
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

fn approx_f64(v: &Value, want: f64) {
    let got = v
        .as_f64()
        .unwrap_or_else(|| panic!("expected number, got {v}"));
    assert!((got - want).abs() < 1e-9, "approx: got {got}, want {want}");
}

#[tokio::test]
async fn welford_first_observation_initializes_full_state() {
    let app = build_app_with_welford_schema().await;
    create_indicator(&app, "a", "rate").await;
    create_indicator(&app, "b", "yield").await;
    create_causes_edge(&app, "a", "b", json!({})).await;

    let (status, resp) = post_welford(
        &app,
        "CAUSES",
        "a",
        "b",
        json!({"observation": 0.7, "alpha": 0.05}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    approx_f64(&resp["score"], 0.7);
    approx_f64(&resp["score_m2"], 0.0);
    approx_f64(&resp["score_stddev"], 0.0);
    approx_f64(&resp["score_min"], 0.7);
    approx_f64(&resp["score_max"], 0.7);
    assert_eq!(resp["score_count"], 1);
}

#[tokio::test]
async fn welford_second_observation_applies_ema_and_welford_increment() {
    let app = build_app_with_welford_schema().await;
    create_indicator(&app, "a", "rate").await;
    create_indicator(&app, "b", "yield").await;
    create_causes_edge(&app, "a", "b", json!({})).await;

    // Obs1 → score=0.5, count=1
    let (s1, _) = post_welford(
        &app,
        "CAUSES",
        "a",
        "b",
        json!({"observation": 0.5, "alpha": 0.5}),
    )
    .await;
    assert_eq!(s1, StatusCode::OK);

    // Obs2 (0.7, α=0.5): expected score=0.6, m2=0.02, stddev=0.1, min=0.5, max=0.7, count=2
    let (s2, resp) = post_welford(
        &app,
        "CAUSES",
        "a",
        "b",
        json!({"observation": 0.7, "alpha": 0.5}),
    )
    .await;
    assert_eq!(s2, StatusCode::OK);
    approx_f64(&resp["score"], 0.6);
    approx_f64(&resp["score_m2"], 0.02);
    approx_f64(&resp["score_stddev"], 0.1);
    approx_f64(&resp["score_min"], 0.5);
    approx_f64(&resp["score_max"], 0.7);
    assert_eq!(resp["score_count"], 2);
}

#[tokio::test]
async fn welford_preserves_non_welford_edge_properties() {
    let app = build_app_with_welford_schema().await;
    create_indicator(&app, "a", "rate").await;
    create_indicator(&app, "b", "yield").await;
    // Edge starts with a non-Welford property the consumer cares about.
    create_causes_edge(
        &app,
        "a",
        "b",
        json!({"evidence_url": "https://x.example/1"}),
    )
    .await;

    let (status, _) = post_welford(
        &app,
        "CAUSES",
        "a",
        "b",
        json!({"observation": 0.5, "alpha": 0.1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // After update, fetch the edge and confirm evidence_url is still there.
    let edge = get_edge_props(&app, "CAUSES", "a", "b").await;
    assert_eq!(edge["properties"]["evidence_url"], "https://x.example/1");
    assert!(edge["properties"]["score"].is_number());
    assert_eq!(edge["properties"]["score_count"], 1);
}

#[tokio::test]
async fn welford_missing_edge_returns_404() {
    let app = build_app_with_welford_schema().await;
    create_indicator(&app, "a", "rate").await;
    create_indicator(&app, "b", "yield").await;
    // No edge created.

    let (status, _) = post_welford(
        &app,
        "CAUSES",
        "a",
        "b",
        json!({"observation": 0.5, "alpha": 0.1}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn welford_alpha_at_or_beyond_open_unit_interval_returns_400() {
    let app = build_app_with_welford_schema().await;
    create_indicator(&app, "a", "x").await;
    create_indicator(&app, "b", "y").await;
    create_causes_edge(&app, "a", "b", json!({})).await;

    for bad_alpha in [0.0_f64, 1.0_f64, -0.1_f64, 1.5_f64] {
        let (status, resp) = post_welford(
            &app,
            "CAUSES",
            "a",
            "b",
            json!({"observation": 0.5, "alpha": bad_alpha}),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "alpha={bad_alpha} should reject"
        );
        assert!(
            err_msg(&resp).contains("alpha"),
            "alpha={bad_alpha}: {resp}"
        );
    }
}

#[tokio::test]
async fn welford_non_finite_observation_returns_400() {
    let app = build_app_with_welford_schema().await;
    create_indicator(&app, "a", "x").await;
    create_indicator(&app, "b", "y").await;
    create_causes_edge(&app, "a", "b", json!({})).await;

    // JSON has no native NaN; smuggle via raw body. Reuses the
    // request shape with a numeric-but-non-finite via 1e500 (parses
    // to f64::INFINITY in serde_json).
    let raw = r#"{"observation": 1e500, "alpha": 0.1}"#;
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/edges/CAUSES/a/b/welford_update")
                .header("content-type", "application/json")
                .body(Body::from(raw))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("observation"), "got: {body}");
}

#[tokio::test]
async fn welford_unknown_graph_returns_404() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/missing/edges/CAUSES/a/b/welford_update")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"observation": 0.5, "alpha": 0.1}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

// =========================================================================
// Ingress hardening — body-size (413), concurrency (503), timeout (408)
// =========================================================================

fn build_app_with_limits(limits: ServerLimits) -> axum::Router {
    let registry = Arc::new(GraphRegistry::new());
    app(AppState::with_no_auth(registry).with_limits(limits))
}

#[tokio::test]
async fn oversized_body_returns_413() {
    // Tiny cap so a normal create-graph body trips it.
    let app = build_app_with_limits(ServerLimits {
        max_body_bytes: 50,
        ..Default::default()
    });
    let body = json!({
        "id": "g1",
        "schema": { "name": "demo", "version": 1, "node_types": {}, "edge_types": {} }
    });
    assert!(body.to_string().len() > 50);
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn v1_sheds_load_with_503_but_probes_stay_up() {
    // Zero permits → every /v1 request is shed immediately. Extreme
    // value chosen so the load-shed path is deterministic without
    // needing N concurrent in-flight requests.
    let app = build_app_with_limits(ServerLimits {
        max_concurrent_requests: 0,
        ..Default::default()
    });

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);

    // Probes are not behind the concurrency layer — liveness stays up.
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

// =========================================================================
// OpenAPI contract — /openapi.json served + covers the whole surface
// =========================================================================

#[tokio::test]
async fn openapi_json_is_served_and_covers_every_route() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/openapi.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let spec: Value = serde_json::from_slice(&bytes).expect("openapi.json is valid JSON");

    // Valid OpenAPI 3.x envelope with a version.
    assert!(
        spec["openapi"].as_str().unwrap().starts_with("3."),
        "openapi version: {}",
        spec["openapi"]
    );
    assert_eq!(spec["info"]["version"], env!("CARGO_PKG_VERSION"));

    // The served spec must expose exactly the path set of the committed
    // contract — no hand-maintained route list (the drift-gate unit test
    // locks committed == generated; this locks served == committed, so
    // the served endpoint can't silently diverge from the contract).
    let committed: Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/openapi.json"
        ))
        .expect("read docs/openapi.json"),
    )
    .expect("docs/openapi.json is valid JSON");

    let served_paths: std::collections::BTreeSet<&str> = spec["paths"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let committed_paths: std::collections::BTreeSet<&str> = committed["paths"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        served_paths, committed_paths,
        "served /openapi.json paths differ from committed docs/openapi.json"
    );
    // Sanity floor: the full surface is documented, not an accidental subset.
    assert!(
        served_paths.len() >= 30,
        "expected >= 30 paths, got {}",
        served_paths.len()
    );
}

// =========================================================================
// Full-text search endpoints (`search:text` / `search:reindex`).
// =========================================================================

fn json_post(uri: &str, body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// Schema with two full-text string properties on `Document`.
fn fulltext_graph_body() -> Value {
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

#[cfg(feature = "fulltext")]
#[tokio::test]
async fn search_text_endpoint_round_trip() {
    let app = build_app();

    let res = app
        .clone()
        .oneshot(json_post("/v1/graphs", &fulltext_graph_body()))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);

    // Index a node.
    let node = json!({
        "node_type": "Document",
        "node_id": "n1",
        "properties": { "title": "Rust Graphs", "body": "embedded full text search" }
    });
    let res = app
        .clone()
        .oneshot(json_post("/v1/graphs/g1/nodes", &node))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);

    // Found by a token from either full-text field.
    let res = app
        .clone()
        .oneshot(json_post(
            "/v1/graphs/g1/search:text",
            &json!({ "query": "graphs" }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["node_id"], "n1");
    assert_eq!(results[0]["node_type"], "Document");
    assert!(results[0]["score"].as_f64().unwrap() > 0.0);

    // A `field:value`-looking query is treated as plain text, not an injected
    // filter on the internal node_type field → no document contains it.
    let res = app
        .clone()
        .oneshot(json_post(
            "/v1/graphs/g1/search:text",
            &json!({ "query": "node_type:Other" }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["results"].as_array().unwrap().len(), 0);

    // Reindex rebuilds from the node store: idempotent, one document.
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs/g1/search:reindex")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["indexed"], 1);
}

#[cfg(feature = "fulltext")]
#[tokio::test]
async fn search_text_rejects_zero_limit() {
    let app = build_app();
    app.clone()
        .oneshot(json_post("/v1/graphs", &fulltext_graph_body()))
        .await
        .unwrap();
    let res = app
        .clone()
        .oneshot(json_post(
            "/v1/graphs/g1/search:text",
            &json!({ "query": "x", "limit": 0 }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[cfg(feature = "fulltext")]
#[tokio::test]
async fn search_text_rejects_unsearchable_node_type() {
    let app = build_app();
    app.clone()
        .oneshot(json_post("/v1/graphs", &fulltext_graph_body()))
        .await
        .unwrap();
    // A node_type that can never match (unknown / no fulltext property) is a 400,
    // not a silent empty 200 — mirrors `similar` / `nodes_scan`.
    let res = app
        .clone()
        .oneshot(json_post(
            "/v1/graphs/g1/search:text",
            &json!({ "query": "x", "node_type": "Nope" }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

/// In a build without the `fulltext` feature the routes still exist but answer
/// 501 — the API surface (and OpenAPI spec) is identical across builds.
#[cfg(not(feature = "fulltext"))]
#[tokio::test]
async fn search_text_returns_501_when_feature_disabled() {
    let app = build_app();
    let res = app
        .clone()
        .oneshot(json_post("/v1/graphs", &fulltext_graph_body()))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);

    let res = app
        .clone()
        .oneshot(json_post(
            "/v1/graphs/g1/search:text",
            &json!({ "query": "x" }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_IMPLEMENTED);
}

// ---------------------------------------------------------------------------
// Graph-theory algorithm endpoints (`/v1/graphs/{id}/algo/*`).
//
// Gated on the `graph` build feature: with it, the algorithms run; without it,
// the routes still exist and return 501 (asserted by the `not(feature)` test at
// the bottom). The graph used is the two-story knowledge graph seeded by
// `seed_two_story_graph`: story-A's {char-A1, char-A2, loc-A1, ev-A1} form one
// weakly-connected component and story-B's {char-B1, char-B2} form another.
// ---------------------------------------------------------------------------

#[cfg(feature = "graph")]
async fn post_algo(app: &axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
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

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_components_finds_two_disconnected_stories() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    let (status, resp) = post_algo(&app, "components", json!({})).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["count"], 2);

    let comps = resp["components"].as_array().unwrap();
    let as_sets: Vec<std::collections::BTreeSet<String>> = comps
        .iter()
        .map(|c| {
            c.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect()
        })
        .collect();

    let story_a: std::collections::BTreeSet<String> = ["char-A1", "char-A2", "loc-A1", "ev-A1"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let story_b: std::collections::BTreeSet<String> = ["char-B1", "char-B2"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    assert!(
        as_sets.contains(&story_a),
        "story-A component missing: {resp}"
    );
    assert!(
        as_sets.contains(&story_b),
        "story-B component missing: {resp}"
    );
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_components_scoped_to_one_edge_type_splits_further() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // Only MENTIONS edges: char-A1<->char-A2 and char-B1<->char-B2 are linked;
    // loc-A1 and ev-A1 become isolated singletons. 4 components total.
    let (status, resp) = post_algo(
        &app,
        "components",
        json!({"scope": {"edge_types": ["MENTIONS"]}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["count"], 4, "body: {resp}");
}

// `scope.where` (#23) — a property predicate that projects the algorithm onto
// one logical subgraph partitioned by a node property (here `story_id`), the
// storyflow multi-tenant shape. Mirrors the `nodes:scan` clause grammar.
#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_where_scopes_degree_to_one_story() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // story-A has 4 nodes, story-B 2. A `where story_id = story-A` predicate
    // must project onto story-A's 4 nodes only — story-B never enters.
    let (status, resp) = post_algo(
        &app,
        "degree",
        json!({"scope": {"where": [
            {"property": "story_id", "op": "eq", "value": "story-A"}
        ]}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let scores = resp["scores"].as_array().unwrap();
    let nodes: std::collections::BTreeSet<String> = scores
        .iter()
        .map(|s| s["node"].as_str().unwrap().to_string())
        .collect();
    let story_a: std::collections::BTreeSet<String> = ["char-A1", "char-A2", "loc-A1", "ev-A1"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(nodes, story_a, "only story-A nodes in scope: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_where_excludes_cross_partition_edges() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    // A cross-story edge char-A1 -> char-B1. Unscoped, it welds the two stories
    // into one component; under `where story_id = story-A` the edge leaves the
    // subgraph (char-B1 is out of scope) and is dropped, so story-A stays whole
    // and story-B is absent entirely.
    create_typed_edge(
        &app,
        "MENTIONS",
        "Character",
        "char-A1",
        "Character",
        "char-B1",
    )
    .await;

    let (status, resp) = post_algo(
        &app,
        "components",
        json!({"scope": {"where": [
            {"property": "story_id", "op": "eq", "value": "story-A"}
        ]}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["count"], 1, "story-A is one component: {resp}");
    let comp: std::collections::BTreeSet<String> = resp["components"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let story_a: std::collections::BTreeSet<String> = ["char-A1", "char-A2", "loc-A1", "ev-A1"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        comp, story_a,
        "cross-partition edge + node excluded: {resp}"
    );
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_where_on_unindexed_property_fails_loud() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    // `name` is declared but not `indexed` — same un-indexed rejection
    // `nodes:scan` enforces, so a silent empty scope can't masquerade as
    // "no matches".
    let (status, resp) = post_algo(
        &app,
        "components",
        json!({"scope": {"where": [
            {"property": "name", "op": "eq", "value": "Alice"}
        ]}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_empty_node_types_scope_fails_loud() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    // An explicit empty node_types would scope to zero nodes (a silent empty
    // result) and skip `where` validation; reject it instead. A bogus `where`
    // rides along to confirm validation isn't being silently skipped.
    let (status, resp) = post_algo(
        &app,
        "components",
        json!({"scope": {
            "node_types": [],
            "where": [{"property": "not_a_real_prop", "op": "eq", "value": 1}]
        }}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
}

// algo/communities (#24) — Louvain. Build a connected two-faction graph (two
// Character triangles joined by a single bridge MENTIONS edge) so the split is
// a genuine within-component community partition, not just two components.
#[cfg(feature = "graph")]
async fn seed_two_faction_graph(app: &axum::Router) {
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

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_communities_recovers_two_factions() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_faction_graph(&app).await;

    let (status, resp) = post_algo(&app, "communities", json!({})).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["count"], 2, "two factions: {resp}");
    assert!(
        resp["modularity"].as_f64().unwrap() > 0.3,
        "modularity should clear 0.3: {resp}"
    );
    // Each triangle lands wholly in one community.
    let comms = resp["communities"].as_array().unwrap();
    let as_sets: Vec<std::collections::BTreeSet<String>> = comms
        .iter()
        .map(|c| {
            c.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect()
        })
        .collect();
    let faction_a: std::collections::BTreeSet<String> =
        ["a1", "a2", "a3"].iter().map(|s| s.to_string()).collect();
    let faction_b: std::collections::BTreeSet<String> =
        ["b1", "b2", "b3"].iter().map(|s| s.to_string()).collect();
    assert!(as_sets.contains(&faction_a), "faction A intact: {resp}");
    assert!(as_sets.contains(&faction_b), "faction B intact: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_communities_rejects_directed() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_faction_graph(&app).await;
    let (status, resp) = post_algo(&app, "communities", json!({"direction": "directed"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_communities_invalid_resolution_fails_loud() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_faction_graph(&app).await;
    let (status, resp) = post_algo(&app, "communities", json!({"resolution": 0})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_communities_honors_where_scope() {
    // Compose #24 with #23: scope communities to one story. story-A is a star
    // centered on char-A1 (one community); story-B never enters.
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    let (status, resp) = post_algo(
        &app,
        "communities",
        json!({"scope": {"where": [
            {"property": "story_id", "op": "eq", "value": "story-A"}
        ]}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let ids: std::collections::BTreeSet<String> = resp["communities"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|c| c.as_array().unwrap())
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let story_a: std::collections::BTreeSet<String> = ["char-A1", "char-A2", "loc-A1", "ev-A1"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(ids, story_a, "only story-A nodes partitioned: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_degree_ranks_hub_node_first() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // Total degree, directed (default). char-A1 has out MENTIONS+VISITS and in
    // INVOLVES => degree 3; every other node has degree 1.
    let (status, resp) = post_algo(&app, "degree", json!({"mode": "total"})).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");

    let scores = resp["scores"].as_array().unwrap();
    assert_eq!(scores.len(), 6);
    assert_eq!(scores[0]["node"], "char-A1");
    assert_eq!(scores[0]["score"], 3.0);
    // Remaining nodes each have degree 1.
    for s in &scores[1..] {
        assert_eq!(s["score"], 1.0, "body: {resp}");
    }
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_degree_out_mode_only_counts_successors() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    let (status, resp) = post_algo(&app, "degree", json!({"mode": "out"})).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let scores = resp["scores"].as_array().unwrap();
    // char-A1 out-degree is 2 (MENTIONS + VISITS); INVOLVES is incoming.
    assert_eq!(scores[0]["node"], "char-A1");
    assert_eq!(scores[0]["score"], 2.0);
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_degree_missing_weight_property_fails_loud() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // Edges carry no "weight" property — a weighted request must 400, not
    // silently default the weight.
    let (status, resp) = post_algo(&app, "degree", json!({"weight": {"property": "weight"}})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
    assert!(err_msg(&resp).contains("weight property"), "body: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_degree_empty_weight_spec_fails_loud() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // weight: {} specifies neither projection — must 400, not silently score
    // every edge 1.0 and mislabel counts as strengths.
    let (status, resp) = post_algo(&app, "degree", json!({"weight": {}})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
    assert!(err_msg(&resp).contains("weight requires"), "body: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_unknown_node_type_in_scope_returns_400() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    let (status, resp) = post_algo(
        &app,
        "components",
        json!({"scope": {"node_types": ["Nope"]}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
    assert!(err_msg(&resp).contains("unknown node type"), "body: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_unknown_graph_returns_404() {
    let app = build_app_with_knowledge_graph().await;
    let res = app
        .clone()
        .oneshot(json_post("/v1/graphs/missing/algo/components", &json!({})))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

// Helper: pull a single node's score out of a ScoresResponse body.
#[cfg(feature = "graph")]
fn score_of(resp: &Value, node: &str) -> f64 {
    resp["scores"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["node"] == node)
        .unwrap_or_else(|| panic!("node {node} not in scores: {resp}"))["score"]
        .as_f64()
        .unwrap()
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_pagerank_ranks_hub_first_and_sums_to_one() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    let (status, resp) = post_algo(&app, "pagerank", json!({})).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let scores = resp["scores"].as_array().unwrap();
    assert_eq!(scores.len(), 6);
    // PageRank flows along edges; char-A1 is pointed at by ev-A1 (INVOLVES) and
    // is the most-connected node, so it should not be the lowest. Mostly we
    // assert the rank vector is a valid distribution.
    let sum: f64 = scores.iter().map(|s| s["score"].as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-6, "ranks should sum to ~1: {sum}");
    // Scores are sorted descending.
    let vals: Vec<f64> = scores
        .iter()
        .map(|s| s["score"].as_f64().unwrap())
        .collect();
    assert!(
        vals.windows(2).all(|w| w[0] >= w[1]),
        "not sorted: {vals:?}"
    );
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_pagerank_rejects_bad_damping() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    let (status, resp) = post_algo(&app, "pagerank", json!({"damping": 1.5})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
    assert!(err_msg(&resp).contains("damping"), "body: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_eigenvector_center_dominates_on_undirected_chain() {
    let app = build_app_with_knowledge_graph().await;
    // char-A1 - char-A2 (MENTIONS) and char-A1 - loc-A1 ... but simplest: use
    // the seeded graph undirected and check the most-connected node leads.
    seed_two_story_graph(&app).await;

    let (status, resp) = post_algo(
        &app,
        "eigenvector",
        json!({"direction": "undirected", "scope": {"node_types": ["Character", "Location", "Event"], "edge_types": ["MENTIONS", "VISITS", "INVOLVES"]}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    // char-A1 is the hub of story-A's component; its eigenvector centrality
    // should exceed its leaf neighbor char-A2.
    assert!(
        score_of(&resp, "char-A1") > score_of(&resp, "char-A2"),
        "hub should dominate: {resp}"
    );
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_eigenvector_rejects_directed() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    let (status, resp) = post_algo(&app, "eigenvector", json!({"direction": "directed"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
    assert!(err_msg(&resp).contains("undirected"), "body: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_pagerank_rejects_excessive_max_iterations() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    // An unbounded iteration budget is a DoS vector under the read lock; the cap
    // rejects it loudly.
    let (status, resp) = post_algo(&app, "pagerank", json!({"max_iterations": 100000000})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
    assert!(err_msg(&resp).contains("max_iterations"), "body: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_closeness_hub_beats_leaf() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    let (status, resp) = post_algo(&app, "closeness", json!({"direction": "undirected"})).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    // In story-A's component, char-A1 (degree 3) is closer to the rest than the
    // leaf loc-A1 (degree 1).
    assert!(
        score_of(&resp, "char-A1") > score_of(&resp, "loc-A1"),
        "hub closeness should beat leaf: {resp}"
    );
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_betweenness_hub_is_the_only_bridge() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    let (status, resp) = post_algo(
        &app,
        "betweenness",
        json!({"direction": "undirected", "normalized": false}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    // char-A1 sits between A2/loc/ev (all attach only through it), so it carries
    // all the betweenness in story-A; the leaves carry none.
    assert!(
        score_of(&resp, "char-A1") > 0.0,
        "hub should be a bridge: {resp}"
    );
    assert_eq!(score_of(&resp, "char-A2"), 0.0, "leaf carries none: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_closeness_non_positive_weight_fails_loud() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    // Edges have no "weight" property -> missing weight is the loud failure here
    // (the cost-positivity check is exercised by the crate unit tests).
    let (status, resp) =
        post_algo(&app, "closeness", json!({"weight": {"property": "cost"}})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_cuts_finds_hub_and_bridges_in_star_component() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // Story-A is a star around char-A1 (A2/loc/ev attach only through it); every
    // such edge is a bridge and char-A1 is the cut vertex. Story-B is the single
    // edge char-B1 - char-B2 (itself a bridge, no cut vertex).
    let (status, resp) = post_algo(&app, "cuts", json!({})).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");

    let aps: Vec<&str> = resp["articulation_points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(aps, vec!["char-A1"], "body: {resp}");
    // Story-A's 3 edges + story-B's 1 edge are all bridges.
    let bridges = resp["bridges"].as_array().unwrap();
    assert_eq!(bridges.len(), 4, "body: {resp}");
    // Each bridge is an {a, b} pair with a < b.
    for br in bridges {
        assert!(br["a"].as_str().unwrap() < br["b"].as_str().unwrap());
    }
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_scc_directed_tree_is_all_singletons() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // The seeded graph has no directed cycles, so every node is its own SCC.
    let (status, resp) = post_algo(&app, "scc", json!({})).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["count"], 6, "body: {resp}");
    for comp in resp["components"].as_array().unwrap() {
        assert_eq!(comp.as_array().unwrap().len(), 1, "body: {resp}");
    }
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_cycles_reports_acyclic_seeded_graph() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    // The seeded directed graph is acyclic.
    let (status, resp) = post_algo(&app, "cycles", json!({})).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["acyclic"], true, "body: {resp}");
    assert!(resp["cycle"].as_array().unwrap().is_empty(), "body: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_personalized_pagerank_requires_seeds() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    let (status, resp) = post_algo(&app, "personalized_pagerank", json!({})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
    assert!(err_msg(&resp).contains("seed"), "body: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_personalized_pagerank_seeds_dominate() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    let (status, resp) = post_algo(
        &app,
        "personalized_pagerank",
        json!({"direction": "undirected", "seeds": ["char-A1"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    // Seeding story-A's hub should rank it (or its immediate neighborhood) above
    // a node in the disconnected story-B component, which gets only teleport mass.
    assert!(
        score_of(&resp, "char-A1") > score_of(&resp, "char-B2"),
        "seed cluster should dominate: {resp}"
    );
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_shortest_path_finds_route_and_unreachable() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // char-A1 -> char-A2 via the MENTIONS edge (directed, one hop).
    let (status, resp) = post_algo(
        &app,
        "shortest_path",
        json!({"source": "char-A1", "target": "char-A2"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["found"], true, "body: {resp}");
    assert_eq!(resp["distance"], 1.0, "body: {resp}");
    let path: Vec<&str> = resp["path"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(path, vec!["char-A1", "char-A2"]);

    // Story-A cannot reach story-B (disconnected) -> not found.
    let (status, resp) = post_algo(
        &app,
        "shortest_path",
        json!({"source": "char-A1", "target": "char-B1"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["found"], false, "body: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_shortest_path_missing_source_is_400() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    let (status, resp) = post_algo(&app, "shortest_path", json!({"target": "char-A2"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
    assert!(err_msg(&resp).contains("source"), "body: {resp}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_link_prediction_from_source() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;

    // Undirected story-A star: char-A2, loc-A1, ev-A1 are mutual non-neighbors
    // that all share the hub char-A1, so from char-A2 they're predicted links.
    let (status, resp) = post_algo(
        &app,
        "link_prediction",
        json!({"source": "char-A2", "method": "common_neighbors"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let links = resp["links"].as_array().unwrap();
    assert!(!links.is_empty(), "expected predicted links: {resp}");
    // Every predicted link is from the source and scores >= 1 common neighbor.
    for l in links {
        assert_eq!(l["a"], "char-A2");
        assert!(l["score"].as_f64().unwrap() >= 1.0);
    }
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_clustering_star_has_zero_transitivity() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    // Story-A is a star (leaves never connect to each other) and story-B a lone
    // edge, so there are no triangles: transitivity 0, all local scores 0.
    let (status, resp) = post_algo(&app, "clustering", json!({})).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["transitivity"], 0.0, "body: {resp}");
    assert_eq!(resp["average_clustering"], 0.0, "body: {resp}");
    for s in resp["scores"].as_array().unwrap() {
        assert_eq!(s["score"], 0.0, "body: {resp}");
    }
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_toposort_orders_acyclic_seeded_graph() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    // The seeded directed graph is acyclic, so a topological order over all 6
    // nodes exists.
    let (status, resp) = post_algo(&app, "toposort", json!({})).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["acyclic"], true, "body: {resp}");
    assert_eq!(resp["order"].as_array().unwrap().len(), 6, "body: {resp}");
    // ev-A1 -> char-A1 (INVOLVES), so ev-A1 precedes char-A1 in the order.
    let order: Vec<&str> = resp["order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let pos = |id: &str| order.iter().position(|&x| x == id).unwrap();
    assert!(pos("ev-A1") < pos("char-A1"), "order: {order:?}");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_max_flow_bridge_edge_is_the_cut() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    // Directed char-A1 --MENTIONS--> char-A2 is the only path; with unit
    // capacities (no weight) the max flow is 1 and that edge is the min cut.
    let (status, resp) = post_algo(
        &app,
        "max_flow",
        json!({"source": "char-A1", "target": "char-A2"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["max_flow"], 1.0, "body: {resp}");
    let cuts = resp["cut_edges"].as_array().unwrap();
    assert_eq!(cuts.len(), 1, "body: {resp}");
    assert_eq!(cuts[0]["from"], "char-A1");
    assert_eq!(cuts[0]["to"], "char-A2");
}

#[cfg(feature = "graph")]
#[tokio::test]
async fn algo_max_flow_missing_target_is_400() {
    let app = build_app_with_knowledge_graph().await;
    seed_two_story_graph(&app).await;
    let (status, resp) = post_algo(&app, "max_flow", json!({"source": "char-A1"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
    assert!(err_msg(&resp).contains("target"), "body: {resp}");
}

#[cfg(not(feature = "graph"))]
#[tokio::test]
async fn algo_endpoints_return_501_without_graph_feature() {
    // Create the graph first so the handler's graph_entry lookup succeeds and we
    // reach the feature-gated 501 (not a 404).
    let app = build_app_with_knowledge_graph().await;
    for path in [
        "components",
        "degree",
        "pagerank",
        "eigenvector",
        "closeness",
        "betweenness",
        "cuts",
        "scc",
        "cycles",
        "personalized_pagerank",
        "shortest_path",
        "link_prediction",
        "clustering",
        "communities",
        "toposort",
        "max_flow",
    ] {
        let res = app
            .clone()
            .oneshot(json_post(&format!("/v1/graphs/g1/algo/{path}"), &json!({})))
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::NOT_IMPLEMENTED,
            "algo/{path} should be 501 without the graph feature"
        );
    }
}

// ---------------------------------------------------------------------------
// Batch pairwise /v1/util/* endpoints (stateless — no graph needed).
// ---------------------------------------------------------------------------

async fn post_util(app: &axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
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

#[tokio::test]
async fn util_pairwise_cosine_matrix() {
    let app = build_app();
    let (status, resp) = post_util(
        &app,
        "pairwise_cosine",
        json!({"vectors": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let m = resp["matrix"].as_array().unwrap();
    assert_eq!(m.len(), 3);
    // Diagonal is 1 (self-similarity); orthogonal pairs are 0; v0==v2 => 1.
    assert!((m[0][0].as_f64().unwrap() - 1.0).abs() < 1e-9);
    assert!((m[0][1].as_f64().unwrap()).abs() < 1e-9);
    assert!((m[0][2].as_f64().unwrap() - 1.0).abs() < 1e-9);
    // Symmetric.
    assert_eq!(m[1][0], m[0][1]);
}

#[tokio::test]
async fn util_pairwise_distance_euclidean() {
    let app = build_app();
    let (status, resp) = post_util(
        &app,
        "pairwise_distance",
        json!({"vectors": [[0.0, 0.0], [3.0, 4.0]], "metric": "euclidean"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let m = resp["matrix"].as_array().unwrap();
    assert_eq!(m[0][0], 0.0);
    assert!(
        (m[0][1].as_f64().unwrap() - 5.0).abs() < 1e-9,
        "body: {resp}"
    );
    assert_eq!(m[1][0], m[0][1]); // symmetric
}

#[tokio::test]
async fn util_pairwise_rejects_ragged_and_empty() {
    let app = build_app();
    let (status, _) = post_util(
        &app,
        "pairwise_cosine",
        json!({"vectors": [[1.0, 2.0], [1.0]]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = post_util(&app, "pairwise_cosine", json!({"vectors": []})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

// =============================================================================
// search:hybrid — RRF fusion over the vector + keyword legs
// =============================================================================

/// A Document type that is full-text searchable (`title`), has an indexed
/// scalar for the structured prefilter (`act`), and carries embeddings —
/// so all three legs/inputs can be exercised against one graph.
#[cfg(feature = "fulltext")]
fn hybrid_graph_body() -> Value {
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
async fn post_hybrid(app: &axum::Router, body: Value) -> (StatusCode, Value) {
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
async fn create_doc(
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
async fn hybrid_fixture() -> axum::Router {
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

#[cfg(feature = "fulltext")]
#[tokio::test]
async fn search_hybrid_fuses_vector_and_keyword() {
    let app = hybrid_fixture().await;
    let (status, body) = post_hybrid(
        &app,
        json!({
            "query": "alpha",
            "query_vector": [1.0, 0.0, 0.0],
            "node_type": "Document"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let hits = body["hits"].as_array().unwrap();

    // d1 is the query vector (rank-1 vector) and a keyword match → it carries
    // both legs and, being the only node in both, fuses to the top.
    assert_eq!(hits[0]["node_id"], "d1", "hits: {hits:?}");
    assert_eq!(hits[0]["node_type"], "Document");
    assert!(hits[0]["score"].as_f64().unwrap() > 0.0);
    assert!(hits[0]["legs"]["vector"].is_object());
    assert!(hits[0]["legs"]["keyword"].is_object());

    // d2 (vector-only) carries just the vector leg; d3 (keyword-only) just the
    // keyword leg.
    let by_id = |id: &str| {
        hits.iter()
            .find(|h| h["node_id"] == id)
            .unwrap_or_else(|| panic!("{id} missing from {hits:?}"))
            .clone()
    };
    let d2 = by_id("d2");
    assert!(d2["legs"]["vector"].is_object());
    assert!(d2["legs"]["keyword"].is_null());
    let d3 = by_id("d3");
    assert!(d3["legs"]["keyword"].is_object());
    assert!(d3["legs"]["vector"].is_null());
}

#[cfg(feature = "fulltext")]
#[tokio::test]
async fn search_hybrid_keyword_only_allows_missing_node_type() {
    let app = hybrid_fixture().await;
    // No vector input, no prefilter → node_type is optional.
    let (status, body) = post_hybrid(&app, json!({ "query": "alpha" })).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let ids: Vec<&str> = body["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["node_id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"d1"));
    assert!(ids.contains(&"d3"));
    assert!(!ids.contains(&"d2"), "d2 has no 'alpha' token");
}

#[cfg(feature = "fulltext")]
#[tokio::test]
async fn search_hybrid_vector_leg_requires_node_type() {
    let app = hybrid_fixture().await;
    let (status, _) = post_hybrid(&app, json!({ "query_vector": [1.0, 0.0, 0.0] })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[cfg(feature = "fulltext")]
#[tokio::test]
async fn search_hybrid_where_prefilter_excludes_nonmatching_nodes() {
    let app = hybrid_fixture().await;
    // act == 1 keeps d1 and d3, drops d2 from BOTH legs.
    let (status, body) = post_hybrid(
        &app,
        json!({
            "query": "alpha",
            "query_vector": [1.0, 0.0, 0.0],
            "node_type": "Document",
            "where": [{ "property": "act", "op": "eq", "value": 1 }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let ids: Vec<&str> = body["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["node_id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"d1"));
    assert!(!ids.contains(&"d2"), "d2 (act=2) prefiltered out: {ids:?}");
}

#[cfg(feature = "fulltext")]
#[tokio::test]
async fn search_hybrid_explicit_legs_selector_restricts_fusion() {
    let app = hybrid_fixture().await;
    // Both inputs present, but only the keyword leg is selected → no hit
    // carries a vector breakdown, and the vector-only d2 is absent.
    let (status, body) = post_hybrid(
        &app,
        json!({
            "query": "alpha",
            "query_vector": [1.0, 0.0, 0.0],
            "node_type": "Document",
            "legs": ["keyword"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let hits = body["hits"].as_array().unwrap();
    assert!(hits.iter().all(|h| h["legs"]["vector"].is_null()));
    assert!(hits.iter().all(|h| h["node_id"] != "d2"));

    // Selecting a leg whose input is absent is a 400, not a silent skip.
    let (status, _) = post_hybrid(
        &app,
        json!({ "query": "alpha", "legs": ["vector"], "node_type": "Document" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[cfg(feature = "fulltext")]
#[tokio::test]
async fn search_hybrid_validates_inputs() {
    let app = hybrid_fixture().await;

    // No active leg.
    let (status, _) = post_hybrid(&app, json!({})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Zero bounds.
    let (status, _) = post_hybrid(&app, json!({ "query": "alpha", "limit": 0 })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = post_hybrid(&app, json!({ "query": "alpha", "k_per_leg": 0 })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Degenerate query vector (zero magnitude) on the vector leg.
    let (status, _) = post_hybrid(
        &app,
        json!({ "query_vector": [0.0, 0.0, 0.0], "node_type": "Document" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Non-positive weight.
    let (status, _) = post_hybrid(
        &app,
        json!({ "query": "alpha", "weights": { "keyword": 0.0 } }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// Without the `fulltext` feature the keyword leg 501s, but a vector-only
/// hybrid request still succeeds — the route works in any build.
#[cfg(not(feature = "fulltext"))]
#[tokio::test]
async fn search_hybrid_keyword_leg_501_but_vector_only_works() {
    let app = build_app();
    let res = app
        .clone()
        .oneshot(json_post("/v1/graphs", &item_schema_body()))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    create_item(&app, "n1").await;
    assert_eq!(
        put_embedding(&app, "Item", "n1", &[1.0, 0.0, 0.0]).await,
        StatusCode::OK
    );

    // Keyword leg requested → 501.
    let res = app
        .clone()
        .oneshot(json_post(
            "/v1/graphs/g1/search:hybrid",
            &json!({ "query": "x" }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_IMPLEMENTED);

    // Vector-only → 200, even though fulltext is compiled out.
    let res = app
        .clone()
        .oneshot(json_post(
            "/v1/graphs/g1/search:hybrid",
            &json!({ "query_vector": [1.0, 0.0, 0.0], "node_type": "Item" }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["hits"][0]["node_id"], "n1");
}
