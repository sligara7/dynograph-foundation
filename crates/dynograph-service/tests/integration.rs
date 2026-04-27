use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use dynograph_service::{AppState, GraphRegistry, NoAuth, Readiness, app};

fn build_app() -> axum::Router {
    let registry = Arc::new(GraphRegistry::new());
    app(AppState::with_no_auth(registry))
}

#[tokio::test]
async fn create_then_get_round_trips_schema_and_content_hash() {
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
    let fetched: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(fetched["id"], "g1");
    assert_eq!(fetched["content_hash"].as_str().unwrap(), first_hash);
    assert_eq!(fetched["schema"]["name"], "demo");
    assert_eq!(fetched["schema"]["version"], 1);
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
    // mis-attributing this as "graph not found" would mask the real
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

#[tokio::test]
async fn ready_returns_503_before_mark_ready_then_flips() {
    let registry = Arc::new(GraphRegistry::new());
    let readiness = Readiness::not_ready();
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
