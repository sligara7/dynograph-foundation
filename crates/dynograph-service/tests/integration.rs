//! End-to-end test for the slice-1 routes: create a graph, fetch it
//! back, assert the wire shape and `content_hash` determinism.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use dynograph_service::{AppState, GraphRegistry, app};

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

    // POST /v1/graphs → 201 + SchemaResponse
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
    assert_eq!(first_hash.len(), 64, "sha256 hex is 64 chars");

    // GET /v1/graphs/g1 → 200 + same SchemaResponse shape, identical content_hash
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
