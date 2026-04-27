//! On-disk persistence integration tests.
//!
//! Runs `dynograph-service` against an `OnDisk` backend rooted at a
//! tempdir. Verifies the create→write→drop→rehydrate→read cycle —
//! the load-bearing behavior of slice 4. RocksDB tests are slow
//! relative to in-memory ones (each open/close is ~50ms on this
//! laptop) but unavoidable; we keep the count low.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tempfile::TempDir;
use tower::ServiceExt;

use dynograph_service::{AppState, GraphRegistry, app};

fn schema_body(id: &str) -> Value {
    json!({
        "id": id,
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
    })
}

fn build_persistent_app(root: &std::path::Path) -> (axum::Router, Arc<GraphRegistry>) {
    let registry = Arc::new(GraphRegistry::on_disk(root));
    let app = app(AppState::with_no_auth(registry.clone()));
    (app, registry)
}

#[tokio::test]
async fn round_trip_survives_drop_and_rehydrate() {
    let tmp = TempDir::new().unwrap();

    // First service incarnation: create graph + write a node.
    {
        let (app, _registry) = build_persistent_app(tmp.path());
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/graphs")
                    .header("content-type", "application/json")
                    .body(Body::from(schema_body("g1").to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::CREATED);

        let create_node = json!({
            "node_type": "Item",
            "node_id": "n1",
            "properties": { "name": "widget" }
        });
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/graphs/g1/nodes")
                    .header("content-type", "application/json")
                    .body(Body::from(create_node.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::CREATED);
    }
    // First service is dropped here — RocksDB lock released.

    // Second service incarnation: rehydrate, read.
    let (app, registry) = build_persistent_app(tmp.path());
    let rehydrated = registry.rehydrate().expect("rehydrate");
    assert_eq!(rehydrated, vec!["g1".to_string()]);

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
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let fetched: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(fetched["properties"]["name"], "widget");
}

#[tokio::test]
async fn delete_graph_removes_on_disk_dir() {
    let tmp = TempDir::new().unwrap();
    let (app, _registry) = build_persistent_app(tmp.path());

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .body(Body::from(schema_body("g1").to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    assert!(
        tmp.path().join("g1").exists(),
        "graph dir should exist after create"
    );

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
    assert_eq!(res.status(), StatusCode::NO_CONTENT);
    assert!(
        !tmp.path().join("g1").exists(),
        "graph dir should be removed after DELETE"
    );
}

#[tokio::test]
async fn rehydrate_is_idempotent() {
    let tmp = TempDir::new().unwrap();

    // Round 1: create.
    {
        let (app, _registry) = build_persistent_app(tmp.path());
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/graphs")
                    .header("content-type", "application/json")
                    .body(Body::from(schema_body("g1").to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::CREATED);
    }

    // Round 2: rehydrate twice, expect the second call to skip the
    // already-loaded id.
    let (_app, registry) = build_persistent_app(tmp.path());
    let first = registry.rehydrate().unwrap();
    assert_eq!(first, vec!["g1".to_string()]);
    let second = registry.rehydrate().unwrap();
    assert!(
        second.is_empty(),
        "second rehydrate should skip already-loaded ids, got {second:?}"
    );
}

#[tokio::test]
async fn rehydrate_on_fresh_root_returns_empty() {
    let tmp = TempDir::new().unwrap();
    let (_app, registry) = build_persistent_app(tmp.path());
    assert!(registry.rehydrate().unwrap().is_empty());
}

#[tokio::test]
async fn rehydrate_fails_loud_on_corrupt_schema_file() {
    let tmp = TempDir::new().unwrap();

    // Plant a corrupt schema.json under a valid graph-id-shaped dir.
    let bad_dir = tmp.path().join("g1");
    std::fs::create_dir_all(&bad_dir).unwrap();
    std::fs::write(bad_dir.join("schema.json"), "{ this is not json").unwrap();

    let (_app, registry) = build_persistent_app(tmp.path());
    let err = registry.rehydrate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("rehydration failed") && msg.contains("g1"),
        "expected loud error citing g1 + rehydration; got {msg}"
    );
}

#[tokio::test]
async fn create_graph_with_invalid_id_returns_400() {
    let tmp = TempDir::new().unwrap();
    let (app, _registry) = build_persistent_app(tmp.path());
    // `..` would be a path-traversal escape; must be rejected before
    // any filesystem op happens.
    let body = json!({
        "id": "../escape",
        "schema": { "name": "demo", "version": 1, "node_types": {}, "edge_types": {} }
    });
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
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    // The escape attempt must not have created a sibling-of-tempdir
    // directory; only the tempdir itself should exist at that level.
    let parent = tmp.path().parent().unwrap();
    assert!(!parent.join("escape").exists());
}
