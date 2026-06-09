//! Integration tests — edges. Split out of `integration.rs`.

mod common;

use common::*;

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
