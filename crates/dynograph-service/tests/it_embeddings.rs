//! Integration tests — embeddings. Split out of `integration.rs`.

mod common;

use common::*;

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
