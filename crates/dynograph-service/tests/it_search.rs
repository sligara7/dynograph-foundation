//! Integration tests — search. Split out of `integration.rs`.

mod common;

use common::*;

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
