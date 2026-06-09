//! Integration tests — nodes. Split out of `integration.rs`.

mod common;

use common::*;

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
