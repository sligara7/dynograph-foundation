//! Integration tests — batch. Split out of `integration.rs`.

mod common;

use common::*;

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
async fn batch_dry_run_stops_at_the_first_failure() {
    // dry_run reports the partition up to and INCLUDING the first failing op,
    // then stops — mirroring the commit path, which aborts there (so later ops
    // would never run). Op 1 is valid but comes after the op-0 failure, so it
    // is never evaluated.
    let app = build_app_with_item_graph().await;
    let ops = json!({
        "dry_run": true,
        "ops": [
            {"op": "replace_node", "node_type": "Item", "node_id": "nope1", "properties": {}},
            {"op": "create_node", "node_type": "Item", "node_id": "ok", "properties": {"name": "ok"}},
        ]
    });
    let (status, resp) = post_batch(&app, ops.clone()).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["valid"], false);
    let results = resp["results"].as_array().unwrap();
    assert_eq!(results.len(), 1, "stops at the first failure: {resp}");
    assert_eq!(results[0]["ok"], false);
    assert_eq!(results[0]["index"], 0);
    assert_eq!(results[0]["op"], "replace_node");

    // The commit path (dry_run:false) stops at the same first failure with the
    // unchanged per-op error shape, committing nothing.
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
