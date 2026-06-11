//! Integration tests — primitives. Split out of `integration.rs`.

mod common;

use common::*;

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

#[tokio::test]
async fn resolve_or_create_incoming_alias_merges_with_vector() {
    let app = build_app_with_character_graph().await;
    create_character(&app, "char-1", "Thomas Anderson", "story-A").await;
    assert_eq!(
        put_embedding(&app, "Character", "char-1", &[1.0, 0.0, 0.0]).await,
        StatusCode::OK
    );

    // Primary name is below the fuzzy zone vs the existing node ("Trinity"
    // scores 53; "Neo" would score 73 and merge name-to-name); the alias
    // carries the existing node's exact name. Since v0.9.3 alias evidence
    // is vector-gated: with a near-identical embedding the merge proceeds
    // as a vector_merge (never auto_merge) and provenance is reported.
    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Trinity", "story_id": "story-A"},
            "incoming_aliases": ["Thomas Anderson"],
            "embedding": [1.0, 0.0, 0.0]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["id"], "char-1", "body: {resp}");
    assert_eq!(resp["was_created"], false);
    assert_eq!(resp["match_kind"], "vector_merge");
    assert_eq!(resp["match_source"], "incoming_alias_to_name");
}

#[tokio::test]
async fn resolve_or_create_incoming_alias_without_vector_creates_new() {
    let app = build_app_with_character_graph().await;
    create_character(&app, "char-1", "Thomas Anderson", "story-A").await;

    // The documented v0.9.3 trade-off: without vector corroboration an
    // alias-only match is insufficient evidence — create new rather than
    // risk merging two distinct characters who share a descriptor.
    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Neo", "story_id": "story-A"},
            "incoming_aliases": ["Thomas Anderson"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["was_created"], true, "body: {resp}");
    assert_eq!(resp["match_kind"], "created_new");
    assert_eq!(resp["match_source"], Value::Null);
}

#[tokio::test]
async fn resolve_or_create_accepts_aliases_as_wire_alias() {
    let app = build_app_with_character_graph().await;
    create_character(&app, "char-1", "Thomas Anderson", "story-A").await;
    assert_eq!(
        put_embedding(&app, "Character", "char-1", &[1.0, 0.0, 0.0]).await,
        StatusCode::OK
    );

    // Same as the with-vector case through the `aliases` spelling of the
    // field — the alias-sourced provenance proves the field was parsed.
    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Trinity", "story_id": "story-A"},
            "aliases": ["Thomas Anderson"],
            "embedding": [1.0, 0.0, 0.0]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["id"], "char-1", "body: {resp}");
    assert_eq!(resp["was_created"], false);
    assert_eq!(resp["match_source"], "incoming_alias_to_name");
}

#[tokio::test]
async fn resolve_or_create_matches_stored_alias() {
    let app = build_app_with_character_graph().await;
    // Node whose stored aliases (JSON-array-encoded string) include the
    // incoming primary name.
    let body = json!({
        "node_type": "Character",
        "node_id": "char-1",
        "properties": {
            "name": "The Cartographer",
            "story_id": "story-A",
            "aliases": "[\"Mira Sandgrove\"]"
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
    assert_eq!(res.status(), StatusCode::CREATED);

    assert_eq!(
        put_embedding(&app, "Character", "char-1", &[1.0, 0.0, 0.0]).await,
        StatusCode::OK
    );

    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Mira Sandgrove", "story_id": "story-A"},
            "embedding": [1.0, 0.0, 0.0]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(
        resp["id"], "char-1",
        "incoming primary should merge via the stored alias: {resp}"
    );
    assert_eq!(resp["was_created"], false);
    assert_eq!(resp["match_kind"], "vector_merge");
    assert_eq!(resp["match_source"], "name_to_stored_alias");
}

/// THE D2 over-merge pin (delta-review finding, 2026-06-11): two DISTINCT
/// characters sharing a generic alias must not merge — v0.9.2 auto-merged
/// them at score 100 via alias↔stored-alias and silently discarded the
/// incoming character's whole profile.
#[tokio::test]
async fn resolve_or_create_distinct_characters_shared_alias_does_not_merge() {
    let app = build_app_with_character_graph().await;
    let body = json!({
        "node_type": "Character",
        "node_id": "char-1",
        "properties": {
            "name": "Aldous Vane",
            "story_id": "story-A",
            "aliases": "[\"the captain\"]"
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
    assert_eq!(res.status(), StatusCode::CREATED);
    assert_eq!(
        put_embedding(&app, "Character", "char-1", &[1.0, 0.0, 0.0]).await,
        StatusCode::OK
    );

    // Distinct character, distinct profile (orthogonal embedding), same
    // generic alias. No embedding at all must also create new.
    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Mira Chen", "story_id": "story-A"},
            "incoming_aliases": ["the captain"],
            "embedding": [0.0, 1.0, 0.0]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(
        resp["was_created"], true,
        "distinct character must NOT merge via the shared alias: {resp}"
    );
    assert_ne!(resp["id"], "char-1");

    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Rolf Tanner", "story_id": "story-A"},
            "incoming_aliases": ["the captain"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(
        resp["was_created"], true,
        "alias-only match without vector support must create new: {resp}"
    );
}

/// O2: an incoming alias matching two distinct in-scope candidates is
/// non-identifying — excluded from merge justification and reported.
#[tokio::test]
async fn resolve_or_create_reports_ambiguous_aliases() {
    let app = build_app_with_character_graph().await;
    for (id, name) in [("char-1", "Aldous Vane"), ("char-2", "Carla Reyes")] {
        let body = json!({
            "node_type": "Character",
            "node_id": id,
            "properties": {
                "name": name,
                "story_id": "story-A",
                "aliases": "[\"the captain\"]"
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
        assert_eq!(res.status(), StatusCode::CREATED);
    }
    assert_eq!(
        put_embedding(&app, "Character", "char-1", &[1.0, 0.0, 0.0]).await,
        StatusCode::OK
    );

    // Even with an embedding near-identical to char-1's, the shared alias
    // is non-identifying in this story and must not justify the merge.
    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Mira Chen", "story_id": "story-A"},
            "incoming_aliases": ["the captain"],
            "embedding": [1.0, 0.0, 0.0]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["was_created"], true, "body: {resp}");
    assert_eq!(resp["ambiguous_aliases"], json!(["the captain"]));
}

#[tokio::test]
async fn resolve_or_create_alias_miss_still_creates_new() {
    let app = build_app_with_character_graph().await;
    create_character(&app, "char-1", "Mira Sandgrove", "story-A").await;

    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Quartermaster Greaves", "story_id": "story-A"},
            "incoming_aliases": ["Brother Aldous", ""]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["was_created"], true, "body: {resp}");
    assert_eq!(resp["match_kind"], "created_new");
}

#[tokio::test]
async fn resolve_or_create_primary_name_beats_other_nodes_alias_on_tie() {
    let app = build_app_with_character_graph().await;
    // char-1 (created first → earlier in scan order) stores an alias equal
    // to char-2's PRIMARY name. An exact-name query for char-2 must merge
    // into char-2 — another node's alias must not hijack exact-name merges.
    let body = json!({
        "node_type": "Character",
        "node_id": "char-1",
        "properties": {
            "name": "The Cartographer",
            "story_id": "story-A",
            "aliases": "[\"Mira Sandgrove\"]"
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
    assert_eq!(res.status(), StatusCode::CREATED);
    create_character(&app, "char-2", "Mira Sandgrove", "story-A").await;

    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Mira Sandgrove", "story_id": "story-A"}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(
        resp["id"], "char-2",
        "exact primary-name match must beat another node's alias: {resp}"
    );
}

#[tokio::test]
async fn resolve_or_create_alias_does_not_cross_scope() {
    let app = build_app_with_character_graph().await;
    create_character(&app, "char-1", "Thomas Anderson", "story-A").await;

    // The alias matches a node in a DIFFERENT scope — must create new.
    let (status, resp) = post_resolve(
        &app,
        json!({
            "node_type": "Character",
            "properties": {"name": "Neo", "story_id": "story-B"},
            "incoming_aliases": ["Thomas Anderson"],
            "scope": {"prop": "story_id", "value": "story-B"}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(
        resp["was_created"], true,
        "alias must not match across scopes: {resp}"
    );
}
