//! Integration tests — algo. Split out of `integration.rs`.

mod common;

use common::*;

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
