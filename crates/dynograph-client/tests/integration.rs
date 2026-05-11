//! End-to-end client ↔ server tests.
//!
//! Each test spins up `dynograph-service` in-process on a free TCP
//! port (`127.0.0.1:0`), exercises the client, and lets axum shut
//! down when the test exits. This is the contract test that pins
//! the wire-shape duplication between `dynograph-service`'s
//! `*_response.rs` modules and `dynograph-client::wire` — if either
//! drifts, the round-trip deserialization fails here in CI before
//! shipping.
//!
//! Tests are NOT marked `#[serial]`; reqwest connections are
//! per-test, and tokio's runtime per `#[tokio::test]` is isolated.

use std::sync::Arc;

use dynograph_client::{ClientError, CreateEdge, DynographClient};
// `Value as DV` keeps assertions on received `properties` honest about
// the wire-shape Value (typed enum from dynograph-core), distinct
// from the `serde_json::Value` we use to *build* request bodies via
// the `json!()` macro.
use dynograph_core::{Schema, Value as DV};
use dynograph_service::{AppState, BearerJwt, GraphRegistry, NoAuth, Readiness, app};
use serde_json::{Map, Value, json};

/// Spin up a fresh in-memory service on a random local port and
/// return `(client_pointing_at_it, shutdown_handle)`. Drop the
/// handle to stop the server.
async fn spawn_service_with_auth(
    auth: Arc<dyn dynograph_service::AuthProvider>,
) -> (DynographClient, tokio::task::JoinHandle<()>) {
    let registry = Arc::new(GraphRegistry::new());
    let readiness = Arc::new(Readiness::ready());
    let state = AppState::new(registry, auth, readiness);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app(state)).await.unwrap();
    });
    let client = DynographClient::new(format!("http://{addr}"));
    (client, server)
}

async fn spawn_service() -> (DynographClient, tokio::task::JoinHandle<()>) {
    spawn_service_with_auth(Arc::new(NoAuth::new())).await
}

fn tiny_schema() -> Schema {
    Schema::from_yaml(
        r#"
schema:
  name: t
  version: 1
  node_types:
    Item:
      properties:
        name: { type: string, required: true }
        tag:  { type: string, indexed: true }
  edge_types:
    Likes:
      from: Item
      to: Item
      properties:
        weight: { type: float }
"#,
    )
    .unwrap()
}

fn props(pairs: &[(&str, Value)]) -> Map<String, Value> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

#[tokio::test]
async fn graph_lifecycle_round_trip() {
    let (client, _server) = spawn_service().await;
    let schema = tiny_schema();

    let created = client.create_graph("g1", &schema).await.unwrap();
    assert_eq!(created.id, "g1");
    assert_eq!(created.content_hash.len(), 64);

    let metadata = client.get_graph("g1").await.unwrap();
    assert_eq!(metadata.id, "g1");
    assert_eq!(metadata.content_hash, created.content_hash);

    let listed = client.list_graphs().await.unwrap();
    assert_eq!(listed, vec!["g1"]);

    let full = client.get_schema("g1").await.unwrap();
    assert_eq!(full.schema.name, "t");

    client.delete_graph("g1").await.unwrap();
    let err = client.get_graph("g1").await.unwrap_err();
    assert_eq!(err.status(), Some(reqwest::StatusCode::NOT_FOUND));
}

#[tokio::test]
async fn replace_schema_returns_new_hash() {
    let (client, _server) = spawn_service().await;
    let initial = client.create_graph("g1", &tiny_schema()).await.unwrap();
    let new_schema = Schema::from_yaml(
        r#"
schema:
  name: t
  version: 2
  node_types:
    Item:
      properties:
        name:     { type: string, required: true }
        tag:      { type: string, indexed: true }
        nickname: { type: string }
  edge_types:
    Likes:
      from: Item
      to: Item
      properties:
        weight: { type: float }
"#,
    )
    .unwrap();
    let updated = client.replace_schema("g1", &new_schema).await.unwrap();
    assert_ne!(updated.content_hash, initial.content_hash);
    assert!(
        updated
            .schema
            .node_types
            .get("Item")
            .unwrap()
            .properties
            .contains_key("nickname")
    );
}

#[tokio::test]
async fn node_crud_and_list_round_trip() {
    let (client, _server) = spawn_service().await;
    client.create_graph("g1", &tiny_schema()).await.unwrap();

    let created = client
        .create_node(
            "g1",
            "Item",
            "n1",
            &props(&[("name", json!("widget")), ("tag", json!("red"))]),
        )
        .await
        .unwrap();
    assert_eq!(created.node_id, "n1");
    assert_eq!(created.properties["name"], DV::String("widget".into()));

    let fetched = client.get_node("g1", "Item", "n1").await.unwrap();
    assert_eq!(fetched.properties["tag"], DV::String("red".into()));

    // PUT REPLACES — properties not in body get dropped (subject to
    // schema defaults re-applying).
    let replaced = client
        .replace_node("g1", "Item", "n1", &props(&[("name", json!("gadget"))]))
        .await
        .unwrap();
    assert_eq!(replaced.properties["name"], DV::String("gadget".into()));
    assert!(!replaced.properties.contains_key("tag"));

    // Re-tag and exercise list-by-prop.
    client
        .replace_node(
            "g1",
            "Item",
            "n1",
            &props(&[("name", json!("gadget")), ("tag", json!("blue"))]),
        )
        .await
        .unwrap();
    client
        .create_node(
            "g1",
            "Item",
            "n2",
            &props(&[("name", json!("widget2")), ("tag", json!("blue"))]),
        )
        .await
        .unwrap();

    let by_type = client.list_nodes("g1", "Item", None).await.unwrap();
    assert_eq!(by_type.nodes.len(), 2);
    let by_filter = client
        .list_nodes("g1", "Item", Some(("tag", "blue")))
        .await
        .unwrap();
    assert_eq!(by_filter.nodes.len(), 2);

    client.delete_node("g1", "Item", "n1").await.unwrap();
    let err = client.get_node("g1", "Item", "n1").await.unwrap_err();
    assert_eq!(err.status(), Some(reqwest::StatusCode::NOT_FOUND));
}

#[tokio::test]
async fn edge_crud_round_trip() {
    let (client, _server) = spawn_service().await;
    client.create_graph("g1", &tiny_schema()).await.unwrap();
    for n in ["a", "b"] {
        client
            .create_node("g1", "Item", n, &props(&[("name", json!(n))]))
            .await
            .unwrap();
    }

    let edge_props = props(&[("weight", json!(0.5))]);
    let created = client
        .create_edge(
            "g1",
            &CreateEdge {
                edge_type: "Likes",
                from_type: "Item",
                from_id: "a",
                to_type: "Item",
                to_id: "b",
                properties: &edge_props,
            },
        )
        .await
        .unwrap();
    assert_eq!(created.from_id, "a");
    assert_eq!(created.to_id, "b");

    let merged = client
        .merge_edge("g1", "Likes", "a", "b", &props(&[("weight", json!(0.9))]))
        .await
        .unwrap();
    assert_eq!(merged.properties["weight"], DV::Float(0.9));

    client.delete_edge("g1", "Likes", "a", "b").await.unwrap();
    let err = client.get_edge("g1", "Likes", "a", "b").await.unwrap_err();
    assert_eq!(err.status(), Some(reqwest::StatusCode::NOT_FOUND));
}

#[tokio::test]
async fn embedding_and_similarity_round_trip() {
    let (client, _server) = spawn_service().await;
    client.create_graph("g1", &tiny_schema()).await.unwrap();
    for (id, _tag) in [("a", "red"), ("b", "blue"), ("c", "red")] {
        client
            .create_node("g1", "Item", id, &props(&[("name", json!(id))]))
            .await
            .unwrap();
    }
    client
        .set_embedding("g1", "Item", "a", &[1.0, 0.0, 0.0])
        .await
        .unwrap();
    client
        .set_embedding("g1", "Item", "b", &[0.95, 0.1, 0.0])
        .await
        .unwrap();
    client
        .set_embedding("g1", "Item", "c", &[0.0, 0.0, 1.0])
        .await
        .unwrap();

    let got = client.get_embedding("g1", "Item", "a").await.unwrap();
    assert_eq!(got.embedding.len(), 3);
    assert!((got.embedding[0] - 1.0).abs() < 1e-6);

    let hits = client
        .similar("g1", "Item", &[1.0, 0.0, 0.0], 3)
        .await
        .unwrap();
    let ids: Vec<&str> = hits.results.iter().map(|h| h.node_id.as_str()).collect();
    assert_eq!(ids[0], "a");
    assert_eq!(ids[1], "b");
    assert_eq!(ids[2], "c");

    client.delete_embedding("g1", "Item", "a").await.unwrap();
    let err = client.get_embedding("g1", "Item", "a").await.unwrap_err();
    assert_eq!(err.status(), Some(reqwest::StatusCode::NOT_FOUND));
}

#[tokio::test]
async fn ops_endpoints_return_expected_text() {
    let (client, _server) = spawn_service().await;
    assert_eq!(client.health().await.unwrap(), "ok");
    assert_eq!(client.ready().await.unwrap(), "ready");
    let metrics = client.metrics().await.unwrap();
    assert!(metrics.contains("dynograph_build_info"), "{metrics}");
}

#[tokio::test]
async fn http_error_preserves_server_body() {
    let (client, _server) = spawn_service().await;
    let err = client.get_graph("does-not-exist").await.unwrap_err();
    match err {
        ClientError::Http { status, body } => {
            assert_eq!(status, reqwest::StatusCode::NOT_FOUND);
            assert!(body.contains("graph not found"), "body was: {body}");
        }
        other => panic!("expected Http error; got {other:?}"),
    }
}

#[tokio::test]
async fn bearer_jwt_rejected_without_token_accepted_with_token() {
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use serde::Serialize;
    use std::time::{SystemTime, UNIX_EPOCH};

    const SECRET: &[u8] = b"slice-12-test-secret";

    #[derive(Serialize)]
    struct Claims {
        sub: String,
        exp: usize,
    }

    let auth: Arc<dyn dynograph_service::AuthProvider> = Arc::new(BearerJwt::new(SECRET));
    let (mut client, _server) = spawn_service_with_auth(auth).await;

    // No token → 401 on a /v1/* request; /health stays public.
    assert_eq!(client.health().await.unwrap(), "ok");
    let err = client.list_graphs().await.unwrap_err();
    assert_eq!(err.status(), Some(reqwest::StatusCode::UNAUTHORIZED));

    // Mint a valid token; replacing the client-side bearer flips the
    // same /v1/* call to 200.
    let exp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as usize
        + 60;
    let token = encode(
        &Header::new(Algorithm::HS256),
        &Claims {
            sub: "alice".to_string(),
            exp,
        },
        &EncodingKey::from_secret(SECRET),
    )
    .unwrap();
    client = client.with_bearer(token);
    assert_eq!(client.list_graphs().await.unwrap(), Vec::<String>::new());
}

// =========================================================================
// v0.5.6 — audit-promoted primitives + new endpoints
// =========================================================================

/// Schema that exercises every v0.5.6-relevant feature: indexed `name`
/// (for nodes:exists / resolve_or_create / nodes:scan), indexed `tag`
/// (for nodes:scan filter), and resolution config (for resolve_or_create).
fn audit_schema() -> Schema {
    Schema::from_yaml(
        r#"
schema:
  name: audit
  version: 1
  node_types:
    Item:
      properties:
        name: { type: string, required: true, indexed: true }
        tag:  { type: string, indexed: true }
      resolution:
        strategy: fuzzy_then_vector
        fuzzy_threshold: 70
        vector_threshold: 0.85
        auto_merge_threshold: 90
  edge_types:
    Likes:
      from: Item
      to: Item
"#,
    )
    .unwrap()
}

async fn setup_audit_graph(client: &DynographClient) {
    client.create_graph("g", &audit_schema()).await.unwrap();
}

#[tokio::test]
async fn batch_round_trip_creates_node_and_edge_atomically() {
    let (client, _server) = spawn_service().await;
    setup_audit_graph(&client).await;

    let body = json!({
        "ops": [
            {"op": "create_node", "node_type": "Item", "node_id": "a",
             "properties": {"name": "Alpha", "tag": "x"}},
            {"op": "create_node", "node_type": "Item", "node_id": "b",
             "properties": {"name": "Beta", "tag": "y"}},
            {"op": "create_edge", "edge_type": "Likes",
             "from_type": "Item", "from_id": "a",
             "to_type": "Item",   "to_id": "b",
             "properties": {}}
        ]
    });
    let resp = client.batch("g", &body).await.unwrap();
    // BatchResponse includes effect counters; assert at least the node count.
    assert_eq!(resp["nodes_created"], 2, "got: {resp}");
    assert_eq!(resp["edges_created"], 1);

    // Verify the writes landed by reading them back.
    let node = client.get_node("g", "Item", "a").await.unwrap();
    assert_eq!(node.node_id, "a");
    let edge = client.get_edge("g", "Likes", "a", "b").await.unwrap();
    assert_eq!(edge.from_id, "a");
    assert_eq!(edge.to_id, "b");
}

#[tokio::test]
async fn resolve_or_create_round_trip_creates_then_auto_merges() {
    let (client, _server) = spawn_service().await;
    setup_audit_graph(&client).await;

    // First call: no candidates, so foundation creates a new node.
    let first = client
        .resolve_or_create(
            "g",
            &json!({
                "node_type": "Item",
                "properties": {"name": "Widget"}
            }),
        )
        .await
        .unwrap();
    assert!(first.was_created);
    assert_eq!(first.match_kind, "created_new");

    // Second call with identical name — fuzzy ratio 100 > auto_merge_threshold.
    let second = client
        .resolve_or_create(
            "g",
            &json!({
                "node_type": "Item",
                "properties": {"name": "Widget"}
            }),
        )
        .await
        .unwrap();
    assert!(!second.was_created);
    assert_eq!(second.match_kind, "auto_merge");
    assert_eq!(second.id, first.id);
}

#[tokio::test]
async fn edges_collect_round_trip_returns_collected_edges() {
    let (client, _server) = spawn_service().await;
    setup_audit_graph(&client).await;

    client
        .create_node("g", "Item", "a", &props(&[("name", json!("A"))]))
        .await
        .unwrap();
    client
        .create_node("g", "Item", "b", &props(&[("name", json!("B"))]))
        .await
        .unwrap();
    client
        .create_edge(
            "g",
            &CreateEdge {
                edge_type: "Likes",
                from_type: "Item",
                from_id: "a",
                to_type: "Item",
                to_id: "b",
                properties: &Map::new(),
            },
        )
        .await
        .unwrap();

    let resp = client
        .edges_collect(
            "g",
            &json!({
                "source": {"type": "Item"},
                "edge_types": ["Likes"],
                "limit": 100
            }),
        )
        .await
        .unwrap();
    let edges = resp["edges"].as_array().expect("edges array");
    assert_eq!(edges.len(), 1, "resp: {resp}");
    assert_eq!(edges[0]["edge_type"], "Likes");
    assert_eq!(edges[0]["from_id"], "a");
    assert_eq!(edges[0]["to_id"], "b");
}

#[tokio::test]
async fn traverse_round_trip_walks_outgoing_edges_from_start() {
    let (client, _server) = spawn_service().await;
    setup_audit_graph(&client).await;

    // Chain: a → b → c via Likes
    for (id, name) in [("a", "A"), ("b", "B"), ("c", "C")] {
        client
            .create_node("g", "Item", id, &props(&[("name", json!(name))]))
            .await
            .unwrap();
    }
    for (from, to) in [("a", "b"), ("b", "c")] {
        client
            .create_edge(
                "g",
                &CreateEdge {
                    edge_type: "Likes",
                    from_type: "Item",
                    from_id: from,
                    to_type: "Item",
                    to_id: to,
                    properties: &Map::new(),
                },
            )
            .await
            .unwrap();
    }

    let resp = client
        .traverse(
            "g",
            &json!({
                "start": {"type": "Item", "id": "a"},
                "traverse": [{"edge_type": "Likes", "direction": "outgoing", "transitive": true}],
                "limit": 10
            }),
        )
        .await
        .unwrap();
    let nodes = resp["nodes"].as_array().expect("nodes array");
    let ids: std::collections::HashSet<&str> = nodes
        .iter()
        .map(|n| n["node_id"].as_str().unwrap())
        .collect();
    // Start (a) is excluded; b and c reachable transitively.
    assert_eq!(ids, ["b", "c"].into_iter().collect());
}

#[tokio::test]
async fn nodes_exists_round_trip_returns_mixed_present_absent() {
    let (client, _server) = spawn_service().await;
    setup_audit_graph(&client).await;

    client
        .create_node(
            "g",
            "Item",
            "widget-1",
            &props(&[("name", json!("Widget"))]),
        )
        .await
        .unwrap();

    let resp = client
        .nodes_exists(
            "g",
            &json!({
                "queries": [
                    {"type": "Item", "name": "Widget"},
                    {"type": "Item", "name": "Ghost"}
                ]
            }),
        )
        .await
        .unwrap();
    assert_eq!(resp.results.len(), 2);
    assert!(resp.results[0].exists);
    assert_eq!(resp.results[0].id.as_deref(), Some("widget-1"));
    assert!(!resp.results[1].exists);
    assert!(resp.results[1].id.is_none());
}

#[tokio::test]
async fn nodes_scan_round_trip_filters_by_indexed_eq() {
    let (client, _server) = spawn_service().await;
    setup_audit_graph(&client).await;

    for (id, name, tag) in [
        ("a", "Alpha", "red"),
        ("b", "Beta", "blue"),
        ("c", "Gamma", "red"),
    ] {
        client
            .create_node(
                "g",
                "Item",
                id,
                &props(&[("name", json!(name)), ("tag", json!(tag))]),
            )
            .await
            .unwrap();
    }

    let resp = client
        .nodes_scan(
            "g",
            &json!({
                "type": "Item",
                "where": [{"property": "tag", "op": "eq", "value": "red"}],
                "return": "ids",
                "limit": 100
            }),
        )
        .await
        .unwrap();
    let ids: std::collections::HashSet<&str> = resp["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(ids, ["a", "c"].into_iter().collect());
}

#[tokio::test]
async fn welford_update_round_trip_initializes_then_evolves_state() {
    let (client, _server) = spawn_service().await;
    setup_audit_graph(&client).await;

    client
        .create_node("g", "Item", "a", &props(&[("name", json!("A"))]))
        .await
        .unwrap();
    client
        .create_node("g", "Item", "b", &props(&[("name", json!("B"))]))
        .await
        .unwrap();
    client
        .create_edge(
            "g",
            &CreateEdge {
                edge_type: "Likes",
                from_type: "Item",
                from_id: "a",
                to_type: "Item",
                to_id: "b",
                properties: &Map::new(),
            },
        )
        .await
        .unwrap();

    let first = client
        .welford_update("g", "Likes", "a", "b", 0.5, 0.5)
        .await
        .unwrap();
    assert_eq!(first.score_count, 1);
    assert!((first.score - 0.5).abs() < 1e-9);

    let second = client
        .welford_update("g", "Likes", "a", "b", 0.7, 0.5)
        .await
        .unwrap();
    assert_eq!(second.score_count, 2);
    // Expected: score = 0.5 + 0.5*(0.7-0.5) = 0.6
    assert!((second.score - 0.6).abs() < 1e-9, "got: {second:?}");
    assert!((second.score_min - 0.5).abs() < 1e-9);
    assert!((second.score_max - 0.7).abs() < 1e-9);
}

// =========================================================================
// v0.5.6 P3 — /v1/util/* pure-math endpoints
// =========================================================================

#[tokio::test]
async fn util_cosine_similarity_round_trip() {
    let (client, _server) = spawn_service().await;
    // [1, 0, 0] · [1, 0, 0] = 1.0
    let resp = client
        .util_cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], None)
        .await
        .unwrap();
    assert!((resp.result - 1.0).abs() < 1e-9);

    // Orthogonal → 0
    let resp = client
        .util_cosine_similarity(&[1.0, 0.0], &[0.0, 1.0], None)
        .await
        .unwrap();
    assert!(resp.result.abs() < 1e-9);
}

#[tokio::test]
async fn util_dot_and_l2_and_euclidean_round_trip() {
    let (client, _server) = spawn_service().await;
    let resp = client
        .util_dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], None)
        .await
        .unwrap();
    assert!((resp.result - 32.0).abs() < 1e-9);

    let resp = client.util_l2_norm(&[3.0, 4.0], None).await.unwrap();
    assert!((resp.result - 5.0).abs() < 1e-9);

    let resp = client
        .util_euclidean_distance(&[0.0, 0.0], &[3.0, 4.0], None)
        .await
        .unwrap();
    assert!((resp.result - 5.0).abs() < 1e-9);
}

#[tokio::test]
async fn util_hadamard_round_trip() {
    let (client, _server) = spawn_service().await;
    let resp = client
        .util_hadamard(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], None)
        .await
        .unwrap();
    assert_eq!(resp.result.len(), 3);
    assert!((resp.result[0] - 4.0).abs() < 1e-9);
    assert!((resp.result[1] - 10.0).abs() < 1e-9);
    assert!((resp.result[2] - 18.0).abs() < 1e-9);
}

#[tokio::test]
async fn util_pearson_round_trip() {
    let (client, _server) = spawn_service().await;
    // Perfect positive correlation
    let resp = client
        .util_pearson_correlation(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 6.0, 8.0, 10.0])
        .await
        .unwrap();
    assert!((resp.result - 1.0).abs() < 1e-9);
}

#[tokio::test]
async fn util_linreg_slope_round_trip() {
    let (client, _server) = spawn_service().await;
    // y = 2x → slope = 2
    let pts = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
    let resp = client.util_linear_regression_slope(&pts).await.unwrap();
    assert!((resp.result - 2.0).abs() < 1e-9);
}

#[tokio::test]
async fn util_fuzzy_string_round_trip() {
    let (client, _server) = spawn_service().await;
    // Identical strings → 100
    let resp = client.util_jaro_winkler("foo", "foo").await.unwrap();
    assert_eq!(resp.result, 100);
    let resp = client
        .util_token_sort_ratio("foo bar", "bar foo")
        .await
        .unwrap();
    // Token sort is permutation-invariant; identical tokens in different order → 100
    assert_eq!(resp.result, 100);
}

#[tokio::test]
async fn util_f32_precision_path() {
    let (client, _server) = spawn_service().await;
    // f32 path takes a different SIMD-friendly impl; same expected value.
    let resp = client
        .util_dot_product(
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            Some(dynograph_client::Precision::F32),
        )
        .await
        .unwrap();
    // f32 has less precision; allow looser tolerance.
    assert!((resp.result - 32.0).abs() < 1e-5);
}

#[tokio::test]
async fn util_mismatched_lengths_returns_400() {
    let (client, _server) = spawn_service().await;
    let err = client
        .util_cosine_similarity(&[1.0, 2.0], &[1.0, 2.0, 3.0], None)
        .await
        .unwrap_err();
    assert_eq!(err.status(), Some(reqwest::StatusCode::BAD_REQUEST));
}

#[tokio::test]
async fn util_pearson_too_few_samples_returns_400() {
    let (client, _server) = spawn_service().await;
    let err = client
        .util_pearson_correlation(&[1.0, 2.0], &[3.0, 4.0])
        .await
        .unwrap_err();
    assert_eq!(err.status(), Some(reqwest::StatusCode::BAD_REQUEST));
}
