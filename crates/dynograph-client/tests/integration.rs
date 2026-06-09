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

// ---------------------------------------------------------------------
// v0.6.x util additions
// ---------------------------------------------------------------------

#[tokio::test]
async fn util_new_distances_round_trip() {
    let (client, _server) = spawn_service().await;
    // squared euclidean of (0,0)->(3,4) = 25; manhattan = 7.
    let sq = client
        .util_squared_euclidean_distance(&[0.0, 0.0], &[3.0, 4.0], None)
        .await
        .unwrap();
    assert!((sq.result - 25.0).abs() < 1e-9);
    let man = client
        .util_manhattan_distance(&[0.0, 0.0], &[3.0, 4.0], None)
        .await
        .unwrap();
    assert!((man.result - 7.0).abs() < 1e-9);
}

#[tokio::test]
async fn util_elementwise_algebra_round_trip() {
    let (client, _server) = spawn_service().await;
    assert_eq!(
        client
            .util_add(&[1.0, 2.0], &[3.0, 4.0], None)
            .await
            .unwrap()
            .result,
        vec![4.0, 6.0]
    );
    assert_eq!(
        client
            .util_subtract(&[3.0, 4.0], &[1.0, 2.0], None)
            .await
            .unwrap()
            .result,
        vec![2.0, 2.0]
    );
    assert_eq!(
        client
            .util_scale(&[1.0, -2.0], 2.0, None)
            .await
            .unwrap()
            .result,
        vec![2.0, -4.0]
    );
    assert_eq!(
        client.util_negate(&[1.0, -2.0], None).await.unwrap().result,
        vec![-1.0, 2.0]
    );
    assert_eq!(
        client
            .util_hadamard_division(&[6.0, 8.0], &[2.0, 4.0], None)
            .await
            .unwrap()
            .result,
        vec![3.0, 2.0]
    );
    assert_eq!(
        client
            .util_elementwise_power(&[2.0, 3.0], 2.0, None)
            .await
            .unwrap()
            .result,
        vec![4.0, 9.0]
    );
}

#[tokio::test]
async fn util_hadamard_division_zero_divisor_returns_400() {
    let (client, _server) = spawn_service().await;
    let err = client
        .util_hadamard_division(&[1.0, 2.0], &[1.0, 0.0], None)
        .await
        .unwrap_err();
    assert_eq!(err.status(), Some(reqwest::StatusCode::BAD_REQUEST));
}

#[tokio::test]
async fn util_l2_normalize_round_trip_and_zero_400() {
    let (client, _server) = spawn_service().await;
    let n = client.util_l2_normalize(&[3.0, 4.0], None).await.unwrap();
    let mag = (n.result[0] * n.result[0] + n.result[1] * n.result[1]).sqrt();
    assert!((mag - 1.0).abs() < 1e-9);
    let err = client
        .util_l2_normalize(&[0.0, 0.0], None)
        .await
        .unwrap_err();
    assert_eq!(err.status(), Some(reqwest::StatusCode::BAD_REQUEST));
}

#[tokio::test]
async fn util_centroid_round_trip_and_ragged_400() {
    let (client, _server) = spawn_service().await;
    let c = client
        .util_centroid(&[vec![1.0, 2.0], vec![3.0, 6.0]], None)
        .await
        .unwrap();
    assert_eq!(c.result, vec![2.0, 4.0]);
    let err = client
        .util_centroid(&[vec![1.0, 2.0], vec![1.0]], None)
        .await
        .unwrap_err();
    assert_eq!(err.status(), Some(reqwest::StatusCode::BAD_REQUEST));
}

#[tokio::test]
async fn util_descriptive_stats_round_trip() {
    let (client, _server) = spawn_service().await;
    let xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    assert!((client.util_mean(&xs).await.unwrap().result - 5.0).abs() < 1e-9);
    assert!((client.util_variance(&xs).await.unwrap().result - 32.0 / 7.0).abs() < 1e-9);
    assert!(
        (client.util_std_dev(&xs).await.unwrap().result - (32.0_f64 / 7.0).sqrt()).abs() < 1e-9
    );
    assert!(
        (client
            .util_median(&[1.0, 2.0, 3.0, 4.0])
            .await
            .unwrap()
            .result
            - 2.5)
            .abs()
            < 1e-9
    );
    assert!(
        (client
            .util_percentile(&[1.0, 2.0, 3.0, 4.0], 100.0)
            .await
            .unwrap()
            .result
            - 4.0)
            .abs()
            < 1e-9
    );
}

#[tokio::test]
async fn util_softmax_round_trip() {
    let (client, _server) = spawn_service().await;
    let p = client.util_softmax(&[1.0, 2.0, 3.0]).await.unwrap().result;
    let sum: f64 = p.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9);
    assert!(p[0] < p[1] && p[1] < p[2]);
}

#[tokio::test]
async fn util_spearman_round_trip_and_degenerate_400() {
    let (client, _server) = spawn_service().await;
    // Monotonic non-linear → 1.0
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [1.0, 8.0, 27.0, 64.0, 125.0];
    let r = client.util_spearman_correlation(&x, &y).await.unwrap();
    assert!((r.result - 1.0).abs() < 1e-9);
    // Constant input → undefined → 400
    let err = client
        .util_spearman_correlation(&[1.0, 2.0, 3.0], &[5.0, 5.0, 5.0])
        .await
        .unwrap_err();
    assert_eq!(err.status(), Some(reqwest::StatusCode::BAD_REQUEST));
}

#[tokio::test]
async fn util_new_endpoints_f32_precision() {
    let (client, _server) = spawn_service().await;
    let r = client
        .util_manhattan_distance(
            &[0.0, 0.0],
            &[3.0, 4.0],
            Some(dynograph_client::Precision::F32),
        )
        .await
        .unwrap();
    assert!((r.result - 7.0).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// search + pairwise (v0.7.0 surface). The in-process service is compiled
// without the `fulltext` feature here, so keyword paths return 501 — asserted
// directly as the typed passthrough. Vector-only hybrid and the util pairwise
// matrices need no feature and round-trip for real.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn util_pairwise_cosine_round_trip() {
    let (client, _server) = spawn_service().await;
    // Three vectors: v0 ∥ v1 (cosine 1), v0 ⟂ v2 (cosine 0).
    let vectors = vec![vec![1.0, 0.0], vec![2.0, 0.0], vec![0.0, 1.0]];
    let m = client
        .util_pairwise_cosine(&vectors, None)
        .await
        .unwrap()
        .matrix;
    assert_eq!(m.len(), 3);
    assert_eq!(m[0].len(), 3);
    assert!((m[0][0] - 1.0).abs() < 1e-9, "self-similarity is 1");
    assert!((m[0][1] - 1.0).abs() < 1e-9, "parallel vectors: cosine 1");
    assert!(m[0][2].abs() < 1e-9, "orthogonal vectors: cosine 0");
    // Symmetric.
    assert!((m[0][1] - m[1][0]).abs() < 1e-9);
}

#[tokio::test]
async fn util_pairwise_distance_round_trip() {
    let (client, _server) = spawn_service().await;
    let vectors = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
    let m = client
        .util_pairwise_distance(&vectors, dynograph_client::DistanceMetric::Euclidean, None)
        .await
        .unwrap()
        .matrix;
    assert_eq!(m.len(), 2);
    assert!(m[0][0].abs() < 1e-9, "distance to self is 0");
    assert!((m[0][1] - 5.0).abs() < 1e-9, "3-4-5 triangle");
    assert!((m[1][0] - 5.0).abs() < 1e-9, "symmetric");
}

#[tokio::test]
async fn util_pairwise_distance_metric_is_honored() {
    let (client, _server) = spawn_service().await;
    let vectors = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
    // Manhattan distance of (3,4) from origin is 7, not the Euclidean 5.
    let m = client
        .util_pairwise_distance(&vectors, dynograph_client::DistanceMetric::Manhattan, None)
        .await
        .unwrap()
        .matrix;
    assert!((m[0][1] - 7.0).abs() < 1e-9, "metric selects manhattan");
}

#[tokio::test]
async fn search_hybrid_vector_only_round_trip() {
    // Vector-only hybrid needs no `fulltext` feature: a `query_vector` +
    // `node_type` request fuses just the vector leg.
    let (client, _server) = spawn_service().await;
    client.create_graph("g1", &tiny_schema()).await.unwrap();
    for id in ["a", "b", "c"] {
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

    let body = json!({
        "query_vector": [1.0, 0.0, 0.0],
        "node_type": "Item",
        "limit": 3,
    });
    let resp = client.search_hybrid("g1", &body).await.unwrap();
    let ids: Vec<&str> = resp.hits.iter().map(|h| h.node_id.as_str()).collect();
    assert_eq!(ids, vec!["a", "b", "c"], "ranked by vector similarity");
    // Single-leg fusion: every hit has a vector leg, no keyword leg.
    assert!(resp.hits[0].legs.vector.is_some());
    assert!(resp.hits[0].legs.keyword.is_none());
    assert_eq!(resp.hits[0].legs.vector.unwrap().rank, 1);
    assert_eq!(resp.hits[0].node_type, "Item");
}

// The in-process service is compiled WITHOUT `fulltext` in the default CI job
// and WITH it in the `--features dynograph-service/fulltext` job. The client
// crate has no compile-time signal for which, so these tests pin the wrapper
// contract — the server's status surfaces as a typed `ClientError::Http` — in a
// way that holds under both: 501 when the feature is off, and the feature-on
// outcome (a 400 from `tiny_schema`'s `Item` having no fulltext-searchable
// property, or a 200 for the bodyless reindex) otherwise.

#[tokio::test]
async fn search_text_surfaces_server_status() {
    let (client, _server) = spawn_service().await;
    client.create_graph("g1", &tiny_schema()).await.unwrap();
    let status = client
        .search_text("g1", "anything", Some("Item"), 10)
        .await
        .unwrap_err()
        .status();
    assert!(
        matches!(
            status,
            Some(reqwest::StatusCode::NOT_IMPLEMENTED | reqwest::StatusCode::BAD_REQUEST)
        ),
        "expected 501 (no fulltext) or 400 (fulltext, non-searchable schema), got {status:?}"
    );
}

#[tokio::test]
async fn search_reindex_surfaces_server_status() {
    let (client, _server) = spawn_service().await;
    client.create_graph("g1", &tiny_schema()).await.unwrap();
    // Feature off => 501 passthrough; feature on => reindex runs (Item has no
    // searchable property, so nothing is indexed) and returns Ok. Both are
    // valid wrapper outcomes; what must never happen is a transport/decode error.
    match client.search_reindex("g1").await {
        Ok(resp) => {
            let _ = resp.indexed;
        }
        Err(e) => assert_eq!(e.status(), Some(reqwest::StatusCode::NOT_IMPLEMENTED)),
    }
}

#[tokio::test]
async fn search_hybrid_keyword_leg_surfaces_server_status() {
    let (client, _server) = spawn_service().await;
    client.create_graph("g1", &tiny_schema()).await.unwrap();
    // A `query` leg engages the keyword path: 501 without fulltext, or 400 with
    // it (Item has no fulltext-searchable property). Either way the client
    // surfaces the server's status as a typed error.
    let body = json!({ "query": "anything", "node_type": "Item", "limit": 5 });
    let status = client
        .search_hybrid("g1", &body)
        .await
        .unwrap_err()
        .status();
    assert!(
        matches!(
            status,
            Some(reqwest::StatusCode::NOT_IMPLEMENTED | reqwest::StatusCode::BAD_REQUEST)
        ),
        "expected 501 (no fulltext) or 400 (fulltext, non-searchable schema), got {status:?}"
    );
}

// ---------------------------------------------------------------------------
// algo/* (v0.7.0). The in-process test service is built WITHOUT the `graph`
// feature, so every algo route returns 501 — asserted as the typed passthrough.
// The typed response shapes are pinned separately by deserializing
// representative payloads (so client/server wire drift fails here in CI). The
// batch dry_run path is NOT graph-gated and round-trips for real.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn algo_methods_pass_through_501_without_graph_feature() {
    let (client, _server) = spawn_service().await;
    client.create_graph("g1", &tiny_schema()).await.unwrap();
    let b = &json!({});
    let n = Some(reqwest::StatusCode::NOT_IMPLEMENTED);
    // Every algo route exists in the contract but 501s without `graph`.
    assert_eq!(
        client.algo_components("g1", b).await.unwrap_err().status(),
        n
    );
    assert_eq!(client.algo_scc("g1", b).await.unwrap_err().status(), n);
    assert_eq!(client.algo_degree("g1", b).await.unwrap_err().status(), n);
    assert_eq!(client.algo_pagerank("g1", b).await.unwrap_err().status(), n);
    assert_eq!(
        client.algo_eigenvector("g1", b).await.unwrap_err().status(),
        n
    );
    assert_eq!(
        client.algo_closeness("g1", b).await.unwrap_err().status(),
        n
    );
    assert_eq!(
        client.algo_betweenness("g1", b).await.unwrap_err().status(),
        n
    );
    assert_eq!(
        client
            .algo_personalized_pagerank("g1", b)
            .await
            .unwrap_err()
            .status(),
        n
    );
    assert_eq!(client.algo_cuts("g1", b).await.unwrap_err().status(), n);
    assert_eq!(client.algo_cycles("g1", b).await.unwrap_err().status(), n);
    assert_eq!(client.algo_toposort("g1", b).await.unwrap_err().status(), n);
    assert_eq!(
        client.algo_clustering("g1", b).await.unwrap_err().status(),
        n
    );
    assert_eq!(
        client
            .algo_shortest_path("g1", b)
            .await
            .unwrap_err()
            .status(),
        n
    );
    assert_eq!(client.algo_max_flow("g1", b).await.unwrap_err().status(), n);
    assert_eq!(
        client
            .algo_link_prediction("g1", b)
            .await
            .unwrap_err()
            .status(),
        n
    );
    assert_eq!(
        client.algo_communities("g1", b).await.unwrap_err().status(),
        n
    );
}

#[test]
fn algo_wire_types_deserialize_server_shapes() {
    use dynograph_client::{
        ClusteringResponse, CommunitiesResponse, ComponentsResponse, CutsResponse, CyclesResponse,
        LinkPredictionResponse, MaxFlowResponse, ScoresResponse, ShortestPathResponse,
        ToposortResponse,
    };
    // Payloads mirror the documented server response shapes; a renamed/retyped
    // client field would fail to deserialize here.
    let s: ScoresResponse =
        serde_json::from_value(json!({"scores": [{"node": "a", "score": 1.5}]})).unwrap();
    assert_eq!(s.scores[0].node, "a");
    assert_eq!(s.scores[0].score, 1.5);

    let c: ComponentsResponse =
        serde_json::from_value(json!({"count": 2, "components": [["a", "b"], ["c"]]})).unwrap();
    assert_eq!(c.count, 2);
    assert_eq!(c.components[1], vec!["c"]);

    let cuts: CutsResponse = serde_json::from_value(
        json!({"articulation_points": ["a"], "bridges": [{"a": "a", "b": "b"}]}),
    )
    .unwrap();
    assert_eq!(cuts.bridges[0].a, "a");
    assert_eq!(cuts.bridges[0].b, "b");

    let cy: CyclesResponse =
        serde_json::from_value(json!({"acyclic": false, "cycle": ["a", "b"]})).unwrap();
    assert!(!cy.acyclic);

    let topo: ToposortResponse =
        serde_json::from_value(json!({"acyclic": true, "order": ["a", "b"]})).unwrap();
    assert!(topo.acyclic);

    let sp: ShortestPathResponse =
        serde_json::from_value(json!({"found": true, "path": ["a", "b"], "distance": 2.0}))
            .unwrap();
    assert!(sp.found);
    assert_eq!(sp.distance, 2.0);

    let lp: LinkPredictionResponse =
        serde_json::from_value(json!({"links": [{"a": "a", "b": "c", "score": 0.5}]})).unwrap();
    assert_eq!(lp.links[0].score, 0.5);

    let cl: ClusteringResponse = serde_json::from_value(
        json!({"scores": [{"node": "a", "score": 1.0}], "transitivity": 0.5, "average_clustering": 0.4}),
    )
    .unwrap();
    assert_eq!(cl.transitivity, 0.5);
    assert_eq!(cl.average_clustering, 0.4);

    let mf: MaxFlowResponse = serde_json::from_value(
        json!({"max_flow": 3.0, "source_side": ["a"], "cut_edges": [{"from": "a", "to": "b"}]}),
    )
    .unwrap();
    assert_eq!(mf.max_flow, 3.0);
    assert_eq!(mf.cut_edges[0].from, "a");

    let com: CommunitiesResponse = serde_json::from_value(
        json!({"count": 2, "communities": [["a"], ["b"]], "modularity": 0.42}),
    )
    .unwrap();
    assert_eq!(com.count, 2);
    assert_eq!(com.modularity, 0.42);
}

#[tokio::test]
async fn batch_dry_run_round_trip_validates_without_committing() {
    let (client, _server) = spawn_service().await;
    client.create_graph("g1", &tiny_schema()).await.unwrap();
    client
        .create_node("g1", "Item", "x", &props(&[("name", json!("x"))]))
        .await
        .unwrap();

    // Valid: create y, then edge x->y (read-your-own-writes). Nothing persists.
    let body = json!({"ops": [
        {"op": "create_node", "node_type": "Item", "node_id": "y", "properties": {"name": "y"}},
        {"op": "create_edge", "edge_type": "Likes", "from_type": "Item", "from_id": "x", "to_type": "Item", "to_id": "y", "properties": {}},
    ]});
    let v = client.batch_dry_run("g1", &body).await.unwrap();
    assert!(v.valid, "{v:?}");
    assert_eq!(v.results.len(), 2);
    assert!(v.results.iter().all(|r| r.ok));
    assert_eq!(v.results[0].op, "create_node");
    let err = client.get_node("g1", "Item", "y").await.unwrap_err();
    assert_eq!(
        err.status(),
        Some(reqwest::StatusCode::NOT_FOUND),
        "y must not persist"
    );

    // Invalid: replace a missing node — valid:false, the failing op is reported.
    let bad = json!({"ops": [
        {"op": "replace_node", "node_type": "Item", "node_id": "nope", "properties": {}},
    ]});
    let v = client.batch_dry_run("g1", &bad).await.unwrap();
    assert!(!v.valid);
    assert!(!v.results[0].ok);
    assert!(v.results[0].error.is_some(), "{v:?}");
}
