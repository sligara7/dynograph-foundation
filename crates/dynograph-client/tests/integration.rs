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
