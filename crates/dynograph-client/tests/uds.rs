//! End-to-end test of the client's Unix-socket transport.
//!
//! Mirrors `integration.rs` but binds the in-process service on a Unix
//! socket (via the same `bind_uds` the binary uses) and drives it through
//! [`DynographClient::connect_unix`]. Proves the UDS transport carries
//! the full request/response cycle — text probe, JSON create/read, a
//! query-string GET, and the non-2xx error path — identically to TCP.

use std::sync::Arc;

use dynograph_client::{CreateEdge, DynographClient};
use dynograph_core::Schema;
use dynograph_service::{AppState, GraphRegistry, NoAuth, Readiness, app, bind_uds};
use serde_json::{Map, Value, json};

/// Spin up a fresh in-memory service on a Unix socket and return a
/// client pointed at it. The `TempDir` and `JoinHandle` are returned so
/// the caller keeps them alive for the test's duration (dropping the dir
/// removes the socket; dropping the handle stops the server).
async fn spawn_uds() -> (
    DynographClient,
    tempfile::TempDir,
    tokio::task::JoinHandle<()>,
) {
    let state = AppState::new(
        Arc::new(GraphRegistry::new()),
        Arc::new(NoAuth::new()),
        Arc::new(Readiness::ready()),
    );
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("dyno.sock");
    let listener = bind_uds(&sock).unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app(state)).await.unwrap();
    });
    let client = DynographClient::connect_unix(&sock);
    (client, dir, server)
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
      properties: {}
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
async fn uds_round_trip_create_read_query_delete() {
    let (client, _dir, _server) = spawn_uds().await;

    // Text probe (bodyless GET → text response) over the socket.
    assert_eq!(client.health().await.unwrap(), "ok");

    // JSON create (POST with body → JSON response).
    client.create_graph("g", &tiny_schema()).await.unwrap();
    let created = client
        .create_node(
            "g",
            "Item",
            "n1",
            &props(&[("name", json!("first")), ("tag", json!("x"))]),
        )
        .await
        .unwrap();
    assert_eq!(created.node_id, "n1");

    // Bodyless GET → JSON response.
    let got = client.get_node("g", "Item", "n1").await.unwrap();
    assert_eq!(got.node_type, "Item");

    // GET with a query string — exercises `path_and_query` over UDS.
    let listed = client
        .list_nodes("g", "Item", Some(("tag", "x")))
        .await
        .unwrap();
    assert_eq!(listed.nodes.len(), 1);
    assert_eq!(listed.nodes[0].node_id, "n1");

    // A second node + an edge between them (more POST bodies).
    client
        .create_node("g", "Item", "n2", &props(&[("name", json!("second"))]))
        .await
        .unwrap();
    let edge = CreateEdge {
        edge_type: "Likes",
        from_type: "Item",
        from_id: "n1",
        to_type: "Item",
        to_id: "n2",
        properties: &Map::new(),
    };
    let e = client.create_edge("g", &edge).await.unwrap();
    assert_eq!(e.from_id, "n1");

    // DELETE (204 → unit) then confirm the non-2xx error path maps to a
    // typed 404 over UDS.
    client.delete_node("g", "Item", "n1").await.unwrap();
    let err = client.get_node("g", "Item", "n1").await.unwrap_err();
    assert_eq!(err.status().map(|s| s.as_u16()), Some(404));
}
