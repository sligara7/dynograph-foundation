//! End-to-end test of the Unix-domain-socket transport.
//!
//! Unlike the `oneshot`-based tests in `integration.rs` (which drive the
//! router as a tower `Service` and never touch a socket), this binds a
//! real `UnixListener` via the same `bind_uds` the binary uses, serves
//! the full `app()` router on it, and connects over an actual
//! `tokio::net::UnixStream`. It proves the new transport carries both a
//! bodyless probe and a JSON write/read round-trip — the same handler
//! stack the TCP listener serves.

use std::path::Path;
use std::sync::Arc;

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Request, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::json;
use tokio::net::UnixStream;

use dynograph_service::{AppState, GraphRegistry, app, bind_uds};

/// Open a fresh connection over the socket, send one request, and
/// return `(status, body_bytes)`. A new connection per call keeps the
/// test free of connection-reuse bookkeeping.
async fn send(
    sock: &Path,
    method: &str,
    path: &str,
    body: Option<Vec<u8>>,
) -> (StatusCode, Vec<u8>) {
    let stream = UnixStream::connect(sock).await.expect("connect to socket");
    let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
        .await
        .expect("http1 handshake");
    // The connection task must be driven for the request to make
    // progress; it ends when the response is fully read.
    tokio::spawn(async move {
        let _ = conn.await;
    });

    let req = Request::builder()
        .method(method)
        .uri(path)
        // HTTP/1.1 requires a Host header; the socket ignores its value.
        .header("host", "localhost")
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body.unwrap_or_default())))
        .expect("build request");

    let resp = sender.send_request(req).await.expect("send request");
    let status = resp.status();
    let bytes = resp
        .into_body()
        .collect()
        .await
        .expect("read body")
        .to_bytes();
    (status, bytes.to_vec())
}

#[tokio::test]
async fn serves_full_api_over_unix_socket() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("dyno.sock");

    let state = AppState::with_no_auth(Arc::new(GraphRegistry::new()));
    let listener = bind_uds(&sock).expect("bind uds");
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        axum::serve(listener, app(state))
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });

    // 1. Bodyless probe round-trips over the socket.
    let (status, body) = send(&sock, "GET", "/health", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, b"ok");

    // 2. A JSON write path: create a graph (the TCP suite asserts the
    //    same 201 for this route — proving parity across transports).
    let create = json!({
        "id": "g1",
        "schema": {
            "name": "demo",
            "version": 1,
            "node_types": {
                "Item": { "properties": { "name": { "type": "string", "required": true } } }
            },
            "edge_types": {}
        }
    });
    let (status, _) = send(
        &sock,
        "POST",
        "/v1/graphs",
        Some(create.to_string().into_bytes()),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    // 3. Read it back over the socket to confirm the body made it
    //    through the handler, not just the framing.
    let (status, body) = send(&sock, "GET", "/v1/graphs/g1", None).await;
    assert_eq!(status, StatusCode::OK);
    let meta: serde_json::Value = serde_json::from_slice(&body).expect("json response");
    assert_eq!(meta["id"], "g1");

    let _ = shutdown_tx.send(());
    server.await.unwrap();
}
