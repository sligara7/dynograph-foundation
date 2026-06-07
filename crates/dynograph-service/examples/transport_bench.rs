//! Transport A/B benchmark: TCP+JSON vs UDS+JSON.
//!
//! The same axum router, the same JSON payloads, and the same hyper
//! HTTP/1.1 client stack are driven over a TCP loopback socket and over
//! a Unix domain socket — the *only* variable is the transport. This is
//! the number that decides whether the Unix-socket transport (Layer 1)
//! is worth adopting, and whether serialization (Layer 2, msgpack) is
//! even the bottleneck worth chasing next.
//!
//! Release is mandatory — debug serde/hyper overhead would swamp the
//! signal:
//!
//!   cargo run --release -p dynograph-service --example transport_bench
//!
//! Scenarios:
//!
//! - `crud_get` — GET one node: tiny request, small response. The
//!   transport-bound case (connection + syscall cost dominate;
//!   serialization is negligible).
//! - `vector_post` — POST cosine_similarity with two 1536-dim f64
//!   vectors: ~50 KB JSON request. The payload-bound case (where binary
//!   serialization would later help).
//!
//! Each runs with a reused keep-alive connection (the realistic
//! pooled-client case) and, for `crud_get`, with a fresh connection per
//! call (isolates connection-setup cost — TCP handshake vs UDS connect).

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::client::conn::http1::SendRequest;
use hyper::{Request, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::json;
use tokio::net::{TcpStream, UnixStream};

use dynograph_service::{AppState, GraphRegistry, app, bind_uds};

const N: usize = 10_000;
const WARMUP: usize = 500;
const VECTOR_DIM: usize = 1536;

#[tokio::main]
async fn main() {
    let state = AppState::with_no_auth(Arc::new(GraphRegistry::new()));
    let router = app(state);

    // Bind both transports and serve the same router on each.
    let tcp_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = tcp_listener.local_addr().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let sock = tmp.path().join("bench.sock");
    let uds_listener = bind_uds(&sock).unwrap();
    {
        let r = router.clone();
        tokio::spawn(async move { axum::serve(tcp_listener, r).await.unwrap() });
    }
    {
        let r = router.clone();
        tokio::spawn(async move { axum::serve(uds_listener, r).await.unwrap() });
    }

    // Seed: one graph + one node so `crud_get` has something to read.
    seed(&mut tcp(addr).await).await;

    // Build the large vector body once; Bytes clones are cheap (Arc).
    let big: Vec<f64> = (0..VECTOR_DIM).map(|i| i as f64 * 0.001).collect();
    let vector_body = Bytes::from(json!({ "a": big, "b": big }).to_string());

    println!(
        "transport_bench — N={N} per scenario (warmup={WARMUP}), vector_dim={VECTOR_DIM}\n\
         release build required for meaningful numbers\n"
    );
    println!(
        "{:<16} {:<5} {:<12} {:>9} {:>9} {:>9} {:>11}",
        "scenario", "tx", "mode", "mean", "p50", "p99", "req/s"
    );
    println!("{}", "-".repeat(76));

    // crud_get — reused connection.
    let tcp_get = run_persistent(|| tcp(addr), get_node_req, StatusCode::OK).await;
    report("crud_get", "tcp", "keepalive", &tcp_get);
    let uds_get = run_persistent(|| uds(&sock), get_node_req, StatusCode::OK).await;
    report("crud_get", "uds", "keepalive", &uds_get);

    // crud_get — fresh connection per call (connection-setup cost).
    let tcp_get_cold = run_fresh(|| tcp(addr), get_node_req, StatusCode::OK).await;
    report("crud_get", "tcp", "fresh-conn", &tcp_get_cold);
    let uds_get_cold = run_fresh(|| uds(&sock), get_node_req, StatusCode::OK).await;
    report("crud_get", "uds", "fresh-conn", &uds_get_cold);

    // vector_post — reused connection (large request payload).
    let vb = vector_body.clone();
    let tcp_vec = run_persistent(
        || tcp(addr),
        move || cosine_req(vb.clone()),
        StatusCode::OK,
    )
    .await;
    report("vector_post", "tcp", "keepalive", &tcp_vec);
    let vb = vector_body.clone();
    let uds_vec = run_persistent(
        || uds(&sock),
        move || cosine_req(vb.clone()),
        StatusCode::OK,
    )
    .await;
    report("vector_post", "uds", "keepalive", &uds_vec);

    println!("\nUDS vs TCP (lower latency is better):");
    delta("crud_get  keepalive ", &tcp_get, &uds_get);
    delta("crud_get  fresh-conn", &tcp_get_cold, &uds_get_cold);
    delta("vector    keepalive ", &tcp_vec, &uds_vec);
}

// ---------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------

async fn tcp(addr: SocketAddr) -> SendRequest<Full<Bytes>> {
    open(TokioIo::new(TcpStream::connect(addr).await.unwrap())).await
}

async fn uds(path: &Path) -> SendRequest<Full<Bytes>> {
    open(TokioIo::new(UnixStream::connect(path).await.unwrap())).await
}

async fn open<IO>(io: IO) -> SendRequest<Full<Bytes>>
where
    IO: hyper::rt::Read + hyper::rt::Write + Send + Unpin + 'static,
{
    let (sender, conn) = hyper::client::conn::http1::handshake(io).await.unwrap();
    tokio::spawn(async move {
        let _ = conn.await;
    });
    sender
}

fn req(method: &str, path: &str, body: Bytes) -> Request<Full<Bytes>> {
    Request::builder()
        .method(method)
        .uri(path)
        .header("host", "localhost")
        .header("content-type", "application/json")
        .body(Full::new(body))
        .unwrap()
}

fn get_node_req() -> Request<Full<Bytes>> {
    req("GET", "/v1/graphs/bench/nodes/Item/n1", Bytes::new())
}

fn cosine_req(body: Bytes) -> Request<Full<Bytes>> {
    req("POST", "/v1/util/cosine_similarity", body)
}

async fn call(sender: &mut SendRequest<Full<Bytes>>, r: Request<Full<Bytes>>, want: StatusCode) {
    // The spawned connection task drives readiness; wait for it before
    // dispatching (also the correct gate between keep-alive requests).
    sender.ready().await.unwrap();
    let resp = sender.send_request(r).await.unwrap();
    assert_eq!(resp.status(), want, "unexpected status");
    // Drain the body — required before the next request on a keep-alive
    // connection, and it's part of the per-call cost we're measuring.
    let _ = resp.into_body().collect().await.unwrap().to_bytes();
}

async fn seed(sender: &mut SendRequest<Full<Bytes>>) {
    let schema = json!({
        "id": "bench",
        "schema": {
            "name": "bench",
            "version": 1,
            "node_types": {
                "Item": { "properties": { "name": { "type": "string", "required": true } } }
            },
            "edge_types": {}
        }
    });
    call(
        sender,
        req("POST", "/v1/graphs", Bytes::from(schema.to_string())),
        StatusCode::CREATED,
    )
    .await;
    let node = json!({ "node_type": "Item", "node_id": "n1", "properties": { "name": "first" } });
    call(
        sender,
        req("POST", "/v1/graphs/bench/nodes", Bytes::from(node.to_string())),
        StatusCode::CREATED,
    )
    .await;
}

// ---------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------

/// Reuse one keep-alive connection across all calls.
async fn run_persistent<C, Fut, R>(connect: C, mut make: R, want: StatusCode) -> Stats
where
    C: Fn() -> Fut,
    Fut: std::future::Future<Output = SendRequest<Full<Bytes>>>,
    R: FnMut() -> Request<Full<Bytes>>,
{
    let mut sender = connect().await;
    for _ in 0..WARMUP {
        call(&mut sender, make(), want).await;
    }
    let mut samples = Vec::with_capacity(N);
    for _ in 0..N {
        let t = Instant::now();
        call(&mut sender, make(), want).await;
        samples.push(t.elapsed());
    }
    Stats::from(samples)
}

/// Open a fresh connection for every call (includes connect+handshake).
async fn run_fresh<C, Fut, R>(connect: C, mut make: R, want: StatusCode) -> Stats
where
    C: Fn() -> Fut,
    Fut: std::future::Future<Output = SendRequest<Full<Bytes>>>,
    R: FnMut() -> Request<Full<Bytes>>,
{
    for _ in 0..WARMUP {
        let mut s = connect().await;
        call(&mut s, make(), want).await;
    }
    let mut samples = Vec::with_capacity(N);
    for _ in 0..N {
        let t = Instant::now();
        let mut s = connect().await;
        call(&mut s, make(), want).await;
        samples.push(t.elapsed());
    }
    Stats::from(samples)
}

struct Stats {
    mean: Duration,
    p50: Duration,
    p99: Duration,
    total: Duration,
}

impl Stats {
    fn from(mut samples: Vec<Duration>) -> Self {
        let total: Duration = samples.iter().sum();
        let mean = total / samples.len() as u32;
        samples.sort_unstable();
        let p50 = samples[samples.len() / 2];
        let p99 = samples[samples.len() * 99 / 100];
        Stats {
            mean,
            p50,
            p99,
            total,
        }
    }

    fn req_per_sec(&self) -> f64 {
        N as f64 / self.total.as_secs_f64()
    }
}

fn report(scenario: &str, tx: &str, mode: &str, s: &Stats) {
    println!(
        "{:<16} {:<5} {:<12} {:>9} {:>9} {:>9} {:>11.0}",
        scenario,
        tx,
        mode,
        us(s.mean),
        us(s.p50),
        us(s.p99),
        s.req_per_sec(),
    );
}

fn delta(label: &str, tcp: &Stats, uds: &Stats) {
    let t = tcp.mean.as_secs_f64();
    let u = uds.mean.as_secs_f64();
    let faster = (t - u) / t * 100.0;
    println!(
        "  {label}: mean {:>8} → {:>8}  ({faster:+.1}% latency, {:.2}x req/s)",
        us(tcp.mean),
        us(uds.mean),
        uds.req_per_sec() / tcp.req_per_sec(),
    );
}

fn us(d: Duration) -> String {
    format!("{:.1}us", d.as_secs_f64() * 1e6)
}
