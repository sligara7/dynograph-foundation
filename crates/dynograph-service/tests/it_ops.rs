//! Integration tests — ops. Split out of `integration.rs`.

mod common;

use common::*;

#[tokio::test]
async fn health_returns_ok() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&bytes[..], b"ok");
}

#[tokio::test]
async fn ready_returns_200_when_marked_ready() {
    // `with_no_auth` defaults to ready, matching slice 1–3 test
    // expectations.
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&bytes[..], b"ready");
}

#[tokio::test]
async fn ready_returns_503_before_mark_ready_then_flips() {
    let registry = Arc::new(GraphRegistry::new());
    let readiness = Arc::new(Readiness::not_ready());
    let state = AppState::new(registry, Arc::new(NoAuth::new()), readiness.clone());
    let app = app(state);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&bytes[..], b"starting");

    readiness.mark_ready();

    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn protected_route_without_token_returns_401() {
    let app = build_app_with_bearer_jwt();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body = std::str::from_utf8(&bytes).unwrap();
    assert!(body.contains("missing Authorization"), "{body}");
}

#[tokio::test]
async fn protected_route_with_valid_token_succeeds() {
    let app = build_app_with_bearer_jwt();
    let token = mint_jwt("alice");
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["graphs"], json!([]));
}

#[tokio::test]
async fn protected_route_with_garbage_token_returns_401() {
    let app = build_app_with_bearer_jwt();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs")
                .header("authorization", "Bearer not.a.real.jwt")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn health_is_public_under_bearer_jwt() {
    let app = build_app_with_bearer_jwt();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn ready_is_public_under_bearer_jwt() {
    let app = build_app_with_bearer_jwt();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn metrics_endpoint_emits_prometheus_text() {
    let app = build_app();
    let (status, body) = fetch_metrics(&app).await;
    assert_eq!(status, StatusCode::OK);
    // Static metrics are always present.
    assert!(body.contains("# TYPE dynograph_build_info gauge"), "{body}");
    assert!(body.contains("dynograph_build_info{version=\""), "{body}");
    assert!(body.contains("dynograph_uptime_seconds"), "{body}");
}

#[tokio::test]
async fn metrics_records_request_under_matched_route_label() {
    let app = build_app();
    // Drive a known route so we see its matched-path label.
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let (_, body) = fetch_metrics(&app).await;
    assert!(
        body.contains(
            r#"dynograph_http_requests_total{method="GET",path="/v1/graphs",status="200"} 1"#
        ),
        "{body}"
    );
    // And the latency-sum counter is present for the same series.
    assert!(
        body.contains(
            r#"dynograph_http_request_duration_microseconds_sum{method="GET",path="/v1/graphs",status="200"}"#
        ),
        "{body}"
    );
}

#[tokio::test]
async fn metrics_uses_route_pattern_not_literal_url() {
    // Two requests to different graph IDs should collapse into one
    // matched-path series — the cardinality protection that makes
    // /metrics safe to expose under arbitrary client traffic.
    let app = build_app();
    for id in ["alpha", "beta"] {
        let body = json!({
            "id": id,
            "schema": { "name": "demo", "version": 1, "node_types": {}, "edge_types": {} }
        });
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/graphs")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::CREATED);
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!("/v1/graphs/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }
    let (_, body) = fetch_metrics(&app).await;
    // One series for the matched pattern, count=2.
    assert!(
        body.contains(
            r#"dynograph_http_requests_total{method="GET",path="/v1/graphs/{id}",status="200"} 2"#
        ),
        "{body}"
    );
    // The literal IDs must not appear as labels.
    assert!(!body.contains("path=\"/v1/graphs/alpha\""), "{body}");
    assert!(!body.contains("path=\"/v1/graphs/beta\""), "{body}");
}

#[tokio::test]
async fn metrics_endpoint_is_not_self_recorded() {
    // Scrapes shouldn't show up in the request-counter series — that
    // would inflate cardinality and primarily measure Prometheus's
    // own scrape interval. Hit /metrics twice; the second body must
    // not contain a series for path="/metrics".
    let app = build_app();
    let _ = fetch_metrics(&app).await;
    let (_, body) = fetch_metrics(&app).await;
    assert!(!body.contains("path=\"/metrics\""), "{body}");
}

#[tokio::test]
async fn metrics_is_public_under_bearer_jwt() {
    // /metrics sits alongside /health and /ready — accessible
    // without a token even when BearerJwt is the configured
    // provider. Operators gate access at the network layer.
    let app = build_app_with_bearer_jwt();
    let (status, body) = fetch_metrics(&app).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("dynograph_build_info"), "{body}");
}

#[tokio::test]
async fn metrics_emits_hnsw_stats_after_an_embedding_set() {
    let app = build_app_with_item_graph().await;
    create_item(&app, "n1").await;
    put_embedding(&app, "Item", "n1", &[0.1, 0.2, 0.3]).await;
    let (status, body) = fetch_metrics(&app).await;
    assert_eq!(status, StatusCode::OK);
    // The Item index now has 1 live entry + 1 insert.
    assert!(
        body.contains(r#"dynograph_hnsw_index_size{graph="g1",node_type="Item"} 1"#),
        "{body}"
    );
    assert!(
        body.contains(r#"dynograph_hnsw_inserts_total{graph="g1",node_type="Item"} 1"#),
        "{body}"
    );
}

/// Verify the middleware exposes `Identity` to handlers via request
/// extensions. We don't change any handler this slice, but the
/// plumbing must already work for slice-11+ consumer-side
/// authorization. End-to-end: a successful POST exercises the
/// middleware (sets `Identity`); the handler runs (it doesn't read
/// `Identity`, but presence is implied by reaching the 201). If
/// the middleware *didn't* run, the request would never reach the
/// handler anyway because the route layer rejects unauthenticated
/// requests.
#[tokio::test]
async fn bearer_jwt_post_creates_graph_under_auth() {
    let app = build_app_with_bearer_jwt();
    let token = mint_jwt("alice");
    let body = json!({
        "id": "g1",
        "schema": { "name": "demo", "version": 1, "node_types": {}, "edge_types": {} }
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
}

#[tokio::test]
async fn oversized_body_returns_413() {
    // Tiny cap so a normal create-graph body trips it.
    let app = build_app_with_limits(ServerLimits {
        max_body_bytes: 50,
        ..Default::default()
    });
    let body = json!({
        "id": "g1",
        "schema": { "name": "demo", "version": 1, "node_types": {}, "edge_types": {} }
    });
    assert!(body.to_string().len() > 50);
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn v1_sheds_load_with_503_but_probes_stay_up() {
    // Zero permits → every /v1 request is shed immediately. Extreme
    // value chosen so the load-shed path is deterministic without
    // needing N concurrent in-flight requests.
    let app = build_app_with_limits(ServerLimits {
        max_concurrent_requests: 0,
        ..Default::default()
    });

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);

    // Probes are not behind the concurrency layer — liveness stays up.
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn openapi_json_is_served_and_covers_every_route() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/openapi.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let spec: Value = serde_json::from_slice(&bytes).expect("openapi.json is valid JSON");

    // Valid OpenAPI 3.x envelope with a version.
    assert!(
        spec["openapi"].as_str().unwrap().starts_with("3."),
        "openapi version: {}",
        spec["openapi"]
    );
    assert_eq!(spec["info"]["version"], env!("CARGO_PKG_VERSION"));

    // The served spec must expose exactly the path set of the committed
    // contract — no hand-maintained route list (the drift-gate unit test
    // locks committed == generated; this locks served == committed, so
    // the served endpoint can't silently diverge from the contract).
    let committed: Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/openapi.json"
        ))
        .expect("read docs/openapi.json"),
    )
    .expect("docs/openapi.json is valid JSON");

    let served_paths: std::collections::BTreeSet<&str> = spec["paths"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let committed_paths: std::collections::BTreeSet<&str> = committed["paths"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        served_paths, committed_paths,
        "served /openapi.json paths differ from committed docs/openapi.json"
    );
    // Sanity floor: the full surface is documented, not an accidental subset.
    assert!(
        served_paths.len() >= 30,
        "expected >= 30 paths, got {}",
        served_paths.len()
    );
}
