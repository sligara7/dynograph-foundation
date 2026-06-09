//! Integration tests — util. Split out of `integration.rs`.

mod common;

use common::*;

#[tokio::test]
async fn util_pairwise_cosine_matrix() {
    let app = build_app();
    let (status, resp) = post_util(
        &app,
        "pairwise_cosine",
        json!({"vectors": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let m = resp["matrix"].as_array().unwrap();
    assert_eq!(m.len(), 3);
    // Diagonal is 1 (self-similarity); orthogonal pairs are 0; v0==v2 => 1.
    assert!((m[0][0].as_f64().unwrap() - 1.0).abs() < 1e-9);
    assert!((m[0][1].as_f64().unwrap()).abs() < 1e-9);
    assert!((m[0][2].as_f64().unwrap() - 1.0).abs() < 1e-9);
    // Symmetric.
    assert_eq!(m[1][0], m[0][1]);
}

#[tokio::test]
async fn util_pairwise_distance_euclidean() {
    let app = build_app();
    let (status, resp) = post_util(
        &app,
        "pairwise_distance",
        json!({"vectors": [[0.0, 0.0], [3.0, 4.0]], "metric": "euclidean"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let m = resp["matrix"].as_array().unwrap();
    assert_eq!(m[0][0], 0.0);
    assert!(
        (m[0][1].as_f64().unwrap() - 5.0).abs() < 1e-9,
        "body: {resp}"
    );
    assert_eq!(m[1][0], m[0][1]); // symmetric
}

#[tokio::test]
async fn util_pairwise_rejects_ragged_and_empty() {
    let app = build_app();
    let (status, _) = post_util(
        &app,
        "pairwise_cosine",
        json!({"vectors": [[1.0, 2.0], [1.0]]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = post_util(&app, "pairwise_cosine", json!({"vectors": []})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}
