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

// ---------------------------------------------------------------------------
// DBSCAN clustering (`POST /v1/util/dbscan`). Stateless pure math over a
// precomputed distance matrix — no graph needed.
// ---------------------------------------------------------------------------

/// Full euclidean distance matrix over 2-D points (as the caller would supply,
/// e.g. via `util/pairwise_distance`).
fn euclidean_matrix(points: &[(f64, f64)]) -> Value {
    let rows: Vec<Vec<f64>> = points
        .iter()
        .map(|&(ax, ay)| {
            points
                .iter()
                .map(|&(bx, by)| ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt())
                .collect()
        })
        .collect();
    json!(rows)
}

#[tokio::test]
async fn dbscan_two_blobs_plus_noise() {
    let app = build_app();
    // Two tight blobs far apart, plus a lone outlier.
    let m = euclidean_matrix(&[
        (0.0, 0.0),
        (0.1, 0.0),
        (0.0, 0.1),
        (10.0, 10.0),
        (10.1, 10.0),
        (10.0, 10.1),
        (100.0, 100.0),
    ]);
    let body = json!({"distance_matrix": m, "eps": 0.5, "min_points": 2});
    let (status, resp) = post_util(&app, "dbscan", body).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    let labels = resp["labels"].as_array().unwrap();
    assert_eq!(resp["num_clusters"], 2, "body: {resp}");
    // Blob A shares a label, blob B shares another, they differ, outlier noise.
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_ne!(labels[0], labels[3]);
    assert_eq!(labels[6], -1);
}

#[tokio::test]
async fn dbscan_eps_too_small_is_all_noise() {
    let app = build_app();
    let m = euclidean_matrix(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
    let body = json!({"distance_matrix": m, "eps": 0.5, "min_points": 2});
    let (status, resp) = post_util(&app, "dbscan", body).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["num_clusters"], 0);
    assert!(resp["labels"].as_array().unwrap().iter().all(|l| l == -1));
}

#[tokio::test]
async fn dbscan_rejects_non_square_matrix() {
    let app = build_app();
    let body = json!({"distance_matrix": [[0.0, 1.0], [1.0]], "eps": 1.0, "min_points": 2});
    let (status, resp) = post_util(&app, "dbscan", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
}

#[tokio::test]
async fn dbscan_rejects_empty_and_bad_params() {
    let app = build_app();
    // Empty matrix.
    let (status, _) =
        post_util(&app, "dbscan", json!({"distance_matrix": [], "eps": 1.0, "min_points": 2})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    // Negative eps.
    let (status, _) = post_util(
        &app,
        "dbscan",
        json!({"distance_matrix": [[0.0, 1.0], [1.0, 0.0]], "eps": -1.0, "min_points": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    // Zero min_points.
    let (status, _) = post_util(
        &app,
        "dbscan",
        json!({"distance_matrix": [[0.0, 1.0], [1.0, 0.0]], "eps": 1.0, "min_points": 0}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}
