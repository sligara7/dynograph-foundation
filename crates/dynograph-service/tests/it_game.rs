//! Integration tests — game. Split out of `integration.rs`.

mod common;

use common::*;

#[tokio::test]
async fn game_prisoners_dilemma_is_nash_pareto_suboptimal() {
    let app = build_app();
    let (status, resp) = post_util(&app, "game/analyze", pd_body()).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    // Defect (1) strictly dominant for both.
    assert_eq!(resp["dominant"][0]["strictly_dominant"], 1);
    assert_eq!(resp["dominant"][1]["strictly_dominant"], 1);
    // Unique pure NE = (defect, defect).
    assert_eq!(resp["pure_nash"], json!([[1, 1]]));
    // The headline: rational defection is collectively worse than (C,C).
    assert_eq!(resp["nash_is_pareto_suboptimal"], true);
    assert_eq!(resp["nash_domination"][0]["nash"], json!([1, 1]));
    assert_eq!(resp["nash_domination"][0]["dominated_by"], json!([0, 0]));
    // No interior mixed NE under strict dominance.
    assert!(resp.get("mixed_2x2").is_none() || resp["mixed_2x2"].is_null());
}

#[tokio::test]
async fn game_matching_pennies_mixed_nash_one_half() {
    let app = build_app();
    let body = json!({
        "players": [{"strategies": ["heads", "tails"]}, {"strategies": ["heads", "tails"]}],
        "payoffs": [
            {"profile": [0, 0], "utilities": [1.0, -1.0]},
            {"profile": [0, 1], "utilities": [-1.0, 1.0]},
            {"profile": [1, 0], "utilities": [-1.0, 1.0]},
            {"profile": [1, 1], "utilities": [1.0, -1.0]}
        ]
    });
    let (status, resp) = post_util(&app, "game/analyze", body).await;
    assert_eq!(status, StatusCode::OK, "body: {resp}");
    assert_eq!(resp["pure_nash"], json!([]), "no pure NE");
    let p0 = resp["mixed_2x2"]["player0_probs"].as_array().unwrap();
    assert!((p0[0].as_f64().unwrap() - 0.5).abs() < 1e-9, "body: {resp}");
    let p1 = resp["mixed_2x2"]["player1_probs"].as_array().unwrap();
    assert!((p1[0].as_f64().unwrap() - 0.5).abs() < 1e-9, "body: {resp}");
}

#[tokio::test]
async fn game_incomplete_payoffs_fails_loud() {
    let app = build_app();
    // Drop the (1,1) cell — incomplete game.
    let mut body = pd_body();
    body["payoffs"].as_array_mut().unwrap().pop();
    let (status, resp) = post_util(&app, "game/analyze", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
}

#[tokio::test]
async fn game_empty_players_fails_loud() {
    let app = build_app();
    let body = json!({"players": [], "payoffs": []});
    let (status, resp) = post_util(&app, "game/analyze", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
}

#[tokio::test]
async fn game_oversized_fails_loud() {
    let app = build_app();
    // 13 players × 2 strategies = 8192 profiles > MAX_GAME_CELLS (4096).
    let players: Vec<Value> = (0..13).map(|_| json!({"strategies": ["a", "b"]})).collect();
    let body = json!({"players": players, "payoffs": []});
    let (status, resp) = post_util(&app, "game/analyze", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {resp}");
}
