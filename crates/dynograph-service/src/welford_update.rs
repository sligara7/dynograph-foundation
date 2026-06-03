//! POST /v1/graphs/{id}/edges/{type}/{from}/{to}/welford_update
//!
//! Server-side atomic Welford-style EMA update of a "score" property
//! family on an existing edge. Reads the edge's current
//! `(score, score_m2, score_stddev, score_min, score_max, score_count)`
//! sextuple, folds in the supplied observation under a fixed-α EMA
//! update + Welford running-variance increment, and writes the new
//! values back atomically under a single write lock.
//!
//! Surfaced by market_graph's edge-strength-learning project: causal
//! edges (CAUSES / AFFECTS / PROPAGATES_TO / CORRELATES_WITH) carry a
//! confidence score updated when new evidence arrives. Today that's a
//! client-side read-modify-write — race-safe only if done inside a
//! single `/batch` call, verbose, and easy to get wrong. After this
//! lands, it's one HTTP call per observation.
//!
//! ## Wire shape
//!
//! ```json
//! POST /v1/graphs/{id}/edges/{type}/{from}/{to}/welford_update
//! {"observation": 0.7, "alpha": 0.05}
//! → {
//!   "score":       0.6200,
//!   "score_m2":    0.0410,
//!   "score_stddev":0.2120,
//!   "score_min":   0.1000,
//!   "score_max":   0.9000,
//!   "score_count": 14
//! }
//! ```
//!
//! ## Math
//!
//! Hybrid: fixed-α EMA for the running estimate (`score`) plus
//! Welford-style m2 accumulation for variance. On observation `x`
//! with coefficient `α`:
//!
//! ```text
//! count_new  = count + 1
//! delta      = x - score_old              // pre-update residual
//! score_new  = score_old + α · delta      // EMA update
//! delta2     = x - score_new              // post-update residual
//! m2_new     = m2 + delta · delta2        // Welford increment
//! stddev_new = sqrt(m2_new / count_new)
//! min_new    = min(min, x)
//! max_new    = max(max, x)
//! ```
//!
//! First-observation case (no prior `score_count` on the edge): all
//! six properties are initialized — `score = x, m2 = 0, stddev = 0,
//! min = max = x, count = 1`.
//!
//! ## Pre-flight validation (400)
//!
//! - `observation` is finite (not NaN / ±∞)
//! - `alpha` is finite AND in the open interval `(0, 1)` — 0 means
//!   the EMA never moves, 1 means no smoothing (just store the latest
//!   value). Both degenerate; reject loudly so callers don't ship
//!   misconfigured pipelines.
//! - Edge exists (else 404)
//!
//! ## Atomicity
//!
//! Whole read-modify-write runs under one `with_engine_write` lock:
//! concurrent updates serialize, and a crash mid-write at the storage
//! layer leaves the edge either pre-update or fully post-update (no
//! partial-property tear). The merge is done via
//! `engine.merge_edge_properties`, which preserves any non-Welford
//! properties already on the edge.
//!
//! ## Property names
//!
//! The six property names are **conventions**, not declared on
//! `edge_type` (foundation doesn't have an edge-property schema yet).
//! Consumers that want different prefixes can add an `score_prop`
//! override in a later iteration; v1 hardcodes the convention to
//! match the market_graph design memo.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use dynograph_core::Value;
use dynograph_storage::StorageEngine;

use crate::registry::RegistryError;

pub(crate) const PROP_SCORE: &str = "score";
pub(crate) const PROP_M2: &str = "score_m2";
pub(crate) const PROP_STDDEV: &str = "score_stddev";
pub(crate) const PROP_MIN: &str = "score_min";
pub(crate) const PROP_MAX: &str = "score_max";
pub(crate) const PROP_COUNT: &str = "score_count";

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct WelfordUpdateRequest {
    pub observation: f64,
    pub alpha: f64,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct WelfordUpdateResponse {
    pub score: f64,
    pub score_m2: f64,
    pub score_stddev: f64,
    pub score_min: f64,
    pub score_max: f64,
    pub score_count: i64,
}

pub(crate) fn run(
    engine: &mut StorageEngine,
    graph_id: &str,
    edge_type: &str,
    from_id: &str,
    to_id: &str,
    req: WelfordUpdateRequest,
) -> Result<WelfordUpdateResponse, RegistryError> {
    if !req.observation.is_finite() {
        return Err(RegistryError::BadRequest(
            "observation must be finite (not NaN or ±∞)".to_string(),
        ));
    }
    if !req.alpha.is_finite() || req.alpha <= 0.0 || req.alpha >= 1.0 {
        return Err(RegistryError::BadRequest(format!(
            "alpha must be finite and in (0, 1), got {}",
            req.alpha
        )));
    }

    let existing = engine
        .get_edge(graph_id, edge_type, from_id, to_id)?
        .ok_or_else(|| RegistryError::EdgeNotFound {
            edge_type: edge_type.to_string(),
            from_id: from_id.to_string(),
            to_id: to_id.to_string(),
        })?;

    let new_state = compute_update(&existing.properties, req.observation, req.alpha);

    // Merge just the six Welford props — preserves any non-Welford
    // properties already on the edge (e.g. `evidence_url`,
    // `last_updated_at` that a consumer may have set alongside score
    // family).
    let mut updates: HashMap<String, Value> = HashMap::with_capacity(6);
    updates.insert(PROP_SCORE.to_string(), Value::Float(new_state.score));
    updates.insert(PROP_M2.to_string(), Value::Float(new_state.score_m2));
    updates.insert(
        PROP_STDDEV.to_string(),
        Value::Float(new_state.score_stddev),
    );
    updates.insert(PROP_MIN.to_string(), Value::Float(new_state.score_min));
    updates.insert(PROP_MAX.to_string(), Value::Float(new_state.score_max));
    updates.insert(PROP_COUNT.to_string(), Value::Int(new_state.score_count));

    let merged = engine
        .merge_edge_properties(graph_id, edge_type, from_id, to_id, updates)?
        .ok_or_else(|| RegistryError::EdgeNotFound {
            edge_type: edge_type.to_string(),
            from_id: from_id.to_string(),
            to_id: to_id.to_string(),
        })?;

    // Re-read the new state from the merged result so the response
    // reflects what's actually persisted (defensive: if a future
    // merge_edge_properties acquires extra coercion it stays
    // visible). Falls back to the computed values if a prop went
    // missing somehow (unreachable: we just wrote all six).
    Ok(WelfordUpdateResponse {
        score: prop_f64(&merged.properties, PROP_SCORE).unwrap_or(new_state.score),
        score_m2: prop_f64(&merged.properties, PROP_M2).unwrap_or(new_state.score_m2),
        score_stddev: prop_f64(&merged.properties, PROP_STDDEV).unwrap_or(new_state.score_stddev),
        score_min: prop_f64(&merged.properties, PROP_MIN).unwrap_or(new_state.score_min),
        score_max: prop_f64(&merged.properties, PROP_MAX).unwrap_or(new_state.score_max),
        score_count: prop_i64(&merged.properties, PROP_COUNT).unwrap_or(new_state.score_count),
    })
}

struct WelfordState {
    score: f64,
    score_m2: f64,
    score_stddev: f64,
    score_min: f64,
    score_max: f64,
    score_count: i64,
}

/// Compute the post-observation state from the edge's current
/// property map. Properties absent from the map seed first-observation
/// defaults. Pure function — no I/O, no engine access — so it's
/// trivially testable as a unit and the run() handler stays focused on
/// the storage transaction.
fn compute_update(current: &HashMap<String, Value>, observation: f64, alpha: f64) -> WelfordState {
    let count_old = prop_i64(current, PROP_COUNT).unwrap_or(0);
    if count_old <= 0 {
        return WelfordState {
            score: observation,
            score_m2: 0.0,
            score_stddev: 0.0,
            score_min: observation,
            score_max: observation,
            score_count: 1,
        };
    }

    let score_old = prop_f64(current, PROP_SCORE).unwrap_or(observation);
    let m2_old = prop_f64(current, PROP_M2).unwrap_or(0.0);
    let min_old = prop_f64(current, PROP_MIN).unwrap_or(observation);
    let max_old = prop_f64(current, PROP_MAX).unwrap_or(observation);

    let count_new = count_old + 1;
    let delta = observation - score_old;
    let score_new = score_old + alpha * delta;
    let delta2 = observation - score_new;
    let m2_new = m2_old + delta * delta2;
    let variance_new = m2_new / count_new as f64;
    let stddev_new = if variance_new > 0.0 {
        variance_new.sqrt()
    } else {
        0.0
    };

    WelfordState {
        score: score_new,
        score_m2: m2_new,
        score_stddev: stddev_new,
        score_min: observation.min(min_old),
        score_max: observation.max(max_old),
        score_count: count_new,
    }
}

fn prop_f64(map: &HashMap<String, Value>, key: &str) -> Option<f64> {
    map.get(key).and_then(|v| v.as_f64())
}

fn prop_i64(map: &HashMap<String, Value>, key: &str) -> Option<i64> {
    map.get(key).and_then(|v| v.as_i64())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn first_observation_initializes_state() {
        let s = compute_update(&HashMap::new(), 0.7, 0.05);
        assert_eq!(s.score, 0.7);
        assert_eq!(s.score_m2, 0.0);
        assert_eq!(s.score_stddev, 0.0);
        assert_eq!(s.score_min, 0.7);
        assert_eq!(s.score_max, 0.7);
        assert_eq!(s.score_count, 1);
    }

    #[test]
    fn second_observation_ema_and_welford() {
        // Edge already has count=1, score=0.5, m2=0, min=0.5, max=0.5.
        // Observe 0.7 with α=0.5. Expectations:
        //   delta       = 0.7 - 0.5     = 0.2
        //   score_new   = 0.5 + 0.5·0.2 = 0.6
        //   delta2      = 0.7 - 0.6     = 0.1
        //   m2_new      = 0 + 0.2·0.1   = 0.02
        //   variance    = 0.02 / 2      = 0.01
        //   stddev      = 0.1
        let current = props(&[
            (PROP_SCORE, Value::Float(0.5)),
            (PROP_M2, Value::Float(0.0)),
            (PROP_MIN, Value::Float(0.5)),
            (PROP_MAX, Value::Float(0.5)),
            (PROP_COUNT, Value::Int(1)),
        ]);
        let s = compute_update(&current, 0.7, 0.5);
        approx(s.score, 0.6);
        approx(s.score_m2, 0.02);
        approx(s.score_stddev, 0.1);
        approx(s.score_min, 0.5);
        approx(s.score_max, 0.7);
        assert_eq!(s.score_count, 2);
    }

    #[test]
    fn min_max_track_extrema_under_repeated_updates() {
        let mut current = HashMap::new();
        let mut state = compute_update(&current, 0.5, 0.1);
        current = stored(state);
        state = compute_update(&current, 0.9, 0.1);
        current = stored(state);
        state = compute_update(&current, 0.1, 0.1);
        approx(state.score_min, 0.1);
        approx(state.score_max, 0.9);
        assert_eq!(state.score_count, 3);
    }

    fn approx(got: f64, want: f64) {
        let diff = (got - want).abs();
        assert!(
            diff < 1e-9,
            "approx fail: got {got}, want {want}, diff {diff}"
        );
    }

    fn stored(s: WelfordState) -> HashMap<String, Value> {
        props(&[
            (PROP_SCORE, Value::Float(s.score)),
            (PROP_M2, Value::Float(s.score_m2)),
            (PROP_STDDEV, Value::Float(s.score_stddev)),
            (PROP_MIN, Value::Float(s.score_min)),
            (PROP_MAX, Value::Float(s.score_max)),
            (PROP_COUNT, Value::Int(s.score_count)),
        ])
    }
}
