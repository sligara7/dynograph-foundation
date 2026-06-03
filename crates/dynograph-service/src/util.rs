//! `POST /v1/util/*` — pure-math utility endpoints.
//!
//! Exposes the load-bearing Tier-3 functions from `dynograph-vector`
//! and `dynograph-resolution` over HTTP so consumers that can't embed
//! foundation as Rust crates (Python services like market_graph,
//! arbitrary HTTP callers) get the same canonical math implementations
//! as the in-process consumers. Avoids the alternative — every
//! consumer re-implementing cosine / Pearson / fuzzy strings in their
//! own language with subtle numerical drift.
//!
//! ## Endpoints
//!
//! | route                                  | input                                   | output                |
//! |----------------------------------------|------------------------------------------|-----------------------|
//! | `POST /v1/util/cosine_similarity`     | `{a:[..], b:[..], precision?}`           | `{result: f64}`       |
//! | `POST /v1/util/dot_product`           | `{a:[..], b:[..], precision?}`           | `{result: f64}`       |
//! | `POST /v1/util/euclidean_distance`    | `{a:[..], b:[..], precision?}`           | `{result: f64}`       |
//! | `POST /v1/util/l2_norm`               | `{v:[..], precision?}`                   | `{result: f64}`       |
//! | `POST /v1/util/hadamard`              | `{a:[..], b:[..], precision?}`           | `{result: [f64]}`     |
//! | `POST /v1/util/pearson_correlation`   | `{a:[..], b:[..]}` (f64 only)            | `{result: f64}`       |
//! | `POST /v1/util/linear_regression_slope` | `{points: [[x, y], ...]}` (f64 only)   | `{result: f64}`       |
//! | `POST /v1/util/jaro_winkler`          | `{a: "s1", b: "s2"}`                     | `{result: u32}` (0..=100) |
//! | `POST /v1/util/token_sort_ratio`      | `{a: "s1", b: "s2"}`                     | `{result: u32}` (0..=100) |
//!
//! ## Precision
//!
//! Vector ops accept an optional `precision` field
//! (`"f32" | "f64"`, default `"f64"`). JSON numbers always parse as
//! f64; for `"f32"` mode the slice is cast element-wise before
//! dispatching to the f32 implementation in `dynograph_vector`.
//! Statistics (`pearson_correlation`, `linear_regression_slope`)
//! are f64-only — the underlying fns have no f32 variant.
//!
//! ## Validation (all 400, pre-flight)
//!
//! - Binary-vector ops: `a` and `b` non-empty AND same length
//!   AND `<= MAX_VECTOR_LEN`.
//! - Unary-vector ops: `v` non-empty AND `<= MAX_VECTOR_LEN`.
//! - Pearson / linear regression: returns 400 with a descriptive
//!   message when the underlying fn yields `None` (insufficient
//!   samples, zero variance, mismatched lengths) — surfaces the
//!   loud-fail rather than the silent-fallback alternative.
//!
//! ## Hot-path / rate-limiting note
//!
//! Each endpoint's CPU is bounded by `MAX_VECTOR_LEN`
//! (`100_000`). Above that, callers should chunk client-side.
//! For very high-rate workloads, revisit
//! `td_http_concurrency_tuning.md` and add per-route limits — this
//! module deliberately defers rate-limiting to the middleware layer
//! once that TD lands.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use dynograph_resolution::{jaro_winkler, token_sort_ratio};
use dynograph_vector::{
    cosine_similarity, cosine_similarity_f64, dot_product, dot_product_f64, euclidean_distance,
    euclidean_distance_f64, hadamard, hadamard_f64, l2_norm, l2_norm_f64, linear_regression_slope,
    pearson_correlation, validate_similarity_vector,
};

use crate::registry::RegistryError;

/// Hard upper bound on a single vector length. Bounds per-request
/// CPU; chunk client-side above this. 100k is comfortably above any
/// realistic embedding (typical 384/768/1536/3072) and below what a
/// pathological client would send to wedge a worker.
pub(crate) const MAX_VECTOR_LEN: usize = 100_000;

/// Hard cap on string length for fuzzy-match endpoints. `jaro_winkler`
/// and `token_sort_ratio` are O(|a|·|b|); without this, an attacker
/// sending two ~1MB strings (axum's default JSON body limit is ~2MB)
/// would cost ~10^12 character comparisons per request — a CPU-DoS
/// vector. 64 KB is well above any natural-name-class input
/// (entity names, fragment text) and below what wedges a worker.
pub(crate) const MAX_STRING_LEN: usize = 64 * 1024;

/// Numeric precision for vector ops. Wire form is the lowercase
/// variant name (`"f32"` / `"f64"`). Public so `dynograph-client`
/// can re-export it and consumers can pass `Precision::F32` instead
/// of stringly-typed `"f32"`.
#[derive(Debug, Deserialize, Serialize, Default, Clone, Copy, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    F32,
    #[default]
    F64,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct BinaryVectorRequest {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    #[serde(default)]
    pub precision: Precision,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct UnaryVectorRequest {
    pub v: Vec<f64>,
    #[serde(default)]
    pub precision: Precision,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct TwoVectorF64Request {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct PointsRequest {
    pub points: Vec<(f64, f64)>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct BinaryStringRequest {
    pub a: String,
    pub b: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ScalarResponse<T> {
    pub result: T,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct VectorResponse {
    pub result: Vec<f64>,
}

// =========================================================================
// Validation
// =========================================================================

fn validate_binary(a: &[f64], b: &[f64]) -> Result<(), RegistryError> {
    if a.is_empty() || b.is_empty() {
        return Err(RegistryError::BadRequest(
            "a and b must be non-empty".to_string(),
        ));
    }
    if a.len() != b.len() {
        return Err(RegistryError::BadRequest(format!(
            "a and b must have the same length, got {} vs {}",
            a.len(),
            b.len()
        )));
    }
    if a.len() > MAX_VECTOR_LEN {
        return Err(RegistryError::BadRequest(format!(
            "vector length {} exceeds maximum {MAX_VECTOR_LEN}",
            a.len()
        )));
    }
    Ok(())
}

fn validate_unary(v: &[f64]) -> Result<(), RegistryError> {
    if v.is_empty() {
        return Err(RegistryError::BadRequest("v must be non-empty".to_string()));
    }
    if v.len() > MAX_VECTOR_LEN {
        return Err(RegistryError::BadRequest(format!(
            "vector length {} exceeds maximum {MAX_VECTOR_LEN}",
            v.len()
        )));
    }
    Ok(())
}

fn cast_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}

/// Reject an embedding that can't produce a meaningful cosine
/// similarity — a non-finite component (NaN/±∞) or zero magnitude.
/// Such a vector scores a silent 0.0 against everything (looks
/// "orthogonal," not "invalid"), so the ingest boundaries
/// (`set_embedding`, `similar`, `resolve-or-create`) screen here and
/// return 400 instead of admitting degenerate data into the index or
/// resolver. Empty-ness is left to each caller's own non-empty check so
/// the message stays specific.
pub(crate) fn validate_embedding_values(emb: &[f32]) -> Result<(), RegistryError> {
    validate_similarity_vector(emb)
        .map_err(|reason| RegistryError::BadRequest(format!("invalid embedding: {reason}")))
}

// =========================================================================
// Binary-vector endpoints
// =========================================================================

pub(crate) fn run_cosine_similarity(
    req: BinaryVectorRequest,
) -> Result<ScalarResponse<f64>, RegistryError> {
    validate_binary(&req.a, &req.b)?;
    let result = match req.precision {
        Precision::F32 => cosine_similarity(&cast_f32(&req.a), &cast_f32(&req.b)) as f64,
        Precision::F64 => cosine_similarity_f64(&req.a, &req.b),
    };
    Ok(ScalarResponse { result })
}

pub(crate) fn run_dot_product(
    req: BinaryVectorRequest,
) -> Result<ScalarResponse<f64>, RegistryError> {
    validate_binary(&req.a, &req.b)?;
    let result = match req.precision {
        Precision::F32 => dot_product(&cast_f32(&req.a), &cast_f32(&req.b)) as f64,
        Precision::F64 => dot_product_f64(&req.a, &req.b),
    };
    Ok(ScalarResponse { result })
}

pub(crate) fn run_euclidean_distance(
    req: BinaryVectorRequest,
) -> Result<ScalarResponse<f64>, RegistryError> {
    validate_binary(&req.a, &req.b)?;
    let result = match req.precision {
        Precision::F32 => euclidean_distance(&cast_f32(&req.a), &cast_f32(&req.b)) as f64,
        Precision::F64 => euclidean_distance_f64(&req.a, &req.b),
    };
    Ok(ScalarResponse { result })
}

pub(crate) fn run_hadamard(req: BinaryVectorRequest) -> Result<VectorResponse, RegistryError> {
    validate_binary(&req.a, &req.b)?;
    let result: Vec<f64> = match req.precision {
        Precision::F32 => hadamard(&cast_f32(&req.a), &cast_f32(&req.b))
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        Precision::F64 => hadamard_f64(&req.a, &req.b),
    };
    Ok(VectorResponse { result })
}

// =========================================================================
// Unary-vector endpoints
// =========================================================================

pub(crate) fn run_l2_norm(req: UnaryVectorRequest) -> Result<ScalarResponse<f64>, RegistryError> {
    validate_unary(&req.v)?;
    let result = match req.precision {
        Precision::F32 => l2_norm(&cast_f32(&req.v)) as f64,
        Precision::F64 => l2_norm_f64(&req.v),
    };
    Ok(ScalarResponse { result })
}

// =========================================================================
// Statistics (f64 only)
// =========================================================================

pub(crate) fn run_pearson_correlation(
    req: TwoVectorF64Request,
) -> Result<ScalarResponse<f64>, RegistryError> {
    // Shared length / non-empty / cap validation, then the
    // pearson-specific n>=3 floor. Final `None` arm catches the
    // remaining zero-variance case with a distinct message.
    validate_binary(&req.a, &req.b)?;
    if req.a.len() < 3 {
        return Err(RegistryError::BadRequest(format!(
            "pearson_correlation requires n >= 3, got {}",
            req.a.len()
        )));
    }
    match pearson_correlation(&req.a, &req.b) {
        Some(result) => Ok(ScalarResponse { result }),
        None => Err(RegistryError::BadRequest(
            "pearson_correlation undefined (likely zero-variance input)".to_string(),
        )),
    }
}

pub(crate) fn run_linear_regression_slope(
    req: PointsRequest,
) -> Result<ScalarResponse<f64>, RegistryError> {
    if req.points.len() < 2 {
        return Err(RegistryError::BadRequest(format!(
            "linear_regression_slope requires >= 2 points, got {}",
            req.points.len()
        )));
    }
    if req.points.len() > MAX_VECTOR_LEN {
        return Err(RegistryError::BadRequest(format!(
            "points length {} exceeds maximum {MAX_VECTOR_LEN}",
            req.points.len()
        )));
    }
    match linear_regression_slope(&req.points) {
        Some(result) => Ok(ScalarResponse { result }),
        None => Err(RegistryError::BadRequest(
            "linear_regression_slope undefined (likely zero variance in x)".to_string(),
        )),
    }
}

// =========================================================================
// String fuzzy-match endpoints
// =========================================================================

pub(crate) fn run_jaro_winkler(
    req: BinaryStringRequest,
) -> Result<ScalarResponse<u32>, RegistryError> {
    validate_strings(&req.a, &req.b)?;
    Ok(ScalarResponse {
        result: jaro_winkler(&req.a, &req.b),
    })
}

pub(crate) fn run_token_sort_ratio(
    req: BinaryStringRequest,
) -> Result<ScalarResponse<u32>, RegistryError> {
    validate_strings(&req.a, &req.b)?;
    Ok(ScalarResponse {
        result: token_sort_ratio(&req.a, &req.b),
    })
}

fn validate_strings(a: &str, b: &str) -> Result<(), RegistryError> {
    if a.len() > MAX_STRING_LEN || b.len() > MAX_STRING_LEN {
        return Err(RegistryError::BadRequest(format!(
            "string length exceeds maximum {MAX_STRING_LEN} bytes, got ({}, {})",
            a.len(),
            b.len()
        )));
    }
    Ok(())
}
