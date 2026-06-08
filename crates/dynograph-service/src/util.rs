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
//! | `POST /v1/util/squared_euclidean_distance` | `{a:[..], b:[..], precision?}`     | `{result: f64}`       |
//! | `POST /v1/util/manhattan_distance`    | `{a:[..], b:[..], precision?}`           | `{result: f64}`       |
//! | `POST /v1/util/add`                   | `{a:[..], b:[..], precision?}`           | `{result: [f64]}`     |
//! | `POST /v1/util/subtract`              | `{a:[..], b:[..], precision?}`           | `{result: [f64]}`     |
//! | `POST /v1/util/scale`                 | `{v:[..], c, precision?}`                | `{result: [f64]}`     |
//! | `POST /v1/util/negate`                | `{v:[..], precision?}`                   | `{result: [f64]}`     |
//! | `POST /v1/util/hadamard_division`     | `{a:[..], b:[..], precision?}`           | `{result: [f64]}` (400 on zero divisor) |
//! | `POST /v1/util/elementwise_power`     | `{v:[..], p, precision?}`                | `{result: [f64]}`     |
//! | `POST /v1/util/l2_normalize`          | `{v:[..], precision?}`                   | `{result: [f64]}` (400 on zero magnitude) |
//! | `POST /v1/util/centroid`              | `{vectors: [[..], ..], precision?}`      | `{result: [f64]}`     |
//! | `POST /v1/util/mean`                  | `{values:[..]}` (f64 only)               | `{result: f64}`       |
//! | `POST /v1/util/variance`              | `{values:[..]}` (f64 only, n>=2)         | `{result: f64}`       |
//! | `POST /v1/util/std_dev`               | `{values:[..]}` (f64 only, n>=2)         | `{result: f64}`       |
//! | `POST /v1/util/median`                | `{values:[..]}` (f64 only)               | `{result: f64}`       |
//! | `POST /v1/util/percentile`            | `{values:[..], p}` (f64 only, 0<=p<=100) | `{result: f64}`       |
//! | `POST /v1/util/softmax`               | `{values:[..]}` (f64 only)               | `{result: [f64]}`     |
//! | `POST /v1/util/spearman_correlation`  | `{a:[..], b:[..]}` (f64 only, n>=3)      | `{result: f64}`       |
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
    add, add_f64, centroid, centroid_f64, cosine_similarity, cosine_similarity_f64, dot_product,
    dot_product_f64, elementwise_power, elementwise_power_f64, euclidean_distance,
    euclidean_distance_f64, hadamard, hadamard_division, hadamard_division_f64, hadamard_f64,
    l2_norm, l2_norm_f64, l2_normalize, l2_normalize_f64, linear_regression_slope,
    manhattan_distance, manhattan_distance_f64, mean, median, negate, negate_f64,
    pearson_correlation, percentile, scale, scale_f64, softmax, spearman_rank_correlation,
    squared_euclidean_distance, squared_euclidean_distance_f64, std_dev, subtract, subtract_f64,
    validate_similarity_vector, variance,
};

use crate::registry::RegistryError;

/// Hard upper bound on a single vector length. Bounds per-request
/// CPU; chunk client-side above this. 100k is comfortably above any
/// realistic embedding (typical 384/768/1536/3072) and below what a
/// pathological client would send to wedge a worker.
pub(crate) const MAX_VECTOR_LEN: usize = 100_000;

/// Hard cap on the number of vectors in a pairwise (matrix-producing) request.
/// The output is an N×N matrix, so this bounds both the O(N²·dim) compute and
/// the response size; the body-size limit already bounds N·dim on input. 1000
/// vectors (a 10^6-entry matrix) is the point of these endpoints — one call
/// instead of a million per-pair round-trips — without an unbounded response.
pub(crate) const MAX_PAIRWISE_VECTORS: usize = 1000;

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
    let result = match req.precision {
        Precision::F32 => widen(hadamard(&cast_f32(&req.a), &cast_f32(&req.b))),
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

// =========================================================================
// v0.6.x additions — more distances, element-wise algebra, descriptive
// stats. Same conventions as above: vector ops honor `precision`
// (f32 cast then dispatch); stats are f64-only; a `None` from the
// underlying fn becomes a 400 (loud-fail) rather than a silent default.
// JSON has no NaN/Infinity literals, so inputs over the wire are always
// finite — no extra finiteness screen is needed here.
// =========================================================================

/// A set of equal-length vectors (for `centroid` and `pairwise_cosine`).
#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct VectorsRequest {
    pub vectors: Vec<Vec<f64>>,
    #[serde(default)]
    pub precision: Precision,
}

/// Distance metric for `pairwise_distance`.
#[derive(Debug, Deserialize, Default, Clone, Copy, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DistanceMetric {
    #[default]
    Euclidean,
    SquaredEuclidean,
    Manhattan,
    /// Cosine distance, `1 - cosine_similarity`.
    Cosine,
}

/// A set of equal-length vectors plus a metric (for `pairwise_distance`).
#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct PairwiseDistanceRequest {
    pub vectors: Vec<Vec<f64>>,
    #[serde(default)]
    pub metric: DistanceMetric,
    #[serde(default)]
    pub precision: Precision,
}

/// An N×N result matrix (row-major), e.g. a pairwise similarity/distance matrix.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct MatrixResponse {
    pub matrix: Vec<Vec<f64>>,
}

/// A vector plus a scalar coefficient (for `scale`).
#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct ScaleRequest {
    pub v: Vec<f64>,
    pub c: f64,
    #[serde(default)]
    pub precision: Precision,
}

/// A vector plus a scalar exponent (for `elementwise_power`).
#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct PowerRequest {
    pub v: Vec<f64>,
    pub p: f64,
    #[serde(default)]
    pub precision: Precision,
}

/// A sample of values for a descriptive statistic (f64-only — no
/// `precision`). Named `values` rather than `v` to read as a sample,
/// not a geometric vector.
#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct SampleRequest {
    pub values: Vec<f64>,
}

/// A sample plus the percentile `p` (in `0..=100`).
#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct PercentileRequest {
    pub values: Vec<f64>,
    pub p: f64,
}

/// Length validation for descriptive-stat samples (distinct message from
/// `validate_unary`, which speaks of a geometric vector `v`).
fn validate_sample(values: &[f64], min_len: usize, ctx: &str) -> Result<(), RegistryError> {
    if values.len() < min_len {
        return Err(RegistryError::BadRequest(format!(
            "{ctx} requires at least {min_len} value(s), got {}",
            values.len()
        )));
    }
    if values.len() > MAX_VECTOR_LEN {
        return Err(RegistryError::BadRequest(format!(
            "values length {} exceeds maximum {MAX_VECTOR_LEN}",
            values.len()
        )));
    }
    Ok(())
}

// --- distances (scalar) ---

pub(crate) fn run_squared_euclidean_distance(
    req: BinaryVectorRequest,
) -> Result<ScalarResponse<f64>, RegistryError> {
    validate_binary(&req.a, &req.b)?;
    let result = match req.precision {
        Precision::F32 => squared_euclidean_distance(&cast_f32(&req.a), &cast_f32(&req.b)) as f64,
        Precision::F64 => squared_euclidean_distance_f64(&req.a, &req.b),
    };
    Ok(ScalarResponse { result })
}

pub(crate) fn run_manhattan_distance(
    req: BinaryVectorRequest,
) -> Result<ScalarResponse<f64>, RegistryError> {
    validate_binary(&req.a, &req.b)?;
    let result = match req.precision {
        Precision::F32 => manhattan_distance(&cast_f32(&req.a), &cast_f32(&req.b)) as f64,
        Precision::F64 => manhattan_distance_f64(&req.a, &req.b),
    };
    Ok(ScalarResponse { result })
}

// --- element-wise algebra (vector) ---

pub(crate) fn run_add(req: BinaryVectorRequest) -> Result<VectorResponse, RegistryError> {
    validate_binary(&req.a, &req.b)?;
    let result = match req.precision {
        Precision::F32 => widen(add(&cast_f32(&req.a), &cast_f32(&req.b))),
        Precision::F64 => add_f64(&req.a, &req.b),
    };
    Ok(VectorResponse { result })
}

pub(crate) fn run_subtract(req: BinaryVectorRequest) -> Result<VectorResponse, RegistryError> {
    validate_binary(&req.a, &req.b)?;
    let result = match req.precision {
        Precision::F32 => widen(subtract(&cast_f32(&req.a), &cast_f32(&req.b))),
        Precision::F64 => subtract_f64(&req.a, &req.b),
    };
    Ok(VectorResponse { result })
}

pub(crate) fn run_hadamard_division(
    req: BinaryVectorRequest,
) -> Result<VectorResponse, RegistryError> {
    validate_binary(&req.a, &req.b)?;
    let result = match req.precision {
        Precision::F32 => hadamard_division(&cast_f32(&req.a), &cast_f32(&req.b)).map(widen),
        Precision::F64 => hadamard_division_f64(&req.a, &req.b),
    };
    result
        .map(|result| VectorResponse { result })
        .ok_or_else(|| {
            RegistryError::BadRequest(
                "hadamard_division undefined: b has a zero component".to_string(),
            )
        })
}

pub(crate) fn run_scale(req: ScaleRequest) -> Result<VectorResponse, RegistryError> {
    validate_unary(&req.v)?;
    let result = match req.precision {
        Precision::F32 => widen(scale(&cast_f32(&req.v), req.c as f32)),
        Precision::F64 => scale_f64(&req.v, req.c),
    };
    Ok(VectorResponse { result })
}

pub(crate) fn run_negate(req: UnaryVectorRequest) -> Result<VectorResponse, RegistryError> {
    validate_unary(&req.v)?;
    let result = match req.precision {
        Precision::F32 => widen(negate(&cast_f32(&req.v))),
        Precision::F64 => negate_f64(&req.v),
    };
    Ok(VectorResponse { result })
}

pub(crate) fn run_elementwise_power(req: PowerRequest) -> Result<VectorResponse, RegistryError> {
    validate_unary(&req.v)?;
    let result = match req.precision {
        Precision::F32 => widen(elementwise_power(&cast_f32(&req.v), req.p as f32)),
        Precision::F64 => elementwise_power_f64(&req.v, req.p),
    };
    Ok(VectorResponse { result })
}

// --- transforms ---

pub(crate) fn run_l2_normalize(req: UnaryVectorRequest) -> Result<VectorResponse, RegistryError> {
    validate_unary(&req.v)?;
    let result = match req.precision {
        Precision::F32 => l2_normalize(&cast_f32(&req.v)).map(widen),
        Precision::F64 => l2_normalize_f64(&req.v),
    };
    result
        .map(|result| VectorResponse { result })
        .ok_or_else(|| {
            RegistryError::BadRequest("l2_normalize undefined: zero-magnitude vector".to_string())
        })
}

pub(crate) fn run_centroid(req: VectorsRequest) -> Result<VectorResponse, RegistryError> {
    if req.vectors.is_empty() {
        return Err(RegistryError::BadRequest(
            "vectors must be non-empty".to_string(),
        ));
    }
    let dim = req.vectors[0].len();
    if dim == 0 {
        return Err(RegistryError::BadRequest(
            "each vector must be non-empty".to_string(),
        ));
    }
    if dim > MAX_VECTOR_LEN {
        return Err(RegistryError::BadRequest(format!(
            "vector length {dim} exceeds maximum {MAX_VECTOR_LEN}"
        )));
    }
    if req.vectors.iter().any(|v| v.len() != dim) {
        return Err(RegistryError::BadRequest(
            "all vectors must have the same length".to_string(),
        ));
    }
    let result = match req.precision {
        Precision::F32 => {
            let cast: Vec<Vec<f32>> = req.vectors.iter().map(|v| cast_f32(v)).collect();
            let refs: Vec<&[f32]> = cast.iter().map(|v| v.as_slice()).collect();
            centroid(&refs).map(widen)
        }
        Precision::F64 => {
            let refs: Vec<&[f64]> = req.vectors.iter().map(|v| v.as_slice()).collect();
            centroid_f64(&refs)
        }
    };
    // Validated above, so `None` shouldn't occur; surface it loudly anyway.
    result
        .map(|result| VectorResponse { result })
        .ok_or_else(|| RegistryError::BadRequest("centroid undefined".to_string()))
}

/// Validate a matrix of vectors for a pairwise op: non-empty, count within the
/// pairwise cap, equal non-empty lengths within `MAX_VECTOR_LEN`. Returns `dim`.
fn validate_matrix(vectors: &[Vec<f64>]) -> Result<usize, RegistryError> {
    if vectors.is_empty() {
        return Err(RegistryError::BadRequest(
            "vectors must be non-empty".to_string(),
        ));
    }
    if vectors.len() > MAX_PAIRWISE_VECTORS {
        return Err(RegistryError::BadRequest(format!(
            "vector count {} exceeds maximum {MAX_PAIRWISE_VECTORS}",
            vectors.len()
        )));
    }
    let dim = vectors[0].len();
    if dim == 0 {
        return Err(RegistryError::BadRequest(
            "each vector must be non-empty".to_string(),
        ));
    }
    if dim > MAX_VECTOR_LEN {
        return Err(RegistryError::BadRequest(format!(
            "vector length {dim} exceeds maximum {MAX_VECTOR_LEN}"
        )));
    }
    if vectors.iter().any(|v| v.len() != dim) {
        return Err(RegistryError::BadRequest(
            "all vectors must have the same length".to_string(),
        ));
    }
    Ok(dim)
}

/// Fill an N×N symmetric matrix where entry `(i, j)` is `f(i, j)`. Computes the
/// upper triangle (and diagonal) once and mirrors it.
// Each iteration writes both `matrix[i][j]` and its mirror `matrix[j][i]` (two
// different rows), which an `iter_mut().enumerate()` can't express — index
// access is the natural form here.
#[allow(clippy::needless_range_loop)]
fn symmetric_matrix(n: usize, mut f: impl FnMut(usize, usize) -> f64) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let value = f(i, j);
            matrix[i][j] = value;
            matrix[j][i] = value;
        }
    }
    matrix
}

/// `util/pairwise_cosine` — full N×N cosine-similarity matrix over the input
/// vectors, in one call instead of N² per-pair round-trips.
pub(crate) fn run_pairwise_cosine(req: VectorsRequest) -> Result<MatrixResponse, RegistryError> {
    validate_matrix(&req.vectors)?;
    let n = req.vectors.len();
    let matrix = match req.precision {
        Precision::F32 => {
            let cast: Vec<Vec<f32>> = req.vectors.iter().map(|v| cast_f32(v)).collect();
            symmetric_matrix(n, |i, j| cosine_similarity(&cast[i], &cast[j]) as f64)
        }
        Precision::F64 => symmetric_matrix(n, |i, j| {
            cosine_similarity_f64(&req.vectors[i], &req.vectors[j])
        }),
    };
    Ok(MatrixResponse { matrix })
}

/// `util/pairwise_distance` — full N×N distance matrix under the chosen metric.
pub(crate) fn run_pairwise_distance(
    req: PairwiseDistanceRequest,
) -> Result<MatrixResponse, RegistryError> {
    validate_matrix(&req.vectors)?;
    let n = req.vectors.len();
    let matrix = match req.precision {
        Precision::F32 => {
            let c: Vec<Vec<f32>> = req.vectors.iter().map(|v| cast_f32(v)).collect();
            symmetric_matrix(n, |i, j| match req.metric {
                DistanceMetric::Euclidean => euclidean_distance(&c[i], &c[j]) as f64,
                DistanceMetric::SquaredEuclidean => squared_euclidean_distance(&c[i], &c[j]) as f64,
                DistanceMetric::Manhattan => manhattan_distance(&c[i], &c[j]) as f64,
                DistanceMetric::Cosine => 1.0 - cosine_similarity(&c[i], &c[j]) as f64,
            })
        }
        Precision::F64 => symmetric_matrix(n, |i, j| {
            let (a, b) = (&req.vectors[i], &req.vectors[j]);
            match req.metric {
                DistanceMetric::Euclidean => euclidean_distance_f64(a, b),
                DistanceMetric::SquaredEuclidean => squared_euclidean_distance_f64(a, b),
                DistanceMetric::Manhattan => manhattan_distance_f64(a, b),
                DistanceMetric::Cosine => 1.0 - cosine_similarity_f64(a, b),
            }
        }),
    };
    Ok(MatrixResponse { matrix })
}

// --- descriptive statistics (f64 only) ---

pub(crate) fn run_mean(req: SampleRequest) -> Result<ScalarResponse<f64>, RegistryError> {
    validate_sample(&req.values, 1, "mean")?;
    mean(&req.values)
        .map(|result| ScalarResponse { result })
        .ok_or_else(|| RegistryError::BadRequest("mean undefined (empty input)".to_string()))
}

pub(crate) fn run_variance(req: SampleRequest) -> Result<ScalarResponse<f64>, RegistryError> {
    validate_sample(&req.values, 2, "variance")?;
    variance(&req.values)
        .map(|result| ScalarResponse { result })
        .ok_or_else(|| RegistryError::BadRequest("variance requires n >= 2".to_string()))
}

pub(crate) fn run_std_dev(req: SampleRequest) -> Result<ScalarResponse<f64>, RegistryError> {
    validate_sample(&req.values, 2, "std_dev")?;
    std_dev(&req.values)
        .map(|result| ScalarResponse { result })
        .ok_or_else(|| RegistryError::BadRequest("std_dev requires n >= 2".to_string()))
}

pub(crate) fn run_median(req: SampleRequest) -> Result<ScalarResponse<f64>, RegistryError> {
    validate_sample(&req.values, 1, "median")?;
    median(&req.values)
        .map(|result| ScalarResponse { result })
        .ok_or_else(|| RegistryError::BadRequest("median undefined (empty input)".to_string()))
}

pub(crate) fn run_percentile(req: PercentileRequest) -> Result<ScalarResponse<f64>, RegistryError> {
    validate_sample(&req.values, 1, "percentile")?;
    if !(0.0..=100.0).contains(&req.p) {
        return Err(RegistryError::BadRequest(format!(
            "percentile p must be in 0..=100, got {}",
            req.p
        )));
    }
    percentile(&req.values, req.p)
        .map(|result| ScalarResponse { result })
        .ok_or_else(|| RegistryError::BadRequest("percentile undefined".to_string()))
}

pub(crate) fn run_softmax(req: SampleRequest) -> Result<VectorResponse, RegistryError> {
    validate_sample(&req.values, 1, "softmax")?;
    Ok(VectorResponse {
        result: softmax(&req.values),
    })
}

pub(crate) fn run_spearman_correlation(
    req: TwoVectorF64Request,
) -> Result<ScalarResponse<f64>, RegistryError> {
    validate_binary(&req.a, &req.b)?;
    if req.a.len() < 3 {
        return Err(RegistryError::BadRequest(format!(
            "spearman_correlation requires n >= 3, got {}",
            req.a.len()
        )));
    }
    spearman_rank_correlation(&req.a, &req.b)
        .map(|result| ScalarResponse { result })
        .ok_or_else(|| {
            RegistryError::BadRequest(
                "spearman_correlation undefined (likely a constant input)".to_string(),
            )
        })
}

/// Widen an f32 result vector to the f64 wire type (JSON numbers are f64).
fn widen(v: Vec<f32>) -> Vec<f64> {
    v.into_iter().map(|x| x as f64).collect()
}
