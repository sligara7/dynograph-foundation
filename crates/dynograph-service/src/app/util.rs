//! `util — stateless pure-math utilities` route handlers — split out of `app.rs`. `use super::*`
//! inherits the shared imports and helpers (`AppState`, `graph_entry`,
//! the `crate::*` wire types) from the parent `app` module.

use super::*;

// =========================================================================
// /v1/util/* — pure-math utility endpoints.
//
// Stateless (no graph_id, no registry access). Each handler is a thin
// adapter over `crate::util::run_*`; the math lives there and stays
// trivially unit-testable.
// =========================================================================

#[utoipa::path(
    post,
    path = "/v1/util/cosine_similarity",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "cosine similarity", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_cosine_similarity(
    Json(req): Json<BinaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_cosine_similarity(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/dot_product",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "dot product", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_dot_product(
    Json(req): Json<BinaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_dot_product(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/euclidean_distance",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "euclidean distance", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_euclidean_distance(
    Json(req): Json<BinaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_euclidean_distance(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/l2_norm",
    request_body = UnaryVectorRequest,
    responses(
        (status = 200, description = "L2 norm", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_l2_norm(
    Json(req): Json<UnaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_l2_norm(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/hadamard",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "elementwise product", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_hadamard(
    Json(req): Json<BinaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_hadamard(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/pearson_correlation",
    request_body = TwoVectorF64Request,
    responses(
        (status = 200, description = "Pearson correlation", body = ScalarResponse<f64>),
        (status = 400, description = "validation error / undefined"),
    ),
    tag = "util",
)]
pub(crate) async fn util_pearson_correlation(
    Json(req): Json<TwoVectorF64Request>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_pearson_correlation(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/linear_regression_slope",
    request_body = PointsRequest,
    responses(
        (status = 200, description = "regression slope", body = ScalarResponse<f64>),
        (status = 400, description = "validation error / undefined"),
    ),
    tag = "util",
)]
pub(crate) async fn util_linear_regression_slope(
    Json(req): Json<PointsRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_linear_regression_slope(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/jaro_winkler",
    request_body = BinaryStringRequest,
    responses(
        (status = 200, description = "Jaro-Winkler similarity (0..=100)", body = ScalarResponse<u32>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_jaro_winkler(
    Json(req): Json<BinaryStringRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_jaro_winkler(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/token_sort_ratio",
    request_body = BinaryStringRequest,
    responses(
        (status = 200, description = "token-sort ratio (0..=100)", body = ScalarResponse<u32>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_token_sort_ratio(
    Json(req): Json<BinaryStringRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_token_sort_ratio(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/squared_euclidean_distance",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "squared euclidean distance", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_squared_euclidean_distance(
    Json(req): Json<BinaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_squared_euclidean_distance(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/manhattan_distance",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "manhattan (L1) distance", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_manhattan_distance(
    Json(req): Json<BinaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_manhattan_distance(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/add",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "element-wise sum", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_add(
    Json(req): Json<BinaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_add(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/subtract",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "element-wise difference", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_subtract(
    Json(req): Json<BinaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_subtract(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/scale",
    request_body = ScaleRequest,
    responses(
        (status = 200, description = "scalar multiple", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_scale(Json(req): Json<ScaleRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_scale(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/negate",
    request_body = UnaryVectorRequest,
    responses(
        (status = 200, description = "negated vector", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_negate(
    Json(req): Json<UnaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_negate(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/hadamard_division",
    request_body = BinaryVectorRequest,
    responses(
        (status = 200, description = "element-wise quotient", body = VectorResponse),
        (status = 400, description = "validation error / zero divisor"),
    ),
    tag = "util",
)]
pub(crate) async fn util_hadamard_division(
    Json(req): Json<BinaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_hadamard_division(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/elementwise_power",
    request_body = PowerRequest,
    responses(
        (status = 200, description = "element-wise power", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_elementwise_power(
    Json(req): Json<PowerRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_elementwise_power(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/l2_normalize",
    request_body = UnaryVectorRequest,
    responses(
        (status = 200, description = "unit-length vector", body = VectorResponse),
        (status = 400, description = "validation error / zero magnitude"),
    ),
    tag = "util",
)]
pub(crate) async fn util_l2_normalize(
    Json(req): Json<UnaryVectorRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_l2_normalize(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/centroid",
    request_body = VectorsRequest,
    responses(
        (status = 200, description = "component-wise mean vector", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_centroid(
    Json(req): Json<VectorsRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_centroid(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/pairwise_cosine",
    request_body = VectorsRequest,
    responses(
        (status = 200, description = "N×N cosine-similarity matrix", body = MatrixResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_pairwise_cosine(
    Json(req): Json<VectorsRequest>,
) -> Result<Response, RegistryError> {
    // O(N²·dim) — offload to the blocking pool so a large matrix can't pin a
    // tokio worker (the timeout layer can't preempt CPU-bound work). Mirrors how
    // the storage/engine wrappers run their synchronous work; a panic in the
    // task is re-raised here, a validation error becomes a 400.
    let resp = tokio::task::spawn_blocking(move || run_pairwise_cosine(req))
        .await
        .unwrap_or_else(|e| std::panic::resume_unwind(e.into_panic()))?;
    Ok(Json(resp).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/pairwise_distance",
    request_body = PairwiseDistanceRequest,
    responses(
        (status = 200, description = "N×N distance matrix under the chosen metric", body = MatrixResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_pairwise_distance(
    Json(req): Json<PairwiseDistanceRequest>,
) -> Result<Response, RegistryError> {
    // O(N²·dim) — offload to the blocking pool (see `util_pairwise_cosine`).
    let resp = tokio::task::spawn_blocking(move || run_pairwise_distance(req))
        .await
        .unwrap_or_else(|e| std::panic::resume_unwind(e.into_panic()))?;
    Ok(Json(resp).into_response())
}

/// Normal-form game-theory analysis (dominant strategies, pure/mixed Nash,
/// Pareto optimality, the `nash_is_pareto_suboptimal` headline). Stateless pure
/// math — see `crate::game` for the wire shape and the explicit out-of-scope
/// boundary. The Pareto pass is O(cells²·players); offload to the blocking pool
/// like the pairwise matrix ops.
#[utoipa::path(
    post,
    path = "/v1/util/game/analyze",
    request_body = GameAnalyzeRequest,
    responses(
        (status = 200, description = "game-theoretic analysis", body = GameAnalyzeResponse),
        (status = 400, description = "validation error (malformed or oversized game)"),
    ),
    tag = "util",
)]
pub(crate) async fn util_game_analyze(
    Json(req): Json<GameAnalyzeRequest>,
) -> Result<Response, RegistryError> {
    let resp = tokio::task::spawn_blocking(move || run_game_analyze(req))
        .await
        .unwrap_or_else(|e| std::panic::resume_unwind(e.into_panic()))?;
    Ok(Json(resp).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/mean",
    request_body = SampleRequest,
    responses(
        (status = 200, description = "arithmetic mean", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_mean(Json(req): Json<SampleRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_mean(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/variance",
    request_body = SampleRequest,
    responses(
        (status = 200, description = "sample variance (n-1)", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_variance(
    Json(req): Json<SampleRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_variance(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/std_dev",
    request_body = SampleRequest,
    responses(
        (status = 200, description = "sample standard deviation", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_std_dev(
    Json(req): Json<SampleRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_std_dev(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/median",
    request_body = SampleRequest,
    responses(
        (status = 200, description = "median (50th percentile)", body = ScalarResponse<f64>),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_median(Json(req): Json<SampleRequest>) -> Result<Response, RegistryError> {
    Ok(Json(run_median(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/percentile",
    request_body = PercentileRequest,
    responses(
        (status = 200, description = "linear-interpolated percentile", body = ScalarResponse<f64>),
        (status = 400, description = "validation error / p out of range"),
    ),
    tag = "util",
)]
pub(crate) async fn util_percentile(
    Json(req): Json<PercentileRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_percentile(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/softmax",
    request_body = SampleRequest,
    responses(
        (status = 200, description = "softmax distribution (sums to 1)", body = VectorResponse),
        (status = 400, description = "validation error"),
    ),
    tag = "util",
)]
pub(crate) async fn util_softmax(
    Json(req): Json<SampleRequest>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_softmax(req)?).into_response())
}

#[utoipa::path(
    post,
    path = "/v1/util/spearman_correlation",
    request_body = TwoVectorF64Request,
    responses(
        (status = 200, description = "Spearman rank correlation", body = ScalarResponse<f64>),
        (status = 400, description = "validation error / undefined"),
    ),
    tag = "util",
)]
pub(crate) async fn util_spearman_correlation(
    Json(req): Json<TwoVectorF64Request>,
) -> Result<Response, RegistryError> {
    Ok(Json(run_spearman_correlation(req)?).into_response())
}
