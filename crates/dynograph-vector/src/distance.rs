//! Vector distance functions, structured for autovectorization.
//!
//! Provides cosine similarity, dot product, and Euclidean distance
//! for f32 vectors. The f32 inner loops are written so an
//! autovectorizing compiler can lower them to SIMD instructions —
//! **but only if the build enables the target features**. With a stock
//! release build (`cargo build --release` on its own), the compiler
//! targets a portable baseline (e.g. `x86-64-v1`) and these loops emit
//! scalar code.
//!
//! To get SIMD, build with one of:
//! - `RUSTFLAGS="-C target-cpu=native"` — best codegen for the host
//!   CPU; the resulting binary may not run on older CPUs of the same
//!   family.
//! - `RUSTFLAGS="-C target-feature=+avx2,+fma"` (x86_64) or
//!   `+neon` (aarch64) — portable across CPUs that have those
//!   features.
//!
//! There is no runtime feature detection here. If your deployment
//! cares about throughput, set the flag explicitly; the inner loops
//! will autovectorize cleanly with either of the options above.

/// Dot product of two f32 slices.
///
/// The 8-wide unrolled accumulator is structured to autovectorize when
/// the build enables AVX2 (x86_64) or NEON (aarch64) — see the module
/// docstring for the required `RUSTFLAGS`. Without those flags the
/// loop emits scalar code; the function is still correct, just not
/// SIMD-accelerated.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    let mut sum = 0.0f32;
    // Process in chunks of 8 for SIMD-friendly access patterns
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;

    for i in 0..chunks {
        let base = i * 8;
        let mut local_sum = 0.0f32;
        local_sum += a[base] * b[base];
        local_sum += a[base + 1] * b[base + 1];
        local_sum += a[base + 2] * b[base + 2];
        local_sum += a[base + 3] * b[base + 3];
        local_sum += a[base + 4] * b[base + 4];
        local_sum += a[base + 5] * b[base + 5];
        local_sum += a[base + 6] * b[base + 6];
        local_sum += a[base + 7] * b[base + 7];
        sum += local_sum;
    }

    let base = chunks * 8;
    for i in 0..remainder {
        sum += a[base + i] * b[base + i];
    }

    sum
}

/// L2 norm (magnitude) of an f32 vector.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    dot_product(v, v).sqrt()
}

/// Cosine similarity between two f32 vectors.
///
/// Returns a value in [-1.0, 1.0] where:
/// - 1.0 = identical direction
/// - 0.0 = orthogonal
/// - -1.0 = opposite direction
///
/// Returns 0.0 only when the cosine is genuinely undefined — either
/// vector has *exactly* zero magnitude, or a non-finite component made
/// the denominator non-finite. The guard is `denom == 0.0 ||
/// !denom.is_finite()`, NOT `denom < EPSILON`: an epsilon floor would
/// wrongly collapse tiny-but-valid vectors (e.g. two parallel `1e-4`
/// vectors, true cosine 1.0) to a fake 0.0.
///
/// A degenerate input scoring 0.0 reads as "orthogonal," not "invalid"
/// — callers that need to fail loud on bad data should screen inputs
/// with [`validate_similarity_vector`] at ingest rather than relying on
/// this return value.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    let denom = norm_a * norm_b;
    if denom == 0.0 || !denom.is_finite() {
        return 0.0;
    }
    // Clamp to [-1, 1] to handle floating point imprecision
    (dot / denom).clamp(-1.0, 1.0)
}

/// Reject a vector that cannot produce a meaningful cosine similarity:
/// one with a non-finite component (`NaN` / `±∞`) or zero magnitude
/// (all components zero, including the empty vector).
///
/// Both cases would otherwise score a silent `0.0` against everything
/// — indistinguishable from a legitimately orthogonal hit — so callers
/// that ingest user- or model-supplied embeddings (the HNSW index, the
/// resolver, the `set_embedding` / `similar` / `resolve-or-create`
/// handlers) should screen here and surface a loud error instead of
/// admitting degenerate data.
///
/// The `Err` carries a human-readable reason suitable for a 400 body.
pub fn validate_similarity_vector(v: &[f32]) -> Result<(), &'static str> {
    let mut sum_sq = 0.0f32;
    for &x in v {
        if !x.is_finite() {
            return Err("vector contains a non-finite component (NaN or ±∞)");
        }
        sum_sq += x * x;
    }
    if sum_sq == 0.0 {
        return Err("vector has zero magnitude (all components zero)");
    }
    Ok(())
}

/// Squared Euclidean distance between two f32 vectors — Euclidean
/// distance without the final `sqrt`.
///
/// Cheaper than [`euclidean_distance`] and order-preserving, so prefer it
/// when you only need to *rank* or *compare* distances (kNN, nearest-
/// centroid, clustering) rather than report a metric value. Same 8-wide
/// unrolled accumulator as [`dot_product`], so it autovectorizes under
/// the same `RUSTFLAGS` (see the module docstring).
#[inline]
pub fn squared_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    let mut sum = 0.0f32;
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;

    for i in 0..chunks {
        let base = i * 8;
        let mut local_sum = 0.0f32;
        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];
        let d4 = a[base + 4] - b[base + 4];
        let d5 = a[base + 5] - b[base + 5];
        let d6 = a[base + 6] - b[base + 6];
        let d7 = a[base + 7] - b[base + 7];
        local_sum += d0 * d0;
        local_sum += d1 * d1;
        local_sum += d2 * d2;
        local_sum += d3 * d3;
        local_sum += d4 * d4;
        local_sum += d5 * d5;
        local_sum += d6 * d6;
        local_sum += d7 * d7;
        sum += local_sum;
    }

    let base = chunks * 8;
    for i in 0..remainder {
        let d = a[base + i] - b[base + i];
        sum += d * d;
    }

    sum
}

/// Euclidean distance between two f32 vectors — `sqrt` of
/// [`squared_euclidean_distance`].
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    squared_euclidean_distance(a, b).sqrt()
}

/// Manhattan (L1 / taxicab) distance between two f32 vectors: the sum of
/// absolute per-component differences. More robust to a single large
/// outlier component than Euclidean. Same 8-wide unrolled structure for
/// autovectorization.
#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    let mut sum = 0.0f32;
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;

    for i in 0..chunks {
        let base = i * 8;
        let mut local_sum = 0.0f32;
        local_sum += (a[base] - b[base]).abs();
        local_sum += (a[base + 1] - b[base + 1]).abs();
        local_sum += (a[base + 2] - b[base + 2]).abs();
        local_sum += (a[base + 3] - b[base + 3]).abs();
        local_sum += (a[base + 4] - b[base + 4]).abs();
        local_sum += (a[base + 5] - b[base + 5]).abs();
        local_sum += (a[base + 6] - b[base + 6]).abs();
        local_sum += (a[base + 7] - b[base + 7]).abs();
        sum += local_sum;
    }

    let base = chunks * 8;
    for i in 0..remainder {
        sum += (a[base + i] - b[base + i]).abs();
    }

    sum
}

/// Element-wise (Hadamard) product of two f32 vectors.
///
/// Returns a new vector `c` where `c[i] = a[i] * b[i]`.
#[inline]
pub fn hadamard(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Return a unit-length (L2-normalized) copy of `v`: each component
/// divided by the vector's magnitude. The standard preprocessing step
/// before cosine / dot-product comparison.
///
/// Returns `None` when the vector has zero magnitude or a non-finite
/// magnitude — the degenerate cases [`validate_similarity_vector`]
/// rejects — so the caller makes the fallback explicit instead of
/// dividing into `NaN`/`∞`.
#[inline]
pub fn l2_normalize(v: &[f32]) -> Option<Vec<f32>> {
    let norm = l2_norm(v);
    if norm == 0.0 || !norm.is_finite() {
        return None;
    }
    Some(v.iter().map(|x| x / norm).collect())
}

/// Component-wise mean (centroid) of a set of equal-length vectors — the
/// average vector, e.g. a cluster prototype or "average embedding."
///
/// Returns `None` if `vectors` is empty, the vectors are not all the same
/// length, or that length is zero — an ambiguous centroid the caller
/// should handle explicitly.
pub fn centroid(vectors: &[&[f32]]) -> Option<Vec<f32>> {
    let dim = vectors.first()?.len();
    if dim == 0 || vectors.iter().any(|v| v.len() != dim) {
        return None;
    }
    let mut acc = vec![0.0f32; dim];
    for v in vectors {
        for (a, x) in acc.iter_mut().zip(v.iter()) {
            *a += *x;
        }
    }
    let n = vectors.len() as f32;
    for a in &mut acc {
        *a /= n;
    }
    Some(acc)
}

/// Element-wise sum `a + b` of two equal-length f32 vectors.
#[inline]
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Element-wise difference `a - b` of two equal-length f32 vectors.
#[inline]
pub fn subtract(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Scalar multiplication: each component of `v` times `c`.
#[inline]
pub fn scale(v: &[f32], c: f32) -> Vec<f32> {
    v.iter().map(|x| x * c).collect()
}

/// Negate a vector (reverse direction) — `scale(v, -1.0)`.
#[inline]
pub fn negate(v: &[f32]) -> Vec<f32> {
    scale(v, -1.0)
}

/// Element-wise (Hadamard) division `a / b`.
///
/// Returns `None` if any component of `b` is zero — dividing would yield
/// a silent `±∞`/`NaN`, so the caller makes the fallback explicit.
#[inline]
pub fn hadamard_division(a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    if b.contains(&0.0) {
        return None;
    }
    Some(a.iter().zip(b.iter()).map(|(x, y)| x / y).collect())
}

/// Raise each component of `v` to the power `p`. Note a negative base
/// with a non-integer `p` yields `NaN` (per IEEE `powf`); screen inputs
/// upstream if that domain is reachable.
#[inline]
pub fn elementwise_power(v: &[f32], p: f32) -> Vec<f32> {
    v.iter().map(|x| x.powf(p)).collect()
}

// =============================================================================
// f64 variants
//
// f32 is the primary path (embedding-shaped, SIMD-friendly).
// f64 variants exist for non-embedding consumers — domain measurements,
// analytic dimension scores, statistics — where the extra precision matters
// and the inputs are typically small (≤ 32 elements), so SIMD chunking
// is not worth the code surface.
// =============================================================================

/// Dot product of two f64 slices.
#[inline]
pub fn dot_product_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm (magnitude) of an f64 vector.
#[inline]
pub fn l2_norm_f64(v: &[f64]) -> f64 {
    dot_product_f64(v, v).sqrt()
}

/// Cosine similarity between two f64 vectors.
///
/// Returns `0.0` only when the cosine is genuinely undefined — a
/// zero-magnitude or non-finite denominator (`denom == 0.0 ||
/// !denom.is_finite()`), NOT an epsilon floor that would collapse
/// tiny-but-valid vectors. Result is clamped to `[-1.0, 1.0]` to handle
/// floating-point imprecision.
#[inline]
pub fn cosine_similarity_f64(a: &[f64], b: &[f64]) -> f64 {
    let dot = dot_product_f64(a, b);
    let norm_a = l2_norm_f64(a);
    let norm_b = l2_norm_f64(b);
    let denom = norm_a * norm_b;
    if denom == 0.0 || !denom.is_finite() {
        return 0.0;
    }
    (dot / denom).clamp(-1.0, 1.0)
}

/// Squared Euclidean distance between two f64 vectors (no final `sqrt`).
/// Order-preserving — prefer when ranking rather than reporting a metric.
#[inline]
pub fn squared_euclidean_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Euclidean distance between two f64 vectors — `sqrt` of
/// [`squared_euclidean_distance_f64`].
#[inline]
pub fn euclidean_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    squared_euclidean_distance_f64(a, b).sqrt()
}

/// Manhattan (L1 / taxicab) distance between two f64 vectors: the sum of
/// absolute per-component differences.
#[inline]
pub fn manhattan_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Element-wise (Hadamard) product of two f64 vectors.
#[inline]
pub fn hadamard_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Unit-length (L2-normalized) copy of an f64 vector. `None` on zero or
/// non-finite magnitude. See [`l2_normalize`].
#[inline]
pub fn l2_normalize_f64(v: &[f64]) -> Option<Vec<f64>> {
    let norm = l2_norm_f64(v);
    if norm == 0.0 || !norm.is_finite() {
        return None;
    }
    Some(v.iter().map(|x| x / norm).collect())
}

/// Component-wise mean (centroid) of a set of equal-length f64 vectors.
/// `None` if empty, ragged, or zero-dimension. See [`centroid`].
pub fn centroid_f64(vectors: &[&[f64]]) -> Option<Vec<f64>> {
    let dim = vectors.first()?.len();
    if dim == 0 || vectors.iter().any(|v| v.len() != dim) {
        return None;
    }
    let mut acc = vec![0.0f64; dim];
    for v in vectors {
        for (a, x) in acc.iter_mut().zip(v.iter()) {
            *a += *x;
        }
    }
    let n = vectors.len() as f64;
    for a in &mut acc {
        *a /= n;
    }
    Some(acc)
}

/// Element-wise sum `a + b` of two equal-length f64 vectors.
#[inline]
pub fn add_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Element-wise difference `a - b` of two equal-length f64 vectors.
#[inline]
pub fn subtract_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Scalar multiplication of an f64 vector by `c`.
#[inline]
pub fn scale_f64(v: &[f64], c: f64) -> Vec<f64> {
    v.iter().map(|x| x * c).collect()
}

/// Negate an f64 vector — `scale_f64(v, -1.0)`.
#[inline]
pub fn negate_f64(v: &[f64]) -> Vec<f64> {
    scale_f64(v, -1.0)
}

/// Element-wise (Hadamard) division `a / b`. `None` if any component of
/// `b` is zero. See [`hadamard_division`].
#[inline]
pub fn hadamard_division_f64(a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    if b.contains(&0.0) {
        return None;
    }
    Some(a.iter().zip(b.iter()).map(|(x, y)| x / y).collect())
}

/// Raise each component of an f64 vector to the power `p`. See
/// [`elementwise_power`].
#[inline]
pub fn elementwise_power_f64(v: &[f64], p: f64) -> Vec<f64> {
    v.iter().map(|x| x.powf(p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 32
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn dot_product_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((dot_product(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn dot_product_large_vector() {
        // Test with 768-dim vectors (typical embedding size)
        let a: Vec<f32> = (0..768).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..768).map(|i| ((768 - i) as f32) * 0.01).collect();
        let result = dot_product(&a, &b);
        // Just verify it computes without panic and is reasonable
        assert!(result.is_finite());
        assert!(result > 0.0);
    }

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let zero = vec![0.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &zero), 0.0);
    }

    #[test]
    fn cosine_tiny_but_valid_vectors_not_collapsed() {
        // Two parallel vectors whose magnitudes are well below
        // f32::EPSILON. The old `denom < EPSILON` guard returned a fake
        // 0.0 ("orthogonal") here; the correct cosine is 1.0.
        let a = vec![1e-4_f32, 2e-4, 3e-4];
        let b = vec![2e-4_f32, 4e-4, 6e-4]; // 2x a, same direction
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn validate_similarity_vector_accepts_normal() {
        assert!(validate_similarity_vector(&[1.0, 2.0, 3.0]).is_ok());
        // Tiny-but-nonzero is valid.
        assert!(validate_similarity_vector(&[0.0, 1e-20, 0.0]).is_ok());
    }

    #[test]
    fn validate_similarity_vector_rejects_zero_magnitude() {
        assert!(validate_similarity_vector(&[0.0, 0.0, 0.0]).is_err());
        // Empty counts as zero magnitude.
        assert!(validate_similarity_vector(&[]).is_err());
    }

    #[test]
    fn validate_similarity_vector_rejects_non_finite() {
        assert!(validate_similarity_vector(&[1.0, f32::NAN, 3.0]).is_err());
        assert!(validate_similarity_vector(&[1.0, f32::INFINITY, 3.0]).is_err());
        assert!(validate_similarity_vector(&[f32::NEG_INFINITY, 2.0]).is_err());
    }

    #[test]
    fn cosine_768_dim_normalized() {
        // Two random-ish normalized 768-dim vectors
        let dim = 768;
        let a: Vec<f32> = (0..dim)
            .map(|i| ((i * 7 + 3) % 100) as f32 / 100.0)
            .collect();
        let b: Vec<f32> = (0..dim)
            .map(|i| ((i * 13 + 7) % 100) as f32 / 100.0)
            .collect();
        let sim = cosine_similarity(&a, &b);
        assert!((-1.0..=1.0).contains(&sim));
    }

    #[test]
    fn euclidean_same_point() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((euclidean_distance(&a, &a)).abs() < 1e-6);
    }

    #[test]
    fn euclidean_known_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn euclidean_unrolled_matches_naive_across_chunk_boundary() {
        // 13 elements exercises both the 8-wide chunk and the 5-element
        // remainder tail of the unrolled accumulator.
        let a: Vec<f32> = (0..13).map(|i| (i as f32) * 0.5).collect();
        let b: Vec<f32> = (0..13).map(|i| (i as f32) * 0.25 + 1.0).collect();
        let naive: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();
        assert!((euclidean_distance(&a, &b) - naive).abs() < 1e-5);
    }

    #[test]
    fn hadamard_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = hadamard(&a, &b);
        assert_eq!(c, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn hadamard_with_zero() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_eq!(hadamard(&a, &b), vec![0.0, 2.0, 0.0]);
    }

    #[test]
    fn hadamard_negatives() {
        let a = vec![1.0, -2.0, 3.0];
        let b = vec![-1.0, -2.0, 3.0];
        assert_eq!(hadamard(&a, &b), vec![-1.0, 4.0, 9.0]);
    }

    // --- f64 variants ---

    #[test]
    fn dot_product_f64_basic() {
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![4.0_f64, 5.0, 6.0];
        assert!((dot_product_f64(&a, &b) - 32.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_similarity_f64_identical() {
        let a = vec![1.0_f64, 2.0, 3.0];
        assert!((cosine_similarity_f64(&a, &a) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_similarity_f64_opposite() {
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![-1.0_f64, -2.0, -3.0];
        assert!((cosine_similarity_f64(&a, &b) - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn cosine_similarity_f64_orthogonal() {
        let a = vec![1.0_f64, 0.0, 0.0];
        let b = vec![0.0_f64, 1.0, 0.0];
        assert!(cosine_similarity_f64(&a, &b).abs() < 1e-12);
    }

    #[test]
    fn cosine_similarity_f64_zero_vector() {
        let a = vec![1.0_f64, 2.0, 3.0];
        let zero = vec![0.0_f64, 0.0, 0.0];
        assert_eq!(cosine_similarity_f64(&a, &zero), 0.0);
    }

    #[test]
    fn euclidean_distance_f64_known() {
        let a = vec![0.0_f64, 0.0];
        let b = vec![3.0_f64, 4.0];
        assert!((euclidean_distance_f64(&a, &b) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn euclidean_distance_f64_same_point() {
        let a = vec![1.0_f64, 2.0, 3.0];
        assert!(euclidean_distance_f64(&a, &a).abs() < 1e-12);
    }

    #[test]
    fn hadamard_f64_basic() {
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![4.0_f64, 5.0, 6.0];
        assert_eq!(hadamard_f64(&a, &b), vec![4.0_f64, 10.0, 18.0]);
    }

    #[test]
    fn l2_norm_f64_basic() {
        let v = vec![3.0_f64, 4.0];
        assert!((l2_norm_f64(&v) - 5.0).abs() < 1e-12);
    }

    // --- squared_euclidean / manhattan ---

    #[test]
    fn squared_euclidean_is_euclidean_without_sqrt() {
        // Use 9 elements so both the 8-wide chunk and the remainder run.
        let a: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..9).map(|i| (i as f32) * 2.0).collect();
        let sq = squared_euclidean_distance(&a, &b);
        let e = euclidean_distance(&a, &b);
        assert!((sq - e * e).abs() < 1e-3, "sq={sq} e^2={}", e * e);
        assert!((squared_euclidean_distance_f64(&[0.0, 0.0], &[3.0, 4.0]) - 25.0).abs() < 1e-12);
    }

    #[test]
    fn manhattan_known() {
        // 9 elements to exercise chunk + remainder paths.
        let a = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0];
        // sum of |differences| = 1+2+3+4+5+6+7+8+9 = 45
        assert!((manhattan_distance(&a, &b) - 45.0).abs() < 1e-4);
        assert!((manhattan_distance_f64(&[1.0, 2.0], &[4.0, 6.0]) - 7.0).abs() < 1e-12);
    }

    // --- l2_normalize ---

    #[test]
    fn l2_normalize_yields_unit_vector() {
        let n = l2_normalize(&[3.0f32, 4.0]).unwrap();
        assert!((l2_norm(&n) - 1.0).abs() < 1e-6);
        assert!((n[0] - 0.6).abs() < 1e-6 && (n[1] - 0.8).abs() < 1e-6);
        let n64 = l2_normalize_f64(&[3.0, 4.0]).unwrap();
        assert!((l2_norm_f64(&n64) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn l2_normalize_rejects_degenerate() {
        assert_eq!(l2_normalize(&[0.0f32, 0.0]), None);
        assert_eq!(l2_normalize(&[f32::NAN, 1.0]), None);
        assert_eq!(l2_normalize_f64(&[0.0, 0.0]), None);
    }

    // --- centroid ---

    #[test]
    fn centroid_averages_componentwise() {
        let a = vec![1.0f32, 2.0];
        let b = vec![3.0f32, 6.0];
        let c = centroid(&[&a, &b]).unwrap();
        assert_eq!(c, vec![2.0, 4.0]);
        let a64 = vec![0.0f64, 0.0];
        let b64 = vec![2.0f64, 4.0];
        assert_eq!(centroid_f64(&[&a64, &b64]).unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn centroid_rejects_empty_and_ragged() {
        let empty: [&[f32]; 0] = [];
        assert_eq!(centroid(&empty), None);
        let a = vec![1.0f32, 2.0];
        let short = vec![1.0f32];
        assert_eq!(centroid(&[&a, &short]), None);
    }

    // --- element-wise algebra (Tier A) ---

    #[test]
    fn add_subtract_componentwise() {
        assert_eq!(add(&[1.0f32, 2.0], &[3.0, 4.0]), vec![4.0, 6.0]);
        assert_eq!(subtract(&[3.0f32, 4.0], &[1.0, 2.0]), vec![2.0, 2.0]);
        assert_eq!(add_f64(&[1.0, 2.0], &[3.0, 4.0]), vec![4.0, 6.0]);
        assert_eq!(subtract_f64(&[3.0, 4.0], &[1.0, 2.0]), vec![2.0, 2.0]);
    }

    #[test]
    fn scale_and_negate() {
        assert_eq!(scale(&[1.0f32, -2.0, 3.0], 2.0), vec![2.0, -4.0, 6.0]);
        assert_eq!(negate(&[1.0f32, -2.0, 3.0]), vec![-1.0, 2.0, -3.0]);
        assert_eq!(scale_f64(&[1.0, -2.0], 0.5), vec![0.5, -1.0]);
        assert_eq!(negate_f64(&[1.0, -2.0]), vec![-1.0, 2.0]);
    }

    #[test]
    fn hadamard_division_basic_and_zero_guard() {
        assert_eq!(
            hadamard_division(&[6.0f32, 8.0], &[2.0, 4.0]),
            Some(vec![3.0, 2.0])
        );
        assert_eq!(hadamard_division(&[1.0f32, 2.0], &[1.0, 0.0]), None);
        assert_eq!(
            hadamard_division_f64(&[6.0, 9.0], &[3.0, 3.0]),
            Some(vec![2.0, 3.0])
        );
        assert_eq!(hadamard_division_f64(&[1.0], &[0.0]), None);
    }

    #[test]
    fn elementwise_power_basic() {
        assert_eq!(
            elementwise_power(&[2.0f32, 3.0, 4.0], 2.0),
            vec![4.0, 9.0, 16.0]
        );
        assert_eq!(elementwise_power_f64(&[4.0, 9.0], 0.5), vec![2.0, 3.0]);
    }
}
