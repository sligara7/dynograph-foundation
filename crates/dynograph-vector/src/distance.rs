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

/// Euclidean distance between two f32 vectors.
///
/// Same 8-wide unrolled accumulator structure as [`dot_product`] so it
/// autovectorizes under the same `RUSTFLAGS` (see the module docstring);
/// without those flags it is correct scalar code.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
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

    sum.sqrt()
}

/// Element-wise (Hadamard) product of two f32 vectors.
///
/// Returns a new vector `c` where `c[i] = a[i] * b[i]`.
#[inline]
pub fn hadamard(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
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

/// Euclidean distance between two f64 vectors.
#[inline]
pub fn euclidean_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

/// Element-wise (Hadamard) product of two f64 vectors.
#[inline]
pub fn hadamard_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
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
}
