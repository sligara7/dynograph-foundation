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
/// Returns 0.0 if either vector has zero magnitude.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    let denom = norm_a * norm_b;
    if denom < f32::EPSILON {
        return 0.0;
    }
    // Clamp to [-1, 1] to handle floating point imprecision
    (dot / denom).clamp(-1.0, 1.0)
}

/// Euclidean distance between two f32 vectors.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensions must match");
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
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
/// Returns `0.0` if either vector has zero magnitude. Result is clamped
/// to `[-1.0, 1.0]` to handle floating-point imprecision.
#[inline]
pub fn cosine_similarity_f64(a: &[f64], b: &[f64]) -> f64 {
    let dot = dot_product_f64(a, b);
    let norm_a = l2_norm_f64(a);
    let norm_b = l2_norm_f64(b);
    let denom = norm_a * norm_b;
    if denom < f64::EPSILON {
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
