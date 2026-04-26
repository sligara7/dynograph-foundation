//! SIMD-accelerated vector distance functions.
//!
//! Provides cosine similarity, dot product, and Euclidean distance
//! for f32 vectors. Uses auto-vectorization hints that the compiler
//! can lower to AVX2/NEON when available.

/// Dot product of two f32 slices.
///
/// The inner loop is structured for auto-vectorization — the compiler
/// will emit SIMD instructions (AVX2 on x86_64, NEON on aarch64)
/// when building with `-C target-cpu=native`.
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
        let a: Vec<f32> = (0..dim).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((i * 13 + 7) % 100) as f32 / 100.0).collect();
        let sim = cosine_similarity(&a, &b);
        assert!(sim >= -1.0 && sim <= 1.0);
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
}
