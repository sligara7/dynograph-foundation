//! Basic statistics over numeric slices.
//!
//! Provides `linear_regression_slope` and `pearson_correlation`. Both
//! return `Option<f64>` and yield `None` for degenerate inputs (insufficient
//! samples, zero variance, mismatched lengths) rather than silently
//! returning `0.0` — so the caller has to make the fallback choice
//! explicit.

/// Ordinary-least-squares slope of `y` regressed on `x` for a sequence
/// of `(x, y)` points.
///
/// Returns `None` if fewer than 2 points are supplied or the `x` values
/// have zero variance (vertical line — slope is undefined).
#[inline]
pub fn linear_regression_slope(points: &[(f64, f64)]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
    let sum_xx: f64 = points.iter().map(|(x, _)| x * x).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return None;
    }
    Some((n * sum_xy - sum_x * sum_y) / denom)
}

/// Pearson correlation coefficient between two equal-length f64 slices.
///
/// Returns `None` if the slices have fewer than 3 elements, mismatched
/// lengths, or either has zero variance.
#[inline]
pub fn pearson_correlation(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len();
    if n < 3 || n != b.len() {
        return None;
    }

    let n_f = n as f64;
    let mean_a: f64 = a.iter().sum::<f64>() / n_f;
    let mean_b: f64 = b.iter().sum::<f64>() / n_f;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if var_a == 0.0 || var_b == 0.0 {
        return None;
    }

    Some(cov / (var_a.sqrt() * var_b.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- linear_regression_slope ---

    #[test]
    fn linreg_perfect_positive() {
        let pts = vec![(0.0, 0.0), (1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
        let slope = linear_regression_slope(&pts).unwrap();
        assert!((slope - 2.0).abs() < 1e-12);
    }

    #[test]
    fn linreg_perfect_negative() {
        let pts = vec![(0.0, 10.0), (1.0, 8.0), (2.0, 6.0), (3.0, 4.0)];
        let slope = linear_regression_slope(&pts).unwrap();
        assert!((slope - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn linreg_horizontal() {
        let pts = vec![(0.0, 5.0), (1.0, 5.0), (2.0, 5.0)];
        let slope = linear_regression_slope(&pts).unwrap();
        assert!(slope.abs() < 1e-12);
    }

    #[test]
    fn linreg_with_noise() {
        // y = 3x + 1 + small noise; OLS should recover ~3
        let pts = vec![
            (1.0, 4.1),
            (2.0, 6.9),
            (3.0, 10.2),
            (4.0, 12.8),
            (5.0, 16.1),
        ];
        let slope = linear_regression_slope(&pts).unwrap();
        assert!((slope - 3.0).abs() < 0.2);
    }

    #[test]
    fn linreg_too_few_points() {
        assert_eq!(linear_regression_slope(&[]), None);
        assert_eq!(linear_regression_slope(&[(1.0, 2.0)]), None);
    }

    #[test]
    fn linreg_zero_x_variance() {
        // All x values identical — vertical line, slope undefined.
        let pts = vec![(1.0, 0.0), (1.0, 5.0), (1.0, 10.0)];
        assert_eq!(linear_regression_slope(&pts), None);
    }

    // --- pearson_correlation ---

    #[test]
    fn pearson_perfect_positive() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&a, &b).unwrap();
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_perfect_negative() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r = pearson_correlation(&a, &b).unwrap();
        assert!((r - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn pearson_weakly_correlated() {
        // Hand-computed: a=[1..5], b=[3,1,4,5,2]
        //   mean_a=3, mean_b=3
        //   cov=2, var_a=10, var_b=10
        //   r = 2 / sqrt(100) = 0.2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![3.0, 1.0, 4.0, 5.0, 2.0];
        let r = pearson_correlation(&a, &b).unwrap();
        assert!((r - 0.2).abs() < 1e-12);
    }

    #[test]
    fn pearson_known_value() {
        // Hand-computed: a=[1,2,3], b=[2,5,4]
        //   mean_a=2, mean_b=11/3
        //   cov=2, var_a=2, var_b=14/3
        //   r = 2/sqrt(28/3) = sqrt(3/7) ≈ 0.6546537
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 5.0, 4.0];
        let r = pearson_correlation(&a, &b).unwrap();
        let expected = (3.0_f64 / 7.0).sqrt();
        assert!((r - expected).abs() < 1e-12);
    }

    #[test]
    fn pearson_too_few_elements() {
        assert_eq!(pearson_correlation(&[], &[]), None);
        assert_eq!(pearson_correlation(&[1.0, 2.0], &[1.0, 2.0]), None);
    }

    #[test]
    fn pearson_mismatched_lengths() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(pearson_correlation(&a, &b), None);
    }

    #[test]
    fn pearson_zero_variance_a() {
        let a = vec![5.0, 5.0, 5.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(pearson_correlation(&a, &b), None);
    }

    #[test]
    fn pearson_zero_variance_b() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![7.0, 7.0, 7.0, 7.0];
        assert_eq!(pearson_correlation(&a, &b), None);
    }
}
