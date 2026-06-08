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

/// Arithmetic mean of a slice. `None` for an empty slice (mean is
/// undefined) rather than a silent `0.0`.
#[inline]
pub fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    Some(xs.iter().sum::<f64>() / xs.len() as f64)
}

/// Sample variance (Bessel-corrected, `n-1` denominator). `None` for
/// fewer than 2 elements. Use this for a sample drawn from a larger
/// population; the streaming counterpart lives in the service's
/// welford update.
#[inline]
pub fn variance(xs: &[f64]) -> Option<f64> {
    let n = xs.len();
    if n < 2 {
        return None;
    }
    let m = xs.iter().sum::<f64>() / n as f64;
    let ss: f64 = xs
        .iter()
        .map(|x| {
            let d = x - m;
            d * d
        })
        .sum();
    Some(ss / (n as f64 - 1.0))
}

/// Sample standard deviation — `sqrt` of [`variance`]. `None` for fewer
/// than 2 elements.
#[inline]
pub fn std_dev(xs: &[f64]) -> Option<f64> {
    variance(xs).map(f64::sqrt)
}

/// Linear-interpolated percentile `p` (in `0..=100`) — the NumPy
/// "linear" / type-7 definition. `None` for an empty slice or a `p`
/// outside `0..=100`. Sorts a copy, so input order is irrelevant. NaNs
/// don't order meaningfully — screen them upstream if possible.
pub fn percentile(xs: &[f64], p: f64) -> Option<f64> {
    if xs.is_empty() || !(0.0..=100.0).contains(&p) {
        return None;
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if s.len() == 1 {
        return Some(s[0]);
    }
    let rank = (p / 100.0) * (s.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return Some(s[lo]);
    }
    let frac = rank - lo as f64;
    Some(s[lo] + (s[hi] - s[lo]) * frac)
}

/// Median — the 50th [`percentile`] (linear interpolation for even
/// counts). `None` for an empty slice.
#[inline]
pub fn median(xs: &[f64]) -> Option<f64> {
    percentile(xs, 50.0)
}

/// Numerically-stable softmax: maps scores to a probability distribution
/// summing to 1, subtracting the max before exponentiating to avoid
/// overflow. Returns an empty vec for empty input. For finite inputs the
/// denominator is always `>= 1` (the max element contributes `e^0`), so
/// the result is a valid distribution; non-finite inputs propagate as
/// `NaN` rather than being silently "fixed."
pub fn softmax(xs: &[f64]) -> Vec<f64> {
    if xs.is_empty() {
        return Vec::new();
    }
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Spearman rank correlation between two equal-length slices — the
/// Pearson correlation of the *ranks*, so it captures monotonic (not
/// just linear) association and resists outliers. Tied values share
/// their average rank. `None` for fewer than 3 elements, mismatched
/// lengths, or a constant input (zero rank variance).
pub fn spearman_rank_correlation(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() < 3 || a.len() != b.len() {
        return None;
    }
    let ra = average_ranks(a);
    let rb = average_ranks(b);
    pearson_correlation(&ra, &rb)
}

/// Fractional ("average") ranks of `xs`: tied values share the mean of
/// the ranks they span (1-based; any affine shift cancels in the
/// downstream correlation).
fn average_ranks(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        xs[i]
            .partial_cmp(&xs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && xs[idx[j]] == xs[idx[i]] {
            j += 1;
        }
        // Positions i..j are tied; their ranks are (i+1)..=j, mean = (i+1+j)/2.
        let avg = (i + 1 + j) as f64 / 2.0;
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    ranks
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

    // --- mean / variance / std_dev ---

    #[test]
    fn mean_basic_and_empty() {
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
        assert_eq!(mean(&[]), None);
    }

    #[test]
    fn variance_and_std_dev_sample() {
        // Sample variance of [2,4,4,4,5,5,7,9] = 4.571428..., std ≈ 2.13809
        let xs = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = variance(&xs).unwrap();
        assert!((v - 32.0 / 7.0).abs() < 1e-12, "v={v}");
        assert!((std_dev(&xs).unwrap() - (32.0_f64 / 7.0).sqrt()).abs() < 1e-12);
        assert_eq!(variance(&[1.0]), None);
        assert_eq!(std_dev(&[]), None);
    }

    // --- percentile / median ---

    #[test]
    fn percentile_linear_interpolation() {
        let xs = vec![1.0, 2.0, 3.0, 4.0]; // type-7
        assert_eq!(percentile(&xs, 0.0), Some(1.0));
        assert_eq!(percentile(&xs, 100.0), Some(4.0));
        // p50 over [1,2,3,4]: rank = 0.5*3 = 1.5 → 2 + 0.5*(3-2) = 2.5
        assert_eq!(percentile(&xs, 50.0), Some(2.5));
        assert_eq!(median(&xs), Some(2.5));
        assert_eq!(median(&[5.0, 1.0, 3.0]), Some(3.0)); // odd, unsorted
    }

    #[test]
    fn percentile_rejects_empty_and_out_of_range() {
        assert_eq!(percentile(&[], 50.0), None);
        assert_eq!(percentile(&[1.0, 2.0], -1.0), None);
        assert_eq!(percentile(&[1.0, 2.0], 101.0), None);
        assert_eq!(median(&[]), None);
    }

    // --- softmax ---

    #[test]
    fn softmax_sums_to_one_and_orders() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "sum={sum}");
        assert!(p[0] < p[1] && p[1] < p[2]); // monotonic in input
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn softmax_is_overflow_stable() {
        // Large magnitudes would overflow a naive exp(); max-subtraction
        // keeps it finite and summing to 1.
        let p = softmax(&[1000.0, 1000.0]);
        assert!((p[0] - 0.5).abs() < 1e-12 && (p[1] - 0.5).abs() < 1e-12);
    }

    // --- spearman ---

    #[test]
    fn spearman_monotonic_nonlinear_is_one() {
        // y = x^3 is monotonic but not linear: Spearman = 1, Pearson < 1.
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|&v| v.powi(3)).collect();
        let s = spearman_rank_correlation(&x, &y).unwrap();
        assert!((s - 1.0).abs() < 1e-12, "spearman={s}");
        assert!(pearson_correlation(&x, &y).unwrap() < 0.99);
    }

    #[test]
    fn spearman_handles_ties() {
        let a = vec![1.0, 2.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 20.0, 30.0];
        assert!((spearman_rank_correlation(&a, &b).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn spearman_degenerate() {
        assert_eq!(spearman_rank_correlation(&[1.0, 2.0], &[1.0, 2.0]), None); // n<3
        assert_eq!(
            spearman_rank_correlation(&[1.0, 2.0, 3.0], &[5.0, 5.0, 5.0]),
            None
        ); // constant
    }
}
