//! DBSCAN density-based clustering over a precomputed distance matrix.
//!
//! [`dbscan`] implements the classic Ester et al. (1996) algorithm: points with
//! at least `min_points` neighbors within radius `eps` are **core** points;
//! density-reachable points form clusters; everything else is **noise**. It is
//! exact (consumer matrices are small, 10²–10³ points) and **deterministic** —
//! points are processed in index order and clusters expand in a fixed worklist
//! order, so a border point reachable from two clusters always lands in the
//! first one that reaches it.
//!
//! ## Input
//!
//! A **precomputed** N×N distance matrix (the caller supplies it — embeddings
//! and vectors aren't this crate's to compute; pair it with the
//! `util/pairwise_distance` matrix op). `distances[i][j]` is the distance from
//! point `i` to point `j`. The matrix is assumed symmetric; only row `p` is
//! read when querying point `p`'s neighborhood.
//!
//! ## Output
//!
//! A label per point: **`-1` = noise**, **`1, 2, 3, …` = cluster id**
//! (1-based, contiguous, assigned in the order clusters are discovered). A
//! point is always a member of its own `eps`-neighborhood, so `min_points`
//! counts the point itself (matching the scikit-learn `min_samples`
//! convention): with `min_points = 1` every point is core and there is no
//! noise.

use crate::error::ClusterError;

/// Internal sentinel: a point not yet assigned to a cluster or to noise.
/// Distinct from the public labels (`-1` noise, `1..` cluster id).
const UNCLASSIFIED: i32 = 0;
/// Public label for a noise point (not density-reachable from any core point).
const NOISE: i32 = -1;

/// Cluster a precomputed distance matrix with DBSCAN.
///
/// Returns one label per point: `-1` for noise, `1..` for cluster membership
/// (see the module docs). Fails loudly (see [`ClusterError`]) on a malformed
/// matrix (empty, non-square, non-finite or negative entry) or parameter
/// (`eps` non-finite/negative, `min_points` zero) rather than clustering
/// garbage.
pub fn dbscan(
    distances: &[Vec<f64>],
    eps: f64,
    min_points: usize,
) -> Result<Vec<i32>, ClusterError> {
    let n = distances.len();
    if n == 0 {
        return Err(ClusterError::Empty);
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(ClusterError::InvalidEps);
    }
    if min_points == 0 {
        return Err(ClusterError::InvalidMinPoints);
    }
    for (i, row) in distances.iter().enumerate() {
        if row.len() != n {
            return Err(ClusterError::NotSquare(format!(
                "row {i} has length {} (expected {n})",
                row.len()
            )));
        }
        for &d in row {
            if !d.is_finite() {
                return Err(ClusterError::NonFiniteDistance);
            }
            if d < 0.0 {
                return Err(ClusterError::NegativeDistance);
            }
        }
    }

    let mut labels = vec![UNCLASSIFIED; n];
    // Per-point generation stamp: the `cluster_id` that last enqueued the point
    // during expansion. Because `cluster_id` is 1-based and strictly increasing,
    // it doubles as a "queued this round" marker with no per-cluster
    // reallocation or reset — and lets `expand_cluster` push each point at most
    // once, keeping the worklist O(N) instead of O(N²) on dense data. `0` means
    // "never queued".
    let mut enqueued = vec![0i32; n];
    let mut cluster_id = 0i32;
    for p in 0..n {
        if labels[p] != UNCLASSIFIED {
            continue;
        }
        let neighbors = region_query(distances, p, eps);
        if neighbors.len() < min_points {
            // Too sparse to seed a cluster. Mark noise for now; a later
            // expansion may still reclaim it as a border point.
            labels[p] = NOISE;
            continue;
        }
        cluster_id += 1;
        expand_cluster(
            distances,
            &mut labels,
            &mut enqueued,
            neighbors,
            cluster_id,
            eps,
            min_points,
        );
    }
    Ok(labels)
}

/// Indices of every point in `p`'s `eps`-neighborhood, including `p` itself (a
/// point is always a member of its own neighborhood, so this holds even if the
/// matrix diagonal is not exactly zero).
fn region_query(distances: &[Vec<f64>], p: usize, eps: f64) -> Vec<usize> {
    (0..distances.len())
        .filter(|&q| q == p || distances[p][q] <= eps)
        .collect()
}

/// Grow cluster `cluster_id` from a core point's neighborhood via a worklist.
/// Density-reachable points are pulled into the cluster; those that are
/// themselves core contribute their neighbors. Points previously marked noise
/// are reclaimed as border points (added to the cluster but not expanded, since
/// noise points are never core).
///
/// `enqueued` stamps each point with the cluster currently being grown so a
/// point is pushed onto the worklist at most once per expansion: without it a
/// point in many core points' neighborhoods (dense data, large `eps`) could be
/// enqueued O(N) times, blowing the worklist up to O(N²). The `region_query`
/// rows are scanned in ascending index order and pushes preserve that order, so
/// the result stays deterministic.
fn expand_cluster(
    distances: &[Vec<f64>],
    labels: &mut [i32],
    enqueued: &mut [i32],
    seeds: Vec<usize>,
    cluster_id: i32,
    eps: f64,
    min_points: usize,
) {
    // `seeds` may grow as core points are found; index forward through it.
    let mut queue = seeds;
    // Stamp the seeds so a later core point doesn't re-enqueue them.
    for &s in &queue {
        enqueued[s] = cluster_id;
    }
    let mut i = 0;
    while i < queue.len() {
        let q = queue[i];
        i += 1;
        if labels[q] == NOISE {
            // Border point: density-reachable but not core. Claim it; don't
            // expand it (it has too few neighbors to be core).
            labels[q] = cluster_id;
            continue;
        }
        if labels[q] != UNCLASSIFIED {
            // Already assigned to this (or, for the seed itself, a) cluster.
            continue;
        }
        labels[q] = cluster_id;
        let q_neighbors = region_query(distances, q, eps);
        if q_neighbors.len() >= min_points {
            for qn in q_neighbors {
                if enqueued[qn] != cluster_id {
                    enqueued[qn] = cluster_id;
                    queue.push(qn);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a full euclidean distance matrix from 2-D points.
    fn distance_matrix(points: &[(f64, f64)]) -> Vec<Vec<f64>> {
        points
            .iter()
            .map(|&(ax, ay)| {
                points
                    .iter()
                    .map(|&(bx, by)| ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn two_blobs_plus_noise() {
        // Two tight blobs far apart, plus one lone outlier.
        let points = [
            (0.0, 0.0),
            (0.1, 0.0),
            (0.0, 0.1), // blob A (indices 0..=2)
            (10.0, 10.0),
            (10.1, 10.0),
            (10.0, 10.1),   // blob B (indices 3..=5)
            (100.0, 100.0), // outlier (index 6)
        ];
        let d = distance_matrix(&points);
        let labels = dbscan(&d, 0.5, 2).unwrap();

        // Blob A shares one label; blob B another; they differ; outlier noise.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
        assert_eq!(labels[6], -1);
        // Two clusters discovered → ids 1 and 2 (1-based, contiguous).
        assert_eq!(labels[0], 1);
        assert_eq!(labels[3], 2);
    }

    #[test]
    fn eps_too_small_is_all_noise() {
        let points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
        let d = distance_matrix(&points);
        // No two points are within 0.5; with min_points=2 every neighborhood is
        // just {self} (size 1) → all noise.
        let labels = dbscan(&d, 0.5, 2).unwrap();
        assert!(labels.iter().all(|&l| l == -1), "labels = {labels:?}");
    }

    #[test]
    fn min_points_one_makes_every_point_its_own_cluster() {
        let points = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)];
        let d = distance_matrix(&points);
        // eps small so nobody connects; min_points=1 → every point is core →
        // no noise, three singleton clusters.
        let labels = dbscan(&d, 0.5, 1).unwrap();
        assert!(!labels.contains(&-1));
        let distinct: std::collections::BTreeSet<i32> = labels.iter().copied().collect();
        assert_eq!(distinct.len(), 3);
        assert_eq!(distinct, [1, 2, 3].into_iter().collect());
    }

    #[test]
    fn all_points_one_cluster() {
        let points = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.0), (0.1, 0.2)];
        let d = distance_matrix(&points);
        let labels = dbscan(&d, 1.0, 2).unwrap();
        assert!(labels.iter().all(|&l| l == 1), "labels = {labels:?}");
    }

    #[test]
    fn dense_fully_connected_blob_is_one_cluster() {
        // Every point within eps of every other (heavy neighborhood overlap):
        // exercises the worklist-dedup path — each point is enqueued once, and
        // the whole blob collapses to a single cluster.
        let points: Vec<(f64, f64)> = (0..20).map(|k| (k as f64 * 0.01, 0.0)).collect();
        let d = distance_matrix(&points);
        let labels = dbscan(&d, 1.0, 3).unwrap();
        assert!(labels.iter().all(|&l| l == 1), "labels = {labels:?}");
    }

    #[test]
    fn border_point_joins_a_cluster_deterministically() {
        // A chain 0—1—2 with eps that links neighbors but min_points=3 so only
        // the middle point (1) is core (neighbors {0,1,2}); 0 and 2 are border
        // points reclaimed into the cluster. 3 is a far outlier → noise.
        let points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (50.0, 0.0)];
        let d = distance_matrix(&points);
        let labels = dbscan(&d, 1.0, 3).unwrap();
        assert_eq!(labels[0], 1);
        assert_eq!(labels[1], 1);
        assert_eq!(labels[2], 1);
        assert_eq!(labels[3], -1);
    }

    #[test]
    fn rejects_empty_matrix() {
        assert_eq!(dbscan(&[], 1.0, 2), Err(ClusterError::Empty));
    }

    #[test]
    fn rejects_non_square() {
        let d = vec![vec![0.0, 1.0], vec![1.0]];
        assert!(matches!(
            dbscan(&d, 1.0, 2),
            Err(ClusterError::NotSquare(_))
        ));
    }

    #[test]
    fn rejects_non_finite_distance() {
        let d = vec![vec![0.0, f64::NAN], vec![f64::NAN, 0.0]];
        assert_eq!(dbscan(&d, 1.0, 2), Err(ClusterError::NonFiniteDistance));
        let d = vec![vec![0.0, f64::INFINITY], vec![f64::INFINITY, 0.0]];
        assert_eq!(dbscan(&d, 1.0, 2), Err(ClusterError::NonFiniteDistance));
    }

    #[test]
    fn rejects_negative_distance() {
        let d = vec![vec![0.0, -1.0], vec![-1.0, 0.0]];
        assert_eq!(dbscan(&d, 1.0, 2), Err(ClusterError::NegativeDistance));
    }

    #[test]
    fn rejects_invalid_eps() {
        let d = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        assert_eq!(dbscan(&d, -1.0, 2), Err(ClusterError::InvalidEps));
        assert_eq!(dbscan(&d, f64::NAN, 2), Err(ClusterError::InvalidEps));
        assert_eq!(dbscan(&d, f64::INFINITY, 2), Err(ClusterError::InvalidEps));
    }

    #[test]
    fn rejects_zero_min_points() {
        let d = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        assert_eq!(dbscan(&d, 1.0, 0), Err(ClusterError::InvalidMinPoints));
    }
}
