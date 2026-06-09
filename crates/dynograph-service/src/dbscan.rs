//! `POST /v1/util/dbscan` — DBSCAN density-based clustering.
//!
//! A stateless, domain-neutral pure-math endpoint (same shape as the
//! `util/pairwise_*` matrix ops and `util/game/analyze`): a precomputed
//! distance matrix goes in, a cluster label per point comes out. The math lives
//! in the `dynograph-cluster` crate; this module is the HTTP/wire seam.
//!
//! ## What it computes
//!
//! DBSCAN (Ester et al., 1996): points with at least `min_points` neighbors
//! within radius `eps` are core points; density-reachable points form clusters
//! of arbitrary shape; the rest is noise. It is exact and deterministic.
//!
//! ## Why `util/`, not `algo/`
//!
//! The input is a **precomputed distance matrix** the caller supplies — not the
//! stored graph's topology. It clusters *points by their pairwise distances*,
//! the same way `util/pairwise_distance` produces that matrix, so it belongs
//! beside the other stateless `util/*` math (no graph id, no `graph` feature
//! gate) rather than under `algo/*` (which operates on a stored graph). DBSCAN
//! is also distinct from the graph suite's Louvain (community detection on
//! edges) — density-on-points vs community-on-graph.
//!
//! ## Wire shape
//!
//! ```json
//! POST /v1/util/dbscan
//! {
//!   "distance_matrix": [[0.0, 0.3, 9.0],
//!                       [0.3, 0.0, 9.1],
//!                       [9.0, 9.1, 0.0]],
//!   "eps": 0.5,
//!   "min_points": 2
//! }
//! ```
//!
//! `distance_matrix` is row-major N×N and assumed symmetric. The response
//! `labels` has one entry per point in the same order: **`-1` = noise**,
//! **`1, 2, … ` = cluster id** (1-based). `num_clusters` is the count of
//! distinct cluster ids (noise excluded). A point is always in its own
//! `eps`-neighborhood, so `min_points` counts the point itself.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::registry::RegistryError;

/// Safety cap on the number of points (matrix rows). DBSCAN is O(N²) over the
/// matrix; cap it (fail loud, same posture as the game/algo caps) rather than
/// risk pinning the blocking worker on a pathological matrix. The narrative use
/// cases are small (10²–10³ points); 4096 is far past them — and the N×N body
/// already bounds N well below this via the request body-size limit.
pub(crate) const MAX_DBSCAN_POINTS: usize = 4096;

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct DbscanRequest {
    /// Row-major N×N precomputed distance matrix (assumed symmetric,
    /// non-negative, finite). The caller computes the distances.
    pub distance_matrix: Vec<Vec<f64>>,
    /// Neighborhood radius: two points are neighbors when their distance is
    /// `<= eps`. Finite and non-negative.
    #[schema(minimum = 0)]
    pub eps: f64,
    /// Minimum neighborhood size (including the point itself) for a point to be
    /// a core point. At least 1.
    #[schema(minimum = 1)]
    pub min_points: usize,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct DbscanResponse {
    /// One label per input point, in input order: `-1` = noise, `1..` =
    /// (1-based) cluster id.
    pub labels: Vec<i32>,
    /// Number of distinct clusters found (noise excluded).
    pub num_clusters: usize,
}

/// Validate the request, run DBSCAN, and shape the response. All failures are
/// 400s (the input is the caller's to fix).
pub(crate) fn run(req: DbscanRequest) -> Result<DbscanResponse, RegistryError> {
    if req.distance_matrix.len() > MAX_DBSCAN_POINTS {
        return Err(RegistryError::BadRequest(format!(
            "distance_matrix has more than the {MAX_DBSCAN_POINTS}-point limit"
        )));
    }
    let labels = dynograph_cluster::dbscan(&req.distance_matrix, req.eps, req.min_points)
        .map_err(|e| RegistryError::BadRequest(e.to_string()))?;
    // Cluster ids are 1-based and contiguous, so the count is the max id (0 if
    // every point is noise).
    let num_clusters = labels.iter().copied().max().unwrap_or(0).max(0) as usize;
    Ok(DbscanResponse {
        labels,
        num_clusters,
    })
}
