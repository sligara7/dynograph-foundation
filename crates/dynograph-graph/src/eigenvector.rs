//! Eigenvector centrality via power iteration.
//!
//! A node is important if it is pointed at by important nodes: the score vector
//! is the dominant eigenvector of the adjacency matrix. We find it by power
//! iteration on `(I + A)` — i.e. `x[v] = x[v] + sum over in-neighbors u of
//! weight(u,v) * x[u]` — renormalizing (L2) each step. The `+ I` shift moves the
//! spectrum by 1 so the dominant eigenvalue is strictly largest in magnitude;
//! without it, bipartite graphs (eigenvalues `±λ`) make plain power iteration
//! oscillate forever. The shift doesn't change the eigenvector, only guarantees
//! convergence. For an undirected graph `in_neighbors` is every incident edge,
//! giving standard symmetric-adjacency eigenvector centrality. Edge weights are
//! **strengths** and must be non-negative (Perron-Frobenius).
//!
//! Scores are normalized to unit L2 norm. The measure is undefined on a graph
//! with no positive-weight edges (no adjacency structure to rank), which
//! surfaces as [`GraphError::NotConverged`] rather than a fake uniform answer.
//!
//! **Intended for undirected (symmetric) graphs.** The `(I + A)` convergence
//! guarantee and the "doesn't change the eigenvector" property rely on `A` being
//! symmetric; on a directed graph the dominant-magnitude eigenvalue of `(I + A)`
//! need not be the Perron eigenvalue, so power iteration can converge to an
//! artifact (e.g. a DAG, whose true eigenvector centrality is degenerate). The
//! service layer only computes this on the undirected graph for that reason.
//! Even undirected, a graph with a degenerate dominant eigenspace (e.g. two
//! isomorphic disconnected components) has a non-unique eigenvector; the result
//! is then seed-dependent — a standard limitation of the measure.

use crate::error::GraphError;
use crate::graph::Graph;

/// Tuning for [`eigenvector_centrality`].
#[derive(Debug, Clone, Copy)]
pub struct EigenvectorConfig {
    /// Convergence threshold on the L1 change between iterations.
    pub tolerance: f64,
    /// Maximum iterations before giving up with [`GraphError::NotConverged`].
    pub max_iterations: usize,
}

impl Default for EigenvectorConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            max_iterations: 100,
        }
    }
}

/// Compute eigenvector centrality for every node (unit-L2-normalized).
///
/// Errors: [`GraphError::InvalidWeight`] on a negative edge weight, or
/// [`GraphError::NotConverged`] if the iteration budget is exhausted or the
/// measure is undefined (no edges to propagate importance). An empty graph
/// returns an empty vector.
pub fn eigenvector_centrality(
    graph: &Graph,
    config: &EigenvectorConfig,
) -> Result<Vec<f64>, GraphError> {
    let n = graph.node_count();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Reject negative weights (Perron-Frobenius assumes a non-negative matrix),
    // and require at least one POSITIVE-weight edge. An edgeless graph — or one
    // whose every edge weight is 0 (structurally edgeless for ranking) — has no
    // adjacency to propagate importance; the `+ I` shift would otherwise return
    // a meaningless uniform vector, so fail loud instead.
    let mut any_positive = false;
    for v in 0..n {
        for &(_, w) in graph.in_neighbors(v) {
            if w < 0.0 {
                return Err(GraphError::InvalidWeight(format!(
                    "eigenvector-centrality weights are strengths and must be non-negative, got {w}"
                )));
            }
            if w > 0.0 {
                any_positive = true;
            }
        }
    }
    if !any_positive {
        return Err(GraphError::NotConverged {
            iterations: config.max_iterations,
        });
    }

    let mut x = vec![1.0 / (n as f64).sqrt(); n];
    for _ in 0..config.max_iterations {
        // Power iteration on (I + A): start each component at its own current
        // value, then add the weighted contribution of its in-neighbors.
        let mut next = x.clone();
        for (v, slot) in next.iter_mut().enumerate() {
            let mut acc = 0.0;
            for &(u, w) in graph.in_neighbors(v) {
                acc += w * x[u];
            }
            *slot += acc;
        }
        let norm = next.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 {
            return Err(GraphError::NotConverged {
                iterations: config.max_iterations,
            });
        }
        for slot in next.iter_mut() {
            *slot /= norm;
        }

        let delta: f64 = (0..n).map(|i| (next[i] - x[i]).abs()).sum();
        x = next;
        if delta < config.tolerance {
            return Ok(x);
        }
    }
    Err(GraphError::NotConverged {
        iterations: config.max_iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    #[test]
    fn empty_graph_returns_empty() {
        let g = GraphBuilder::new().build(false);
        assert!(
            eigenvector_centrality(&g, &EigenvectorConfig::default())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn edgeless_graph_is_undefined() {
        let mut b = GraphBuilder::new();
        b.add_node("a");
        b.add_node("b");
        let g = b.build(false);
        assert!(matches!(
            eigenvector_centrality(&g, &EigenvectorConfig::default()),
            Err(GraphError::NotConverged { .. })
        ));
    }

    #[test]
    fn all_zero_weight_graph_is_undefined() {
        // Edges exist but every weight is 0 => structurally edgeless for ranking.
        // Must fail loud, not return a fake uniform vector.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 0.0).unwrap();
        b.add_edge("b", "c", 0.0).unwrap();
        let g = b.build(false);
        assert!(matches!(
            eigenvector_centrality(&g, &EigenvectorConfig::default()),
            Err(GraphError::NotConverged { .. })
        ));
    }

    #[test]
    fn symmetric_path_center_scores_highest() {
        // Undirected path a-b-c: b (the center) has the highest eigenvector
        // centrality; a and c are symmetric.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        let g = b.build(false);
        let ev = eigenvector_centrality(&g, &EigenvectorConfig::default()).unwrap();
        let cb = ev[g.idx_of("b").unwrap()];
        let ca = ev[g.idx_of("a").unwrap()];
        let cc = ev[g.idx_of("c").unwrap()];
        assert!(cb > ca && cb > cc, "center should dominate: {ev:?}");
        assert!((ca - cc).abs() < 1e-6, "endpoints should be symmetric");
    }

    #[test]
    fn triangle_is_uniform() {
        // Undirected triangle: all nodes equivalent => equal centrality.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        b.add_edge("c", "a", 1.0).unwrap();
        let g = b.build(false);
        let ev = eigenvector_centrality(&g, &EigenvectorConfig::default()).unwrap();
        for w in ev.windows(2) {
            assert!((w[0] - w[1]).abs() < 1e-6, "expected uniform: {ev:?}");
        }
    }

    #[test]
    fn negative_weight_is_rejected() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", -2.0).unwrap();
        let g = b.build(false);
        assert!(matches!(
            eigenvector_centrality(&g, &EigenvectorConfig::default()),
            Err(GraphError::InvalidWeight(_))
        ));
    }
}
