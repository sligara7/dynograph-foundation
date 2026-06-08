//! PageRank via power iteration.
//!
//! Rank flows along **outgoing** edges (each node distributes its rank to the
//! nodes it points at), so a node accrues importance by being pointed at by
//! important nodes. Edge weights are treated as **strength**: a node splits its
//! rank among its out-edges in proportion to weight (equal split when
//! unweighted). Dangling nodes (no out-edges) redistribute their mass uniformly
//! so total rank is conserved at 1.

use crate::error::GraphError;
use crate::graph::Graph;

/// Tuning for [`pagerank`].
#[derive(Debug, Clone, Copy)]
pub struct PageRankConfig {
    /// Damping factor (probability of following an edge vs. teleporting).
    /// Standard value 0.85; must be in `[0, 1]`.
    pub damping: f64,
    /// Convergence threshold on the L1 change between iterations.
    pub tolerance: f64,
    /// Maximum iterations before giving up with [`GraphError::NotConverged`].
    pub max_iterations: usize,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            tolerance: 1e-6,
            max_iterations: 100,
        }
    }
}

/// Compute PageRank for every node, returning a rank vector that sums to ~1.
///
/// Errors: [`GraphError::InvalidWeight`] on a negative edge weight (strength
/// must be non-negative), or [`GraphError::NotConverged`] if the iteration
/// budget is exhausted. An empty graph returns an empty vector.
pub fn pagerank(graph: &Graph, config: &PageRankConfig) -> Result<Vec<f64>, GraphError> {
    let n = graph.node_count();
    if n == 0 {
        return Ok(Vec::new());
    }
    let nf = n as f64;
    let d = config.damping;

    // Weighted out-degree (strength) per node; reject negative weights.
    let mut out_strength = vec![0.0; n];
    for (u, slot) in out_strength.iter_mut().enumerate() {
        let mut sum = 0.0;
        for &(_, w) in graph.out_neighbors(u) {
            if w < 0.0 {
                return Err(GraphError::InvalidWeight(format!(
                    "PageRank weights are strengths and must be non-negative, got {w}"
                )));
            }
            sum += w;
        }
        *slot = sum;
    }

    let mut rank = vec![1.0 / nf; n];
    for _ in 0..config.max_iterations {
        let mut next = vec![(1.0 - d) / nf; n];
        let mut dangling = 0.0;
        for u in 0..n {
            if out_strength[u] == 0.0 {
                // No out-edges: hold the mass to redistribute uniformly.
                dangling += rank[u];
                continue;
            }
            let share = d * rank[u] / out_strength[u];
            for &(v, w) in graph.out_neighbors(u) {
                next[v] += share * w;
            }
        }
        let spread = d * dangling / nf;
        if spread != 0.0 {
            for slot in next.iter_mut() {
                *slot += spread;
            }
        }

        let delta: f64 = (0..n).map(|i| (next[i] - rank[i]).abs()).sum();
        rank = next;
        if delta < config.tolerance {
            return Ok(rank);
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
        let g = GraphBuilder::new().build(true);
        assert!(pagerank(&g, &PageRankConfig::default()).unwrap().is_empty());
    }

    #[test]
    fn ranks_sum_to_one() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        b.add_edge("c", "a", 1.0).unwrap();
        b.add_edge("a", "c", 1.0).unwrap();
        let g = b.build(true);
        let pr = pagerank(&g, &PageRankConfig::default()).unwrap();
        let sum: f64 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum was {sum}");
    }

    #[test]
    fn symmetric_cycle_is_uniform() {
        // a->b->c->a: a perfectly symmetric directed cycle => equal rank.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        b.add_edge("c", "a", 1.0).unwrap();
        let g = b.build(true);
        let pr = pagerank(&g, &PageRankConfig::default()).unwrap();
        for p in &pr {
            assert!((p - 1.0 / 3.0).abs() < 1e-6, "expected ~1/3, got {p}");
        }
    }

    #[test]
    fn hub_with_many_inlinks_ranks_highest() {
        // a,b,c all point at d; d points at a. d should rank highest.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "d", 1.0).unwrap();
        b.add_edge("b", "d", 1.0).unwrap();
        b.add_edge("c", "d", 1.0).unwrap();
        b.add_edge("d", "a", 1.0).unwrap();
        let g = b.build(true);
        let pr = pagerank(&g, &PageRankConfig::default()).unwrap();
        let d = pr[g.idx_of("d").unwrap()];
        for id in ["a", "b", "c"] {
            assert!(d > pr[g.idx_of(id).unwrap()], "d should outrank {id}");
        }
    }

    #[test]
    fn dangling_node_conserves_mass() {
        // c is dangling (no out-edges). Total must still sum to 1.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        let g = b.build(true);
        let pr = pagerank(&g, &PageRankConfig::default()).unwrap();
        let sum: f64 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum was {sum}");
    }

    #[test]
    fn negative_weight_is_rejected() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", -1.0).unwrap();
        let g = b.build(true);
        assert!(matches!(
            pagerank(&g, &PageRankConfig::default()),
            Err(GraphError::InvalidWeight(_))
        ));
    }

    #[test]
    fn too_few_iterations_does_not_converge() {
        // A hub graph starts far from its stationary distribution, so one
        // iteration from the uniform seed can't reach the tolerance (unlike a
        // symmetric cycle, which is already at its fixed point).
        let mut b = GraphBuilder::new();
        b.add_edge("a", "d", 1.0).unwrap();
        b.add_edge("b", "d", 1.0).unwrap();
        b.add_edge("c", "d", 1.0).unwrap();
        b.add_edge("d", "a", 1.0).unwrap();
        let g = b.build(true);
        let cfg = PageRankConfig {
            damping: 0.85,
            tolerance: 1e-12,
            max_iterations: 1,
        };
        assert!(matches!(
            pagerank(&g, &cfg),
            Err(GraphError::NotConverged { .. })
        ));
    }
}
