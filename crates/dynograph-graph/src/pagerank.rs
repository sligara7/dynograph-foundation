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
    // Uniform teleport => standard PageRank.
    let teleport = vec![1.0 / n as f64; n];
    power_iteration(graph, &teleport, config)
}

/// Personalized PageRank (a.k.a. random walk with restart): like [`pagerank`],
/// but the `(1 - damping)` teleport mass returns to the `seeds` instead of
/// spreading uniformly, so scores measure relevance *to the seed set*. Seeds are
/// node indices; a repeated seed weights that node more heavily.
///
/// Errors: [`GraphError::InvalidParameter`] if `seeds` is empty, plus the same
/// weight / convergence errors as [`pagerank`]. An empty graph returns empty.
pub fn personalized_pagerank(
    graph: &Graph,
    seeds: &[usize],
    config: &PageRankConfig,
) -> Result<Vec<f64>, GraphError> {
    let n = graph.node_count();
    if n == 0 {
        return Ok(Vec::new());
    }
    if seeds.is_empty() {
        return Err(GraphError::InvalidParameter(
            "personalized PageRank requires at least one seed node".to_string(),
        ));
    }
    let mut teleport = vec![0.0; n];
    let share = 1.0 / seeds.len() as f64;
    for &s in seeds {
        if s >= n {
            return Err(GraphError::InvalidParameter(format!(
                "seed index {s} is out of range for a {n}-node graph"
            )));
        }
        teleport[s] += share;
    }
    power_iteration(graph, &teleport, config)
}

/// Shared power-iteration core. `teleport` is the restart distribution (must sum
/// to 1): uniform for plain PageRank, seed-concentrated for personalized. Both
/// the teleport term and the dangling-mass redistribution follow it, so the rank
/// vector stays a distribution summing to ~1.
fn power_iteration(
    graph: &Graph,
    teleport: &[f64],
    config: &PageRankConfig,
) -> Result<Vec<f64>, GraphError> {
    let n = graph.node_count();
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

    let mut rank = teleport.to_vec();
    for _ in 0..config.max_iterations {
        // Teleport term: (1-d) of the mass restarts per the teleport vector.
        let mut next: Vec<f64> = teleport.iter().map(|&t| (1.0 - d) * t).collect();
        let mut dangling = 0.0;
        for (u, (&rank_u, &strength_u)) in rank.iter().zip(out_strength.iter()).enumerate() {
            if strength_u == 0.0 {
                // No out-strength: hold the mass to redistribute via teleport.
                dangling += rank_u;
                continue;
            }
            let share = d * rank_u / strength_u;
            for &(v, w) in graph.out_neighbors(u) {
                next[v] += share * w;
            }
        }
        if dangling != 0.0 {
            let mass = d * dangling;
            for (slot, &t) in next.iter_mut().zip(teleport.iter()) {
                *slot += mass * t;
            }
        }

        let delta: f64 = next.iter().zip(&rank).map(|(a, b)| (a - b).abs()).sum();
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
    fn personalized_pagerank_requires_seeds() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        let g = b.build(true);
        assert!(matches!(
            personalized_pagerank(&g, &[], &PageRankConfig::default()),
            Err(GraphError::InvalidParameter(_))
        ));
    }

    #[test]
    fn personalized_pagerank_rejects_out_of_range_seed() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        let g = b.build(true);
        // Index 9 is out of range for a 2-node graph: fail loud, don't panic.
        assert!(matches!(
            personalized_pagerank(&g, &[9], &PageRankConfig::default()),
            Err(GraphError::InvalidParameter(_))
        ));
    }

    #[test]
    fn personalized_pagerank_decays_with_distance_from_seed() {
        // Undirected chain a-b-c-d. Seeding {a}, relevance should decay with
        // distance: closer nodes score higher than farther ones.
        let mut bld = GraphBuilder::new();
        bld.add_edge("a", "b", 1.0).unwrap();
        bld.add_edge("b", "c", 1.0).unwrap();
        bld.add_edge("c", "d", 1.0).unwrap();
        let g = bld.build(false);
        let a = g.idx_of("a").unwrap();
        let ppr = personalized_pagerank(&g, &[a], &PageRankConfig::default()).unwrap();
        let sum: f64 = ppr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "ppr should sum to 1: {sum}");
        let (pa, pb, pc, pd) = (
            ppr[a],
            ppr[g.idx_of("b").unwrap()],
            ppr[g.idx_of("c").unwrap()],
            ppr[g.idx_of("d").unwrap()],
        );
        // The seed region (a and its neighbor b) dominates the far region, and
        // relevance decays with distance from there (b > c > d). (a itself is not
        // the strict max — its degree-2 neighbor b accrues more flow.)
        assert!(pa > pc && pb > pc && pc > pd, "expected decay: {ppr:?}");
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
