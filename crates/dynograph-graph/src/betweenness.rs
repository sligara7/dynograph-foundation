//! Betweenness centrality via Brandes' algorithm — how often a node lies on
//! shortest paths between other nodes.
//!
//! Exact (consumer graphs are small). For each source we compute single-source
//! shortest paths (BFS unweighted, Dijkstra weighted) and accumulate dependency
//! contributions back along the shortest-path DAG. Weighted runs treat edge
//! weight as a **cost** and require strictly positive weights.
//!
//! Rescaling matches the standard convention (and NetworkX): when `normalized`,
//! divide by the number of ordered pairs `(n-1)(n-2)` — which also cancels the
//! double-counting of undirected pairs; when not normalized, halve undirected
//! scores so each unordered pair is counted once.

use crate::error::GraphError;
use crate::graph::Graph;
use crate::paths::{require_positive_costs, shortest_paths};

/// Compute betweenness centrality for every node.
///
/// Errors: [`GraphError::InvalidWeight`] if `weighted` and any edge weight is
/// not strictly positive. An empty graph returns an empty vector.
pub fn betweenness_centrality(
    graph: &Graph,
    weighted: bool,
    normalized: bool,
) -> Result<Vec<f64>, GraphError> {
    let n = graph.node_count();
    if n == 0 {
        return Ok(Vec::new());
    }
    if weighted {
        require_positive_costs(graph)?;
    }

    let mut cb = vec![0.0; n];
    for s in 0..n {
        let sssp = shortest_paths(graph, s, weighted);
        // Brandes dependency accumulation, walking nodes in reverse
        // finalization (non-increasing distance) order.
        let mut delta = vec![0.0; n];
        for &w in sssp.order.iter().rev() {
            for &v in &sssp.preds[w] {
                delta[v] += (sssp.sigma[v] / sssp.sigma[w]) * (1.0 + delta[w]);
            }
            if w != s {
                cb[w] += delta[w];
            }
        }
    }

    rescale(&mut cb, n, normalized, graph.is_directed());
    Ok(cb)
}

/// Standard Brandes rescaling. `normalized` divides by the count of ordered
/// endpoint pairs `(n-1)(n-2)` (which simultaneously removes undirected
/// double-counting); otherwise undirected scores are halved.
fn rescale(cb: &mut [f64], n: usize, normalized: bool, directed: bool) {
    if normalized {
        if n > 2 {
            let scale = 1.0 / ((n - 1) as f64 * (n - 2) as f64);
            for c in cb.iter_mut() {
                *c *= scale;
            }
        }
        // n <= 2: every score is already 0; nothing to scale.
    } else if !directed {
        for c in cb.iter_mut() {
            *c *= 0.5;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    fn undirected_path_abc() -> Graph {
        // a - b - c. b is on the only shortest path between a and c.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        b.build(false)
    }

    #[test]
    fn empty_graph_returns_empty() {
        let g = GraphBuilder::new().build(false);
        assert!(betweenness_centrality(&g, false, true).unwrap().is_empty());
    }

    #[test]
    fn path_center_unnormalized() {
        let g = undirected_path_abc();
        let cb = betweenness_centrality(&g, false, false).unwrap();
        // Only the pair (a,c) has a between node, b. Unordered count once => 1.
        assert!((cb[g.idx_of("b").unwrap()] - 1.0).abs() < 1e-9, "{cb:?}");
        assert!((cb[g.idx_of("a").unwrap()]).abs() < 1e-9);
        assert!((cb[g.idx_of("c").unwrap()]).abs() < 1e-9);
    }

    #[test]
    fn path_center_normalized() {
        let g = undirected_path_abc();
        let cb = betweenness_centrality(&g, false, true).unwrap();
        // n=3: scale = 1/((3-1)(3-2)) = 1/2. Raw ordered count for b is 2
        // (a->c and c->a), so normalized = 2 * 1/2 = 1.0.
        assert!((cb[g.idx_of("b").unwrap()] - 1.0).abs() < 1e-9, "{cb:?}");
    }

    #[test]
    fn star_center_is_maximal() {
        // Undirected star: hub lies between every pair of the 3 leaves.
        let mut b = GraphBuilder::new();
        b.add_edge("hub", "a", 1.0).unwrap();
        b.add_edge("hub", "b", 1.0).unwrap();
        b.add_edge("hub", "c", 1.0).unwrap();
        let g = b.build(false);
        let cb = betweenness_centrality(&g, false, false).unwrap();
        // 3 unordered leaf pairs all pass through hub => 3.0.
        assert!((cb[g.idx_of("hub").unwrap()] - 3.0).abs() < 1e-9, "{cb:?}");
        for leaf in ["a", "b", "c"] {
            assert!(cb[g.idx_of(leaf).unwrap()].abs() < 1e-9);
        }
    }

    #[test]
    fn parallel_shortest_paths_split_credit() {
        // Square a-b, a-c, b-d, c-d (undirected). Between a and d there are two
        // equal shortest paths (via b, via c), so b and c each get half credit.
        let mut bld = GraphBuilder::new();
        bld.add_edge("a", "b", 1.0).unwrap();
        bld.add_edge("a", "c", 1.0).unwrap();
        bld.add_edge("b", "d", 1.0).unwrap();
        bld.add_edge("c", "d", 1.0).unwrap();
        let g = bld.build(false);
        let cb = betweenness_centrality(&g, false, false).unwrap();
        let b = cb[g.idx_of("b").unwrap()];
        let c = cb[g.idx_of("c").unwrap()];
        // Each of b,c lies on one of the two a<->d shortest paths => 0.5 each.
        assert!((b - 0.5).abs() < 1e-9, "b={b} cb={cb:?}");
        assert!((c - 0.5).abs() < 1e-9, "c={c} cb={cb:?}");
    }

    #[test]
    fn weighted_detour_changes_betweenness() {
        // a-b(1) b-c(1) a-c(1): unweighted, a-c is direct so b is NOT between.
        // Weighted a-c cost 5 forces a->b->c, putting b between a and c.
        let mut bld = GraphBuilder::new();
        bld.add_edge("a", "b", 1.0).unwrap();
        bld.add_edge("b", "c", 1.0).unwrap();
        bld.add_edge("a", "c", 5.0).unwrap();
        let g = bld.build(false);
        let unweighted = betweenness_centrality(&g, false, false).unwrap();
        assert!(unweighted[g.idx_of("b").unwrap()].abs() < 1e-9);
        let weighted = betweenness_centrality(&g, true, false).unwrap();
        assert!(
            (weighted[g.idx_of("b").unwrap()] - 1.0).abs() < 1e-9,
            "{weighted:?}"
        );
    }

    #[test]
    fn non_positive_weight_rejected_when_weighted() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", -1.0).unwrap();
        let g = b.build(false);
        assert!(matches!(
            betweenness_centrality(&g, true, false),
            Err(GraphError::InvalidWeight(_))
        ));
    }
}
