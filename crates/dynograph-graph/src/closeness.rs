//! Closeness centrality — how short the shortest paths from a node to the rest
//! of the graph are.
//!
//! For each node `s` we run single-source shortest paths (BFS unweighted,
//! Dijkstra weighted) along **outgoing** edges and score
//! `(reachable / (n-1)) * (reachable / sum_of_distances)` — the Wasserman-Faust
//! variant, which behaves sensibly on disconnected graphs by scaling down nodes
//! that can only reach part of the graph. Weighted runs treat edge weight as a
//! **cost** and require strictly positive weights.

use crate::error::GraphError;
use crate::graph::Graph;
use crate::paths::{require_positive_costs, shortest_paths};

/// Compute closeness centrality for every node. An unreachable or isolated node
/// scores 0. An empty graph (and the trivial one-node graph) returns all zeros.
///
/// Errors: [`GraphError::InvalidWeight`] if `weighted` and any edge weight is
/// not strictly positive (Dijkstra costs must be > 0).
pub fn closeness_centrality(graph: &Graph, weighted: bool) -> Result<Vec<f64>, GraphError> {
    let n = graph.node_count();
    let mut scores = vec![0.0; n];
    if n <= 1 {
        return Ok(scores);
    }
    if weighted {
        require_positive_costs(graph)?;
    }

    for (s, slot) in scores.iter_mut().enumerate() {
        let sssp = shortest_paths(graph, s, weighted);
        let mut sum = 0.0;
        let mut reachable = 0usize;
        for (v, &d) in sssp.dist.iter().enumerate() {
            if v != s && d.is_finite() {
                sum += d;
                reachable += 1;
            }
        }
        if sum > 0.0 {
            let r = reachable as f64;
            *slot = (r / (n as f64 - 1.0)) * (r / sum);
        }
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    #[test]
    fn empty_and_singleton_are_zero() {
        let g = GraphBuilder::new().build(false);
        assert!(closeness_centrality(&g, false).unwrap().is_empty());

        let mut b = GraphBuilder::new();
        b.add_node("solo");
        let g = b.build(false);
        assert_eq!(closeness_centrality(&g, false).unwrap(), vec![0.0]);
    }

    #[test]
    fn path_center_is_closest() {
        // Undirected path a-b-c-d. b and c (interior) are closer to everything
        // than the endpoints a and d.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        b.add_edge("c", "d", 1.0).unwrap();
        let g = b.build(false);
        let c = closeness_centrality(&g, false).unwrap();
        let (a, bb, cc, d) = (
            c[g.idx_of("a").unwrap()],
            c[g.idx_of("b").unwrap()],
            c[g.idx_of("c").unwrap()],
            c[g.idx_of("d").unwrap()],
        );
        assert!(bb > a && cc > d, "interior should beat endpoints: {c:?}");
        assert!((a - d).abs() < 1e-9 && (bb - cc).abs() < 1e-9, "symmetry");
    }

    #[test]
    fn star_center_has_max_closeness() {
        // Undirected star: hub reaches all at distance 1 => closeness 1.0
        // (Wasserman-Faust: reachable=n-1, sum=n-1 => (1)*(1)=1).
        let mut b = GraphBuilder::new();
        b.add_edge("hub", "a", 1.0).unwrap();
        b.add_edge("hub", "b", 1.0).unwrap();
        b.add_edge("hub", "c", 1.0).unwrap();
        let g = b.build(false);
        let c = closeness_centrality(&g, false).unwrap();
        assert!((c[g.idx_of("hub").unwrap()] - 1.0).abs() < 1e-9, "{c:?}");
        // A leaf reaches hub at 1 and the two other leaves at 2.
        let leaf = c[g.idx_of("a").unwrap()];
        assert!(leaf < 1.0 && leaf > 0.0);
    }

    #[test]
    fn disconnected_component_scales_down() {
        // a-b connected; c isolated. a can only reach 1 of 2 others, so the
        // Wasserman-Faust factor (reachable/(n-1)) penalizes it below 1.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_node("c");
        let g = b.build(false);
        let c = closeness_centrality(&g, false).unwrap();
        // a: reachable=1, n-1=2, sum=1 => (1/2)*(1/1)=0.5
        assert!((c[g.idx_of("a").unwrap()] - 0.5).abs() < 1e-9, "{c:?}");
        assert_eq!(c[g.idx_of("c").unwrap()], 0.0);
    }

    #[test]
    fn weighted_uses_costs() {
        // a-b cost 1, b-c cost 1, a-c cost 5. From a: dist b=1, c=2 (via b, not
        // the direct cost-5 edge).
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        b.add_edge("a", "c", 5.0).unwrap();
        let g = b.build(false);
        let c = closeness_centrality(&g, true).unwrap();
        // a: sum = 1 (b) + 2 (c) = 3, reachable 2, n-1=2 => (2/2)*(2/3)=0.6667
        assert!(
            (c[g.idx_of("a").unwrap()] - 2.0 / 3.0).abs() < 1e-9,
            "{c:?}"
        );
    }

    #[test]
    fn non_positive_weight_rejected_when_weighted() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 0.0).unwrap();
        let g = b.build(false);
        assert!(matches!(
            closeness_centrality(&g, true),
            Err(GraphError::InvalidWeight(_))
        ));
    }
}
