//! Single-pair shortest path: the route and distance between two nodes.
//!
//! Reuses the shared SSSP primitive (BFS unweighted, Dijkstra weighted) from
//! `paths`, then reconstructs one shortest route by walking the predecessor
//! chain back from the target. Follows **outgoing** edges (directed); on an
//! undirected graph that is every incident edge. Weighted runs treat the edge
//! weight as a **cost** and require strictly positive weights.

use crate::error::GraphError;
use crate::graph::Graph;
use crate::paths::{require_positive_costs, shortest_paths};

/// A shortest path from a source to a target.
#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    /// Node indices in order, `source` first and `target` last.
    pub nodes: Vec<usize>,
    /// Total path cost — hop count when unweighted, summed edge cost otherwise.
    pub distance: f64,
}

/// Find one shortest path from `source` to `target`. Returns `Ok(None)` when the
/// target is unreachable, and a single-node path of distance 0 when
/// `source == target`.
///
/// Errors: [`GraphError::InvalidWeight`] if `weighted` and any edge weight is
/// not strictly positive.
pub fn shortest_path(
    graph: &Graph,
    source: usize,
    target: usize,
    weighted: bool,
) -> Result<Option<Path>, GraphError> {
    if weighted {
        require_positive_costs(graph)?;
    }
    if source == target {
        return Ok(Some(Path {
            nodes: vec![source],
            distance: 0.0,
        }));
    }

    let sssp = shortest_paths(graph, source, weighted);
    if !sssp.dist[target].is_finite() {
        return Ok(None);
    }

    // Walk one predecessor chain back to the source. Predecessors strictly
    // decrease distance, so the walk always terminates at the source.
    let mut nodes = vec![target];
    let mut cur = target;
    while cur != source {
        cur = sssp.preds[cur][0];
        nodes.push(cur);
    }
    nodes.reverse();
    Ok(Some(Path {
        nodes,
        distance: sssp.dist[target],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    fn ids(g: &crate::graph::Graph, path: &Path) -> Vec<String> {
        path.nodes.iter().map(|&i| g.id_of(i).to_string()).collect()
    }

    #[test]
    fn source_equals_target_is_trivial() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        let g = b.build(true);
        let a = g.idx_of("a").unwrap();
        let p = shortest_path(&g, a, a, false).unwrap().unwrap();
        assert_eq!(p.nodes, vec![a]);
        assert_eq!(p.distance, 0.0);
    }

    #[test]
    fn directed_chain_path() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        let g = b.build(true);
        let p = shortest_path(&g, g.idx_of("a").unwrap(), g.idx_of("c").unwrap(), false)
            .unwrap()
            .unwrap();
        assert_eq!(ids(&g, &p), vec!["a", "b", "c"]);
        assert_eq!(p.distance, 2.0);
    }

    #[test]
    fn unreachable_returns_none() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_node("c");
        let g = b.build(true);
        let p = shortest_path(&g, g.idx_of("a").unwrap(), g.idx_of("c").unwrap(), false).unwrap();
        assert!(p.is_none());
    }

    #[test]
    fn directed_respects_orientation() {
        // a -> b only; b cannot reach a.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        let g = b.build(true);
        assert!(
            shortest_path(&g, g.idx_of("b").unwrap(), g.idx_of("a").unwrap(), false)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn weighted_prefers_cheaper_detour() {
        // Direct a-c costs 5; a-b-c costs 2.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        b.add_edge("a", "c", 5.0).unwrap();
        let g = b.build(false);
        let p = shortest_path(&g, g.idx_of("a").unwrap(), g.idx_of("c").unwrap(), true)
            .unwrap()
            .unwrap();
        assert_eq!(ids(&g, &p), vec!["a", "b", "c"]);
        assert_eq!(p.distance, 2.0);
    }

    #[test]
    fn non_positive_weight_rejected_when_weighted() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 0.0).unwrap();
        let g = b.build(true);
        assert!(matches!(
            shortest_path(&g, g.idx_of("a").unwrap(), g.idx_of("b").unwrap(), true),
            Err(GraphError::InvalidWeight(_))
        ));
    }
}
