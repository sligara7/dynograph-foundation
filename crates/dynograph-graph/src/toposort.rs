//! Topological sort of a **directed acyclic** graph via Kahn's algorithm.
//!
//! Repeatedly emits a node with no remaining in-edges. Among the currently
//! available nodes it always picks the smallest index (a min-heap), so the
//! result is the unique lexicographically-smallest topological order — stable
//! and reproducible. Returns `None` when the graph has a cycle (no valid order).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::graph::Graph;

/// Return one topological order (node indices), or `None` if the directed graph
/// contains a cycle. The order is the lexicographically-smallest valid one.
pub fn topological_sort(graph: &Graph) -> Option<Vec<usize>> {
    let n = graph.node_count();
    let mut in_degree = vec![0usize; n];
    for v in 0..n {
        for &(w, _) in graph.out_neighbors(v) {
            in_degree[w] += 1;
        }
    }

    // Min-heap of currently in-degree-0 nodes (smallest index first).
    let mut ready: BinaryHeap<Reverse<usize>> =
        (0..n).filter(|&v| in_degree[v] == 0).map(Reverse).collect();

    let mut order = Vec::with_capacity(n);
    while let Some(Reverse(v)) = ready.pop() {
        order.push(v);
        for &(w, _) in graph.out_neighbors(v) {
            in_degree[w] -= 1;
            if in_degree[w] == 0 {
                ready.push(Reverse(w));
            }
        }
    }

    // If some nodes never reached in-degree 0, they're in a cycle.
    if order.len() == n { Some(order) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, GraphBuilder};

    /// True if `order` lists every node once and respects every edge (u before v
    /// for each edge u->v).
    fn is_valid_topo(g: &Graph, order: &[usize]) -> bool {
        if order.len() != g.node_count() {
            return false;
        }
        let mut pos = vec![usize::MAX; g.node_count()];
        for (i, &v) in order.iter().enumerate() {
            pos[v] = i;
        }
        (0..g.node_count()).all(|u| g.out_neighbors(u).iter().all(|&(v, _)| pos[u] < pos[v]))
    }

    #[test]
    fn empty_graph_is_empty_order() {
        let g = GraphBuilder::new().build(true);
        assert_eq!(topological_sort(&g), Some(Vec::new()));
    }

    #[test]
    fn chain_orders_in_sequence() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        let g = b.build(true);
        let order = topological_sort(&g).unwrap();
        assert!(is_valid_topo(&g, &order));
        assert_eq!(
            order,
            vec![
                g.idx_of("a").unwrap(),
                g.idx_of("b").unwrap(),
                g.idx_of("c").unwrap()
            ]
        );
    }

    #[test]
    fn diamond_is_valid() {
        // a->b, a->c, b->d, c->d.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("a", "c", 1.0).unwrap();
        b.add_edge("b", "d", 1.0).unwrap();
        b.add_edge("c", "d", 1.0).unwrap();
        let g = b.build(true);
        let order = topological_sort(&g).unwrap();
        assert!(is_valid_topo(&g, &order), "{order:?}");
    }

    #[test]
    fn cycle_has_no_topological_order() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        b.add_edge("c", "a", 1.0).unwrap();
        let g = b.build(true);
        assert_eq!(topological_sort(&g), None);
    }

    #[test]
    fn ties_broken_by_smallest_index() {
        // r (idx 0) -> a (1) and r -> b (2). After r, both a and b are ready; the
        // min-heap emits the smaller index first, so the order is deterministic.
        let mut bld = GraphBuilder::new();
        bld.add_edge("r", "a", 1.0).unwrap();
        bld.add_edge("r", "b", 1.0).unwrap();
        let g = bld.build(true);
        // Independent (edgeless) nodes likewise come out in ascending index order.
        let order = topological_sort(&g).unwrap();
        assert_eq!(order, vec![0, 1, 2]);
        assert!(is_valid_topo(&g, &order));
    }
}
