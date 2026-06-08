//! Directed cycle detection — is the graph a DAG, and if not, a witness cycle.
//!
//! An **iterative** colored DFS (white/gray/black): a back edge to a node still
//! on the current DFS path (gray) closes a cycle. The traversal is iterative so a
//! deep graph up to the service's 100k-node cap can't overflow the call stack.
//! Self-loops are dropped by `GraphBuilder`, so the smallest detectable cycle has
//! two distinct nodes.
//!
//! For the full set of nodes that lie on *some* cycle, use
//! [`strongly_connected_components`](crate::strongly_connected_components): a node
//! is on a cycle iff its SCC has more than one node. This module answers the
//! complementary "is there a cycle, and show me one".

use crate::graph::Graph;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Color {
    White,
    Gray,
    Black,
}

/// One DFS frame, replacing recursion.
#[derive(Clone, Copy)]
struct Frame {
    node: usize,
    child_idx: usize,
}

/// Find one directed cycle, if any. The returned node sequence `[v0, v1, .., vk]`
/// is a cycle closed by the edge `vk -> v0`. Returns `None` iff the graph is
/// acyclic (a DAG).
pub fn find_cycle(graph: &Graph) -> Option<Vec<usize>> {
    let n = graph.node_count();
    let mut color = vec![Color::White; n];
    let mut stack: Vec<Frame> = Vec::new();

    for start in 0..n {
        if color[start] != Color::White {
            continue;
        }
        color[start] = Color::Gray;
        stack.push(Frame {
            node: start,
            child_idx: 0,
        });

        while let Some(&Frame { node: u, child_idx }) = stack.last() {
            let neighbors = graph.out_neighbors(u);
            if child_idx < neighbors.len() {
                stack.last_mut().unwrap().child_idx += 1;
                let (v, _) = neighbors[child_idx];
                match color[v] {
                    Color::White => {
                        color[v] = Color::Gray;
                        stack.push(Frame {
                            node: v,
                            child_idx: 0,
                        });
                    }
                    Color::Gray => {
                        // Back edge to an ancestor on the current path: the cycle
                        // is the path from v up to u, closed by the edge u -> v.
                        let from = stack
                            .iter()
                            .position(|f| f.node == v)
                            .expect("gray on stack");
                        return Some(stack[from..].iter().map(|f| f.node).collect());
                    }
                    Color::Black => {} // fully explored; no cycle through v
                }
            } else {
                color[u] = Color::Black;
                stack.pop();
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, GraphBuilder};

    /// Assert `cycle` is a genuine directed cycle in `g` (consecutive edges plus
    /// the closing edge all exist).
    fn assert_valid_cycle(g: &Graph, cycle: &[usize]) {
        assert!(cycle.len() >= 2, "cycle too short: {cycle:?}");
        for i in 0..cycle.len() {
            let u = cycle[i];
            let v = cycle[(i + 1) % cycle.len()];
            assert!(
                g.out_neighbors(u).iter().any(|&(w, _)| w == v),
                "missing edge {u}->{v} in cycle {cycle:?}"
            );
        }
    }

    #[test]
    fn dag_has_no_cycle() {
        // Diamond a->b, a->c, b->d, c->d: acyclic.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("a", "c", 1.0).unwrap();
        b.add_edge("b", "d", 1.0).unwrap();
        b.add_edge("c", "d", 1.0).unwrap();
        let g = b.build(true);
        assert!(find_cycle(&g).is_none());
    }

    #[test]
    fn directed_triangle_has_cycle() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        b.add_edge("c", "a", 1.0).unwrap();
        let g = b.build(true);
        let cycle = find_cycle(&g).expect("expected a cycle");
        assert_eq!(cycle.len(), 3);
        assert_valid_cycle(&g, &cycle);
    }

    #[test]
    fn two_node_cycle_detected() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "a", 1.0).unwrap();
        let g = b.build(true);
        let cycle = find_cycle(&g).expect("expected a cycle");
        assert_valid_cycle(&g, &cycle);
    }

    #[test]
    fn cycle_in_one_of_several_components() {
        // Acyclic part x->y plus a cyclic part p->q->p.
        let mut b = GraphBuilder::new();
        b.add_edge("x", "y", 1.0).unwrap();
        b.add_edge("p", "q", 1.0).unwrap();
        b.add_edge("q", "p", 1.0).unwrap();
        let g = b.build(true);
        let cycle = find_cycle(&g).expect("expected a cycle");
        assert_valid_cycle(&g, &cycle);
    }

    #[test]
    fn undirected_edge_is_a_two_cycle() {
        // Built undirected, a-b is symmetric (a->b and b->a), i.e. a 2-cycle.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        let g = b.build(false);
        assert!(find_cycle(&g).is_some());
    }
}
