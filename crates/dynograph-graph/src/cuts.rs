//! Articulation points (cut vertices) and bridges (cut edges) — the structural
//! keystones whose removal disconnects the graph.
//!
//! Both come from one undirected DFS low-link pass (Tarjan). The traversal is
//! **iterative** (an explicit stack, not recursion) so a deep graph — up to the
//! service's 100k-node cap — can't overflow the call stack.
//!
//! Treats the graph as **undirected** and **simple**: it follows `out_neighbors`
//! as symmetric adjacency (the service builds undirected for this measure), and
//! parallel edges between the same pair are already merged by `GraphBuilder`, so
//! the standard single-parent-skip is correct. (Two genuinely parallel edges
//! would make their shared endpoints non-cut, but the merged graph can't see
//! that — documented, not a silent surprise.)

use crate::graph::Graph;

const UNVISITED: usize = usize::MAX;

/// The cut structure of a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cuts {
    /// Articulation-point node indices, ascending.
    pub articulation_points: Vec<usize>,
    /// Bridge edges as `(a, b)` index pairs with `a < b`, ascending.
    pub bridges: Vec<(usize, usize)>,
}

/// One DFS frame, replacing recursion: the node, its DFS-tree parent
/// (`UNVISITED` for a root), and the index of the next neighbor to visit.
#[derive(Clone, Copy)]
struct Frame {
    node: usize,
    parent: usize,
    child_idx: usize,
}

/// Compute articulation points and bridges in a single iterative DFS.
pub fn cut_structure(graph: &Graph) -> Cuts {
    let n = graph.node_count();
    let mut disc = vec![UNVISITED; n];
    let mut low = vec![0usize; n];
    let mut is_ap = vec![false; n];
    let mut bridges = Vec::new();
    let mut timer = 0usize;

    for start in 0..n {
        if disc[start] != UNVISITED {
            continue;
        }
        // Begin a DFS tree rooted at `start`.
        disc[start] = timer;
        low[start] = timer;
        timer += 1;
        let mut root_children = 0usize;
        let mut stack = vec![Frame {
            node: start,
            parent: UNVISITED,
            child_idx: 0,
        }];

        while let Some(&Frame {
            node: u,
            parent,
            child_idx,
        }) = stack.last()
        {
            let neighbors = graph.out_neighbors(u);
            if child_idx < neighbors.len() {
                stack.last_mut().unwrap().child_idx += 1;
                let (v, _) = neighbors[child_idx];
                if v == parent {
                    // Skip the single edge back to our DFS parent.
                    continue;
                }
                if disc[v] == UNVISITED {
                    // Tree edge: descend into v.
                    disc[v] = timer;
                    low[v] = timer;
                    timer += 1;
                    if u == start {
                        root_children += 1;
                    }
                    stack.push(Frame {
                        node: v,
                        parent: u,
                        child_idx: 0,
                    });
                } else {
                    // Back edge: v is an ancestor (or visited); tighten low[u].
                    low[u] = low[u].min(disc[v]);
                }
            } else {
                // u is finished. Propagate its low to its parent and test the
                // parent for the articulation / bridge conditions on edge p-u.
                let u_low = low[u];
                stack.pop();
                if let Some(&Frame { node: p, .. }) = stack.last() {
                    low[p] = low[p].min(u_low);
                    if p != start && u_low >= disc[p] {
                        is_ap[p] = true;
                    }
                    if u_low > disc[p] {
                        bridges.push((p.min(u), p.max(u)));
                    }
                }
            }
        }

        // A root is an articulation point iff it has more than one DFS child.
        if root_children > 1 {
            is_ap[start] = true;
        }
    }

    let articulation_points = (0..n).filter(|&v| is_ap[v]).collect();
    bridges.sort_unstable();
    Cuts {
        articulation_points,
        bridges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    fn idx(g: &Graph, id: &str) -> usize {
        g.idx_of(id).unwrap()
    }

    #[test]
    fn empty_graph_has_no_cuts() {
        let g = GraphBuilder::new().build(false);
        let c = cut_structure(&g);
        assert!(c.articulation_points.is_empty());
        assert!(c.bridges.is_empty());
    }

    #[test]
    fn path_center_is_articulation_all_edges_bridges() {
        // a - b - c: b is a cut vertex; both edges are bridges.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        let g = b.build(false);
        let c = cut_structure(&g);
        assert_eq!(c.articulation_points, vec![idx(&g, "b")]);
        let (a, bb, cc) = (idx(&g, "a"), idx(&g, "b"), idx(&g, "c"));
        let mut expected = vec![(a.min(bb), a.max(bb)), (bb.min(cc), bb.max(cc))];
        expected.sort_unstable();
        assert_eq!(c.bridges, expected);
    }

    #[test]
    fn triangle_has_no_cuts() {
        // A cycle is 2-edge- and 2-vertex-connected: no cut vertices or bridges.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        b.add_edge("c", "a", 1.0).unwrap();
        let g = b.build(false);
        let c = cut_structure(&g);
        assert!(c.articulation_points.is_empty(), "{c:?}");
        assert!(c.bridges.is_empty(), "{c:?}");
    }

    #[test]
    fn bridge_between_two_triangles() {
        // Triangle (a,b,c) and triangle (d,e,f), joined by a single edge c-d.
        // c-d is the only bridge; c and d are articulation points.
        let mut b = GraphBuilder::new();
        for (x, y) in [("a", "b"), ("b", "c"), ("c", "a")] {
            b.add_edge(x, y, 1.0).unwrap();
        }
        for (x, y) in [("d", "e"), ("e", "f"), ("f", "d")] {
            b.add_edge(x, y, 1.0).unwrap();
        }
        b.add_edge("c", "d", 1.0).unwrap();
        let g = b.build(false);
        let c = cut_structure(&g);
        let (ci, di) = (idx(&g, "c"), idx(&g, "d"));
        assert_eq!(c.articulation_points, {
            let mut v = vec![ci, di];
            v.sort_unstable();
            v
        });
        assert_eq!(c.bridges, vec![(ci.min(di), ci.max(di))]);
    }

    #[test]
    fn shared_vertex_is_articulation_without_bridges() {
        // Two triangles sharing vertex x: x is a cut vertex, but no bridges
        // (every edge is still inside a cycle).
        let mut b = GraphBuilder::new();
        for (p, q) in [("x", "a"), ("a", "b"), ("b", "x")] {
            b.add_edge(p, q, 1.0).unwrap();
        }
        for (p, q) in [("x", "c"), ("c", "d"), ("d", "x")] {
            b.add_edge(p, q, 1.0).unwrap();
        }
        let g = b.build(false);
        let c = cut_structure(&g);
        assert_eq!(c.articulation_points, vec![idx(&g, "x")]);
        assert!(c.bridges.is_empty(), "{c:?}");
    }

    #[test]
    fn disconnected_components_each_analyzed() {
        // Two separate paths; each middle node is an articulation point.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        b.add_edge("x", "y", 1.0).unwrap();
        b.add_edge("y", "z", 1.0).unwrap();
        let g = b.build(false);
        let c = cut_structure(&g);
        let mut expected = vec![idx(&g, "b"), idx(&g, "y")];
        expected.sort_unstable();
        assert_eq!(c.articulation_points, expected);
        assert_eq!(c.bridges.len(), 4);
    }
}
