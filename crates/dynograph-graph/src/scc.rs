//! Strongly-connected components of a **directed** graph via Tarjan's algorithm.
//!
//! Two nodes are in the same SCC iff each is reachable from the other following
//! edge direction. The traversal is **iterative** (explicit stacks, not
//! recursion) so a deep graph — up to the service's 100k-node cap — can't
//! overflow the call stack. Result shape is the shared [`Components`] partition
//! (the same as weakly-connected components, just under directed reachability).
//!
//! On an undirected graph each connected component is trivially strongly
//! connected, so this coincides with `connected_components` there.

use crate::components::Components;
use crate::graph::Graph;

const UNVISITED: usize = usize::MAX;

/// Partition the graph into strongly-connected components.
pub fn strongly_connected_components(graph: &Graph) -> Components {
    let n = graph.node_count();
    let mut index = vec![UNVISITED; n]; // DFS discovery index
    let mut lowlink = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut labels = vec![UNVISITED; n];
    let mut scc_stack: Vec<usize> = Vec::new(); // nodes on the current SCC path
    let mut counter = 0usize;
    let mut count = 0usize;

    // Explicit DFS work stack of (node, next-neighbor-index).
    let mut work: Vec<(usize, usize)> = Vec::new();

    for start in 0..n {
        if index[start] != UNVISITED {
            continue;
        }
        index[start] = counter;
        lowlink[start] = counter;
        counter += 1;
        scc_stack.push(start);
        on_stack[start] = true;
        work.push((start, 0));

        while let Some(&(v, child_idx)) = work.last() {
            let neighbors = graph.out_neighbors(v);
            if child_idx < neighbors.len() {
                work.last_mut().unwrap().1 += 1;
                let (w, _) = neighbors[child_idx];
                if index[w] == UNVISITED {
                    // Tree edge: descend into w.
                    index[w] = counter;
                    lowlink[w] = counter;
                    counter += 1;
                    scc_stack.push(w);
                    on_stack[w] = true;
                    work.push((w, 0));
                } else if on_stack[w] {
                    // Back/cross edge to a node still on the current path.
                    lowlink[v] = lowlink[v].min(index[w]);
                }
            } else {
                // v finished. If it roots an SCC, pop the SCC off scc_stack.
                if lowlink[v] == index[v] {
                    loop {
                        let w = scc_stack.pop().expect("scc stack underflow");
                        on_stack[w] = false;
                        labels[w] = count;
                        if w == v {
                            break;
                        }
                    }
                    count += 1;
                }
                work.pop();
                if let Some(&(parent, _)) = work.last() {
                    lowlink[parent] = lowlink[parent].min(lowlink[v]);
                }
            }
        }
    }

    Components { labels, count }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    #[test]
    fn empty_graph_has_no_components() {
        let g = GraphBuilder::new().build(true);
        let c = strongly_connected_components(&g);
        assert_eq!(c.count, 0);
    }

    #[test]
    fn directed_cycle_is_one_scc() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        b.add_edge("c", "a", 1.0).unwrap();
        let g = b.build(true);
        let c = strongly_connected_components(&g);
        assert_eq!(c.count, 1);
        assert_eq!(c.labels, vec![0, 0, 0]);
    }

    #[test]
    fn directed_path_is_all_singletons() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        let g = b.build(true);
        let c = strongly_connected_components(&g);
        assert_eq!(c.count, 3);
        // Each node is its own SCC.
        let groups = c.groups();
        assert!(groups.iter().all(|grp| grp.len() == 1));
    }

    #[test]
    fn two_cycles_joined_one_way_are_two_sccs() {
        // a<->b is one SCC, c<->d another, with a one-way bridge b->c that does
        // not merge them (no path back from c to b).
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "a", 1.0).unwrap();
        b.add_edge("c", "d", 1.0).unwrap();
        b.add_edge("d", "c", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        let g = b.build(true);
        let c = strongly_connected_components(&g);
        assert_eq!(c.count, 2);
        // a,b share a label distinct from c,d's shared label.
        let (a, bb, cc, dd) = (
            g.idx_of("a").unwrap(),
            g.idx_of("b").unwrap(),
            g.idx_of("c").unwrap(),
            g.idx_of("d").unwrap(),
        );
        assert_eq!(c.labels[a], c.labels[bb]);
        assert_eq!(c.labels[cc], c.labels[dd]);
        assert_ne!(c.labels[a], c.labels[cc]);
    }

    #[test]
    fn self_loop_does_not_break_singleton() {
        // Self-loops are dropped by the builder; an isolated node is its own SCC.
        let mut b = GraphBuilder::new();
        b.add_node("solo");
        b.add_edge("a", "b", 1.0).unwrap();
        let g = b.build(true);
        let c = strongly_connected_components(&g);
        assert_eq!(c.count, 3);
    }
}
