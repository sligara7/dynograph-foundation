//! Connected-component detection.
//!
//! For an **undirected** graph this is the usual connected components. For a
//! **directed** graph it computes **weakly**-connected components — components
//! under the underlying undirected graph (i.e. ignoring edge direction). Strong
//! connectivity (Tarjan SCC) is a separate, later algorithm.

use std::collections::VecDeque;

use crate::graph::Graph;

/// Per-node component labels, contiguous in `0..count`.
///
/// `labels[i]` is the component index of node `i`; `count` is the number of
/// distinct components. Labels are assigned in order of first node visited, so
/// node `0` is always in component `0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Components {
    /// Component label per node index.
    pub labels: Vec<usize>,
    /// Number of distinct components (0 for an empty graph).
    pub count: usize,
}

impl Components {
    /// Group node indices by component label. `groups()[c]` is the list of node
    /// indices in component `c`.
    pub fn groups(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.count];
        for (node, &label) in self.labels.iter().enumerate() {
            out[label].push(node);
        }
        out
    }
}

/// Compute (weakly-)connected components via BFS over the union of out- and
/// in-adjacency. Using both directions makes this correct for directed graphs
/// (weak connectivity) and is a harmless no-op duplication for undirected ones
/// (where the two lists are identical).
pub fn connected_components(graph: &Graph) -> Components {
    let n = graph.node_count();
    let mut labels = vec![usize::MAX; n];
    let mut count = 0;

    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }
        // New component: BFS the whole reachable (undirected) region.
        let label = count;
        count += 1;
        labels[start] = label;
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            for &(v, _w) in graph.out_neighbors(u) {
                if labels[v] == usize::MAX {
                    labels[v] = label;
                    queue.push_back(v);
                }
            }
            for &(v, _w) in graph.in_neighbors(u) {
                if labels[v] == usize::MAX {
                    labels[v] = label;
                    queue.push_back(v);
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
        let g = GraphBuilder::new().build(false);
        let c = connected_components(&g);
        assert_eq!(c.count, 0);
        assert!(c.labels.is_empty());
    }

    #[test]
    fn isolated_nodes_are_separate_components() {
        let mut b = GraphBuilder::new();
        b.add_node("a");
        b.add_node("b");
        b.add_node("c");
        let g = b.build(false);
        let c = connected_components(&g);
        assert_eq!(c.count, 3);
    }

    #[test]
    fn connected_undirected_chain_is_one_component() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        let g = b.build(false);
        let c = connected_components(&g);
        assert_eq!(c.count, 1);
        assert_eq!(c.labels, vec![0, 0, 0]);
    }

    #[test]
    fn two_disconnected_groups() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("c", "d", 1.0).unwrap();
        let g = b.build(false);
        let c = connected_components(&g);
        assert_eq!(c.count, 2);
        let groups = c.groups();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0], vec![0, 1]);
        assert_eq!(groups[1], vec![2, 3]);
    }

    #[test]
    fn directed_edges_are_weakly_connected() {
        // a -> b -> c with no reverse edges is ONE weakly-connected component.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        let g = b.build(true);
        let c = connected_components(&g);
        assert_eq!(c.count, 1);
    }
}
