//! Centrality measures. PR A ships degree centrality; PageRank, closeness,
//! betweenness, and eigenvector centrality slot in alongside it later.

use crate::graph::Graph;

/// Which incident edges to count for [`degree_centrality`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegreeMode {
    /// In-degree (predecessors). Equals total for undirected graphs.
    In,
    /// Out-degree (successors). Equals total for undirected graphs.
    Out,
    /// Both directions. For undirected graphs each neighbor is counted once.
    Total,
}

/// Degree centrality per node index.
///
/// When `weighted` is false, the score is the number of incident edges. When
/// `weighted` is true, it is the sum of incident edge weights (the node
/// "strength"). For undirected graphs `In`, `Out`, and `Total` all read the same
/// (symmetric) adjacency, so each neighbor is counted exactly once.
pub fn degree_centrality(graph: &Graph, mode: DegreeMode, weighted: bool) -> Vec<f64> {
    let n = graph.node_count();
    let score = |adj: &[(usize, f64)]| -> f64 {
        if weighted {
            adj.iter().map(|&(_, w)| w).sum()
        } else {
            adj.len() as f64
        }
    };

    (0..n)
        .map(|u| {
            if !graph.is_directed() {
                // Undirected: out and in are identical; count once.
                score(graph.out_neighbors(u))
            } else {
                match mode {
                    DegreeMode::In => score(graph.in_neighbors(u)),
                    DegreeMode::Out => score(graph.out_neighbors(u)),
                    DegreeMode::Total => {
                        score(graph.out_neighbors(u)) + score(graph.in_neighbors(u))
                    }
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    fn star_directed() -> Graph {
        // hub "h" -> a, b, c (out-degree 3 at hub, in-degree 1 at each leaf)
        let mut b = GraphBuilder::new();
        b.add_edge("h", "a", 1.0).unwrap();
        b.add_edge("h", "b", 1.0).unwrap();
        b.add_edge("h", "c", 1.0).unwrap();
        b.build(true)
    }

    #[test]
    fn directed_out_degree() {
        let g = star_directed();
        let d = degree_centrality(&g, DegreeMode::Out, false);
        assert_eq!(d[g.idx_of("h").unwrap()], 3.0);
        assert_eq!(d[g.idx_of("a").unwrap()], 0.0);
    }

    #[test]
    fn directed_in_degree() {
        let g = star_directed();
        let d = degree_centrality(&g, DegreeMode::In, false);
        assert_eq!(d[g.idx_of("h").unwrap()], 0.0);
        assert_eq!(d[g.idx_of("a").unwrap()], 1.0);
    }

    #[test]
    fn directed_total_degree() {
        let g = star_directed();
        let d = degree_centrality(&g, DegreeMode::Total, false);
        assert_eq!(d[g.idx_of("h").unwrap()], 3.0);
        assert_eq!(d[g.idx_of("a").unwrap()], 1.0);
    }

    #[test]
    fn undirected_counts_each_neighbor_once() {
        let mut b = GraphBuilder::new();
        b.add_edge("h", "a", 1.0).unwrap();
        b.add_edge("h", "b", 1.0).unwrap();
        let g = b.build(false);
        // Total must not double-count on an undirected graph.
        let d = degree_centrality(&g, DegreeMode::Total, false);
        assert_eq!(d[g.idx_of("h").unwrap()], 2.0);
        assert_eq!(d[g.idx_of("a").unwrap()], 1.0);
    }

    #[test]
    fn weighted_degree_is_strength() {
        let mut b = GraphBuilder::new();
        b.add_edge("h", "a", 2.5).unwrap();
        b.add_edge("h", "b", 1.5).unwrap();
        let g = b.build(true);
        let d = degree_centrality(&g, DegreeMode::Out, true);
        assert_eq!(d[g.idx_of("h").unwrap()], 4.0);
    }
}
