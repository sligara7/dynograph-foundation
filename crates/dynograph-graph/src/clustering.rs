//! Clustering coefficient and transitivity — how much a node's neighbors are
//! themselves connected (triadic closure). Treats the graph as **undirected**.
//!
//! - **Local clustering** of a node `v` with degree `k`: the fraction of the
//!   `k(k-1)/2` possible edges among `v`'s neighbors that actually exist. Nodes
//!   with degree < 2 have no defined coefficient and score 0.
//! - **Average clustering**: the mean local coefficient over all nodes.
//! - **Transitivity** (global clustering): `3·triangles / connected-triples`,
//!   i.e. the fraction of length-2 paths that close into a triangle.

use std::collections::HashSet;

use crate::graph::Graph;

/// Per-node local clustering plus the two global summaries.
#[derive(Debug, Clone, PartialEq)]
pub struct Clustering {
    /// Local clustering coefficient per node index (0 for degree < 2).
    pub local: Vec<f64>,
    /// Global transitivity: `3·triangles / connected-triples` (0 if no triples).
    pub transitivity: f64,
    /// Mean of `local` over all nodes (0 for an empty graph).
    pub average: f64,
}

/// Undirected neighbor sets (self excluded).
fn neighbor_sets(graph: &Graph) -> Vec<HashSet<usize>> {
    (0..graph.node_count())
        .map(|u| {
            graph
                .out_neighbors(u)
                .iter()
                .map(|&(v, _)| v)
                .filter(|&v| v != u)
                .collect()
        })
        .collect()
}

/// Compute local clustering, average clustering, and global transitivity.
pub fn clustering(graph: &Graph) -> Clustering {
    let n = graph.node_count();
    let nbr = neighbor_sets(graph);
    let mut local = vec![0.0; n];

    // `closed` accumulates Σ_v |N(i)∩N(v)| over neighbors i of v = 2·(edges among
    // v's neighbors); `triples` accumulates Σ_v k(k-1) = 2·(connected triples).
    let mut closed_total = 0usize;
    let mut triples_total = 0usize;
    for (v, nv) in nbr.iter().enumerate() {
        let k = nv.len();
        if k < 2 {
            continue;
        }
        // Number of (ordered) neighbor pairs of v that share an edge.
        let among: usize = nv.iter().map(|&i| nbr[i].intersection(nv).count()).sum();
        local[v] = among as f64 / (k * (k - 1)) as f64;
        closed_total += among;
        triples_total += k * (k - 1);
    }

    let transitivity = if triples_total == 0 {
        0.0
    } else {
        closed_total as f64 / triples_total as f64
    };
    let average = if n == 0 {
        0.0
    } else {
        local.iter().sum::<f64>() / n as f64
    };

    Clustering {
        local,
        transitivity,
        average,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    #[test]
    fn empty_graph() {
        let c = clustering(&GraphBuilder::new().build(false));
        assert!(c.local.is_empty());
        assert_eq!(c.transitivity, 0.0);
        assert_eq!(c.average, 0.0);
    }

    #[test]
    fn triangle_is_fully_clustered() {
        // Every node's two neighbors are connected => local = 1 everywhere,
        // transitivity = 1.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        b.add_edge("c", "a", 1.0).unwrap();
        let c = clustering(&b.build(false));
        assert!(c.local.iter().all(|&v| (v - 1.0).abs() < 1e-9), "{c:?}");
        assert!((c.transitivity - 1.0).abs() < 1e-9);
        assert!((c.average - 1.0).abs() < 1e-9);
    }

    #[test]
    fn path_has_zero_clustering() {
        // a-b-c: b's neighbors a,c are not connected => all coefficients 0.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        let c = clustering(&b.build(false));
        assert!(c.local.iter().all(|&v| v == 0.0), "{c:?}");
        assert_eq!(c.transitivity, 0.0);
    }

    #[test]
    fn star_center_has_zero_clustering() {
        // Hub's leaves are mutually unconnected => 0 everywhere.
        let mut b = GraphBuilder::new();
        b.add_edge("h", "a", 1.0).unwrap();
        b.add_edge("h", "b", 1.0).unwrap();
        b.add_edge("h", "c", 1.0).unwrap();
        let c = clustering(&b.build(false));
        assert_eq!(c.transitivity, 0.0);
        assert!(c.local.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn square_with_one_diagonal() {
        // Nodes a,b,c,d: square a-b-c-d-a plus diagonal a-c.
        // deg: a=3 (b,d,c), b=2 (a,c), c=3 (b,d,a), d=2 (a,c).
        // local(a): neighbors {b,d,c}; edges among them: b-c, a? no — among b,d,c:
        //   b-c yes, c-d yes, b-d no => 2 edges of 3 possible => 2/3.
        let mut g = GraphBuilder::new();
        g.add_edge("a", "b", 1.0).unwrap();
        g.add_edge("b", "c", 1.0).unwrap();
        g.add_edge("c", "d", 1.0).unwrap();
        g.add_edge("d", "a", 1.0).unwrap();
        g.add_edge("a", "c", 1.0).unwrap();
        let g = g.build(false);
        let c = clustering(&g);
        let la = c.local[g.idx_of("a").unwrap()];
        assert!((la - 2.0 / 3.0).abs() < 1e-9, "local(a)={la}");
        // b's neighbors a,c are connected (diagonal) => local(b)=1.
        assert!((c.local[g.idx_of("b").unwrap()] - 1.0).abs() < 1e-9);
    }
}
