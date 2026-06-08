//! Maximum flow and minimum s-t cut via the Edmonds-Karp algorithm (BFS
//! augmenting paths). Edge weights are **capacities** (non-negative). By the
//! max-flow min-cut theorem, the max flow value equals the min-cut capacity.
//!
//! For a directed graph, capacities follow edge direction; for an undirected
//! graph each edge carries its capacity in both directions (the standard
//! undirected-flow model). The residual graph is kept in `BTreeMap`s so BFS
//! visits neighbors in index order — the max-flow value is unique regardless,
//! but this makes the *reported* min cut deterministic when several exist.
//!
//! Exact and fine for the small graphs this serves (Edmonds-Karp is O(V·E²)).

use std::collections::{BTreeMap, VecDeque};

use crate::error::GraphError;
use crate::graph::Graph;

/// Tolerance for treating a residual capacity as zero.
const EPS: f64 = 1e-9;

/// Maximum flow value plus the minimum cut it certifies.
#[derive(Debug, Clone, PartialEq)]
pub struct MinCut {
    /// The maximum flow from source to target (= min-cut capacity).
    pub max_flow: f64,
    /// Node indices on the source side of the minimum cut (reachable from the
    /// source in the residual graph), ascending.
    pub source_side: Vec<usize>,
    /// Cut edges `(u, v)` crossing from the source side to the sink side,
    /// ascending. Their capacities sum to `max_flow`.
    pub cut_edges: Vec<(usize, usize)>,
}

/// Compute the maximum flow and minimum cut between `source` and `target`.
///
/// Errors: [`GraphError::InvalidParameter`] if `source == target`, or
/// [`GraphError::InvalidWeight`] on a negative capacity.
pub fn max_flow_min_cut(graph: &Graph, source: usize, target: usize) -> Result<MinCut, GraphError> {
    if source == target {
        return Err(GraphError::InvalidParameter(
            "max flow requires distinct source and target nodes".to_string(),
        ));
    }
    let n = graph.node_count();

    // Residual capacities. A real edge u->v adds capacity to res[u][v]; we ensure
    // the reverse slot res[v][u] exists (>= 0) so flow can be cancelled.
    let mut res: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); n];
    for u in 0..n {
        for &(v, c) in graph.out_neighbors(u) {
            if c < 0.0 {
                return Err(GraphError::InvalidWeight(format!(
                    "max-flow capacities must be non-negative, got {c}"
                )));
            }
            // A positive capacity below the residual-zero tolerance would be
            // silently treated as no edge (BFS skips it), under-reporting the
            // flow. Reject it loudly rather than return a wrong-but-successful
            // answer — the caller should rescale such capacities.
            if c > 0.0 && c < EPS {
                return Err(GraphError::InvalidWeight(format!(
                    "max-flow capacity {c} is positive but below the {EPS} numerical \
                     resolution; rescale capacities"
                )));
            }
            *res[u].entry(v).or_insert(0.0) += c;
            res[v].entry(u).or_insert(0.0);
        }
    }

    let mut max_flow = 0.0;
    loop {
        // BFS for an augmenting path; parent[v] = predecessor on the path.
        let mut parent = vec![usize::MAX; n];
        parent[source] = source;
        let mut queue = VecDeque::new();
        queue.push_back(source);
        while let Some(u) = queue.pop_front() {
            if u == target {
                break;
            }
            for (&v, &cap) in &res[u] {
                if parent[v] == usize::MAX && cap > EPS {
                    parent[v] = u;
                    queue.push_back(v);
                }
            }
        }
        if parent[target] == usize::MAX {
            break; // no augmenting path remains
        }

        // Bottleneck residual capacity along the path.
        let mut bottleneck = f64::INFINITY;
        let mut v = target;
        while v != source {
            let u = parent[v];
            bottleneck = bottleneck.min(res[u][&v]);
            v = u;
        }
        // Push the flow, updating forward and reverse residuals.
        let mut v = target;
        while v != source {
            let u = parent[v];
            *res[u].get_mut(&v).unwrap() -= bottleneck;
            *res[v].get_mut(&u).unwrap() += bottleneck;
            v = u;
        }
        max_flow += bottleneck;
    }

    // Min cut: nodes reachable from the source in the final residual graph.
    let mut reachable = vec![false; n];
    reachable[source] = true;
    let mut queue = VecDeque::from([source]);
    while let Some(u) = queue.pop_front() {
        for (&v, &cap) in &res[u] {
            if !reachable[v] && cap > EPS {
                reachable[v] = true;
                queue.push_back(v);
            }
        }
    }
    let source_side: Vec<usize> = (0..n).filter(|&v| reachable[v]).collect();

    // Cut edges: original edges from the source side to the sink side. Looping
    // only over source-side tails visits each crossing edge once.
    let mut cut_edges = Vec::new();
    for &u in &source_side {
        for &(v, _) in graph.out_neighbors(u) {
            if !reachable[v] {
                cut_edges.push((u, v));
            }
        }
    }
    cut_edges.sort_unstable();
    cut_edges.dedup();

    Ok(MinCut {
        max_flow,
        source_side,
        cut_edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    #[test]
    fn single_edge_flow() {
        let mut b = GraphBuilder::new();
        b.add_edge("s", "t", 5.0).unwrap();
        let g = b.build(true);
        let (s, t) = (g.idx_of("s").unwrap(), g.idx_of("t").unwrap());
        let cut = max_flow_min_cut(&g, s, t).unwrap();
        assert!((cut.max_flow - 5.0).abs() < 1e-9);
        assert_eq!(cut.source_side, vec![s]);
        assert_eq!(cut.cut_edges, vec![(s, t)]);
    }

    #[test]
    fn classic_network_max_flow_is_five() {
        // s->a(3), s->b(2), a->b(1), a->t(2), b->t(3). Max flow = 5.
        let mut b = GraphBuilder::new();
        b.add_edge("s", "a", 3.0).unwrap();
        b.add_edge("s", "b", 2.0).unwrap();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("a", "t", 2.0).unwrap();
        b.add_edge("b", "t", 3.0).unwrap();
        let g = b.build(true);
        let cut = max_flow_min_cut(&g, g.idx_of("s").unwrap(), g.idx_of("t").unwrap()).unwrap();
        assert!((cut.max_flow - 5.0).abs() < 1e-9, "{cut:?}");
        // Both source edges are saturated, so only s is on the source side and
        // the cut is exactly {s->a, s->b} (capacity 3+2 = 5).
        assert_eq!(cut.source_side, vec![g.idx_of("s").unwrap()]);
        assert_eq!(cut.cut_edges.len(), 2);
    }

    #[test]
    fn bottleneck_limits_flow() {
        // s->a(10), a->t(1): the a->t bottleneck caps flow at 1.
        let mut b = GraphBuilder::new();
        b.add_edge("s", "a", 10.0).unwrap();
        b.add_edge("a", "t", 1.0).unwrap();
        let g = b.build(true);
        let cut = max_flow_min_cut(&g, g.idx_of("s").unwrap(), g.idx_of("t").unwrap()).unwrap();
        assert!((cut.max_flow - 1.0).abs() < 1e-9, "{cut:?}");
    }

    #[test]
    fn disconnected_has_zero_flow() {
        let mut b = GraphBuilder::new();
        b.add_edge("s", "a", 5.0).unwrap();
        b.add_node("t");
        let g = b.build(true);
        let cut = max_flow_min_cut(&g, g.idx_of("s").unwrap(), g.idx_of("t").unwrap()).unwrap();
        assert_eq!(cut.max_flow, 0.0);
        assert!(cut.cut_edges.is_empty());
    }

    #[test]
    fn undirected_edge_carries_capacity_both_ways() {
        // Undirected s-a(3), a-t(3): max flow 3.
        let mut b = GraphBuilder::new();
        b.add_edge("s", "a", 3.0).unwrap();
        b.add_edge("a", "t", 3.0).unwrap();
        let g = b.build(false);
        let cut = max_flow_min_cut(&g, g.idx_of("s").unwrap(), g.idx_of("t").unwrap()).unwrap();
        assert!((cut.max_flow - 3.0).abs() < 1e-9, "{cut:?}");
    }

    #[test]
    fn same_source_and_target_rejected() {
        let mut b = GraphBuilder::new();
        b.add_edge("s", "t", 1.0).unwrap();
        let g = b.build(true);
        let s = g.idx_of("s").unwrap();
        assert!(matches!(
            max_flow_min_cut(&g, s, s),
            Err(GraphError::InvalidParameter(_))
        ));
    }

    #[test]
    fn sub_resolution_capacity_rejected() {
        // A positive-but-tiny capacity would silently read as zero flow; reject it.
        let mut b = GraphBuilder::new();
        b.add_edge("s", "t", 1e-12).unwrap();
        let g = b.build(true);
        assert!(matches!(
            max_flow_min_cut(&g, g.idx_of("s").unwrap(), g.idx_of("t").unwrap()),
            Err(GraphError::InvalidWeight(_))
        ));
    }

    #[test]
    fn negative_capacity_rejected() {
        let mut b = GraphBuilder::new();
        b.add_edge("s", "t", -1.0).unwrap();
        let g = b.build(true);
        assert!(matches!(
            max_flow_min_cut(&g, g.idx_of("s").unwrap(), g.idx_of("t").unwrap()),
            Err(GraphError::InvalidWeight(_))
        ));
    }
}
