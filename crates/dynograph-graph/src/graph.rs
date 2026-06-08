//! The in-memory graph abstraction every algorithm in this crate operates on.
//!
//! A [`Graph`] uses **dense `usize` node indices** (`0..node_count`) internally
//! so algorithms can key per-node state into flat `Vec`s rather than hash maps.
//! String ids (the storage-layer node ids) are interned by [`GraphBuilder`] and
//! recoverable via [`Graph::id_of`] / [`Graph::idx_of`]; algorithm results are
//! returned index-keyed and the caller maps them back to ids.
//!
//! The builder is the single seam between this crate and the outside world: the
//! service layer scans storage, projects each edge to a finite `f64` weight, and
//! feeds `(from, to, weight)` triples in. This crate never touches storage.

use std::collections::HashMap;

use crate::error::GraphError;

/// What to do with an edge whose endpoints are the same node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelfLoopPolicy {
    /// Drop self-loops entirely (the usual choice for centrality/components).
    #[default]
    Drop,
    /// Keep self-loops (relevant for some weighted-degree / Markov uses).
    Keep,
}

/// How to combine multiple edges between the same ordered pair of nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParallelEdgePolicy {
    /// Sum the weights (the natural default: two unit edges => weight 2).
    #[default]
    Sum,
    /// Keep the maximum weight.
    Max,
    /// Keep the first weight seen; ignore later duplicates.
    KeepFirst,
}

/// Accumulates nodes and weighted edges, then materializes a [`Graph`].
///
/// Node ids are interned on first sight (via either [`add_node`](Self::add_node)
/// or as an endpoint of [`add_edge`](Self::add_edge)), so the caller does not
/// have to pre-declare every node. To restrict a run to a scope, the caller
/// simply does not feed edges/nodes outside it.
#[derive(Debug, Default)]
pub struct GraphBuilder {
    ids: Vec<String>,
    index: HashMap<String, usize>,
    edges: Vec<(usize, usize, f64)>,
    self_loops: SelfLoopPolicy,
    parallel: ParallelEdgePolicy,
}

impl GraphBuilder {
    /// A new builder with default policies (drop self-loops, sum parallels).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the self-loop policy. Defaults to [`SelfLoopPolicy::Drop`].
    pub fn self_loop_policy(mut self, policy: SelfLoopPolicy) -> Self {
        self.self_loops = policy;
        self
    }

    /// Set the parallel-edge policy. Defaults to [`ParallelEdgePolicy::Sum`].
    pub fn parallel_edge_policy(mut self, policy: ParallelEdgePolicy) -> Self {
        self.parallel = policy;
        self
    }

    /// Intern a node id, returning its dense index. Idempotent.
    pub fn add_node(&mut self, id: &str) -> usize {
        if let Some(&idx) = self.index.get(id) {
            return idx;
        }
        let idx = self.ids.len();
        self.ids.push(id.to_string());
        self.index.insert(id.to_string(), idx);
        idx
    }

    /// Add a weighted edge between two node ids (interning either endpoint if
    /// not yet seen). For unweighted graphs pass `1.0`.
    ///
    /// Fails loudly if `weight` is not finite — a NaN/Inf weight is always an
    /// upstream projection bug, never a meaningful input.
    pub fn add_edge(&mut self, from: &str, to: &str, weight: f64) -> Result<(), GraphError> {
        if !weight.is_finite() {
            return Err(GraphError::NonFiniteWeight {
                from: from.to_string(),
                to: to.to_string(),
                weight,
            });
        }
        let f = self.add_node(from);
        let t = self.add_node(to);
        self.edges.push((f, t, weight));
        Ok(())
    }

    /// Materialize the graph. `directed` selects whether edges are one-way
    /// (`out`/`in` adjacency are distinct) or symmetric (both adjacency lists
    /// hold every incident edge, so degree/components/etc. read uniformly).
    ///
    /// Parallel edges are merged per the configured policy; self-loops are
    /// handled per the configured policy. For undirected graphs `(u,v)` and
    /// `(v,u)` are the same edge and merge together.
    pub fn build(self, directed: bool) -> Graph {
        let n = self.ids.len();

        // Merge parallel edges first, keyed canonically. For undirected graphs
        // the key is order-independent so (u,v) and (v,u) collapse.
        let mut merged: HashMap<(usize, usize), f64> = HashMap::new();
        for (u, v, w) in self.edges {
            if u == v && self.self_loops == SelfLoopPolicy::Drop {
                continue;
            }
            let key = if directed || u <= v { (u, v) } else { (v, u) };
            merged
                .entry(key)
                .and_modify(|existing| *existing = combine(*existing, w, self.parallel))
                .or_insert(w);
        }

        let mut out_adj = vec![Vec::new(); n];
        let mut in_adj = vec![Vec::new(); n];
        for ((u, v), w) in merged {
            if directed {
                out_adj[u].push((v, w));
                in_adj[v].push((u, w));
            } else {
                // Symmetric: every incident edge appears in both endpoints'
                // out and in lists. A kept self-loop is added once.
                out_adj[u].push((v, w));
                in_adj[u].push((v, w));
                if u != v {
                    out_adj[v].push((u, w));
                    in_adj[v].push((u, w));
                }
            }
        }

        Graph {
            ids: self.ids,
            index: self.index,
            out_adj,
            in_adj,
            directed,
        }
    }
}

fn combine(existing: f64, incoming: f64, policy: ParallelEdgePolicy) -> f64 {
    match policy {
        ParallelEdgePolicy::Sum => existing + incoming,
        ParallelEdgePolicy::Max => existing.max(incoming),
        ParallelEdgePolicy::KeepFirst => existing,
    }
}

/// An immutable, densely-indexed weighted graph.
///
/// For a **directed** graph, `out_adj[u]` are `u`'s successors and `in_adj[u]`
/// its predecessors. For an **undirected** graph both lists hold all of `u`'s
/// neighbors (so algorithms can ignore the distinction). Each adjacency entry is
/// `(neighbor_index, weight)`.
#[derive(Debug, Clone)]
pub struct Graph {
    ids: Vec<String>,
    index: HashMap<String, usize>,
    out_adj: Vec<Vec<(usize, f64)>>,
    in_adj: Vec<Vec<(usize, f64)>>,
    directed: bool,
}

impl Graph {
    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.ids.len()
    }

    /// Whether the graph was built directed.
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// The string id of a dense index. Panics on out-of-range index — indices
    /// only ever come from this graph, so an out-of-range one is a logic bug.
    pub fn id_of(&self, idx: usize) -> &str {
        &self.ids[idx]
    }

    /// The dense index of a string id, if present.
    pub fn idx_of(&self, id: &str) -> Option<usize> {
        self.index.get(id).copied()
    }

    /// Outgoing neighbors of `idx` (successors for directed, all neighbors for
    /// undirected) as `(neighbor_index, weight)`.
    pub fn out_neighbors(&self, idx: usize) -> &[(usize, f64)] {
        &self.out_adj[idx]
    }

    /// Incoming neighbors of `idx` (predecessors for directed, all neighbors for
    /// undirected) as `(neighbor_index, weight)`.
    pub fn in_neighbors(&self, idx: usize) -> &[(usize, f64)] {
        &self.in_adj[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_is_idempotent() {
        let mut b = GraphBuilder::new();
        assert_eq!(b.add_node("a"), 0);
        assert_eq!(b.add_node("b"), 1);
        assert_eq!(b.add_node("a"), 0);
        let g = b.build(true);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.id_of(0), "a");
        assert_eq!(g.idx_of("b"), Some(1));
        assert_eq!(g.idx_of("missing"), None);
    }

    #[test]
    fn add_edge_interns_endpoints() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        let g = b.build(true);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.out_neighbors(0), &[(1, 1.0)]);
        assert_eq!(g.in_neighbors(1), &[(0, 1.0)]);
        assert!(g.out_neighbors(1).is_empty());
    }

    #[test]
    fn non_finite_weight_is_rejected() {
        let mut b = GraphBuilder::new();
        assert!(matches!(
            b.add_edge("a", "b", f64::NAN),
            Err(GraphError::NonFiniteWeight { .. })
        ));
        assert!(matches!(
            b.add_edge("a", "b", f64::INFINITY),
            Err(GraphError::NonFiniteWeight { .. })
        ));
    }

    #[test]
    fn undirected_edge_is_symmetric() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 2.0).unwrap();
        let g = b.build(false);
        assert_eq!(g.out_neighbors(0), &[(1, 2.0)]);
        assert_eq!(g.out_neighbors(1), &[(0, 2.0)]);
        assert_eq!(g.in_neighbors(0), &[(1, 2.0)]);
        assert_eq!(g.in_neighbors(1), &[(0, 2.0)]);
    }

    #[test]
    fn parallel_edges_sum_by_default() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("a", "b", 3.0).unwrap();
        let g = b.build(true);
        assert_eq!(g.out_neighbors(0), &[(1, 4.0)]);
    }

    #[test]
    fn parallel_edge_max_policy() {
        let mut b = GraphBuilder::new().parallel_edge_policy(ParallelEdgePolicy::Max);
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("a", "b", 3.0).unwrap();
        let g = b.build(true);
        assert_eq!(g.out_neighbors(0), &[(1, 3.0)]);
    }

    #[test]
    fn undirected_parallel_edges_merge_regardless_of_direction() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "a", 1.0).unwrap();
        let g = b.build(false);
        // One undirected edge of summed weight 2, appearing in both lists once.
        assert_eq!(g.out_neighbors(0), &[(1, 2.0)]);
        assert_eq!(g.out_neighbors(1), &[(0, 2.0)]);
    }

    #[test]
    fn self_loops_dropped_by_default() {
        let mut b = GraphBuilder::new();
        b.add_edge("a", "a", 5.0).unwrap();
        b.add_edge("a", "b", 1.0).unwrap();
        let g = b.build(true);
        assert_eq!(g.out_neighbors(0), &[(1, 1.0)]);
    }

    #[test]
    fn self_loops_kept_when_requested() {
        let mut b = GraphBuilder::new().self_loop_policy(SelfLoopPolicy::Keep);
        b.add_edge("a", "a", 5.0).unwrap();
        let g = b.build(false);
        // Kept exactly once even for undirected.
        assert_eq!(g.out_neighbors(0), &[(0, 5.0)]);
    }
}
