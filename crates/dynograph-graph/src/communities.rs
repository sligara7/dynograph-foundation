//! Community detection via the **Louvain** method (modularity maximization),
//! treating the graph as **undirected**.
//!
//! Where connected components answer "what's reachable" and clustering answers
//! "how cliquey is a node," Louvain answers "what are the dense sub-communities
//! *within* a connected graph" — the faction/cluster-discovery primitive. It
//! greedily maximizes **modularity**
//!
//! ```text
//! Q = (1/2m) Σ_uv [A_uv - γ k_u k_v / 2m] δ(c_u, c_v)
//! ```
//!
//! where `m` is total edge weight, `k_u` the weighted degree, `γ` the
//! resolution (1.0 = classic modularity; higher → more, smaller communities),
//! and `δ` is 1 when `u`,`v` share a community. Edge weights are **strengths**
//! (higher = tighter tie), like PageRank/eigenvector.
//!
//! Two alternating phases repeat until modularity stops improving:
//! 1. **Local moving** — each node, in index order, leaves its community and
//!    joins the neighboring community giving the largest modularity gain (or
//!    stays). Repeat until a full pass moves nothing.
//! 2. **Aggregation** — collapse each community into one super-node (intra-
//!    community weight becomes a self-loop, inter-community weight becomes a
//!    super-edge) and recurse.
//!
//! **Determinism.** Nodes are processed in ascending index order, communities
//! are renumbered by ascending representative, and gain ties are broken toward
//! the smallest community index. With the builder's already-sorted adjacency
//! the result is fully reproducible — no randomized restarts (consumer graphs
//! are small, so the exact greedy pass is affordable and stable).

use std::collections::HashMap;

use crate::error::GraphError;
use crate::graph::Graph;

/// A modularity-maximizing partition of the graph's nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct Communities {
    /// Community label per node index, in `0..num_communities`. Labels are
    /// assigned deterministically (by ascending smallest member index).
    pub labels: Vec<usize>,
    /// Number of communities.
    pub num_communities: usize,
    /// Modularity of the returned partition under the requested resolution.
    /// `0.0` for an edgeless or empty graph (modularity is undefined there).
    pub modularity: f64,
}

/// Safety cap on local-moving passes per level. The loop is monotonic
/// (modularity only rises) and converges quickly on the small graphs here; the
/// cap is a fail-safe against a pathological oscillation pinning a worker,
/// matching the iteration caps on the power-iteration centralities.
const MAX_PASSES_PER_LEVEL: usize = 100;

/// Detect communities by Louvain modularity maximization at the given
/// `resolution` (γ; pass `1.0` for classic modularity). The graph is treated
/// as undirected via its symmetric adjacency.
///
/// Errors:
/// - [`GraphError::InvalidParameter`] if `resolution` is not a positive finite
///   number (a non-positive γ makes every merge look beneficial; NaN poisons
///   every gain).
/// - [`GraphError::InvalidWeight`] on a negative edge weight — like
///   PageRank/eigenvector, weights are **strengths** and must be non-negative;
///   modularity is ill-defined for negative weights (they can cancel `2m` to
///   zero or flip its sign), so reject rather than return a silently-wrong
///   partition.
pub fn louvain(graph: &Graph, resolution: f64) -> Result<Communities, GraphError> {
    if !resolution.is_finite() || resolution <= 0.0 {
        return Err(GraphError::InvalidParameter(format!(
            "resolution must be a positive finite number, got {resolution}"
        )));
    }
    let n = graph.node_count();
    if n == 0 {
        return Ok(Communities {
            labels: Vec::new(),
            num_communities: 0,
            modularity: 0.0,
        });
    }

    // ---- Level 0 adjacency, derived from the (undirected) graph. ----
    // `adj[u]` holds (neighbor, weight) for u != v; a self-loop on the input
    // graph (if the builder kept it) folds into `self_loops[u]` rather than
    // being dropped, so its weight still counts toward the degree and carries
    // across aggregation levels (where intra-community weight also lives there).
    let mut adj: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
    let mut self_loops = vec![0.0f64; n];
    for (u, self_loop) in self_loops.iter_mut().enumerate() {
        let mut row = Vec::new();
        for &(v, w) in graph.out_neighbors(u) {
            if w < 0.0 {
                return Err(GraphError::InvalidWeight(format!(
                    "community detection weights are strengths and must be non-negative, got {w}"
                )));
            }
            if v == u {
                *self_loop += w;
            } else {
                row.push((v, w));
            }
        }
        adj.push(row);
    }

    // 2m = Σ degrees (m = total edge weight); invariant across aggregation levels.
    let two_m: f64 = (0..n).map(|u| degree(&adj[u], self_loops[u])).sum::<f64>();
    // Edgeless graph: every node is its own community, modularity 0.
    if two_m == 0.0 {
        return Ok(Communities {
            labels: (0..n).collect(),
            num_communities: n,
            modularity: 0.0,
        });
    }
    let m = two_m / 2.0;

    // `node_to_super[orig]` is the community index of original node `orig` at
    // the current level (starts as identity); composed across levels.
    let mut node_to_super: Vec<usize> = (0..n).collect();

    loop {
        let level_n = adj.len();
        let comm = local_moving(&adj, &self_loops, m, resolution);

        // Renumber the communities that survived this level to a dense
        // 0..k range, ordered by ascending old community index (determinism).
        let (relabel, k) = dense_relabel(&comm);
        // Compose: every original node now points at its renumbered community.
        for c in node_to_super.iter_mut() {
            *c = relabel[comm[*c]];
        }

        // Converged when local moving didn't reduce the community count.
        if k == level_n {
            return Ok(finalize(graph, &node_to_super, k, m, resolution));
        }

        // ---- Aggregate into the next level's graph. ----
        let (next_adj, next_self) = aggregate(&adj, &self_loops, &comm, &relabel, k);
        adj = next_adj;
        self_loops = next_self;
    }
}

/// Weighted degree of a node: incident off-diagonal weight plus twice its
/// self-loop weight (the standard convention making `Σ_u k_u = 2m`).
fn degree(neighbors: &[(usize, f64)], self_loop: f64) -> f64 {
    neighbors.iter().map(|&(_, w)| w).sum::<f64>() + 2.0 * self_loop
}

/// One level of local moving. Returns the community label per super-node.
fn local_moving(
    adj: &[Vec<(usize, f64)>],
    self_loops: &[f64],
    m: f64,
    resolution: f64,
) -> Vec<usize> {
    let n = adj.len();
    let k: Vec<f64> = (0..n).map(|u| degree(&adj[u], self_loops[u])).collect();
    let mut comm: Vec<usize> = (0..n).collect();
    // tot[c] = Σ of degrees of nodes currently in community c.
    let mut tot = k.clone();

    for _pass in 0..MAX_PASSES_PER_LEVEL {
        let mut moved = false;
        for u in 0..n {
            let old = comm[u];
            // Weight from u into each neighboring community (u's own edges).
            let mut to_comm: HashMap<usize, f64> = HashMap::new();
            for &(v, w) in &adj[u] {
                *to_comm.entry(comm[v]).or_insert(0.0) += w;
            }
            // Remove u from its community before scoring candidates.
            tot[old] -= k[u];

            // Baseline: returning to the old community (a net no-op move).
            let mut best_comm = old;
            let mut best_gain = to_comm.get(&old).copied().unwrap_or(0.0)
                - resolution * tot[old] * k[u] / (2.0 * m);

            // Consider neighbor communities in ascending index order so gain
            // ties resolve to the smallest community index (determinism).
            let mut candidates: Vec<usize> = to_comm.keys().copied().collect();
            candidates.sort_unstable();
            for c in candidates {
                let gain = to_comm[&c] - resolution * tot[c] * k[u] / (2.0 * m);
                if gain > best_gain {
                    best_gain = gain;
                    best_comm = c;
                }
            }

            tot[best_comm] += k[u];
            comm[u] = best_comm;
            if best_comm != old {
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    comm
}

/// Map the distinct community ids in `comm` to a dense `0..k` range, ordered by
/// ascending original id. Returns `(old_id -> new_id, k)`. `old_id` indexes a
/// vector of length `comm.len()` (community ids are node indices), so unused
/// slots hold `usize::MAX` and must never be read.
fn dense_relabel(comm: &[usize]) -> (Vec<usize>, usize) {
    let mut relabel = vec![usize::MAX; comm.len()];
    let mut next = 0;
    // Ascending old-id order: scan in node order and assign on first sight.
    for &c in comm {
        if relabel[c] == usize::MAX {
            relabel[c] = next;
            next += 1;
        }
    }
    (relabel, next)
}

/// Collapse each community into a super-node: intra-community off-diagonal
/// weight and carried self-loops become the super-node's self-loop; inter-
/// community weight becomes super-edges (stored in both directions).
fn aggregate(
    adj: &[Vec<(usize, f64)>],
    self_loops: &[f64],
    comm: &[usize],
    relabel: &[usize],
    k: usize,
) -> (Vec<Vec<(usize, f64)>>, Vec<f64>) {
    let n = adj.len();
    let mut new_self = vec![0.0f64; k];
    // Inter-community weight, keyed by ordered pair (a < b).
    let mut inter: HashMap<(usize, usize), f64> = HashMap::new();

    for u in 0..n {
        let cu = relabel[comm[u]];
        new_self[cu] += self_loops[u]; // carry existing self-loops
        for &(v, w) in &adj[u] {
            if u < v {
                // Count each undirected off-diagonal edge once.
                let cv = relabel[comm[v]];
                if cu == cv {
                    new_self[cu] += w;
                } else {
                    let key = if cu < cv { (cu, cv) } else { (cv, cu) };
                    *inter.entry(key).or_insert(0.0) += w;
                }
            }
        }
    }

    let mut new_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); k];
    for ((a, b), w) in inter {
        new_adj[a].push((b, w));
        new_adj[b].push((a, w));
    }
    // Sort each list by neighbor for deterministic downstream iteration.
    for list in new_adj.iter_mut() {
        list.sort_unstable_by_key(|&(v, _)| v);
    }
    (new_adj, new_self)
}

/// Build the [`Communities`] result: the per-node labels (already composed in
/// `labels`) and the modularity of that partition computed on the original
/// level-0 graph.
fn finalize(
    graph: &Graph,
    labels: &[usize],
    num_communities: usize,
    m: f64,
    resolution: f64,
) -> Communities {
    let n = graph.node_count();
    // tot[c] = Σ degrees in c; intra[c] = Σ_in(c)/2 = intra-community edge
    // weight (each off-diagonal edge once) plus self-loop weight. A self-loop
    // contributes `2w` to its node's degree and `w` to intra (it is an
    // in-community link), matching the level-0 `self_loops` folding.
    let mut tot = vec![0.0f64; num_communities];
    let mut intra = vec![0.0f64; num_communities];
    for u in 0..n {
        let cu = labels[u];
        for &(v, w) in graph.out_neighbors(u) {
            if v == u {
                tot[cu] += 2.0 * w;
                intra[cu] += w;
            } else {
                tot[cu] += w;
                if u < v && labels[v] == cu {
                    intra[cu] += w;
                }
            }
        }
    }
    let mut q = 0.0;
    for c in 0..num_communities {
        // 2*intra (both directions) / 2m, minus the null-model term.
        q += intra[c] / m - resolution * (tot[c] / (2.0 * m)).powi(2);
    }

    Communities {
        labels: labels.to_vec(),
        num_communities,
        modularity: q,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    /// Two triangles joined by a single bridge edge — the textbook planted
    /// two-community graph. Louvain must recover the two triangles.
    fn two_triangles() -> Graph {
        let mut b = GraphBuilder::new();
        for (x, y) in [
            ("a1", "a2"),
            ("a2", "a3"),
            ("a3", "a1"),
            ("b1", "b2"),
            ("b2", "b3"),
            ("b3", "b1"),
            ("a1", "b1"), // the bridge
        ] {
            b.add_edge(x, y, 1.0).unwrap();
        }
        b.build(false)
    }

    fn group_of(g: &Graph, c: &Communities, id: &str) -> usize {
        c.labels[g.idx_of(id).unwrap()]
    }

    #[test]
    fn empty_graph() {
        let c = louvain(&GraphBuilder::new().build(false), 1.0).unwrap();
        assert!(c.labels.is_empty());
        assert_eq!(c.num_communities, 0);
        assert_eq!(c.modularity, 0.0);
    }

    #[test]
    fn edgeless_graph_is_all_singletons() {
        let mut b = GraphBuilder::new();
        b.add_node("a");
        b.add_node("b");
        let c = louvain(&b.build(false), 1.0).unwrap();
        assert_eq!(c.num_communities, 2);
        assert_eq!(c.modularity, 0.0);
    }

    #[test]
    fn recovers_two_planted_triangles() {
        let g = two_triangles();
        let c = louvain(&g, 1.0).unwrap();
        assert_eq!(c.num_communities, 2, "two communities: {c:?}");
        // The three a-nodes share a community; likewise the b-nodes; and the
        // two communities differ.
        let ca = group_of(&g, &c, "a1");
        assert_eq!(group_of(&g, &c, "a2"), ca);
        assert_eq!(group_of(&g, &c, "a3"), ca);
        let cb = group_of(&g, &c, "b1");
        assert_eq!(group_of(&g, &c, "b2"), cb);
        assert_eq!(group_of(&g, &c, "b3"), cb);
        assert_ne!(ca, cb);
        assert!(
            c.modularity > 0.3,
            "modularity {} should clear 0.3",
            c.modularity
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let g = two_triangles();
        let a = louvain(&g, 1.0).unwrap();
        let b = louvain(&g, 1.0).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn weights_respect_strengths() {
        // A 4-clique-ish graph where weak cross edges and strong within-pair
        // edges should split {a,b} from {c,d}. Without weights the symmetric
        // structure has no preferred split; strong intra edges drive it.
        let mut bld = GraphBuilder::new();
        bld.add_edge("a", "b", 10.0).unwrap();
        bld.add_edge("c", "d", 10.0).unwrap();
        bld.add_edge("b", "c", 1.0).unwrap();
        bld.add_edge("a", "d", 1.0).unwrap();
        let g = bld.build(false);
        let c = louvain(&g, 1.0).unwrap();
        assert_eq!(c.num_communities, 2, "{c:?}");
        assert_eq!(group_of(&g, &c, "a"), group_of(&g, &c, "b"));
        assert_eq!(group_of(&g, &c, "c"), group_of(&g, &c, "d"));
        assert_ne!(group_of(&g, &c, "a"), group_of(&g, &c, "c"));
    }

    #[test]
    fn invalid_resolution_is_rejected() {
        let g = two_triangles();
        assert!(matches!(
            louvain(&g, 0.0),
            Err(GraphError::InvalidParameter(_))
        ));
        assert!(matches!(
            louvain(&g, -1.0),
            Err(GraphError::InvalidParameter(_))
        ));
        assert!(matches!(
            louvain(&g, f64::NAN),
            Err(GraphError::InvalidParameter(_))
        ));
    }

    #[test]
    fn kept_self_loop_is_counted_not_dropped() {
        // With SelfLoopPolicy::Keep, a self-loop's weight must fold into the
        // node's strength rather than being silently dropped (which previously
        // zeroed 2m and forced the all-singletons path). A lone self-looped node
        // is one community wholly containing the graph => modularity 0.
        let mut b =
            crate::graph::GraphBuilder::new().self_loop_policy(crate::graph::SelfLoopPolicy::Keep);
        b.add_edge("a", "a", 3.0).unwrap();
        let c = louvain(&b.build(false), 1.0).unwrap();
        assert_eq!(c.num_communities, 1, "{c:?}");
        assert!(c.modularity.abs() < 1e-9, "modularity {}", c.modularity);
    }

    #[test]
    fn negative_weight_is_rejected() {
        // Strengths must be non-negative (like PageRank/eigenvector); a negative
        // weight is rejected loudly rather than corrupting modularity.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", -1.0).unwrap();
        assert!(matches!(
            louvain(&b.build(false), 1.0),
            Err(GraphError::InvalidWeight(_))
        ));
    }

    #[test]
    fn single_clique_is_one_community() {
        // A triangle has no good split — Louvain should keep it whole.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        b.add_edge("c", "a", 1.0).unwrap();
        let c = louvain(&b.build(false), 1.0).unwrap();
        assert_eq!(c.num_communities, 1);
    }
}
