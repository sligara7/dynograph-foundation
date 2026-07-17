//! Community detection via the **Leiden** method (modularity maximization),
//! treating the graph as **undirected**.
//!
//! Leiden (Traag, Waltman & van Eck, 2019) is the successor to Louvain (Blondel
//! et al., 2008): same modularity objective, but it repairs Louvain's one real
//! defect — communities that come out *badly connected*, or even internally
//! *disconnected*. Where connected components answer "what's reachable" and
//! clustering answers "how cliquey is a node," Leiden answers "what are the
//! dense sub-communities *within* a connected graph" — the faction/cluster
//! discovery primitive. It greedily maximizes **modularity**
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
//! Three phases repeat until the partition stops changing:
//! 1. **Local moving** — each node, in index order, leaves its community and
//!    joins the neighboring community giving the largest modularity gain (or
//!    stays). Repeat until a full pass moves nothing. (This is Louvain's inner
//!    loop, seeded from the previous level's partition `P`.)
//! 2. **Refinement** — starting from *singletons*, and never crossing a `P`
//!    community boundary, merge each node into a *well-connected* sub-community.
//!    A node is only eligible to move if it is itself well-connected to its `P`
//!    community, and it may only join a sub-community that is well-connected to
//!    that community. The result `P_refined` **subdivides** `P`, and every one
//!    of its parts is internally connected — this is the step Louvain lacks and
//!    the reason Leiden never emits a disconnected community.
//! 3. **Aggregation** — collapse each *refined* sub-community into one super-node
//!    (intra weight becomes a self-loop, inter weight becomes a super-edge), but
//!    seed the next level's community assignment from `P` (each refined part sits
//!    inside exactly one `P` community). Recurse. Aggregating at the finer
//!    `P_refined` granularity — while still grouping by `P` — is what carries the
//!    connectivity guarantee up the levels.
//!
//! **Determinism.** The reference Leiden randomizes the refinement move (picking
//! among quality-preserving targets by a Boltzmann weight) to escape local
//! optima. We deliberately run the **deterministic greedy** variant instead:
//! nodes are processed in ascending index order, each takes the strictly-best
//! well-connected target, gain ties break toward the smallest community index,
//! and communities are renumbered by ascending representative. The connectivity
//! guarantee comes from the refinement *structure* (eligible-node and
//! well-connected-target predicates), **not** from the randomness, so dropping
//! the randomness costs only the occasional escape from a local optimum — not
//! the guarantee. With the builder's already-sorted adjacency the result is
//! fully reproducible, matching the no-randomized-restarts posture of the
//! power-iteration centralities (consumer graphs are small, so the exact greedy
//! pass is affordable and stable).

use std::collections::{HashMap, HashSet};

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

/// Detect communities by Leiden modularity maximization at the given
/// `resolution` (γ; pass `1.0` for classic modularity). The graph is treated
/// as undirected via its symmetric adjacency, and every returned community is
/// guaranteed to be internally connected.
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
pub fn leiden(graph: &Graph, resolution: f64) -> Result<Communities, GraphError> {
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

    // `orig_to_node[orig]` is the current-level super-node holding original node
    // `orig` (starts as identity); composed through each level's *refined*
    // partition. `seed` is the community each current-level node starts local
    // moving in (level 0: singletons); after a level it is `P` carried onto the
    // aggregated (refined) nodes.
    let mut orig_to_node: Vec<usize> = (0..n).collect();
    let mut seed: Vec<usize> = (0..n).collect();

    loop {
        let level_n = adj.len();

        // 1. Local moving from the seed partition → P (dense 0..p).
        let (part, _p) = densify(&local_moving(&adj, &self_loops, m, resolution, &seed));

        // 2. Refine within P → P_refined (dense 0..r). Subdivides P; every part
        //    is well-connected, hence internally connected.
        let (refined, r) = densify(&refine(&adj, &self_loops, two_m, resolution, &part));

        // Converged when refinement can't split anything finer than the current
        // nodes — aggregation would reproduce the same graph, so stop and return
        // P mapped back onto the original nodes.
        if r == level_n {
            return Ok(finalize(
                graph,
                &compose(&orig_to_node, &part),
                m,
                resolution,
            ));
        }

        // 3. Aggregate the refined sub-communities into the next level's graph,
        //    and seed each new node with its P community (refined ⊆ P).
        let (next_adj, next_self) = aggregate(&adj, &self_loops, &refined, r);
        let mut next_seed = vec![0usize; r];
        for (&rc, &pc) in refined.iter().zip(part.iter()) {
            next_seed[rc] = pc;
        }
        // Compose original→node through this level's refinement.
        for o in orig_to_node.iter_mut() {
            *o = refined[*o];
        }
        adj = next_adj;
        self_loops = next_self;
        seed = next_seed;
    }
}

/// Map each original node to its `P` community: `part[orig_to_node[orig]]`.
fn compose(orig_to_node: &[usize], part: &[usize]) -> Vec<usize> {
    orig_to_node.iter().map(|&node| part[node]).collect()
}

/// Weighted degree of a node: incident off-diagonal weight plus twice its
/// self-loop weight (the standard convention making `Σ_u k_u = 2m`).
fn degree(neighbors: &[(usize, f64)], self_loop: f64) -> f64 {
    neighbors.iter().map(|&(_, w)| w).sum::<f64>() + 2.0 * self_loop
}

/// One level of local moving, seeded from `seed` (community label per node,
/// dense `0..p`). Returns the community label per super-node (a subset of the
/// seed labels — the caller renumbers to a dense range).
fn local_moving(
    adj: &[Vec<(usize, f64)>],
    self_loops: &[f64],
    m: f64,
    resolution: f64,
    seed: &[usize],
) -> Vec<usize> {
    let n = adj.len();
    let k: Vec<f64> = (0..n).map(|u| degree(&adj[u], self_loops[u])).collect();
    let mut comm: Vec<usize> = seed.to_vec();
    // tot[c] = Σ of degrees of nodes currently in community c. `seed` is dense
    // 0..p, so a length-n vector covers every live label.
    let mut tot = vec![0.0f64; n];
    for (u, &c) in comm.iter().enumerate() {
        tot[c] += k[u];
    }

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

/// Refine `part` into `P_refined`, which subdivides `part` into internally
/// well-connected sub-communities. Returns the refined community label per node
/// (community ids are node indices — the caller renumbers to a dense range).
///
/// This is the phase Louvain lacks. Starting from singletons, we visit each
/// `part` community `S` and merge its nodes together, but only ever:
/// - moving a node `u` that is **well-connected to `S`** (its within-`S` weight
///   clears the modularity null-model threshold), and
/// - into a sub-community that is itself **well-connected to `S`**.
///
/// A node once merged is fixed (single pass, ascending order). Because every
/// merge joins well-connected pieces, each refined sub-community is connected.
fn refine(
    adj: &[Vec<(usize, f64)>],
    self_loops: &[f64],
    two_m: f64,
    resolution: f64,
    part: &[usize],
) -> Vec<usize> {
    let n = adj.len();
    let k: Vec<f64> = (0..n).map(|u| degree(&adj[u], self_loops[u])).collect();

    // Group node indices by their P community (ascending label order).
    let p = part.iter().copied().max().map_or(0, |mx| mx + 1);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); p];
    for (u, &c) in part.iter().enumerate() {
        groups[c].push(u);
    }

    // Refined partition: start as singletons. A community's id is a node index.
    // `tot_ref[c]` is its degree sum (k_C); `size[c]` its member count (an
    // untouched singleton has size 1); `e_cross[c]` its crossing weight to the
    // rest of its P community, E(C, S∖C). Maintaining `e_cross` incrementally —
    // rather than rescanning the community's edges — makes the well-connected
    // test an O(1) comparison. Its closed-form delta when singleton `u` merges
    // into `c`: the u↔c weight (once crossing from both sides) turns internal,
    // so E(c, S∖c) += E(u, S∖u) − 2·E(u, c); the rest of u's crossing weight
    // stays crossing.
    let mut refined: Vec<usize> = (0..n).collect();
    let mut tot_ref: Vec<f64> = k.clone();
    let mut size: Vec<usize> = vec![1; n];

    for s_nodes in &groups {
        if s_nodes.len() <= 1 {
            continue; // a lone node is already its own (trivially connected) part
        }
        let in_s: HashSet<usize> = s_nodes.iter().copied().collect();

        // Seed each singleton's crossing weight (E({u}, S∖u) = u's within-S
        // degree) and k_S = Σ k over S.
        let mut e_cross: HashMap<usize, f64> = HashMap::new();
        let mut k_s = 0.0;
        for &u in s_nodes {
            k_s += k[u];
            let d: f64 = adj[u]
                .iter()
                .filter(|&&(v, _)| in_s.contains(&v))
                .map(|&(_, w)| w)
                .sum();
            e_cross.insert(u, d);
        }

        // `groups` is built in ascending node order, so `s_nodes` is already
        // sorted — the ascending pass the determinism note relies on.
        for &u in s_nodes {
            // Only untouched singletons move (a moved or absorbed node has a
            // size other than 1).
            if size[u] != 1 {
                continue;
            }
            // Eligible only if u is well-connected to S:
            //   E(u, S∖u) ≥ γ k_u (k_S − k_u) / 2m.
            if e_cross[&u] * two_m < resolution * k[u] * (k_s - k[u]) {
                continue;
            }

            // Weight from u into each neighboring refined community within S.
            let mut e_to: HashMap<usize, f64> = HashMap::new();
            for &(v, w) in &adj[u] {
                if in_s.contains(&v) {
                    *e_to.entry(refined[v]).or_insert(0.0) += w;
                }
            }

            // Best strictly-positive-gain move into a well-connected community,
            // ties → smallest community id (candidates sorted ascending).
            let mut candidates: Vec<usize> = e_to.keys().copied().collect();
            candidates.sort_unstable();
            let mut best = u; // stay a singleton (gain 0)
            let mut best_gain = 0.0;
            for c in candidates {
                if c == u {
                    continue;
                }
                // C well-connected to S: E(C, S∖C) ≥ γ k_C (k_S − k_C) / 2m.
                if e_cross[&c] * two_m < resolution * tot_ref[c] * (k_s - tot_ref[c]) {
                    continue;
                }
                let gain = e_to[&c] - resolution * tot_ref[c] * k[u] / two_m;
                if gain > best_gain {
                    best_gain = gain;
                    best = c;
                }
            }

            if best != u {
                let delta = e_cross[&u] - 2.0 * e_to[&best];
                refined[u] = best;
                tot_ref[best] += k[u];
                size[best] += 1;
                size[u] = 0; // u's own singleton community is now empty
                *e_cross.get_mut(&best).unwrap() += delta;
            }
        }
    }
    refined
}

/// Renumber the community ids in `comm` to a dense `0..k` range, ordered by
/// ascending first appearance in node order (determinism), and return the dense
/// label per node plus `k`. Community ids index a scratch vector of length
/// `comm.len()` (they are node indices), so every id is in range.
fn densify(comm: &[usize]) -> (Vec<usize>, usize) {
    let mut relabel = vec![usize::MAX; comm.len()];
    let mut next = 0;
    // Ascending old-id order: scan in node order and assign on first sight.
    for &c in comm {
        if relabel[c] == usize::MAX {
            relabel[c] = next;
            next += 1;
        }
    }
    (comm.iter().map(|&c| relabel[c]).collect(), next)
}

/// Collapse each refined sub-community into a super-node: intra off-diagonal
/// weight and carried self-loops become the super-node's self-loop; inter
/// weight becomes super-edges (stored in both directions). `labels` are the
/// dense `0..k` refined community ids per node.
fn aggregate(
    adj: &[Vec<(usize, f64)>],
    self_loops: &[f64],
    labels: &[usize],
    k: usize,
) -> (Vec<Vec<(usize, f64)>>, Vec<f64>) {
    let n = adj.len();
    let mut new_self = vec![0.0f64; k];
    // Inter-community weight, keyed by ordered pair (a < b).
    let mut inter: HashMap<(usize, usize), f64> = HashMap::new();

    for u in 0..n {
        let cu = labels[u];
        new_self[cu] += self_loops[u]; // carry existing self-loops
        for &(v, w) in &adj[u] {
            if u < v {
                // Count each undirected off-diagonal edge once.
                let cv = labels[v];
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
/// level-0 graph. `labels` need not be dense; they are renumbered here by
/// ascending smallest member index for a stable, reproducible labeling.
fn finalize(graph: &Graph, labels: &[usize], m: f64, resolution: f64) -> Communities {
    let (labels, num_communities) = densify(labels);

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
        labels,
        num_communities,
        modularity: q,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    /// Two triangles joined by a single bridge edge — the textbook planted
    /// two-community graph. Leiden must recover the two triangles.
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

    /// Every community's induced subgraph is connected (the Leiden guarantee
    /// Louvain does not make). Returns false if any community splits into two or
    /// more reachability islands over the graph's undirected adjacency.
    fn all_communities_connected(g: &Graph, c: &Communities) -> bool {
        let n = g.node_count();
        for label in 0..c.num_communities {
            let members: Vec<usize> = (0..n).filter(|&u| c.labels[u] == label).collect();
            if members.is_empty() {
                continue;
            }
            let in_c: std::collections::HashSet<usize> = members.iter().copied().collect();
            // BFS within the community from its first member.
            let mut seen = std::collections::HashSet::new();
            let mut stack = vec![members[0]];
            seen.insert(members[0]);
            while let Some(u) = stack.pop() {
                for &(v, _) in g.out_neighbors(u) {
                    if in_c.contains(&v) && seen.insert(v) {
                        stack.push(v);
                    }
                }
            }
            if seen.len() != members.len() {
                return false;
            }
        }
        true
    }

    #[test]
    fn empty_graph() {
        let c = leiden(&GraphBuilder::new().build(false), 1.0).unwrap();
        assert!(c.labels.is_empty());
        assert_eq!(c.num_communities, 0);
        assert_eq!(c.modularity, 0.0);
    }

    #[test]
    fn edgeless_graph_is_all_singletons() {
        let mut b = GraphBuilder::new();
        b.add_node("a");
        b.add_node("b");
        let c = leiden(&b.build(false), 1.0).unwrap();
        assert_eq!(c.num_communities, 2);
        assert_eq!(c.modularity, 0.0);
    }

    #[test]
    fn recovers_two_planted_triangles() {
        let g = two_triangles();
        let c = leiden(&g, 1.0).unwrap();
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
        assert!(all_communities_connected(&g, &c));
    }

    #[test]
    fn deterministic_across_runs() {
        let g = two_triangles();
        let a = leiden(&g, 1.0).unwrap();
        let b = leiden(&g, 1.0).unwrap();
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
        let c = leiden(&g, 1.0).unwrap();
        assert_eq!(c.num_communities, 2, "{c:?}");
        assert_eq!(group_of(&g, &c, "a"), group_of(&g, &c, "b"));
        assert_eq!(group_of(&g, &c, "c"), group_of(&g, &c, "d"));
        assert_ne!(group_of(&g, &c, "a"), group_of(&g, &c, "c"));
    }

    #[test]
    fn invalid_resolution_is_rejected() {
        let g = two_triangles();
        assert!(matches!(
            leiden(&g, 0.0),
            Err(GraphError::InvalidParameter(_))
        ));
        assert!(matches!(
            leiden(&g, -1.0),
            Err(GraphError::InvalidParameter(_))
        ));
        assert!(matches!(
            leiden(&g, f64::NAN),
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
        let c = leiden(&b.build(false), 1.0).unwrap();
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
            leiden(&b.build(false), 1.0),
            Err(GraphError::InvalidWeight(_))
        ));
    }

    #[test]
    fn single_clique_is_one_community() {
        // A triangle has no good split — Leiden should keep it whole.
        let mut b = GraphBuilder::new();
        b.add_edge("a", "b", 1.0).unwrap();
        b.add_edge("b", "c", 1.0).unwrap();
        b.add_edge("c", "a", 1.0).unwrap();
        let c = leiden(&b.build(false), 1.0).unwrap();
        assert_eq!(c.num_communities, 1);
    }

    #[test]
    fn four_planted_communities_stay_connected() {
        // Four 4-cliques in a ring, each joined to the next by one bridge edge.
        // Leiden should recover four dense communities, and — the property
        // Louvain can violate — every one must be internally connected.
        let mut b = GraphBuilder::new();
        let cliques = [
            ["a", "b", "c", "d"],
            ["e", "f", "g", "h"],
            ["i", "j", "k", "l"],
            ["m", "n", "o", "p"],
        ];
        for clique in &cliques {
            for i in 0..clique.len() {
                for j in (i + 1)..clique.len() {
                    b.add_edge(clique[i], clique[j], 1.0).unwrap();
                }
            }
        }
        // Ring of bridges between consecutive cliques.
        b.add_edge("a", "e", 1.0).unwrap();
        b.add_edge("e", "i", 1.0).unwrap();
        b.add_edge("i", "m", 1.0).unwrap();
        b.add_edge("m", "a", 1.0).unwrap();
        let g = b.build(false);
        let c = leiden(&g, 1.0).unwrap();
        assert_eq!(c.num_communities, 4, "{c:?}");
        assert!(all_communities_connected(&g, &c));
        // Each clique lands wholly in one community.
        for clique in &cliques {
            let first = group_of(&g, &c, clique[0]);
            for id in &clique[1..] {
                assert_eq!(group_of(&g, &c, id), first, "{id} split from {}", clique[0]);
            }
        }
    }
}
