//! Link prediction — score node pairs that are *not* currently connected by how
//! much their neighborhoods overlap, as candidate edges for a consumer's
//! second-pass confirmation. Treats the graph as **undirected** (neighbor =
//! incident node).
//!
//! Three standard scores over the common neighbors `N(u) ∩ N(v)`:
//! - **Common neighbors** — `|N(u) ∩ N(v)|`.
//! - **Jaccard** — `|N(u) ∩ N(v)| / |N(u) ∪ N(v)|`.
//! - **Adamic-Adar** — `Σ_{w ∈ N(u)∩N(v)} 1 / ln |N(w)|`, down-weighting hubs.
//!   (Any common neighbor `w` has degree ≥ 2, so `ln |N(w)| > 0`.)
//!
//! Already-adjacent pairs and pairs with no common neighbor are omitted (score 0).

use std::collections::{HashMap, HashSet};

use crate::graph::Graph;

/// Which neighborhood-overlap score to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkPredictionMethod {
    CommonNeighbors,
    Jaccard,
    AdamicAdar,
}

/// Neighbor sets per node (undirected adjacency, self excluded).
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

fn adamic_adar_term(degree: usize) -> f64 {
    1.0 / (degree as f64).ln()
}

/// Predict links from a single `source` to every non-adjacent node that shares
/// at least one neighbor with it. Returns `(target, score)` pairs (unordered).
pub fn link_prediction_from(
    graph: &Graph,
    source: usize,
    method: LinkPredictionMethod,
) -> Vec<(usize, f64)> {
    let nbr = neighbor_sets(graph);
    let src = &nbr[source];
    let mut out = Vec::new();
    for (t, t_nbr) in nbr.iter().enumerate() {
        if t == source || src.contains(&t) {
            continue;
        }
        let common: Vec<usize> = src.intersection(t_nbr).copied().collect();
        if common.is_empty() {
            continue;
        }
        out.push((t, pair_score(method, &common, src, t_nbr, &nbr)));
    }
    out
}

/// Predict links across every non-adjacent pair that shares a neighbor. Returns
/// `(u, v, score)` with `u < v`. Computed by, for each node `w`, crediting every
/// pair of `w`'s neighbors with a shared neighbor — O(Σ deg²), exact and fine for
/// the small graphs this serves.
pub fn link_prediction_all(
    graph: &Graph,
    method: LinkPredictionMethod,
) -> Vec<(usize, usize, f64)> {
    let nbr = neighbor_sets(graph);
    // Per unordered pair: (common-neighbor count, accumulated Adamic-Adar sum).
    let mut acc: HashMap<(usize, usize), (usize, f64)> = HashMap::new();
    for w_nbr in &nbr {
        if w_nbr.len() < 2 {
            continue;
        }
        let neighbors: Vec<usize> = w_nbr.iter().copied().collect();
        let aa = adamic_adar_term(w_nbr.len());
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let (u, v) = (
                    neighbors[i].min(neighbors[j]),
                    neighbors[i].max(neighbors[j]),
                );
                let entry = acc.entry((u, v)).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += aa;
            }
        }
    }

    let mut out = Vec::new();
    for ((u, v), (common, aa_sum)) in acc {
        if nbr[u].contains(&v) {
            continue; // already an edge
        }
        out.push((
            u,
            v,
            score_from(method, common, nbr[u].len(), nbr[v].len(), aa_sum),
        ));
    }
    out
}

fn pair_score(
    method: LinkPredictionMethod,
    common: &[usize],
    a: &HashSet<usize>,
    b: &HashSet<usize>,
    nbr: &[HashSet<usize>],
) -> f64 {
    // Only sum Adamic-Adar terms when that method is requested.
    let aa_sum = if method == LinkPredictionMethod::AdamicAdar {
        common.iter().map(|&w| adamic_adar_term(nbr[w].len())).sum()
    } else {
        0.0
    };
    score_from(method, common.len(), a.len(), b.len(), aa_sum)
}

/// Combine the resolved overlap quantities into the requested score. Single
/// definition shared by the single-source and all-pairs paths so they can't
/// drift. `aa_sum` is the pre-summed Adamic-Adar contribution (unused by the
/// other methods).
fn score_from(
    method: LinkPredictionMethod,
    common: usize,
    degree_u: usize,
    degree_v: usize,
    aa_sum: f64,
) -> f64 {
    match method {
        LinkPredictionMethod::CommonNeighbors => common as f64,
        LinkPredictionMethod::Jaccard => {
            let union = degree_u + degree_v - common;
            common as f64 / union as f64
        }
        LinkPredictionMethod::AdamicAdar => aa_sum,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, GraphBuilder};
    use LinkPredictionMethod::*;

    fn path_abcd() -> Graph {
        // a - b - c - d (undirected).
        let mut g = GraphBuilder::new();
        g.add_edge("a", "b", 1.0).unwrap();
        g.add_edge("b", "c", 1.0).unwrap();
        g.add_edge("c", "d", 1.0).unwrap();
        g.build(false)
    }

    #[test]
    fn from_source_common_neighbors() {
        let g = path_abcd();
        let a = g.idx_of("a").unwrap();
        let scored = link_prediction_from(&g, a, CommonNeighbors);
        // a's only non-adjacent node sharing a neighbor is c (via b).
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].0, g.idx_of("c").unwrap());
        assert_eq!(scored[0].1, 1.0);
    }

    #[test]
    fn from_source_adamic_adar_weights_by_degree() {
        let g = path_abcd();
        let a = g.idx_of("a").unwrap();
        let scored = link_prediction_from(&g, a, AdamicAdar);
        // Common neighbor b has degree 2 => 1/ln(2).
        assert!((scored[0].1 - 1.0 / 2f64.ln()).abs() < 1e-9, "{scored:?}");
    }

    #[test]
    fn from_source_jaccard() {
        let g = path_abcd();
        let a = g.idx_of("a").unwrap();
        let scored = link_prediction_from(&g, a, Jaccard);
        // N(a)={b}, N(c)={b,d}; common=1, union=|{b,d}|=2 => 0.5.
        assert!((scored[0].1 - 0.5).abs() < 1e-9, "{scored:?}");
    }

    #[test]
    fn all_pairs_star_predicts_leaf_pairs() {
        // Star hub-a, hub-b, hub-c: every leaf pair shares the hub.
        let mut g = GraphBuilder::new();
        g.add_edge("hub", "a", 1.0).unwrap();
        g.add_edge("hub", "b", 1.0).unwrap();
        g.add_edge("hub", "c", 1.0).unwrap();
        let g = g.build(false);
        let pairs = link_prediction_all(&g, CommonNeighbors);
        // 3 leaf pairs (a,b),(a,c),(b,c), each common=1; no hub-leaf pairs (adjacent).
        assert_eq!(pairs.len(), 3, "{pairs:?}");
        assert!(pairs.iter().all(|&(_, _, s)| s == 1.0));
    }

    #[test]
    fn all_pairs_adamic_adar_uses_hub_degree() {
        let mut g = GraphBuilder::new();
        g.add_edge("hub", "a", 1.0).unwrap();
        g.add_edge("hub", "b", 1.0).unwrap();
        g.add_edge("hub", "c", 1.0).unwrap();
        let g = g.build(false);
        let pairs = link_prediction_all(&g, AdamicAdar);
        // hub degree 3 => each pair scores 1/ln(3).
        assert!(
            pairs
                .iter()
                .all(|&(_, _, s)| (s - 1.0 / 3f64.ln()).abs() < 1e-9)
        );
    }

    #[test]
    fn adjacent_pairs_are_excluded() {
        // Triangle: every pair is already adjacent => no predictions.
        let mut g = GraphBuilder::new();
        g.add_edge("a", "b", 1.0).unwrap();
        g.add_edge("b", "c", 1.0).unwrap();
        g.add_edge("c", "a", 1.0).unwrap();
        let g = g.build(false);
        assert!(link_prediction_all(&g, CommonNeighbors).is_empty());
    }
}
