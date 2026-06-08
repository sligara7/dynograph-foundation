//! Single-source shortest paths — the shared primitive behind closeness and
//! betweenness centrality. Computes, from one source, the distance to every
//! node, the number of shortest paths (`sigma`), the shortest-path predecessor
//! lists, and the finalization order — everything Brandes' betweenness
//! accumulation needs; closeness uses only `dist`.
//!
//! Unweighted graphs use BFS (every edge costs 1). Weighted graphs use Dijkstra
//! with the edge weight as a **cost**; callers must ensure costs are strictly
//! positive (the caller validates and returns a domain error otherwise) — with
//! zero/negative costs Dijkstra's finalization order would be wrong.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use crate::error::GraphError;
use crate::graph::Graph;

/// Shortest-path-cost weights must be strictly positive for Dijkstra to settle
/// nodes in the correct order. Shared by the path-based centrality measures
/// (closeness, betweenness).
pub(crate) fn require_positive_costs(graph: &Graph) -> Result<(), GraphError> {
    for u in 0..graph.node_count() {
        for &(_, w) in graph.out_neighbors(u) {
            if w <= 0.0 {
                return Err(GraphError::InvalidWeight(format!(
                    "closeness/betweenness weights are path costs and must be strictly positive, got {w}"
                )));
            }
        }
    }
    Ok(())
}

/// Tolerance for treating two path lengths as equal (shortest-path ties). BFS
/// distances are exact integers; for weighted graphs this absorbs FP rounding
/// when deciding whether an alternate path ties the current best.
const EPS: f64 = 1e-9;

/// Result of a single-source shortest-path computation.
pub(crate) struct Sssp {
    /// Distance from the source to each node; `f64::INFINITY` if unreachable.
    pub dist: Vec<f64>,
    /// Number of distinct shortest paths from the source to each node.
    pub sigma: Vec<f64>,
    /// Shortest-path predecessors of each node (nodes one hop before it on some
    /// shortest path from the source).
    pub preds: Vec<Vec<usize>>,
    /// Nodes in non-decreasing distance (finalization) order — the order
    /// Brandes' dependency accumulation walks in reverse.
    pub order: Vec<usize>,
}

/// Compute single-source shortest paths from `s`, following outgoing edges
/// (which for an undirected graph means all incident edges). Includes the path
/// counts (`sigma`), predecessor lists, and finalization order betweenness needs.
pub(crate) fn shortest_paths(g: &Graph, s: usize, weighted: bool) -> Sssp {
    if weighted {
        dijkstra(g, s, true)
    } else {
        bfs(g, s, true)
    }
}

/// Compute only the distance vector from `s` — for callers (closeness) that
/// don't need path counts/predecessors, skipping their per-source allocation
/// (notably the `Vec<Vec<usize>>` predecessor lists, one per source).
pub(crate) fn distances(g: &Graph, s: usize, weighted: bool) -> Vec<f64> {
    if weighted {
        dijkstra(g, s, false).dist
    } else {
        bfs(g, s, false).dist
    }
}

fn bfs(g: &Graph, s: usize, track_paths: bool) -> Sssp {
    let n = g.node_count();
    let mut dist = vec![f64::INFINITY; n];
    let mut sigma = vec![0.0; n];
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut order = Vec::new();

    dist[s] = 0.0;
    sigma[s] = 1.0;
    let mut queue = VecDeque::new();
    queue.push_back(s);

    while let Some(v) = queue.pop_front() {
        if track_paths {
            order.push(v);
        }
        for &(w, _cost) in g.out_neighbors(v) {
            // First discovery of w: set its distance and enqueue.
            if dist[w].is_infinite() {
                dist[w] = dist[v] + 1.0;
                queue.push_back(w);
            }
            // w found on (another) shortest path through v.
            if track_paths && (dist[w] - (dist[v] + 1.0)).abs() < EPS {
                sigma[w] += sigma[v];
                preds[w].push(v);
            }
        }
    }

    Sssp {
        dist,
        sigma,
        preds,
        order,
    }
}

fn dijkstra(g: &Graph, s: usize, track_paths: bool) -> Sssp {
    let n = g.node_count();
    let mut dist = vec![f64::INFINITY; n];
    let mut sigma = vec![0.0; n];
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut order = Vec::new();
    let mut done = vec![false; n];

    dist[s] = 0.0;
    sigma[s] = 1.0;
    let mut heap = BinaryHeap::new();
    heap.push(State { dist: 0.0, node: s });

    while let Some(State { dist: d, node: v }) = heap.pop() {
        // Lazy deletion: a node can appear multiple times; process its first
        // (smallest-distance) pop and skip the rest.
        if done[v] {
            continue;
        }
        done[v] = true;
        if track_paths {
            order.push(v);
        }

        for &(w, cost) in g.out_neighbors(v) {
            let new_dist = d + cost;
            if new_dist < dist[w] - EPS {
                // Strictly shorter path found: reset counts/preds.
                dist[w] = new_dist;
                if track_paths {
                    sigma[w] = sigma[v];
                    preds[w].clear();
                    preds[w].push(v);
                }
                heap.push(State {
                    dist: new_dist,
                    node: w,
                });
            } else if track_paths && (new_dist - dist[w]).abs() < EPS {
                // Tie: another shortest path to w through v. Because costs are
                // strictly positive, dist[v] < dist[w], so v is already
                // finalized — all of w's sigma is accumulated before it settles.
                sigma[w] += sigma[v];
                preds[w].push(v);
            }
        }
    }

    Sssp {
        dist,
        sigma,
        preds,
        order,
    }
}

/// Min-heap entry for Dijkstra. `BinaryHeap` is a max-heap, so `Ord` is reversed
/// on distance (smallest distance = greatest priority). Distances pushed here
/// are always finite, so the `partial_cmp` unwrap is safe.
struct State {
    dist: f64,
    node: usize,
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.dist == other.dist
    }
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse on distance for min-heap behavior; tie-break on node for a
        // total, deterministic order.
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
