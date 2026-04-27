//! HNSW (Hierarchical Navigable Small World) index.
//!
//! A multi-layer graph structure for approximate nearest neighbor search.
//! Each layer is a navigable small-world graph where higher layers are
//! sparser (fewer nodes) and lower layers are denser. Search starts at
//! the top layer and greedily descends, using each layer to quickly
//! narrow the search space.
//!
//! Key parameters:
//! - M: max connections per node per layer (default 16)
//! - ef_construction: beam width during insertion (default 200)
//! - ef_search: beam width during query (default 50)
//! - ml: level generation factor (default 1/ln(M))

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use crate::distance::cosine_similarity;

/// Configuration for the HNSW index.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Vector dimensionality.
    pub dim: usize,
    /// Max connections per node per layer.
    pub m: usize,
    /// Max connections for layer 0 (typically 2*M).
    pub m0: usize,
    /// Beam width during construction.
    pub ef_construction: usize,
    /// Beam width during search.
    pub ef_search: usize,
    /// Level multiplier (1/ln(M)).
    pub ml: f64,
}

impl HnswConfig {
    pub fn new(dim: usize) -> Self {
        let m = 16;
        Self {
            dim,
            m,
            m0: m * 2,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f64).ln(),
        }
    }

    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.m0 = m * 2;
        self.ml = 1.0 / (m as f64).ln();
        self
    }

    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }
}

/// A search result: node ID + similarity score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
}

/// Internal node representation.
struct Node {
    id: String,
    vector: Vec<f32>,
    /// Connections per layer: layer_index -> list of neighbor node indices
    connections: Vec<Vec<usize>>,
    level: usize,
    /// Tombstone flag — tombstoned slots stay in `nodes` with their
    /// connections intact so neighbor indices in other nodes remain
    /// valid, but are excluded from search results and distance
    /// computations. `remove(id)` sets this; re-insert of the same id
    /// creates a fresh slot and doesn't reuse the tombstoned one.
    tombstoned: bool,
}

/// Live counters exposed by `HnswIndex::stats()`. Values are snapshots
/// at call time. All counters are monotonic (never decrease) except
/// `index_size` and `tombstoned_count`, which are gauges.
#[derive(Debug, Clone)]
pub struct HnswStats {
    pub searches_total: u64,
    pub search_duration_micros_sum: u64,
    pub inserts_total: u64,
    pub removes_total: u64,
    pub index_size: usize,
    pub tombstoned_count: usize,
}

/// The HNSW index.
pub struct HnswIndex {
    config: HnswConfig,
    nodes: Vec<Node>,
    /// Map from string ID to internal index. Tombstoned ids are removed
    /// from this map so `contains()` / `get_vector()` return the right
    /// answer after `remove()`.
    id_to_index: HashMap<String, usize>,
    /// Entry point (top-level node index). Always points at a non-
    /// tombstoned node while the index has any live nodes. `None` only
    /// when every node has been tombstoned (effectively empty).
    entry_point: Option<usize>,
    /// Current max level in the index
    max_level: usize,
    /// Running count of tombstoned slots (for metrics + live-size).
    tombstoned_count: AtomicUsize,
    searches_total: AtomicU64,
    search_duration_micros_sum: AtomicU64,
    inserts_total: AtomicU64,
    removes_total: AtomicU64,
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            id_to_index: HashMap::new(),
            entry_point: None,
            max_level: 0,
            tombstoned_count: AtomicUsize::new(0),
            searches_total: AtomicU64::new(0),
            search_duration_micros_sum: AtomicU64::new(0),
            inserts_total: AtomicU64::new(0),
            removes_total: AtomicU64::new(0),
        }
    }

    /// Vector dimensionality the index was constructed with. Every
    /// `insert` and `search` must use vectors of this length —
    /// callers validate against this rather than relying on the
    /// internal `assert!` in `insert` panicking.
    pub fn dim(&self) -> usize {
        self.config.dim
    }

    /// Convenience: build an index with default `HnswConfig` for the
    /// given dimensionality. Equivalent to
    /// `HnswIndex::new(HnswConfig::new(dim))`.
    pub fn with_dim(dim: usize) -> Self {
        Self::new(HnswConfig::new(dim))
    }

    /// Total slot count (includes tombstoned slots).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the index is empty. `true` when there are no live nodes,
    /// either because nothing has been inserted or because every slot
    /// has been tombstoned.
    pub fn is_empty(&self) -> bool {
        self.entry_point.is_none()
    }

    /// Number of live (non-tombstoned) vectors. Differs from `len()`
    /// once anything has been removed.
    pub fn live_count(&self) -> usize {
        self.nodes.len() - self.tombstoned_count.load(Ordering::Relaxed)
    }

    /// Snapshot of runtime counters for observability (tech-debt #47).
    pub fn stats(&self) -> HnswStats {
        HnswStats {
            searches_total: self.searches_total.load(Ordering::Relaxed),
            search_duration_micros_sum: self.search_duration_micros_sum.load(Ordering::Relaxed),
            inserts_total: self.inserts_total.load(Ordering::Relaxed),
            removes_total: self.removes_total.load(Ordering::Relaxed),
            index_size: self.live_count(),
            tombstoned_count: self.tombstoned_count.load(Ordering::Relaxed),
        }
    }

    /// Generate a random level for a new node.
    fn random_level(&self) -> usize {
        let r: f64 = rand::random();
        (-r.ln() * self.config.ml).floor() as usize
    }

    /// Insert a vector with an associated ID. If `id` already exists,
    /// the old slot is tombstoned and a new slot is created with
    /// freshly-computed layer neighbors — so the graph stays correctly
    /// linked after the vector changes (tech-debt #45). The old
    /// tombstoned slot becomes garbage until a future compaction.
    pub fn insert(&mut self, id: &str, vector: &[f32]) {
        assert_eq!(
            vector.len(),
            self.config.dim,
            "vector dim {} != config dim {}",
            vector.len(),
            self.config.dim
        );

        // Re-insert: tombstone the old slot so the new vector gets
        // freshly-computed neighbor connections. Overwriting in place
        // leaves connections pointing at the old content cluster — a
        // silent quality-drift bug when the vector moves far.
        if self.id_to_index.contains_key(id) {
            self.remove(id);
        }

        let new_level = self.random_level();
        let new_idx = self.nodes.len();

        let mut node = Node {
            id: id.to_string(),
            vector: vector.to_vec(),
            connections: vec![Vec::new(); new_level + 1],
            level: new_level,
            tombstoned: false,
        };

        self.id_to_index.insert(id.to_string(), new_idx);

        if self.entry_point.is_none() {
            // First node
            self.entry_point = Some(new_idx);
            self.max_level = new_level;
            self.nodes.push(node);
            self.inserts_total.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let entry = self.entry_point.unwrap();

        // Phase 1: Greedily traverse from top to new_level+1
        let mut current = entry;
        for level in (new_level + 1..=self.max_level).rev() {
            current = self.greedy_closest(current, &node.vector, level);
        }

        // Phase 2: Insert at each level from min(new_level, max_level) down to 0
        let insert_from = new_level.min(self.max_level);
        for level in (0..=insert_from).rev() {
            let ef = self.config.ef_construction;
            let neighbors = self.search_layer(current, &node.vector, ef, level);

            // Select M best neighbors
            let max_conn = if level == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let selected: Vec<usize> = neighbors
                .iter()
                .take(max_conn)
                .map(|&(idx, _)| idx)
                .collect();

            // Add bidirectional connections
            node.connections[level] = selected.clone();

            // We need to push the node first so we can modify neighbors
            // But node isn't in self.nodes yet — we'll fix connections after push
        }

        self.nodes.push(node);

        // Now fix bidirectional connections
        let insert_from = new_level.min(self.max_level);
        for level in 0..=insert_from {
            let max_conn = if level == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let connections = self.nodes[new_idx].connections[level].clone();
            for &neighbor_idx in &connections {
                if neighbor_idx >= self.nodes.len() {
                    continue;
                }
                if level >= self.nodes[neighbor_idx].connections.len() {
                    continue;
                }
                self.nodes[neighbor_idx].connections[level].push(new_idx);
                // Prune if over capacity
                if self.nodes[neighbor_idx].connections[level].len() > max_conn {
                    // Collect scores first (avoids borrow conflict)
                    let nv = self.nodes[neighbor_idx].vector.clone();
                    let conn = self.nodes[neighbor_idx].connections[level].clone();
                    let mut scored: Vec<(usize, f32)> = conn
                        .iter()
                        .map(|&idx| (idx, cosine_similarity(&nv, &self.nodes[idx].vector)))
                        .collect();
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    self.nodes[neighbor_idx].connections[level] = scored
                        .into_iter()
                        .take(max_conn)
                        .map(|(idx, _)| idx)
                        .collect();
                }
            }
        }

        // Update entry point if new node has higher level
        if new_level > self.max_level {
            self.entry_point = Some(new_idx);
            self.max_level = new_level;
        }

        self.inserts_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Tombstone the slot associated with `id`. Returns `true` if a live
    /// slot was tombstoned, `false` if `id` wasn't present. The old
    /// slot's `connections` stay intact so other nodes' neighbor
    /// indices remain valid; searches skip tombstoned nodes.
    ///
    /// If the tombstoned slot was the graph's entry point, a new entry
    /// is chosen by scanning for any remaining live node. When the last
    /// node is tombstoned, `entry_point` becomes `None` and the index
    /// effectively empties.
    pub fn remove(&mut self, id: &str) -> bool {
        let Some(&idx) = self.id_to_index.get(id) else {
            return false;
        };
        self.nodes[idx].tombstoned = true;
        self.id_to_index.remove(id);
        self.tombstoned_count.fetch_add(1, Ordering::Relaxed);
        self.removes_total.fetch_add(1, Ordering::Relaxed);

        if self.entry_point == Some(idx) {
            // Pick any surviving live node as the new entry. Walks the
            // nodes vec; O(n) in the size of the index. Remove is a
            // cold path compared to search so this is acceptable.
            self.entry_point = self
                .nodes
                .iter()
                .enumerate()
                .find(|(i, n)| *i != idx && !n.tombstoned)
                .map(|(i, _)| i);
            // Recompute max_level based on the new entry's level.
            self.max_level = self.entry_point.map(|i| self.nodes[i].level).unwrap_or(0);
        }

        true
    }

    /// Search for the k nearest neighbors to a query vector.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        if self.is_empty() {
            return Vec::new();
        }

        assert_eq!(
            query.len(),
            self.config.dim,
            "query dim {} != config dim {}",
            query.len(),
            self.config.dim
        );

        let start = Instant::now();
        let entry = self.entry_point.unwrap();

        // Phase 1: Greedily traverse from top to layer 1
        let mut current = entry;
        for level in (1..=self.max_level).rev() {
            current = self.greedy_closest(current, query, level);
        }

        // Phase 2: Search layer 0 with ef_search beam width
        let ef = self.config.ef_search.max(k);
        let candidates = self.search_layer(current, query, ef, 0);

        // Return top-k live results. Tombstoned candidates are filtered
        // here rather than in search_layer so the beam width (ef) still
        // considers them as traversal waypoints — their neighbor
        // connections remain valid paths even when the tombstoned node
        // itself shouldn't land in the result set.
        let results: Vec<SearchResult> = candidates
            .into_iter()
            .filter(|(idx, _)| !self.nodes[*idx].tombstoned)
            .take(k)
            .map(|(idx, score)| SearchResult {
                id: self.nodes[idx].id.clone(),
                score,
            })
            .collect();

        let elapsed_micros = start.elapsed().as_micros() as u64;
        self.searches_total.fetch_add(1, Ordering::Relaxed);
        self.search_duration_micros_sum
            .fetch_add(elapsed_micros, Ordering::Relaxed);

        results
    }

    /// Get the vector for a given ID.
    pub fn get_vector(&self, id: &str) -> Option<&[f32]> {
        self.id_to_index
            .get(id)
            .map(|&idx| self.nodes[idx].vector.as_slice())
    }

    /// Check if an ID exists in the index.
    pub fn contains(&self, id: &str) -> bool {
        self.id_to_index.contains_key(id)
    }

    /// Batch insert multiple vectors.
    pub fn insert_batch(&mut self, items: &[(&str, &[f32])]) {
        for &(id, vector) in items {
            self.insert(id, vector);
        }
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /// Greedily find the closest node to query at a given layer.
    fn greedy_closest(&self, start: usize, query: &[f32], level: usize) -> usize {
        let mut current = start;
        let mut best_score = cosine_similarity(query, &self.nodes[current].vector);

        loop {
            let mut improved = false;
            let connections = if level < self.nodes[current].connections.len() {
                &self.nodes[current].connections[level]
            } else {
                break;
            };

            for &neighbor in connections {
                let score = cosine_similarity(query, &self.nodes[neighbor].vector);
                if score > best_score {
                    best_score = score;
                    current = neighbor;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        current
    }

    /// Search a single layer with beam width ef, returning sorted (idx, score) pairs.
    fn search_layer(
        &self,
        entry: usize,
        query: &[f32],
        ef: usize,
        level: usize,
    ) -> Vec<(usize, f32)> {
        let entry_score = cosine_similarity(query, &self.nodes[entry].vector);

        // candidates: min-heap by score (worst first for easy eviction)
        // best: max-heap by score (best first for result)
        let mut visited = HashSet::new();
        visited.insert(entry);

        // We use a sorted vec approach for simplicity
        let mut candidates: Vec<(usize, f32)> = vec![(entry, entry_score)];
        let mut results: Vec<(usize, f32)> = vec![(entry, entry_score)];

        while let Some(&(current_idx, _current_score)) = candidates.last() {
            candidates.pop();

            // If current is worse than worst result and we have enough, stop
            if results.len() >= ef {
                let worst_result = results.iter().map(|r| r.1).fold(f32::MAX, f32::min);
                if _current_score < worst_result {
                    break;
                }
            }

            // Explore neighbors
            let connections = if level < self.nodes[current_idx].connections.len() {
                &self.nodes[current_idx].connections[level]
            } else {
                continue;
            };

            for &neighbor in connections {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                let score = cosine_similarity(query, &self.nodes[neighbor].vector);

                let should_add = results.len() < ef
                    || score > results.iter().map(|r| r.1).fold(f32::MAX, f32::min);

                if should_add {
                    candidates.push((neighbor, score));
                    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    results.push((neighbor, score));
                    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(dim: usize) -> HnswConfig {
        HnswConfig::new(dim)
            .with_ef_construction(100)
            .with_ef_search(50)
    }

    #[test]
    fn empty_index() {
        let index = HnswIndex::new(make_config(3));
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert!(index.search(&[1.0, 0.0, 0.0], 5).is_empty());
    }

    #[test]
    fn insert_and_search_single() {
        let mut index = HnswIndex::new(make_config(3));
        index.insert("a", &[1.0, 0.0, 0.0]);

        let results = index.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn insert_and_search_multiple() {
        let mut index = HnswIndex::new(make_config(3));
        index.insert("x_axis", &[1.0, 0.0, 0.0]);
        index.insert("y_axis", &[0.0, 1.0, 0.0]);
        index.insert("z_axis", &[0.0, 0.0, 1.0]);
        index.insert("xy_diag", &[0.707, 0.707, 0.0]);

        // Search for something close to x_axis
        let results = index.search(&[0.9, 0.1, 0.0], 2);
        assert_eq!(results.len(), 2);
        // x_axis should be closest
        assert_eq!(results[0].id, "x_axis");
        // xy_diag should be second
        assert_eq!(results[1].id, "xy_diag");
    }

    #[test]
    fn search_returns_k_results() {
        let mut index = HnswIndex::new(make_config(3));
        for i in 0..20 {
            let angle = (i as f32) * std::f32::consts::PI / 10.0;
            index.insert(&format!("v{}", i), &[angle.cos(), angle.sin(), 0.0]);
        }

        let results = index.search(&[1.0, 0.0, 0.0], 5);
        assert_eq!(results.len(), 5);
        // Results should be sorted by score descending
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }

    #[test]
    fn get_vector() {
        let mut index = HnswIndex::new(make_config(3));
        index.insert("test", &[1.0, 2.0, 3.0]);

        let v = index.get_vector("test").unwrap();
        assert_eq!(v, &[1.0, 2.0, 3.0]);

        assert!(index.get_vector("missing").is_none());
    }

    #[test]
    fn contains() {
        let mut index = HnswIndex::new(make_config(3));
        index.insert("exists", &[1.0, 0.0, 0.0]);

        assert!(index.contains("exists"));
        assert!(!index.contains("missing"));
    }

    #[test]
    fn update_existing_vector() {
        let mut index = HnswIndex::new(make_config(3));
        index.insert("a", &[1.0, 0.0, 0.0]);
        index.insert("a", &[0.0, 1.0, 0.0]); // update

        // Re-insert tombstones the old slot and creates a new one, so
        // total slot count is 2 but live count is 1. Contains/get_vector
        // see only the live slot. This is the tech-debt #45 fix: the
        // new vector gets re-linked into the HNSW graph, not just
        // overwritten in place.
        assert_eq!(index.len(), 2, "two slots (one tombstoned + one live)");
        assert_eq!(index.live_count(), 1, "only one live vector for id 'a'");
        assert!(index.contains("a"));
        let v = index.get_vector("a").unwrap();
        assert_eq!(v, &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn batch_insert() {
        let mut index = HnswIndex::new(make_config(3));
        index.insert_batch(&[
            ("a", &[1.0, 0.0, 0.0]),
            ("b", &[0.0, 1.0, 0.0]),
            ("c", &[0.0, 0.0, 1.0]),
        ]);

        assert_eq!(index.len(), 3);
    }

    #[test]
    fn high_dimensional_search() {
        let dim = 768; // typical embedding dimension
        let mut index = HnswIndex::new(make_config(dim));

        // Insert 100 random-ish vectors
        for i in 0..100 {
            let v: Vec<f32> = (0..dim)
                .map(|j| (((i * 7 + j * 13) % 100) as f32 / 100.0) - 0.5)
                .collect();
            index.insert(&format!("v{}", i), &v);
        }

        // Query with a known vector
        let query: Vec<f32> = (0..dim)
            .map(|j| ((j * 13) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let results = index.search(&query, 10);

        assert_eq!(results.len(), 10);
        // First result should be v0 (which has the same formula with i=0)
        assert_eq!(results[0].id, "v0");
        assert!(results[0].score > 0.9); // should be very similar
    }

    #[test]
    fn search_more_than_available() {
        let mut index = HnswIndex::new(make_config(3));
        index.insert("a", &[1.0, 0.0, 0.0]);
        index.insert("b", &[0.0, 1.0, 0.0]);

        // Ask for 10 but only 2 exist
        let results = index.search(&[1.0, 0.0, 0.0], 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn remove_drops_node_from_search_results() {
        let mut index = HnswIndex::new(make_config(3));
        index.insert("keep", &[1.0, 0.0, 0.0]);
        index.insert("drop", &[0.9, 0.4359, 0.0]); // ~0.9 cosine to query
        index.insert("far", &[0.0, 0.0, 1.0]);

        let before = index.search(&[1.0, 0.0, 0.0], 3);
        assert!(
            before.iter().any(|r| r.id == "drop"),
            "'drop' should appear in search results before remove: {:?}",
            before
        );

        assert!(index.remove("drop"));
        assert!(!index.contains("drop"));
        assert_eq!(index.live_count(), 2);
        assert_eq!(index.len(), 3, "slot retained as tombstone");

        let after = index.search(&[1.0, 0.0, 0.0], 3);
        assert!(
            !after.iter().any(|r| r.id == "drop"),
            "'drop' must not appear in search results after tombstone: {:?}",
            after
        );
        assert_eq!(after.len(), 2, "two live results remain");
    }

    #[test]
    fn remove_idempotent_and_returns_bool() {
        let mut index = HnswIndex::new(make_config(3));
        index.insert("x", &[1.0, 0.0, 0.0]);
        assert!(index.remove("x"));
        assert!(!index.remove("x"), "second remove of same id returns false");
        assert!(!index.remove("never_existed"));
    }

    #[test]
    fn remove_entry_point_picks_successor() {
        let mut index = HnswIndex::new(make_config(3));
        index.insert("a", &[1.0, 0.0, 0.0]);
        index.insert("b", &[0.0, 1.0, 0.0]);
        index.insert("c", &[0.0, 0.0, 1.0]);

        // Tombstone them one by one; search must keep working until
        // the last live node is gone.
        assert!(index.remove("a"));
        assert!(!index.search(&[1.0, 1.0, 0.0], 2).is_empty());
        assert!(index.remove("b"));
        assert!(!index.search(&[0.0, 0.0, 1.0], 1).is_empty());
        assert!(index.remove("c"));
        assert!(
            index.search(&[1.0, 0.0, 0.0], 1).is_empty(),
            "all-tombstoned index searches empty"
        );
        assert!(index.is_empty());
    }

    #[test]
    fn reinsert_after_remove_creates_fresh_slot() {
        let mut index = HnswIndex::new(make_config(3));
        index.insert("id", &[1.0, 0.0, 0.0]);
        index.remove("id");
        index.insert("id", &[0.0, 1.0, 0.0]);

        assert!(index.contains("id"));
        assert_eq!(index.get_vector("id").unwrap(), &[0.0, 1.0, 0.0]);
        assert_eq!(index.live_count(), 1);
        assert_eq!(index.len(), 2, "tombstoned slot + live slot");
    }

    #[test]
    fn stats_counters_track_operations() {
        let mut index = HnswIndex::new(make_config(3));
        let before = index.stats();
        assert_eq!(before.inserts_total, 0);
        assert_eq!(before.searches_total, 0);
        assert_eq!(before.removes_total, 0);
        assert_eq!(before.index_size, 0);

        index.insert("a", &[1.0, 0.0, 0.0]);
        index.insert("b", &[0.0, 1.0, 0.0]);
        let _ = index.search(&[1.0, 0.0, 0.0], 1);
        index.remove("a");
        // Re-insert with same id → one remove (from insert's duplicate path
        // would not fire here since 'a' was just removed) + one insert.
        index.insert("a", &[0.5, 0.5, 0.0]);

        let after = index.stats();
        assert_eq!(after.inserts_total, 3, "a, b, a again");
        assert_eq!(after.removes_total, 1, "just the explicit remove");
        assert_eq!(after.searches_total, 1);
        assert_eq!(after.index_size, 2, "a live, b live; first-a tombstoned");
        assert_eq!(after.tombstoned_count, 1);
    }
}
