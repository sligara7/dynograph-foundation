//! Key encoding for RocksDB column families.
//!
//! All keys use a consistent format that enables efficient prefix scans:
//! - Nodes: `{graph_id}\x00{node_type}\x00{node_id}`
//! - Edges: `{graph_id}\x00{edge_type}\x00{from_id}\x00{to_id}`
//! - Adjacency (out): `{graph_id}\x00{from_id}\x00{edge_type}\x00{to_id}`
//! - Adjacency (in): `{graph_id}\x00{to_id}\x00{edge_type}\x00{from_id}`
//! - Node index: `{graph_id}\x00{node_type}\x00{prop_name}\x00{prop_value}\x00{node_id}`

use dynograph_core::{DynoError, Value};

const SEP: u8 = 0x00;

/// Reject a string that would corrupt the key encoding.
///
/// Keys join their segments with `SEP` (NUL, `0x00`), so a segment
/// that itself contains NUL is indistinguishable from a field boundary
/// once encoded: a scan's `splitn(_, SEP)` splits it at the wrong
/// position, and an index value can forge a `(value, node_id)`
/// boundary — both silent
/// wrong-data hazards. Reject at write time (loudly) rather than
/// persist an ambiguous key. Only NUL is rejected: it is the sole byte
/// that breaks the encoding; other control bytes are stored verbatim.
pub fn validate_key_segment(field: &str, value: &str) -> Result<(), DynoError> {
    if value.as_bytes().contains(&SEP) {
        return Err(DynoError::InvalidKeySegment {
            field: field.to_string(),
            message: "must not contain a NUL byte (0x00), the key-segment separator".to_string(),
        });
    }
    Ok(())
}

/// Encode a node key.
pub fn node_key(graph_id: &str, node_type: &str, node_id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(graph_id.len() + node_type.len() + node_id.len() + 2);
    key.extend_from_slice(graph_id.as_bytes());
    key.push(SEP);
    key.extend_from_slice(node_type.as_bytes());
    key.push(SEP);
    key.extend_from_slice(node_id.as_bytes());
    key
}

/// Encode a prefix for scanning all nodes of a type in a graph.
pub fn node_type_prefix(graph_id: &str, node_type: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(graph_id.len() + node_type.len() + 2);
    key.extend_from_slice(graph_id.as_bytes());
    key.push(SEP);
    key.extend_from_slice(node_type.as_bytes());
    key.push(SEP);
    key
}

/// Encode an edge key.
pub fn edge_key(graph_id: &str, edge_type: &str, from_id: &str, to_id: &str) -> Vec<u8> {
    let mut key =
        Vec::with_capacity(graph_id.len() + edge_type.len() + from_id.len() + to_id.len() + 3);
    key.extend_from_slice(graph_id.as_bytes());
    key.push(SEP);
    key.extend_from_slice(edge_type.as_bytes());
    key.push(SEP);
    key.extend_from_slice(from_id.as_bytes());
    key.push(SEP);
    key.extend_from_slice(to_id.as_bytes());
    key
}

/// Encode an outgoing adjacency key.
pub fn adj_out_key(graph_id: &str, from_id: &str, edge_type: &str, to_id: &str) -> Vec<u8> {
    let mut key =
        Vec::with_capacity(graph_id.len() + from_id.len() + edge_type.len() + to_id.len() + 3);
    key.extend_from_slice(graph_id.as_bytes());
    key.push(SEP);
    key.extend_from_slice(from_id.as_bytes());
    key.push(SEP);
    key.extend_from_slice(edge_type.as_bytes());
    key.push(SEP);
    key.extend_from_slice(to_id.as_bytes());
    key
}

/// Encode an incoming adjacency key.
pub fn adj_in_key(graph_id: &str, to_id: &str, edge_type: &str, from_id: &str) -> Vec<u8> {
    let mut key =
        Vec::with_capacity(graph_id.len() + to_id.len() + edge_type.len() + from_id.len() + 3);
    key.extend_from_slice(graph_id.as_bytes());
    key.push(SEP);
    key.extend_from_slice(to_id.as_bytes());
    key.push(SEP);
    key.extend_from_slice(edge_type.as_bytes());
    key.push(SEP);
    key.extend_from_slice(from_id.as_bytes());
    key
}

/// Encode a prefix for scanning all outgoing edges from a node.
pub fn adj_out_prefix(graph_id: &str, from_id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(graph_id.len() + from_id.len() + 2);
    key.extend_from_slice(graph_id.as_bytes());
    key.push(SEP);
    key.extend_from_slice(from_id.as_bytes());
    key.push(SEP);
    key
}

/// Encode a prefix for scanning all incoming edges to a node.
pub fn adj_in_prefix(graph_id: &str, to_id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(graph_id.len() + to_id.len() + 2);
    key.extend_from_slice(graph_id.as_bytes());
    key.push(SEP);
    key.extend_from_slice(to_id.as_bytes());
    key.push(SEP);
    key
}

/// Encode a prefix for scanning *every* key belonging to one graph.
/// Used by `StorageEngine::clear_graph` to drop all of a graph's data
/// in a single range tombstone per CF. Includes the trailing separator
/// so e.g. graph "g1" doesn't sweep "g10".
pub fn graph_prefix(graph_id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(graph_id.len() + 1);
    key.extend_from_slice(graph_id.as_bytes());
    key.push(SEP);
    key
}

/// Canonical byte encoding of a property value for use in index keys.
///
/// Returns `None` for values that cannot be equality-indexed (lists, maps,
/// floats — floats omitted because bit-equality on f64 is a poor match for
/// user intent, and no indexed property today declares a float). Null is
/// treated as absent and also returns None.
pub fn value_to_index_bytes(value: &Value) -> Option<Vec<u8>> {
    match value {
        Value::String(s) => Some(s.as_bytes().to_vec()),
        Value::Int(n) => Some(n.to_string().into_bytes()),
        Value::Bool(b) => Some(if *b { b"1".to_vec() } else { b"0".to_vec() }),
        Value::Null | Value::Float(_) | Value::List(_) | Value::Map(_) => None,
        // Value is #[non_exhaustive]; new variants are non-indexable until
        // someone adds explicit handling here.
        _ => None,
    }
}

/// Encode an index key for a (node_type, property, value, node_id) tuple.
pub fn node_idx_key(
    graph_id: &str,
    node_type: &str,
    prop_name: &str,
    prop_value: &[u8],
    node_id: &str,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(
        graph_id.len() + node_type.len() + prop_name.len() + prop_value.len() + node_id.len() + 4,
    );
    key.extend_from_slice(graph_id.as_bytes());
    key.push(SEP);
    key.extend_from_slice(node_type.as_bytes());
    key.push(SEP);
    key.extend_from_slice(prop_name.as_bytes());
    key.push(SEP);
    key.extend_from_slice(prop_value);
    key.push(SEP);
    key.extend_from_slice(node_id.as_bytes());
    key
}

/// Prefix for scanning all nodes of a type where the given property equals
/// the given value. Used by `scan_nodes_by_property`.
pub fn node_idx_value_prefix(
    graph_id: &str,
    node_type: &str,
    prop_name: &str,
    prop_value: &[u8],
) -> Vec<u8> {
    let mut key = Vec::with_capacity(
        graph_id.len() + node_type.len() + prop_name.len() + prop_value.len() + 4,
    );
    key.extend_from_slice(graph_id.as_bytes());
    key.push(SEP);
    key.extend_from_slice(node_type.as_bytes());
    key.push(SEP);
    key.extend_from_slice(prop_name.as_bytes());
    key.push(SEP);
    key.extend_from_slice(prop_value);
    key.push(SEP);
    key
}

/// Extract the node_id suffix from an index key. The caller must pass the
/// exact value prefix used to scan.
pub fn node_idx_key_node_id<'a>(key: &'a [u8], value_prefix: &[u8]) -> Option<&'a [u8]> {
    key.strip_prefix(value_prefix)
}

/// Exclusive upper bound for a prefix-range scan or `delete_range_cf`.
/// Increments the last non-`0xFF` byte and truncates trailing `0xFF`s,
/// e.g. `[g, 1, 0x00]` → `[g, 1, 0x01]`. Returns `None` if every byte
/// in the prefix is `0xFF` (no bounded successor exists — caller must
/// fall back to per-key iteration).
// Only the RocksDB backend calls this (range tombstones); without that
// feature it's exercised solely by its own unit tests, so allow dead_code
// in a non-rocksdb production build rather than gate it (and its tests) out.
#[cfg_attr(not(feature = "rocksdb"), allow(dead_code))]
pub fn next_prefix(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    while let Some(last) = end.last_mut() {
        if *last < 0xFF {
            *last += 1;
            return Some(end);
        }
        end.pop();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_key_format() {
        let key = node_key("graph1", "Character", "abc-123");
        let expected = b"graph1\x00Character\x00abc-123";
        assert_eq!(key, expected);
    }

    #[test]
    fn node_type_prefix_enables_scan() {
        let prefix = node_type_prefix("graph1", "Character");
        let key1 = node_key("graph1", "Character", "aaa");
        let key2 = node_key("graph1", "Character", "zzz");
        let key3 = node_key("graph1", "Location", "aaa");

        assert!(key1.starts_with(&prefix));
        assert!(key2.starts_with(&prefix));
        assert!(!key3.starts_with(&prefix));
    }

    #[test]
    fn edge_key_format() {
        let key = edge_key("g1", "KNOWS", "alice", "bob");
        let expected = b"g1\x00KNOWS\x00alice\x00bob";
        assert_eq!(key, expected);
    }

    #[test]
    fn adj_out_prefix_scans_all_neighbors() {
        let prefix = adj_out_prefix("g1", "alice");
        let k1 = adj_out_key("g1", "alice", "KNOWS", "bob");
        let k2 = adj_out_key("g1", "alice", "VISITS", "tavern");
        let k3 = adj_out_key("g1", "bob", "KNOWS", "alice");

        assert!(k1.starts_with(&prefix));
        assert!(k2.starts_with(&prefix));
        assert!(!k3.starts_with(&prefix));
    }

    #[test]
    fn node_idx_key_format() {
        let k = node_idx_key("g1", "Fragment", "story_id", b"sA", "f1");
        let expected = b"g1\x00Fragment\x00story_id\x00sA\x00f1";
        assert_eq!(k, expected);
    }

    #[test]
    fn node_idx_value_prefix_selects_one_story() {
        let prefix = node_idx_value_prefix("g1", "Fragment", "story_id", b"sA");
        let k_same = node_idx_key("g1", "Fragment", "story_id", b"sA", "f1");
        let k_other_story = node_idx_key("g1", "Fragment", "story_id", b"sB", "f2");
        let k_other_prop = node_idx_key("g1", "Fragment", "arc_id", b"sA", "f3");
        let k_other_type = node_idx_key("g1", "Character", "story_id", b"sA", "c1");
        assert!(k_same.starts_with(&prefix));
        assert!(!k_other_story.starts_with(&prefix));
        assert!(!k_other_prop.starts_with(&prefix));
        assert!(!k_other_type.starts_with(&prefix));
    }

    #[test]
    fn value_to_index_bytes_covers_supported_types() {
        assert_eq!(
            value_to_index_bytes(&Value::String("abc".into())),
            Some(b"abc".to_vec())
        );
        assert_eq!(value_to_index_bytes(&Value::Int(42)), Some(b"42".to_vec()));
        assert_eq!(
            value_to_index_bytes(&Value::Bool(true)),
            Some(b"1".to_vec())
        );
        assert_eq!(
            value_to_index_bytes(&Value::Bool(false)),
            Some(b"0".to_vec())
        );
        assert_eq!(value_to_index_bytes(&Value::Null), None);
        assert_eq!(value_to_index_bytes(&Value::Float(1.0)), None);
        assert_eq!(value_to_index_bytes(&Value::List(vec![])), None);
    }

    #[test]
    fn next_prefix_increments_last_byte() {
        assert_eq!(next_prefix(b"g1\x00"), Some(b"g1\x01".to_vec()));
        assert_eq!(next_prefix(b"abc"), Some(b"abd".to_vec()));
    }

    #[test]
    fn next_prefix_carries_through_trailing_ff() {
        assert_eq!(next_prefix(&[b'a', 0xFF]), Some(vec![b'b']));
        assert_eq!(next_prefix(&[b'a', 0xFF, 0xFF]), Some(vec![b'b']));
    }

    #[test]
    fn next_prefix_all_ff_returns_none() {
        assert_eq!(next_prefix(&[0xFF, 0xFF]), None);
        assert_eq!(next_prefix(&[]), None);
    }

    #[test]
    fn next_prefix_bounds_a_real_scan() {
        // For prefix `g1\x00alice\x00`, every key starting with that
        // prefix must compare strictly less than next_prefix(prefix).
        let prefix = adj_out_prefix("g1", "alice");
        let end = next_prefix(&prefix).unwrap();
        let in_range = adj_out_key("g1", "alice", "KNOWS", "bob");
        let out_of_range = adj_out_key("g1", "alice2", "KNOWS", "bob");
        assert!(in_range.as_slice() < end.as_slice());
        assert!(out_of_range.as_slice() >= end.as_slice());
    }

    #[test]
    fn validate_key_segment_accepts_normal_and_control_but_rejects_nul() {
        assert!(validate_key_segment("node_id", "abc-123").is_ok());
        // Non-NUL control bytes don't break the encoding, so they pass.
        assert!(validate_key_segment("node_id", "a\tb\nc").is_ok());

        let err = validate_key_segment("node_id", "a\x00b").unwrap_err();
        assert!(
            matches!(err, DynoError::InvalidKeySegment { ref field, .. } if field == "node_id"),
            "{err:?}"
        );
    }

    #[test]
    fn nul_in_id_would_collide_without_the_guard() {
        // The hazard the guard prevents: a node_id containing NUL
        // encodes to the same bytes as a different (type, id) split.
        let sneaky = node_key("g1", "T", "a\x00b");
        let collision = node_key("g1", "T\x00a", "b");
        assert_eq!(
            sneaky, collision,
            "NUL in a segment makes two distinct tuples encode identically — \
             validate_key_segment must reject it before this can be stored"
        );
    }

    #[test]
    fn node_idx_key_node_id_extracts_suffix() {
        let prefix = node_idx_value_prefix("g1", "Fragment", "story_id", b"sA");
        let k = node_idx_key("g1", "Fragment", "story_id", b"sA", "f1");
        assert_eq!(node_idx_key_node_id(&k, &prefix), Some(b"f1".as_ref()));
    }
}

/// Property tests for the key encoding. The unit tests above pin a few
/// concrete shapes; these assert the two invariants the encoding lives
/// or dies by — across thousands of generated inputs:
///
///   1. **Round-trip** — splitting an encoded key on `SEP` recovers the
///      exact segments. Since the encoder is the only producer of `SEP`
///      bytes (segments are NUL-free), this is also a proof of
///      **injectivity**: two distinct tuples can't decode to one key, so
///      they can't encode to one either. That is the ST1 collision
///      guarantee (`validate_key_segment` enforces the NUL-free
///      precondition at write time).
///   2. **Prefix selection** — a scan prefix matches exactly the keys it
///      should and no others, and `next_prefix` is a correct exclusive
///      upper bound for the whole prefix range.
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// A NUL-free key segment — the precondition `validate_key_segment`
    /// enforces. Bounded length keeps cases small; the `[^\u{0}]` class
    /// still draws multibyte UTF-8 and non-NUL control bytes, the inputs
    /// most likely to trip a naive split. Empty is allowed (segments can
    /// legitimately be empty at this layer).
    fn nul_free_segment() -> impl Strategy<Value = String> {
        "[^\u{0}]{0,12}"
    }

    /// Split an encoded key into its `SEP`-separated segments.
    fn split_segments(key: &[u8]) -> Vec<&[u8]> {
        key.split(|&b| b == SEP).collect()
    }

    /// A string drawn from a tiny alphabet that includes NUL, so both
    /// the accept and reject branches of `validate_key_segment` get
    /// exercised (a fully-random `String` would essentially never draw
    /// U+0000).
    fn maybe_nul_string() -> impl Strategy<Value = String> {
        prop::collection::vec(
            prop_oneof![Just('\u{0}'), Just('a'), Just('z'), Just('\t')],
            0..10,
        )
        .prop_map(|cs| cs.into_iter().collect())
    }

    proptest! {
        /// The guard accepts a segment iff it contains no NUL — exactly
        /// the precondition every round-trip property below assumes.
        #[test]
        fn validate_key_segment_rejects_exactly_nul(s in maybe_nul_string()) {
            prop_assert_eq!(
                validate_key_segment("f", &s).is_ok(),
                !s.contains('\u{0}')
            );
        }

        #[test]
        fn node_key_round_trips(g in nul_free_segment(), t in nul_free_segment(), id in nul_free_segment()) {
            let key = node_key(&g, &t, &id);
            let parts = split_segments(&key);
            prop_assert_eq!(parts, vec![g.as_bytes(), t.as_bytes(), id.as_bytes()]);
        }

        #[test]
        fn edge_key_round_trips(g in nul_free_segment(), t in nul_free_segment(), from in nul_free_segment(), to in nul_free_segment()) {
            let key = edge_key(&g, &t, &from, &to);
            let parts = split_segments(&key);
            prop_assert_eq!(parts, vec![g.as_bytes(), t.as_bytes(), from.as_bytes(), to.as_bytes()]);
        }

        #[test]
        fn node_idx_key_round_trips(g in nul_free_segment(), t in nul_free_segment(), p in nul_free_segment(), v in nul_free_segment(), id in nul_free_segment()) {
            let key = node_idx_key(&g, &t, &p, v.as_bytes(), &id);
            let parts = split_segments(&key);
            prop_assert_eq!(
                parts,
                vec![g.as_bytes(), t.as_bytes(), p.as_bytes(), v.as_bytes(), id.as_bytes()]
            );
        }

        /// The headline ST1 property: distinct NUL-free node tuples never
        /// encode to the same bytes. (Implied by round-trip, asserted
        /// directly so a regression names the right thing.)
        #[test]
        fn distinct_node_tuples_never_collide(
            a in (nul_free_segment(), nul_free_segment(), nul_free_segment()),
            b in (nul_free_segment(), nul_free_segment(), nul_free_segment()),
        ) {
            prop_assume!(a != b);
            prop_assert_ne!(
                node_key(&a.0, &a.1, &a.2),
                node_key(&b.0, &b.1, &b.2)
            );
        }

        /// `node_type_prefix` selects exactly the keys of its (graph,
        /// type): every matching tuple is prefixed, and any tuple whose
        /// (graph, type) differs is not.
        #[test]
        fn node_type_prefix_selects_exactly(
            g in nul_free_segment(), t in nul_free_segment(), id in nul_free_segment(),
            g2 in nul_free_segment(), t2 in nul_free_segment(),
        ) {
            let prefix = node_type_prefix(&g, &t);
            prop_assert!(node_key(&g, &t, &id).starts_with(&prefix));
            prop_assume!((g2.as_str(), t2.as_str()) != (g.as_str(), t.as_str()));
            prop_assert!(!node_key(&g2, &t2, &id).starts_with(&prefix));
        }

        /// `next_prefix` is a correct exclusive upper bound: every key
        /// that begins with `prefix` sorts strictly below it.
        #[test]
        fn next_prefix_upper_bounds_the_range(
            prefix in prop::collection::vec(any::<u8>(), 1..16),
            suffix in prop::collection::vec(any::<u8>(), 0..16),
        ) {
            // All-0xFF prefixes have no bounded successor by design.
            prop_assume!(prefix.iter().any(|&b| b != 0xFF));
            let end = next_prefix(&prefix).expect("non-all-0xFF prefix has a successor");
            let mut key = prefix.clone();
            key.extend_from_slice(&suffix);
            prop_assert!(key.as_slice() < end.as_slice());
        }
    }
}
