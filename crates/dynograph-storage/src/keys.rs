//! Key encoding for RocksDB column families.
//!
//! All keys use a consistent format that enables efficient prefix scans:
//! - Nodes: `{graph_id}\x00{node_type}\x00{node_id}`
//! - Edges: `{graph_id}\x00{edge_type}\x00{from_id}\x00{to_id}`
//! - Adjacency (out): `{graph_id}\x00{from_id}\x00{edge_type}\x00{to_id}`
//! - Adjacency (in): `{graph_id}\x00{to_id}\x00{edge_type}\x00{from_id}`
//! - Node index: `{graph_id}\x00{node_type}\x00{prop_name}\x00{prop_value}\x00{node_id}`

use dynograph_core::Value;

const SEP: u8 = 0x00;

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

/// Prefix for checking whether any index entries exist for a
/// (node_type, property) pair. Used by lazy backfill to decide whether the
/// index has been populated yet.
pub fn node_idx_property_prefix(
    graph_id: &str,
    node_type: &str,
    prop_name: &str,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(graph_id.len() + node_type.len() + prop_name.len() + 3);
    key.extend_from_slice(graph_id.as_bytes());
    key.push(SEP);
    key.extend_from_slice(node_type.as_bytes());
    key.push(SEP);
    key.extend_from_slice(prop_name.as_bytes());
    key.push(SEP);
    key
}

/// Extract the node_id suffix from an index key. The caller must pass the
/// exact value prefix used to scan.
pub fn node_idx_key_node_id<'a>(key: &'a [u8], value_prefix: &[u8]) -> Option<&'a [u8]> {
    key.strip_prefix(value_prefix)
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
    fn node_idx_property_prefix_spans_all_values() {
        let prefix = node_idx_property_prefix("g1", "Fragment", "story_id");
        let k1 = node_idx_key("g1", "Fragment", "story_id", b"sA", "f1");
        let k2 = node_idx_key("g1", "Fragment", "story_id", b"sB", "f2");
        let k_other_prop = node_idx_key("g1", "Fragment", "arc_id", b"sA", "f3");
        assert!(k1.starts_with(&prefix));
        assert!(k2.starts_with(&prefix));
        assert!(!k_other_prop.starts_with(&prefix));
    }

    #[test]
    fn value_to_index_bytes_covers_supported_types() {
        assert_eq!(
            value_to_index_bytes(&Value::String("abc".into())),
            Some(b"abc".to_vec())
        );
        assert_eq!(value_to_index_bytes(&Value::Int(42)), Some(b"42".to_vec()));
        assert_eq!(value_to_index_bytes(&Value::Bool(true)), Some(b"1".to_vec()));
        assert_eq!(value_to_index_bytes(&Value::Bool(false)), Some(b"0".to_vec()));
        assert_eq!(value_to_index_bytes(&Value::Null), None);
        assert_eq!(value_to_index_bytes(&Value::Float(1.0)), None);
        assert_eq!(value_to_index_bytes(&Value::List(vec![])), None);
    }

    #[test]
    fn node_idx_key_node_id_extracts_suffix() {
        let prefix = node_idx_value_prefix("g1", "Fragment", "story_id", b"sA");
        let k = node_idx_key("g1", "Fragment", "story_id", b"sA", "f1");
        assert_eq!(node_idx_key_node_id(&k, &prefix), Some(b"f1".as_ref()));
    }
}
