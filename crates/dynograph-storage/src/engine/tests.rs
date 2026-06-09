use super::*;
use dynograph_core::{Schema, props};

fn test_schema() -> Schema {
    Schema::from_yaml(
        r#"
schema:
  name: test
  version: 1
  node_types:
    Character:
      properties:
        name:
          type: string
          required: true
        role:
          type: enum
          values: [protagonist, antagonist, supporting]
    Location:
      properties:
        name:
          type: string
          required: true
  edge_types:
    KNOWS:
      from: Character
      to: Character
    VISITS:
      from: Character
      to: Location
"#,
    )
    .unwrap()
}

/// Build a `StorageEngine` for a backend-agnostic engine test,
/// honoring the `DYNOGRAPH_TEST_BACKEND` env var: unset (or
/// `"memory"`) gives the in-memory backend, `"rocksdb"` a fresh
/// temp-dir RocksDB store. CI runs the storage suite once per
/// backend (the "backend matrix") so engine logic is exercised
/// against both — the in-memory string map and the array-indexed
/// column-family handles that diverge (the C1 batch panic only
/// reproduced on RocksDB, because every batch test was in-memory).
///
/// In rocksdb mode the temp dir is intentionally leaked via
/// `keep()`: the engine holds it open for the whole test and there's
/// no place to park the `TempDir` guard when returning only the
/// engine. CI runners are ephemeral; local rocksdb runs leave the
/// dirs under the system temp dir for the OS to reclaim. Tests that
/// are specifically about one backend keep calling
/// `new_in_memory` / `new_rocksdb` directly.
fn test_engine(schema: Schema) -> StorageEngine {
    match std::env::var("DYNOGRAPH_TEST_BACKEND").as_deref() {
        // Only honour the rocksdb backend when it's actually compiled in.
        // Without the feature the arm is removed, so a stray
        // `DYNOGRAPH_TEST_BACKEND=rocksdb` falls through to in-memory — the
        // only backend this build has — rather than panicking on the stub.
        #[cfg(feature = "rocksdb")]
        Ok("rocksdb") => {
            let dir = tempfile::tempdir()
                .expect("create temp dir for rocksdb test")
                .keep();
            let path = dir.to_str().expect("temp path is valid utf-8");
            StorageEngine::new_rocksdb(schema, path).expect("open rocksdb test engine")
        }
        _ => StorageEngine::new_in_memory(schema),
    }
}

// Backend-agnostic engine tests (run against both backends via the
// matrix — see `test_engine`).

#[test]
fn create_and_get_node() {
    let mut engine = test_engine(test_schema());
    let props = props! { "name" => "Alice", "role" => "protagonist" };
    let node = engine.create_node("g1", "Character", "c1", props).unwrap();
    assert_eq!(node.node_id, "c1");
    let fetched = engine.get_node("g1", "Character", "c1").unwrap().unwrap();
    assert_eq!(fetched.properties["name"].as_str().unwrap(), "Alice");
}

#[test]
fn create_node_rejects_nul_in_node_id() {
    let mut engine = test_engine(test_schema());
    let err = engine
        .create_node("g1", "Character", "c\x001", props! { "name" => "Alice" })
        .unwrap_err();
    assert!(
        matches!(err, DynoError::InvalidKeySegment { ref field, .. } if field == "node_id"),
        "{err:?}"
    );
    // The reject must happen before any put — no orphaned body left.
    assert_eq!(engine.count_nodes("g1", "Character").unwrap(), 0);
}

#[test]
fn create_node_rejects_nul_in_graph_id() {
    let mut engine = test_engine(test_schema());
    let err = engine
        .create_node("g\x001", "Character", "c1", props! { "name" => "Alice" })
        .unwrap_err();
    assert!(
        matches!(err, DynoError::InvalidKeySegment { ref field, .. } if field == "graph_id"),
        "{err:?}"
    );
}

#[test]
fn create_node_allows_nul_in_non_indexed_property_value() {
    // `name` is not indexed, so its value lives in the msgpack body,
    // never in a key — a NUL there is harmless and must be allowed.
    let mut engine = test_engine(test_schema());
    let node = engine
        .create_node("g1", "Character", "c1", props! { "name" => "a\x00b" })
        .unwrap();
    assert_eq!(node.properties["name"].as_str().unwrap(), "a\x00b");
    let fetched = engine.get_node("g1", "Character", "c1").unwrap().unwrap();
    assert_eq!(fetched.properties["name"].as_str().unwrap(), "a\x00b");
}

#[test]
fn create_node_rejects_nul_in_indexed_property_value() {
    // `story_id` IS indexed → its value becomes a key segment.
    let mut engine = test_engine(indexed_schema());
    let err = engine
        .create_node(
            "g1",
            "Fragment",
            "f1",
            props! { "name" => "A", "story_id" => "s\x00A" },
        )
        .unwrap_err();
    assert!(
        matches!(err, DynoError::InvalidKeySegment { ref field, .. } if field == "story_id"),
        "{err:?}"
    );
    assert_eq!(engine.count_nodes("g1", "Fragment").unwrap(), 0);
}

#[test]
fn replace_node_properties_rejects_nul_in_indexed_value() {
    let mut engine = test_engine(indexed_schema());
    engine
        .create_node(
            "g1",
            "Fragment",
            "f1",
            props! { "name" => "A", "story_id" => "sA" },
        )
        .unwrap();
    let err = engine
        .replace_node_properties(
            "g1",
            "Fragment",
            "f1",
            props! { "name" => "A", "story_id" => "s\x00B" },
        )
        .unwrap_err();
    assert!(
        matches!(err, DynoError::InvalidKeySegment { ref field, .. } if field == "story_id"),
        "{err:?}"
    );
}

#[test]
fn scan_nodes_by_property_rejects_nul_in_query_value() {
    // A NUL query value can't match any stored key, so without the
    // guard the scan would silently return empty rather than fail.
    let engine = test_engine(indexed_schema());
    let err = engine
        .scan_nodes_by_property(
            "g1",
            "Fragment",
            "story_id",
            &Value::String("s\x00A".into()),
        )
        .unwrap_err();
    assert!(
        matches!(err, DynoError::InvalidKeySegment { ref field, .. } if field == "story_id"),
        "{err:?}"
    );
}

#[test]
fn create_edge_rejects_nul_in_endpoint_id() {
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "alice", props! { "name" => "A" })
        .unwrap();
    let err = engine
        .create_edge(
            "g1",
            "KNOWS",
            "Character",
            "alice",
            "Character",
            "b\x00ob",
            HashMap::new(),
        )
        .unwrap_err();
    assert!(
        matches!(err, DynoError::InvalidKeySegment { ref field, .. } if field == "to_id"),
        "{err:?}"
    );
}

#[test]
fn create_node_validates_schema() {
    let mut engine = test_engine(test_schema());
    let result = engine.create_node("g1", "Character", "c1", HashMap::new());
    assert!(result.is_err());
}

#[test]
fn create_node_validates_enum() {
    let mut engine = test_engine(test_schema());
    let props = props! { "name" => "Bob", "role" => "villain" };
    assert!(engine.create_node("g1", "Character", "c1", props).is_err());
}

#[test]
fn get_nonexistent_node_returns_none() {
    let engine = test_engine(test_schema());
    assert!(
        engine
            .get_node("g1", "Character", "missing")
            .unwrap()
            .is_none()
    );
}

#[test]
fn delete_node() {
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
        .unwrap();
    assert!(engine.delete_node("g1", "Character", "c1").unwrap());
    assert!(engine.get_node("g1", "Character", "c1").unwrap().is_none());
    assert!(!engine.delete_node("g1", "Character", "c1").unwrap());
}

#[test]
fn delete_node_removes_outgoing_edges_and_peer_inverse_adjacency() {
    // Tech-debt C1 regression: delete_node used to leave dangling
    // CF_EDGES entries and inverse adjacency on neighbor nodes.
    // After deleting alice, bob's incoming-edge scan should be empty
    // and the alice→bob edge should be unresolvable.
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "alice", props! { "name" => "Alice" })
        .unwrap();
    engine
        .create_node("g1", "Character", "bob", props! { "name" => "Bob" })
        .unwrap();
    engine
        .create_edge(
            "g1",
            "KNOWS",
            "Character",
            "alice",
            "Character",
            "bob",
            HashMap::new(),
        )
        .unwrap();

    // Sanity: edge exists pre-delete.
    assert!(
        engine
            .get_edge("g1", "KNOWS", "alice", "bob")
            .unwrap()
            .is_some()
    );
    assert_eq!(
        engine.scan_incoming_edges("g1", "bob", None).unwrap().len(),
        1
    );

    engine.delete_node("g1", "Character", "alice").unwrap();

    // Edge no longer resolves from CF_EDGES.
    assert!(
        engine
            .get_edge("g1", "KNOWS", "alice", "bob")
            .unwrap()
            .is_none(),
        "edge should be gone after endpoint delete"
    );
    // bob's incoming-edge scan should not return the alice→bob entry.
    assert_eq!(
        engine.scan_incoming_edges("g1", "bob", None).unwrap().len(),
        0,
        "peer inverse adjacency should be cleaned up"
    );
}

#[test]
fn delete_node_removes_incoming_edges_and_peer_outgoing_adjacency() {
    // Symmetric case: delete the destination of an edge; the source's
    // outgoing-edge scan should no longer include the deleted endpoint.
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "alice", props! { "name" => "Alice" })
        .unwrap();
    engine
        .create_node("g1", "Character", "bob", props! { "name" => "Bob" })
        .unwrap();
    engine
        .create_edge(
            "g1",
            "KNOWS",
            "Character",
            "alice",
            "Character",
            "bob",
            HashMap::new(),
        )
        .unwrap();

    engine.delete_node("g1", "Character", "bob").unwrap();

    assert!(
        engine
            .get_edge("g1", "KNOWS", "alice", "bob")
            .unwrap()
            .is_none(),
    );
    assert_eq!(
        engine
            .scan_outgoing_edges("g1", "alice", None)
            .unwrap()
            .len(),
        0,
        "alice's outgoing-edge scan must not reference the deleted bob"
    );
}

#[test]
fn delete_node_with_mixed_incoming_and_outgoing_edges() {
    // alice has both an outgoing edge (alice → bob) and an incoming
    // edge (carol → alice via VISITS — Character VISITS Location;
    // we'll use a Location-typed `loc1` with a back-link via KNOWS
    // — but the test_schema only allows VISITS Character→Location.
    // So: alice → loc1 (VISITS), bob → alice (KNOWS).
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "alice", props! { "name" => "A" })
        .unwrap();
    engine
        .create_node("g1", "Character", "bob", props! { "name" => "B" })
        .unwrap();
    engine
        .create_node("g1", "Location", "loc1", props! { "name" => "Tavern" })
        .unwrap();
    engine
        .create_edge(
            "g1",
            "VISITS",
            "Character",
            "alice",
            "Location",
            "loc1",
            HashMap::new(),
        )
        .unwrap();
    engine
        .create_edge(
            "g1",
            "KNOWS",
            "Character",
            "bob",
            "Character",
            "alice",
            HashMap::new(),
        )
        .unwrap();

    engine.delete_node("g1", "Character", "alice").unwrap();

    // Both edges gone from CF_EDGES.
    assert!(
        engine
            .get_edge("g1", "VISITS", "alice", "loc1")
            .unwrap()
            .is_none()
    );
    assert!(
        engine
            .get_edge("g1", "KNOWS", "bob", "alice")
            .unwrap()
            .is_none()
    );
    // Loc1 has no incoming visits anymore.
    assert_eq!(
        engine
            .scan_incoming_edges("g1", "loc1", None)
            .unwrap()
            .len(),
        0
    );
    // Bob has no outgoing knows anymore.
    assert_eq!(
        engine.scan_outgoing_edges("g1", "bob", None).unwrap().len(),
        0
    );
}

#[test]
fn create_and_get_edge() {
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
        .unwrap();
    engine
        .create_node("g1", "Character", "c2", props! { "name" => "Bob" })
        .unwrap();
    let edge = engine
        .create_edge(
            "g1",
            "KNOWS",
            "Character",
            "c1",
            "Character",
            "c2",
            HashMap::new(),
        )
        .unwrap();
    assert_eq!(edge.edge_type, "KNOWS");
    let fetched = engine.get_edge("g1", "KNOWS", "c1", "c2").unwrap().unwrap();
    assert_eq!(fetched.from_id, "c1");
}

#[test]
fn edge_validates_types() {
    let mut engine = test_engine(test_schema());
    assert!(
        engine
            .create_edge(
                "g1",
                "KNOWS",
                "Location",
                "l1",
                "Character",
                "c1",
                HashMap::new()
            )
            .is_err()
    );
}

#[test]
fn cross_type_edge() {
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
        .unwrap();
    engine
        .create_node("g1", "Location", "loc1", props! { "name" => "Tavern" })
        .unwrap();
    let edge = engine
        .create_edge(
            "g1",
            "VISITS",
            "Character",
            "c1",
            "Location",
            "loc1",
            HashMap::new(),
        )
        .unwrap();
    assert_eq!(edge.edge_type, "VISITS");
}

#[test]
fn count_nodes_by_type() {
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
        .unwrap();
    engine
        .create_node("g1", "Character", "c2", props! { "name" => "Bob" })
        .unwrap();
    engine
        .create_node("g1", "Location", "loc1", props! { "name" => "Tavern" })
        .unwrap();
    assert_eq!(engine.count_nodes("g1", "Character").unwrap(), 2);
    assert_eq!(engine.count_nodes("g1", "Location").unwrap(), 1);
}

#[test]
fn graph_isolation() {
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
        .unwrap();
    engine
        .create_node("g2", "Character", "c1", props! { "name" => "Bob" })
        .unwrap();
    assert_eq!(
        engine
            .get_node("g1", "Character", "c1")
            .unwrap()
            .unwrap()
            .properties["name"]
            .as_str()
            .unwrap(),
        "Alice"
    );
    assert_eq!(
        engine
            .get_node("g2", "Character", "c1")
            .unwrap()
            .unwrap()
            .properties["name"]
            .as_str()
            .unwrap(),
        "Bob"
    );
}

// Reverse-index tests (CF_NODE_IDX)

fn indexed_schema() -> Schema {
    Schema::from_yaml(
        r#"
schema:
  name: test_indexed
  version: 1
  node_types:
    Fragment:
      properties:
        name: { type: string, required: true }
        story_id: { type: string, required: true, indexed: true }
    Character:
      properties:
        name: { type: string, required: true }
        story_id: { type: string, indexed: true }
  edge_types: {}
"#,
    )
    .unwrap()
}

#[test]
fn create_populates_index_and_scan_filters_by_value() {
    let mut engine = test_engine(indexed_schema());
    engine
        .create_node(
            "g1",
            "Fragment",
            "f1",
            props! { "name" => "A", "story_id" => "sA" },
        )
        .unwrap();
    engine
        .create_node(
            "g1",
            "Fragment",
            "f2",
            props! { "name" => "B", "story_id" => "sA" },
        )
        .unwrap();
    engine
        .create_node(
            "g1",
            "Fragment",
            "f3",
            props! { "name" => "C", "story_id" => "sB" },
        )
        .unwrap();

    let sid_a = Value::from("sA");
    let got_a = engine
        .scan_nodes_by_property("g1", "Fragment", "story_id", &sid_a)
        .unwrap();
    let mut ids: Vec<_> = got_a.iter().map(|n| n.node_id.clone()).collect();
    ids.sort();
    assert_eq!(ids, vec!["f1", "f2"]);

    let sid_b = Value::from("sB");
    let got_b = engine
        .scan_nodes_by_property("g1", "Fragment", "story_id", &sid_b)
        .unwrap();
    assert_eq!(got_b.len(), 1);
    assert_eq!(got_b[0].node_id, "f3");
}

#[test]
fn scan_filters_by_node_type() {
    // Same story_id across Fragment and Character — scan must not bleed types.
    let mut engine = test_engine(indexed_schema());
    engine
        .create_node(
            "g1",
            "Fragment",
            "f1",
            props! { "name" => "F", "story_id" => "sA" },
        )
        .unwrap();
    engine
        .create_node(
            "g1",
            "Character",
            "c1",
            props! { "name" => "C", "story_id" => "sA" },
        )
        .unwrap();

    let sid = Value::from("sA");
    let frags = engine
        .scan_nodes_by_property("g1", "Fragment", "story_id", &sid)
        .unwrap();
    assert_eq!(frags.len(), 1);
    assert_eq!(frags[0].node_id, "f1");
    assert_eq!(frags[0].node_type, "Fragment");

    let chars = engine
        .scan_nodes_by_property("g1", "Character", "story_id", &sid)
        .unwrap();
    assert_eq!(chars.len(), 1);
    assert_eq!(chars[0].node_id, "c1");
    assert_eq!(chars[0].node_type, "Character");
}

#[test]
fn update_moves_index_entry() {
    let mut engine = test_engine(indexed_schema());
    engine
        .create_node(
            "g1",
            "Fragment",
            "f1",
            props! { "name" => "A", "story_id" => "sA" },
        )
        .unwrap();

    // Reparent f1 from sA to sB.
    engine
        .replace_node_properties(
            "g1",
            "Fragment",
            "f1",
            props! { "name" => "A", "story_id" => "sB" },
        )
        .unwrap();

    let sid_a = Value::from("sA");
    let sid_b = Value::from("sB");
    assert_eq!(
        engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &sid_a)
            .unwrap()
            .len(),
        0,
        "old story_id should no longer match"
    );
    let hits_b = engine
        .scan_nodes_by_property("g1", "Fragment", "story_id", &sid_b)
        .unwrap();
    assert_eq!(hits_b.len(), 1);
    assert_eq!(hits_b[0].node_id, "f1");
}

#[test]
fn create_node_overwrite_reconciles_index() {
    // create-or-replace: re-creating an existing id with a different
    // indexed value must drop the old index entry, not leave it
    // dangling (which would make scan_nodes_by_property return the
    // node under its stale value).
    let mut engine = test_engine(indexed_schema());
    engine
        .create_node(
            "g1",
            "Fragment",
            "f1",
            props! { "name" => "A", "story_id" => "sA" },
        )
        .unwrap();
    // Overwrite the SAME id with a new story_id (bare create, not
    // replace_node_properties).
    engine
        .create_node(
            "g1",
            "Fragment",
            "f1",
            props! { "name" => "A", "story_id" => "sB" },
        )
        .unwrap();

    assert_eq!(
        engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &Value::from("sA"))
            .unwrap()
            .len(),
        0,
        "stale index entry under the old value must be gone"
    );
    let hits_b = engine
        .scan_nodes_by_property("g1", "Fragment", "story_id", &Value::from("sB"))
        .unwrap();
    assert_eq!(hits_b.len(), 1);
    assert_eq!(hits_b[0].node_id, "f1");
}

#[test]
fn delete_cleans_up_index_entries() {
    let mut engine = test_engine(indexed_schema());
    engine
        .create_node(
            "g1",
            "Fragment",
            "f1",
            props! { "name" => "A", "story_id" => "sA" },
        )
        .unwrap();

    assert!(engine.delete_node("g1", "Fragment", "f1").unwrap());

    let sid = Value::from("sA");
    let hits = engine
        .scan_nodes_by_property("g1", "Fragment", "story_id", &sid)
        .unwrap();
    assert_eq!(hits.len(), 0);
}

#[test]
fn non_indexed_property_returns_empty() {
    // `name` isn't declared indexed, so no CF_NODE_IDX entries are written
    // and scans against it see nothing.
    let mut engine = test_engine(indexed_schema());
    engine
        .create_node(
            "g1",
            "Fragment",
            "f1",
            props! { "name" => "Alice", "story_id" => "sA" },
        )
        .unwrap();

    let name = Value::from("Alice");
    let hits = engine
        .scan_nodes_by_property("g1", "Fragment", "name", &name)
        .unwrap();
    assert_eq!(hits.len(), 0);
}

#[test]
fn unsupported_value_types_return_empty() {
    let engine = test_engine(indexed_schema());
    assert_eq!(
        engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &Value::Float(1.0))
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &Value::Null)
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &Value::List(vec![]))
            .unwrap()
            .len(),
        0
    );
}

#[cfg(feature = "rocksdb")]
#[test]
fn index_survives_through_rocksdb_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_str().unwrap().to_string();

    {
        let mut engine = StorageEngine::new_rocksdb(indexed_schema(), &path).unwrap();
        engine
            .create_node(
                "g1",
                "Fragment",
                "f1",
                props! { "name" => "A", "story_id" => "sA" },
            )
            .unwrap();
        engine
            .create_node(
                "g1",
                "Fragment",
                "f2",
                props! { "name" => "B", "story_id" => "sB" },
            )
            .unwrap();
        engine
            .replace_node_properties(
                "g1",
                "Fragment",
                "f2",
                props! { "name" => "B", "story_id" => "sA" },
            )
            .unwrap();
        engine
            .create_node(
                "g1",
                "Fragment",
                "f3",
                props! { "name" => "C", "story_id" => "sA" },
            )
            .unwrap();
        engine.delete_node("g1", "Fragment", "f3").unwrap();
    }

    // Reopen. f1 and f2 should both be under sA; f3 should be gone.
    let engine = StorageEngine::new_rocksdb(indexed_schema(), &path).unwrap();
    let sid_a = Value::from("sA");
    let hits = engine
        .scan_nodes_by_property("g1", "Fragment", "story_id", &sid_a)
        .unwrap();
    let mut ids: Vec<_> = hits.iter().map(|n| n.node_id.clone()).collect();
    ids.sort();
    assert_eq!(ids, vec!["f1", "f2"]);

    let sid_b = Value::from("sB");
    let empty = engine
        .scan_nodes_by_property("g1", "Fragment", "story_id", &sid_b)
        .unwrap();
    assert_eq!(empty.len(), 0, "update should have removed sB entry for f2");
}

// RocksDB tests

#[cfg(feature = "rocksdb")]
#[test]
fn rocksdb_create_and_get_node() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine =
        StorageEngine::new_rocksdb(test_schema(), dir.path().to_str().unwrap()).unwrap();

    engine
        .create_node(
            "g1",
            "Character",
            "c1",
            props! { "name" => "Alice", "role" => "protagonist" },
        )
        .unwrap();
    let node = engine.get_node("g1", "Character", "c1").unwrap().unwrap();
    assert_eq!(node.properties["name"].as_str().unwrap(), "Alice");
}

#[cfg(feature = "rocksdb")]
#[test]
fn rocksdb_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_str().unwrap().to_string();

    // Write
    {
        let mut engine = StorageEngine::new_rocksdb(test_schema(), &path).unwrap();
        engine
            .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
            .unwrap();
        engine
            .create_node("g1", "Character", "c2", props! { "name" => "Bob" })
            .unwrap();
        engine
            .create_edge(
                "g1",
                "KNOWS",
                "Character",
                "c1",
                "Character",
                "c2",
                HashMap::new(),
            )
            .unwrap();
        // engine drops here, RocksDB flushes
    }

    // Re-open and verify data survived
    {
        let engine = StorageEngine::new_rocksdb(test_schema(), &path).unwrap();
        let alice = engine.get_node("g1", "Character", "c1").unwrap().unwrap();
        assert_eq!(alice.properties["name"].as_str().unwrap(), "Alice");
        let bob = engine.get_node("g1", "Character", "c2").unwrap().unwrap();
        assert_eq!(bob.properties["name"].as_str().unwrap(), "Bob");
        let edge = engine.get_edge("g1", "KNOWS", "c1", "c2").unwrap().unwrap();
        assert_eq!(edge.from_id, "c1");
        assert_eq!(engine.count_nodes("g1", "Character").unwrap(), 2);
    }
}

#[cfg(feature = "rocksdb")]
#[test]
fn rocksdb_scan_and_count() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine =
        StorageEngine::new_rocksdb(test_schema(), dir.path().to_str().unwrap()).unwrap();

    engine
        .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
        .unwrap();
    engine
        .create_node("g1", "Character", "c2", props! { "name" => "Bob" })
        .unwrap();
    engine
        .create_node("g1", "Location", "loc1", props! { "name" => "Tavern" })
        .unwrap();

    assert_eq!(engine.count_nodes("g1", "Character").unwrap(), 2);
    assert_eq!(engine.count_nodes("g1", "Location").unwrap(), 1);

    let chars = engine.scan_nodes("g1", "Character").unwrap();
    assert_eq!(chars.len(), 2);
}

#[cfg(feature = "rocksdb")]
#[test]
fn rocksdb_delete_node() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine =
        StorageEngine::new_rocksdb(test_schema(), dir.path().to_str().unwrap()).unwrap();

    engine
        .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
        .unwrap();
    assert!(engine.delete_node("g1", "Character", "c1").unwrap());
    assert!(engine.get_node("g1", "Character", "c1").unwrap().is_none());
}

#[cfg(feature = "rocksdb")]
#[test]
fn rocksdb_outgoing_edges() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine =
        StorageEngine::new_rocksdb(test_schema(), dir.path().to_str().unwrap()).unwrap();

    engine
        .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
        .unwrap();
    engine
        .create_node("g1", "Character", "c2", props! { "name" => "Bob" })
        .unwrap();
    engine
        .create_node("g1", "Location", "loc1", props! { "name" => "Tavern" })
        .unwrap();
    engine
        .create_edge(
            "g1",
            "KNOWS",
            "Character",
            "c1",
            "Character",
            "c2",
            HashMap::new(),
        )
        .unwrap();
    engine
        .create_edge(
            "g1",
            "VISITS",
            "Character",
            "c1",
            "Location",
            "loc1",
            HashMap::new(),
        )
        .unwrap();

    let all = engine.scan_outgoing_edges("g1", "c1", None).unwrap();
    assert_eq!(all.len(), 2);

    let knows = engine
        .scan_outgoing_edges("g1", "c1", Some("KNOWS"))
        .unwrap();
    assert_eq!(knows.len(), 1);
    assert_eq!(knows[0].to_id, "c2");
}

// -- Tech-debt C4 regression tests: batch atomicity for mixed put + delete

#[test]
fn batch_buffers_mixed_put_and_delete_visible_within_batch() {
    let mut engine = test_engine(test_schema());
    // Pre-existing node we'll delete inside the batch.
    engine
        .create_node("g1", "Character", "to_delete", props! { "name" => "Goner" })
        .unwrap();

    engine.begin_batch();
    engine
        .create_node("g1", "Character", "to_create", props! { "name" => "New" })
        .unwrap();
    engine.delete_node("g1", "Character", "to_delete").unwrap();

    // Within-batch reads see buffered ops — read-your-own-writes.
    assert!(
        engine
            .get_node("g1", "Character", "to_create")
            .unwrap()
            .is_some(),
        "buffered create should be visible within batch (read-your-own-writes)"
    );
    assert!(
        engine
            .get_node("g1", "Character", "to_delete")
            .unwrap()
            .is_none(),
        "buffered delete should be visible within batch (read-your-own-writes)"
    );

    engine.commit_batch().unwrap();

    // Post-commit: both ops applied atomically — same view as within-batch.
    assert!(
        engine
            .get_node("g1", "Character", "to_create")
            .unwrap()
            .is_some()
    );
    assert!(
        engine
            .get_node("g1", "Character", "to_delete")
            .unwrap()
            .is_none()
    );
}

#[test]
fn batch_discard_with_mixed_ops_leaves_pre_batch_state() {
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "keep_me", props! { "name" => "Stay" })
        .unwrap();

    engine.begin_batch();
    engine
        .create_node("g1", "Character", "ghost", props! { "name" => "Phantom" })
        .unwrap();
    engine.delete_node("g1", "Character", "keep_me").unwrap();
    engine.discard_batch();

    // Discarded ops must not affect disk.
    assert!(
        engine
            .get_node("g1", "Character", "ghost")
            .unwrap()
            .is_none()
    );
    assert!(
        engine
            .get_node("g1", "Character", "keep_me")
            .unwrap()
            .is_some()
    );
}

#[test]
fn batched_replace_on_indexed_node_visible_to_index_scans() {
    // Replace on an indexed node applies an index delete + put
    // atomically. v0.5.5+ buffer-aware scans surface the swapped
    // index state mid-batch; the within-batch view matches the
    // post-commit view (with the only difference being whether
    // the change has hit disk).
    let mut engine = test_engine(indexed_schema());
    engine
        .create_node(
            "g1",
            "Fragment",
            "f1",
            props! { "name" => "A", "story_id" => "sA" },
        )
        .unwrap();

    engine.begin_batch();
    engine
        .replace_node_properties(
            "g1",
            "Fragment",
            "f1",
            props! { "name" => "A", "story_id" => "sB" },
        )
        .unwrap();

    // Within-batch: index reflects the swap — f1 lives under sB,
    // sA index entry is gone (read-your-own-writes).
    let sa = Value::from("sA");
    let sb = Value::from("sB");
    assert_eq!(
        engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &sa)
            .unwrap()
            .len(),
        0,
        "old story_id index entry should be tombstoned within batch"
    );
    let hits_b_in_batch = engine
        .scan_nodes_by_property("g1", "Fragment", "story_id", &sb)
        .unwrap();
    assert_eq!(
        hits_b_in_batch.len(),
        1,
        "new story_id index entry should be visible within batch"
    );
    assert_eq!(hits_b_in_batch[0].node_id, "f1");

    engine.commit_batch().unwrap();

    // Post-commit: same view as within-batch.
    assert_eq!(
        engine
            .scan_nodes_by_property("g1", "Fragment", "story_id", &sa)
            .unwrap()
            .len(),
        0
    );
    let hits_b = engine
        .scan_nodes_by_property("g1", "Fragment", "story_id", &sb)
        .unwrap();
    assert_eq!(hits_b.len(), 1);
    assert_eq!(hits_b[0].node_id, "f1");
}

#[test]
fn batch_prefix_delete_visible_within_batch() {
    // delete_node uses prefix_delete on adj_out/adj_in. Pre-v0.5.5
    // these tombstones were invisible to scans until commit; v0.5.5+
    // overlays the buffer on scans, so the deletion is observable
    // mid-batch.
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "alice", props! { "name" => "A" })
        .unwrap();
    engine
        .create_node("g1", "Character", "bob", props! { "name" => "B" })
        .unwrap();
    engine
        .create_edge(
            "g1",
            "KNOWS",
            "Character",
            "alice",
            "Character",
            "bob",
            HashMap::new(),
        )
        .unwrap();

    engine.begin_batch();
    engine.delete_node("g1", "Character", "alice").unwrap();

    // Within-batch reads see the buffered prefix-delete: alice is
    // gone, her outgoing adjacency is empty, bob's incoming is empty.
    assert!(
        engine
            .get_node("g1", "Character", "alice")
            .unwrap()
            .is_none(),
        "buffered delete_node should be visible within batch"
    );
    assert_eq!(
        engine
            .scan_outgoing_edges("g1", "alice", None)
            .unwrap()
            .len(),
        0,
        "buffered prefix_delete on adj_out should be visible within batch"
    );
    assert_eq!(
        engine.scan_incoming_edges("g1", "bob", None).unwrap().len(),
        0,
        "buffered prefix_delete on adj_in should be visible within batch"
    );

    engine.commit_batch().unwrap();

    // Post-commit: same view as within-batch.
    assert!(
        engine
            .get_node("g1", "Character", "alice")
            .unwrap()
            .is_none()
    );
    assert!(
        engine
            .get_edge("g1", "KNOWS", "alice", "bob")
            .unwrap()
            .is_none()
    );
    assert_eq!(
        engine.scan_incoming_edges("g1", "bob", None).unwrap().len(),
        0
    );
}

/// New v0.5.5 positive tests for buffer-aware reads. Each exercises
/// one of the read paths against an active batch: get + scan + the
/// PrefixDelete-then-Put resurrection corner.

#[test]
fn buffer_aware_get_sees_in_batch_create() {
    let mut engine = test_engine(test_schema());
    engine.begin_batch();
    engine
        .create_node("g1", "Character", "alice", props! { "name" => "Alice" })
        .unwrap();
    let n = engine
        .get_node("g1", "Character", "alice")
        .unwrap()
        .expect("buffered create must be readable mid-batch");
    assert_eq!(
        n.properties.get("name").and_then(|v| v.as_str()),
        Some("Alice")
    );
}

#[test]
fn buffer_aware_get_sees_in_batch_delete_of_existing_node() {
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "alice", props! { "name" => "Alice" })
        .unwrap();
    engine.begin_batch();
    engine.delete_node("g1", "Character", "alice").unwrap();
    assert!(
        engine
            .get_node("g1", "Character", "alice")
            .unwrap()
            .is_none(),
        "buffered delete must shadow the backend node"
    );
}

#[test]
fn buffer_aware_scan_overlays_in_batch_creates_and_deletes() {
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "pre", props! { "name" => "Pre" })
        .unwrap();
    engine.begin_batch();
    engine
        .create_node("g1", "Character", "new", props! { "name" => "New" })
        .unwrap();
    engine.delete_node("g1", "Character", "pre").unwrap();

    let mut nodes = engine.scan_nodes("g1", "Character").unwrap();
    nodes.sort_by(|a, b| a.node_id.cmp(&b.node_id));
    assert_eq!(nodes.len(), 1, "expected only the in-batch new node");
    assert_eq!(nodes[0].node_id, "new");
}

#[test]
fn buffer_aware_get_late_put_resurrects_after_prefix_delete() {
    // [delete_node X (prefix-delete on adj), Put X again] — the
    // late Put wins. Ordering is preserved end-to-end at commit;
    // mid-batch reads must see the same final state.
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "phoenix", props! { "name" => "v1" })
        .unwrap();
    engine.begin_batch();
    engine.delete_node("g1", "Character", "phoenix").unwrap();
    engine
        .create_node("g1", "Character", "phoenix", props! { "name" => "v2" })
        .unwrap();
    let n = engine
        .get_node("g1", "Character", "phoenix")
        .unwrap()
        .expect("late put after delete must resurrect the key");
    assert_eq!(
        n.properties.get("name").and_then(|v| v.as_str()),
        Some("v2"),
        "late put's value must win over earlier delete"
    );
}

#[test]
fn buffer_aware_discard_keeps_backend_unchanged() {
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "keep", props! { "name" => "Keep" })
        .unwrap();
    engine.begin_batch();
    engine
        .create_node("g1", "Character", "ghost", props! { "name" => "Ghost" })
        .unwrap();
    engine.delete_node("g1", "Character", "keep").unwrap();
    // Within-batch view: ghost present, keep gone.
    assert!(
        engine
            .get_node("g1", "Character", "ghost")
            .unwrap()
            .is_some()
    );
    assert!(
        engine
            .get_node("g1", "Character", "keep")
            .unwrap()
            .is_none()
    );
    engine.discard_batch();
    // Post-discard view: backend unchanged. Ghost never existed,
    // keep is still there.
    assert!(
        engine
            .get_node("g1", "Character", "ghost")
            .unwrap()
            .is_none()
    );
    assert!(
        engine
            .get_node("g1", "Character", "keep")
            .unwrap()
            .is_some()
    );
}

#[test]
fn replace_schema_swaps_validation_rules() {
    // Old schema: Person has `name: string, required`. New schema:
    // adds `age: int, required` with a default. Verify (a) the
    // schema accessor returns the new shape and (b) writes
    // post-swap apply the new validation rules (default applied
    // for `age`).
    let old = Schema::from_yaml(
        r#"
schema:
  name: t
  version: 1
  node_types:
    Person:
      properties:
        name: { type: string, required: true }
  edge_types: {}
"#,
    )
    .unwrap();
    let new = Schema::from_yaml(
        r#"
schema:
  name: t
  version: 2
  node_types:
    Person:
      properties:
        name: { type: string, required: true }
        age:  { type: int,    required: true, default: 0 }
  edge_types: {}
"#,
    )
    .unwrap();

    let mut engine = test_engine(old);
    engine
        .create_node("g1", "Person", "alice", HashMap::new())
        .unwrap_err(); // missing required `name` — sanity check old schema in force

    engine.replace_schema(new);
    assert_eq!(engine.schema().version, 2);

    // New schema's `age` default applies on create — `name` still
    // required.
    let mut props = HashMap::new();
    props.insert("name".to_string(), Value::String("alice".into()));
    engine.create_node("g1", "Person", "alice", props).unwrap();
    let stored = engine.get_node("g1", "Person", "alice").unwrap().unwrap();
    assert_eq!(stored.properties.get("age"), Some(&Value::Int(0)));
}

fn embedding_schema() -> Schema {
    Schema::from_yaml(
        r#"
schema:
  name: t
  version: 1
  node_types:
    Item:
      properties:
        name: { type: string, required: true }
  edge_types: {}
"#,
    )
    .unwrap()
}

fn make_item(engine: &mut StorageEngine, id: &str) {
    let mut props = HashMap::new();
    props.insert("name".to_string(), Value::String(id.into()));
    engine.create_node("g1", "Item", id, props).unwrap();
}

#[test]
fn embedding_round_trip() {
    let mut engine = test_engine(embedding_schema());
    make_item(&mut engine, "n1");
    let v = vec![1.0_f32, 2.0, 3.0, 4.0];
    engine.set_embedding("g1", "Item", "n1", &v).unwrap();
    let got = engine
        .get_embedding("g1", "Item", "n1")
        .unwrap()
        .expect("embedding present");
    assert_eq!(got, v);
}

#[test]
fn embedding_overwrites_in_place() {
    let mut engine = test_engine(embedding_schema());
    make_item(&mut engine, "n1");
    engine
        .set_embedding("g1", "Item", "n1", &[1.0, 2.0])
        .unwrap();
    engine
        .set_embedding("g1", "Item", "n1", &[9.0, 8.0, 7.0])
        .unwrap();
    let got = engine.get_embedding("g1", "Item", "n1").unwrap().unwrap();
    assert_eq!(got, vec![9.0, 8.0, 7.0]);
}

#[test]
fn set_embedding_on_missing_node_errors_loudly() {
    let mut engine = test_engine(embedding_schema());
    let err = engine
        .set_embedding("g1", "Item", "ghost", &[1.0, 2.0])
        .unwrap_err();
    assert!(matches!(
        err,
        DynoError::NodeNotFound { ref node_type, ref node_id }
            if node_type == "Item" && node_id == "ghost"
    ));
}

#[test]
fn set_embedding_rejects_empty_vector() {
    let mut engine = test_engine(embedding_schema());
    make_item(&mut engine, "n1");
    let err = engine.set_embedding("g1", "Item", "n1", &[]).unwrap_err();
    assert!(matches!(err, DynoError::Validation { .. }));
}

#[test]
fn set_embedding_rejects_non_finite() {
    let mut engine = test_engine(embedding_schema());
    make_item(&mut engine, "n1");
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = engine
            .set_embedding("g1", "Item", "n1", &[1.0, bad, 3.0])
            .unwrap_err();
        assert!(
            matches!(err, DynoError::Validation { ref message, .. } if message.contains("non-finite")),
            "{bad} should be rejected loudly, got {err:?}"
        );
    }
    // Nothing was persisted — the reject ran before the write.
    assert!(engine.get_embedding("g1", "Item", "n1").unwrap().is_none());
}

#[test]
fn decode_embedding_rejects_non_finite() {
    // NaN bytes (exponent all ones, non-zero mantissa) on the read
    // path must fail loud rather than yield a poison float.
    let nan_bytes = f32::NAN.to_le_bytes();
    let err = decode_embedding(&nan_bytes).unwrap_err();
    assert!(
        matches!(err, DynoError::Storage(ref m) if m.contains("non-finite")),
        "{err:?}"
    );
}

#[test]
fn delete_embedding_returns_existed_bool() {
    let mut engine = test_engine(embedding_schema());
    make_item(&mut engine, "n1");
    engine.set_embedding("g1", "Item", "n1", &[1.0]).unwrap();
    assert!(engine.delete_embedding("g1", "Item", "n1").unwrap());
    // Idempotent — second delete returns false but is not an error.
    assert!(!engine.delete_embedding("g1", "Item", "n1").unwrap());
    // And the embedding is gone for real.
    assert!(engine.get_embedding("g1", "Item", "n1").unwrap().is_none());
}

#[test]
fn delete_node_cascades_to_embedding() {
    let mut engine = test_engine(embedding_schema());
    make_item(&mut engine, "n1");
    engine
        .set_embedding("g1", "Item", "n1", &[0.5, 0.5])
        .unwrap();
    engine.delete_node("g1", "Item", "n1").unwrap();
    assert!(engine.get_embedding("g1", "Item", "n1").unwrap().is_none());
}

#[test]
fn scan_embeddings_by_type_returns_all_and_only_that_type() {
    let mut engine = test_engine(embedding_schema());
    for id in ["a", "b", "c"] {
        make_item(&mut engine, id);
    }
    engine
        .set_embedding("g1", "Item", "a", &[1.0, 0.0])
        .unwrap();
    engine
        .set_embedding("g1", "Item", "b", &[0.0, 1.0])
        .unwrap();
    // c has no embedding.
    let mut scanned = engine.scan_embeddings_by_type("g1", "Item").unwrap();
    scanned.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(scanned.len(), 2);
    assert_eq!(scanned[0].0, "a");
    assert_eq!(scanned[0].1, vec![1.0, 0.0]);
    assert_eq!(scanned[1].0, "b");
    assert_eq!(scanned[1].1, vec![0.0, 1.0]);
}

#[cfg(feature = "rocksdb")]
#[test]
fn rocksdb_embedding_persistence() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    let v = vec![0.25_f32, 0.5, 0.75, 1.0];
    {
        let mut engine = StorageEngine::new_rocksdb(embedding_schema(), path).unwrap();
        make_item(&mut engine, "n1");
        engine.set_embedding("g1", "Item", "n1", &v).unwrap();
    }
    let engine = StorageEngine::new_rocksdb(embedding_schema(), path).unwrap();
    let got = engine.get_embedding("g1", "Item", "n1").unwrap().unwrap();
    assert_eq!(got, v);
}

/// Regression for the C1 panic: a batched `delete_node` buffers a
/// `Delete` on `CF_EMBEDDINGS` (the embedding cascade), and on the
/// RocksDB backend `commit_batch` used to index a 5-element handle
/// array with `CfId::Embeddings as usize == 5`, panicking out of
/// bounds. Reachable in production via the `/batch` endpoint. The
/// in-memory backend keys CFs by string so it never hit this; the
/// bug only surfaces on RocksDB, hence the explicit backend here.
#[cfg(feature = "rocksdb")]
#[test]
fn rocksdb_batched_delete_node_with_embedding_commits() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut engine =
        StorageEngine::new_rocksdb(embedding_schema(), tmp.path().to_str().unwrap()).unwrap();

    make_item(&mut engine, "n1");
    engine
        .set_embedding("g1", "Item", "n1", &[0.1_f32, 0.2, 0.3, 0.4])
        .unwrap();

    engine.begin_batch();
    engine.delete_node("g1", "Item", "n1").unwrap();
    let committed = engine.commit_batch().unwrap();

    assert!(committed > 0, "batch should have committed ops");
    assert!(engine.get_node("g1", "Item", "n1").unwrap().is_none());
    assert!(
        engine.get_embedding("g1", "Item", "n1").unwrap().is_none(),
        "embedding cascade should have removed the sidecar embedding"
    );
}

/// Companion to the above: the batched `set_embedding` *put* path
/// also resolves the `CF_EMBEDDINGS` handle on commit, the same
/// handle that was missing from the old array.
#[cfg(feature = "rocksdb")]
#[test]
fn rocksdb_batched_set_embedding_commits() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut engine =
        StorageEngine::new_rocksdb(embedding_schema(), tmp.path().to_str().unwrap()).unwrap();

    make_item(&mut engine, "n1");

    let v = vec![0.5_f32, 0.25, 0.125, 0.0625];
    engine.begin_batch();
    engine.set_embedding("g1", "Item", "n1", &v).unwrap();
    engine.commit_batch().unwrap();

    assert_eq!(
        engine.get_embedding("g1", "Item", "n1").unwrap().unwrap(),
        v
    );
}

#[test]
fn decode_embedding_rejects_non_multiple_of_4() {
    let err = decode_embedding(&[1, 2, 3, 4, 5]).unwrap_err();
    assert!(
        matches!(err, DynoError::Storage(ref m) if m.contains("not a multiple of 4")),
        "{err:?}"
    );
}

/// alice→bob KNOWS in each named graph. Optionally also sets an
/// embedding on alice. Used by clear_graph + S7 adjacency tests.
fn seed_alice_bob(engine: &mut StorageEngine, graphs: &[&str], with_embedding: bool) {
    for graph in graphs {
        engine
            .create_node(graph, "Character", "alice", props! { "name" => "A" })
            .unwrap();
        engine
            .create_node(graph, "Character", "bob", props! { "name" => "B" })
            .unwrap();
        engine
            .create_edge(
                graph,
                "KNOWS",
                "Character",
                "alice",
                "Character",
                "bob",
                HashMap::new(),
            )
            .unwrap();
        if with_embedding {
            engine
                .set_embedding(graph, "Character", "alice", &[0.1, 0.2, 0.3])
                .unwrap();
        }
    }
}

#[test]
fn clear_graph_drops_every_cf_for_graph_only() {
    // Two graphs share the schema and overlap in node/edge ids.
    // After clear_graph("g1"), every g1 key — nodes, edges, both
    // adjacency CFs, and embeddings — must be gone, but g2's data
    // must survive untouched. Idempotent: a second clear is Ok.
    let mut engine = test_engine(test_schema());
    seed_alice_bob(&mut engine, &["g1", "g2"], true);

    engine.clear_graph("g1").unwrap();

    // g1 is gone everywhere.
    assert!(
        engine
            .get_node("g1", "Character", "alice")
            .unwrap()
            .is_none()
    );
    assert!(
        engine
            .get_edge("g1", "KNOWS", "alice", "bob")
            .unwrap()
            .is_none()
    );
    assert_eq!(
        engine
            .scan_outgoing_edges("g1", "alice", None)
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        engine.scan_incoming_edges("g1", "bob", None).unwrap().len(),
        0
    );
    assert!(
        engine
            .get_embedding("g1", "Character", "alice")
            .unwrap()
            .is_none()
    );

    // g2 is untouched.
    assert!(
        engine
            .get_node("g2", "Character", "alice")
            .unwrap()
            .is_some()
    );
    assert!(
        engine
            .get_edge("g2", "KNOWS", "alice", "bob")
            .unwrap()
            .is_some()
    );
    assert_eq!(
        engine
            .get_embedding("g2", "Character", "alice")
            .unwrap()
            .unwrap(),
        vec![0.1, 0.2, 0.3]
    );

    // Idempotent re-clear.
    engine.clear_graph("g1").unwrap();
    engine.clear_graph("never_existed").unwrap();
}

#[test]
fn clear_graph_invalidates_read_cache() {
    // get_node populates the CF_NODES read cache; clear_graph
    // prefix-deletes CF_NODES, which must also invalidate the cache
    // or a subsequent read would serve the cleared node from cache.
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "c1", props! { "name" => "Alice" })
        .unwrap();
    // Prime the cache.
    assert!(engine.get_node("g1", "Character", "c1").unwrap().is_some());

    engine.clear_graph("g1").unwrap();

    assert!(
        engine.get_node("g1", "Character", "c1").unwrap().is_none(),
        "cleared node must not be served from a stale read cache"
    );
}

#[test]
fn clear_graph_does_not_sweep_prefix_neighbors() {
    // Regression: a naive prefix without the trailing separator
    // would let `clear_graph("g1")` also wipe "g10", "g1_test", etc.
    // The graph_prefix helper appends \x00, so the boundary is exact.
    let mut engine = test_engine(test_schema());
    engine
        .create_node("g1", "Character", "a", props! { "name" => "A" })
        .unwrap();
    engine
        .create_node("g10", "Character", "a", props! { "name" => "A10" })
        .unwrap();
    engine
        .create_node("g1_test", "Character", "a", props! { "name" => "AT" })
        .unwrap();

    engine.clear_graph("g1").unwrap();

    assert!(engine.get_node("g1", "Character", "a").unwrap().is_none());
    assert!(engine.get_node("g10", "Character", "a").unwrap().is_some());
    assert!(
        engine
            .get_node("g1_test", "Character", "a")
            .unwrap()
            .is_some()
    );
}

#[cfg(feature = "rocksdb")]
#[test]
fn rocksdb_short_graph_id_adjacency_is_correct() {
    // S7 regression: with `fixed_prefix(48)` on adjacency CFs, a
    // short graph_id like "g1" let RocksDB group keys by a prefix
    // that crossed the graph_id/node_id/edge_type boundaries —
    // potentially returning incorrect adjacency results once the
    // CF spilled to disk. Without the extractor, plain seek + range
    // iteration is correct on any key shape. This test exercises
    // the same shape with multiple graphs sharing node_ids.
    let dir = tempfile::tempdir().unwrap();
    let mut engine =
        StorageEngine::new_rocksdb(test_schema(), dir.path().to_str().unwrap()).unwrap();
    seed_alice_bob(&mut engine, &["g1", "g2"], false);
    // Per-graph adjacency must not bleed across graphs.
    let g1_out = engine.scan_outgoing_edges("g1", "alice", None).unwrap();
    let g2_out = engine.scan_outgoing_edges("g2", "alice", None).unwrap();
    assert_eq!(g1_out.len(), 1);
    assert_eq!(g2_out.len(), 1);
    assert_eq!(g1_out[0].to_id, "bob");
    assert_eq!(g2_out[0].to_id, "bob");
}
