use super::*;
use dynograph_core::{Schema, props};

/// A schema with one full-text node type (`Document` with `title`/`body`
/// fulltext) and one without (`Tag`).
fn ft_schema() -> Schema {
    Schema::from_yaml(
        r#"
schema:
  name: ft
  version: 1
  node_types:
    Document:
      properties:
        title: { type: string, fulltext: true }
        body:  { type: string, fulltext: true }
        author: { type: string, indexed: true }
    Tag:
      properties:
        name: { type: string, indexed: true }
  edge_types: {}
"#,
    )
    .unwrap()
}

/// RocksDB engine over a fresh temp dir (leaked — the engine holds it open
/// for the test, mirroring `test_engine`'s rocksdb arm).
#[cfg(feature = "rocksdb")]
fn rocks_engine(schema: Schema) -> StorageEngine {
    let dir = tempfile::tempdir().expect("temp dir").keep();
    let path = dir.to_str().expect("utf-8 temp path");
    StorageEngine::new_rocksdb(schema, path).expect("open rocksdb engine")
}

#[test]
fn no_index_built_when_schema_has_no_fulltext() {
    // Schema with no fulltext property → search is a clean empty, never an
    // error, and no index is constructed.
    let schema = Schema::from_yaml(
        r#"
schema:
  name: plain
  version: 1
  node_types:
    Tag:
      properties:
        name: { type: string, indexed: true }
  edge_types: {}
"#,
    )
    .unwrap();
    let engine = StorageEngine::new_in_memory(schema);
    assert!(
        engine
            .search_fulltext("g1", "anything", None, 10)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn create_indexes_and_delete_clears_in_memory() {
    let mut engine = StorageEngine::new_in_memory(ft_schema());
    engine
        .create_node(
            "g1",
            "Document",
            "n1",
            props! { "title" => "Rust Graphs", "body" => "embedded full text search" },
        )
        .unwrap();

    // Findable by a token from either fulltext field.
    let hits = engine.search_fulltext("g1", "graphs", None, 10).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].node_id, "n1");
    assert_eq!(hits[0].node_type, "Document");
    assert_eq!(
        engine
            .search_fulltext("g1", "search", None, 10)
            .unwrap()
            .len(),
        1
    );

    // Delete clears the document.
    engine.delete_node("g1", "Document", "n1").unwrap();
    assert!(
        engine
            .search_fulltext("g1", "graphs", None, 10)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn replace_properties_reindexes() {
    let mut engine = StorageEngine::new_in_memory(ft_schema());
    engine
        .create_node(
            "g1",
            "Document",
            "n1",
            props! { "title" => "alpha", "body" => "first" },
        )
        .unwrap();
    assert_eq!(
        engine
            .search_fulltext("g1", "alpha", None, 10)
            .unwrap()
            .len(),
        1
    );

    engine
        .replace_node_properties(
            "g1",
            "Document",
            "n1",
            props! { "title" => "beta", "body" => "second" },
        )
        .unwrap();
    // Old token gone, new token present — replace semantics held.
    assert!(
        engine
            .search_fulltext("g1", "alpha", None, 10)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        engine
            .search_fulltext("g1", "beta", None, 10)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn node_type_with_no_fulltext_is_not_searchable() {
    let mut engine = StorageEngine::new_in_memory(ft_schema());
    engine
        .create_node("g1", "Tag", "t1", props! { "name" => "important" })
        .unwrap();
    // Tag has no fulltext property → never indexed.
    assert!(
        engine
            .search_fulltext("g1", "important", None, 10)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn batch_commit_makes_writes_visible_and_discard_rolls_back() {
    let mut engine = StorageEngine::new_in_memory(ft_schema());

    // Committed batch → searchable.
    engine.begin_batch();
    engine
        .create_node(
            "g1",
            "Document",
            "n1",
            props! { "title" => "committed", "body" => "x" },
        )
        .unwrap();
    // Buffered: not yet visible.
    assert!(
        engine
            .search_fulltext("g1", "committed", None, 10)
            .unwrap()
            .is_empty()
    );
    engine.commit_batch().unwrap();
    assert_eq!(
        engine
            .search_fulltext("g1", "committed", None, 10)
            .unwrap()
            .len(),
        1
    );

    // Discarded batch → rolled back, never visible.
    engine.begin_batch();
    engine
        .create_node(
            "g1",
            "Document",
            "n2",
            props! { "title" => "transient", "body" => "y" },
        )
        .unwrap();
    engine.discard_batch();
    assert!(
        engine
            .search_fulltext("g1", "transient", None, 10)
            .unwrap()
            .is_empty()
    );
    // The earlier committed doc is untouched by the rollback.
    assert_eq!(
        engine
            .search_fulltext("g1", "committed", None, 10)
            .unwrap()
            .len(),
        1
    );
}

#[cfg(feature = "rocksdb")]
#[test]
fn reindex_rebuilds_and_survives_rocksdb_reopen() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().to_str().expect("utf-8 temp path").to_string();

    {
        let mut engine =
            StorageEngine::new_rocksdb(ft_schema(), &path).expect("open rocksdb engine");
        engine
            .create_node(
                "g1",
                "Document",
                "n1",
                props! { "title" => "persistent", "body" => "z" },
            )
            .unwrap();
        assert_eq!(
            engine
                .search_fulltext("g1", "persistent", None, 10)
                .unwrap()
                .len(),
            1
        );
    }

    // Reopen the same dir: the on-disk Tantivy index reloads and the doc is
    // still searchable without re-indexing.
    let engine = StorageEngine::new_rocksdb(ft_schema(), &path).expect("reopen rocksdb engine");
    assert_eq!(
        engine
            .search_fulltext("g1", "persistent", None, 10)
            .unwrap()
            .len(),
        1
    );

    // reindex_fulltext is idempotent: rebuild from RocksDB, still one hit.
    let n = engine.reindex_fulltext("g1").unwrap();
    assert_eq!(n, 1);
    assert_eq!(
        engine
            .search_fulltext("g1", "persistent", None, 10)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn scoped_by_graph_and_node_type() {
    let mut engine = rocks_engine(ft_schema());
    engine
        .create_node(
            "g1",
            "Document",
            "d1",
            props! { "title" => "common", "body" => "a" },
        )
        .unwrap();
    engine
        .create_node(
            "g2",
            "Document",
            "d2",
            props! { "title" => "common", "body" => "b" },
        )
        .unwrap();
    engine
        .create_node("g1", "Tag", "t1", props! { "name" => "common" })
        .unwrap();

    // Graph scoping.
    let g1 = engine.search_fulltext("g1", "common", None, 10).unwrap();
    assert_eq!(g1.len(), 1);
    assert_eq!(g1[0].node_id, "d1");
    // node_type filter (Tag isn't indexed anyway, so only Document matches).
    let docs = engine
        .search_fulltext("g1", "common", Some("Document"), 10)
        .unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].node_id, "d1");
}

#[test]
fn clear_graph_drops_fulltext_documents() {
    let mut engine = StorageEngine::new_in_memory(ft_schema());
    engine
        .create_node(
            "g1",
            "Document",
            "n1",
            props! { "title" => "scrubme", "body" => "a" },
        )
        .unwrap();
    assert_eq!(
        engine
            .search_fulltext("g1", "scrubme", None, 10)
            .unwrap()
            .len(),
        1
    );
    engine.clear_graph("g1").unwrap();
    assert!(
        engine
            .search_fulltext("g1", "scrubme", None, 10)
            .unwrap()
            .is_empty()
    );
}

/// #1 regression guard: a discarded batch must not drop a prior committed
/// full-text document (the writer is clean at begin_batch, so the rollback
/// only reverts the batch's own ops).
#[test]
fn discard_batch_preserves_prior_committed_fulltext() {
    let mut engine = StorageEngine::new_in_memory(ft_schema());
    engine
        .create_node(
            "g1",
            "Document",
            "n1",
            props! { "title" => "keepme", "body" => "a" },
        )
        .unwrap();
    assert_eq!(
        engine
            .search_fulltext("g1", "keepme", None, 10)
            .unwrap()
            .len(),
        1
    );

    engine.begin_batch();
    engine
        .create_node(
            "g1",
            "Document",
            "n2",
            props! { "title" => "dropme", "body" => "b" },
        )
        .unwrap();
    engine.discard_batch();

    // Prior committed doc survives; the discarded batch's doc never appears.
    assert_eq!(
        engine
            .search_fulltext("g1", "keepme", None, 10)
            .unwrap()
            .len(),
        1
    );
    assert!(
        engine
            .search_fulltext("g1", "dropme", None, 10)
            .unwrap()
            .is_empty()
    );
}

/// #2: enabling full-text on a live engine (via replace_schema) doesn't
/// build an index, so search/reindex fail loud instead of silently empty /
/// Ok(0).
#[test]
fn fulltext_enabled_at_runtime_fails_loud_until_reopen() {
    let plain = Schema::from_yaml(
        r#"
schema:
  name: p
  version: 1
  node_types:
    Document:
      properties:
        title: { type: string }
  edge_types: {}
"#,
    )
    .unwrap();
    let mut engine = StorageEngine::new_in_memory(plain);
    // Genuinely no full-text → clean empty, no error.
    assert!(
        engine
            .search_fulltext("g1", "x", None, 10)
            .unwrap()
            .is_empty()
    );
    assert_eq!(engine.reindex_fulltext("g1").unwrap(), 0);

    // Enable full-text at runtime; no index is built for the live engine.
    engine.replace_schema(ft_schema());
    assert!(engine.search_fulltext("g1", "x", None, 10).is_err());
    assert!(engine.reindex_fulltext("g1").is_err());
}

/// #6: a rebuild can't run inside an open batch.
#[test]
fn reindex_inside_batch_errors() {
    let mut engine = StorageEngine::new_in_memory(ft_schema());
    engine.begin_batch();
    let err = engine.reindex_fulltext("g1").unwrap_err();
    engine.discard_batch();
    match err {
        DynoError::Storage(msg) => assert!(msg.contains("batch"), "got: {msg}"),
        other => panic!("expected Storage error, got {other:?}"),
    }
}
