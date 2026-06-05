//! Domain-neutral full-text / BM25 index primitive for dynograph.
//!
//! Wraps an embedded [Tantivy] inverted index behind a small, storage-agnostic
//! API: index a node's full-text fields ([`TextIndex::upsert`]), remove it
//! ([`TextIndex::delete`]), and run tokenized, BM25-ranked keyword search over
//! it ([`TextIndex::search`]). The crate knows nothing about schemas, graphs as
//! a concept, or any consumer domain — callers pass already-extracted `(graph,
//! node_type, node_id, fields)` and get back scored `node_id`s.
//!
//! # Consistency model
//!
//! Tantivy has its own commit lifecycle and cannot join RocksDB's atomic write
//! batch, so this index is a **derived, rebuildable** view: the caller's primary
//! store is the source of truth. Writes ([`upsert`]/[`delete`]) are buffered and
//! only become visible to [`search`] after an explicit [`commit`]. This matches
//! Tantivy's design (commits are comparatively expensive) and lets callers batch.
//!
//! [Tantivy]: https://github.com/quickwit-oss/tantivy
//! [`upsert`]: TextIndex::upsert
//! [`delete`]: TextIndex::delete
//! [`search`]: TextIndex::search
//! [`commit`]: TextIndex::commit

use std::path::Path;
use std::sync::Mutex;

use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, STORED, STRING, Schema, TEXT, Value};
use tantivy::tokenizer::TokenStream;
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

/// NUL byte joins `graph_id` and `node_id` into the per-document unique key.
/// NUL can't appear in either segment (the storage layer rejects NUL-bearing
/// key segments before writing), so the composite is unambiguous.
const UID_SEP: char = '\u{0}';

/// Tantivy writer heap budget. Tantivy requires a minimum per-index arena;
/// 50 MB is comfortably above the floor and bounds memory before an auto-flush.
const WRITER_HEAP_BYTES: usize = 50_000_000;

/// Errors surfaced by [`TextIndex`].
#[derive(Debug, thiserror::Error)]
pub enum TextError {
    /// Failed to open/create the on-disk index directory.
    #[error("failed to open full-text index at {path}: {source}")]
    Open {
        path: String,
        #[source]
        source: tantivy::TantivyError,
    },
    /// Any other Tantivy-level failure (add/delete/commit/search).
    #[error("full-text index error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),
}

/// Handles for the fixed Tantivy document schema this crate maintains.
struct Fields {
    /// `graph_id\0node_id`, STRING (indexed, not stored) — the delete key.
    /// Composite so delete-by-term is correct even if multiple graphs ever
    /// share one index directory.
    uid: Field,
    /// STRING + STORED — filter searches to one graph and echo back on hits.
    graph_id: Field,
    /// STRING + STORED — optional type filter + returned on hits.
    node_type: Field,
    /// STRING + STORED — returned on hits (the caller's join key).
    node_id: Field,
    /// TEXT — the concatenation of all full-text property values, tokenized
    /// and BM25-scored. (First cut: one combined field; per-property field
    /// targeting is a future extension — see crate docs / design notes.)
    text: Field,
}

/// An embedded full-text index over a directory on disk.
///
/// Open-or-create with [`TextIndex::open`]. Cheap to clone-by-reopen but not
/// `Clone`; hold one per index directory. Internally synchronized, so `&self`
/// methods are safe to call concurrently.
pub struct TextIndex {
    index: Index,
    reader: IndexReader,
    writer: Mutex<IndexWriter>,
    fields: Fields,
}

impl TextIndex {
    /// Open the index at `path`, creating it (and the directory) if absent.
    ///
    /// The Tantivy document schema is fixed by this crate; reopening a
    /// directory written by an incompatible schema returns [`TextError::Open`].
    pub fn open(path: &Path) -> Result<Self, TextError> {
        let mut builder = Schema::builder();
        let uid = builder.add_text_field("uid", STRING);
        let graph_id = builder.add_text_field("graph_id", STRING | STORED);
        let node_type = builder.add_text_field("node_type", STRING | STORED);
        let node_id = builder.add_text_field("node_id", STRING | STORED);
        let text = builder.add_text_field("text", TEXT);
        let schema = builder.build();

        let map_open = |source: tantivy::TantivyError| TextError::Open {
            path: path.display().to_string(),
            source,
        };

        std::fs::create_dir_all(path).map_err(|e| TextError::Open {
            path: path.display().to_string(),
            source: tantivy::TantivyError::SystemError(e.to_string()),
        })?;
        let dir = tantivy::directory::MmapDirectory::open(path).map_err(|e| TextError::Open {
            path: path.display().to_string(),
            source: tantivy::TantivyError::from(e),
        })?;
        let index = Index::open_or_create(dir, schema).map_err(map_open)?;

        let writer: IndexWriter = index.writer(WRITER_HEAP_BYTES).map_err(map_open)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(map_open)?;

        Ok(Self {
            index,
            reader,
            writer: Mutex::new(writer),
            fields: Fields {
                uid,
                graph_id,
                node_type,
                node_id,
                text,
            },
        })
    }

    fn uid(graph_id: &str, node_id: &str) -> String {
        format!("{graph_id}{UID_SEP}{node_id}")
    }

    /// Index (or replace) the full-text fields of one node.
    ///
    /// `fields` is the subset of the node's properties declared `fulltext: true`,
    /// as `(property_name, value)` pairs already extracted by the caller. Values
    /// are concatenated into the searchable text. Replace semantics: any existing
    /// document for `(graph_id, node_id)` is removed first, so re-`upsert` is
    /// idempotent. Buffered until [`commit`](Self::commit).
    pub fn upsert(
        &self,
        graph_id: &str,
        node_type: &str,
        node_id: &str,
        fields: &[(String, String)],
    ) -> Result<(), TextError> {
        let text: String = fields
            .iter()
            .map(|(_, v)| v.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let mut doc = TantivyDocument::new();
        doc.add_text(self.fields.uid, Self::uid(graph_id, node_id));
        doc.add_text(self.fields.graph_id, graph_id);
        doc.add_text(self.fields.node_type, node_type);
        doc.add_text(self.fields.node_id, node_id);
        doc.add_text(self.fields.text, text);

        let writer = self.writer.lock().unwrap();
        // Delete-then-add: the new doc has a higher opstamp than the delete, so
        // it survives — only the prior version (lower opstamp) is removed.
        writer.delete_term(Term::from_field_text(
            self.fields.uid,
            &Self::uid(graph_id, node_id),
        ));
        writer.add_document(doc)?;
        Ok(())
    }

    /// Remove a node from the index. Idempotent — deleting an absent node is a
    /// no-op. Buffered until [`commit`](Self::commit).
    pub fn delete(&self, graph_id: &str, node_id: &str) -> Result<(), TextError> {
        let writer = self.writer.lock().unwrap();
        writer.delete_term(Term::from_field_text(
            self.fields.uid,
            &Self::uid(graph_id, node_id),
        ));
        Ok(())
    }

    /// Commit buffered writes and make them visible to [`search`](Self::search).
    /// Forces a reader reload so results are consistent immediately on return.
    pub fn commit(&self) -> Result<(), TextError> {
        let mut writer = self.writer.lock().unwrap();
        writer.commit()?;
        drop(writer);
        self.reader.reload()?;
        Ok(())
    }

    /// BM25 keyword search within one graph.
    ///
    /// `query` is tokenized with the same analyzer as the indexed text and
    /// matched as a conjunction: every token must occur (AND semantics). The
    /// raw string is **not** run through Tantivy's query grammar, so colons,
    /// parentheses, `+`/`-`, and `field:value`-looking input are treated as
    /// plain text — they can neither reference the index's internal fields nor
    /// raise parse errors. A query that yields no usable tokens (empty or all
    /// punctuation) matches nothing. Results are scoped to `graph_id`,
    /// optionally restricted to one `node_type`, capped at `limit`, and
    /// returned highest-BM25-score first.
    pub fn search(
        &self,
        graph_id: &str,
        query: &str,
        node_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<TextHit>, TextError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let searcher = self.reader.searcher();

        // Tokenize the raw input with the SAME analyzer the `text` field uses,
        // then AND a TermQuery per token. We deliberately do NOT feed `query`
        // to Tantivy's QueryParser: that would resolve `field:value` tokens
        // against the index's internal fields (graph_id, node_type, uid, ...)
        // and would error or silently change meaning on punctuation. Tokenizing
        // treats the input as plain keywords and pins AND semantics.
        let mut analyzer = self.index.tokenizer_for_field(self.fields.text)?;
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        {
            let mut stream = analyzer.token_stream(query);
            while stream.advance() {
                let term = Term::from_field_text(self.fields.text, &stream.token().text);
                clauses.push((
                    Occur::Must,
                    Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)),
                ));
            }
        }
        // No usable tokens (empty or all-punctuation query) → match nothing,
        // rather than letting the bare graph-scope clause return every node.
        if clauses.is_empty() {
            return Ok(Vec::new());
        }

        // Scope to the graph, and optionally to one node_type.
        clauses.push((
            Occur::Must,
            Box::new(TermQuery::new(
                Term::from_field_text(self.fields.graph_id, graph_id),
                IndexRecordOption::Basic,
            )),
        ));
        if let Some(nt) = node_type {
            clauses.push((
                Occur::Must,
                Box::new(TermQuery::new(
                    Term::from_field_text(self.fields.node_type, nt),
                    IndexRecordOption::Basic,
                )),
            ));
        }
        let bool_query = BooleanQuery::new(clauses);

        let hits = searcher.search(&bool_query, &TopDocs::with_limit(limit).order_by_score())?;
        let mut out = Vec::with_capacity(hits.len());
        for (score, addr) in hits {
            let doc: TantivyDocument = searcher.doc(addr)?;
            let get = |f: Field| {
                doc.get_first(f)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string()
            };
            out.push(TextHit {
                node_id: get(self.fields.node_id),
                node_type: get(self.fields.node_type),
                score,
            });
        }
        Ok(out)
    }
}

/// One search result: the matched node's id and type, plus its BM25 score.
#[derive(Debug, Clone, PartialEq)]
pub struct TextHit {
    pub node_id: String,
    pub node_type: String,
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn open_tmp() -> (tempfile::TempDir, TextIndex) {
        let dir = tempfile::tempdir().unwrap();
        let idx = TextIndex::open(dir.path()).unwrap();
        (dir, idx)
    }

    #[test]
    fn upsert_then_search_finds_doc() {
        let (_d, idx) = open_tmp();
        idx.upsert(
            "g1",
            "Document",
            "n1",
            &fields(&[
                ("title", "The Quick Brown Fox"),
                ("body", "jumps over the lazy dog"),
            ]),
        )
        .unwrap();
        idx.commit().unwrap();

        let hits = idx.search("g1", "fox", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node_id, "n1");
        assert_eq!(hits[0].node_type, "Document");
        assert!(hits[0].score > 0.0);

        // A term in the second field is searchable too (fields are concatenated).
        assert_eq!(idx.search("g1", "lazy", None, 10).unwrap().len(), 1);
        // A term in neither field matches nothing.
        assert!(idx.search("g1", "elephant", None, 10).unwrap().is_empty());
    }

    #[test]
    fn uncommitted_writes_are_invisible() {
        let (_d, idx) = open_tmp();
        idx.upsert("g1", "Document", "n1", &fields(&[("body", "hello world")]))
            .unwrap();
        // No commit yet.
        assert!(idx.search("g1", "hello", None, 10).unwrap().is_empty());
        idx.commit().unwrap();
        assert_eq!(idx.search("g1", "hello", None, 10).unwrap().len(), 1);
    }

    #[test]
    fn upsert_replaces_not_duplicates() {
        let (_d, idx) = open_tmp();
        idx.upsert("g1", "Document", "n1", &fields(&[("body", "alpha")]))
            .unwrap();
        idx.commit().unwrap();
        // Replace the same node's content.
        idx.upsert("g1", "Document", "n1", &fields(&[("body", "beta")]))
            .unwrap();
        idx.commit().unwrap();

        // Old term gone, new term present, and exactly one doc total.
        assert!(idx.search("g1", "alpha", None, 10).unwrap().is_empty());
        let beta = idx.search("g1", "beta", None, 10).unwrap();
        assert_eq!(beta.len(), 1);
        assert_eq!(beta[0].node_id, "n1");
    }

    #[test]
    fn delete_removes_doc() {
        let (_d, idx) = open_tmp();
        idx.upsert("g1", "Document", "n1", &fields(&[("body", "deletable")]))
            .unwrap();
        idx.commit().unwrap();
        assert_eq!(idx.search("g1", "deletable", None, 10).unwrap().len(), 1);

        idx.delete("g1", "n1").unwrap();
        idx.commit().unwrap();
        assert!(idx.search("g1", "deletable", None, 10).unwrap().is_empty());
        // Deleting again is a harmless no-op.
        idx.delete("g1", "n1").unwrap();
        idx.commit().unwrap();
    }

    #[test]
    fn search_is_scoped_by_graph() {
        let (_d, idx) = open_tmp();
        idx.upsert("g1", "Document", "n1", &fields(&[("body", "shared term")]))
            .unwrap();
        idx.upsert("g2", "Document", "n2", &fields(&[("body", "shared term")]))
            .unwrap();
        idx.commit().unwrap();

        let g1 = idx.search("g1", "shared", None, 10).unwrap();
        assert_eq!(g1.len(), 1);
        assert_eq!(g1[0].node_id, "n1");
        let g2 = idx.search("g2", "shared", None, 10).unwrap();
        assert_eq!(g2.len(), 1);
        assert_eq!(g2[0].node_id, "n2");
        // Same node_id in different graphs doesn't collide on delete.
        idx.delete("g1", "n1").unwrap();
        idx.commit().unwrap();
        assert!(idx.search("g1", "shared", None, 10).unwrap().is_empty());
        assert_eq!(idx.search("g2", "shared", None, 10).unwrap().len(), 1);
    }

    #[test]
    fn node_type_filter_restricts_results() {
        let (_d, idx) = open_tmp();
        idx.upsert("g1", "Document", "d1", &fields(&[("body", "common")]))
            .unwrap();
        idx.upsert("g1", "Note", "x1", &fields(&[("body", "common")]))
            .unwrap();
        idx.commit().unwrap();

        assert_eq!(idx.search("g1", "common", None, 10).unwrap().len(), 2);
        let docs = idx.search("g1", "common", Some("Document"), 10).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].node_id, "d1");
    }

    #[test]
    fn multi_term_query_is_conjunctive() {
        let (_d, idx) = open_tmp();
        idx.upsert(
            "g1",
            "Document",
            "n1",
            &fields(&[("body", "the quick brown fox")]),
        )
        .unwrap();
        idx.upsert(
            "g1",
            "Document",
            "n2",
            &fields(&[("body", "only brown here")]),
        )
        .unwrap();
        idx.commit().unwrap();

        // Every token must occur (AND), order-independent: only n1 has both.
        let both = idx.search("g1", "quick brown", None, 10).unwrap();
        assert_eq!(both.len(), 1);
        assert_eq!(both[0].node_id, "n1");
        // A single shared term still matches both docs.
        assert_eq!(idx.search("g1", "brown", None, 10).unwrap().len(), 2);
    }

    #[test]
    fn punctuation_query_is_treated_as_text_not_grammar() {
        let (_d, idx) = open_tmp();
        idx.upsert(
            "g1",
            "Document",
            "n1",
            &fields(&[("body", "meeting at 3 30 pm")]),
        )
        .unwrap();
        idx.commit().unwrap();

        // A colon-bearing query that the old QueryParser path rejected now
        // tokenizes to plain terms (3, 30) and searches — no error.
        let hits = idx.search("g1", "3:30", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node_id, "n1");
        // An all-punctuation query yields no usable tokens → empty, not error,
        // and crucially NOT every doc in the graph.
        assert!(
            idx.search("g1", "!!! ??? :::", None, 10)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn field_prefix_in_query_does_not_inject_a_filter() {
        let (_d, idx) = open_tmp();
        // A Document and a Note, neither containing the literal words below.
        idx.upsert("g1", "Document", "d1", &fields(&[("body", "common")]))
            .unwrap();
        idx.upsert("g1", "Note", "x1", &fields(&[("body", "common")]))
            .unwrap();
        idx.commit().unwrap();

        // `node_type:Note` is treated as the text tokens node/type/note, NOT as
        // a filter on the internal node_type field. No body contains those
        // words, so the result is empty (the old QueryParser path would have
        // returned the Note via field injection).
        assert!(
            idx.search("g1", "node_type:Note", None, 10)
                .unwrap()
                .is_empty()
        );
        // The legitimate node_type PARAMETER still filters correctly.
        let docs = idx.search("g1", "common", Some("Note"), 10).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].node_id, "x1");
    }

    #[test]
    fn survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let idx = TextIndex::open(dir.path()).unwrap();
            idx.upsert("g1", "Document", "n1", &fields(&[("body", "persistent")]))
                .unwrap();
            idx.commit().unwrap();
        }
        // Reopen the same directory: committed data is still searchable.
        let idx = TextIndex::open(dir.path()).unwrap();
        let hits = idx.search("g1", "persistent", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node_id, "n1");
    }

    #[test]
    fn zero_limit_returns_empty() {
        let (_d, idx) = open_tmp();
        idx.upsert("g1", "Document", "n1", &fields(&[("body", "hello")]))
            .unwrap();
        idx.commit().unwrap();
        assert!(idx.search("g1", "hello", None, 0).unwrap().is_empty());
    }
}
