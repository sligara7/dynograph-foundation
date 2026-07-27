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

/// Default Tantivy writer-arena budget, in MB. The arena is reserved up front
/// per index, so on a multi-graph host this is the dominant fixed memory cost.
/// Override at runtime with `DYNOGRAPH_FULLTEXT_WRITER_HEAP_MB`.
const DEFAULT_WRITER_HEAP_MB: usize = 50;
/// Tantivy's practical per-writer floor; smaller arenas make `writer()` error.
const MIN_WRITER_HEAP_BYTES: usize = 15_000_000;

/// Resolve the writer-arena size in bytes: `DYNOGRAPH_FULLTEXT_WRITER_HEAP_MB`
/// when set and parseable, else [`DEFAULT_WRITER_HEAP_MB`] — floored at the
/// Tantivy minimum either way so a too-small value can't break index open.
/// Read once per index open (not on any hot path).
fn writer_heap_bytes() -> usize {
    std::env::var("DYNOGRAPH_FULLTEXT_WRITER_HEAP_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_WRITER_HEAP_MB)
        .saturating_mul(1_000_000)
        .max(MIN_WRITER_HEAP_BYTES)
}

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
    /// A write was attempted on an index opened read-only.
    ///
    /// Loud on purpose. A read-only index that silently swallowed writes would
    /// report success while the caller's data went nowhere, which is worse than
    /// refusing: the caller would only find out by searching for something that
    /// was never there.
    #[error(
        "full-text index at {path} is open READ-ONLY, so `{operation}` cannot be served; \
         another process holds the writer"
    )]
    ReadOnly {
        path: String,
        operation: &'static str,
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
    /// `None` when opened read-only. Optional rather than a second type
    /// because every read path is identical and duplicating them to model the
    /// absence of a writer would be a worse trade.
    writer: Option<Mutex<IndexWriter>>,
    fields: Fields,
    /// Only for error messages — a refusal that cannot say WHICH index it is
    /// about is not much of a diagnosis on a host with several graphs.
    path: String,
}

impl TextIndex {
    /// Open the index at `path`, creating it (and the directory) if absent.
    ///
    /// The Tantivy document schema is fixed by this crate; reopening a
    /// directory written by an incompatible schema returns [`TextError::Open`].
    pub fn open(path: &Path) -> Result<Self, TextError> {
        let (schema, fields) = Self::build_schema();
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
        Self::finish(index, fields, path.display().to_string())
    }

    /// Open an EXISTING index for reading only, taking no writer lock — so this
    /// succeeds while another process holds the writer.
    ///
    /// The view is consistent as of this call and does not refresh; seeing later
    /// writes means reopening. That is deliberate: staleness stays bounded and
    /// knowable (it is exactly as old as the open), which is what a caller needs
    /// in order to say how current its answer is.
    ///
    /// Unlike [`open`](Self::open) this never CREATES. A read-only open of a
    /// directory that holds no index is an error, not an empty index: silently
    /// inventing one would report "no results" for a design that exists and is
    /// merely somewhere else.
    pub fn open_read_only(path: &Path) -> Result<Self, TextError> {
        let (_schema, fields) = Self::build_schema();
        let map_open = |source: tantivy::TantivyError| TextError::Open {
            path: path.display().to_string(),
            source,
        };
        let dir = tantivy::directory::MmapDirectory::open(path).map_err(|e| TextError::Open {
            path: path.display().to_string(),
            source: tantivy::TantivyError::from(e),
        })?;
        let index = Index::open(dir).map_err(map_open)?;
        Self::finish_read_only(index, fields, path.display().to_string())
    }

    /// Open an ephemeral, RAM-backed index. Nothing is persisted — intended for
    /// the in-memory storage backend and tests. Identical API and semantics to
    /// [`open`](Self::open) otherwise.
    pub fn open_in_ram() -> Result<Self, TextError> {
        let (schema, fields) = Self::build_schema();
        let index = Index::create_in_ram(schema);
        Self::finish(index, fields, "<in-ram>".to_string())
    }

    /// Build the fixed document schema and the field handles for it. The field
    /// set and order are stable so an on-disk index reopens without a schema
    /// mismatch.
    fn build_schema() -> (Schema, Fields) {
        let mut builder = Schema::builder();
        let uid = builder.add_text_field("uid", STRING);
        let graph_id = builder.add_text_field("graph_id", STRING | STORED);
        let node_type = builder.add_text_field("node_type", STRING | STORED);
        let node_id = builder.add_text_field("node_id", STRING | STORED);
        let text = builder.add_text_field("text", TEXT);
        (
            builder.build(),
            Fields {
                uid,
                graph_id,
                node_type,
                node_id,
                text,
            },
        )
    }

    /// Build the writer + reader over an opened index.
    fn finish(index: Index, fields: Fields, path: String) -> Result<Self, TextError> {
        let writer: IndexWriter = index.writer(writer_heap_bytes())?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;
        Ok(Self {
            index,
            reader,
            writer: Some(Mutex::new(writer)),
            fields,
            path,
        })
    }

    /// Build a reader ONLY, taking no writer lock.
    ///
    /// This is what makes a second process able to read an index the first is
    /// writing: Tantivy's `INDEX_WRITER_LOCK` admits exactly one writer, and
    /// asking for one is the only thing that was ever in the way.
    fn finish_read_only(index: Index, fields: Fields, path: String) -> Result<Self, TextError> {
        // Manual reload: a read-only view is a view as of open, and refreshing
        // it behind the caller's back would make "how old is this?" unanswerable
        // — the exact property the read-only path was chosen for.
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        Ok(Self {
            index,
            reader,
            writer: None,
            fields,
            path,
        })
    }

    /// The writer, or a loud refusal naming the operation that wanted it.
    fn writer(&self, operation: &'static str) -> Result<&Mutex<IndexWriter>, TextError> {
        self.writer.as_ref().ok_or_else(|| TextError::ReadOnly {
            path: self.path.clone(),
            operation,
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

        let writer = self.writer("upsert")?.lock().unwrap();
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
        let writer = self.writer("delete")?.lock().unwrap();
        writer.delete_term(Term::from_field_text(
            self.fields.uid,
            &Self::uid(graph_id, node_id),
        ));
        Ok(())
    }

    /// Remove every document belonging to `graph_id`. Used to drop a graph's
    /// index when the graph is deleted, and to clear before a full rebuild.
    /// Idempotent. Buffered until [`commit`](Self::commit).
    pub fn delete_graph(&self, graph_id: &str) -> Result<(), TextError> {
        let writer = self.writer("delete_graph")?.lock().unwrap();
        writer.delete_term(Term::from_field_text(self.fields.graph_id, graph_id));
        Ok(())
    }

    /// Commit buffered writes and make them visible to [`search`](Self::search).
    /// Forces a reader reload so results are consistent immediately on return.
    pub fn commit(&self) -> Result<(), TextError> {
        let mut writer = self.writer("commit")?.lock().unwrap();
        writer.commit()?;
        drop(writer);
        self.reader.reload()?;
        Ok(())
    }

    /// Discard buffered (uncommitted) writes, reverting to the last commit.
    /// Mirrors a storage-layer batch rollback so a discarded batch leaves no
    /// stray full-text entries.
    pub fn rollback(&self) -> Result<(), TextError> {
        let mut writer = self.writer("rollback")?.lock().unwrap();
        writer.rollback()?;
        Ok(())
    }

    /// BM25 keyword search within one graph.
    ///
    /// `query` is tokenized with the same analyzer as the indexed text and
    /// matched as a **ranked disjunction**. Exactly two things are guaranteed:
    /// a document must contain at least one token to appear at all, and results
    /// come back in descending BM25 score order.
    ///
    /// Matching more tokens raises a document's score, because BM25 sums a
    /// contribution per matched term — but that is a tendency, **not** an
    /// ordering guarantee. BM25 also weighs term rarity (IDF) and penalizes
    /// length, so a short document matching one rare token can outrank a long
    /// one matching several common tokens. Callers must not treat "matched the
    /// most terms" as a property they can rely on; rank by the score returned,
    /// and if a single answer is needed, treat the top hit as the best
    /// candidate rather than a decision.
    ///
    /// This is what makes a natural-language question usable as a query — under
    /// the previous conjunctive rule one incidental word the corpus never used
    /// ("do we already have a requirement for X?") reduced a perfect match to
    /// zero hits, which reads as "no such thing exists" and is the worst
    /// possible answer for a caller deciding whether to create a duplicate.
    /// Narrowing is still available and is now the caller's choice: pass fewer,
    /// better words, or filter with `node_type`.
    ///
    /// The raw string is **not** run through Tantivy's query grammar, so colons,
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
        // then OR a TermQuery per token. We deliberately do NOT feed `query`
        // to Tantivy's QueryParser: that would resolve `field:value` tokens
        // against the index's internal fields (graph_id, node_type, uid, ...)
        // and would error or silently change meaning on punctuation. Tokenizing
        // treats the input as plain keywords; BM25 then does the discriminating
        // by SCORE rather than by exclusion.
        let mut analyzer = self.index.tokenizer_for_field(self.fields.text)?;
        let mut term_clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        {
            let mut stream = analyzer.token_stream(query);
            while stream.advance() {
                let term = Term::from_field_text(self.fields.text, &stream.token().text);
                term_clauses.push((
                    Occur::Should,
                    Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)),
                ));
            }
        }
        // No usable tokens (empty or all-punctuation query) → match nothing,
        // rather than letting the bare graph-scope clause return every node.
        if term_clauses.is_empty() {
            return Ok(Vec::new());
        }

        // The term disjunction is nested and then required as a single clause.
        // This is load-bearing: a flat mix of Should terms with the Must scope
        // clauses would make the terms wholly optional, and every node in the
        // graph would match on the scope clause alone — the exact failure the
        // empty-token guard above exists to prevent. Nested, a pure-Should
        // BooleanQuery demands at least one of its clauses, so "at least one
        // token, ranked by how many" falls out without depending on a
        // minimum-should-match setter.
        let mut clauses: Vec<(Occur, Box<dyn Query>)> =
            vec![(Occur::Must, Box::new(BooleanQuery::new(term_clauses)))];

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
    fn multi_term_query_ranks_rather_than_excludes() {
        // FLIPPED DELIBERATELY (was `multi_term_query_is_conjunctive`). The old
        // contract — every token must occur — was pinned by this test and is
        // now the defect it guards against: a partial match is the normal shape
        // of a real question, and answering it with silence is what pushed
        // callers back to grepping prose files. Discrimination moved from
        // exclusion to ranking; the assertion below is the same scenario read
        // the new way, so the behaviour change is on the record rather than
        // erased by deleting a test.
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

        // Both documents contain at least one token, so both come back rather
        // than the partial match being discarded — that inclusion is the change.
        // These two fixtures are comparable in length, so the one matching both
        // tokens also scores higher here; that is the summation showing through,
        // not a promise that a fuller match always leads (see
        // `more_matching_tokens_ranks_higher` for why it cannot be).
        let both = idx.search("g1", "quick brown", None, 10).unwrap();
        assert_eq!(both.len(), 2, "a partial match is a match, ranked lower");
        assert_eq!(
            both[0].node_id, "n1",
            "both tokens match, and both docs are short"
        );
        assert!(both[0].score > both[1].score);
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
    fn an_unmatched_token_no_longer_annihilates_a_match() {
        let (_d, idx) = open_tmp();
        idx.upsert(
            "g1",
            "Requirement",
            "r1",
            &fields(&[("body", "a design survives a reflow2 upgrade")]),
        )
        .unwrap();
        idx.commit().unwrap();

        // The regression this test exists for: under the old conjunctive rule
        // one token the corpus never used reduced a perfect match to zero, so
        // asking a question in natural language answered "no such thing".
        let exact = idx.search("g1", "upgrade", None, 10).unwrap();
        assert_eq!(exact.len(), 1, "the bare term must still match");

        let with_noise = idx.search("g1", "upgrade zzzznotaword", None, 10).unwrap();
        assert_eq!(
            with_noise.len(),
            1,
            "an unmatched extra token must lower the score, never erase the hit"
        );

        // A whole natural-language question, of which only some words occur.
        let question = idx
            .search(
                "g1",
                "does an existing design survive an upgrade?",
                None,
                10,
            )
            .unwrap();
        assert_eq!(question.len(), 1);
        assert_eq!(question[0].node_id, "r1");
    }

    #[test]
    fn more_matching_tokens_ranks_higher() {
        let (_d, idx) = open_tmp();
        idx.upsert("g1", "Doc", "both", &fields(&[("body", "alpha beta")]))
            .unwrap();
        idx.upsert("g1", "Doc", "one", &fields(&[("body", "alpha only")]))
            .unwrap();
        idx.commit().unwrap();

        // Disjunction must not flatten relevance: BM25 sums a contribution per
        // matched term, so an extra matched term raises the score. Note what
        // this does and does not pin. The two documents here are deliberately
        // the same length and share the same common token, which isolates the
        // summation; it is NOT a general law that matching more terms outranks
        // matching fewer, because BM25 also weighs term rarity and penalizes
        // length — a short document matching one rare token can beat a long one
        // matching several common ones. The contract is "ordered by score", and
        // this test pins that the score responds to term count at all.
        let hits = idx.search("g1", "alpha beta", None, 10).unwrap();
        assert_eq!(hits.len(), 2, "both documents match at least one token");
        assert_eq!(
            hits[0].node_id, "both",
            "with length and shared terms held constant, the extra match scores higher"
        );
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn a_disjunction_still_cannot_return_the_whole_graph() {
        let (_d, idx) = open_tmp();
        idx.upsert("g1", "Doc", "hit", &fields(&[("body", "alpha")]))
            .unwrap();
        idx.upsert("g1", "Doc", "miss", &fields(&[("body", "unrelated")]))
            .unwrap();
        idx.commit().unwrap();

        // The nesting is what stops the graph-scope clause matching on its own.
        // If the term clauses were flattened in beside it, "miss" would come
        // back here, and every search would silently return the entire graph.
        let hits = idx.search("g1", "alpha", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node_id, "hit");

        // And a query of pure noise still matches nothing at all.
        assert!(
            idx.search("g1", "zzzznotaword qqqqnope", None, 10)
                .unwrap()
                .is_empty()
        );
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
    fn open_in_ram_indexes_and_searches() {
        let idx = TextIndex::open_in_ram().unwrap();
        idx.upsert(
            "g1",
            "Document",
            "n1",
            &fields(&[("body", "ephemeral ram index")]),
        )
        .unwrap();
        idx.commit().unwrap();
        let hits = idx.search("g1", "ram", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node_id, "n1");
    }

    #[test]
    fn rollback_discards_uncommitted_writes() {
        let (_d, idx) = open_tmp();
        idx.upsert("g1", "Document", "n1", &fields(&[("body", "committed")]))
            .unwrap();
        idx.commit().unwrap();
        // Buffer a second write, then roll back before committing it.
        idx.upsert("g1", "Document", "n2", &fields(&[("body", "transient")]))
            .unwrap();
        idx.rollback().unwrap();
        idx.commit().unwrap();

        // The committed doc survives; the rolled-back one never appears.
        assert_eq!(idx.search("g1", "committed", None, 10).unwrap().len(), 1);
        assert!(idx.search("g1", "transient", None, 10).unwrap().is_empty());
    }

    #[test]
    fn delete_graph_removes_only_that_graph() {
        let (_d, idx) = open_tmp();
        idx.upsert("g1", "Document", "n1", &fields(&[("body", "term")]))
            .unwrap();
        idx.upsert("g1", "Document", "n2", &fields(&[("body", "term")]))
            .unwrap();
        idx.upsert("g2", "Document", "n3", &fields(&[("body", "term")]))
            .unwrap();
        idx.commit().unwrap();

        idx.delete_graph("g1").unwrap();
        idx.commit().unwrap();
        assert!(idx.search("g1", "term", None, 10).unwrap().is_empty());
        assert_eq!(idx.search("g2", "term", None, 10).unwrap().len(), 1);
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
