# Design: Domain-Neutral Full-Text / BM25 Index Primitive

> Status: in progress.
> Scope decision (locked): **new `dynograph-text` crate**, backed by **Tantivy**,
> **lexical search only** in the first cut (no built-in vector fusion).
> Background and rationale: see `document-queryability-discussion.md`.
>
> Progress:
> - [x] Step 1 — `dynograph-core` `fulltext` flag + helpers + validation.
> - [x] Step 2 — `dynograph-text` crate (`TextIndex` over Tantivy 0.26).
> - [x] Step 3 — `dynograph-storage` feature-gated write-path hooks + reindex.
> - [ ] Step 4 — `dynograph-service` `search:text` endpoint + OpenAPI.
>
> As-built notes (steps 1–3):
> - Commit cadence: a non-batched node write commits the index immediately;
>   batched writes become visible at `commit_batch` and are reverted by
>   `discard_batch` (via `TextIndex::rollback`). RocksDB stays authoritative.
> - The index is built only when `Schema::has_any_fulltext_properties()` — no
>   writer arena for schemas that don't use full-text. In-memory engine uses a
>   RAM index (`TextIndex::open_in_ram`); RocksDB uses a `fulltext/` subdir of
>   the data dir.
> - `reindex_fulltext` is clear-then-rebuild (`TextIndex::delete_graph` then
>   re-upsert every fulltext node from the authoritative store).
> - `librocksdb-sys` needs `clang`/`libclang-dev` at build time (storage tests).

## Goal

Give the foundation a true full-text search capability — tokenized, BM25-ranked
keyword/phrase search over declared text properties — as a **domain-neutral**
primitive. No `Fragment`, `story_id`, chunking, or narrative semantics anywhere
in the foundation. Consumers opt in per-property via the schema, exactly as they
already do for `indexed: true`.

This fills the foundation's one real capability gap: today it supports structured
property filters (`CF_NODE_IDX`) and vector similarity (`CF_EMBEDDINGS` + HNSW),
but has **no inverted index** — keyword search is impossible without a full scan.

### Non-goals (first cut)
- No vector/lexical fusion (RRF, hybrid) in the foundation. That stays a consumer
  concern for now; revisit once the lexical primitive is proven.
- No chunking or document-splitting (consumer/ingestion concern).
- No relevance-tuning UI, synonyms, or learned ranking.

## Existing patterns this mirrors

The foundation already has a clean "declare in schema → maintain a secondary index
on write → query it via the service" pattern. The full-text primitive copies it:

| Concern | Structured index (exists) | Vector index (exists) | Full-text (new) |
|---|---|---|---|
| Schema flag | `indexed: true` on `PropertyDef` (`schema.rs:115`) | `embedding_field` on node type | `fulltext: true` on `PropertyDef` |
| Storage | `CF_NODE_IDX` RocksDB CF | `CF_EMBEDDINGS` CF + in-mem HNSW | Tantivy index dir (new `dynograph-text`) |
| Write hook | `write_index_entries` / `delete_index_entries` (`engine.rs:766/789`) | `set_embedding` (`engine.rs:357`) | `text_index.upsert` / `.delete` |
| Query | `/v1/graphs/{id}/nodes:scan` | `/v1/graphs/{id}/similar` | `/v1/graphs/{id}/search:text` (new) |

## 1. `dynograph-core` — schema flag

Add one field to `PropertyDef` (`schema.rs:109`), mirroring `indexed`:

```rust
pub struct PropertyDef {
    pub prop_type: PropertyType,
    #[serde(default)] pub required: bool,
    #[serde(default)] pub indexed: bool,
    #[serde(default)] pub fulltext: bool,   // NEW
    // ... rest unchanged ...
}
```

Helpers paralleling the existing `indexed_properties` / `has_indexed_properties`
(`schema.rs:654/670`):

```rust
impl Schema {
    /// Names of string properties on this node type carrying `fulltext: true`.
    pub fn fulltext_properties(&self, node_type: &str) -> Vec<&str>;
    pub fn has_fulltext_properties(&self, node_type: &str) -> bool;
}
```

Validation rule (fail loud at schema-load, consistent with the project's
fail-loud philosophy): `fulltext: true` is only valid on `PropertyType::String`.
Reject `fulltext` on Int/Float/Bool/Enum/etc. with a clear error.

Schema YAML, fully domain-neutral:

```yaml
node_types:
  Document:                      # consumer's name; foundation doesn't care
    properties:
      title:   { type: string, fulltext: true }
      body:    { type: string, fulltext: true }
      author:  { type: string, indexed: true }   # structured filter, not FTS
```

## 2. `dynograph-text` — the new crate

A self-contained inverted-index crate. Tantivy is isolated here so consumers who
don't want the dependency don't pay for it (it is **not** pulled into
`dynograph-storage`'s default build — see §5 wiring).

### Public API (storage-agnostic, takes ids + text)

```rust
pub struct TextIndex { /* tantivy::Index + IndexWriter + schema fields */ }

pub struct TextHit { pub node_id: String, pub node_type: String, pub score: f32 }

impl TextIndex {
    /// Open or create the index at `path`. The Tantivy document schema is
    /// fixed: stored keyword fields `graph_id`, `node_type`, `node_id` (for
    /// filtering + retrieval), a non-stored composite key `uid` =
    /// `graph_id\0node_id` (for correct delete-by-term even if graphs share a
    /// dir), and one combined `text` TEXT field (tokenized + BM25).
    ///
    /// IMPLEMENTED (step 2): all full-text property values are concatenated
    /// into the single `text` field. Per-property field targeting (e.g.
    /// `title:foo`) is intentionally out of the first cut — a Tantivy schema is
    /// fixed at creation, so dynamic per-property fields across many node types
    /// would couple the index schema to the dynograph schema. Future extension:
    /// a Tantivy JSON field to recover per-key search without that coupling.
    pub fn open(path: &Path) -> Result<Self, TextError>;

    /// Index (or replace) the full-text fields of one node. `fields` is the
    /// subset of the node's properties declared `fulltext: true`, already
    /// extracted to (prop_name -> string) by the caller. Replace semantics:
    /// delete-by-term(node_id) then add, so re-upsert is idempotent.
    pub fn upsert(
        &self,
        graph_id: &str,
        node_type: &str,
        node_id: &str,
        fields: &[(String, String)],
    ) -> Result<(), TextError>;

    /// Remove a node from the index (delete-by-term on node_id).
    pub fn delete(&self, graph_id: &str, node_id: &str) -> Result<(), TextError>;

    /// BM25 search. `node_type` optionally restricts results to one type.
    /// `query` is parsed by Tantivy's QueryParser over the TEXT fields
    /// (supports terms, phrases, AND/OR). Returns top-k by BM25 score.
    pub fn search(
        &self,
        graph_id: &str,
        query: &str,
        node_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<TextHit>, TextError>;

    /// Commit buffered writes. See §4 for the commit-cadence policy.
    pub fn commit(&self) -> Result<(), TextError>;
}
```

### Tokenization / analyzer
IMPLEMENTED (step 2): the `text` field uses Tantivy's `default` tokenizer
(simple tokenizer + lowercasing, **no stemming**). Predictable for a first cut:
exact-token-after-lowercasing matching. A single fixed analyzer — no per-property
config. Future extensions: switch to `en_stem` for stemmed recall, and/or an
optional `analyzer:` key on the property.

## 3. `dynograph-storage` — write-path hooks

The storage engine extracts the `fulltext` fields and forwards them to the
`TextIndex` at the same points it already maintains `CF_NODE_IDX`:

- **`create_node` (`engine.rs:841`)** — after `write_index_entries`, if
  `schema.has_fulltext_properties(node_type)`, build the `(name, string)` list
  from `properties` and call `text_index.upsert(...)`.
- **`delete_node` (`engine.rs:929`)** — alongside the `CF_NODE_IDX` and
  `CF_EMBEDDINGS` cleanup, call `text_index.delete(graph_id, node_id)`.
- **update path** — same delete-then-write reconciliation the structured index
  uses; for FTS `upsert` already has replace semantics, so a single `upsert`
  suffices.

The engine holds an `Option<TextIndex>` (None when the consumer compiled/configured
without full-text — see §5), so all hooks are `if let Some(idx) = &self.text_index`.

## 4. Consistency model (the one real design tension)

Tantivy is a **separate index with its own commit lifecycle** — it cannot join
RocksDB's atomic `WriteBatch`. So node writes (RocksDB) and their full-text
entries (Tantivy) are not transactionally atomic. Decision for the first cut:

- **RocksDB is the source of truth.** Tantivy is a derived, rebuildable index.
- **Commit cadence:** buffer Tantivy writes and `commit()` on a cadence (e.g. per
  atomic batch boundary, and/or a time/op threshold), not per single node. This
  matches Tantivy's design (commits are relatively expensive) and bounds the
  staleness window to "last commit."
- **Crash / drift recovery:** because RocksDB is authoritative and stores the full
  node bodies, the Tantivy index can always be **rebuilt by scanning `CF_NODES`**
  (see §6). A drift or a corrupt index dir is recoverable, never data loss.
- **Documented contract:** full-text results are **eventually consistent** within
  the commit window. This is acceptable for search and must be stated in the API
  docs so consumers don't assume read-your-write on `search:text`.

This "authoritative store + rebuildable derived index" stance is exactly how the
foundation already treats the HNSW (rebuilt from `CF_EMBEDDINGS` on rehydrate).

## 5. `dynograph-service` — endpoint + wiring

New endpoint, registered in `app()` (`app.rs:320`) next to `nodes:scan` / `similar`:

```
POST /v1/graphs/{id}/search:text
  body:  { "query": "...", "node_type": "Document"?, "limit": 10? }
  resp:  { "results": [ { "node_id": "...", "node_type": "...", "score": 7.3 }, ... ],
           "wire_version": "...", "content_hash": "..." }
```

Handler: resolve the graph, call `text_index.search(graph_id, query, node_type, limit)`,
return hits. Optionally hydrate full nodes (mirror how `similar` returns ids +
scores and lets the caller fetch). Keep parity with existing response envelope
(`wire_version`, `content_hash`).

**Feature-gating the dependency.** Put the Tantivy-backed `dynograph-text` behind
a cargo feature (e.g. `fulltext`) on `dynograph-storage`/`dynograph-service` so a
consumer that doesn't need FTS isn't forced to compile Tantivy. When the feature
is off, `text_index` is `None`, the schema accepts `fulltext: true` but logs/ignores
it (or rejects at load — decide), and `search:text` returns `501 Not Implemented`.

Config: an index directory path alongside the RocksDB path
(`dynograph.example.toml`), per-graph subdir.

## 6. Rebuild / backfill path

A maintenance operation to (re)build the Tantivy index from authoritative
`CF_NODES`, needed for: enabling FTS on an existing graph, recovering from a
corrupt/missing index dir, or a Tantivy schema change.

- Walk all nodes of every type with `has_fulltext_properties`.
- For each, extract the `fulltext` fields and `upsert`.
- Single `commit()` at the end.

Expose as an admin endpoint (`POST /v1/graphs/{id}/search:reindex`) and/or a
startup reconcile. This is the FTS analogue of the embedding rehydrate helper at
`engine.rs:433`.

## 7. Testing

- **Core:** `fulltext` round-trips through schema YAML/JSON; validation rejects
  `fulltext` on non-string types; `fulltext_properties` helper correctness.
- **Text crate:** upsert → search returns the doc; delete removes it; replace
  semantics (re-upsert doesn't duplicate); phrase vs term queries; `node_type`
  filter; BM25 ordering sanity (more term hits ranks higher).
- **Storage integration:** create_node indexes; delete_node de-indexes; update
  replaces; **index survives RocksDB reopen + reindex** (mirror the existing
  `index_survives_through_rocksdb_reopen` test at `engine.rs:2360`).
- **Service:** `search:text` happy path; `node_type` filter; empty query rejected
  (fail loud, like `nodes:scan`); `501` when feature disabled.
- **Rebuild:** drop the index dir, reindex from `CF_NODES`, search works.

## 8. Open questions (decide during implementation)

1. **Commit cadence** — per-batch boundary only, or also a background timer? Start
   with per-batch + an explicit `commit` on the reindex path; measure.
2. **`fulltext: true` when feature disabled** — reject at schema load, or accept +
   warn? Leaning reject-at-load for fail-loud consistency, but that couples schema
   acceptance to build features. Needs a call.
3. **Result hydration** — ids+scores only (like `similar`) vs. inline node bodies.
   Start with ids+scores for symmetry; add an `include_nodes` flag later if needed.
4. **Multi-graph index layout** — one Tantivy dir per graph (simpler isolation,
   matches per-graph HNSW) vs. one shared dir with a `graph_id` filter field.
   Leaning per-graph dir.

## Sequencing

1. `dynograph-core`: `fulltext` flag + helpers + validation + tests. (small, no deps)
2. `dynograph-text`: crate + `TextIndex` over Tantivy + unit tests. (isolated)
3. `dynograph-storage`: feature-gated `text_index`, write-path hooks, reindex,
   reopen test.
4. `dynograph-service`: `search:text` endpoint, `search:reindex`, config, `501`
   path, OpenAPI.

Each step is independently reviewable; (1) and (2) have no interdependency and can
land in parallel.
