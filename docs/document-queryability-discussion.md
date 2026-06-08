# Document Queryability — Discussion Summary

> Status: design notes / decision record. Captures a discussion about whether
> dynograph-foundation can store documents as queryable nodes, what the existing
> stack already provides, and which single addition genuinely belongs in the
> foundation. The actionable outcome is a **domain-neutral full-text/BM25 index
> primitive** (see "Decision" and the dedicated design doc).

## Original question

> Can dynograph-foundation be used to store documents as nodes, with edges
> defining how documents relate, such that each document's **contents** are
> queryable (not just stored)?

## The systems involved

Three distinct pieces, in a clean layering (verified — not forks):

| Component | Path | Role |
|---|---|---|
| **dynograph-foundation** | `git_projects/dynograph-foundation` | Domain-neutral, schema-driven graph storage base. Crates: `dynograph-core`, `dynograph-storage`, `dynograph-resolution`, `dynograph-vector`, `dynograph-service`, `dynograph-client`. Storage is RocksDB column families + per-(graph, node_type) in-memory HNSW. |
| **storyflow `dynograph`** | `git_projects/storyflow/services/dynograph` | A **consumer** of the foundation (git dependency, `tag = "v0.6.0"`), adding domain crates: `dynograph-engine`, `dynograph-server`, `dynograph-extract`, `dynograph-context`, `dynograph-query`, `dynograph-self`. Carries the narrative domain (Story/Fragment/Character, `story_id` access control, GraphRAG endpoints). |
| **embeddings-rs** | `git_projects/storyflow/services/embeddings-rs` | Stateless local embedding sidecar. Runs `nomic-embed-text-v1.5` (768-dim) via ONNX Runtime. `POST /embed`, `/embed-batch`, `/health`. No external API, no storage, no chunking. |

Relationship confirmed via `Cargo.toml`: storyflow's dynograph **depends on** the
foundation crates — it is not a fork. The foundation provides the domain-neutral
base; storyflow extends it. **Test for "belongs in the foundation": is it
domain-neutral?** Anything mentioning `Fragment`, `story_id`, or narrative
semantics fails by definition.

## What "queryable contents" means here

Three common interpretations; the stack supports two of three:

| Query style | Supported? | Mechanism |
|---|---|---|
| **Structured metadata filters** (author = X, date range, type = Y) | ✅ | `nodes:scan` predicates on properties marked `indexed: true` |
| **Semantic / meaning search** ("docs about supply-chain risk") | ✅ (turnkey via embeddings-rs) | embeddings-rs generates vectors → HNSW `find_similar` (cosine) |
| **Full-text / keyword search** (BM25, inverted index) | ❌ | No inverted index anywhere in the foundation. Only a substring-`contains()` fallback in storyflow when the sidecar is down. |

So: **documents can be modeled as nodes with relationship edges, and their
contents are queryable by structured filter and by semantic similarity, today.**
The gap is true full-text/keyword search.

## Key architectural facts discovered (with file references)

1. **embeddings-rs fills the "compute embeddings" gap** the foundation
   deliberately omits. It's a stateless `text → vector` function. It does **not**
   call dynograph; **dynograph calls it**
   (`storyflow/.../dynograph-server/src/compat/embeddings.rs`,
   `EMBEDDING_URL=http://embeddings:8401`).

2. **The storyflow HNSW is a single, global, per-graph index — not per-type.**
   `find_similar(_node_type, …)` ignores `node_type`
   (`storyflow/.../local_backend.rs:188`); type separation is done by
   **post-filtering** resolved nodes (`compat/graphrag.rs:309`). Over-fetch
   (`limit * 2`) compensates for the mixed neighbor space.

3. **The foundation ALREADY persists embeddings and supports HNSW rebuild on
   load** — storyflow bypasses it:
   - `dynograph-storage/src/engine.rs:30` — `CF_EMBEDDINGS` column family.
   - `engine.rs:357` — `set_embedding(...)` persists f32 bytes to `CF_EMBEDDINGS`.
   - `engine.rs:433-441` — load helper: "call this on rehydrate to populate the
     in-memory HNSW per-type" via `prefix_scan(CF_EMBEDDINGS)`.
   - But storyflow's own `dynograph-engine/src/lib.rs:449` `set_embedding` only
     does `vector_index.insert` — it keeps a **separate in-memory HNSW** and never
     persists. Hence storyflow "tech-debt #43: restart drops every embedding."
   - **Conclusion: #43 is a storyflow adoption gap, not a foundation gap. The
     foundation is already correct.**

4. **The foundation has no full-text / lexical / BM25 / tantivy code at all**
   (grep across the repo = empty). This is its one real capability gap.

## "Should X live inside embeddings-rs?" — No, by the statefulness test

The guiding principle: **embeddings-rs is stateless (`text → vector`); anything
that must remember something between requests does not belong in it.**

| Candidate | Verdict | Correct owner |
|---|---|---|
| Full-text / keyword index | No (stateful inverted index, tied to doc lifecycle) | Foundation (domain-neutral primitive) or a dedicated search service |
| Persistent vectors | No (storage concern; the index isn't embeddings-rs's) | Foundation `CF_EMBEDDINGS` (already exists) |
| Chunk storage + retrieval orchestration | No | Consumer ingestion layer + graph nodes/edges |
| Chunk **splitting** (text → chunks) | Borderline — it's stateless and tokenizer-coupled | A shared lib (preferred) or, weak case, embeddings-rs |

## The three sketched additions and where they actually belong

Originally sketched against the **storyflow** service:

1. **Chunk-level retrieval** — `Fragment` → `FragmentChunk` (+ `HAS_CHUNK` edge),
   embed at chunk granularity (remove `embedding_field` from the parent to avoid
   double-counting in the shared HNSW), search rolls chunk hits up to distinct
   parents. **Domain-specific. Does not touch the foundation** — the foundation
   already supports it structurally (arbitrary node types + edges +
   `embedding_field`).

2. **Persistent vectors** — **Already a foundation feature** (see fact #3).
   The real work is a storyflow-side change to adopt the foundation's existing
   `set_embedding` / load-on-rehydrate API instead of its bespoke global HNSW.
   **No foundation change.**

3. **Hybrid (keyword + vector) search** — a lexical/BM25 index fused with vector
   results (e.g. Reciprocal Rank Fusion, which sidesteps embeddings-rs's
   un-normalized vector magnitudes). The **generic half** of this — a full-text
   index primitive — is the one piece that genuinely belongs in, and improves,
   the foundation. The `Fragment`/`story_id`/RRF-rollup specifics stay in
   storyflow.

## Per-item verdict: does it improve dynograph-foundation?

| Item | Improves the foundation? | Why |
|---|---|---|
| #1 Chunking | No | Domain-specific; foundation already supports it structurally |
| #2 Persistent vectors | No | Foundation already has it; storyflow just doesn't use it |
| #3 Full-text / BM25 | **Yes** | Foundation's only true gap; domain-neutral; benefits every consumer |

## Decision

**Focus strictly on #3: a domain-neutral full-text / BM25 index primitive for
dynograph-foundation** — minus all `Fragment` / `story_id` / narrative specifics.

Rough shape (to be detailed in a dedicated design doc):
- Opt-in `fulltext: true` flag on a `PropertyDef` in `dynograph-core`, mirroring
  the existing `indexed: true`.
- A lexical index in `dynograph-storage` (or a new `dynograph-text` crate),
  sitting alongside `CF_NODES` / `CF_EMBEDDINGS` / `CF_NODE_IDX`, kept in sync on
  node writes/deletes.
- `dynograph-service` endpoints: a full-text search endpoint, plus an optional
  fused/hybrid endpoint (lexical + vector).

**Guardrail:** the foundation earns its value by knowing nothing about any
consumer's domain. The moment "document," "chunk," or "story" leaks into
`dynograph-core`, it stops being a foundation. Push only the generic index
primitive down; keep domain semantics in the consumer layer.

### Near-term storyflow fix (separate from #3, noted for completeness)
Route storyflow's `dynograph-engine` through the foundation's existing persistent
embedding API to retire tech-debt #43 by adoption rather than new code.
