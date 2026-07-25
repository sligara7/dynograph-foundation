# Changelog

Notable changes to `dynograph-foundation`. Format loosely follows
[Keep a Changelog](https://keepachangelog.com); versions match the
workspace `version` in `Cargo.toml`.

## v0.11.0 — 2026-07-24

### Changed
- **Full-text search ranks a multi-term query instead of excluding on it.**
  `TextIndex::search` keeps the same signature and the same guarantees about
  query grammar (the raw string is still never fed to Tantivy's parser, so
  `field:value` input cannot inject a filter), but tokens are now matched as a
  **ranked disjunction**: a document must contain at least one token, and BM25
  sorts documents matching more — and rarer — tokens above those matching fewer.
  A document containing every token still comes back first.

  Previously every token had to occur. That made a natural-language question
  unusable as a query, because one incidental word the corpus never happened to
  use collapsed a perfect match to zero results:

  ```
  search("upgrade")              -> 1 hit, score 8.29
  search("upgrade zzzznotaword") -> 0 hits
  ```

  Zero hits reads as "no such thing exists", which is the worst available answer
  for the caller this index primarily serves: consumers search before creating a
  node and treat an empty result as licence to create. A false negative
  therefore **manufactured duplicates** — the index degraded the graph it exists
  to protect. Discrimination has moved from exclusion to ranking; callers wanting
  a narrower result should pass fewer, better words or filter by `node_type`.

  **Consumers should expect more hits per query than before, ordered by
  relevance**, and should read the top result rather than assume the set is
  already filtered. `multi_term_query_is_conjunctive` was flipped (not deleted)
  to `multi_term_query_ranks_rather_than_excludes` so the contract change is on
  the record; three tests were added covering an unmatched token, rank ordering,
  and the guarantee that a disjunction still cannot return the whole graph or
  match pure noise.

## v0.10.0 — 2026-07-17

### Changed
- **Community detection now uses Leiden instead of Louvain.** `POST
  /v1/graphs/{id}/algo/communities` keeps the same request/response contract
  (`{scope?, weight?, direction?, resolution?}` → `{count, communities,
  modularity}`), but the partition is now computed by the Leiden method (Traag,
  Waltman & van Eck, 2019), the successor to Louvain. Leiden adds a refinement
  phase that repairs Louvain's one real defect — communities that come out
  badly connected or even internally disconnected — so **every returned
  community is now guaranteed to be internally connected**. We run the
  deterministic greedy variant (no randomized refinement move), preserving the
  suite's reproducibility invariant; the connectivity guarantee comes from the
  refinement structure, not the randomness. Partitions for a given graph may
  differ from the old Louvain output (results are read-only and not persisted).
  Guarded by a new connectivity test on planted-community graphs.

## v0.9.4 — 2026-07-11

### Fixed
- **UDS transport: request paths with invalid URI characters no longer panic.**
  A path segment containing a space or other RFC 3986-invalid character
  (e.g. a node name like `Sir Testwell` interpolated into
  `/v1/graphs/{g}/nodes/{name}`) hit `hyperlocal`'s internal `Uri` unwrap and
  panicked the in-flight request, dropping the connection (consumers saw a
  gateway 502 with no server error). The client now percent-encodes invalid
  path characters at the transport seam — matching the TCP arm, where
  reqwest's URL parser already did this — and pre-validates the assembled
  path into a typed `ClientError::Unix` so any still-invalid path is an
  error, never a panic. Regression tests pin both the encoding and the
  no-panic invariant at the exact former panic site.

## v0.9.3 — 2026-06-12

Source-aware entity resolution — fixes the alias over-merge class
shipped in v0.9.2: two distinct same-scope entities sharing a generic
alias ("the captain") exact-matched at score 100 and silently
auto-merged, discarding the incoming entity's profile.

### Added

- `EntityResolver::resolve_sourced`: resolution with match provenance.
  Tier-1 auto-merge considers name↔name pairs only; any alias-sourced
  pair (incoming or stored, either direction) requires vector
  corroboration (cosine ≥ `vector_threshold`) at ANY fuzzy score —
  including an exact 100 — and falls back to `CreateNew` when the
  caller supplies no embedding.
- Alias-ambiguity exclusion: an incoming alias matching ≥2 distinct
  in-scope candidates above the fuzzy threshold is non-identifying by
  construction — excluded from merge justification and reported.
- `/resolve-or-create` response gains `match_source`
  (`name_to_name` / `name_to_stored_alias` / `incoming_alias_to_name` /
  `incoming_alias_to_stored_alias`; `null` on `created_new`) and
  `ambiguous_aliases: [String]`. Both additive.
- `MatchSource` / `ResolutionOutcome` exported from
  `dynograph-resolution`; client wire struct carries the new response
  fields with serde defaults (compatible with pre-0.9.3 servers).

### Changed

- A stored alias equal to another entity's primary name can no longer
  outscore-and-hijack that entity's exact-name auto-merge (tier-1 is
  name-pairs-only; previously only a sort-stability tiebreak protected
  the equal-score case).

### Deprecated

- `EntityResolver::resolve_with_aliases` — flattens match provenance,
  so alias pairs auto-merge on fuzzy score alone. Use `resolve_sourced`.

### Known consumer note

- Callers that send no embeddings get `created_new` for every
  alias-sourced match — the fail-safe direction (recoverable
  duplication instead of silent merge corruption). Alias dedup re-arms
  once the caller supplies embeddings on the resolve call.

## v0.9.2 — 2026-06-10

Alias-aware entity resolution — additive; default behavior for requests
without aliases (and corpora without stored aliases) is unchanged.

### Added

- `EntityResolver::resolve_with_aliases`: resolves the primary name first;
  if it would create, each incoming alias is resolved in turn against the
  same candidates and the first merge wins. Empty aliases and
  case-insensitive duplicates of the primary are skipped.
- `/resolve-or-create` accepts `incoming_aliases: [String]` (wire alias:
  `aliases`) and feeds each candidate node's stored `aliases` property —
  a `List` of strings or a JSON-array-encoded string — into the candidate
  list, so an incoming name can merge into a node it only matches via a
  stored alias. Previously an `incoming_aliases` request field was
  silently dropped by serde and stored aliases were invisible to
  resolution over HTTP.

### Fixed (within the new surface)

- Tie precedence: all primary-name candidates precede all alias
  candidates, so on an equal fuzzy score an entity's primary name beats
  another entity's stored alias — an alias can't hijack exact-name merges.

## v0.9.1 — 2026-06-09

Internal refactor only — no behavior, API, or wire-contract change
(`cargo-semver-checks` and the OpenAPI drift gate both confirm the public
surface is unchanged).

### Changed

- Split the monolithic `crates/dynograph-storage/src/engine.rs` (3660 lines)
  into an `engine/` module: the `StorageEngine` public API methods are grouped
  by concern (`nodes` / `edges` / `scan` / `batch` / `fulltext` / `embeddings`),
  with the shared private machinery in `engine/mod.rs` and the test suites in
  their own files. Pure code move — `StorageEngine` / `StoredNode` / `StoredEdge`
  and all behavior are unchanged. Completes the monolithic-file split arc begun
  in v0.9.0 (`algo.rs` / `app.rs` / `integration.rs`).

## v0.9.0 — 2026-06-09

Adds density-based clustering to the stateless math surface and makes the
RocksDB backend an opt-out build feature, plus a round of internal
restructuring (the monolithic `algo.rs` / `app.rs` / `integration.rs` split into
modules, and a `KvBackend` storage trait). All additive — default runtime
behavior is unchanged.

### Added (`dynograph-cluster` — new crate)

- A pure, dependency-free **density-based clustering** crate (the fourth math
  sibling of `dynograph-vector` / `dynograph-graph` / `dynograph-game`): a
  precomputed distance matrix in, a cluster label per point out. `dbscan(
  distance_matrix, eps, min_points) -> Result<Vec<i32>, ClusterError>`
  implements the classic Ester et al. (1996) algorithm — exact and
  deterministic, `-1` = noise and `1..` = (1-based) cluster id, fail-loud
  validation of the matrix and parameters. The caller supplies the distances
  (the same way it supplies dimension vectors); this is matrix in / labels out,
  not a graph algorithm. Distinct from `dynograph-graph`'s Louvain
  (community-on-edges) — this is density-on-points.

### Added (`dynograph-service`)

- **`POST /v1/util/dbscan`** — DBSCAN over a caller-supplied distance matrix,
  exposed beside the other stateless `util/*` pure-math ops (no graph id, no
  `graph` feature gate — it takes a matrix, not the stored graph's topology).
  Returns `{ labels, num_clusters }`. Pairs with `util/pairwise_distance`, which
  produces exactly the kind of matrix it consumes. Fail-loud on a malformed
  matrix (empty / non-square / non-finite / negative) or parameter (`eps`
  non-finite/negative, `min_points` zero), and capped at 4096 points.

### Changed

- **RocksDB is now an optional cargo feature** (`dynograph-storage` and
  `dynograph-service` both gain a `rocksdb` feature, **on by default**). The
  default build and the published image are unchanged (RocksDB on); building
  with `--no-default-features` produces an in-memory-only binary that skips the
  RocksDB C++ compile and fails loud at startup if `storage.root` is set.

### Internal

- Storage is now backend-agnostic behind a `KvBackend` trait (RocksDB and the
  in-memory map sit behind one trait; a new backend is one file). No public
  behavior change; includes a `MemoryBackend` `prefix_scan` ordering fix to
  honour the key-order contract.
- Split the monolithic `algo.rs` → `algo/` module, `app.rs` → `app/` module
  (handlers + apidoc), and the integration test file → per-domain `it_*.rs` with
  a shared `common/` (pure code moves; behavior and the wire contract unchanged).
- Expanded endpoint-catalog and schema-driven-configuration docs.

## v0.8.0 — 2026-06-09

Closes the consumer graph-analytics integration arc: a single graph can now be
**partitioned by a node property** and analyzed per-partition, **community
detection** finds dense sub-groups within a connected graph, `/batch` gains an
honest **validate-before-apply** mode, and the typed **client** wraps the whole
v0.7.0+ surface so consumers reach it without hand-rolling HTTP. Plus a new
domain-neutral **game-theory analyzer**. All additive — no breaking changes.

### Added (`dynograph-game` — new crate)

- A pure, dependency-free normal-form game-theory crate (the third math sibling
  of `dynograph-vector` / `dynograph-graph`): a payoff matrix in, the strategic
  analysis out. Per-player strict/weak **dominant strategies**, **pure-strategy
  Nash** equilibria, **Pareto-optimal** outcomes, the headline
  **`nash_is_pareto_suboptimal`** classification (the prisoner's-dilemma
  "rational play → collectively worse" signal) with the Pareto outcome that
  dominates each suboptimal equilibrium, and the closed-form interior **2×2
  mixed** Nash. Out of scope by design: general/non-2×2 mixed Nash, cooperative
  solution concepts, iterated-game simulation.

### Added (`dynograph-service`)

- **`scope.where` on every `algo/*` endpoint** — an optional property predicate
  (the `nodes:scan` / `search:hybrid` clause grammar) that projects an algorithm
  onto one logical subgraph **partitioned by a node property**, not just by
  node/edge type. Enables per-tenant analytics (e.g. per-story PageRank) over a
  single shared graph; only matching nodes and the edges between them enter the
  projection. An explicit empty `scope.node_types` is now a `400` (was a silent
  empty graph).
- **`POST /v1/graphs/{id}/algo/communities`** — Louvain community detection
  (modularity maximization, undirected): the faction/cluster-discovery primitive
  that components (reachability) and clustering (local cliquiness) don't cover.
  Returns the partition plus its modularity; honors the shared `scope`/`where`
  and edge weights. Behind the `graph` feature like the rest of `algo/*`.
- **`/batch` `dry_run` (validate-only)** — runs every op against the batch
  buffer to produce a per-op pass/fail report, then discards without mutating
  the graph (HTTP 200 with a `BatchValidation` body). Unblocks
  preview-before-apply flows. The `200` body is now a `oneOf` of the commit
  summary or the dry-run report; the commit path is byte-for-byte unchanged.
- **`POST /v1/util/game/analyze`** — the game-theory analyzer over HTTP
  (stateless pure math, always-on like the other `util/*` ops). Caps the profile
  count and offloads to the blocking pool like the pairwise matrix ops.

### Added (`dynograph-client`)

- Typed wrappers for the v0.7.0+ surface so consumers reach it through the
  client instead of raw HTTP: **`search_hybrid`**, **`search_text`**,
  **`search_reindex`**; **`util_pairwise_cosine`** / **`util_pairwise_distance`**
  (with `DistanceMetric`); the **15 `algo_*`** methods plus **`algo_communities`**
  (untyped request, typed responses); and a typed **`batch_dry_run`**. Feature-
  gated server routes surface a `501` as a typed `ClientError::Http`.

### Changed (packaging)

- Bump workspace version 0.7.0 → 0.8.0; promote docs/version strings and the
  generated `docs/openapi.json` (`info.version`) to match.

## v0.7.0 — 2026-06-08

The domain-neutral graph-theory algorithm suite plus batch vector math, so
consumers get **all** their graph / vector / stats math from the foundation
behind one stable wire contract instead of reimplementing it (numpy adjacency
SVD, hand-rolled cosine loops, etc.). All additive — no breaking changes.

### Added (`dynograph-graph` — new crate)

- A pure, dependency-free graph-theory crate (the topology sibling of
  `dynograph-vector`): exact algorithms over a densely-indexed in-memory `Graph`.
  Components (weak + strongly-connected/Tarjan); centrality (degree, PageRank,
  personalized PageRank / random-walk-with-restart, eigenvector, closeness,
  betweenness); structure (articulation points & bridges, directed cycle
  detection, topological sort, clustering coefficient / transitivity); paths &
  flow (single-pair shortest path, max-flow / min-cut); and link prediction
  (common-neighbors / Jaccard / Adamic-Adar). Deep traversals are iterative so
  they can't overflow the stack at the node cap.

### Added (`dynograph-service`)

- **`POST /v1/graphs/{id}/algo/*`** — 17 graph-algorithm endpoints over the
  generic node/edge graph. Each request supplies the domain-specific parts (a
  subgraph `scope`, an edge-weight projection, a direction) and gets back a
  generic result (scores, partitions, paths, cuts). Behind the optional `graph`
  build feature (mirroring `fulltext`): the routes and OpenAPI contract always
  exist; without the feature they return `501`. The published image enables it.
- **`POST /v1/util/pairwise_cosine`** and **`POST /v1/util/pairwise_distance`** —
  batch N×N matrix forms so a consumer ranking N entities does one call instead
  of N² per-pair round-trips. Offloaded to the blocking pool.

### Changed (packaging)

- Bump workspace version 0.6.3 → 0.7.0; promote docs/version strings and the
  generated `docs/openapi.json` (`info.version`) to match. Release image and the
  smoke test build with `--features graph,dynograph-service/fulltext`.

## v0.6.3 — 2026-06-08

A domain-neutral hybrid-search primitive that fuses the retrieval legs the
foundation already has, so consumers' NL→graph / GraphRAG layers stop
reimplementing rank fusion.

### Added (dynograph-service)

- **`POST /v1/graphs/{id}/search:hybrid`** — fans out to the vector (HNSW) and
  keyword (BM25) legs and **Reciprocal-Rank-Fuses** their ranked outputs into a
  single ranked node list (`score = Σ_leg weight_leg / (k_rrf + rank_leg)`,
  `k_rrf = 60`). Rank-based on purpose, so it's immune to un-normalized
  embedding magnitudes — no score normalization needed. Each hit carries a
  per-leg `{rank, score}` breakdown.
  - An optional structured `where` clause acts as an **intersect prefilter**
    (same grammar as `nodes:scan`), constraining every ranked leg; it is not a
    fusion leg of its own. At least one ranked leg (`query` and/or
    `query_vector`) is required — a pure `where` filter is what `nodes:scan` is
    for.
  - The vector leg requires `node_type` (HNSW indexes are per-type; a cross-type
    fan-out would silently skip mismatched-dim indexes, so it fails loud).
  - The keyword leg is behind the opt-in `fulltext` cargo feature; requesting it
    in a build without the feature returns `501`, exactly like `search:text`.
    Vector-only requests succeed in any build.
  - Optional per-leg `weights`, `k_per_leg` (candidates per leg pre-fusion), and
    `limit` (final cap). Foundation never embeds — the caller supplies
    `query_vector`.

### Changed (packaging)

- Bump workspace version 0.6.2 → 0.6.3; promote docs/version strings and the
  generated `docs/openapi.json` (`info.version`) to match.

## v0.6.2 — 2026-06-07

Turn on full-text/BM25 search in the shipped artifact and cover it in CI. The
full-text primitive itself — the `dynograph-text` embedded-Tantivy index, the
`fulltext: true` property flag, and the `POST /v1/graphs/{id}/search:text` /
`search:reindex` endpoints — shipped in v0.5.9 behind an opt-in `fulltext` cargo
feature (off by default), but the published image was built without it, so those
endpoints returned `501 Not Implemented` in the container.

### Changed (packaging)

- The release Docker image now builds with `--features
  dynograph-service/fulltext`, so full-text search is live in the shipped
  container instead of returning 501. The feature remains opt-in for source
  builds; only the published artifact changes.

### CI

- New `fulltext` job builds, tests, and clippy-lints the workspace with the
  `fulltext` feature on, exercising the `#[cfg(feature = "fulltext")]`
  storage/service wiring that the default-feature legs skip. The `smoke-test`
  job now builds the binary with the same feature, so it boots the exact
  configuration the published image ships.

## v0.6.1 — 2026-06-07

Domain-neutral vector/stats math: more distances, element-wise algebra,
descriptive statistics, and filtered HNSW search — exposed over 17 new
`POST /v1/util/*` endpoints and wrapped in `dynograph-client`.

### Added (dynograph-vector)

- **Distances:** `squared_euclidean_distance`, `manhattan_distance`
  (each with an f64 variant). `euclidean_distance` now delegates to the
  squared form.
- **Element-wise algebra:** `add`, `subtract`, `scale`, `negate`,
  `hadamard_division` (→ `None` on a zero divisor), `elementwise_power`
  (each with an f64 variant).
- **Transforms:** `l2_normalize` (→ `None` on zero/non-finite
  magnitude), `centroid` (component-wise mean; `None` if empty/ragged),
  each with an f64 variant.
- **Descriptive statistics:** `mean`, `variance` / `std_dev` (sample,
  `n-1`), `median`, `percentile` (linear interpolation), `softmax`
  (numerically stable), `spearman_rank_correlation`. All return `None`
  on degenerate input (no silent default).
- **HNSW:** `HnswIndex::search_filtered(query, k, predicate)` — k-NN
  with an id predicate (ACL / type / exclude-self), a post-filter over
  the same beam `search` uses.

### Added (service)

- 17 new `POST /v1/util/*` endpoints exposing the stateless math above:
  `squared_euclidean_distance`, `manhattan_distance`, `add`, `subtract`,
  `scale`, `negate`, `hadamard_division`, `elementwise_power`,
  `l2_normalize`, `centroid`, `mean`, `variance`, `std_dev`, `median`,
  `percentile`, `softmax`, `spearman_correlation`. Vector ops honor the
  `precision` field (f32/f64); statistics are f64-only. A degenerate
  result (zero divisor, zero magnitude, out-of-range percentile,
  constant input) is a loud 400, not a silent default.

### Added (client)

- `dynograph-client` methods for all 17 new util endpoints.

## v0.6.0 — 2026-06-07

Optional Unix-domain-socket transport, served alongside TCP, with
matching client support. A faster same-host path for co-located
consumers; TCP-only remains the default, so the release is
backward-compatible at the transport level. The client's `ClientError`
becomes `#[non_exhaustive]` (the one source-compatibility note).

### Added (service)

- **Unix-domain-socket transport** — set `[server].uds_path` (or
  `DYNOGRAPH_UDS_PATH`) to serve the full `/v1` API on a Unix socket
  *in addition to* the TCP `bind` address. Same router, auth, limits,
  and OpenAPI on both transports; a faster path for co-located
  consumers with no TCP/IP stack overhead. TCP-only
  remains the default. A stale socket left by a crashed prior run is
  reclaimed on start; an existing non-socket file at the path is a
  hard startup error rather than something silently overwritten.

### Added (client)

- **`DynographClient::connect_unix(path)`** — reach the service over its
  Unix-socket transport. Connections are pooled/kept-alive (where the
  UDS win over TCP is largest, per the `transport_bench` example).
  Identical method surface to the TCP client (`new`) — every call
  behaves the same; only the constructor differs. `base_url()` returns
  the socket path for UDS clients.

### Changed (client)

- `ClientError` is now `#[non_exhaustive]` and gains a `Unix(String)`
  variant for Unix-transport failures (connect / timeout / malformed
  request), kept separate from the reqwest-backed `Network`. Downstream
  `match`es need a wildcard arm.

## v0.5.6 — 2026-05-10

Tier-3 primitives exposed over HTTP; Tier-2 primitives wrapped in
`dynograph-client`. Closes the "everything in foundation must be
reachable from a Python consumer and a future Rust extraction
crate" gap surfaced by a consumer capabilities inventory.

### Added (service)

- **`POST /v1/graphs/{id}/nodes:exists`** — batch `(type, name)`
  existence check. Returns per-query `{exists, id}` in request order
  so the caller can zip queries with results. Replaces N round-trips
  via `list_nodes` (a two-pass extraction relevance gate)
  with a single HTTP call. Pre-flight rejects requests where `name`
  isn't `indexed: true` — the un-indexed-rejection policy
  `/resolve-or-create` and `/edges:collect` already use.

- **`POST /v1/graphs/{id}/nodes:scan`** — predicate-filtered scan over
  a single node type. Seven AND-combined operators (`eq` / `neq` /
  `in` / `gt` / `lt` / `gte` / `lte`). Seed strategy: the first `eq`
  clause drives an index-backed `scan_nodes_by_property`; without one
  it falls back to a full per-type scan. Remaining clauses evaluated
  in memory per row. Range ops support `Int` and lex-ordered
  `String` (Datetime rides the String path). `Op::In` lists capped at
  `MAX_IN_LIST_LEN` (1_000) to bound CPU.

- **`POST /v1/graphs/{id}/edges/{type}/{from}/{to}/welford_update`**
  — atomic Welford-style EMA update of the
  `(score, score_m2, score_stddev, score_min, score_max, score_count)`
  property family. Hybrid: fixed-α EMA for the running estimate,
  Welford m2 accumulation for variance. Whole read-modify-write
  serializes under one `with_engine_write` lock; preserves any
  non-Welford properties already on the edge. Replaces client-side
  read-modify-write that was race-safe only inside a `/batch` call.

- **`POST /v1/util/*`** — nine pure-math utility endpoints exposing
  the load-bearing Tier-3 functions:
  - `cosine_similarity`, `dot_product`, `euclidean_distance`, `l2_norm`,
    `hadamard` (vector ops; `precision: "f32" | "f64"`, default `f64`)
  - `pearson_correlation`, `linear_regression_slope` (f64 only)
  - `jaro_winkler`, `token_sort_ratio` (string fuzzy match)
  Stateless — no `graph_id` in the path. Per-request CPU bounded by
  `MAX_VECTOR_LEN` (100_000); chunk client-side above that. Sits
  under `/v1/` so auth middleware still applies.

### Added (client)

- Typed `dynograph-client` methods for every audit-promoted primitive
  and every v0.5.6 endpoint:
  - `batch` / `resolve_or_create` / `edges_collect` / `traverse` (P2)
  - `nodes_exists` / `nodes_scan` / `welford_update` (new in v0.5.6)
  - `util_*` (nine math endpoints)
  Complex-shaped routes (batch, edges_collect, traverse, nodes_scan)
  take `&serde_json::Value` for the request and return
  `serde_json::Value` for the response — same untyped-body pattern
  `create_node` already uses for properties. Simple-shaped routes
  return typed wire structs (`ResolveOrCreateResponse`,
  `NodesExistsResponse`, `WelfordUpdateResponse`, `UtilScalarResponse`,
  `UtilVectorResponse`, `UtilRatioResponse`). Future PRs can replace
  any complex-route wrapper with a typed shell.

### Changed

- **`MAX_LIMIT` consolidated.** Pre-v0.5.6, three modules each
  defined their own `pub(crate) const MAX_LIMIT: usize = 10_000;`
  and rewrote the same `if limit == 0 || limit > MAX_LIMIT` check
  inline. v0.5.6 hoists both to `crate::validation` —
  `validate_limit(limit, context)` and the shared `MAX_LIMIT` const.
  `edges_collect`, `traverse`, and `nodes_scan` now share one
  implementation. Error wording unchanged.

- **`/nodes:scan` reuses `NodeResponse`** (`{node_type, node_id,
  properties}`) for its `Nodes` return shape. Bespoke `{type, id,
  properties}` shape would have drifted from every other node-returning
  endpoint; picked the existing shape so consumers can deserialize
  into one struct.

### Internal

- 3 unit tests + 7 integration tests for welford; 16 integration
  tests for nodes:scan; 6 integration tests for nodes:exists; 9
  client integration tests for util endpoints; 7 client integration
  tests for the audit-promoted primitives. Total workspace: 24
  client + 155 service + 56 service-unit + 9 persistence passing.

## v0.5.5 — 2026-05-06

Contract change: read-your-own-writes within a batch.

### Changed

- **`engine.get` and `engine.prefix_scan` now consult the batch
  buffer.** Pre-v0.5.5 the buffer was write-only — `engine.put` /
  `engine.delete` / `engine.prefix_delete` queued ops while reads
  bypassed the buffer and went straight to RocksDB. Reads inside an
  active batch saw the **pre-batch** state, regardless of what
  earlier ops in the same batch had buffered. That contract was
  documented and tested but proved a footgun for downstream consumers
  that naturally expected transactional read-your-own-writes
  semantics — surfaced concretely by a consumer's fragment-integration
  handler, where an entity created early in the batch was
  invisible to subsequent `resolve_entity` (= `scan_nodes`) calls
  for the rest of the batch, silently dropping cross-entity edges.

  v0.5.5 makes reads buffer-aware:

  - `get(cf, key)` walks the buffer in reverse for the latest
    matching op; a buffered `Put` returns its value, a `Delete` or
    covering `PrefixDelete` returns `None`. A miss falls through to
    the cache + backend.
  - `prefix_scan(cf, prefix)` reads backend results, then overlays
    the buffer in insertion order: `Put` upserts, `Delete` removes,
    `PrefixDelete` prunes everything starting with that prefix. Late
    puts can resurrect a key that an earlier `PrefixDelete` in the
    same batch tombstoned — order is preserved end-to-end.

  Compositions that previously failed now succeed:

  - `create_node X` then `replace_node X` — the replace sees the
    buffered create.
  - `create_edge X→Y` then `merge_edge X→Y` — the merge composes on
    top of the buffered create.
  - `create_edge X→Y` then `delete_node X` — the cascade sees the
    buffered edge and tombstones it; no orphan survives.

  Discarded batches are unchanged: the buffer drops, the in-batch
  view is invisible to anyone after `discard_batch()`.

  Two integration tests in `crates/dynograph-service/tests/integration.rs`
  flipped: `batch_modify_after_create_in_same_batch_fails` →
  `batch_modify_after_create_in_same_batch_succeeds`, and
  `batch_orphan_edge_when_delete_node_in_same_batch` →
  `batch_delete_node_cascades_in_batch_edges`. Both anticipated this
  flip in their pre-v0.5.5 doc comments.

  Performance: the buffer is bounded (the heaviest known consumer case
  caps at ~67 ops; `MAX_BATCH_OPS` is 1000). The reverse-walk in `get` is
  O(buffer) per call; the overlay in `prefix_scan` is O(scan + buffer)
  with logarithmic upsert/remove via BTreeMap. Microseconds in
  practice.

### Migration notes

If your code relied on the pre-v0.5.5 "reads see pre-batch state"
contract — e.g., as a way to build a delta against the snapshot at
`begin_batch()` time — that path no longer works. Capture the
pre-batch state explicitly before `begin_batch()` if you still need
it. We don't expect any out-of-tree consumers to have done this; the
contract was always fragile (a single in-batch write would break the
delta), and the only known consumer of foundation has been
audited.

## v0.5.4 — 2026-05-06

Fourth and final primitive identified by the foundation
audit (2026-05-04). With this release all four enumerated
primitives (`/batch`, `/resolve-or-create`, `/edges:collect`,
`/traverse`) are in main.

### Added

- **`POST /v1/graphs/{id}/traverse`** — typed BFS over one or more
  edge-type steps from a single start node. Backs a consumer's
  `compute_predecessors` shape: transitive walk along a single
  edge type from a start node, scoped by a node property. Used
  today by `state_at_epoch`, `events_between_epochs`, and the
  `add_precedes` cycle check — one HTTP call after migration vs.
  one round-trip per edge plus client-side BFS today.

  Request: required `start: {type, id}`; required `traverse` array
  (max 10 steps), each `{edge_type, direction, transitive?}`;
  optional `scope: {prop, value}` (prop must be `indexed: true` on
  the start type and every per-step candidate type); optional
  `return: "ids" | "nodes"` (default `"ids"`); required `limit` in
  `1..=10_000`.

  `direction` is `"outgoing"` / `"incoming"` / `"both"`.
  `transitive: true` BFS-walks the step's edges to exhaustion
  before advancing; `transitive: false` is one hop per visited
  node.

  Response: `{nodes: [{node_type, node_id, properties?}], truncated}`.
  `properties` is omitted in `return: "ids"` mode via
  `skip_serializing_if`.

  Whole BFS runs under one `with_engine_read` lock — candidate
  scans, edge walks, and node lookups see one consistent snapshot.

  Pre-flight validation (all 400, no scans on failure): unknown
  `start.type`; empty `start.id`; empty `traverse`; chain length
  > 10; unknown `edge_type`; `limit` out of range; un-indexed
  `scope.prop` on any candidate type. Same masked-misconfiguration
  rejection rule used by `/edges:collect` and `/resolve-or-create`.

  PR: [#9](https://github.com/sligara7/dynograph-foundation/pull/9).

### Two semantic decisions worth knowing

- **Start is never in the result.** Mirrors
  `compute_predecessors`'s "predecessors of X (not including X)"
  shape; caller already has the start id. Visited set still holds
  start so a cycle back through it short-circuits.
- **Start not found in storage → 404** (loud failure per the
  no-silent-fallback principle). Start exists but fails the scope
  filter → `200` with empty `nodes` (legitimate "no matches", not
  a misconfiguration).

### Two BFS bookkeeping notes

- **`emitted` keyed by `(node_type, node_id)`** — result-set
  dedup with UNION semantics across steps. A peer reached at step
  0 stays in results even if step 1's edges would also visit it.
- **`queued_steps` keyed by `(node_id, step_idx)`** — work-item
  cycle guard. Same id at a different `step_idx` is a different
  work item, which is how multi-step chains advance.

Transitive steps fan out to BOTH `(peer, step_idx)` (continue
transitively) AND `(peer, step_idx+1)` (advance), so a chain like
`[transitive A, then B]` doesn't starve step B.

### Internal

- `peer_cache` keyed by `peer_id` collapses repeat `fetch_peer`
  lookups across transitive iterations and across multiple sources
  pointing at the same target. Cache hits are validated against
  the current candidate set so `direction: "both"` with different
  per-direction candidates stays correct.

## v0.5.3 — 2026-05-05

Third of four primitives identified by the foundation
audit (2026-05-04). Closes the read-side fan-out gap.

### Added

- **`POST /v1/graphs/{id}/edges:collect`** — fan-out edge
  collection across a typed source set. Replaces a
  `collect_*_edges` master pattern (today walks N entity
  types × M nodes × K edge types via per-node `outgoing_edges` —
  hundreds of round-trips per call); one HTTP call after
  migration. Used by 13+ knowledge-graph routes plus the
  projection step in pagerank/louvain/shortest-path.

  Request: `source.type` accepts `"*"` (all node types) or a
  single name or an array; optional `source.filter: {prop, value}`
  scope (prop must be `indexed: true`); required `edge_types`
  (non-empty); required `limit` in `1..=10_000`; optional
  `format: "edges" | "adjacency"` (default `"edges"`); optional
  `resolve_target: bool` (default false).

  Response shape varies by format. `"edges"` returns
  `{edges: [...], truncated: bool}` with each edge carrying
  `edge_type`, `from_type` (known from the source-type scan),
  `from_id`, `to_id`, `properties`, plus `target` when
  `resolve_target=true`. `"adjacency"` returns
  `{adjacency: {from_id: [...]}, truncated: bool}` for
  client-side algorithms (pagerank/louvain/shortest-path
  projection).

  Whole call runs under one `with_engine_read` lock —
  candidate scan + per-source-node `scan_outgoing_edges` +
  optional target resolution all see a consistent snapshot.

  Pre-flight validation (all 400, no scans on failure): empty
  `edge_types`; unknown `edge_type`; unknown `source.type` name
  (single, list, or post-wildcard); `limit` out of range;
  `source.filter.prop` not `indexed: true` on every covered
  source type (otherwise `scan_nodes_by_property` silently
  returns empty per-type, masking misconfiguration — same
  rejection rule `/resolve-or-create` uses).

  PR: [#7](https://github.com/sligara7/dynograph-foundation/pull/7).

### Two design notes worth knowing

- **No edge-type-prefix scan exists in storage.** Adjacency CFs
  are keyed `(graph_id, node_id, edge_type, peer_id)` — by node,
  not edge type. So wildcard `source.type` without filter walks
  O(N) prefix scans (one per node). A future CF
  `(graph_id, edge_type, from_id, to_id)` would enable true
  edge-type scans; out of scope until profiling proves per-node
  walk is the bottleneck.
- **`StoredEdge` carries no `to_type`.** When `resolve_target=true`,
  foundation discovers each target's type by walking the schema's
  `EdgeTypeDef.to` declaration and trying `get_node` for each
  candidate. Bounded by the schema (typically 1-3 candidates per
  edge type); wildcard endpoints with `resolve_target=true` pay
  full per-edge fanout cost.

### Internal

- New `crate::validation` module with `validate_indexed_property`
  helper, shared between `/resolve-or-create` and `/edges:collect`
  (was duplicated). Net cleanup; no external behavior change.

## v0.5.2 — 2026-05-05

Second of four primitives identified by the foundation
audit (2026-05-04). Closes the LLM-extraction migration gate.

### Added

- **`POST /v1/graphs/{id}/resolve-or-create`** — fuzzy/vector entity
  resolution with create-on-miss semantics. Exposes the existing
  `dynograph-resolution` crate (token_sort_ratio + jaro_winkler with
  cosine-similarity tiebreaker) over HTTP. A consumer's LLM
  extraction funnels every entity through an embedded
  resolve-or-create call today; after migration each
  entity is one HTTP call to this route.

  Body carries `node_type` + `properties` (including the query name
  at `properties.name`) + optional `embedding` for vector tiebreaking
  + optional `scope: {prop, value}` to filter candidates by an
  indexed property (e.g. `story_id`). Returns
  `{id, was_created, match_kind}` where `match_kind` is one of
  `auto_merge` / `vector_merge` / `created_new` — extension over the
  audit's `{id, was_created}` sketch, distinguishes auto-merge from
  vector-merge for consumer-side observability and threshold tuning.

  Pre-flight validation pushes every checkable failure ahead of any
  writes (all 400, no state changes): unknown node_type; type with
  no `resolution` block in schema (no silent fallback to defaults —
  explicit-is-better); missing or non-string `properties.name`;
  scope.prop not declared as `indexed: true` (otherwise
  `scan_nodes_by_property` silently returns empty, masking
  misconfiguration as "everything was a new entity"); embedding
  empty; embedding dim mismatched against existing index.

  CreateNew dispatch generates a UUIDv4, writes the node, then sets
  the embedding sidecar + inserts into HNSW. **Sequential, not
  batched** — `set_embedding`'s existence check reads the storage
  backend, not the batch buffer (same read-your-own-writes
  constraint documented in `batch.rs` for v0.5.1). All checkable
  failures are pre-flighted; only a pure storage-I/O fault between
  the two writes can tear the pair, which is caller-retry-safe.

  PR: [#5](https://github.com/sligara7/dynograph-foundation/pull/5).
  Aliases (mentioned in the audit memo) deliberately omitted from
  v1 — the underlying resolver doesn't natively support multi-name
  queries; orchestrating that at the HTTP layer would re-implement
  the threshold logic. Extend the resolver crate properly if a real
  workload needs them.

## v0.5.1 — 2026-05-05

First of four primitives identified by the foundation
audit (2026-05-04). Closes the dominant atomicity gap: every
multi-write handler today depends on the in-process
write lock making the sequence atomic, which doesn't survive the
move to HTTP.

### Added

- **`POST /v1/graphs/{id}/batch`** — atomic multi-op transaction.
  Body `{"ops": [...]}` accepts any combination of `create_node`,
  `replace_node`, `delete_node`, `create_edge`, `merge_edge`,
  `delete_edge` ops; field shapes mirror the existing
  single-handler request bodies. Whole batch runs under one
  engine write lock + storage `begin_batch` / `commit_batch`.
  All-or-nothing: any per-op failure discards the batch and
  returns `400` with a structured JSON error
  (`{error, op_index, op_type}`) identifying the failing op —
  the one place the service deviates from the plain-text error
  convention, because batch callers need the index to debug a
  partial rejection. Success returns 200 + per-kind counts +
  `ops_applied`. Soft cap: 1000 ops/batch (audit's heaviest
  known case is ~67). Empty `ops` and `> 1000 ops` are both 400.

  Two storage-layer constraints documented in
  `crates/dynograph-service/src/batch.rs` module doc and locked
  in by hazard tests in `tests/integration.rs`:

  1. *No read-your-own-writes within a batch.* The engine batch
     buffer is write-only — `engine.put()` buffers but
     `engine.get()` reads the backend. Ops whose precondition is
     a `get()` (`merge_edge`, `replace_node`, `delete_*`) see
     pre-batch state. `create_node X` then `replace_node X` in
     one batch fails with "node not found" → rollback.
  2. *Cascade-delete misses in-batch creates.* `delete_node X`
     in the same batch as `create_edge X→Y` leaves an orphaned
     edge — cascade reads pre-batch adjacency.

  Neither blocks any audit-enumerated workload (`integrate_fragment`
  and friends are all-creates or modifications-of-pre-existing).
  Lifting either would require a buffer-aware `engine.get()`;
  out of scope for this release.

  PR: [#3](https://github.com/sligara7/dynograph-foundation/pull/3).
  Consumer side-B acceptance gate: a `mutation.integrate_fragment_atomic`
  test — must stay green when a consumer rewrites its
  fragment-integration handler to call `/batch`.

## v0.5.0 — 2026-05-04

Automation + safety release. Locks down the drift classes that bit
the v0.3.x line, ships build provenance so deployments are
self-identifying, and finishes the public-enum non-exhaustive pass
v0.4.0 started.

### Breaking

- **`Value`, `PropertyType`, `EdgeEndpoint` are now `#[non_exhaustive]`.**
  Same discipline v0.4.0 applied to `DynoError`. External callers
  doing exhaustive `match` on any of these need to add a wildcard arm
  (`_ => …`). Internal patterns within `dynograph-core` are unaffected.
  Future variant additions to these enums no longer require a major
  bump.

### Added

- **`GET /buildinfo`** — JSON build provenance:
  ```json
  {"version": "0.5.0", "git_sha": "abc1234", "git_dirty": false, "uptime_seconds": 142.391}
  ```
  Public endpoint, sibling of `/metrics`/`/health`/`/ready`.
- **`dynograph_build_info` gauge** gains `git_sha` and `git_dirty`
  labels. After v0.5.0, "what code is running on this host?" is one
  curl: `curl /metrics | grep build_info` or `curl /buildinfo`.
- **GHCR publish workflow** (`.github/workflows/release.yml`) runs on
  every `v*` tag push, builds the Docker image, and pushes to
  `ghcr.io/sligara7/dynograph-foundation:${tag}` + `:latest`. README
  and `docs/service.md` now lead with the docker-pull example.
- **CI: doc-version-drift guard** (`scripts/check-doc-versions.sh`).
  Mechanically catches the v0.3.x drift class — Cargo.toml advances
  but README/docs still advertise the previous tag.
- **CI: `cargo-deny`** — security advisories (RUSTSEC), license
  compliance, dup-version detection, unknown-registry detection.
  Config in `deny.toml`.
- **CI: `typos`** — spell-checks source/docs/comments.
- **CI: `msrv-check`** job — builds against rust 1.94 (the declared
  `rust-version`). Catches drift where we accidentally use a feature
  stabilized after MSRV, or a transitive dep silently raises its own.

### Changed

- **CI stable toolchain pinned to 1.95.0** (was `@stable`, the moving
  pointer). The `clippy::unnecessary-sort-by` expansion in 1.95 broke
  CI on the v0.4.0 release branch with no source change. Bump
  `RUST_TOOLCHAIN` env var deliberately when reviewing rustc release
  notes.
- **`SEMVER_BASELINE_REV` bumped v0.3.1 → v0.4.0.** v0.4.0 is now the
  most-recent compilable, properly-versioned release tag.

## v0.4.0 — 2026-05-04

Cleanup release. Bumps minor (in 0.x convention) to honor a breaking
change shipped under v0.3.2's patch tag.

### Breaking

- **`DynoError` is now `#[non_exhaustive]`.** Callers doing exhaustive
  match on `DynoError` must add a wildcard arm (`_ => …`). This formalizes
  the v0.3.2 addition of `DynoError::EdgeValidation` (which itself was a
  breaking change shipped under a patch bump) and prevents future variant
  additions from repeating the same semver mistake.

### Fixed

- **Workspace compiles again.** v0.3.2 added `DynoError::EdgeValidation`
  but did not extend the exhaustive match in
  `dynograph-service::registry::status_for_dyno_error`, so the v0.3.2
  tag did not build. `EdgeValidation` is now mapped to `400 Bad Request`
  alongside the other client-validation variants.
- **`cargo fmt --check` passes again.** Edge-validation code in
  `dynograph-core/src/schema.rs` shipped unformatted in v0.3.2.
  Reformatted.
- **CI `semver-checks` job actually runs.** The previous setup invoked
  `cargo-semver-checks-action@v2` with no baseline configuration, which
  defaults to crates.io — but no foundation crate is published there,
  so the job failed on every PR with `not found in registry`. Replaced
  with a manual `cargo semver-checks --baseline-rev v0.3.1` invocation
  (v0.3.1 = most recent compilable tag).

### Changed

- **Workspace `version` advances `0.3.0` → `0.4.0`.** `Cargo.toml` was
  frozen at `"0.3.0"` across v0.3.0/0.3.1/0.3.2, so binaries from any
  of those tags self-reported `wire_version` as `"0.3.0"` regardless of
  the commit. The `Cargo.toml` version is now the single source of
  truth and tracks each release tag.
- **README + `docs/*` rewritten** for accuracy. Removed references to a
  published GHCR image (no such image exists; consumers build locally
  from this repo's `Dockerfile`).

## v0.3.2 — 2026-04-30

### Fixed

- **`engine::create_edge` now validates the property bag** against the
  edge type's declared properties. Until v0.3.2, edge endpoint validation
  ran but property validation was skipped — required properties could be
  missing, enum values could fall outside the declared set, and the
  handler still returned `200`. Surfaced by a consumer's `SUBTEXT_OF`
  lifecycle probe returning HTTP 200 on `relationship_type="totally_made_up"`.
- New `DynoError::EdgeValidation { edge_type, property, message }`
  variant so edge-property failures name the offending edge instead of
  overloading the node-scoped `Validation`.
- `Schema::validate_edge_properties(edge_type, &mut HashMap)` mirrors
  `validate_node`'s shape — applies declared defaults, enforces
  required-presence, validates each value.

> **Note:** the v0.3.2 tag does not compile in `dynograph-service`
> (missing match arm; fixed in v0.3.3). Library-only consumers of
> `dynograph-core` / `dynograph-storage` are unaffected.

## v0.3.1 — 2026-04-27

### Fixed

- **`PropertyDef`, `NodeTypeDef`, `EdgeTypeDef` are now externally
  constructible** via `..Default::default()` syntax. v0.3.0's
  `#[non_exhaustive]` annotation prevented external struct-literal
  construction; discoverable only on consumer attempt. Drops
  `#[non_exhaustive]` from those three structs and adds `Default`
  derives. `Schema` and `ResolutionConfig` keep `#[non_exhaustive]`
  (they enter via deserialization, not struct literals).
- `PropertyType` gains `#[derive(Default)]` with `String` as the
  default (dominant type in real schemas). `EdgeEndpoint::default()`
  returns `Single("*")`.
- Strictly additive on the wire: serde shape unchanged,
  `content_hash` unchanged.

## v0.3.0 — 2026-04-27

The "embedded → service" release. Foundation gains an HTTP service,
an async client crate, a Docker image, and a sidecar embedding
store. A handful of correctness fixes in storage and the vector
index landed at the end of the cycle (TD-1/2/3).

### Added

- **`dynograph` HTTP service** (`crates/dynograph-service`) with
  multi-graph `GraphRegistry`, RocksDB persistence + restart
  rehydration, node/edge CRUD under `/v1/graphs/{id}`, schema
  split (`POST /v1/graphs`, `PUT /v1/graphs/{id}/schema`) with
  additive-only evolution enforcement, `/ready` + `/metrics`
  (Prometheus), and pluggable auth (`NoAuth` / `BearerJwt`).
- **`dynograph-client`** async Rust HTTP client crate (`reqwest` +
  `rustls-tls`) covering every `/v1/*` route.
- **Sidecar embedding store** + **HNSW similarity search**
  exposed as `POST /v1/graphs/{id}/similar`. Embeddings cascade
  with their owning node on delete.
- **Docker image** built from the in-tree `Dockerfile` / `docker-compose.yml`. No published image; build locally.
- **`docs/migration.md`** — embedded → service playbook.

### Changed (behavioral)

- **`Storage::delete_node` now cascades** to incident edges *and*
  peer-side adjacency entries (TD/C1). Previously it left
  dangling edges that `get_edge` would still resolve and that
  `scan_incoming_edges` on the peer would still return.
- **`update_node_properties` → `replace_node_properties`** and
  **`update_edge_properties` → `merge_edge_properties`** (S1).
  The rename makes REPLACE-vs-MERGE semantics explicit at the
  storage layer. Behavior of the underlying calls is unchanged;
  only the names moved.
- **`validate_node` now takes `&mut`** because it applies schema
  defaults inline (C3). Callers passing `&` need a one-character
  bump.
- **`commit_batch` is atomic across deletes too** (C4). Mixed
  put/delete batches no longer split into two rocksdb writes.
- **Datetime property validation** is now strict per RFC 3339 (C2).

### Fixed (TD-1/2/3, post-tag tightening)

- **HNSW correctness + perf** (TD-1) — vector index returned
  approximate neighbors that occasionally missed exact matches at
  small `M`; bound + heuristics corrected.
- **Storage cache + adjacency + scan-decode + lifecycle** (TD-2) —
  several edge cases around cache invalidation on delete and
  msgpack decode reuse on hot-path scans.
- **Schema + resolver tightening** (TD-3) — surfaces stricter
  errors on malformed schema input rather than silent partial
  acceptance.

### Migration notes

#### `delete_node` cascade

If you have code that **deletes a node and immediately creates a
new node with the same id** (typically as a workaround for
"update properties"), replace the pair with
`replace_node_properties`:

```rust
// Before — silently relied on edges hanging around through the
// delete+recreate gap. With v0.3.0's cascading delete those
// edges are now correctly destroyed, so this pattern drops every
// edge attached to the node.
storage.delete_node(graph_id, "Item", id)?;
storage.create_node(graph_id, "Item", id, new_props)?;

// After — full property replacement, edges + adjacency
// untouched.
storage.replace_node_properties(graph_id, "Item", id, new_props)?;
```

If you actually want partial-update / merge semantics on a node,
do a `get_node` + caller-side merge + `replace_node_properties`
round-trip. (Edges have native `merge_edge_properties`; nodes
deliberately don't, to keep the storage layer's REPLACE-vs-MERGE
distinction explicit.)

#### Method renames

```text
update_node_properties → replace_node_properties
update_edge_properties → merge_edge_properties
```

Mechanical rename; no semantic change.

#### `validate_node` signature

```rust
// Before
schema.validate_node(node_type, &props)?;
// After
schema.validate_node(node_type, &mut props)?;
```

The mutation is schema defaults being applied inline.

## v0.2.1

- `feat(schema)`: optional `description` field on `PropertyDef`.

## v0.2.0

- Initial public-ish baseline of the foundation crates
  (`dynograph-core`, `dynograph-storage`, `dynograph-resolution`,
  `dynograph-vector`). Embedded-only; no service.

## v0.1.0

- Initial workspace skeleton.
