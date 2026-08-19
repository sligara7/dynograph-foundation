# Endpoint reference

The complete published surface of `dynograph-foundation` (v0.13.0): **80 routes**
in the OpenAPI contract (the operational probes + everything under `/v1`), plus
the meta `GET /openapi.json` that serves the contract itself. This is the
human-readable catalog; the machine-readable source of truth is
[`docs/openapi.json`](openapi.json), and [`docs/api.md`](api.md) carries worked
request/response examples for the core CRUD + primitive routes.

**Conventions.** All paths are under `/v1` unless noted. `{id}` is a graph id.
Bodies and responses are JSON. Auth is a bearer JWT when the service is
configured for it (operational probes are public). Two surfaces are behind
optional build features and return **`501`** when the feature is off (the
routes and OpenAPI contract always exist):

- **`graph`** — every `algo/*` endpoint.
- **`fulltext`** — `search:text`, `search:reindex`, and the keyword leg of
  `search:hybrid`.

The published image enables both.

---

## Operational (public)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe. |
| `GET` | `/ready` | Readiness probe (distinct from `/health`). |
| `GET` | `/metrics` | Prometheus text-format scrape. |
| `GET` | `/buildinfo` | Build provenance: `version`, `git_sha`, `git_dirty`. |
| `GET` | `/openapi.json` | The live OpenAPI 3 contract. |

## Graph lifecycle & schema

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/graphs` | Create a graph from a **schema** (see [Schema-driven configuration](../README.md#schema-driven-configuration)). |
| `GET` | `/graphs` | List graph ids. |
| `GET` | `/graphs/{id}` | Graph metadata (id, wire version, content hash). |
| `DELETE` | `/graphs/{id}` | Delete a graph and all its data. |
| `GET` | `/graphs/{id}/schema` | Full schema view (the shape consumer codegen reads). |
| `PUT` | `/graphs/{id}/schema` | Replace a schema (additive-only compat rules, atomic). |

## Nodes

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/graphs/{id}/nodes` | Create a typed node (schema-validated). |
| `GET` | `/graphs/{id}/nodes?type=X[&prop=Y&value=Z]` | List nodes of a type, optional single-property filter. |
| `GET` | `/graphs/{id}/nodes/{type}/{node_id}` | Fetch one node. |
| `PUT` | `/graphs/{id}/nodes/{type}/{node_id}` | Full replacement of the node's properties. |
| `DELETE` | `/graphs/{id}/nodes/{type}/{node_id}` | Delete a node (cascades to its edges). |

## Edges

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/graphs/{id}/edges` | Create a typed edge between two nodes. |
| `GET` | `/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}` | Fetch one edge. |
| `PATCH` | `/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}` | Partial-update the edge's properties. |
| `DELETE` | `/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}` | Delete one edge. |

## Embeddings (vector sidecar)

Foundation does not embed — the caller supplies vectors. Each `node_type` has
its own HNSW index.

| Method | Path | Description |
|--------|------|-------------|
| `PUT` | `/graphs/{id}/nodes/{type}/{node_id}/embedding` | Set an embedding; updates the per-type HNSW index in lockstep. |
| `GET` | `/graphs/{id}/nodes/{type}/{node_id}/embedding` | Fetch a node's embedding. |
| `DELETE` | `/graphs/{id}/nodes/{type}/{node_id}/embedding` | Remove an embedding (and its index entry). |

## Search

| Method | Path | Description | Feature |
|--------|------|-------------|---------|
| `POST` | `/graphs/{id}/similar` | HNSW vector similarity over a type's index. | |
| `POST` | `/graphs/{id}/search:text` | BM25 keyword search over the full-text index. | `fulltext` |
| `POST` | `/graphs/{id}/search:hybrid` | Reciprocal-Rank-Fusion of the vector + keyword legs (optional `where` prefilter). | keyword leg: `fulltext` |
| `POST` | `/graphs/{id}/search:reindex` | Rebuild the full-text index from the node store. | `fulltext` |

## Primitives (composed multi-op / query helpers)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/graphs/{id}/batch` | Atomic multi-op transaction; optional `dry_run` validate-before-apply report. |
| `POST` | `/graphs/{id}/resolve-or-create` | Fuzzy/vector entity resolution with create-on-miss. |
| `POST` | `/graphs/{id}/nodes:scan` | Predicate-filtered scan over one node type (AND-combined `where` clauses). |
| `POST` | `/graphs/{id}/nodes:exists` | Batch `(type, name)` existence check. |
| `POST` | `/graphs/{id}/edges:collect` | Fan-out edge collection across a typed source set. |
| `POST` | `/graphs/{id}/edges:adjacent` | Single-node 1-hop adjacency (in/out/both). |
| `POST` | `/graphs/{id}/traverse` | Typed BFS traversal from a start node. |
| `POST` | `/graphs/{id}/edges/{edge_type}/{from_id}/{to_id}/welford_update` | Atomic Welford/EMA update of an edge's score family. |

## Graph algorithms — `algo/*` (feature: `graph`)

Each request supplies the domain-specific parts — a subgraph **`scope`** (node
types, edge types, and an optional **`where`** property predicate), an
edge-weight projection, and a direction — and gets back a generic result.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/graphs/{id}/algo/components` | Weakly-connected components. |
| `POST` | `/graphs/{id}/algo/scc` | Strongly-connected components (Tarjan). |
| `POST` | `/graphs/{id}/algo/degree` | Degree centrality. |
| `POST` | `/graphs/{id}/algo/pagerank` | PageRank (weights = strength). |
| `POST` | `/graphs/{id}/algo/personalized_pagerank` | Random walk with restart to a seed set. |
| `POST` | `/graphs/{id}/algo/eigenvector` | Eigenvector centrality (undirected). |
| `POST` | `/graphs/{id}/algo/closeness` | Closeness centrality (weights = path cost). |
| `POST` | `/graphs/{id}/algo/betweenness` | Betweenness centrality (weights = path cost). |
| `POST` | `/graphs/{id}/algo/communities` | Leiden community detection + modularity. |
| `POST` | `/graphs/{id}/algo/clustering` | Local clustering coefficients + global transitivity. |
| `POST` | `/graphs/{id}/algo/cuts` | Articulation points and bridges. |
| `POST` | `/graphs/{id}/algo/cycles` | Directed cycle detection (+ a witness cycle). |
| `POST` | `/graphs/{id}/algo/toposort` | Topological order (or a not-acyclic flag). |
| `POST` | `/graphs/{id}/algo/shortest_path` | Single-pair shortest path. |
| `POST` | `/graphs/{id}/algo/max_flow` | Maximum flow / minimum s-t cut. |
| `POST` | `/graphs/{id}/algo/link_prediction` | Neighborhood-overlap link scoring (common-neighbors / Jaccard / Adamic-Adar). |

## Utility math — `util/*` (stateless, no graph)

Canonical pure-math implementations so every consumer shares one numerically
identical result instead of re-deriving cosine / Pearson / Nash in each language.

**Vector ops** (binary unless noted; optional `f32`/`f64` precision):

| Path | Returns |
|------|---------|
| `util/cosine_similarity`, `util/dot_product` | scalar |
| `util/euclidean_distance`, `util/squared_euclidean_distance`, `util/manhattan_distance` | scalar |
| `util/l2_norm` *(unary)* | scalar |
| `util/add`, `util/subtract`, `util/hadamard`, `util/hadamard_division` | vector |
| `util/scale`, `util/negate`, `util/elementwise_power`, `util/l2_normalize` *(unary)* | vector |
| `util/centroid`, `util/softmax` *(vector set)* | vector |
| `util/pairwise_cosine`, `util/pairwise_distance` *(vector set)* | N×N matrix |

**Statistics & correlation** (f64): `util/mean`, `util/variance`, `util/std_dev`,
`util/median`, `util/percentile`, `util/pearson_correlation`,
`util/spearman_correlation`, `util/linear_regression_slope`.

**String similarity**: `util/jaro_winkler`, `util/token_sort_ratio` (0–100).

**Game theory**: `util/game/analyze` — normal-form analysis (dominant
strategies, pure & 2×2-mixed Nash, Pareto optimality, `nash_is_pareto_suboptimal`).

**Clustering**: `util/dbscan` — DBSCAN density-based clustering over a
caller-supplied N×N distance matrix (`eps`, `min_points`); returns a label per
point (`-1` = noise, `1..` = cluster id) and the cluster count. Lives under
`util/` (matrix in, labels out) rather than `algo/*` because it clusters points
by their pairwise distances, not the stored graph's topology — distinct from
the graph suite's Leiden (community-on-edges).
