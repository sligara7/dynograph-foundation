# Running `dynograph`

The `dynograph` binary is the deployable form of `dynograph-service`.
One process serves all `/v1/*` REST endpoints over HTTP, with
configurable storage backend, auth provider, and operational probes.

## Quick start

Published Docker image:

```bash
docker run --rm -p 8080:8080 -v dynograph-data:/data \
    -e DYNOGRAPH_STORAGE_ROOT=/data \
    ghcr.io/sligara7/dynograph-foundation:0.14.0
```

Or build from this repo:

```bash
docker compose up                                        # build + run
curl http://localhost:8080/health                        # → "ok"
curl -X POST http://localhost:8080/v1/graphs \
    -H 'content-type: application/json' \
    -d '{"id":"g1","schema":{"name":"demo","version":1,"node_types":{},"edge_types":{}}}'
```

Or native:

```bash
cargo run --release --bin dynograph -- --config dynograph.example.toml
```

(Or omit `--config` to run with built-in defaults: in-memory storage,
`127.0.0.1:8080`, no auth.)

## Configuration

`dynograph` reads `--config <path>` (TOML) at startup, then applies
env-var overrides on top. Both inputs are optional — defaults
ship as in-memory storage on `127.0.0.1:8080` with `noauth`.

See [`dynograph.example.toml`](../dynograph.example.toml) for the
full annotated shape. The three sections:

### `[server]`

| Key | Default | Description |
|---|---|---|
| `bind` | `127.0.0.1:8080` | Listen address. `0.0.0.0:8080` for non-localhost. The Docker image overrides via `ENV DYNOGRAPH_BIND`. |
| `uds_path` | _(absent)_ | Optional Unix-socket path served *in addition to* `bind` — the full `/v1` API on a faster same-host transport for co-located consumers. Absent means TCP only. A stale socket from a prior run is reclaimed on start; an existing non-socket file at the path is a startup error. |

### `[storage]`

| Key | Default | Description |
|---|---|---|
| `root` | _(absent)_ | RocksDB root. When set, persistent mode; absent means in-memory. |

In-memory mode starts ready immediately; on-disk mode rehydrates
existing graphs from `{root}/{id}/` and flips `/ready` to 200 only
after rehydration finishes (see [`/ready`](#operational-probes)
below).

### `[auth]`

| Key | Default | Description |
|---|---|---|
| `provider` | `"noauth"` | `"noauth"` accepts every request as `"anonymous"`. `"bearer_jwt"` requires a valid HS256 JWT. |
| `secret` | — | Literal symmetric secret (dev convenience). |
| `secret_env` | — | Env var holding the secret (production-recommended — keeps the secret out of TOML). |
| `issuer` | — | When set, tokens must carry a matching `iss` claim. |
| `audience` | — | When set, tokens must carry a matching `aud` claim. |

Exactly one of `secret` or `secret_env` must be set when
`provider = "bearer_jwt"`. Both or neither is a startup error.

### Env-var overrides

| Variable | Overrides |
|---|---|
| `DYNOGRAPH_BIND` | `[server].bind` |
| `DYNOGRAPH_UDS_PATH` | `[server].uds_path` |
| `DYNOGRAPH_STORAGE_ROOT` | `[storage].root` (creates the dir if absent) |
| `RUST_LOG` | tracing filter for stderr logs (e.g. `info`, `debug`, `dynograph_service=trace`) |

## Operational probes

| Path | Auth | 200 means | 503 means |
|---|---|---|---|
| `/health` | public | process is up | n/a — never 503 from this route |
| `/ready` | public | startup work complete (rehydrate done on `OnDisk`) | still rehydrating |
| `/metrics` | public | Prometheus text-format metrics | n/a |
| `/buildinfo` | public | JSON build provenance | n/a |

`/health` and `/ready` together implement the standard k8s liveness
+ readiness pattern: `/health` failures restart the pod; `/ready`
failures keep the pod out of the load-balancer rotation.

`/metrics` and `/buildinfo` are intentionally public — the assumption
is that the network / ingress layer (k8s `NetworkPolicy`, Caddy IP
allowlist) gates scrape access when needed. Same model as `/health`.

### `GET /buildinfo`

```json
{
  "version": "0.5.0",
  "git_sha": "abc1234",
  "git_dirty": false,
  "uptime_seconds": 142.391
}
```

`version` is `CARGO_PKG_VERSION` at build time. `git_sha` is the
short HEAD sha at build time, or `"unknown"` when the binary was
built outside a git checkout (e.g. from a release tarball).
`git_dirty` is `true` if the working tree had uncommitted changes
when the binary was built — useful for catching "what did I deploy?"
mistakes after a hot patch.

The same triple is also surfaced as labels on the
`dynograph_build_info` Prometheus gauge:

```
dynograph_build_info{version="0.5.0",git_sha="abc1234",git_dirty="false"} 1
```

## BearerJwt

```toml
[auth]
provider = "bearer_jwt"
secret_env = "DYNOGRAPH_JWT_SECRET"
issuer = "https://auth.example.com"
audience = "dynograph"
```

```bash
DYNOGRAPH_JWT_SECRET=$(openssl rand -hex 32) dynograph --config dynograph.toml
```

Tokens are HS256, must carry `sub` (becomes the request's
`Identity`) and `exp` (mandatory; `jsonwebtoken`'s default 60s
clock-skew leeway applies). `iss` / `aud` are enforced when set.
JWKS / asymmetric algorithms (RS256, ES256) are not implemented —
they need an async key fetcher and an HTTP-client dep that no
consumer has asked for yet. When one does, layer a separate
`AuthProvider` impl.

```bash
# mint a dev token (jq + jose ↩ or any JWT lib)
TOKEN=$(jose-util jwt sign -k dev.jwk -p '{"sub":"alice","exp":'$(($(date +%s) + 3600))'}')
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/v1/graphs
```

`/health`, `/ready`, `/metrics` stay public even under
`bearer_jwt` — k8s probes don't carry tokens.

## Storage modes

**In-memory** (`[storage].root` absent): every graph lives in a
HashMap-backed engine. Restart loses everything. Useful for tests,
dev, and short-lived workers. Starts `/ready = 200` immediately.

**On-disk** (`[storage].root = "/data"`): every graph is a
`{root}/{id}/` dir with `schema.json` (canonical schema for
rehydration) + `db/` (RocksDB column-family store). On startup,
`rehydrate()` walks `root` and registers each valid graph dir
before flipping `/ready` to 200. Corrupt schema or a RocksDB
open failure aborts startup loud — fail-loud policy.

Schema replacement (`PUT /v1/graphs/{id}/schema`) writes the new
`schema.json` before the in-memory swap. A disk-write failure
leaves the in-memory state untouched; no skew across a process
restart.

## Graceful shutdown

`SIGINT` and `SIGTERM` trigger graceful shutdown. axum drains
in-flight requests, then exits. RocksDB flushes its WAL on drop.
The Docker image's default entrypoint is `tini`-free; if you need
PID-1-correct signal handling, run with `--init` or wrap in
`tini`.

## Logs

Defaults to JSON-able structured logs on stderr at `INFO`. Set
`RUST_LOG` to tune (e.g. `RUST_LOG=info,dynograph_service=debug`).
Each request is logged at `INFO` with method + path + status +
latency.

## Domain-neutral math: call DF, don't reimplement

`dynograph` is the single home for **all** domain-neutral math its consumers
need — graph topology, per-vector algebra, and descriptive statistics. The
policy is: **call these endpoints instead of reimplementing the math locally**
(no hand-rolled numpy adjacency SVD, no duplicated cosine loops). Keeping one
audited implementation behind a stable wire contract is the whole point.

These families, all under the published OpenAPI spec (`GET /openapi.json`):

- **Vector & stats** — `POST /v1/util/*`: per-pair algebra (`cosine_similarity`,
  `euclidean_distance`, `add`, `scale`, `l2_normalize`, …), reductions
  (`centroid`, `mean`, `variance`, `percentile`, `softmax`, `pearson_correlation`,
  …), and **batch matrix** forms — `pairwise_cosine` and `pairwise_distance`
  take N vectors and return the full N×N matrix in one call, so a consumer
  ranking N entities does **one** request instead of N² per-pair round-trips.
  Stateless; no graph required. Vectors capped at `MAX_VECTOR_LEN` (100k) and
  pairwise inputs at 1000 vectors.

- **Graph topology** — `POST /v1/graphs/{id}/algo/*`: components, strongly-
  connected components, degree/PageRank/eigenvector/closeness/betweenness
  centrality, personalized PageRank, articulation points & bridges, cycle
  detection, shortest path, link prediction, clustering/transitivity,
  topological sort, and max-flow/min-cut. Each request supplies the
  domain-specific parts — a subgraph `scope` (node/edge-type filter), an
  edge-weight projection, and a direction — and gets back a generic result
  (scores, partitions, paths). Behind the optional `graph` build feature; the
  published image enables it. Without the feature the routes return `501`.

- **Stateless analysis** — also `POST /v1/util/*`, but operating on a
  caller-supplied matrix rather than vectors or a stored graph: `game/analyze`
  (normal-form game theory — dominant strategies, pure & 2×2-mixed Nash, Pareto
  optimality, `nash_is_pareto_suboptimal`) and `dbscan` (density-based
  clustering of a precomputed N×N distance matrix → a label per point, `-1` =
  noise). These live under `util/` (not `algo/`) and need no `graph` feature —
  they take the matrix in the request, not the graph's topology. `dbscan` is
  distinct from the graph suite's Leiden: density-on-points vs community-on-edges.

The smoke test (`scripts/smoke-test.sh`, run in CI after the release build)
probes `/v1/util/pairwise_cosine` on the booted binary, so a build that fails to
expose the math surface fails the pipeline rather than shipping.

## Resource shape

The default RocksDB tuning (per `cf_options` in
`dynograph-storage::engine`) targets correctness, not throughput.
For high-volume workloads, RocksDB exposes block-cache size,
write-buffer size, and bloom-filter bits as build-time options on
each column family — currently hard-coded; tuneable via env vars
is a future-slice concern.

Per-graph HNSW indexes live in process memory. Memory cost per
index = `node_count × (dim × 4 + neighbor_count × 8)`. For 1M
nodes at dim=768, plan for ~3.5 GB per indexed type. Multi-graph
deployments multiply this.
