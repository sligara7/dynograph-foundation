# Running `dynograph`

The `dynograph` binary is the deployable form of `dynograph-service`.
One process serves all `/v1/*` REST endpoints over HTTP, with
configurable storage backend, auth provider, and operational probes.

## Quick start (Docker)

```bash
docker compose up                                        # build + run
curl http://localhost:8080/health                        # → "ok"
curl -X POST http://localhost:8080/v1/graphs \
    -H 'content-type: application/json' \
    -d '{"id":"g1","schema":{"name":"demo","version":1,"node_types":{},"edge_types":{}}}'
```

`docker-compose.yml` builds locally + persists `/data` in a named
volume. For published images, use:

```bash
docker run --rm -p 8080:8080 -v dynograph-data:/data \
    -e DYNOGRAPH_STORAGE_ROOT=/data \
    ghcr.io/sligara7/dynograph-foundation:0.3.0
```

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
| `DYNOGRAPH_STORAGE_ROOT` | `[storage].root` (creates the dir if absent) |
| `RUST_LOG` | tracing filter for stderr logs (e.g. `info`, `debug`, `dynograph_service=trace`) |

## Operational probes

| Path | Auth | 200 means | 503 means |
|---|---|---|---|
| `/health` | public | process is up | n/a — never 503 from this route |
| `/ready` | public | startup work complete (rehydrate done on `OnDisk`) | still rehydrating |
| `/metrics` | public | Prometheus text-format metrics | n/a |

`/health` and `/ready` together implement the standard k8s liveness
+ readiness pattern: `/health` failures restart the pod; `/ready`
failures keep the pod out of the load-balancer rotation.

`/metrics` is intentionally public — the assumption is that the
network / ingress layer (k8s `NetworkPolicy`, Caddy IP allowlist)
gates Prometheus scrape access. Same model as `/health`.

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
JWKS / asymmetric algorithms (RS256, ES256) are not in 0.3.0 —
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
