# Migrating from embedded → service

This is the playbook for moving an existing consumer that depends
on `dynograph-storage` (or its higher-level wrappers) directly off
the embedded path and onto the new HTTP service. The canonical
migration target is storyflow's `services/dynograph/crates/dynograph-server`,
but the pattern generalizes.

## What changes

| Before | After |
|---|---|
| `cargo` dep on `dynograph-storage = "0.2.x"` | `cargo` dep on `dynograph-client = "0.3"` (Rust consumers), or any HTTP client (other languages) |
| `StorageEngine::new_rocksdb(schema, path)` in-process | `dynograph` binary running as a sidecar/service |
| Direct method calls (`engine.create_node(…)`) | HTTP requests (`POST /v1/graphs/{id}/nodes`) |
| Single-graph state per process | Multi-graph `GraphRegistry`, addressed by `id` in URLs |
| Schema in compiled-in YAML / Rust constants | Schema POST'd to `POST /v1/graphs` at deploy time, replaced via `PUT /v1/graphs/{id}/schema` |
| Embedding lookups via `dynograph-vector::HnswIndex` directly | `POST /v1/graphs/{id}/similar` over HTTP |
| Auth: consumer-managed | foundation handles authentication via `[auth].provider`; consumer layers authorization on top |

## What stays

- The storage layer (`dynograph-storage`) is unchanged. The
  embedded path remains supported for short-lived workers, tests,
  and dev. Foundation v0.3.0 is strictly additive — embedded
  consumers can stay on `dynograph-storage` indefinitely.
- The `Schema` / `Value` / `DynoError` types from `dynograph-core`
  are reused on both sides of the migration.
- `wire_version` + `content_hash` drift detection contract is
  identical to the C-partial substrate storyflow already uses.

## Step-by-step (storyflow-flavored)

The order minimizes downtime and avoids dual-write windows.

### 1. Stand up the service alongside the existing embedded path

Add `dynograph` to docker-compose pointing at a fresh RocksDB root
(or use the in-memory mode for early integration). Initially it
serves no traffic — you're just verifying it boots.

```yaml
# docker-compose.yml addition
services:
  dynograph:
    image: ghcr.io/sligara7/dynograph-foundation:0.3.0
    ports: [ "8080:8080" ]
    volumes: [ "dynograph-data:/data" ]
    environment:
      DYNOGRAPH_STORAGE_ROOT: /data
      RUST_LOG: info
volumes:
  dynograph-data:
```

### 2. POST the schema

The schema YAML the embedded consumer used becomes the body of
`POST /v1/graphs`. Each existing in-process graph becomes a
distinct `id`:

```bash
curl -X POST http://dynograph:8080/v1/graphs \
    -H 'content-type: application/json' \
    -d "$(jq -n --arg id 'storyflow' --slurpfile schema schema.json '{id: $id, schema: $schema[0]}')"
```

Capture the `content_hash` from the response — that's the
fingerprint codegen will key off.

### 3. Migrate one read path at a time

Pick the lowest-risk endpoint first (a `GET /v1/graphs/{id}/schema`
in lieu of an in-process schema accessor is a good first
candidate). Replace the embedded call with an HTTP call via
`dynograph-client`:

```rust
// Before:
let schema = engine.schema().clone();

// After:
let resp = dynograph_client.get_schema("storyflow").await?;
let schema = resp.schema;
```

Verify with the existing test suite. Repeat for each read path.

### 4. Migrate writes

Writes are the riskier transition because of the dual-write
window. The cleanest pattern is **strangler-fig**: keep the
embedded path running and start writing to both. When the HTTP
path passes a parity-check window, drop the embedded writes.

```rust
// Strangler-fig:
engine.create_node("storyflow", "Item", &id, props.clone())?;
client.create_node("storyflow", "Item", &id, &props_json).await?;
// Compare or assert equivalence; then remove the engine call once stable.
```

Foundation does **not** provide a "fork writes" middleware — that's
a consumer-level concern and the trade-off (latency vs safety)
varies per shop.

### 5. Schema-evolution safety

`PUT /v1/graphs/{id}/schema` enforces additive-only evolution at
the wire — removed types/properties, changed types, narrowed edge
endpoints, and `optional → required-without-default` transitions
all return `400` with all violations enumerated. Callers should
treat the body as a structured-enough error to display in admin
tooling.

For migrations that genuinely require breaking changes (e.g.
schema-as-of-snapshot replays where types disappear), the path is:

1. `DELETE /v1/graphs/{id}` (destructive)
2. `POST /v1/graphs` with the new schema
3. Replay data writes against the fresh graph

There's no `force=true` opt-in on `PUT` and no plan to add one;
the cost of breaking a live consumer's codegen output silently is
worse than forcing the explicit destroy + recreate.

### 6. Auth

Foundation v0.3.0 provides authentication (`NoAuth` or
`BearerJwt`) but never authorization. A typical consumer layout:

- `[auth].provider = "bearer_jwt"` on the foundation service.
- Consumers run their own gateway (Caddy, an nginx wrapper, or a
  thin axum proxy) that mints/validates application-level tokens
  and re-mints foundation-level JWTs with whatever `sub` the
  application maps to.
- Consumers enforce per-user access rules (which graphs a user can
  read, which routes are allowed) at the gateway layer, not in
  foundation.

## Drift detection

`wire_version` and `content_hash` on every schema-bearing response
implement the same drift contract storyflow's C-partial substrate
uses:

- `wire_version` mismatch → consumer was built against a
  different foundation crate version. Action: rebuild against
  the matching foundation tag.
- `content_hash` mismatch → schema has changed. Action: re-run
  codegen against the new schema, redeploy.

Consumers should fail-fast at startup on either mismatch — silent
acceptance leads to drift bugs that surface much later in
extraction-time validation.

## Rollback

Keeping the embedded path on a feature flag during the migration
makes rollback trivial. Once the HTTP path is stable in production,
drop the embedded code path; you can always re-add it from git
history if needed.

`docker-compose.yml`'s `dynograph-data` volume is the persistent
state. Backups follow standard RocksDB conventions: snapshot the
volume (or use RocksDB's checkpoint API if exposed in a future
slice). For a hard rollback, stop the service, restore the volume
from backup, restart.
