# syntax=docker/dockerfile:1.7

# ---------- builder ----------
# Pinned to the workspace MSRV (`rust-version` in Cargo.toml). Bump
# both in lockstep.
FROM rust:1.94-slim-bookworm AS builder

# rocksdb vendors RocksDB and builds it from source, so the builder
# needs a C++ toolchain plus clang for bindgen. cmake + pkg-config
# are required by the rocksdb-sys build script.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        cmake \
        libclang-dev \
        pkg-config \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .
# BuildKit cache mounts: the cargo registry + target dir persist
# across iterative rebuilds on the same host, so a source-only
# change skips the ~12-minute RocksDB recompile. The mount goes
# away when the layer ends, so we copy the binary out to a stable
# path the runtime stage can read. CI/cross-host rebuilds need
# `cache-from`/`cache-to` (deferred to slice 13's publish workflow).
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --bin dynograph \
 && cp /build/target/release/dynograph /usr/local/bin/dynograph

# ---------- runtime ----------
FROM debian:bookworm-slim

# debian-slim already ships libstdc++6 + libgcc-s1, which the
# vendored RocksDB links against dynamically. ca-certificates is
# included so any future BearerJwt JWKS path (deferred from slice 9)
# can verify TLS without rebuilding the runtime image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/dynograph /usr/local/bin/dynograph

# Mount this volume for persistent storage. Pair with
# `DYNOGRAPH_STORAGE_ROOT=/data` (or `[storage].root = "/data"` in a
# mounted dynograph.toml) to flip the registry into on-disk mode.
VOLUME ["/data"]

# Default to all-interfaces inside the container — the host's port
# mapping is the access boundary. Override via DYNOGRAPH_BIND or a
# mounted dynograph.toml.
ENV DYNOGRAPH_BIND=0.0.0.0:8080
EXPOSE 8080

LABEL org.opencontainers.image.source="https://github.com/sligara7/dynograph-foundation"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.description="dynograph-service — schema-driven graph HTTP service over RocksDB + HNSW"

ENTRYPOINT ["/usr/local/bin/dynograph"]
