# syntax=docker/dockerfile:1.7

# ---------- builder ----------
# Pinned to the rust toolchain the workspace tests run against (1.94
# stabilizes the `is_multiple_of` slice 8a uses; older nightlies will
# fail to compile dynograph-storage). Bump alongside the workspace's
# minimum rustc.
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
RUN cargo build --release --bin dynograph

# ---------- runtime ----------
FROM debian:bookworm-slim

# debian-slim already ships libstdc++6 + libgcc-s1, which the
# vendored RocksDB links against dynamically. ca-certificates is
# included so any future BearerJwt JWKS path (deferred from slice 9)
# can verify TLS without rebuilding the runtime image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/dynograph /usr/local/bin/dynograph

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
