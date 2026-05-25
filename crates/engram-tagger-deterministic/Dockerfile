# Reference sidecar for engram's HTTP-tagger wire contract.
#
# Multi-stage build: chef + cook the workspace deps once, then build
# just the sidecar binary. Runtime is debian-slim with no native ML
# dependencies — gline-rs uses ort which is statically linked.
#
# The GLiNER ONNX model is NOT baked into the image (~580MB asset).
# Operators mount it from the host or a named volume at /models/gliner:
#
#   docker run --rm \
#     -e EMBEDDER_ENDPOINT=http://host.docker.internal:11434/v1 \
#     -e GLINER_MODEL_DIR=/models/gliner \
#     -e TOPIC_TAXONOMY_PATH=/etc/engram/topic-taxonomy.toml \
#     -v $HOME/models/gliner_small-v2.1:/models/gliner:ro \
#     -v $PWD/crates/engram-tagger-deterministic/topic-taxonomy.toml:/etc/engram/topic-taxonomy.toml:ro \
#     -p 8081:8081 \
#     engram-tagger-deterministic:latest
#
# Then point engram at it: provider = "http", endpoint =
# "http://localhost:8081" in [tagger] / [tagger.http].

# --- Build stage ---------------------------------------------------------

FROM rust:1.95-slim-bookworm AS builder

WORKDIR /workspace

# System deps for compiling the native sub-tree:
# - build-essential: gcc + g++ + make. `cc-rs` (used by several deps under
#   ort-sys / tokenizers) probes for a C++ compiler by literal name; the
#   slim base ships `cc` but not `c++`, so without g++ the build fails
#   with `ToolNotFound: failed to find tool "c++"`.
# - clang + libclang-dev: `bindgen` (pulled in by tokenizers transitively)
#   needs libclang at compile time. Without these the build dies later
#   with "Unable to find libclang".
# - pkg-config + libssl-dev: reqwest's OpenSSL fallback path (rustls is
#   the default but the build script probes for OpenSSL anyway).
# - cmake: required by a handful of native build scripts in the ort tree.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    libclang-dev \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy the whole workspace — sidecar depends on engram-core, engram-embed,
# and engram-tagger-protocol so we need the full layout. .dockerignore at
# the repo root keeps target/ and .git/ out of the build context.
COPY . .

RUN cargo build --release -p engram-tagger-deterministic

# --- Runtime stage -------------------------------------------------------

FROM debian:bookworm-slim AS runtime

# Runtime deps:
# - ca-certificates: TLS to remote embedder/sidecar endpoints.
# - curl: required by the HEALTHCHECK directive at the bottom of this
#   Dockerfile (the slim base image doesn't ship curl). Without it the
#   container reports "unhealthy" even when /health returns 200 from the
#   host, because Docker can't execute the healthcheck command inside.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 sidecar

USER sidecar
WORKDIR /home/sidecar

COPY --from=builder /workspace/target/release/engram-tagger-deterministic /usr/local/bin/

# Sensible defaults — operators override via -e.
ENV BIND_ADDR=0.0.0.0:8081 \
    EMBEDDER_ENDPOINT=http://host.docker.internal:11434/v1 \
    EMBEDDER_MODEL=bge-m3 \
    GLINER_MODEL_DIR=/models/gliner \
    TOPIC_TAXONOMY_PATH=/etc/engram/topic-taxonomy.toml \
    KIND_THRESHOLD=0.55 \
    TOPIC_THRESHOLD=0.45 \
    MODEL_ID=deterministic/gliner-small-v2.1+regex+bge-m3 \
    MODEL_VERSION=1 \
    RUST_LOG=info

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8081/health || exit 1

ENTRYPOINT ["/usr/local/bin/engram-tagger-deterministic"]
