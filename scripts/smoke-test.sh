#!/usr/bin/env bash
# Boot the built `dynograph` binary in its default (in-memory, no-auth)
# mode and probe the operational endpoints. The release.yml image
# publish never actually runs the binary, so this is the gate that
# catches a binary which fails to start — or a route that 500s on a
# fresh process — before it ships.
#
# Run in CI after `cargo build --release --bin dynograph`. Also runnable
# locally against a debug build:
#   cargo build --bin dynograph
#   DYNOGRAPH_BIN=target/debug/dynograph ./scripts/smoke-test.sh
set -euo pipefail

BIN="${DYNOGRAPH_BIN:-target/release/dynograph}"
# An uncommon port so a stray service on the default 8080 can't make the
# probes pass against the wrong server (the post-startup liveness check
# below is the real guard against that; this just reduces the odds).
BIND="${DYNOGRAPH_BIND:-127.0.0.1:18080}"
BASE="http://${BIND}"

if [[ ! -x "$BIN" ]]; then
  echo "smoke-test: binary not found at $BIN (build it first)" >&2
  exit 1
fi

# Start in the background. No DYNOGRAPH_STORAGE_ROOT → in-memory backend,
# ready immediately; default auth is no-auth, so the probes need no token.
DYNOGRAPH_BIND="$BIND" "$BIN" &
PID=$!
# Always stop the server, even if a probe fails. SIGTERM triggers the
# binary's graceful shutdown.
cleanup() {
  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for the listener to accept connections (up to ~20s), failing fast
# if the process dies during startup.
reachable=0
for _ in {1..40}; do
  if curl -fsS "${BASE}/health" >/dev/null 2>&1; then
    reachable=1
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "smoke-test: process exited before becoming reachable" >&2
    exit 1
  fi
  sleep 0.5
done
if [[ "$reachable" -ne 1 ]]; then
  echo "smoke-test: server never became reachable on ${BASE}" >&2
  exit 1
fi

# If our process died but the port still answered, we were probing some
# other service (e.g. the binary failed to bind a port already in use).
# Fail rather than report a false green.
if ! kill -0 "$PID" 2>/dev/null; then
  echo "smoke-test: dynograph process is not running (port in use?) — probes would hit the wrong server" >&2
  exit 1
fi

fail=0
for path in /health /ready /buildinfo /openapi.json; do
  # `|| echo 000` keeps a curl-level failure (a transient connection
  # error) from aborting the whole script under `set -e` — it becomes a
  # non-200 code that fails this probe instead of skipping the rest.
  code=$(curl -s -o /dev/null -w '%{http_code}' "${BASE}${path}" || echo "000")
  if [[ "$code" == "200" ]]; then
    echo "smoke-test: GET ${path} -> ${code}"
  else
    echo "smoke-test: GET ${path} -> ${code} (expected 200)" >&2
    fail=1
  fi
done

# The served contract must be parseable and look like an OpenAPI doc.
if ! curl -fsS "${BASE}/openapi.json" | grep -q '"openapi"'; then
  echo "smoke-test: /openapi.json is missing the \"openapi\" field" >&2
  fail=1
fi

if [[ "$fail" -eq 0 ]]; then
  echo "smoke-test: all probes passed"
fi
exit "$fail"
