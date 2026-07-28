#!/usr/bin/env bash
# Asserts that the public API of every workspace crate is compatible with the
# most recent release tag, so a breaking change cannot reach a consumer
# unannounced. Consumers pin by git tag and link these crates directly, so a
# break lands in THEIR build rather than in a registry's version resolution.
#
# Two things here are not obvious, and both are why this file exists rather
# than a one-line `cargo semver-checks --workspace` in the CI yaml.
#
# 1. THE CRATE LIST MUST BE EXPLICIT. `--workspace` checks NOTHING here.
#    cargo-semver-checks skips crates with `publish = false`, and this
#    workspace sets exactly that, deliberately (see the Cargo.toml comment:
#    the crates are consumed by git tag and are not crates.io-ready). So
#    `--workspace` printed "Skipping <crate>" ten times and exited 0 — a
#    green check that had examined nothing, from the interaction of two
#    individually-correct decisions. Naming packages with `-p` overrides the
#    skip. The list is derived from `cargo metadata`, never hand-kept, so a
#    new crate is covered the day it is added.
#
# 2. THE BASELINE IS DERIVED. It used to be a hand-maintained literal pinned
#    at v0.4.0 while the workspace was at 0.11.0 — seven minors, during which
#    Cargo's 0.x rules (the minor is the breaking position below 1.0) let an
#    old baseline permit every break it was meant to catch.
#
# The theme both share: a check that has stopped looking reports exactly what
# a check that looked and found nothing reports. So this script refuses to
# pass silently — an empty crate list, a missing tag, or a "Skipping" line in
# the output all FAIL rather than sail through.
#
# Local usage: `./scripts/check-semver.sh`. CI runs the same.

set -euo pipefail

if ! command -v cargo-semver-checks >/dev/null 2>&1; then
    echo "FAIL: cargo-semver-checks not installed (cargo install cargo-semver-checks --locked)" >&2
    exit 1
fi

BASELINE="${SEMVER_BASELINE_REV:-$(git describe --tags --abbrev=0 --match 'v[0-9]*' 2>/dev/null || true)}"
if [ -z "$BASELINE" ]; then
    echo "FAIL: no v* tag found, so there is no baseline to compare against." >&2
    echo "      Refusing to report a pass from a comparison that did not happen." >&2
    exit 1
fi

# Workspace members, from cargo rather than from memory.
mapfile -t PKGS < <(cargo metadata --format-version 1 --no-deps 2>/dev/null \
    | python3 -c 'import json,sys; print("\n".join(sorted(p["name"] for p in json.load(sys.stdin)["packages"])))')

if [ "${#PKGS[@]}" -eq 0 ]; then
    echo "FAIL: no workspace packages found; there is nothing to check and that is not a pass." >&2
    exit 1
fi

ARGS=()
for p in "${PKGS[@]}"; do
    ARGS+=(-p "$p")
done

echo "Baseline: $BASELINE"
echo "Packages (${#PKGS[@]}): ${PKGS[*]}"
echo

OUT=$(mktemp)
trap 'rm -f "$OUT"' EXIT

set +e
cargo semver-checks "${ARGS[@]}" --baseline-rev "$BASELINE" 2>&1 | tee "$OUT"
RESULT=${PIPESTATUS[0]}
set -e

# A skipped crate is the failure this script was written to catch: it looks
# identical to a crate that passed. Never let it through, whatever the exit
# code said.
if grep -qE '^\s*Skipping ' "$OUT"; then
    echo >&2
    echo "FAIL: cargo-semver-checks SKIPPED at least one crate, so its public API was" >&2
    echo "      never compared. This is how the check silently did nothing for seven" >&2
    echo "      minors. A skipped crate is not a passing crate." >&2
    grep -E '^\s*Skipping ' "$OUT" >&2
    exit 1
fi

exit "$RESULT"
