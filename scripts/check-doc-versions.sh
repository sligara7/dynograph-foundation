#!/usr/bin/env bash
# Asserts that the hardcoded facts in README.md and docs/*.md still match
# the code: version strings against Cargo.toml's workspace `version`, and
# route counts against docs/openapi.json. Catches the class of drift where
# a release bump or a route change goes in but the docs still advertise the
# previous state.
#
# Targeted patterns (NOT a generic "any X.Y.Z" scan, which would
# false-positive on historical references in CHANGELOG-style prose):
#
#   - `tag = "vX.Y.Z"`                      (git-dep tag in Rust toml blocks)
#   - `"wire_version":  "X.Y.Z"`            (JSON example bodies)
#   - `latest tag (`vX.Y.Z`)`               (prose phrasing in README)
#   - `version (e.g. "X.Y.Z")`              (api.md's wire-stability section)
#   - `tagged per release (`:X.Y.Z`)`       (image-tag prose in README — the
#                                            v0.9.1→v0.9.3 slip this class missed)
#   - `published surface of ... (vX.Y.Z)`   (endpoints.md's opening line — the
#                                            v0.9.1-at-0.11.0 slip this class missed)
#
# Route counts (source of truth: docs/openapi.json, which is itself pinned
# to the generated spec by a contract-gate test in dynograph-service, so it
# cannot drift from the handlers):
#
#   - `**N routes**`                        (endpoints.md — every operation)
#   - `all N `/v1` routes`                  (README — the /v1 operations only)
#
# CHANGELOG.md is intentionally excluded — historical version
# references there are correct.
#
# Silence is failure. Every pattern below is anchored on a literal phrase
# ("tag = ", "wire_version", "latest tag", "ghcr.io/..."), so rewording a
# doc would otherwise disable the corresponding check with no signal — a
# guard that finds nothing looks exactly like a guard that has stopped
# looking. Each pattern is therefore required to match at least once
# repo-wide, and a pattern that matches nothing fails the run. If you
# deliberately remove the last occurrence of a documented fact, delete its
# pattern here in the same change.
#
# Local usage: `./scripts/check-doc-versions.sh`. CI runs the same.

set -euo pipefail

CARGO_VERSION=$(awk -F'"' '/^version = / {print $2; exit}' Cargo.toml)

if [ -z "$CARGO_VERSION" ]; then
    echo "FAIL: could not extract workspace version from Cargo.toml" >&2
    exit 1
fi

OPENAPI="docs/openapi.json"

if [ ! -f "$OPENAPI" ]; then
    echo "FAIL: $OPENAPI not found; cannot check documented route counts" >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "FAIL: python3 not found; cannot read route counts from $OPENAPI" >&2
    exit 1
fi

# Total operations, then /v1-only operations, one per line.
read -r TOTAL_ROUTES V1_ROUTES < <(python3 - "$OPENAPI" <<'PY'
import json, sys

with open(sys.argv[1]) as fh:
    spec = json.load(fh)

METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}
ops = [(path, m) for path, item in spec["paths"].items() for m in item if m in METHODS]
print(len(ops), sum(1 for path, _ in ops if path.startswith("/v1")))
PY
)

echo "Workspace version: $CARGO_VERSION"
echo "OpenAPI routes:    $TOTAL_ROUTES total, $V1_ROUTES under /v1"
echo

EXIT=0

# Every pattern's repo-wide match count, keyed by label. A label that ends
# on 0 means the doc phrase it was anchored on is gone.
declare -A HITS=()

check_pattern() {
    local file=$1
    local pattern=$2
    local label=$3

    HITS["$label"]=${HITS["$label"]:-0}

    while IFS=: read -r linenum line; do
        HITS["$label"]=$(( HITS["$label"] + 1 ))
        # Extract first X.Y.Z (with optional v prefix) on the matched line.
        local version
        version=$(printf '%s' "$line" | grep -oE 'v?[0-9]+\.[0-9]+\.[0-9]+' | head -1 | sed 's/^v//')
        if [ -n "$version" ] && [ "$version" != "$CARGO_VERSION" ]; then
            printf 'DRIFT: %s:%s (%s): found %s, expected %s\n' \
                "$file" "$linenum" "$label" "$version" "$CARGO_VERSION"
            printf '       > %s\n' "$line"
            EXIT=1
        fi
    done < <(grep -nE "$pattern" "$file" 2>/dev/null || true)
}

# Like check_pattern, but the pattern must bracket the number tightly
# enough that the matched substring contains exactly the count to compare
# (an X.Y.Z on the same line would otherwise be read as the number).
check_count() {
    local file=$1
    local pattern=$2
    local label=$3
    local expected=$4

    HITS["$label"]=${HITS["$label"]:-0}

    while IFS=: read -r linenum line; do
        HITS["$label"]=$(( HITS["$label"] + 1 ))
        local found
        found=$(printf '%s' "$line" | grep -oE "$pattern" | grep -oE '[0-9]+' | head -1)
        if [ -n "$found" ] && [ "$found" != "$expected" ]; then
            printf 'DRIFT: %s:%s (%s): found %s, expected %s\n' \
                "$file" "$linenum" "$label" "$found" "$expected"
            printf '       > %s\n' "$line"
            EXIT=1
        fi
    done < <(grep -nE "$pattern" "$file" 2>/dev/null || true)
}

FILES=("README.md")
for d in docs/*.md; do
    [ -f "$d" ] && FILES+=("$d")
done

for f in "${FILES[@]}"; do
    check_pattern "$f" 'tag = "v[0-9]+\.[0-9]+\.[0-9]+"'                 "git-dep tag"
    check_pattern "$f" '"wire_version":[[:space:]]*"[0-9]+\.[0-9]+\.[0-9]+"' "wire_version JSON"
    check_pattern "$f" 'latest tag.*\(`v[0-9]+\.[0-9]+\.[0-9]+`\)'       "latest-tag prose"
    check_pattern "$f" 'version \(e\.g\.[[:space:]]*`"[0-9]+\.[0-9]+\.[0-9]+"`\)' "wire-version stability example"
    check_pattern "$f" 'ghcr\.io/sligara7/dynograph-foundation:[0-9]+\.[0-9]+\.[0-9]+' "docker image tag"
    check_pattern "$f" 'tagged per release \(`:[0-9]+\.[0-9]+\.[0-9]+`\)' "image-tag prose"
    check_pattern "$f" 'published surface of .*\(v[0-9]+\.[0-9]+\.[0-9]+\)' "published-surface version"

    check_count "$f" '\*\*[0-9]+ routes\*\*'       "total route count" "$TOTAL_ROUTES"
    check_count "$f" 'all [0-9]+ `/v1` routes'     "/v1 route count"   "$V1_ROUTES"
done

# A pattern that matches nothing is not a pass — it is a check that has
# gone blind, which is the failure mode this guard exists to prevent.
for label in "${!HITS[@]}"; do
    if [ "${HITS[$label]}" -eq 0 ]; then
        printf 'BLIND: (%s): matched nothing in README.md or docs/*.md — the doc\n' "$label"
        printf '       phrase this check is anchored on was reworded or removed, so the\n'
        printf '       check silently stopped running. Re-anchor the pattern in\n'
        printf '       scripts/check-doc-versions.sh, or drop it if the fact is gone.\n'
        EXIT=1
    fi
done

if [ "$EXIT" -eq 0 ]; then
    echo "OK: doc version refs match Cargo.toml ($CARGO_VERSION), route counts match $OPENAPI,"
    echo "    and all ${#HITS[@]} checks matched live doc text."
fi

exit $EXIT
