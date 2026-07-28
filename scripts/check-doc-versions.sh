#!/usr/bin/env bash
# Asserts that the hardcoded facts in README.md and docs/*.md still match the
# code: version strings against Cargo.toml's workspace `version`, and route
# counts against docs/openapi.json. Catches the class of drift where a release
# bump or a route change goes in but the docs still advertise the previous
# state.
#
# Checks are TARGETED, not a generic "any X.Y.Z" scan, which would
# false-positive on historical references in CHANGELOG-style prose. Each one
# is anchored on a literal phrase and brackets the value it compares, so the
# number checked is the first match inside the MATCHED SUBSTRING rather than
# anywhere on the line (a line often carries both a version and a count).
#
# CHANGELOG.md is intentionally excluded — historical version references
# there are correct.
#
# SILENCE IS FAILURE. Because each check is anchored on a literal phrase,
# rewording a doc would otherwise disable it with no signal — and a check that
# has stopped looking reports exactly what a check that looked and found
# nothing reports. So a check matching nothing anywhere FAILS the run. If you
# deliberately remove the last occurrence of a documented fact, delete its
# check here in the same change.
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

# Total operations and /v1-only operations, space-separated on one line.
read -r TOTAL_ROUTES V1_ROUTES < <(python3 - "$OPENAPI" <<'PY'
import json, sys

with open(sys.argv[1]) as fh:
    spec = json.load(fh)

# Path-item objects also carry non-method keys (parameters, summary, ...).
METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}
ops = [(path, m) for path, item in spec["paths"].items() for m in item if m in METHODS]
print(len(ops), sum(1 for path, _ in ops if path.startswith("/v1")))
PY
)

echo "Workspace version: $CARGO_VERSION"
echo "OpenAPI routes:    $TOTAL_ROUTES total, $V1_ROUTES under /v1"
echo

EXIT=0
CHECKS=0

FILES=("README.md")
for d in docs/*.md; do
    [ -f "$d" ] && FILES+=("$d")
done

# check <label> <expected> <pattern> [value-regex]
#
# `pattern` must bracket the value tightly enough that the matched substring
# contains exactly the thing being compared. `value-regex` defaults to a
# semver triple; count checks pass a bare integer.
check() {
    local label=$1 expected=$2 pattern=$3 value_re=${4:-'[0-9]+\.[0-9]+\.[0-9]+'}
    local matched=0 file linenum line found
    CHECKS=$((CHECKS + 1))

    while IFS=: read -r file linenum line; do
        matched=1
        found=$(printf '%s' "$line" | grep -oE "$pattern" | grep -oE "$value_re" | head -1)
        if [ -n "$found" ] && [ "$found" != "$expected" ]; then
            printf 'DRIFT: %s:%s (%s): found %s, expected %s\n' \
                "$file" "$linenum" "$label" "$found" "$expected"
            printf '       > %s\n' "$line"
            EXIT=1
        fi
    done < <(grep -HnE "$pattern" "${FILES[@]}" 2>/dev/null || true)

    if [ "$matched" -eq 0 ]; then
        printf 'BLIND: (%s): matched nothing in README.md or docs/*.md — the phrase\n' "$label"
        printf '       this check is anchored on was reworded or removed, so the check\n'
        printf '       silently stopped running. Re-anchor it here, or drop it if the\n'
        printf '       fact is gone.\n'
        EXIT=1
    fi
}

check "git-dep tag"                    "$CARGO_VERSION" 'tag = "v[0-9]+\.[0-9]+\.[0-9]+"'
check "wire_version JSON"              "$CARGO_VERSION" '"wire_version":[[:space:]]*"[0-9]+\.[0-9]+\.[0-9]+"'
check "latest-tag prose"               "$CARGO_VERSION" 'latest tag.*\(`v[0-9]+\.[0-9]+\.[0-9]+`\)'
check "wire-version stability example" "$CARGO_VERSION" 'version \(e\.g\.[[:space:]]*`"[0-9]+\.[0-9]+\.[0-9]+"`\)'
check "docker image tag"               "$CARGO_VERSION" 'ghcr\.io/sligara7/dynograph-foundation:[0-9]+\.[0-9]+\.[0-9]+'
check "image-tag prose"                "$CARGO_VERSION" 'tagged per release \(`:[0-9]+\.[0-9]+\.[0-9]+`\)'
check "published-surface version"      "$CARGO_VERSION" 'published surface of .*\(v[0-9]+\.[0-9]+\.[0-9]+\)'
check "total route count"              "$TOTAL_ROUTES"  '\*\*[0-9]+ routes\*\*'   '[0-9]+'
check "/v1 route count"                "$V1_ROUTES"     'all [0-9]+ `/v1` routes' '[0-9]+'

if [ "$EXIT" -eq 0 ]; then
    echo "OK: doc version refs match Cargo.toml ($CARGO_VERSION), route counts match $OPENAPI,"
    echo "    and all $CHECKS checks matched live doc text."
fi

exit $EXIT
