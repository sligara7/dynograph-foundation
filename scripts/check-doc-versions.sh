#!/usr/bin/env bash
# Asserts that hardcoded version strings in README.md and docs/*.md
# match the workspace `version` in Cargo.toml. Catches the class of
# drift where a release bump in Cargo.toml goes in but the docs still
# advertise the previous tag.
#
# Targeted patterns (NOT a generic "any X.Y.Z" scan, which would
# false-positive on historical references in CHANGELOG-style prose):
#
#   - `tag = "vX.Y.Z"`                      (git-dep tag in Rust toml blocks)
#   - `"wire_version":  "X.Y.Z"`            (JSON example bodies)
#   - `latest tag (`vX.Y.Z`)`               (prose phrasing in README)
#   - `version (e.g. "X.Y.Z")`              (api.md's wire-stability section)
#
# CHANGELOG.md is intentionally excluded — historical version
# references there are correct.
#
# Brittleness: each pattern below is anchored on a literal phrase
# ("tag = ", "wire_version", "latest tag", "ghcr.io/..."). Reworking
# any of those phrases in the docs disables the corresponding check
# silently — update both sides together.
#
# Local usage: `./scripts/check-doc-versions.sh`. CI runs the same.

set -euo pipefail

CARGO_VERSION=$(awk -F'"' '/^version = / {print $2; exit}' Cargo.toml)

if [ -z "$CARGO_VERSION" ]; then
    echo "FAIL: could not extract workspace version from Cargo.toml" >&2
    exit 1
fi

echo "Workspace version: $CARGO_VERSION"
echo

EXIT=0

check_pattern() {
    local file=$1
    local pattern=$2
    local label=$3

    while IFS=: read -r linenum line; do
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
done

if [ "$EXIT" -eq 0 ]; then
    echo "OK: all targeted doc version refs match Cargo.toml ($CARGO_VERSION)"
fi

exit $EXIT
