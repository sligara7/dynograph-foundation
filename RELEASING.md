# Releasing dynograph-foundation

The human checklist for cutting a release. Written 2026-06-12, right
after the v0.9.3 release walked into every gap this file now closes:
no CHANGELOG entry in the feature PR, a doc-prose version site the
drift checker didn't cover, and GHCR images publishing without GitHub
Releases (v0.9.2/v0.9.3 had to be backfilled by hand).

The principle: **everything that can be checked is checked by CI;
this file is the ordered path through those checks**, not a list of
things to remember. If you find yourself remembering something CI
doesn't enforce, extend CI and add the step here.

## 1. In the release PR (one PR carries the whole bump)

- [ ] **Code + tests** for the change itself. House standard: pins for
      the bug class being fixed, not just the instance.
- [ ] **`Cargo.toml` workspace `version`** — single site; every crate
      inherits via `version.workspace = true`. `Cargo.lock` updates on
      the next build; commit it.
- [ ] **`CHANGELOG.md`**: add a `## vX.Y.Z — YYYY-MM-DD` section at the
      top. The release workflow lifts these notes verbatim into the
      GitHub Release, so write them for consumers. (v0.9.3 shipped
      without one — don't repeat that.)
- [ ] **Committed OpenAPI spec** if any wire type or route changed:
      `UPDATE_OPENAPI=1 cargo test -p dynograph-service openapi_spec`
      regenerates `docs/openapi.json`. The drift test fails CI until
      you do; the spec's `info.version` tracks the crate version, so a
      version bump alone also requires a regen.
- [ ] **Doc version sites**: `scripts/check-doc-versions.sh` (also a CI
      job, `doc-version-drift`) asserts the targeted patterns in
      README + docs/ match `Cargo.toml`. If you add a NEW place that
      mentions a version, add a pattern to the script in the same PR —
      untargeted prose is exactly how README's "tagged per release"
      line sat at 0.9.1 for two releases.
- [ ] **Local CI-equivalent before pushing** (the hosted run then just
      confirms): `cargo test --workspace`,
      `cargo clippy --workspace --all-targets -- -D warnings`,
      `cargo fmt --all -- --check`, `bash scripts/check-doc-versions.sh`.

## 2. Merge

PR → CI green (all jobs; the matrix includes the `fulltext` feature
build) → merge. **Tag the merge commit, not the branch head.**

## 3. Tag (this is the release trigger)

```bash
git checkout main && git pull
git tag vX.Y.Z          # on the verified merge commit
git push origin vX.Y.Z
```

The tag push triggers `.github/workflows/release.yml`, which does BOTH:

1. **GHCR image**: builds + pushes
   `ghcr.io/sligara7/dynograph-foundation:X.Y.Z` (unprefixed semver —
   the v is stripped) and `:latest`. ~3–4 min with a warm Actions
   cache, ~10 min cold (RocksDB).
2. **GitHub Release**: created from the tag with notes extracted from
   the tag's CHANGELOG section (warns + falls back to a stub if the
   section is missing — see step 1).

Verify both: the Actions run is green, and
`docker manifest inspect ghcr.io/sligara7/dynograph-foundation:X.Y.Z`
resolves.

## 4. Downstream: the storyflow pin bump

The consumer repo pins by git tag AND by image tag. In `storyflow`:

```bash
scripts/bump-version.sh foundation X.Y.Z   # versions.env + 7 Cargo tags + compose image
(cd services/dynograph && cargo update --workspace)   # Cargo.lock → the new tag
```

Then the coupled proof obligations (CI + safe-deploy gates re-check
all of these):

- `cargo check --workspace` in `services/dynograph` — compiles against
  the new crates. Keep `dynograph-resolution`'s public enums
  shape-stable when possible; the engine matches them exhaustively.
- `make -C services/generation_plus regen-schemas` — gen_plus codegen
  vs the dynograph schema. A no-op IS the proof when no schema yaml
  changed; commit the diff when it isn't.
- side_b contract run against the new foundation
  (`tests/side_b/runner` from the gen_plus image) — update probe
  expectations HONESTLY when semantics deliberately changed, with
  flip-back notes if the change is a documented dormancy.

One PR, flagged for review (it joins the deploy delta), merged on
local-green + stamp; the local dev stack's foundation container flips
on merged main (`docker compose up -d foundation`).

## Failure modes already paid for (don't rediscover)

- **Tag without CHANGELOG section** → Release ships a stub note
  (workflow warns). Fix: write the section in the release PR.
- **`:vX.Y.Z` vs `:X.Y.Z` image tags** — fixed 2026-05-11; container
  tags are unprefixed semver, git tags carry the `v`.
- **Bumping Cargo.toml but not docs** → `doc-version-drift` CI fails.
  Bumping docs the checker doesn't target → extend the checker (see
  step 1).
- **GHCR published but no Release object** → closed by the
  `github-release` job (this file's reason for existing). If the
  Releases page is ever behind again: `gh release create vX.Y.Z
  --notes-file <(awk section from CHANGELOG)` backfills.
- **Local RocksDB builds failing on `stdbool.h`** (bindgen vs gcc-15
  headers): `BINDGEN_EXTRA_CLANG_ARGS=-I/usr/lib/gcc/x86_64-linux-gnu/15/include`.
