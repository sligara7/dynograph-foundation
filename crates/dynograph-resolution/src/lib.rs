//! DynoGraph Resolution — Entity resolution with fuzzy matching + vector similarity.
//!
//! Provides three-tier entity resolution:
//! - Score >= auto_merge_threshold (default 90): auto-merge
//! - Score in [fuzzy_threshold, auto_merge_threshold): vector tiebreaker
//! - Score < fuzzy_threshold (default 70): create new entity
//!
//! Defaults above are the canonical `ResolutionConfig` defaults in
//! `dynograph-core` (`auto_merge` 90, `fuzzy` 70, `vector` 0.85).
//!
//! Since v0.9.3 the source-aware path (`EntityResolver::resolve_sourced`)
//! is the recommended entry point: auto-merge is reserved for name↔name
//! evidence, alias-sourced matches always require vector corroboration,
//! and incoming aliases matching ≥2 candidates are excluded as
//! non-identifying (`ResolutionOutcome::ambiguous_aliases`).

mod fuzzy;
mod resolver;

pub use fuzzy::{jaro_winkler, token_sort_ratio};
pub use resolver::{Candidate, EntityResolver, MatchSource, ResolutionOutcome, ResolutionResult};

// Types owned by other crates that cross THIS crate's public boundary, so a
// caller can name every type in a signature it has to satisfy without taking
// an undeclared dependency at a version nobody stated.
//
// `HnswIndex` is the sharp case: `resolve` / `resolve_with_aliases` /
// `resolve_sourced` take `Option<&HnswIndex>` as a PARAMETER, so a caller
// cannot merely read it — it must construct one. Before this re-export that
// meant adding `dynograph-vector` to your own manifest and guessing a
// compatible version, for a coupling neither crate declared.
//
// Re-exported rather than wrapped because `dynograph-vector` is a REQUIRED
// dependency here: the type is always present, and a wrapper would fork the
// shared vocabulary for no gain. Wrapping is reserved for types arriving
// through an OPTIONAL dependency, which a consumer cannot depend on at all —
// that is why `FulltextHit` in dynograph-storage is a new type and these are
// not. See `req:boundaries-own-their-types`.
pub use dynograph_core::{ResolutionConfig, ResolutionStrategy};
pub use dynograph_vector::{HnswConfig, HnswIndex};
