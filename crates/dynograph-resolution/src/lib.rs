//! DynoGraph Resolution — Entity resolution with fuzzy matching + vector similarity.
//!
//! Provides three-tier entity resolution:
//! - Score >= auto_merge_threshold (default 90): auto-merge
//! - Score in [fuzzy_threshold, auto_merge_threshold): vector tiebreaker
//! - Score < fuzzy_threshold (default 70): create new entity
//!
//! Defaults above are the canonical `ResolutionConfig` defaults in
//! `dynograph-core` (`auto_merge` 90, `fuzzy` 70, `vector` 0.85).

mod fuzzy;
mod resolver;

pub use fuzzy::{jaro_winkler, token_sort_ratio};
pub use resolver::{Candidate, EntityResolver, ResolutionResult};
