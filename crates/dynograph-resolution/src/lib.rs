//! DynoGraph Resolution — Entity resolution with fuzzy matching + vector similarity.
//!
//! Provides three-tier entity resolution:
//! - Score >= auto_merge_threshold (default 95): auto-merge
//! - Score in [fuzzy_threshold, auto_merge_threshold): vector tiebreaker
//! - Score < fuzzy_threshold (default 70): create new entity

pub mod fuzzy;
pub mod resolver;

pub use fuzzy::{jaro_winkler, token_sort_ratio};
pub use resolver::{EntityResolver, ResolutionResult, Candidate};
