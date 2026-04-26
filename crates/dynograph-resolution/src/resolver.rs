//! Entity resolver — combines fuzzy matching and vector similarity
//! into the three-tier resolution system.

use dynograph_core::ResolutionConfig;
use dynograph_vector::{cosine_similarity, HnswIndex};

use crate::fuzzy;

/// A candidate entity found during resolution.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub id: String,
    pub name: String,
    pub fuzzy_score: u32,
    pub vector_score: Option<f32>,
}

/// The resolution decision.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionResult {
    /// Fuzzy score >= auto_merge_threshold. Merge with this entity.
    AutoMerge { candidate: String },
    /// Fuzzy score in tiebreaker zone AND vector score >= vector_threshold.
    VectorMerge { candidate: String },
    /// No match found above thresholds. Create a new entity.
    CreateNew,
}

/// Entity resolver that implements the three-tier resolution strategy.
pub struct EntityResolver {
    auto_merge_threshold: u32,
    fuzzy_threshold: u32,
    vector_threshold: f32,
}

impl EntityResolver {
    /// Create a resolver from a schema ResolutionConfig.
    pub fn from_config(config: &ResolutionConfig) -> Self {
        Self {
            auto_merge_threshold: config.auto_merge_threshold,
            fuzzy_threshold: config.fuzzy_threshold,
            vector_threshold: config.vector_threshold as f32,
        }
    }

    /// Create a resolver with default thresholds.
    pub fn default() -> Self {
        Self {
            auto_merge_threshold: 90,
            fuzzy_threshold: 70,
            vector_threshold: 0.85,
        }
    }

    /// Resolve a name against a list of existing entity names.
    ///
    /// Returns the resolution decision and the list of candidates considered.
    pub fn resolve(
        &self,
        query_name: &str,
        existing: &[(String, String)], // (id, name) pairs
        query_embedding: Option<&[f32]>,
        vector_index: Option<&HnswIndex>,
    ) -> (ResolutionResult, Vec<Candidate>) {
        if existing.is_empty() {
            return (ResolutionResult::CreateNew, Vec::new());
        }

        // Phase 1: Fuzzy matching against all existing names
        let mut candidates: Vec<Candidate> = existing
            .iter()
            .map(|(id, name)| {
                let fuzzy_score = fuzzy::token_sort_ratio(query_name, name)
                    .max(fuzzy::jaro_winkler(query_name, name));
                Candidate {
                    id: id.clone(),
                    name: name.clone(),
                    fuzzy_score,
                    vector_score: None,
                }
            })
            .collect();

        // Sort by fuzzy score descending
        candidates.sort_by(|a, b| b.fuzzy_score.cmp(&a.fuzzy_score));

        // Check top candidate
        if let Some(best) = candidates.first() {
            // Tier 1: Auto-merge
            if best.fuzzy_score >= self.auto_merge_threshold {
                return (
                    ResolutionResult::AutoMerge {
                        candidate: best.id.clone(),
                    },
                    candidates,
                );
            }

            // Tier 2: Tiebreaker zone — use vector similarity
            if best.fuzzy_score >= self.fuzzy_threshold {
                if let (Some(embedding), Some(index)) = (query_embedding, vector_index) {
                    // Search for vector matches among candidates in the zone
                    for c in candidates.iter_mut() {
                        if c.fuzzy_score < self.fuzzy_threshold {
                            break; // Below zone, stop checking
                        }
                        if let Some(vec) = index.get_vector(&c.id) {
                            let vscore = cosine_similarity(embedding, vec);
                            c.vector_score = Some(vscore);
                        }
                    }

                    // Find best combined candidate in the zone
                    let best_vector = candidates
                        .iter()
                        .filter(|c| c.fuzzy_score >= self.fuzzy_threshold)
                        .filter_map(|c| c.vector_score.map(|v| (c, v)))
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    if let Some((best_c, vscore)) = best_vector {
                        if vscore >= self.vector_threshold {
                            return (
                                ResolutionResult::VectorMerge {
                                    candidate: best_c.id.clone(),
                                },
                                candidates,
                            );
                        }
                    }
                }
            }
        }

        // Tier 3: No match — create new
        (ResolutionResult::CreateNew, candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynograph_vector::HnswConfig;

    fn default_resolver() -> EntityResolver {
        EntityResolver::default()
    }

    #[test]
    fn empty_existing_creates_new() {
        let resolver = default_resolver();
        let (result, candidates) = resolver.resolve("Alice", &[], None, None);
        assert_eq!(result, ResolutionResult::CreateNew);
        assert!(candidates.is_empty());
    }

    #[test]
    fn exact_match_auto_merges() {
        let resolver = default_resolver();
        let existing = vec![("id1".to_string(), "Alice".to_string())];
        let (result, _) = resolver.resolve("Alice", &existing, None, None);
        assert!(matches!(result, ResolutionResult::AutoMerge { candidate } if candidate == "id1"));
    }

    #[test]
    fn near_exact_match_auto_merges() {
        let resolver = default_resolver();
        let existing = vec![("id1".to_string(), "Marcus Whitfield".to_string())];
        let (result, _) = resolver.resolve("Marcus Whitfeld", &existing, None, None);
        // Jaro-Winkler for near-matches should be >= 95
        assert!(
            matches!(result, ResolutionResult::AutoMerge { .. }),
            "Expected AutoMerge, got {:?}",
            result
        );
    }

    #[test]
    fn completely_different_creates_new() {
        let resolver = default_resolver();
        let existing = vec![("id1".to_string(), "Alice".to_string())];
        let (result, _) = resolver.resolve("Xylophone", &existing, None, None);
        assert_eq!(result, ResolutionResult::CreateNew);
    }

    #[test]
    fn tiebreaker_zone_without_vector_creates_new() {
        let resolver = default_resolver();
        // "Professor" vs "Professor Whitfield" — fuzzy score in 70-94 range
        let existing = vec![("id1".to_string(), "Professor Whitfield".to_string())];
        let (result, candidates) = resolver.resolve("the old professor", &existing, None, None);
        // Without vector index, tiebreaker zone defaults to CreateNew
        assert_eq!(result, ResolutionResult::CreateNew);
        // But we should have a candidate with a score
        assert!(!candidates.is_empty());
    }

    #[test]
    fn tiebreaker_zone_with_vector_match() {
        let resolver = default_resolver();
        // Use names that land in the fuzzy tiebreaker zone (70-94)
        let existing = vec![("id1".to_string(), "Professor Edwin Whitfield".to_string())];

        let mut index = HnswIndex::new(HnswConfig::new(3));
        index.insert("id1", &[0.9, 0.1, 0.0]);

        let query_embedding = [0.85, 0.15, 0.0]; // very similar

        let (result, candidates) = resolver.resolve(
            "Edwin Whitfield",
            &existing,
            Some(&query_embedding),
            Some(&index),
        );

        // Should be in tiebreaker zone and vector should push it over
        let top = &candidates[0];
        assert!(
            top.fuzzy_score >= 70 && top.fuzzy_score < 95,
            "Expected tiebreaker zone, got fuzzy_score={}",
            top.fuzzy_score
        );
        assert!(
            matches!(result, ResolutionResult::VectorMerge { .. }),
            "Expected VectorMerge, got {:?}",
            result
        );
    }

    #[test]
    fn tiebreaker_zone_with_weak_vector_creates_new() {
        let resolver = default_resolver();
        let existing = vec![("id1".to_string(), "Professor Whitfield".to_string())];

        let mut index = HnswIndex::new(HnswConfig::new(3));
        index.insert("id1", &[1.0, 0.0, 0.0]);

        // Orthogonal embedding — low vector similarity
        let query_embedding = [0.0, 1.0, 0.0];

        let (result, _) = resolver.resolve(
            "the old professor",
            &existing,
            Some(&query_embedding),
            Some(&index),
        );

        assert_eq!(result, ResolutionResult::CreateNew);
    }

    #[test]
    fn multiple_candidates_best_wins() {
        let resolver = default_resolver();
        let existing = vec![
            ("id1".to_string(), "Alice Smith".to_string()),
            ("id2".to_string(), "Alice Johnson".to_string()),
            ("id3".to_string(), "Bob Wilson".to_string()),
        ];
        let (result, candidates) = resolver.resolve("Alice Smyth", &existing, None, None);
        // "Alice Smyth" is closest to "Alice Smith"
        assert!(
            matches!(&result, ResolutionResult::AutoMerge { candidate } if candidate == "id1"),
            "Expected AutoMerge with id1, got {:?}",
            &result
        );
        // Candidates should be sorted by score
        assert!(candidates[0].fuzzy_score >= candidates[1].fuzzy_score);
    }

    #[test]
    fn from_schema_config() {
        let config = ResolutionConfig {
            strategy: "fuzzy_then_vector".to_string(),
            fuzzy_threshold: 60,
            vector_threshold: 0.9,
            auto_merge_threshold: 90,
        };
        let resolver = EntityResolver::from_config(&config);
        // Verify custom thresholds work — exact match should still auto-merge at 90+
        let existing = vec![("id1".to_string(), "Alice".to_string())];
        let (result, _) = resolver.resolve("Alice", &existing, None, None);
        assert!(matches!(result, ResolutionResult::AutoMerge { .. }));
    }

    #[test]
    fn reordered_name_matches() {
        let resolver = default_resolver();
        let existing = vec![("id1".to_string(), "John Smith".to_string())];
        let (result, _) = resolver.resolve("Smith, John", &existing, None, None);
        // token_sort_ratio handles reordering
        assert!(
            matches!(result, ResolutionResult::AutoMerge { .. }),
            "Expected AutoMerge for reordered name, got {:?}",
            result
        );
    }
}
