//! Entity resolver — combines fuzzy matching and vector similarity
//! into the three-tier resolution system.

use dynograph_core::{ResolutionConfig, ResolutionStrategy};
use dynograph_vector::{HnswIndex, cosine_similarity};

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

/// Provenance of the winning match pair: which field of the query matched
/// which field of the candidate. Anything other than `NameToName` is
/// "alias-sourced" and is never allowed to auto-merge on fuzzy score
/// alone — two distinct entities can legitimately share a descriptor
/// ("the captain"), so alias evidence requires vector corroboration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchSource {
    /// Query primary name matched a candidate's primary name.
    NameToName,
    /// Query primary name matched one of a candidate's stored aliases.
    NameToStoredAlias,
    /// An incoming alias matched a candidate's primary name.
    IncomingAliasToName,
    /// An incoming alias matched one of a candidate's stored aliases.
    IncomingAliasToStoredAlias,
}

impl MatchSource {
    /// True when either side of the winning pair is an alias.
    pub fn is_alias_sourced(&self) -> bool {
        !matches!(self, MatchSource::NameToName)
    }
}

/// Result of a source-aware resolution (`EntityResolver::resolve_sourced`).
#[derive(Debug, Clone)]
pub struct ResolutionOutcome {
    pub result: ResolutionResult,
    /// Provenance of the winning pair. `None` when `result` is `CreateNew`.
    pub match_source: Option<MatchSource>,
    /// Incoming aliases excluded from merge justification because they
    /// matched two or more distinct in-scope candidates above the fuzzy
    /// threshold — by construction non-identifying in this scope.
    pub ambiguous_aliases: Vec<String>,
    /// Candidates considered by the query that produced the decision
    /// (the primary name's candidates when nothing merged), best-first.
    pub candidates: Vec<Candidate>,
}

/// Entity resolver that implements the three-tier resolution strategy.
pub struct EntityResolver {
    auto_merge_threshold: u32,
    fuzzy_threshold: u32,
    vector_threshold: f32,
}

impl Default for EntityResolver {
    /// Canonical thresholds, single-sourced from core's
    /// `ResolutionConfig` defaults (auto-merge 90, fuzzy zone 70-89,
    /// vector cutoff 0.85) so they can't drift from the schema. The
    /// strategy is irrelevant here — the resolver only reads the three
    /// thresholds.
    fn default() -> Self {
        Self::from_config(&ResolutionConfig::new(ResolutionStrategy::default()))
    }
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

    /// Resolve a name against a list of existing entity names.
    ///
    /// Returns the resolution decision and the list of candidates
    /// considered, sorted best-first by fuzzy score. Callers pass
    /// `&[(&str, &str)]` so a `Vec<(String, String)>` doesn't need to
    /// be cloned at every call site — this previously forced every
    /// caller to allocate even when their data was already in-place.
    /// (The function still allocates owned `Candidate.id`/`.name`
    /// strings on its own — the candidates outlive the input slice.)
    pub fn resolve(
        &self,
        query_name: &str,
        existing: &[(&str, &str)], // (id, name) pairs
        query_embedding: Option<&[f32]>,
        vector_index: Option<&HnswIndex>,
    ) -> (ResolutionResult, Vec<Candidate>) {
        if existing.is_empty() {
            return (ResolutionResult::CreateNew, Vec::new());
        }

        // Phase 1: Fuzzy matching against all existing names. The query
        // is normalized once and scored against each candidate's
        // normalized form — `PreparedName` owns the lowercase/token-sort
        // contract so it isn't re-derived (or accidentally reordered)
        // per candidate.
        let query_prepared = fuzzy::PreparedName::new(query_name);
        let mut candidates: Vec<Candidate> = existing
            .iter()
            .map(|(id, name)| Candidate {
                id: (*id).to_string(),
                name: (*name).to_string(),
                fuzzy_score: query_prepared.score(&fuzzy::PreparedName::new(name)),
                vector_score: None,
            })
            .collect();

        // Sort by fuzzy score descending
        candidates.sort_by_key(|c| std::cmp::Reverse(c.fuzzy_score));

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
            if best.fuzzy_score >= self.fuzzy_threshold
                && let (Some(embedding), Some(index)) = (query_embedding, vector_index)
            {
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
                    .max_by(|a, b| a.1.total_cmp(&b.1));

                if let Some((best_c, vscore)) = best_vector
                    && vscore >= self.vector_threshold
                {
                    return (
                        ResolutionResult::VectorMerge {
                            candidate: best_c.id.clone(),
                        },
                        candidates,
                    );
                }
            }
        }

        // Tier 3: No match — create new
        (ResolutionResult::CreateNew, candidates)
    }

    /// Resolve a primary name plus alternate names (incoming aliases).
    ///
    /// The primary name is tried first; if it would create a new entity,
    /// each alias is resolved in turn against the same candidate list and
    /// the first merge wins. This catches cases like an incoming "Neo"
    /// carrying the alias "Thomas Anderson" matching an existing
    /// "Thomas Anderson" entity. Aliases that are empty or equal to the
    /// primary name (case-insensitively) are skipped.
    ///
    /// Returns the winning decision plus the candidate list from the query
    /// that produced it (primary's candidates when nothing merged).
    #[deprecated(
        since = "0.9.3",
        note = "flattens match provenance, so alias pairs auto-merge on fuzzy \
                score alone and two distinct entities sharing a generic alias \
                silently merge. Use `resolve_sourced`, which keeps name and \
                stored-alias candidates apart and vector-gates alias evidence."
    )]
    pub fn resolve_with_aliases(
        &self,
        query_name: &str,
        incoming_aliases: &[&str],
        existing: &[(&str, &str)],
        query_embedding: Option<&[f32]>,
        vector_index: Option<&HnswIndex>,
    ) -> (ResolutionResult, Vec<Candidate>) {
        let (primary, primary_candidates) =
            self.resolve(query_name, existing, query_embedding, vector_index);
        if !matches!(primary, ResolutionResult::CreateNew) {
            return (primary, primary_candidates);
        }

        let query_lower = query_name.to_lowercase();
        for alias in incoming_aliases {
            if alias.is_empty() || alias.to_lowercase() == query_lower {
                continue;
            }
            let (result, candidates) = self.resolve(alias, existing, query_embedding, vector_index);
            if !matches!(result, ResolutionResult::CreateNew) {
                return (result, candidates);
            }
        }

        (primary, primary_candidates)
    }

    /// Source-aware three-tier resolution (v0.9.3).
    ///
    /// Same tiers as `resolve`, with match provenance threaded through the
    /// decision instead of flattened away:
    ///
    /// - **Tier 1 (auto-merge)** considers ONLY name↔name pairs. A
    ///   stored-alias pair can no longer outscore-and-hijack an auto-merge,
    ///   and an incoming alias never auto-merges on fuzzy score alone.
    /// - **Tier 2 (vector tiebreak)** considers every pair at or above
    ///   `fuzzy_threshold`. Name↔name pairs reach it in the classic
    ///   `[fuzzy, auto_merge)` zone; alias-sourced pairs reach it at ANY
    ///   score — including an exact 100 — because alias evidence always
    ///   requires vector corroboration (≥ `vector_threshold`). Without a
    ///   query embedding, a vector index, or a stored vector for the
    ///   candidate, an alias-sourced match falls through to `CreateNew`.
    /// - **Ambiguity (O2):** an incoming alias matching ≥2 distinct
    ///   candidates above `fuzzy_threshold` is by construction
    ///   non-identifying in this scope; it is excluded from merge
    ///   justification entirely and reported in `ambiguous_aliases`.
    ///
    /// `primary` and `stored_aliases` are both `(id, text)` pairs; the same
    /// id may appear once in `primary` and many times in `stored_aliases`.
    pub fn resolve_sourced(
        &self,
        query_name: &str,
        incoming_aliases: &[&str],
        primary: &[(&str, &str)],
        stored_aliases: &[(&str, &str)],
        query_embedding: Option<&[f32]>,
        vector_index: Option<&HnswIndex>,
    ) -> ResolutionOutcome {
        if primary.is_empty() && stored_aliases.is_empty() {
            return ResolutionOutcome {
                result: ResolutionResult::CreateNew,
                match_source: None,
                ambiguous_aliases: Vec::new(),
                candidates: Vec::new(),
            };
        }

        // Primary name first.
        let (decision, primary_scored) = self.decide_one_query(
            query_name,
            false,
            primary,
            stored_aliases,
            query_embedding,
            vector_index,
        );
        if let Some((result, source)) = decision {
            return ResolutionOutcome {
                result,
                match_source: Some(source),
                ambiguous_aliases: Vec::new(),
                candidates: primary_scored,
            };
        }

        // Incoming aliases, first merge wins; ambiguous aliases are
        // excluded from merge justification (O2).
        let query_lower = query_name.to_lowercase();
        let mut ambiguous_aliases: Vec<String> = Vec::new();
        for alias in incoming_aliases {
            if alias.is_empty() || alias.to_lowercase() == query_lower {
                continue;
            }
            let (decision, scored) = self.decide_one_query(
                alias,
                true,
                primary,
                stored_aliases,
                query_embedding,
                vector_index,
            );
            let distinct_hits = {
                let mut ids: Vec<&str> = scored
                    .iter()
                    .filter(|c| c.fuzzy_score >= self.fuzzy_threshold)
                    .map(|c| c.id.as_str())
                    .collect();
                ids.sort_unstable();
                ids.dedup();
                ids.len()
            };
            if distinct_hits >= 2 {
                ambiguous_aliases.push((*alias).to_string());
                continue;
            }
            if let Some((result, source)) = decision {
                return ResolutionOutcome {
                    result,
                    match_source: Some(source),
                    ambiguous_aliases,
                    candidates: scored,
                };
            }
        }

        ResolutionOutcome {
            result: ResolutionResult::CreateNew,
            match_source: None,
            ambiguous_aliases,
            candidates: primary_scored,
        }
    }

    /// Score one query string against every (primary + stored-alias) pair
    /// and apply the source-aware tiers. Returns the decision (None = this
    /// query justifies no merge) and the scored candidate list, best-first.
    /// Primary pairs are scored before alias pairs so the stable sort keeps
    /// a name ahead of an equal-scoring alias.
    fn decide_one_query(
        &self,
        query: &str,
        query_is_alias: bool,
        primary: &[(&str, &str)],
        stored_aliases: &[(&str, &str)],
        query_embedding: Option<&[f32]>,
        vector_index: Option<&HnswIndex>,
    ) -> (Option<(ResolutionResult, MatchSource)>, Vec<Candidate>) {
        let query_prepared = fuzzy::PreparedName::new(query);
        // (candidate, is_stored_alias) — Candidate.name carries the text
        // that was scored (the alias text for alias pairs), matching what
        // the flattened `resolve` path reported for those entries.
        let mut scored: Vec<(Candidate, bool)> = primary
            .iter()
            .map(|(id, text)| (*id, *text, false))
            .chain(stored_aliases.iter().map(|(id, text)| (*id, *text, true)))
            .map(|(id, text, is_alias)| {
                (
                    Candidate {
                        id: id.to_string(),
                        name: text.to_string(),
                        fuzzy_score: query_prepared.score(&fuzzy::PreparedName::new(text)),
                        vector_score: None,
                    },
                    is_alias,
                )
            })
            .collect();
        scored.sort_by_key(|(c, _)| std::cmp::Reverse(c.fuzzy_score));

        // Tier 1: auto-merge on the best NAME↔NAME pair only.
        if !query_is_alias
            && let Some((best_name, _)) = scored.iter().find(|(_, is_alias)| !is_alias)
            && best_name.fuzzy_score >= self.auto_merge_threshold
        {
            let result = ResolutionResult::AutoMerge {
                candidate: best_name.id.clone(),
            };
            let candidates = scored.into_iter().map(|(c, _)| c).collect();
            return (Some((result, MatchSource::NameToName)), candidates);
        }

        // Tier 2: vector tiebreak over every pair in the zone. Name↔name
        // pairs at/above auto_merge were handled above (or, for an alias
        // query, are alias-sourced by definition and belong here).
        if let (Some(embedding), Some(index)) = (query_embedding, vector_index) {
            for (c, _) in scored.iter_mut() {
                if c.fuzzy_score < self.fuzzy_threshold {
                    break; // sorted — below the zone, stop.
                }
                if let Some(vec) = index.get_vector(&c.id) {
                    c.vector_score = Some(cosine_similarity(embedding, vec));
                }
            }
            let best_vector = scored
                .iter()
                .filter(|(c, _)| c.fuzzy_score >= self.fuzzy_threshold)
                .filter_map(|(c, is_alias)| c.vector_score.map(|v| (c, *is_alias, v)))
                .max_by(|a, b| a.2.total_cmp(&b.2));
            if let Some((best_c, is_stored_alias, vscore)) = best_vector
                && vscore >= self.vector_threshold
            {
                let source = match (query_is_alias, is_stored_alias) {
                    (false, false) => MatchSource::NameToName,
                    (false, true) => MatchSource::NameToStoredAlias,
                    (true, false) => MatchSource::IncomingAliasToName,
                    (true, true) => MatchSource::IncomingAliasToStoredAlias,
                };
                let result = ResolutionResult::VectorMerge {
                    candidate: best_c.id.clone(),
                };
                let candidates = scored.iter().map(|(c, _)| c.clone()).collect();
                return (Some((result, source)), candidates);
            }
        }

        (None, scored.into_iter().map(|(c, _)| c).collect())
    }
}

#[cfg(test)]
// The legacy `resolve_with_aliases` tests deliberately keep pinning the
// deprecated flattened path until it is removed.
#[allow(deprecated)]
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
        let existing = [("id1", "Alice")];
        let (result, _) = resolver.resolve("Alice", &existing, None, None);
        assert!(matches!(result, ResolutionResult::AutoMerge { candidate } if candidate == "id1"));
    }

    #[test]
    fn near_exact_match_auto_merges() {
        let resolver = default_resolver();
        let existing = [("id1", "Marcus Whitfield")];
        let (result, _) = resolver.resolve("Marcus Whitfeld", &existing, None, None);
        // Jaro-Winkler for near-matches should be >= 95
        assert!(
            matches!(result, ResolutionResult::AutoMerge { .. }),
            "Expected AutoMerge, got {:?}",
            result
        );
    }

    #[test]
    fn token_reordered_name_auto_merges() {
        // Exercises the token-sort branch of the (now hoisted) fuzzy
        // scoring end-to-end: a reordered name must still auto-merge,
        // since `token_sort_ratio` normalizes token order to 100.
        let resolver = default_resolver();
        let existing = [("id1", "Marcus Whitfield")];
        let (result, _) = resolver.resolve("Whitfield Marcus", &existing, None, None);
        assert!(
            matches!(&result, ResolutionResult::AutoMerge { candidate } if candidate == "id1"),
            "reordered tokens should auto-merge, got {result:?}"
        );
    }

    #[test]
    fn completely_different_creates_new() {
        let resolver = default_resolver();
        let existing = [("id1", "Alice")];
        let (result, _) = resolver.resolve("Xylophone", &existing, None, None);
        assert_eq!(result, ResolutionResult::CreateNew);
    }

    #[test]
    fn tiebreaker_zone_without_vector_creates_new() {
        let resolver = default_resolver();
        // "Professor" vs "Professor Whitfield" — fuzzy score in 70-94 range
        let existing = [("id1", "Professor Whitfield")];
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
        let existing = [("id1", "Professor Edwin Whitfield")];

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
        let existing = [("id1", "Professor Whitfield")];

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
        let existing = [
            ("id1", "Alice Smith"),
            ("id2", "Alice Johnson"),
            ("id3", "Bob Wilson"),
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
        let config = ResolutionConfig::new(ResolutionStrategy::FuzzyThenVector)
            .with_fuzzy_threshold(60)
            .with_vector_threshold(0.9)
            .with_auto_merge_threshold(90);
        let resolver = EntityResolver::from_config(&config);
        // Verify custom thresholds work — exact match should still auto-merge at 90+
        let existing = [("id1", "Alice")];
        let (result, _) = resolver.resolve("Alice", &existing, None, None);
        assert!(matches!(result, ResolutionResult::AutoMerge { .. }));
    }

    #[test]
    fn alias_merges_when_primary_misses() {
        let resolver = default_resolver();
        let existing = [("id1", "Thomas Anderson")];
        let (result, _) =
            resolver.resolve_with_aliases("Neo", &["Thomas Anderson"], &existing, None, None);
        assert!(
            matches!(&result, ResolutionResult::AutoMerge { candidate } if candidate == "id1"),
            "alias should auto-merge, got {result:?}"
        );
    }

    #[test]
    fn primary_match_wins_before_aliases_are_tried() {
        let resolver = default_resolver();
        let existing = [("id1", "Alice"), ("id2", "Beatrix")];
        // Primary matches id1; the alias would match id2 — primary wins.
        let (result, _) =
            resolver.resolve_with_aliases("Alice", &["Beatrix"], &existing, None, None);
        assert!(
            matches!(&result, ResolutionResult::AutoMerge { candidate } if candidate == "id1"),
            "primary match must take precedence, got {result:?}"
        );
    }

    #[test]
    fn empty_and_primary_equal_aliases_are_skipped() {
        let resolver = default_resolver();
        let existing = [("id1", "Completely Unrelated")];
        // "" and a case-variant of the primary must be skipped, not resolved.
        let (result, _) =
            resolver.resolve_with_aliases("Xylophone", &["", "XYLOPHONE"], &existing, None, None);
        assert_eq!(result, ResolutionResult::CreateNew);
    }

    #[test]
    fn all_aliases_missing_returns_primary_create_new() {
        let resolver = default_resolver();
        let existing = [("id1", "Alice")];
        let (result, candidates) = resolver.resolve_with_aliases(
            "Xylophone",
            &["Quagmire", "Zeppelin"],
            &existing,
            None,
            None,
        );
        assert_eq!(result, ResolutionResult::CreateNew);
        // Candidate list is the PRIMARY query's, not the last alias's.
        assert_eq!(candidates[0].name, "Alice");
    }

    #[test]
    fn reordered_name_matches() {
        let resolver = default_resolver();
        let existing = [("id1", "John Smith")];
        let (result, _) = resolver.resolve("Smith, John", &existing, None, None);
        // token_sort_ratio handles reordering
        assert!(
            matches!(result, ResolutionResult::AutoMerge { .. }),
            "Expected AutoMerge for reordered name, got {:?}",
            result
        );
    }

    // ---- resolve_sourced (v0.9.3): O1 match-source gating + O2 ambiguity ----

    /// THE D2 over-merge pin: two DISTINCT characters sharing a generic
    /// alias must NOT merge on alias evidence alone. v0.9.2 auto-merged
    /// this at score 100 (alias↔stored-alias exact match).
    #[test]
    fn distinct_characters_shared_alias_without_vector_creates_new() {
        let resolver = default_resolver();
        let primary = [("id1", "Aldous Vane")];
        let stored = [("id1", "the captain")];
        let outcome =
            resolver.resolve_sourced("Mira Chen", &["the captain"], &primary, &stored, None, None);
        assert_eq!(
            outcome.result,
            ResolutionResult::CreateNew,
            "alias-only exact match without vector support must CreateNew"
        );
        assert_eq!(outcome.match_source, None);
    }

    /// Same shape WITH strong vector corroboration: the same character
    /// described twice IS cosine-similar, so the merge proceeds — as a
    /// VectorMerge, never an AutoMerge — and provenance is reported.
    #[test]
    fn alias_match_with_strong_vector_merges_as_vector_merge() {
        let resolver = default_resolver();
        let primary = [("id1", "Aldous Vane")];
        let stored = [("id1", "the captain")];
        let mut index = HnswIndex::new(HnswConfig::new(3));
        index.insert("id1", &[0.9, 0.1, 0.0]);
        let embedding = [0.85, 0.15, 0.0];
        let outcome = resolver.resolve_sourced(
            "Mira Chen",
            &["the captain"],
            &primary,
            &stored,
            Some(&embedding),
            Some(&index),
        );
        assert!(
            matches!(&outcome.result, ResolutionResult::VectorMerge { candidate } if candidate == "id1"),
            "expected VectorMerge, got {:?}",
            outcome.result
        );
        assert_eq!(
            outcome.match_source,
            Some(MatchSource::IncomingAliasToStoredAlias)
        );
    }

    /// Weak vector evidence on an alias-exact match still creates new —
    /// distinct profiles sharing a descriptor are not cosine-similar.
    #[test]
    fn alias_match_with_weak_vector_creates_new() {
        let resolver = default_resolver();
        let primary = [("id1", "Aldous Vane")];
        let stored = [("id1", "the captain")];
        let mut index = HnswIndex::new(HnswConfig::new(3));
        index.insert("id1", &[1.0, 0.0, 0.0]);
        let embedding = [0.0, 1.0, 0.0]; // orthogonal
        let outcome = resolver.resolve_sourced(
            "Mira Chen",
            &["the captain"],
            &primary,
            &stored,
            Some(&embedding),
            Some(&index),
        );
        assert_eq!(outcome.result, ResolutionResult::CreateNew);
    }

    /// The #130 keep-the-win direction under the new rules: an incoming
    /// alias matching an existing primary name merges WITH vector
    /// corroboration (IncomingAliasToName), and — the documented v0.9.3
    /// trade-off — creates new WITHOUT it.
    #[test]
    fn incoming_alias_to_name_is_vector_gated() {
        let resolver = default_resolver();
        let primary = [("id1", "Thomas Anderson")];
        let stored: [(&str, &str); 0] = [];

        let without =
            resolver.resolve_sourced("Neo", &["Thomas Anderson"], &primary, &stored, None, None);
        assert_eq!(
            without.result,
            ResolutionResult::CreateNew,
            "no embedding ⇒ alias evidence is insufficient"
        );

        // NB: "Neo" itself token-sort-scores 73 vs "Thomas Anderson" —
        // inside the fuzzy zone — so with an embedding the PRIMARY round
        // would legitimately vector-merge as name_to_name before the alias
        // is tried. "Trinity" (53) stays below the zone, isolating the
        // incoming-alias path.
        let mut index = HnswIndex::new(HnswConfig::new(3));
        index.insert("id1", &[0.9, 0.1, 0.0]);
        let embedding = [0.85, 0.15, 0.0];
        let with = resolver.resolve_sourced(
            "Trinity",
            &["Thomas Anderson"],
            &primary,
            &stored,
            Some(&embedding),
            Some(&index),
        );
        assert!(
            matches!(&with.result, ResolutionResult::VectorMerge { candidate } if candidate == "id1"),
            "expected VectorMerge, got {:?}",
            with.result
        );
        assert_eq!(with.match_source, Some(MatchSource::IncomingAliasToName));
    }

    /// name↔name behavior is unchanged: exact primary-name match still
    /// auto-merges with no vector evidence at all.
    #[test]
    fn name_to_name_auto_merge_unchanged() {
        let resolver = default_resolver();
        let primary = [("id1", "Alice")];
        let stored: [(&str, &str); 0] = [];
        let outcome = resolver.resolve_sourced("Alice", &[], &primary, &stored, None, None);
        assert!(
            matches!(&outcome.result, ResolutionResult::AutoMerge { candidate } if candidate == "id1")
        );
        assert_eq!(outcome.match_source, Some(MatchSource::NameToName));
    }

    /// A node storing an alias equal to another node's primary name must
    /// not hijack that node's exact-name auto-merge: tier 1 considers
    /// name↔name pairs only.
    #[test]
    fn stored_alias_cannot_hijack_name_auto_merge() {
        let resolver = default_resolver();
        let primary = [("id1", "Alice")];
        let stored = [("id2", "Alice")]; // id2 stores "Alice" as an alias
        let outcome = resolver.resolve_sourced("Alice", &[], &primary, &stored, None, None);
        assert!(
            matches!(&outcome.result, ResolutionResult::AutoMerge { candidate } if candidate == "id1"),
            "the primary-name owner must win, got {:?}",
            outcome.result
        );
        assert_eq!(outcome.match_source, Some(MatchSource::NameToName));
    }

    /// Query primary name hitting only a stored alias is alias-sourced:
    /// vector-gated like every other alias pair.
    #[test]
    fn name_to_stored_alias_is_vector_gated() {
        let resolver = default_resolver();
        let primary = [("id1", "Aldous Vane")];
        let stored = [("id1", "The Captain")];

        let without = resolver.resolve_sourced("the captain", &[], &primary, &stored, None, None);
        assert_eq!(without.result, ResolutionResult::CreateNew);

        let mut index = HnswIndex::new(HnswConfig::new(3));
        index.insert("id1", &[0.9, 0.1, 0.0]);
        let embedding = [0.85, 0.15, 0.0];
        let with = resolver.resolve_sourced(
            "the captain",
            &[],
            &primary,
            &stored,
            Some(&embedding),
            Some(&index),
        );
        assert!(
            matches!(&with.result, ResolutionResult::VectorMerge { candidate } if candidate == "id1")
        );
        assert_eq!(with.match_source, Some(MatchSource::NameToStoredAlias));
    }

    /// O2: an incoming alias matching two DISTINCT candidates above the
    /// fuzzy threshold is non-identifying — excluded from merge
    /// justification even when vector evidence would have cleared it,
    /// and reported in `ambiguous_aliases`.
    #[test]
    fn ambiguous_incoming_alias_is_excluded_and_reported() {
        let resolver = default_resolver();
        let primary = [("id1", "Aldous Vane"), ("id2", "Carla Reyes")];
        let stored = [("id1", "the captain"), ("id2", "the captain")];
        let mut index = HnswIndex::new(HnswConfig::new(3));
        index.insert("id1", &[0.9, 0.1, 0.0]);
        index.insert("id2", &[0.0, 0.0, 1.0]);
        let embedding = [0.85, 0.15, 0.0]; // strongly similar to id1
        let outcome = resolver.resolve_sourced(
            "Mira Chen",
            &["the captain"],
            &primary,
            &stored,
            Some(&embedding),
            Some(&index),
        );
        assert_eq!(
            outcome.result,
            ResolutionResult::CreateNew,
            "ambiguous alias must not justify a merge, got {:?}",
            outcome.result
        );
        assert_eq!(outcome.ambiguous_aliases, vec!["the captain".to_string()]);
    }

    /// O2 counts DISTINCT candidates: one node matched via both its name
    /// and its stored alias is one candidate, not an ambiguity.
    #[test]
    fn same_node_matched_twice_is_not_ambiguous() {
        let resolver = default_resolver();
        let primary = [("id1", "The Captain")];
        let stored = [("id1", "the captain")];
        let mut index = HnswIndex::new(HnswConfig::new(3));
        index.insert("id1", &[0.9, 0.1, 0.0]);
        let embedding = [0.85, 0.15, 0.0];
        let outcome = resolver.resolve_sourced(
            "Mira Chen",
            &["the captain"],
            &primary,
            &stored,
            Some(&embedding),
            Some(&index),
        );
        assert!(
            matches!(&outcome.result, ResolutionResult::VectorMerge { candidate } if candidate == "id1"),
            "single-candidate alias with vector support should merge, got {:?}",
            outcome.result
        );
        assert!(outcome.ambiguous_aliases.is_empty());
    }

    /// The classic name↔name tiebreaker zone still works through the
    /// sourced path: fuzzy zone hit + strong vector ⇒ VectorMerge with
    /// NameToName provenance.
    #[test]
    fn sourced_name_tiebreaker_zone_with_vector_merges() {
        let resolver = default_resolver();
        let primary = [("id1", "Professor Edwin Whitfield")];
        let stored: [(&str, &str); 0] = [];
        let mut index = HnswIndex::new(HnswConfig::new(3));
        index.insert("id1", &[0.9, 0.1, 0.0]);
        let embedding = [0.85, 0.15, 0.0];
        let outcome = resolver.resolve_sourced(
            "Edwin Whitfield",
            &[],
            &primary,
            &stored,
            Some(&embedding),
            Some(&index),
        );
        assert!(
            matches!(outcome.result, ResolutionResult::VectorMerge { .. }),
            "expected VectorMerge, got {:?}",
            outcome.result
        );
        assert_eq!(outcome.match_source, Some(MatchSource::NameToName));
    }

    #[test]
    fn sourced_empty_candidates_creates_new() {
        let resolver = default_resolver();
        let outcome = resolver.resolve_sourced("Alice", &["Al"], &[], &[], None, None);
        assert_eq!(outcome.result, ResolutionResult::CreateNew);
        assert!(outcome.candidates.is_empty());
        assert!(outcome.ambiguous_aliases.is_empty());
    }
}
