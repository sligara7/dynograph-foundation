//! POST /v1/graphs/{id}/search:hybrid — Reciprocal-Rank-Fusion over the
//! retrieval legs foundation already has.
//!
//! One domain-neutral hybrid-search primitive: fan out to the vector
//! (HNSW) and keyword (BM25) legs, optionally constrain both to a
//! structured `where` prefilter, and Reciprocal-Rank-Fuse the ranked
//! outputs into a single ranked node list. Without it every consumer's
//! NL→graph / GraphRAG layer reimplements rank fusion — the "math in
//! consumers" anti-pattern the consolidation policy (ISSUES #21) forbids.
//! RRF is rank math, so it lives here.
//!
//! ## Wire shape
//!
//! ```jsonc
//! POST /v1/graphs/{id}/search:hybrid
//! {
//!   "query": "the dagger Aria stole",   // optional → keyword/BM25 leg
//!   "query_vector": [/* f32 */],         // optional → vector leg (DF never embeds)
//!   "node_type": "Fragment",            // filter on every leg; REQUIRED if vector/where active
//!   "where": [{"property":"act","op":"eq","value":2}],  // optional structured prefilter (intersect)
//!   "legs": ["vector","keyword"],       // optional; default = whichever inputs are present
//!   "k_per_leg": 60,                     // candidates each leg contributes pre-fusion
//!   "limit": 20,                          // final cap
//!   "weights": {"vector":1.0,"keyword":1.0}  // optional per-leg RRF weights
//! }
//! → { "hits": [ { "node_id", "node_type", "score",
//!                 "legs": { "vector": {"rank":2,"score":0.81},
//!                           "keyword": {"rank":1,"score":7.4} } } ] }
//! ```
//!
//! ## Fusion
//!
//! `RRF(n) = Σ_leg  weight_leg / (k_rrf + rank_leg(n))`, `k_rrf = 60`,
//! default per-leg `weight = 1.0`. Rank-based on purpose — immune to the
//! un-normalized embedding magnitudes that make raw-score fusion fragile,
//! so no score normalization is needed. Each leg already returns an
//! ordered list; fusion only needs the per-leg rank.
//!
//! ## Design decisions (see ISSUES #22)
//!
//! - **Structured = prefilter, not a fusion leg.** A property scan is
//!   unordered and has no honest rank, so `where` clauses define an
//!   *allowed set* each ranked leg retrieves within ([`nodes_scan::matching_node_ids`]).
//! - **Vector leg requires `node_type`.** HNSW indexes are per-type;
//!   fanning across mismatched-dim indexes would silently skip some
//!   (a no-silent-fallbacks violation), so it fails loud instead.
//! - The keyword leg is behind the `fulltext` build feature; a request
//!   that asks for it in a build without the feature returns 501, exactly
//!   like `search:text`. A vector leg (optionally with a `where` prefilter)
//!   needs no feature and succeeds in any build.
//! - At least one *ranked* leg is required (`query` and/or `query_vector`).
//!   A pure `where` filter with no ranked leg is a 400 — that is what
//!   `nodes:scan` is for; the prefilter only constrains the ranked legs.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use dynograph_storage::StorageEngine;
use dynograph_vector::HnswIndex;

use crate::nodes_scan::{self, WhereClause};
use crate::registry::RegistryError;
use crate::util::validate_embedding_values;
use crate::validation::validate_limit;

/// RRF constant. 60 is the value from the original Cormack et al. paper
/// and the de-facto default; it damps the contribution of low ranks so a
/// single leg can't dominate on rank-1 alone.
const K_RRF: f32 = 60.0;

/// Over-fetch ceiling for the keyword leg when a structured prefilter is
/// active: BM25 returns its top-k *before* the prefilter intersects, so we
/// fetch up to this cap and intersect after. The precise fix is a filtered
/// full-text search; tracked as a follow-up.
#[cfg(feature = "fulltext")]
const KEYWORD_PREFILTER_FETCH: usize = crate::validation::MAX_LIMIT;

fn default_k_per_leg() -> usize {
    60
}

fn default_hybrid_limit() -> usize {
    20
}

/// Which retrieval legs a request targets. Used both for the optional
/// explicit `legs` selector and to key the per-leg breakdown in a hit.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LegName {
    Vector,
    Keyword,
}

/// Optional per-leg RRF weights. Omitted legs default to `1.0`.
#[derive(Debug, Deserialize, Default, ToSchema)]
// In a build without `fulltext` the keyword weight is never read (the
// keyword leg 501s before fusion), but it's part of the published wire
// contract, so the "unread field" lint is a false positive there.
#[cfg_attr(not(feature = "fulltext"), allow(dead_code))]
pub(crate) struct LegWeights {
    pub vector: Option<f32>,
    pub keyword: Option<f32>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct SearchHybridBody {
    /// Keyword/BM25 leg input. Tokenized like `search:text` (AND
    /// semantics, no query grammar). Omit for vector-only search.
    #[serde(default)]
    pub query: Option<String>,
    /// Vector leg input. Foundation does **not** embed — the caller
    /// supplies the query vector (sidecar's job). Omit for keyword-only.
    #[serde(default)]
    pub query_vector: Option<Vec<f32>>,
    /// Type filter applied to every active leg. Required when the vector
    /// leg or a structured `where` prefilter is active (both are
    /// per-type); optional for keyword-only.
    #[serde(default)]
    pub node_type: Option<String>,
    /// Optional structured prefilter (intersect): only nodes matching all
    /// clauses survive in every leg. Same clause grammar as `nodes:scan`.
    #[serde(default, rename = "where")]
    pub clauses: Vec<WhereClause>,
    /// Restrict to a subset of legs. Default = whichever inputs are
    /// present. Naming a leg whose input is absent is a 400.
    #[serde(default)]
    pub legs: Option<Vec<LegName>>,
    /// Candidates each leg contributes pre-fusion (1..=MAX_LIMIT).
    // `usize` advertises `minimum: 0`, contradicting the `validate_limit`
    // 1..=MAX_LIMIT bound; pin it. `10_000` mirrors `validation::MAX_LIMIT`
    // (utoipa `#[schema]` can't reference a const — keep in sync by hand).
    #[schema(minimum = 1, maximum = 10_000)]
    #[serde(default = "default_k_per_leg")]
    pub k_per_leg: usize,
    /// Final result cap after fusion (1..=MAX_LIMIT).
    #[schema(minimum = 1, maximum = 10_000)]
    #[serde(default = "default_hybrid_limit")]
    pub limit: usize,
    /// Optional per-leg RRF weights (each leg defaults to 1.0).
    #[serde(default)]
    pub weights: LegWeights,
}

/// One leg's contribution to a fused hit.
#[derive(Debug, Serialize, Clone, Copy, ToSchema)]
pub(crate) struct HybridLegInfo {
    /// 1-based rank within that leg's ranked output.
    // `usize` would advertise `minimum: 0` in the generated spec; the rank
    // is 1-based, so pin it lest clients infer 0-based ranks.
    #[schema(minimum = 1)]
    pub rank: usize,
    /// That leg's native score (BM25 for keyword, similarity for vector) —
    /// echoed for transparency; the fused `score` is what orders hits.
    pub score: f32,
}

/// Per-leg breakdown attached to each hit. Only legs that ranked the node
/// are present.
#[derive(Debug, Serialize, Default, ToSchema)]
pub(crate) struct HybridLegBreakdown {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<HybridLegInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyword: Option<HybridLegInfo>,
}

/// One fused result, ordered by RRF `score` (highest first).
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct HybridHit {
    pub node_id: String,
    pub node_type: String,
    /// Fused Reciprocal-Rank-Fusion score.
    pub score: f32,
    pub legs: HybridLegBreakdown,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct SearchHybridResponse {
    pub hits: Vec<HybridHit>,
}

/// One leg's ranked output, fed to [`rrf_fuse`]. `hits` is in rank order
/// (best first).
pub(crate) struct RankedLeg {
    pub name: LegName,
    pub weight: f32,
    pub hits: Vec<LegHit>,
}

/// A single ranked entry from a leg, before fusion.
pub(crate) struct LegHit {
    pub node_id: String,
    pub node_type: String,
    pub score: f32,
}

/// Per-node fusion accumulator. Keyed externally by `(node_type, node_id)`,
/// so the type lives in the map key, not here. `Default` gives the zero
/// state for every field except `best_rank`, which the insert overrides
/// with `usize::MAX`.
#[derive(Default)]
struct Fused {
    score: f32,
    /// Best (lowest) rank this node achieved in any leg — the first
    /// tie-break key.
    best_rank: usize,
    vector: Option<HybridLegInfo>,
    keyword: Option<HybridLegInfo>,
}

/// Reciprocal-Rank-Fuse the legs into a single ranked list. Pure: no I/O,
/// fully determined by its inputs.
///
/// `score = Σ_leg weight_leg / (k_rrf + rank_leg(node))`. Hits are ordered
/// by fused score (desc), then best single-leg rank (asc), then
/// `(node_type, node_id)` (asc) so the result is fully deterministic.
/// Truncated to `limit`.
///
/// A node's identity is `(node_type, node_id)`, not `node_id` alone — the
/// keyword leg with no `node_type` filter can return hits from several
/// types, and two types may legitimately share a `node_id`. Keying by the
/// pair keeps those distinct while still merging the *same* node seen by
/// both legs (a vector hit and a keyword hit agree on the type, since the
/// vector leg is always type-scoped).
pub(crate) fn rrf_fuse(legs: &[RankedLeg], k_rrf: f32, limit: usize) -> Vec<HybridHit> {
    let mut fused: HashMap<(String, String), Fused> = HashMap::new();
    for leg in legs {
        for (i, hit) in leg.hits.iter().enumerate() {
            let rank = i + 1;
            let contribution = leg.weight / (k_rrf + rank as f32);
            let entry = fused
                .entry((hit.node_type.clone(), hit.node_id.clone()))
                .or_insert_with(|| Fused {
                    best_rank: usize::MAX,
                    ..Default::default()
                });
            entry.score += contribution;
            entry.best_rank = entry.best_rank.min(rank);
            let info = HybridLegInfo {
                rank,
                score: hit.score,
            };
            match leg.name {
                LegName::Vector => entry.vector = Some(info),
                LegName::Keyword => entry.keyword = Some(info),
            }
        }
    }

    let mut entries: Vec<((String, String), Fused)> = fused.into_iter().collect();
    entries.sort_by(|(a_key, a), (b_key, b)| {
        // Scores are finite (positive weight / positive denom), so the
        // partial_cmp never returns None; the fallback keeps it total.
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.best_rank.cmp(&b.best_rank))
            .then(a_key.cmp(b_key))
    });
    entries.truncate(limit);
    entries
        .into_iter()
        .map(|((node_type, node_id), f)| HybridHit {
            node_id,
            node_type,
            score: f.score,
            legs: HybridLegBreakdown {
                vector: f.vector,
                keyword: f.keyword,
            },
        })
        .collect()
}

/// Resolve a per-leg weight, failing loud on a non-positive / non-finite
/// override (a silent leg-nullifying weight would mask intent).
fn resolve_weight(provided: Option<f32>, leg: &str) -> Result<f32, RegistryError> {
    let w = provided.unwrap_or(1.0);
    if !w.is_finite() || w <= 0.0 {
        return Err(RegistryError::BadRequest(format!(
            "weights.{leg} must be a finite number > 0, got {w}"
        )));
    }
    Ok(w)
}

pub(crate) fn run(
    engine: &StorageEngine,
    indexes: &HashMap<String, HnswIndex>,
    graph_id: &str,
    body: SearchHybridBody,
) -> Result<SearchHybridResponse, RegistryError> {
    validate_limit(body.limit, "limit")?;
    validate_limit(body.k_per_leg, "k_per_leg")?;

    // A present-but-blank input is a client slip, not a leg toggle — 400
    // rather than silently treating the leg as absent.
    if let Some(q) = &body.query
        && q.trim().is_empty()
    {
        return Err(RegistryError::BadRequest(
            "query must be non-empty when provided (omit it for vector-only search)".to_string(),
        ));
    }
    if let Some(v) = &body.query_vector
        && v.is_empty()
    {
        return Err(RegistryError::BadRequest(
            "query_vector must be non-empty when provided (omit it for keyword-only search)"
                .to_string(),
        ));
    }
    let keyword_input = body.query.is_some();
    let vector_input = body.query_vector.is_some();

    // Resolve active legs: an explicit `legs` selector restricts to the
    // named legs (each must have its input present); otherwise every leg
    // whose input is present runs.
    let (vector_active, keyword_active) = match &body.legs {
        Some(selected) => {
            let mut vector = false;
            let mut keyword = false;
            for leg in selected {
                match leg {
                    LegName::Vector if !vector_input => {
                        return Err(RegistryError::BadRequest(
                            "leg `vector` requested but `query_vector` is absent".to_string(),
                        ));
                    }
                    LegName::Keyword if !keyword_input => {
                        return Err(RegistryError::BadRequest(
                            "leg `keyword` requested but `query` is absent".to_string(),
                        ));
                    }
                    LegName::Vector => vector = true,
                    LegName::Keyword => keyword = true,
                }
            }
            (vector, keyword)
        }
        None => (vector_input, keyword_input),
    };
    if !vector_active && !keyword_active {
        return Err(RegistryError::BadRequest(
            "supply `query` and/or `query_vector` (and select at least one leg)".to_string(),
        ));
    }

    let has_prefilter = !body.clauses.is_empty();
    let node_type = body
        .node_type
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    if (vector_active || has_prefilter) && node_type.is_none() {
        return Err(RegistryError::BadRequest(
            "`node_type` is required when the vector leg or a structured `where` prefilter is active"
                .to_string(),
        ));
    }

    // Structured prefilter → the allowed node-id set every leg intersects
    // against. Clause validation (indexed-property + value-shape) happens
    // here, so an empty result is an honest "no matches", not a masked typo.
    let allowed: Option<HashSet<String>> = if has_prefilter {
        let nt = node_type.expect("node_type is required when a prefilter is active");
        Some(nodes_scan::matching_node_ids(
            engine,
            graph_id,
            nt,
            &body.clauses,
        )?)
    } else {
        None
    };
    let keep = |id: &str| allowed.as_ref().is_none_or(|set| set.contains(id));

    let mut legs: Vec<RankedLeg> = Vec::new();

    if vector_active {
        let nt = node_type.expect("node_type is required when the vector leg is active");
        if !engine.schema().node_types.contains_key(nt) {
            return Err(RegistryError::BadRequest(format!(
                "unknown node type: {nt}"
            )));
        }
        let embedding = body
            .query_vector
            .as_deref()
            .expect("vector_active implies query_vector is present");
        // A degenerate query (non-finite / zero magnitude) scores 0.0
        // against everything — reject it loudly, like `/similar`.
        validate_embedding_values(embedding)?;
        let hits = match indexes.get(nt) {
            // No embedding has ever been set for this type — honest empty
            // leg (matches `/similar`), not an error.
            None => Vec::new(),
            Some(index) => {
                if index.dim() != embedding.len() {
                    return Err(RegistryError::EmbeddingDimMismatch {
                        node_type: nt.to_string(),
                        expected: index.dim(),
                        actual: embedding.len(),
                    });
                }
                index
                    .search_filtered(embedding, body.k_per_leg, keep)
                    .into_iter()
                    .map(|sr| LegHit {
                        node_id: sr.id.to_string(),
                        node_type: nt.to_string(),
                        score: sr.score,
                    })
                    .collect()
            }
        };
        legs.push(RankedLeg {
            name: LegName::Vector,
            weight: resolve_weight(body.weights.vector, "vector")?,
            hits,
        });
    }

    if keyword_active {
        #[cfg(feature = "fulltext")]
        {
            let query = body
                .query
                .as_deref()
                .expect("keyword_active implies query is present");
            // Fail loud on a node_type filter that can never match — an
            // unknown type, or a known type with no `fulltext` property
            // (shared with `search:text`).
            if let Some(nt) = node_type {
                crate::validation::validate_fulltext_searchable(engine.schema(), nt)?;
            }
            // BM25 ranks before the prefilter intersects, so over-fetch
            // when a prefilter is active, then intersect and take k.
            let fetch = if allowed.is_some() {
                KEYWORD_PREFILTER_FETCH
            } else {
                body.k_per_leg
            };
            let hits = engine
                .search_fulltext(graph_id, query, node_type, fetch)?
                .into_iter()
                .filter(|h| keep(&h.node_id))
                .take(body.k_per_leg)
                .map(|h| LegHit {
                    node_id: h.node_id,
                    node_type: h.node_type,
                    score: h.score,
                })
                .collect();
            legs.push(RankedLeg {
                name: LegName::Keyword,
                weight: resolve_weight(body.weights.keyword, "keyword")?,
                hits,
            });
        }
        #[cfg(not(feature = "fulltext"))]
        {
            return Err(RegistryError::NotImplemented(
                "the keyword leg requires the `fulltext` build feature \
                 (compile with --features fulltext); omit `query` for vector-only hybrid search"
                    .to_string(),
            ));
        }
    }

    Ok(SearchHybridResponse {
        hits: rrf_fuse(&legs, K_RRF, body.limit),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leg(name: LegName, ids_scores: &[(&str, f32)]) -> RankedLeg {
        RankedLeg {
            name,
            weight: 1.0,
            hits: ids_scores
                .iter()
                .map(|(id, s)| LegHit {
                    node_id: (*id).to_string(),
                    node_type: "T".to_string(),
                    score: *s,
                })
                .collect(),
        }
    }

    /// `1/(k+rank)` for an easy hand-check against the expected math.
    fn rrf(rank: usize) -> f32 {
        1.0 / (K_RRF + rank as f32)
    }

    #[test]
    fn single_leg_passes_through_in_rank_order() {
        let legs = vec![leg(LegName::Vector, &[("a", 0.9), ("b", 0.5), ("c", 0.1)])];
        let hits = rrf_fuse(&legs, K_RRF, 10);
        assert_eq!(
            hits.iter().map(|h| h.node_id.as_str()).collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
        // Rank-1 score is exactly 1/(60+1).
        assert!((hits[0].score - rrf(1)).abs() < 1e-7);
        assert_eq!(hits[0].legs.vector.unwrap().rank, 1);
        assert!(hits[0].legs.keyword.is_none());
    }

    #[test]
    fn overlap_across_legs_boosts_the_common_node() {
        // "b" appears in BOTH legs (vector rank-2 + keyword rank-1); "a" and
        // "c" each appear in a single leg. b's fused score 1/62 + 1/61 beats
        // a's lone 1/61, so the cross-leg node rises to the top even though
        // it was never rank-1 in the vector leg.
        let legs = vec![
            leg(LegName::Vector, &[("a", 0.9), ("b", 0.8), ("c", 0.7)]),
            leg(LegName::Keyword, &[("b", 7.0)]),
        ];
        let hits = rrf_fuse(&legs, K_RRF, 10);
        assert_eq!(hits[0].node_id, "b");
        let b_expected = rrf(2) + rrf(1);
        assert!((hits[0].score - b_expected).abs() < 1e-7);
        // b carries both legs' breakdown; the single-leg "a" follows.
        let b_legs = &hits[0].legs;
        assert_eq!(b_legs.vector.unwrap().rank, 2);
        assert_eq!(b_legs.keyword.unwrap().rank, 1);
        assert_eq!(b_legs.keyword.unwrap().score, 7.0);
        assert_eq!(hits[1].node_id, "a");
        assert!(hits[1].legs.keyword.is_none());
    }

    #[test]
    fn distinct_node_types_sharing_a_node_id_are_not_merged() {
        // The keyword leg with no node_type filter can return two different
        // types that happen to share a node_id. They are distinct nodes
        // (identity is (node_type, node_id)) and must NOT collapse into one
        // fused entry — doing so would corrupt scores and drop a real hit.
        let legs = vec![RankedLeg {
            name: LegName::Keyword,
            weight: 1.0,
            hits: vec![
                LegHit {
                    node_id: "x".into(),
                    node_type: "A".into(),
                    score: 5.0,
                },
                LegHit {
                    node_id: "x".into(),
                    node_type: "B".into(),
                    score: 3.0,
                },
            ],
        }];
        let hits = rrf_fuse(&legs, K_RRF, 10);
        assert_eq!(hits.len(), 2, "must not merge same id / different type");
        // Rank-1 (type A) outscores rank-2 (type B); both retain their type.
        assert_eq!(
            (hits[0].node_type.as_str(), hits[0].node_id.as_str()),
            ("A", "x")
        );
        assert_eq!(
            (hits[1].node_type.as_str(), hits[1].node_id.as_str()),
            ("B", "x")
        );
    }

    #[test]
    fn weights_scale_a_legs_contribution() {
        // Same ranks, but the keyword leg is weighted 3x, so its rank-1
        // node "b" outscores the vector rank-1 node "a".
        let legs = vec![
            RankedLeg {
                name: LegName::Vector,
                weight: 1.0,
                hits: vec![LegHit {
                    node_id: "a".into(),
                    node_type: "T".into(),
                    score: 0.9,
                }],
            },
            RankedLeg {
                name: LegName::Keyword,
                weight: 3.0,
                hits: vec![LegHit {
                    node_id: "b".into(),
                    node_type: "T".into(),
                    score: 7.0,
                }],
            },
        ];
        let hits = rrf_fuse(&legs, K_RRF, 10);
        assert_eq!(hits[0].node_id, "b");
        assert!((hits[0].score - 3.0 * rrf(1)).abs() < 1e-7);
        assert!((hits[1].score - rrf(1)).abs() < 1e-7);
    }

    #[test]
    fn tie_break_prefers_best_rank_then_node_id() {
        // Two nodes each appear once at rank 1 in different legs → equal
        // fused score and equal best_rank. Tie-break falls to node_id asc.
        let legs = vec![
            leg(LegName::Vector, &[("z", 0.9)]),
            leg(LegName::Keyword, &[("a", 9.0)]),
        ];
        let hits = rrf_fuse(&legs, K_RRF, 10);
        assert_eq!(hits[0].node_id, "a");
        assert_eq!(hits[1].node_id, "z");

        // Now give "z" a better best_rank (rank 1) than "a" (rank 2) at the
        // same fused score: best_rank wins before node_id.
        let legs = vec![
            leg(LegName::Vector, &[("z", 0.9), ("a", 0.5)]),
            leg(LegName::Keyword, &[("a", 9.0), ("z", 1.0)]),
        ];
        let hits = rrf_fuse(&legs, K_RRF, 10);
        // Both have score 1/61 + 1/62; "z" has best_rank 1, "a" best_rank 1
        // too (a is rank-1 in keyword). Equal → node_id asc → "a" first.
        assert_eq!(hits[0].node_id, "a");
    }

    #[test]
    fn limit_truncates_after_fusion() {
        let legs = vec![leg(LegName::Vector, &[("a", 0.9), ("b", 0.5), ("c", 0.1)])];
        let hits = rrf_fuse(&legs, K_RRF, 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].node_id, "a");
        assert_eq!(hits[1].node_id, "b");
    }

    #[test]
    fn empty_legs_fuse_to_nothing() {
        assert!(rrf_fuse(&[], K_RRF, 10).is_empty());
        let legs = vec![leg(LegName::Vector, &[])];
        assert!(rrf_fuse(&legs, K_RRF, 10).is_empty());
    }

    #[test]
    fn resolve_weight_rejects_non_positive_and_non_finite() {
        assert!(resolve_weight(None, "vector").is_ok());
        assert_eq!(resolve_weight(Some(2.5), "vector").unwrap(), 2.5);
        assert!(resolve_weight(Some(0.0), "vector").is_err());
        assert!(resolve_weight(Some(-1.0), "keyword").is_err());
        assert!(resolve_weight(Some(f32::NAN), "vector").is_err());
        assert!(resolve_weight(Some(f32::INFINITY), "vector").is_err());
    }
}
