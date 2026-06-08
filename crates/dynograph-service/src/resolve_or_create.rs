//! POST /v1/graphs/{id}/resolve-or-create — fuzzy/vector entity
//! resolution with create-on-miss semantics.
//!
//! Closes audit primitive #2 (2026-05-04 audit). Exposes
//! the existing `dynograph-resolution` crate (token_sort_ratio +
//! jaro_winkler + cosine-similarity tiebreaker) over HTTP. An
//! LLM extraction pipeline funnels every entity through this gate
//! today via an embedded resolve-or-create call; after migration,
//! every entity resolution is one HTTP call.
//!
//! ## Wire shape
//!
//! ```json
//! POST /v1/graphs/{id}/resolve-or-create
//! {
//!   "node_type": "Character",
//!   "properties": {"name": "Mira Sandgrove", ...},
//!   "embedding": [0.1, 0.2, ...],         // optional
//!   "scope": {"prop": "story_id", "value": "X"}   // optional
//! }
//! → {"id": "...", "was_created": true|false, "match_kind": "auto_merge|vector_merge|created_new"}
//! ```
//!
//! `properties.name` (string) is **required** — it's the query string
//! the resolver compares against existing nodes' `name` properties.
//! Same convention candidates use, so a unified contract: a node's
//! "name" lives at `properties.name`.
//!
//! ## Validation order (all 400 on failure, no state changes)
//!
//! 1. `node_type` exists in the schema
//! 2. The schema declares a `ResolutionConfig` for this node_type
//!    (no silent fallback to defaults — explicit-is-better)
//! 3. `properties.name` is present and is a string
//! 4. If `scope` is set: `scope.prop` is declared as `indexed: true`
//!    on the node type (otherwise `scan_nodes_by_property` would
//!    silently return empty → resolution would always `CreateNew`,
//!    masking a misconfiguration as "everything was a new entity")
//! 5. If `embedding` is set and an HNSW index already exists for the
//!    type: dim must match (else `EmbeddingDimMismatch` 400)
//!
//! ## Atomicity on `CreateNew`
//!
//! `engine.create_node` + (optional) `engine.set_embedding` run
//! sequentially under the caller's write lock. They could be wrapped
//! in `begin_batch` / `commit_batch` for atomicity (v0.5.5+ supports
//! read-your-own-writes within a batch, so `set_embedding`'s node
//! existence check would see the buffered create) — TODO follow-up.
//! For now every failure mode that could leave a torn state (node
//! created but no embedding) is pushed into pre-flight validation:
//! empty embedding, dim mismatch, missing/non-string name, unknown
//! type, non-indexed scope prop. If a pure storage-I/O failure (e.g.
//! disk) still tears the pair, the caller can retry — resolution
//! will auto-merge to the existing node and `set_embedding` again.
//! Same contract the single-call create_node + set_embedding pair
//! has had since v0.3.0.
//!
//! ## Aliases
//!
//! The audit memo's wire-shape sketch included an `aliases: [String]`
//! field; foundation's underlying resolver doesn't natively support
//! multi-name queries. Implementing aliases by orchestrating multiple
//! `resolve()` calls would re-implement the threshold logic at the
//! HTTP layer, which is a smell. Out of scope for v1 — extend the
//! resolver crate if a real workload needs them.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use dynograph_core::{ResolutionConfig, Value};
use dynograph_resolution::{EntityResolver, ResolutionResult};
use dynograph_storage::StorageEngine;
use dynograph_vector::HnswIndex;

use crate::registry::RegistryError;
use crate::validation::validate_indexed_property;

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct ResolveOrCreateRequest {
    pub node_type: String,
    #[serde(default)]
    #[schema(value_type = Object)]
    pub properties: HashMap<String, Value>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub scope: Option<ScopeFilter>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct ScopeFilter {
    pub prop: String,
    #[schema(value_type = Object)]
    pub value: Value,
}

#[derive(Debug, Serialize, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MatchKind {
    /// Resolver returned `AutoMerge` — fuzzy score ≥ auto_merge_threshold.
    AutoMerge,
    /// Resolver returned `VectorMerge` — fuzzy zone hit + embedding
    /// similarity ≥ vector_threshold.
    VectorMerge,
    /// Resolver returned `CreateNew` — caller's payload created a
    /// fresh node with a fresh UUIDv4 id.
    CreatedNew,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ResolveOrCreateResponse {
    pub id: String,
    pub was_created: bool,
    pub match_kind: MatchKind,
}

/// Pre-flight checks + resolution + (optional) create. Caller must
/// hold the `with_state_write` lock — we touch storage AND the HNSW
/// indexes map in lockstep.
pub(crate) fn run(
    engine: &mut StorageEngine,
    indexes: &mut HashMap<String, HnswIndex>,
    graph_id: &str,
    req: ResolveOrCreateRequest,
) -> Result<ResolveOrCreateResponse, RegistryError> {
    // ---- Pre-flight validation (no state changes past this block) ----

    // (1+2) node_type exists AND declares a ResolutionConfig.
    let resolution_config: ResolutionConfig = {
        let nt = engine
            .schema()
            .node_types
            .get(&req.node_type)
            .ok_or_else(|| {
                RegistryError::BadRequest(format!("unknown node type: {}", req.node_type))
            })?;
        nt.resolution
            .as_ref()
            .ok_or_else(|| {
                RegistryError::BadRequest(format!(
                    "node type {} has no entity resolution configured (schema must declare `resolution:`)",
                    req.node_type
                ))
            })?
            .clone()
    };

    // (3) properties.name present + string.
    let query_name = req
        .properties
        .get("name")
        .ok_or_else(|| {
            RegistryError::BadRequest(
                "properties.name is required (used as the resolution query)".to_string(),
            )
        })?
        .as_str()
        .ok_or_else(|| RegistryError::BadRequest("properties.name must be a string".to_string()))?
        .to_string();

    // (4) If scoped, the scope prop must be indexed — otherwise
    // scan_nodes_by_property silently returns empty.
    if let Some(ref scope) = req.scope {
        validate_indexed_property(engine.schema(), &req.node_type, &scope.prop, "scope")?;
    }

    // (5) Embedding pre-flight: non-empty + finite/non-degenerate + dim
    // matches existing index. All validations land 400 BEFORE any writes
    // so a CreateNew dispatch can't tear (node created, embedding rejected).
    if let Some(ref emb) = req.embedding {
        if emb.is_empty() {
            return Err(RegistryError::BadRequest(
                "embedding must be non-empty".to_string(),
            ));
        }
        // Degenerate embedding (non-finite / zero magnitude) → a 0.0
        // vector score that quietly falls below `vector_threshold` and
        // masquerades as a legitimate CreateNew. Reject before any write.
        crate::util::validate_embedding_values(emb)?;
        if let Some(idx) = indexes.get(&req.node_type)
            && idx.dim() != emb.len()
        {
            return Err(RegistryError::EmbeddingDimMismatch {
                node_type: req.node_type.clone(),
                expected: idx.dim(),
                actual: emb.len(),
            });
        }
    }

    // ---- Fetch candidates ----

    let candidates = match &req.scope {
        Some(s) => engine.scan_nodes_by_property(graph_id, &req.node_type, &s.prop, &s.value)?,
        None => engine.scan_nodes(graph_id, &req.node_type)?,
    };

    // (id, name) pairs — silently skip candidates without a string
    // `name` property. Logged at debug because the alternative is to
    // fail the whole call when one node in the corpus is malformed,
    // which is hostile. A schema where `name` is required + string
    // (the only sane setup for name-based resolution) makes this
    // unreachable in practice.
    let pairs: Vec<(&str, &str)> = candidates
        .iter()
        .filter_map(
            |n| match n.properties.get("name").and_then(|v| v.as_str()) {
                Some(name) => Some((n.node_id.as_str(), name)),
                None => {
                    tracing::debug!(
                        node_type = %req.node_type,
                        node_id = %n.node_id,
                        "skipping resolution candidate: no string `name` property"
                    );
                    None
                }
            },
        )
        .collect();

    // ---- Resolve ----

    let resolver = EntityResolver::from_config(&resolution_config);
    let (result, _candidates) = resolver.resolve(
        &query_name,
        &pairs,
        req.embedding.as_deref(),
        indexes.get(&req.node_type),
    );

    // ---- Dispatch ----

    match result {
        ResolutionResult::AutoMerge { candidate } => Ok(ResolveOrCreateResponse {
            id: candidate,
            was_created: false,
            match_kind: MatchKind::AutoMerge,
        }),
        ResolutionResult::VectorMerge { candidate } => Ok(ResolveOrCreateResponse {
            id: candidate,
            was_created: false,
            match_kind: MatchKind::VectorMerge,
        }),
        ResolutionResult::CreateNew => {
            // Sequential: create_node, then set_embedding (if any),
            // then HNSW insert. See module doc on `## Atomicity` for
            // the torn-state window and follow-up batch-wrap plan.
            let node_id = Uuid::new_v4().to_string();
            let node_type = req.node_type.clone();
            let properties = req.properties;
            let embedding = req.embedding;

            engine.create_node(graph_id, &node_type, &node_id, properties)?;

            if let Some(emb) = embedding {
                engine.set_embedding(graph_id, &node_type, &node_id, &emb)?;
                let idx = match indexes.get_mut(&node_type) {
                    Some(i) => i,
                    None => indexes
                        .entry(node_type.clone())
                        .or_insert_with(|| HnswIndex::with_dim(emb.len())),
                };
                idx.insert(&node_id, &emb);
            }

            Ok(ResolveOrCreateResponse {
                id: node_id,
                was_created: true,
                match_kind: MatchKind::CreatedNew,
            })
        }
    }
}
