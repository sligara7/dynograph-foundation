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
//! ## Aliases (both directions)
//!
//! - **Incoming**: `incoming_aliases: [String]` (wire alias: `aliases`)
//!   are alternate names for the query entity. The primary name is
//!   resolved first; if it would create, each alias is resolved in turn
//!   and the first merge wins (`EntityResolver::resolve_with_aliases` —
//!   the threshold logic stays in the resolver crate, per the original
//!   design note). Catches a query like "Neo" arriving with alias
//!   "Thomas Anderson" matching an existing "Thomas Anderson" node.
//! - **Stored**: each candidate node's `aliases` property contributes
//!   additional `(id, alias)` pairs to the candidate list, so an
//!   incoming primary name can merge into a node it only matches via a
//!   stored alias. Both a `List` of strings and a JSON-array-encoded
//!   string (clients that serialize list-valued properties to JSON
//!   strings) are accepted.
//!
//! Before v0.9.x neither direction existed over HTTP: an
//! `incoming_aliases` field in the request was silently dropped by
//! serde, and stored aliases were not part of the candidate list.

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
    /// Alternate names for the query entity, tried (in order) when the
    /// primary `properties.name` doesn't merge. `aliases` is accepted as
    /// a wire alias for this field; sending BOTH spellings in one request
    /// is rejected as a duplicate field.
    #[serde(default, alias = "aliases")]
    pub incoming_aliases: Vec<String>,
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

/// Extract a node's stored aliases for the candidate list. Accepts a
/// `List` of strings or a JSON-array-encoded string (for clients that
/// serialize list-valued properties to JSON strings); anything else
/// yields no aliases. Non-string elements and empty strings are
/// dropped — in BOTH forms, so one malformed element (a stray null
/// from an extraction pipeline) can't silently void a node's entire
/// alias set.
fn stored_aliases(value: Option<&Value>) -> Vec<String> {
    let aliases: Vec<String> = match value {
        Some(Value::List(items)) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        Some(Value::String(s)) => serde_json::from_str::<Vec<serde_json::Value>>(s)
            .unwrap_or_default()
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    };
    aliases.into_iter().filter(|a| !a.is_empty()).collect()
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
    // unreachable in practice. Each candidate's stored `aliases` add
    // further (id, alias) pairs so an incoming name can merge into a
    // node it only matches via an alias.
    //
    // ALL primary-name pairs precede ALL alias pairs: the resolver's
    // sort is stable, so on a fuzzy-score tie an entity's primary name
    // beats another entity's alias — a node storing an alias equal to
    // some other node's primary name must not hijack that node's
    // exact-name merges.
    let mut owned_pairs: Vec<(String, String)> = Vec::new();
    let mut alias_pairs: Vec<(String, String)> = Vec::new();
    for n in &candidates {
        match n.properties.get("name").and_then(|v| v.as_str()) {
            Some(name) => owned_pairs.push((n.node_id.clone(), name.to_string())),
            None => {
                tracing::debug!(
                    node_type = %req.node_type,
                    node_id = %n.node_id,
                    "skipping resolution candidate: no string `name` property"
                );
                continue;
            }
        }
        for alias in stored_aliases(n.properties.get("aliases")) {
            alias_pairs.push((n.node_id.clone(), alias));
        }
    }
    owned_pairs.append(&mut alias_pairs);
    let pairs: Vec<(&str, &str)> = owned_pairs
        .iter()
        .map(|(id, name)| (id.as_str(), name.as_str()))
        .collect();

    // ---- Resolve ----

    let alias_refs: Vec<&str> = req.incoming_aliases.iter().map(String::as_str).collect();
    let resolver = EntityResolver::from_config(&resolution_config);
    let (result, _candidates) = resolver.resolve_with_aliases(
        &query_name,
        &alias_refs,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_aliases_accepts_list_of_strings() {
        let v = Value::List(vec![
            Value::String("The Cartographer".into()),
            Value::String("".into()), // empty dropped
            Value::Int(7),            // non-string dropped
        ]);
        assert_eq!(stored_aliases(Some(&v)), vec!["The Cartographer".to_string()]);
    }

    #[test]
    fn stored_aliases_accepts_json_encoded_string() {
        let v = Value::String(r#"["The Cartographer", "M. Sandgrove", ""]"#.into());
        assert_eq!(
            stored_aliases(Some(&v)),
            vec!["The Cartographer".to_string(), "M. Sandgrove".to_string()]
        );
    }

    #[test]
    fn stored_aliases_json_string_tolerates_non_string_elements() {
        // One malformed element must not void the whole alias set —
        // same element-level tolerance as the List form.
        let v = Value::String(r#"["The Cartographer", null, 7]"#.into());
        assert_eq!(stored_aliases(Some(&v)), vec!["The Cartographer".to_string()]);
    }

    #[test]
    fn stored_aliases_ignores_garbage() {
        assert!(stored_aliases(None).is_empty());
        assert!(stored_aliases(Some(&Value::String("not json".into()))).is_empty());
        assert!(stored_aliases(Some(&Value::Int(3))).is_empty());
        assert!(stored_aliases(Some(&Value::Map(Default::default()))).is_empty());
    }
}
