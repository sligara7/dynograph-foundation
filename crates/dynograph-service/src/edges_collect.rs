//! POST /v1/graphs/{id}/edges:collect — fan-out edge collection
//! across a typed source set.
//!
//! Closes audit primitive #3 (2026-05-04 audit). Replaces a
//! `collect_*_edges` master pattern that walks
//! N entity types × M nodes × K edge types via per-node
//! `outgoing_edges` — hundreds of round-trips per call today; one
//! HTTP call after migration. Used by 13+ knowledge-graph routes
//! plus the projection step in pagerank/leiden/shortest-path.
//!
//! ## Wire shape
//!
//! ```json
//! POST /v1/graphs/{id}/edges:collect
//! {
//!   "source": {
//!     "type": "*" | "Character" | ["Character", "Event"],
//!     "filter": {"prop": "story_id", "value": "X"}    // optional
//!   },
//!   "edge_types": ["MENTIONS", "EXPLORES"],            // required, non-empty
//!   "format": "edges" | "adjacency",                   // optional, default "edges"
//!   "resolve_target": false,                            // optional, default false
//!   "limit": 200                                        // required, 1..=10_000
//! }
//! ```
//!
//! ## Validation (all 400, all pre-flight)
//!
//! - `edge_types` non-empty + every entry known to schema
//! - `limit` in `1..=10_000`
//! - `source.type` names (single, list, or after wildcard expansion)
//!   all known to schema
//! - `source.filter.prop` declared as `indexed: true` on every
//!   covered source type (otherwise `scan_nodes_by_property` silently
//!   returns empty per-type → masked misconfiguration)
//!
//! ## Performance characteristics
//!
//! - With `source.filter`: O(M_filtered × K_outgoing) where M_filtered
//!   uses the indexed-property fast scan.
//! - Without filter, single source type: O(M_type × K_outgoing).
//! - Wildcard `source.type = "*"` without filter: O(N) over the entire
//!   graph (every node × outgoing scan). Adjacency CFs are
//!   node-keyed, not edge-type-keyed, so there's no edge-type-prefix
//!   shortcut. A future storage CF keyed
//!   `(graph_id, edge_type, from_id, to_id)` would enable true
//!   edge-type scans; out of scope until profiling proves the per-node
//!   walk is the bottleneck.
//!
//! Iteration order is unspecified — depends on HashMap iter order
//! across types and storage scan order within each type. Algorithm
//! consumers (pagerank/leiden) don't care about order; "show me a
//! few sample edges" callers shouldn't either with a `limit` cap.
//!
//! ## `resolve_target` and to_type discovery
//!
//! `StoredEdge` doesn't carry `to_type` (edge keys are
//! `(edge_type, from_id, to_id)` — type is validated at write time
//! and not preserved). When `resolve_target = true`, foundation
//! discovers each target's type via the schema's `EdgeTypeDef.to`
//! declaration:
//!
//! - `Single(t)` where `t != "*"` → 1 `get_node` lookup
//! - `Single("*")` (or `Multiple` containing `"*"`) → try every node
//!   type in the schema until a match
//! - `Multiple([t1, t2, ...])` → try each in order
//!
//! Cost is bounded by the schema (typically 1-3 candidate types per
//! edge type). Wildcard edge endpoints with `resolve_target = true`
//! pay the full per-edge fanout cost; document accordingly in the
//! consuming schema if it matters.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use dynograph_core::{EdgeEndpoint, Schema, Value};
use dynograph_storage::StorageEngine;

use crate::registry::RegistryError;
use crate::validation::{validate_indexed_property, validate_limit};

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct EdgesCollectRequest {
    pub source: SourceSpec,
    pub edge_types: Vec<String>,
    #[serde(default)]
    pub format: ResponseFormat,
    #[serde(default)]
    pub resolve_target: bool,
    pub limit: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct SourceSpec {
    #[serde(rename = "type")]
    pub type_filter: SourceTypeFilter,
    #[serde(default)]
    pub filter: Option<PropertyFilter>,
}

/// Source-type matcher. Untagged so the JSON can be either a string
/// (`"*"` or a single type name) or an array of names.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum SourceTypeFilter {
    Single(String),
    Multiple(Vec<String>),
}

impl SourceTypeFilter {
    /// Resolve to the concrete list of node-type names to scan, given
    /// the schema. `"*"` (anywhere) expands to every type in the
    /// schema. Validates that every named type exists.
    fn resolve<'a>(&'a self, schema: &'a Schema) -> Result<Vec<&'a str>, RegistryError> {
        let names: Vec<&str> = match self {
            SourceTypeFilter::Single(s) => vec![s.as_str()],
            SourceTypeFilter::Multiple(ss) => ss.iter().map(String::as_str).collect(),
        };
        let expanded = expand_node_type_wildcards(&names, schema);
        for n in &expanded {
            if !schema.node_types.contains_key(*n) {
                return Err(RegistryError::BadRequest(format!(
                    "source.type references unknown node type: {n}"
                )));
            }
        }
        Ok(expanded)
    }
}

#[derive(Debug, Deserialize, ToSchema)]
#[schema(as = CollectPropertyFilter)]
pub(crate) struct PropertyFilter {
    pub prop: String,
    #[schema(value_type = Object)]
    pub value: Value,
}

#[derive(Debug, Deserialize, Default, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ResponseFormat {
    #[default]
    Edges,
    Adjacency,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct CollectedEdge {
    pub edge_type: String,
    pub from_type: String,
    pub from_id: String,
    pub to_id: String,
    #[schema(value_type = Object)]
    pub properties: HashMap<String, Value>,
    /// Present only when `resolve_target = true` AND the target node
    /// was found. Caller can rely on `target.is_some()` ↔ resolution
    /// succeeded; absence with `resolve_target = true` means the edge
    /// points at a node id no schema-candidate type could find — a
    /// referential-integrity gap worth surfacing in caller logs (see
    /// the v0.5.1 batch.rs hazard documentation for one source of such
    /// orphans).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<TargetNode>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct AdjacencyEntry {
    pub edge_type: String,
    pub to_id: String,
    #[schema(value_type = Object)]
    pub properties: HashMap<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<TargetNode>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct TargetNode {
    pub node_type: String,
    pub node_id: String,
    #[schema(value_type = Object)]
    pub properties: HashMap<String, Value>,
}

/// Untagged so the JSON has the shape the caller asked for: an
/// `{"edges": [...], "truncated": ...}` object for `format = "edges"`,
/// an `{"adjacency": {...}, "truncated": ...}` object for
/// `format = "adjacency"`. Caller distinguishes by which field is
/// present (always exactly one).
#[derive(Debug, Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum EdgesCollectResponse {
    Edges {
        edges: Vec<CollectedEdge>,
        truncated: bool,
    },
    Adjacency {
        #[schema(value_type = Object)]
        adjacency: HashMap<String, Vec<AdjacencyEntry>>,
        truncated: bool,
    },
}

/// Resolve the `source.type`+`source.filter` pair to the concrete
/// list of source nodes, fan-out per-source `scan_outgoing_edges`,
/// filter to requested edge types, optionally resolve target nodes,
/// shape into the requested response format. Caller holds the
/// `with_engine_read` lock.
pub(crate) fn run(
    engine: &StorageEngine,
    graph_id: &str,
    req: EdgesCollectRequest,
) -> Result<EdgesCollectResponse, RegistryError> {
    // ---- Pre-flight validation ----

    if req.edge_types.is_empty() {
        return Err(RegistryError::BadRequest(
            "edge_types must be non-empty".to_string(),
        ));
    }
    validate_limit(req.limit, "limit")?;

    let schema = engine.schema();

    // Every requested edge_type must exist in the schema.
    for et in &req.edge_types {
        if !schema.edge_types.contains_key(et) {
            return Err(RegistryError::BadRequest(format!(
                "edge_types references unknown edge type: {et}"
            )));
        }
    }

    let source_types = req.source.type_filter.resolve(schema)?;

    // If a property filter is supplied, validate it's indexed on
    // EVERY covered source type. Mixed-indexed cases would silently
    // drop nodes from un-indexed types — same masked-misconfiguration
    // failure mode `/resolve-or-create` rejects.
    if let Some(ref f) = req.source.filter {
        for st in &source_types {
            validate_indexed_property(schema, st, &f.prop, "source.filter")?;
        }
    }

    let edge_type_set: std::collections::HashSet<&str> =
        req.edge_types.iter().map(String::as_str).collect();

    // Pre-resolve candidate target types per requested edge_type when
    // resolve_target is on. Without this, every matching edge would
    // re-walk `schema.edge_types[edge_type].to` in the inner loop —
    // O(edges) HashMap lookups for what's a fixed function of the
    // request. Surfaces non_exhaustive EdgeEndpoint variants up-front
    // (before any scans) too.
    let candidates_by_edge_type: HashMap<&str, Vec<&str>> = if req.resolve_target {
        req.edge_types
            .iter()
            .map(|et| {
                let edge_def = schema
                    .edge_types
                    .get(et)
                    .expect("edge_types validated above");
                let candidates = candidate_target_types(&edge_def.to, schema, et)?;
                Ok((et.as_str(), candidates))
            })
            .collect::<Result<_, RegistryError>>()?
    } else {
        HashMap::new()
    };

    // ---- Fan-out (single pass; branch on format inside the loop) ----

    let mut edges_acc: Vec<CollectedEdge> = Vec::new();
    let mut adj_acc: HashMap<String, Vec<AdjacencyEntry>> = HashMap::new();
    let mut count: usize = 0;
    let mut truncated = false;

    'outer: for source_type in &source_types {
        let source_nodes = match &req.source.filter {
            Some(f) => engine.scan_nodes_by_property(graph_id, source_type, &f.prop, &f.value)?,
            None => engine.scan_nodes(graph_id, source_type)?,
        };

        for node in source_nodes {
            let outgoing = engine.scan_outgoing_edges(graph_id, &node.node_id, None)?;
            for edge in outgoing {
                if !edge_type_set.contains(edge.edge_type.as_str()) {
                    continue;
                }

                let target = if req.resolve_target {
                    let candidates = candidates_by_edge_type
                        .get(edge.edge_type.as_str())
                        .expect("pre-cached for every requested edge_type");
                    fetch_target(engine, graph_id, candidates, &edge.to_id)?
                } else {
                    None
                };

                match req.format {
                    ResponseFormat::Edges => edges_acc.push(CollectedEdge {
                        edge_type: edge.edge_type,
                        from_type: (*source_type).to_string(),
                        from_id: edge.from_id,
                        to_id: edge.to_id,
                        properties: edge.properties,
                        target,
                    }),
                    ResponseFormat::Adjacency => {
                        adj_acc
                            .entry(edge.from_id)
                            .or_default()
                            .push(AdjacencyEntry {
                                edge_type: edge.edge_type,
                                to_id: edge.to_id,
                                properties: edge.properties,
                                target,
                            });
                    }
                }

                count += 1;
                if count >= req.limit {
                    truncated = true;
                    break 'outer;
                }
            }
        }
    }

    Ok(match req.format {
        ResponseFormat::Edges => EdgesCollectResponse::Edges {
            edges: edges_acc,
            truncated,
        },
        ResponseFormat::Adjacency => EdgesCollectResponse::Adjacency {
            adjacency: adj_acc,
            truncated,
        },
    })
}

/// Returns the candidate node-type names that an `EdgeEndpoint` can
/// connect to, expanding wildcards via the schema. `edge_type` is
/// only used for the error message in the non_exhaustive arm.
fn candidate_target_types<'a>(
    endpoint: &'a EdgeEndpoint,
    schema: &'a Schema,
    edge_type: &str,
) -> Result<Vec<&'a str>, RegistryError> {
    let names: Vec<&str> = match endpoint {
        EdgeEndpoint::Single(t) => vec![t.as_str()],
        EdgeEndpoint::Multiple(ts) => ts.iter().map(String::as_str).collect(),
        // EdgeEndpoint is #[non_exhaustive] in dynograph-core. If a
        // variant lands without updating this dispatch, fail loudly
        // here rather than panicking or silently widening the
        // candidate set. Recovery is either a foundation upgrade
        // (preferred) or rolling the schema back to a known variant.
        _ => {
            return Err(RegistryError::BadRequest(format!(
                "edge type {edge_type}: EdgeTypeDef.to uses an EdgeEndpoint variant unknown to this foundation build (likely needs a foundation upgrade or a schema rollback) — debug repr: {endpoint:?}"
            )));
        }
    };
    Ok(expand_node_type_wildcards(&names, schema))
}

/// If any name in `requested` is the wildcard `"*"`, returns every
/// node type in the schema. Otherwise returns the requested names
/// as-is. Shared between `SourceTypeFilter::resolve` and
/// `candidate_target_types` so the wildcard-expansion semantics are
/// defined exactly once.
fn expand_node_type_wildcards<'a>(requested: &[&'a str], schema: &'a Schema) -> Vec<&'a str> {
    if requested.contains(&EdgeEndpoint::WILDCARD) {
        schema.node_types.keys().map(String::as_str).collect()
    } else {
        requested.to_vec()
    }
}

/// Walk pre-resolved candidate node types until `engine.get_node`
/// returns one. `None` means orphan (no candidate type has this id) —
/// referential-integrity gap, caller-visible via `target.is_none()`
/// in the response.
fn fetch_target(
    engine: &StorageEngine,
    graph_id: &str,
    candidate_types: &[&str],
    to_id: &str,
) -> Result<Option<TargetNode>, RegistryError> {
    for ct in candidate_types {
        if let Some(stored) = engine.get_node(graph_id, ct, to_id)? {
            return Ok(Some(TargetNode {
                node_type: (*ct).to_string(),
                node_id: stored.node_id,
                properties: stored.properties,
            }));
        }
    }
    Ok(None)
}
