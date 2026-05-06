//! POST /v1/graphs/{id}/edges:collect — fan-out edge collection
//! across a typed source set.
//!
//! Closes audit primitive #3 (storyflow audit 2026-05-04). Replaces
//! the storyflow `collect_story_edges` master pattern that walks
//! N entity types × M nodes × K edge types via per-node
//! `outgoing_edges` — hundreds of round-trips per call today; one
//! HTTP call after migration. Used by 13+ knowledge-graph routes
//! plus the projection step in pagerank/louvain/shortest-path.
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
//! consumers (pagerank/louvain) don't care about order; "show me a
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

use dynograph_core::{EdgeEndpoint, Schema, Value};
use dynograph_storage::StorageEngine;

use crate::registry::RegistryError;

/// Hard upper bound on a single response. Above this, callers should
/// shard via filter or pagination (not yet supported — add when a real
/// workload pushes past).
pub(crate) const MAX_LIMIT: usize = 10_000;

#[derive(Debug, Deserialize)]
pub(crate) struct EdgesCollectRequest {
    pub source: SourceSpec,
    pub edge_types: Vec<String>,
    #[serde(default)]
    pub format: ResponseFormat,
    #[serde(default)]
    pub resolve_target: bool,
    pub limit: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SourceSpec {
    #[serde(rename = "type")]
    pub type_filter: SourceTypeFilter,
    #[serde(default)]
    pub filter: Option<PropertyFilter>,
}

/// Source-type matcher. Untagged so the JSON can be either a string
/// (`"*"` or a single type name) or an array of names.
#[derive(Debug, Deserialize)]
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
            SourceTypeFilter::Single(s) if s == EdgeEndpoint::WILDCARD => {
                schema.node_types.keys().map(String::as_str).collect()
            }
            SourceTypeFilter::Single(s) => vec![s.as_str()],
            SourceTypeFilter::Multiple(ss) => {
                if ss.iter().any(|s| s == EdgeEndpoint::WILDCARD) {
                    schema.node_types.keys().map(String::as_str).collect()
                } else {
                    ss.iter().map(String::as_str).collect()
                }
            }
        };
        for n in &names {
            if !schema.node_types.contains_key(*n) {
                return Err(RegistryError::BadRequest(format!(
                    "source.type references unknown node type: {n}"
                )));
            }
        }
        Ok(names)
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct PropertyFilter {
    pub prop: String,
    pub value: Value,
}

#[derive(Debug, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ResponseFormat {
    #[default]
    Edges,
    Adjacency,
}

#[derive(Debug, Serialize)]
pub(crate) struct CollectedEdge {
    pub edge_type: String,
    pub from_type: String,
    pub from_id: String,
    pub to_id: String,
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

#[derive(Debug, Serialize)]
pub(crate) struct AdjacencyEntry {
    pub edge_type: String,
    pub to_id: String,
    pub properties: HashMap<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<TargetNode>,
}

#[derive(Debug, Serialize)]
pub(crate) struct TargetNode {
    pub node_type: String,
    pub node_id: String,
    pub properties: HashMap<String, Value>,
}

/// Untagged so the JSON has the shape the caller asked for: an
/// `{"edges": [...], "truncated": ...}` object for `format = "edges"`,
/// an `{"adjacency": {...}, "truncated": ...}` object for
/// `format = "adjacency"`. Caller distinguishes by which field is
/// present (always exactly one).
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum EdgesCollectResponse {
    Edges {
        edges: Vec<CollectedEdge>,
        truncated: bool,
    },
    Adjacency {
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
    if req.limit == 0 || req.limit > MAX_LIMIT {
        return Err(RegistryError::BadRequest(format!(
            "limit must be in 1..={MAX_LIMIT}, got {}",
            req.limit
        )));
    }

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
            let nt = schema
                .node_types
                .get(*st)
                .expect("source_types validated above");
            let pd = nt.properties.get(&f.prop).ok_or_else(|| {
                RegistryError::BadRequest(format!(
                    "source.filter.prop {:?} is not declared on node type {st}",
                    f.prop
                ))
            })?;
            if !pd.indexed {
                return Err(RegistryError::BadRequest(format!(
                    "source.filter.prop {:?} is not indexed on node type {st} — cannot scope-filter",
                    f.prop
                )));
            }
        }
    }

    let edge_type_set: std::collections::HashSet<&str> =
        req.edge_types.iter().map(String::as_str).collect();

    // ---- Fan-out ----

    let mut collected: Vec<(String, CollectedEdge)> = Vec::new();
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
                    resolve_target_node(engine, graph_id, schema, &edge.edge_type, &edge.to_id)?
                } else {
                    None
                };

                collected.push((
                    edge.from_id.clone(),
                    CollectedEdge {
                        edge_type: edge.edge_type,
                        from_type: (*source_type).to_string(),
                        from_id: edge.from_id,
                        to_id: edge.to_id,
                        properties: edge.properties,
                        target,
                    },
                ));

                if collected.len() >= req.limit {
                    truncated = true;
                    break 'outer;
                }
            }
        }
    }

    // ---- Shape into requested format ----

    Ok(match req.format {
        ResponseFormat::Edges => EdgesCollectResponse::Edges {
            edges: collected.into_iter().map(|(_, e)| e).collect(),
            truncated,
        },
        ResponseFormat::Adjacency => {
            let mut adjacency: HashMap<String, Vec<AdjacencyEntry>> = HashMap::new();
            for (from_id, e) in collected {
                adjacency.entry(from_id).or_default().push(AdjacencyEntry {
                    edge_type: e.edge_type,
                    to_id: e.to_id,
                    properties: e.properties,
                    target: e.target,
                });
            }
            EdgesCollectResponse::Adjacency {
                adjacency,
                truncated,
            }
        }
    })
}

/// Try to find the target node by walking the candidate types
/// declared in the schema's `EdgeTypeDef.to`. First successful
/// `get_node` wins. Returns `None` if no candidate type has the node
/// (orphan — referential integrity gap; caller-visible).
fn resolve_target_node(
    engine: &StorageEngine,
    graph_id: &str,
    schema: &Schema,
    edge_type: &str,
    to_id: &str,
) -> Result<Option<TargetNode>, RegistryError> {
    let edge_def = schema.edge_types.get(edge_type).ok_or_else(|| {
        // Validated at the top level, but defensive — schema could in
        // principle be replaced under a long-running call.
        RegistryError::BadRequest(format!("edge_type vanished from schema: {edge_type}"))
    })?;

    // EdgeEndpoint is #[non_exhaustive]; the wildcard arm guards
    // against a future variant landing in dynograph-core without
    // updating this dispatch. Returning a 500-ish bad request beats
    // silently widening to "try every type" (which would mask the
    // missing handler) or panicking (which would log without the
    // graph_id context).
    let candidate_types: Vec<&str> = match &edge_def.to {
        EdgeEndpoint::Single(t) if t == EdgeEndpoint::WILDCARD => {
            schema.node_types.keys().map(String::as_str).collect()
        }
        EdgeEndpoint::Single(t) => vec![t.as_str()],
        EdgeEndpoint::Multiple(ts) => {
            if ts.iter().any(|t| t == EdgeEndpoint::WILDCARD) {
                schema.node_types.keys().map(String::as_str).collect()
            } else {
                ts.iter().map(String::as_str).collect()
            }
        }
        _ => {
            return Err(RegistryError::BadRequest(format!(
                "edge type {edge_type} uses an EdgeEndpoint variant this foundation version doesn't recognize"
            )));
        }
    };

    for ct in candidate_types {
        if let Some(stored) = engine.get_node(graph_id, ct, to_id)? {
            return Ok(Some(TargetNode {
                node_type: ct.to_string(),
                node_id: stored.node_id,
                properties: stored.properties,
            }));
        }
    }
    Ok(None)
}
