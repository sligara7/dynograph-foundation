//! POST /v1/graphs/{id}/traverse — typed BFS over one or more
//! edge-type steps from a single start node.
//!
//! Backs the `compute_predecessors` shape — transitive walk along a
//! single edge type from a start node, scoped by some node property —
//! plus its multi-step generalization.
//!
//! ## Wire shape
//!
//! ```json
//! POST /v1/graphs/{id}/traverse
//! {
//!   "start": {"type": "NarrativeEpoch", "id": "epoch-X"},
//!   "traverse": [
//!     {"edge_type": "PRECEDES", "direction": "outgoing", "transitive": true}
//!   ],
//!   "scope":  {"prop": "story_id", "value": "Y"},   // optional
//!   "return": "ids" | "nodes",                       // optional, default "ids"
//!   "limit":  1000                                    // required, 1..=10_000
//! }
//! ```
//!
//! ## Semantics
//!
//! - **Start is never in the result.** Caller already has the start
//!   id; mirrors `compute_predecessors`'s "predecessors of X (not
//!   including X)" shape. If a future caller wants start included,
//!   they prepend it client-side.
//! - **Start not found in storage → 404** (loud failure, matches the
//!   foundation's no-silent-fallback principle). Start exists but
//!   fails the scope filter → empty `nodes` with 200 (legitimate
//!   "no matches", not a misconfiguration).
//! - **Result-dedup keyed by `(node_type, node_id)`** — each typed-id
//!   appears in the result set at most once. The multi-step shape is
//!   UNION semantics (closer to SQL UNION than JOIN): a peer reached
//!   at step 0 stays in results even if step 1's edges would also
//!   visit it.
//! - **Queue dedup keyed by `(node_id, step_idx)`** — each (node,
//!   step) pair is processed at most once. This is the cycle guard
//!   AND the reason a node visited at step 0 can still be processed
//!   at step 1 (different step → different queue key).
//! - **`transitive: true`** — fans out to both `(peer, step_idx)`
//!   (continue transitively at the same step) AND `(peer, step_idx+1)`
//!   (advance to the next step, if any). Without the latter, a chain
//!   like `[transitive A, then B]` would never apply step B.
//!   **`transitive: false`** — fans out only to `(peer, step_idx+1)`
//!   (one hop, then advance). Multi-step requests can mix both.
//! - **`direction: both`** — walks outgoing AND incoming on the same
//!   step. Useful for cycle detection where edge orientation isn't
//!   relevant. Peer-type candidates draw from both `EdgeTypeDef.from`
//!   and `EdgeTypeDef.to` (resolved per edge based on which CF the
//!   edge came from).
//!
//! ## Validation (all 400, all pre-flight, no scans before reject)
//!
//! - `start.type` declared in schema; `start.id` non-empty.
//! - `traverse` non-empty AND length ≤ `MAX_STEPS` (10) — sanity
//!   cap; longer chains are almost certainly a bug or a missing
//!   primitive.
//! - Each step's `edge_type` declared in schema.
//! - `limit` in `1..=MAX_LIMIT` (10_000).
//! - When `scope` is provided: `scope.prop` is `indexed: true` on
//!   `start.type` AND on every node type the BFS could land on
//!   (every candidate type for every step's direction). Same
//!   masked-misconfiguration rejection rule used by `/edges:collect`
//!   and `/resolve-or-create`. We check only nodes we visit (no
//!   `scan_nodes_by_property`), but un-declared / un-indexed props
//!   silently drop nodes from the scope filter — failing loud at
//!   the request boundary instead.
//!
//! ## Performance
//!
//! Read-only; whole BFS runs under one `with_engine_read` lock so
//! the candidate scan + per-node edge walks see a consistent
//! snapshot. Per visit: one `scan_outgoing_edges` and/or
//! `scan_incoming_edges` (storage prefix-scans on `(graph, node)` and
//! post-filters on `edge_type` in memory — no edge-type prefix CF
//! exists today) plus one `get_node` per peer to resolve type and
//! read properties for the scope check. A `peer_cache` keyed by
//! `peer_id` collapses repeat lookups on the same id across
//! transitive iterations and across multiple sources pointing at the
//! same target. Cost is `O(unique_peers × avg_candidates + visits)`,
//! capped by `limit`.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use dynograph_core::{EdgeEndpoint, Schema, Value};
use dynograph_storage::StorageEngine;

use crate::registry::RegistryError;
use crate::validation::{validate_indexed_property, validate_limit};

/// Hard cap on traverse-step chain length. Longer chains are almost
/// certainly a bug or a missing primitive (a schema-aware path query,
/// a recursive-CTE-shaped endpoint, etc.) — fail at the request
/// boundary rather than service it.
pub(crate) const MAX_STEPS: usize = 10;

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct TraverseRequest {
    pub start: StartSpec,
    pub traverse: Vec<TraverseStep>,
    #[serde(default)]
    pub scope: Option<PropertyFilter>,
    #[serde(default, rename = "return")]
    pub return_format: ReturnFormat,
    pub limit: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct StartSpec {
    #[serde(rename = "type")]
    pub node_type: String,
    pub id: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct TraverseStep {
    pub edge_type: String,
    pub direction: Direction,
    #[serde(default)]
    pub transitive: bool,
}

#[derive(Debug, Deserialize, Clone, Copy, ToSchema)]
#[serde(rename_all = "snake_case")]
#[schema(as = TraverseDirection)]
pub(crate) enum Direction {
    Outgoing,
    Incoming,
    Both,
}

#[derive(Debug, Deserialize, ToSchema)]
#[schema(as = TraversePropertyFilter)]
pub(crate) struct PropertyFilter {
    pub prop: String,
    #[schema(value_type = Object)]
    pub value: Value,
}

#[derive(Debug, Deserialize, Default, Clone, Copy, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReturnFormat {
    #[default]
    Ids,
    Nodes,
}

/// Per-result entry. `properties` is populated only when
/// `return: "nodes"`; for `return: "ids"` the field is omitted from
/// the wire shape via `skip_serializing_if`. Caller distinguishes by
/// presence of the field, OR by request echo (the request shape is
/// the source of truth for which they asked for).
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct TraversedNode {
    pub node_type: String,
    pub node_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Object)]
    pub properties: Option<HashMap<String, Value>>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct TraverseResponse {
    pub nodes: Vec<TraversedNode>,
    pub truncated: bool,
}

/// Pre-resolved per-step candidate types — outgoing and incoming
/// candidate sets computed once before the BFS so the inner loop
/// doesn't re-walk the schema's `EdgeEndpoint` for every edge.
struct StepCandidates {
    /// Candidate node types for peers when walking OUTGOING edges of
    /// this step's edge_type (i.e. the `to` endpoints).
    outgoing: Vec<String>,
    /// Candidate node types for peers when walking INCOMING edges of
    /// this step's edge_type (i.e. the `from` endpoints).
    incoming: Vec<String>,
}

/// Run the traversal. Caller holds the `with_engine_read` lock so
/// candidate scans + edge walks + node lookups see a consistent
/// snapshot.
pub(crate) fn run(
    engine: &StorageEngine,
    graph_id: &str,
    req: TraverseRequest,
) -> Result<TraverseResponse, RegistryError> {
    // ---- Pre-flight validation ----

    if req.traverse.is_empty() {
        return Err(RegistryError::BadRequest(
            "traverse must be non-empty".to_string(),
        ));
    }
    if req.traverse.len() > MAX_STEPS {
        return Err(RegistryError::BadRequest(format!(
            "traverse length {} exceeds maximum {MAX_STEPS}",
            req.traverse.len()
        )));
    }
    validate_limit(req.limit, "limit")?;
    if req.start.id.is_empty() {
        return Err(RegistryError::BadRequest(
            "start.id must be non-empty".to_string(),
        ));
    }

    let schema = engine.schema();

    if !schema.node_types.contains_key(req.start.node_type.as_str()) {
        return Err(RegistryError::BadRequest(format!(
            "start.type references unknown node type: {}",
            req.start.node_type
        )));
    }

    // Validate edge_types up-front and pre-resolve candidate types
    // per step (per direction). Surfaces non_exhaustive EdgeEndpoint
    // variants here, before any storage I/O.
    let mut step_candidates: Vec<StepCandidates> = Vec::with_capacity(req.traverse.len());
    for step in &req.traverse {
        let edge_def = schema.edge_types.get(&step.edge_type).ok_or_else(|| {
            RegistryError::BadRequest(format!(
                "traverse step references unknown edge type: {}",
                step.edge_type
            ))
        })?;
        let outgoing = match step.direction {
            Direction::Outgoing | Direction::Both => {
                candidate_endpoint_types(&edge_def.to, schema, &step.edge_type)?
            }
            Direction::Incoming => Vec::new(),
        };
        let incoming = match step.direction {
            Direction::Incoming | Direction::Both => {
                candidate_endpoint_types(&edge_def.from, schema, &step.edge_type)?
            }
            Direction::Outgoing => Vec::new(),
        };
        step_candidates.push(StepCandidates { outgoing, incoming });
    }

    // Scope-prop validation: must be `indexed: true` on the start
    // type AND every per-step candidate type. We don't actually call
    // `scan_nodes_by_property` here, but the rule is "if a prop can
    // be silently dropped (because it isn't declared on the type),
    // reject loudly at the boundary" — same posture as edges_collect.
    if let Some(ref scope) = req.scope {
        validate_indexed_property(schema, &req.start.node_type, &scope.prop, "scope")?;
        for sc in &step_candidates {
            for t in sc.outgoing.iter().chain(sc.incoming.iter()) {
                validate_indexed_property(schema, t, &scope.prop, "scope")?;
            }
        }
    }

    // ---- Fetch start; honor scope; seed BFS ----

    let start_node = engine
        .get_node(graph_id, &req.start.node_type, &req.start.id)?
        .ok_or_else(|| RegistryError::NodeNotFound {
            node_type: req.start.node_type.clone(),
            node_id: req.start.id.clone(),
        })?;

    let start_in_scope = node_matches_scope(&start_node.properties, req.scope.as_ref());

    // Two dedup sets, two purposes:
    //   - `emitted` keys `(type, id)` — result-set dedup with UNION
    //     semantics across steps. Pre-seeded with start so start never
    //     lands in results, and a cycle back through start is silent.
    //   - `queued_steps` keys `(id, step_idx)` — work-item dedup AND
    //     cycle guard. A given id can legitimately enter the queue at
    //     a different step_idx (that's how multi-step chains advance);
    //     the same (id, step) pair is processed at most once.
    let mut emitted: HashSet<(String, String)> = HashSet::new();
    emitted.insert((req.start.node_type.clone(), req.start.id.clone()));
    let mut queued_steps: HashSet<(String, usize)> = HashSet::new();

    let mut queue: VecDeque<(String, usize)> = VecDeque::new();
    if start_in_scope {
        queued_steps.insert((req.start.id.clone(), 0));
        queue.push_back((req.start.id.clone(), 0));
    }

    // Peer-resolution cache: a given peer_id can be probed many times
    // — across transitive iterations and across multiple sources
    // pointing at the same target. Caching avoids re-running
    // `fetch_peer` (which can do up to `candidate_types.len()`
    // CF point-lookups + msgpack decodes) for the same id. Cache
    // hits are validated against the current candidate set; a cached
    // type that isn't in the current candidates is dropped to a
    // fresh resolution (safe even when the same id exists under
    // multiple types, which is rare but legal).
    let mut peer_cache: HashMap<String, (String, dynograph_storage::StoredNode)> = HashMap::new();

    // Hoisted out of the inner loop: the format choice doesn't change
    // per peer.
    let want_props = req.return_format == ReturnFormat::Nodes;

    let mut results: Vec<TraversedNode> = Vec::new();
    let mut truncated = false;

    'bfs: while let Some((curr_id, step_idx)) = queue.pop_front() {
        let step = &req.traverse[step_idx];
        let cands = &step_candidates[step_idx];

        let outgoing = if matches!(step.direction, Direction::Outgoing | Direction::Both) {
            engine.scan_outgoing_edges(graph_id, &curr_id, Some(&step.edge_type))?
        } else {
            Vec::new()
        };
        let incoming = if matches!(step.direction, Direction::Incoming | Direction::Both) {
            engine.scan_incoming_edges(graph_id, &curr_id, Some(&step.edge_type))?
        } else {
            Vec::new()
        };

        // Each peer = (peer_id, candidates_to_resolve_peer_type).
        // Outgoing edges' peer is `to_id` resolved via `cands.outgoing`;
        // incoming edges' peer is `from_id` resolved via `cands.incoming`.
        // Chain rather than collecting into a `Vec<...>` first.
        let peers = outgoing
            .into_iter()
            .map(|e| (e.to_id, cands.outgoing.as_slice()))
            .chain(
                incoming
                    .into_iter()
                    .map(|e| (e.from_id, cands.incoming.as_slice())),
            );

        for (peer_id, peer_candidates) in peers {
            let resolved = match peer_cache.get(&peer_id) {
                Some(hit) if peer_candidates.iter().any(|c| c == &hit.0) => Some(hit.clone()),
                _ => {
                    let result = fetch_peer(engine, graph_id, peer_candidates, &peer_id)?;
                    if let Some(ref hit) = result {
                        peer_cache.insert(peer_id.clone(), hit.clone());
                    }
                    result
                }
            };
            // Orphan (no candidate type has this id) is silently
            // skipped — referential-integrity gap on the storage
            // side, same posture as edges_collect's `target.is_none()`.
            let Some((peer_type, peer_node)) = resolved else {
                continue;
            };

            if !node_matches_scope(&peer_node.properties, req.scope.as_ref()) {
                continue;
            }

            if emitted.insert((peer_type.clone(), peer_id.clone())) {
                results.push(TraversedNode {
                    node_type: peer_type,
                    node_id: peer_id.clone(),
                    properties: if want_props {
                        Some(peer_node.properties)
                    } else {
                        None
                    },
                });
                if results.len() >= req.limit {
                    truncated = true;
                    break 'bfs;
                }
            }

            // Fan-out semantics live in the module doc; this block is
            // its mechanical realization. `transitive` ordered last so
            // its `push_back` can move (rather than clone) peer_id.
            let next_step = step_idx + 1;
            if next_step < req.traverse.len() && queued_steps.insert((peer_id.clone(), next_step)) {
                queue.push_back((peer_id.clone(), next_step));
            }
            if step.transitive && queued_steps.insert((peer_id.clone(), step_idx)) {
                queue.push_back((peer_id, step_idx));
            }
        }
    }

    Ok(TraverseResponse {
        nodes: results,
        truncated,
    })
}

/// True when the node either has no scope filter to satisfy, or its
/// `properties[prop] == value`. A missing prop (un-declared on this
/// type, or just absent on this row) cleanly fails — that's the
/// silent-drop the indexed-validation guard fronts.
fn node_matches_scope(props: &HashMap<String, Value>, scope: Option<&PropertyFilter>) -> bool {
    match scope {
        None => true,
        Some(s) => props.get(&s.prop) == Some(&s.value),
    }
}

/// Walk candidate node types until `engine.get_node` returns one;
/// return that `(type, node)` pair. `None` means orphan (no
/// candidate type has this id) — the edge points at a node id
/// foundation can't resolve from the schema's endpoint declaration.
fn fetch_peer(
    engine: &StorageEngine,
    graph_id: &str,
    candidate_types: &[String],
    peer_id: &str,
) -> Result<Option<(String, dynograph_storage::StoredNode)>, RegistryError> {
    for ct in candidate_types {
        if let Some(stored) = engine.get_node(graph_id, ct, peer_id)? {
            return Ok(Some((ct.clone(), stored)));
        }
    }
    Ok(None)
}

/// Returns the candidate node-type names that an `EdgeEndpoint` can
/// reach, expanding `"*"` via the schema. Owned `String`s (not
/// borrowed) because the caller stores them in `StepCandidates` which
/// outlives the immediate validation pass and is read across BFS
/// iterations. Adjacent to `edges_collect::candidate_target_types`,
/// which is borrowed-shape; merging the two into a single helper in
/// `validation.rs` is a future cleanup once a third caller appears
/// or the borrowed version grows the post-expansion existence check
/// this version has.
fn candidate_endpoint_types(
    endpoint: &EdgeEndpoint,
    schema: &Schema,
    edge_type: &str,
) -> Result<Vec<String>, RegistryError> {
    let names: Vec<&str> = match endpoint {
        EdgeEndpoint::Single(t) => vec![t.as_str()],
        EdgeEndpoint::Multiple(ts) => ts.iter().map(String::as_str).collect(),
        // EdgeEndpoint is #[non_exhaustive]. A future variant that
        // lands without updating this dispatch would silently widen
        // (or narrow) the candidate set; better to fail at the
        // request boundary with a foundation-upgrade hint.
        _ => {
            return Err(RegistryError::BadRequest(format!(
                "edge type {edge_type}: EdgeTypeDef endpoint uses an EdgeEndpoint variant unknown to this foundation build (likely needs a foundation upgrade or a schema rollback) — debug repr: {endpoint:?}"
            )));
        }
    };
    let expanded: Vec<String> = if names.contains(&EdgeEndpoint::WILDCARD) {
        schema.node_types.keys().cloned().collect()
    } else {
        names.into_iter().map(str::to_string).collect()
    };
    // Verify every named type exists. Catches a schema that names
    // a node type in `EdgeTypeDef.from`/`.to` that no longer exists
    // — would otherwise silently produce zero candidates.
    for n in &expanded {
        if !schema.node_types.contains_key(n) {
            return Err(RegistryError::BadRequest(format!(
                "edge type {edge_type}: endpoint references unknown node type {n:?}"
            )));
        }
    }
    Ok(expanded)
}
