//! POST /v1/graphs/{id}/nodes:scan — predicate-filtered scan over a
//! single node type.
//!
//! Surfaced by market_graph's query service: "find all Persons with
//! `influence_level = market_moving`" and similar property-filtered
//! scans. Today's `list_nodes` query string takes one `(prop, value)`
//! pair only and is `eq`-only; this primitive generalizes to a list
//! of AND-combined predicates over the seven standard ops.
//!
//! ## Wire shape
//!
//! ```json
//! POST /v1/graphs/{id}/nodes:scan
//! {
//!   "type": "Person",
//!   "where": [
//!     {"property": "influence_level", "op": "eq", "value": "market_moving"}
//!   ],
//!   "limit": 100,
//!   "return": "nodes"      // optional: "nodes" (default) or "ids"
//! }
//! → {
//!   "results": [ {...node, ...} | "<id>" ],
//!   "truncated": false
//! }
//! ```
//!
//! ## Operators
//!
//! | op    | semantics                            | rhs shape                    |
//! |-------|--------------------------------------|------------------------------|
//! | `eq`  | property value == rhs                | scalar Value                 |
//! | `neq` | property value != rhs                | scalar Value                 |
//! | `in`  | property value ∈ rhs                 | array of scalar Values       |
//! | `gt`  | property value > rhs                 | scalar Value (Int / String)  |
//! | `lt`  | property value < rhs                 | scalar Value (Int / String)  |
//! | `gte` | property value >= rhs                | scalar Value (Int / String)  |
//! | `lte` | property value <= rhs                | scalar Value (Int / String)  |
//!
//! Range ops on `Float` and on heterogenous Value variants
//! (Bool/Null/List/Map) are rejected pre-flight — declarative
//! comparison on those types is either ambiguous (Float NaN, List
//! ordering) or out of scope. `String` range works lexicographically;
//! `Datetime` rides on the String path (storage stores Datetime as
//! ISO-8601 string — lexicographic order matches chronological order
//! for that format).
//!
//! ## Validation (all 400, pre-flight, no scan started)
//!
//! - `where` is non-empty (an unbounded "give me all" scan is what
//!   `list_nodes` is for — `nodes:scan` is for predicate filtering)
//! - `limit` in `1..=MAX_LIMIT`
//! - Each clause's `property` is declared on the type AND flagged
//!   `indexed: true`. Same un-indexed-rejection policy
//!   `/resolve-or-create`, `/edges:collect`, `/traverse`, and
//!   `/nodes:exists` use — non-indexed equality scans silently
//!   return empty, which would mask schema misconfig as "no matches".
//!   This validation also catches unknown node types (no need for a
//!   separate `type` existence check up front).
//! - Each clause's `value` is shape-correct for the op (e.g. `in`
//!   requires an array; range ops require an ordered scalar variant)
//! - `in` lists are capped at `MAX_IN_LIST_LEN` so a pathologically
//!   long `in` list can't translate to `O(candidates × in_len)`
//!   in-memory comparisons.
//!
//! ## Performance
//!
//! The seed strategy is "use the first `eq` clause to drive
//! `scan_nodes_by_property` (index-backed); fall back to
//! `scan_nodes` (full type scan) if no `eq` clause is present." The
//! remaining clauses are evaluated in memory per candidate row.
//! That makes the common single-`eq` case index-fast and gracefully
//! degrades for range-only filters.

use serde::{Deserialize, Serialize};

use dynograph_core::Value;
use dynograph_storage::{StorageEngine, StoredNode};

use crate::node_response::NodeResponse;
use crate::registry::RegistryError;
use crate::validation::{validate_indexed_property, validate_limit};

/// Upper bound on the rhs list for `Op::In`. Keeps the per-candidate
/// membership check from blowing up: with `MAX_LIMIT` candidates and
/// an N-element `in` list, the inner-loop cost is O(MAX_LIMIT × N).
/// 1_000 is comfortably above any practical "is this one of a known
/// set?" enum query and well below pathological pasted-CSV inputs.
pub(crate) const MAX_IN_LIST_LEN: usize = 1_000;

#[derive(Debug, Deserialize)]
pub(crate) struct NodesScanRequest {
    #[serde(rename = "type")]
    pub node_type: String,
    #[serde(rename = "where")]
    pub clauses: Vec<WhereClause>,
    pub limit: usize,
    #[serde(default)]
    pub r#return: ReturnShape,
}

#[derive(Debug, Deserialize)]
pub(crate) struct WhereClause {
    pub property: String,
    pub op: Op,
    pub value: Value,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Op {
    Eq,
    Neq,
    In,
    Gt,
    Lt,
    Gte,
    Lte,
}

#[derive(Debug, Deserialize, Default, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReturnShape {
    #[default]
    Nodes,
    Ids,
}

/// Untagged so the wire produces either `{"results": ["id", ...], ...}`
/// or `{"results": [{...node}, ...], ...}` depending on `return`. The
/// node shape is the shared `NodeResponse` — same `{node_type,
/// node_id, properties}` triple `GET /nodes` and the rest of foundation
/// emit, so consumers can deserialize into a single struct.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum NodesScanResponse {
    Ids {
        results: Vec<String>,
        truncated: bool,
    },
    Nodes {
        results: Vec<NodeResponse>,
        truncated: bool,
    },
}

pub(crate) fn run(
    engine: &StorageEngine,
    graph_id: &str,
    req: NodesScanRequest,
) -> Result<NodesScanResponse, RegistryError> {
    validate_limit(req.limit, "limit")?;
    if req.clauses.is_empty() {
        return Err(RegistryError::BadRequest(
            "where must be non-empty (use GET /v1/graphs/{id}/nodes for an unbounded type scan)"
                .to_string(),
        ));
    }

    let schema = engine.schema();
    for clause in &req.clauses {
        // validate_indexed_property catches unknown node type AND
        // unknown property AND un-indexed property in one pass; no
        // need for a separate type-existence check upstream.
        validate_indexed_property(schema, &req.node_type, &clause.property, "where")?;
        validate_clause_value_shape(clause)?;
    }

    // First `eq` clause drives an index-backed scan_nodes_by_property
    // (one CF prefix scan). Without an eq clause, fall back to a
    // full per-type scan. Either way, every clause (including the
    // seed) is re-evaluated in memory so the AND semantics stay
    // uniform.
    let seed_eq = req.clauses.iter().find(|c| c.op == Op::Eq);
    let candidates: Vec<StoredNode> = match seed_eq {
        Some(c) => {
            engine.scan_nodes_by_property(graph_id, &req.node_type, &c.property, &c.value)?
        }
        None => engine.scan_nodes(graph_id, &req.node_type)?,
    };

    let mut matched: Vec<StoredNode> = Vec::new();
    let mut truncated = false;
    for node in candidates {
        if !clauses_match(&req.clauses, &node) {
            continue;
        }
        matched.push(node);
        if matched.len() >= req.limit {
            truncated = true;
            break;
        }
    }

    Ok(match req.r#return {
        ReturnShape::Ids => NodesScanResponse::Ids {
            results: matched.into_iter().map(|n| n.node_id).collect(),
            truncated,
        },
        ReturnShape::Nodes => NodesScanResponse::Nodes {
            results: matched.into_iter().map(NodeResponse::from).collect(),
            truncated,
        },
    })
}

/// Confirm the `value` shape is sensible for the op. Cheap O(1) check
/// done pre-flight so a bad request rejects without doing any storage
/// I/O. The deeper "rhs Value variant is ordered" check for range ops
/// happens here too — a request that asks `gt` on a `Bool` or `List`
/// fails immediately rather than producing zero matches. The `in`-list
/// length cap also lives here so a hostile request rejects before
/// allocating a candidate set.
fn validate_clause_value_shape(clause: &WhereClause) -> Result<(), RegistryError> {
    match clause.op {
        Op::Eq | Op::Neq => Ok(()),
        Op::In => match &clause.value {
            Value::List(items) => {
                if items.len() > MAX_IN_LIST_LEN {
                    return Err(RegistryError::BadRequest(format!(
                        "where.value for op `in` exceeds maximum length {MAX_IN_LIST_LEN}, got {}",
                        items.len()
                    )));
                }
                Ok(())
            }
            other => Err(RegistryError::BadRequest(format!(
                "where.value for op `in` must be a list, got {}",
                other.type_name()
            ))),
        },
        Op::Gt | Op::Lt | Op::Gte | Op::Lte => match &clause.value {
            Value::Int(_) | Value::String(_) => Ok(()),
            other => Err(RegistryError::BadRequest(format!(
                "where.value for range op (gt/lt/gte/lte) must be int or string, got {}",
                other.type_name()
            ))),
        },
    }
}

fn clauses_match(clauses: &[WhereClause], node: &StoredNode) -> bool {
    clauses.iter().all(|c| clause_matches(c, node))
}

fn clause_matches(clause: &WhereClause, node: &StoredNode) -> bool {
    let Some(lhs) = node.properties.get(&clause.property) else {
        return false;
    };
    match clause.op {
        Op::Eq => lhs == &clause.value,
        Op::Neq => lhs != &clause.value,
        Op::In => match &clause.value {
            Value::List(items) => items.iter().any(|item| item == lhs),
            _ => false, // unreachable: shape validated pre-flight
        },
        Op::Gt => value_cmp(lhs, &clause.value).is_some_and(|o| o.is_gt()),
        Op::Lt => value_cmp(lhs, &clause.value).is_some_and(|o| o.is_lt()),
        Op::Gte => value_cmp(lhs, &clause.value).is_some_and(|o| o.is_ge()),
        Op::Lte => value_cmp(lhs, &clause.value).is_some_and(|o| o.is_le()),
    }
}

/// Ordered comparison between two `Value`s for range ops. Only
/// matching variants compare; cross-variant comparisons (e.g.
/// `Int(5) > String("hello")`) return None — the node is silently
/// excluded rather than crashing. Pre-flight validates the rhs is
/// `Int` or `String`; this fn additionally guards against an lhs
/// that's a different variant (e.g. a property declared `int` but
/// holding a string due to a hand-edited document) — those rows
/// just don't match.
fn value_cmp(lhs: &Value, rhs: &Value) -> Option<std::cmp::Ordering> {
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => Some(a.cmp(b)),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        _ => None,
    }
}
