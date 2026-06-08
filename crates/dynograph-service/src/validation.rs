//! Cross-route schema validation helpers. Lives separately from
//! `registry.rs` (which holds `RegistryError` itself) and from the
//! per-route modules so multiple routes can share validators without
//! one route module taking a dependency on another.

use dynograph_core::Schema;

use crate::registry::RegistryError;

/// Shared upper bound for any route that takes a `limit` field.
/// 10_000 is the historical ceiling — well above any realistic
/// single-call workload, well below "I just want to list the whole
/// graph" (that's what scan endpoints with pagination are for, when
/// pagination ships).
pub(crate) const MAX_LIMIT: usize = 10_000;

/// Validate `limit` is in `1..=MAX_LIMIT`. `context` is the field
/// name (e.g. `"limit"`) interpolated into the error message so
/// callers don't each rewrite the same bounds-check.
pub(crate) fn validate_limit(limit: usize, context: &str) -> Result<(), RegistryError> {
    if limit == 0 || limit > MAX_LIMIT {
        return Err(RegistryError::BadRequest(format!(
            "{context} must be in 1..={MAX_LIMIT}, got {limit}",
        )));
    }
    Ok(())
}

/// Reject a `node_type` filter that can never full-text match — an
/// unknown type, or a known type that declares no `fulltext` property.
/// Shared by `search:text` and `search:hybrid`'s keyword leg so the
/// 400 message stays identical (and a client typo surfaces loudly
/// instead of as an empty 200). Only the keyword leg uses it, so it's
/// gated to the `fulltext` build like its callers.
#[cfg(feature = "fulltext")]
pub(crate) fn validate_fulltext_searchable(
    schema: &Schema,
    node_type: &str,
) -> Result<(), RegistryError> {
    if !schema.has_fulltext_properties(node_type) {
        return Err(RegistryError::BadRequest(format!(
            "node type '{node_type}' is not full-text searchable \
             (unknown type, or it declares no fulltext properties)"
        )));
    }
    Ok(())
}

/// Confirm that `prop` is declared on `node_type` in `schema` AND
/// flagged `indexed: true`. Used by routes that filter by property
/// via `scan_nodes_by_property` — un-indexed lookups silently return
/// empty (the storage scan walks the index CF only), which would
/// mask a misconfiguration as "no candidates found." Reject loudly
/// at the request boundary instead.
///
/// `context` is interpolated into the error message so callers can
/// surface where the validation fired (e.g. `"scope"` or
/// `"source.filter"`) without each call site rewriting the same
/// validation logic.
pub(crate) fn validate_indexed_property(
    schema: &Schema,
    node_type: &str,
    prop: &str,
    context: &str,
) -> Result<(), RegistryError> {
    let nt = schema.node_types.get(node_type).ok_or_else(|| {
        RegistryError::BadRequest(format!(
            "{context}.prop refers to unknown node type {node_type:?}"
        ))
    })?;
    let pd = nt.properties.get(prop).ok_or_else(|| {
        RegistryError::BadRequest(format!(
            "{context}.prop {prop:?} is not declared on node type {node_type}"
        ))
    })?;
    if !pd.indexed {
        return Err(RegistryError::BadRequest(format!(
            "{context}.prop {prop:?} is not indexed on node type {node_type} — cannot scope-filter (declare `indexed: true` in schema)"
        )));
    }
    Ok(())
}
