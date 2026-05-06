//! Cross-route schema validation helpers. Lives separately from
//! `registry.rs` (which holds `RegistryError` itself) and from the
//! per-route modules so multiple routes can share validators without
//! one route module taking a dependency on another.

use dynograph_core::Schema;

use crate::registry::RegistryError;

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
