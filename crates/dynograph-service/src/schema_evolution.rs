//! Schema-evolution compatibility checks for `PUT /v1/graphs/{id}/schema`.
//!
//! The rule is **additive**: a new schema is accepted if every node
//! and edge that was valid under the old schema would still be valid
//! under the new one. Concretely:
//!
//! - Removing a node or edge type would orphan existing data.
//! - Removing a property would silently drop stored values on the
//!   next write.
//! - Changing a property's type would invalidate existing values.
//! - Optional → required without a default would fail validation for
//!   existing rows that don't have the value.
//! - Narrowing an edge's `from`/`to` endpoint set would break
//!   existing edges whose endpoints are no longer accepted.
//!
//! Constraint-tightening (range narrowing, enum-values shrinking) is
//! *not* checked yet — those are real breaking changes but require
//! data inspection to detect, so they're deferred to a future slice.
//! For now, an evolved schema with tighter constraints will pass this
//! check; subsequent writes that violate the tighter constraint will
//! fail at validate-time, which is the existing fail-loud behavior.
//!
//! Reports **all** violations, not just the first. Callers see the
//! full incompat set in one response.

use dynograph_core::{EdgeEndpoint, EdgeTypeDef, NodeTypeDef, PropertyDef, PropertyType, Schema};

#[derive(Debug, Clone, PartialEq)]
pub enum EvolutionError {
    RemovedNodeType(String),
    RemovedEdgeType(String),
    RemovedNodeProperty {
        node_type: String,
        property: String,
    },
    RemovedEdgeProperty {
        edge_type: String,
        property: String,
    },
    ChangedNodePropertyType {
        node_type: String,
        property: String,
        old: PropertyType,
        new: PropertyType,
    },
    ChangedEdgePropertyType {
        edge_type: String,
        property: String,
        old: PropertyType,
        new: PropertyType,
    },
    NodePropertyBecameRequiredWithoutDefault {
        node_type: String,
        property: String,
    },
    EdgePropertyBecameRequiredWithoutDefault {
        edge_type: String,
        property: String,
    },
    NarrowedEdgeFrom {
        edge_type: String,
    },
    NarrowedEdgeTo {
        edge_type: String,
    },
}

impl std::fmt::Display for EvolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RemovedNodeType(t) => write!(f, "removed node type: {t}"),
            Self::RemovedEdgeType(t) => write!(f, "removed edge type: {t}"),
            Self::RemovedNodeProperty {
                node_type,
                property,
            } => write!(f, "removed node property: {node_type}.{property}"),
            Self::RemovedEdgeProperty {
                edge_type,
                property,
            } => write!(f, "removed edge property: {edge_type}.{property}"),
            Self::ChangedNodePropertyType {
                node_type,
                property,
                old,
                new,
            } => write!(
                f,
                "changed node property type: {node_type}.{property} {old:?} -> {new:?}"
            ),
            Self::ChangedEdgePropertyType {
                edge_type,
                property,
                old,
                new,
            } => write!(
                f,
                "changed edge property type: {edge_type}.{property} {old:?} -> {new:?}"
            ),
            Self::NodePropertyBecameRequiredWithoutDefault {
                node_type,
                property,
            } => write!(
                f,
                "node property became required without a default: {node_type}.{property}"
            ),
            Self::EdgePropertyBecameRequiredWithoutDefault {
                edge_type,
                property,
            } => write!(
                f,
                "edge property became required without a default: {edge_type}.{property}"
            ),
            Self::NarrowedEdgeFrom { edge_type } => {
                write!(f, "edge `from` endpoint narrowed: {edge_type}")
            }
            Self::NarrowedEdgeTo { edge_type } => {
                write!(f, "edge `to` endpoint narrowed: {edge_type}")
            }
        }
    }
}

pub fn validate_compatible(old: &Schema, new: &Schema) -> Result<(), Vec<EvolutionError>> {
    let mut errors = Vec::new();

    for (name, old_type) in &old.node_types {
        match new.node_types.get(name) {
            None => errors.push(EvolutionError::RemovedNodeType(name.clone())),
            Some(new_type) => check_node_type(name, old_type, new_type, &mut errors),
        }
    }

    for (name, old_edge) in &old.edge_types {
        match new.edge_types.get(name) {
            None => errors.push(EvolutionError::RemovedEdgeType(name.clone())),
            Some(new_edge) => check_edge_type(name, old_edge, new_edge, &mut errors),
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn check_node_type(
    name: &str,
    old: &NodeTypeDef,
    new: &NodeTypeDef,
    errors: &mut Vec<EvolutionError>,
) {
    for (prop_name, old_prop) in &old.properties {
        match new.properties.get(prop_name) {
            None => errors.push(EvolutionError::RemovedNodeProperty {
                node_type: name.to_string(),
                property: prop_name.clone(),
            }),
            Some(new_prop) if new_prop.prop_type != old_prop.prop_type => {
                errors.push(EvolutionError::ChangedNodePropertyType {
                    node_type: name.to_string(),
                    property: prop_name.clone(),
                    old: old_prop.prop_type.clone(),
                    new: new_prop.prop_type.clone(),
                });
            }
            Some(_) => {}
        }
    }
    // New required property without a default fails validation for
    // every existing row of this type — equivalent to a removal.
    for (prop_name, new_prop) in &new.properties {
        if !old.properties.contains_key(prop_name)
            && new_prop.required
            && new_prop.default.is_none()
        {
            errors.push(EvolutionError::NodePropertyBecameRequiredWithoutDefault {
                node_type: name.to_string(),
                property: prop_name.clone(),
            });
        }
    }
    // An existing property tightening from optional → required
    // without a default has the same effect.
    for (prop_name, old_prop) in &old.properties {
        if let Some(new_prop) = new.properties.get(prop_name)
            && tightened_to_required_without_default(old_prop, new_prop)
        {
            errors.push(EvolutionError::NodePropertyBecameRequiredWithoutDefault {
                node_type: name.to_string(),
                property: prop_name.clone(),
            });
        }
    }
}

fn check_edge_type(
    name: &str,
    old: &EdgeTypeDef,
    new: &EdgeTypeDef,
    errors: &mut Vec<EvolutionError>,
) {
    if !endpoint_widened_or_same(&old.from, &new.from) {
        errors.push(EvolutionError::NarrowedEdgeFrom {
            edge_type: name.to_string(),
        });
    }
    if !endpoint_widened_or_same(&old.to, &new.to) {
        errors.push(EvolutionError::NarrowedEdgeTo {
            edge_type: name.to_string(),
        });
    }
    for (prop_name, old_prop) in &old.properties {
        match new.properties.get(prop_name) {
            None => errors.push(EvolutionError::RemovedEdgeProperty {
                edge_type: name.to_string(),
                property: prop_name.clone(),
            }),
            Some(new_prop) if new_prop.prop_type != old_prop.prop_type => {
                errors.push(EvolutionError::ChangedEdgePropertyType {
                    edge_type: name.to_string(),
                    property: prop_name.clone(),
                    old: old_prop.prop_type.clone(),
                    new: new_prop.prop_type.clone(),
                });
            }
            Some(_) => {}
        }
    }
    for (prop_name, new_prop) in &new.properties {
        if !old.properties.contains_key(prop_name)
            && new_prop.required
            && new_prop.default.is_none()
        {
            errors.push(EvolutionError::EdgePropertyBecameRequiredWithoutDefault {
                edge_type: name.to_string(),
                property: prop_name.clone(),
            });
        }
    }
    for (prop_name, old_prop) in &old.properties {
        if let Some(new_prop) = new.properties.get(prop_name)
            && tightened_to_required_without_default(old_prop, new_prop)
        {
            errors.push(EvolutionError::EdgePropertyBecameRequiredWithoutDefault {
                edge_type: name.to_string(),
                property: prop_name.clone(),
            });
        }
    }
}

fn tightened_to_required_without_default(old: &PropertyDef, new: &PropertyDef) -> bool {
    !old.required && new.required && new.default.is_none()
}

/// True when every node type the old endpoint accepted is still
/// accepted by the new endpoint. Wildcard `*` is special-cased: an
/// old wildcard requires a new wildcard (otherwise data with
/// arbitrary endpoint types could break); a new wildcard always
/// accepts the old set.
fn endpoint_widened_or_same(old: &EdgeEndpoint, new: &EdgeEndpoint) -> bool {
    if endpoint_is_wildcard(new) {
        return true;
    }
    let old_types: Vec<&str> = match old {
        EdgeEndpoint::Single(t) => vec![t.as_str()],
        EdgeEndpoint::Multiple(ts) => ts.iter().map(String::as_str).collect(),
    };
    for t in &old_types {
        if *t == "*" {
            return false;
        }
        if !new.accepts(t) {
            return false;
        }
    }
    true
}

fn endpoint_is_wildcard(ep: &EdgeEndpoint) -> bool {
    match ep {
        EdgeEndpoint::Single(t) => t == "*",
        EdgeEndpoint::Multiple(ts) => ts.iter().any(|t| t == "*"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_from(yaml: &str) -> Schema {
        Schema::from_yaml(yaml).unwrap()
    }

    fn base() -> Schema {
        schema_from(
            r#"
schema:
  name: t
  version: 1
  node_types:
    Person:
      properties:
        name: { type: string, required: true }
        age:  { type: int }
  edge_types:
    Knows:
      from: Person
      to: Person
      properties:
        since: { type: int }
"#,
        )
    }

    #[test]
    fn identical_schemas_are_compatible() {
        validate_compatible(&base(), &base()).unwrap();
    }

    #[test]
    fn adding_a_node_type_is_compatible() {
        let new = schema_from(
            r#"
schema:
  name: t
  version: 2
  node_types:
    Person:
      properties:
        name: { type: string, required: true }
        age:  { type: int }
    Place:
      properties:
        name: { type: string }
  edge_types:
    Knows:
      from: Person
      to: Person
      properties:
        since: { type: int }
"#,
        );
        validate_compatible(&base(), &new).unwrap();
    }

    #[test]
    fn adding_optional_property_is_compatible() {
        let new = schema_from(
            r#"
schema:
  name: t
  version: 2
  node_types:
    Person:
      properties:
        name: { type: string, required: true }
        age:  { type: int }
        nickname: { type: string }
  edge_types:
    Knows:
      from: Person
      to: Person
      properties:
        since: { type: int }
"#,
        );
        validate_compatible(&base(), &new).unwrap();
    }

    #[test]
    fn adding_required_property_with_default_is_compatible() {
        let new = schema_from(
            r#"
schema:
  name: t
  version: 2
  node_types:
    Person:
      properties:
        name: { type: string, required: true }
        age:  { type: int }
        tier: { type: string, required: true, default: standard }
  edge_types:
    Knows:
      from: Person
      to: Person
      properties:
        since: { type: int }
"#,
        );
        validate_compatible(&base(), &new).unwrap();
    }

    #[test]
    fn relaxing_required_to_optional_is_compatible() {
        let new = schema_from(
            r#"
schema:
  name: t
  version: 2
  node_types:
    Person:
      properties:
        name: { type: string }
        age:  { type: int }
  edge_types:
    Knows:
      from: Person
      to: Person
      properties:
        since: { type: int }
"#,
        );
        validate_compatible(&base(), &new).unwrap();
    }

    #[test]
    fn removing_node_type_rejected() {
        let new = schema_from(
            r#"
schema:
  name: t
  version: 2
  node_types: {}
  edge_types: {}
"#,
        );
        let errs = validate_compatible(&base(), &new).unwrap_err();
        assert!(errs.contains(&EvolutionError::RemovedNodeType("Person".into())));
        assert!(errs.contains(&EvolutionError::RemovedEdgeType("Knows".into())));
    }

    #[test]
    fn removing_node_property_rejected() {
        let new = schema_from(
            r#"
schema:
  name: t
  version: 2
  node_types:
    Person:
      properties:
        name: { type: string, required: true }
  edge_types:
    Knows:
      from: Person
      to: Person
      properties:
        since: { type: int }
"#,
        );
        let errs = validate_compatible(&base(), &new).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            &errs[0],
            EvolutionError::RemovedNodeProperty { node_type, property }
                if node_type == "Person" && property == "age"
        ));
    }

    #[test]
    fn changing_node_property_type_rejected() {
        let new = schema_from(
            r#"
schema:
  name: t
  version: 2
  node_types:
    Person:
      properties:
        name: { type: string, required: true }
        age:  { type: string }
  edge_types:
    Knows:
      from: Person
      to: Person
      properties:
        since: { type: int }
"#,
        );
        let errs = validate_compatible(&base(), &new).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            &errs[0],
            EvolutionError::ChangedNodePropertyType { node_type, property, .. }
                if node_type == "Person" && property == "age"
        ));
    }

    #[test]
    fn adding_required_without_default_rejected() {
        let new = schema_from(
            r#"
schema:
  name: t
  version: 2
  node_types:
    Person:
      properties:
        name: { type: string, required: true }
        age:  { type: int }
        ssn:  { type: string, required: true }
  edge_types:
    Knows:
      from: Person
      to: Person
      properties:
        since: { type: int }
"#,
        );
        let errs = validate_compatible(&base(), &new).unwrap_err();
        assert!(matches!(
            errs.first(),
            Some(EvolutionError::NodePropertyBecameRequiredWithoutDefault { node_type, property })
                if node_type == "Person" && property == "ssn"
        ));
    }

    #[test]
    fn tightening_optional_to_required_without_default_rejected() {
        let new = schema_from(
            r#"
schema:
  name: t
  version: 2
  node_types:
    Person:
      properties:
        name: { type: string, required: true }
        age:  { type: int, required: true }
  edge_types:
    Knows:
      from: Person
      to: Person
      properties:
        since: { type: int }
"#,
        );
        let errs = validate_compatible(&base(), &new).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            &errs[0],
            EvolutionError::NodePropertyBecameRequiredWithoutDefault { node_type, property }
                if node_type == "Person" && property == "age"
        ));
    }

    #[test]
    fn narrowing_edge_endpoint_rejected() {
        let new = schema_from(
            r#"
schema:
  name: t
  version: 2
  node_types:
    Person:
      properties:
        name: { type: string, required: true }
        age:  { type: int }
    Place:
      properties:
        name: { type: string }
  edge_types:
    Knows:
      from: Person
      to: Place
      properties:
        since: { type: int }
"#,
        );
        let errs = validate_compatible(&base(), &new).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            &errs[0],
            EvolutionError::NarrowedEdgeTo { edge_type } if edge_type == "Knows"
        ));
    }

    #[test]
    fn widening_edge_endpoint_to_wildcard_is_compatible() {
        let new = schema_from(
            r#"
schema:
  name: t
  version: 2
  node_types:
    Person:
      properties:
        name: { type: string, required: true }
        age:  { type: int }
  edge_types:
    Knows:
      from: Person
      to: "*"
      properties:
        since: { type: int }
"#,
        );
        validate_compatible(&base(), &new).unwrap();
    }

    #[test]
    fn widening_to_multiple_endpoint_compatible_narrowing_rejected() {
        let old = schema_from(
            r#"
schema:
  name: t
  version: 1
  node_types:
    Person: { properties: {} }
    Animal: { properties: {} }
  edge_types:
    Knows:
      from: Person
      to: [Person, Animal]
      properties: {}
"#,
        );
        let widened = schema_from(
            r#"
schema:
  name: t
  version: 2
  node_types:
    Person: { properties: {} }
    Animal: { properties: {} }
    Place:  { properties: {} }
  edge_types:
    Knows:
      from: Person
      to: [Person, Animal, Place]
      properties: {}
"#,
        );
        validate_compatible(&old, &widened).unwrap();

        let narrowed = schema_from(
            r#"
schema:
  name: t
  version: 2
  node_types:
    Person: { properties: {} }
    Animal: { properties: {} }
  edge_types:
    Knows:
      from: Person
      to: [Person]
      properties: {}
"#,
        );
        let errs = validate_compatible(&old, &narrowed).unwrap_err();
        assert!(matches!(
            &errs[0],
            EvolutionError::NarrowedEdgeTo { edge_type } if edge_type == "Knows"
        ));
    }

    #[test]
    fn multiple_violations_all_reported() {
        // Removes a node type, removes a property, narrows an edge.
        let new = schema_from(
            r#"
schema:
  name: t
  version: 2
  node_types:
    Place:
      properties:
        name: { type: string }
  edge_types:
    Knows:
      from: Place
      to: Place
      properties: {}
"#,
        );
        let errs = validate_compatible(&base(), &new).unwrap_err();
        assert!(errs.contains(&EvolutionError::RemovedNodeType("Person".into())));
        // Knows.since is removed too; check that.
        assert!(errs.iter().any(|e| matches!(
            e,
            EvolutionError::RemovedEdgeProperty { edge_type, property }
                if edge_type == "Knows" && property == "since"
        )));
        // And edge endpoints narrowed (Person → Place).
        assert!(
            errs.iter()
                .any(|e| matches!(e, EvolutionError::NarrowedEdgeFrom { .. }))
        );
        assert!(errs.len() >= 4);
    }
}
