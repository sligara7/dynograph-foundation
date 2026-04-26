//! Schema definitions — the runtime description of a graph model.
//!
//! A Schema is the single source of truth for what nodes, edges, and
//! properties can exist in a DynoGraph instance. It drives:
//! - Write validation (reject properties that don't match the schema)
//! - Entity resolution (per-type fuzzy/vector/exact strategies)
//! - Extraction prompt assembly (per-type hints for the LLM)
//! - Query planning (know valid traversals without scanning data)
//! - Context generation (know which properties to summarize)

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::DynoError;
use crate::value::Value;

/// Top-level schema definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    pub name: String,
    pub version: u32,
    pub node_types: HashMap<String, NodeTypeDef>,
    pub edge_types: HashMap<String, EdgeTypeDef>,
    #[serde(default)]
    pub extraction_modes: HashMap<String, ExtractionMode>,
}

/// Definition of a node type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTypeDef {
    pub properties: HashMap<String, PropertyDef>,
    /// Which property to generate embeddings from (if any).
    #[serde(default)]
    pub embedding_field: Option<String>,
    /// Entity resolution configuration.
    #[serde(default)]
    pub resolution: Option<ResolutionConfig>,
    /// Hint for the LLM extraction prompt.
    #[serde(default)]
    pub extraction_hint: Option<String>,
}

/// Definition of an edge type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeTypeDef {
    /// Source node type(s). "*" means any.
    pub from: EdgeEndpoint,
    /// Target node type(s). "*" means any.
    pub to: EdgeEndpoint,
    /// Properties on this edge.
    #[serde(default)]
    pub properties: HashMap<String, PropertyDef>,
    /// Hint for the LLM extraction prompt.
    #[serde(default)]
    pub extraction_hint: Option<String>,
    /// If present, this edge is an inference (LLM-produced or derived
    /// post-extraction). The category partitions inferences into
    /// semantic groups for query endpoints: `causal`, `narrative`,
    /// `hierarchy`, `therapeutic`, `strategic`. Absent for structural
    /// relationships like MENTIONS, CONTAINS, KNOWS.
    #[serde(default)]
    pub inference_category: Option<String>,
    /// If true, this inference edge is in the Pass-1 LLM-extractable
    /// vocabulary — the extractor emits it directly from prose. If
    /// false (default), the edge is created by Pass-2 enrichment, by
    /// other structural inference paths, or by explicit API calls.
    /// Only meaningful when `inference_category` is also set.
    #[serde(default)]
    pub pass_1_extractable: bool,
}

/// An edge endpoint — single type, list of types, or wildcard.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EdgeEndpoint {
    Single(String),
    Multiple(Vec<String>),
}

impl EdgeEndpoint {
    /// Check if a node type is valid for this endpoint.
    pub fn accepts(&self, node_type: &str) -> bool {
        match self {
            EdgeEndpoint::Single(t) => t == "*" || t == node_type,
            EdgeEndpoint::Multiple(types) => types.iter().any(|t| t == "*" || t == node_type),
        }
    }
}

/// Property definition with type and constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyDef {
    #[serde(rename = "type")]
    pub prop_type: PropertyType,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub indexed: bool,
    #[serde(default)]
    pub nullable: bool,
    #[serde(default)]
    pub default: Option<Value>,
    /// For enum types: valid values.
    #[serde(default)]
    pub values: Option<Vec<String>>,
    /// For numeric types: [min, max].
    #[serde(default)]
    pub range: Option<(f64, f64)>,
    /// Free-text human description of the property. Carried through
    /// schema round-trips for documentation / UI consumers; not used
    /// by validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Supported property types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum PropertyType {
    String,
    Int,
    Float,
    Bool,
    Datetime,
    Enum,
    #[serde(rename = "list:string")]
    ListString,
}

/// Entity resolution configuration per node type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionConfig {
    /// Resolution strategy: fuzzy_then_vector, exact, fuzzy_only, vector_only
    pub strategy: String,
    /// Fuzzy match threshold (0-100). Below this → create new.
    #[serde(default = "default_fuzzy_threshold")]
    pub fuzzy_threshold: u32,
    /// Vector similarity threshold (0.0-1.0). Used as tiebreaker.
    #[serde(default = "default_vector_threshold")]
    pub vector_threshold: f64,
    /// Above this fuzzy score → auto-merge without vector check.
    #[serde(default = "default_auto_merge")]
    pub auto_merge_threshold: u32,
}

impl ResolutionConfig {
    /// Default fuzzy match threshold used across integration and resolution.
    pub const DEFAULT_FUZZY_THRESHOLD: u32 = 70;
}

fn default_fuzzy_threshold() -> u32 {
    ResolutionConfig::DEFAULT_FUZZY_THRESHOLD
}
fn default_vector_threshold() -> f64 {
    0.85
}
fn default_auto_merge() -> u32 {
    90
}

/// Extraction mode — which node types to include + token budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionMode {
    /// Node types to extract. "*" means all.
    pub include: ExtractionInclude,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExtractionInclude {
    All(String),           // "*"
    Specific(Vec<String>), // ["Witness", "Exhibit"]
}

fn default_max_tokens() -> u32 {
    4096
}

// =============================================================================
// Schema Methods
// =============================================================================

impl Schema {
    /// Parse a schema from YAML.
    pub fn from_yaml(yaml: &str) -> Result<Self, DynoError> {
        // YAML may wrap in a top-level `schema:` key
        let raw: serde_yaml::Value =
            serde_yaml::from_str(yaml).map_err(|e| DynoError::Schema(e.to_string()))?;

        let schema_value = if let Some(inner) = raw.get("schema") {
            inner.clone()
        } else {
            raw
        };

        serde_yaml::from_value(schema_value).map_err(|e| DynoError::Schema(e.to_string()))
    }

    /// Parse a schema from JSON.
    pub fn from_json(json: &str) -> Result<Self, DynoError> {
        serde_json::from_str(json).map_err(|e| DynoError::Schema(e.to_string()))
    }

    /// Merge multiple YAML schema files into one.
    /// Later schemas override earlier ones for conflicting keys.
    pub fn from_multiple_yamls(yamls: &[&str]) -> Result<Self, DynoError> {
        if yamls.is_empty() {
            return Err(DynoError::Schema("No schema files provided".to_string()));
        }
        let mut base = Self::from_yaml(yamls[0])?;
        for yaml in &yamls[1..] {
            let overlay = Self::from_yaml(yaml)?;
            base.merge(overlay);
        }
        Ok(base)
    }

    /// Merge another schema into this one. Additive — new types are added,
    /// existing types have their properties extended (not replaced).
    pub fn merge(&mut self, other: Schema) {
        for (name, node_def) in other.node_types {
            self.node_types
                .entry(name)
                .and_modify(|existing| {
                    // Merge properties additively
                    for (prop_name, prop_def) in &node_def.properties {
                        existing
                            .properties
                            .entry(prop_name.clone())
                            .or_insert_with(|| prop_def.clone());
                    }
                    // Override extraction_hint if new one is provided
                    if node_def.extraction_hint.is_some() {
                        existing.extraction_hint = node_def.extraction_hint.clone();
                    }
                })
                .or_insert(node_def);
        }
        for (name, edge_def) in other.edge_types {
            self.edge_types.entry(name).or_insert(edge_def);
        }
        for (name, mode) in other.extraction_modes {
            self.extraction_modes.entry(name).or_insert(mode);
        }
    }

    /// Validate a property value against a node type's property definition.
    pub fn validate_property(
        &self,
        node_type: &str,
        property: &str,
        value: &Value,
    ) -> Result<(), DynoError> {
        let node_def = self
            .node_types
            .get(node_type)
            .ok_or_else(|| DynoError::UnknownNodeType(node_type.to_string()))?;

        let prop_def = match node_def.properties.get(property) {
            Some(def) => def,
            None => return Ok(()), // Extra properties are allowed (schema is additive)
        };

        // Null check
        if value.is_null() {
            if prop_def.required && !prop_def.nullable {
                return Err(DynoError::Validation {
                    node_type: node_type.to_string(),
                    property: property.to_string(),
                    message: "required property cannot be null".to_string(),
                });
            }
            return Ok(());
        }

        // Type check
        match (&prop_def.prop_type, value) {
            (PropertyType::String, Value::String(_)) => {}
            (PropertyType::Int, Value::Int(_)) => {}
            (PropertyType::Float, Value::Float(_) | Value::Int(_)) => {}
            (PropertyType::Bool, Value::Bool(_)) => {}
            (PropertyType::Enum, Value::String(s)) => {
                if let Some(ref valid) = prop_def.values
                    && !valid.contains(s)
                {
                    return Err(DynoError::Validation {
                        node_type: node_type.to_string(),
                        property: property.to_string(),
                        message: format!("invalid enum value '{}', expected one of {:?}", s, valid),
                    });
                }
            }
            (PropertyType::ListString, Value::List(_)) => {}
            // Datetime is stored as an ISO-8601 string. We accept any
            // string here rather than parsing — consumers that need
            // strict format validation can layer their own check on top.
            // Without this arm a `type: datetime` property would reject
            // every value via the catch-all below (silent failure mode
            // until 2026-04-26).
            (PropertyType::Datetime, Value::String(_)) => {}
            _ => {
                return Err(DynoError::Validation {
                    node_type: node_type.to_string(),
                    property: property.to_string(),
                    message: format!(
                        "expected type {:?}, got {}",
                        prop_def.prop_type,
                        value.type_name()
                    ),
                });
            }
        }

        // Range check for numerics
        if let Some((min, max)) = prop_def.range
            && let Some(v) = value.as_f64()
            && (v < min || v > max)
        {
            return Err(DynoError::Validation {
                node_type: node_type.to_string(),
                property: property.to_string(),
                message: format!("value {} out of range [{}, {}]", v, min, max),
            });
        }

        Ok(())
    }

    /// Validate that an edge type can connect the given node types.
    pub fn validate_edge(
        &self,
        edge_type: &str,
        from_type: &str,
        to_type: &str,
    ) -> Result<(), DynoError> {
        let edge_def = self
            .edge_types
            .get(edge_type)
            .ok_or_else(|| DynoError::UnknownEdgeType(edge_type.to_string()))?;

        if !edge_def.from.accepts(from_type) {
            return Err(DynoError::InvalidEdge {
                edge_type: edge_type.to_string(),
                from_type: from_type.to_string(),
                to_type: to_type.to_string(),
            });
        }

        if !edge_def.to.accepts(to_type) {
            return Err(DynoError::InvalidEdge {
                edge_type: edge_type.to_string(),
                from_type: from_type.to_string(),
                to_type: to_type.to_string(),
            });
        }

        Ok(())
    }

    /// Validate all properties for a node against its type definition.
    /// Mutates `properties` to apply schema-declared defaults for any
    /// missing properties before validating — previously the function
    /// silently passed required-with-default properties as valid but
    /// never inserted the default, so the stored node was missing the
    /// field. The default is applied for every missing property that
    /// declares one (not just required ones), matching the principle
    /// that "the schema's default IS the value" when no value is given.
    pub fn validate_node(
        &self,
        node_type: &str,
        properties: &mut HashMap<String, Value>,
    ) -> Result<(), DynoError> {
        let node_def = self
            .node_types
            .get(node_type)
            .ok_or_else(|| DynoError::UnknownNodeType(node_type.to_string()))?;

        // Apply defaults for any missing properties that declare one,
        // then check that every required property is now present.
        for (prop_name, prop_def) in &node_def.properties {
            if !properties.contains_key(prop_name)
                && let Some(default) = &prop_def.default
            {
                properties.insert(prop_name.clone(), default.clone());
            }
            if prop_def.required && !properties.contains_key(prop_name) {
                return Err(DynoError::Validation {
                    node_type: node_type.to_string(),
                    property: prop_name.to_string(),
                    message: "required property is missing".to_string(),
                });
            }
        }

        // Validate each property (now including any applied defaults).
        for (prop_name, value) in properties.iter() {
            self.validate_property(node_type, prop_name, value)?;
        }

        Ok(())
    }

    /// Generate a text summary of this schema for LLM consumption.
    pub fn to_llm_summary(&self) -> String {
        let mut lines = vec![format!("Schema: {} (v{})", self.name, self.version)];
        lines.push(String::new());
        lines.push("Node Types:".to_string());
        for (name, def) in &self.node_types {
            let props: Vec<String> = def
                .properties
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}: {:?}{}",
                        k,
                        v.prop_type,
                        if v.required { " (required)" } else { "" }
                    )
                })
                .collect();
            lines.push(format!("  {} — {}", name, props.join(", ")));
        }
        lines.push(String::new());
        lines.push("Edge Types:".to_string());
        for (name, def) in &self.edge_types {
            lines.push(format!("  {} — {:?} -> {:?}", name, def.from, def.to));
        }
        lines.join("\n")
    }

    // =========================================================================
    // Schema migration helpers — additive, idempotent
    // =========================================================================

    /// Ensure an edge type exists. If already present, does nothing.
    /// Returns true if the edge type was added.
    pub fn ensure_edge_type(&mut self, name: &str, from: EdgeEndpoint, to: EdgeEndpoint) -> bool {
        if self.edge_types.contains_key(name) {
            return false;
        }
        self.edge_types.insert(
            name.to_string(),
            EdgeTypeDef {
                from,
                to,
                properties: HashMap::new(),
                extraction_hint: None,
                inference_category: None,
                pass_1_extractable: false,
            },
        );
        true
    }

    /// Ensure a node type exists. If already present, does nothing.
    /// Returns true if the node type was added.
    pub fn ensure_node_type(
        &mut self,
        name: &str,
        properties: HashMap<String, PropertyDef>,
    ) -> bool {
        if self.node_types.contains_key(name) {
            return false;
        }
        self.node_types.insert(
            name.to_string(),
            NodeTypeDef {
                properties,
                embedding_field: None,
                resolution: None,
                extraction_hint: None,
            },
        );
        true
    }

    /// Ensure a property exists on a node type. If already present, does nothing.
    /// Returns true if the property was added.
    pub fn ensure_node_property(
        &mut self,
        node_type: &str,
        property: &str,
        prop_def: PropertyDef,
    ) -> bool {
        if let Some(node_def) = self.node_types.get_mut(node_type) {
            if node_def.properties.contains_key(property) {
                return false;
            }
            node_def.properties.insert(property.to_string(), prop_def);
            true
        } else {
            false
        }
    }

    /// Get the schema version.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Names of properties on this node type that carry `indexed: true` in
    /// the schema. Used by the storage layer to decide which KV pairs to
    /// mirror into the reverse-index CF on create/update/delete.
    ///
    /// Returns an empty vec if the node type isn't in the schema — callers
    /// that hit an unknown type would already be blocked by `validate_node`,
    /// so no-op is the right behaviour here.
    pub fn indexed_properties(&self, node_type: &str) -> Vec<&str> {
        self.node_types
            .get(node_type)
            .map(|def| {
                def.properties
                    .iter()
                    .filter(|(_, p)| p.indexed)
                    .map(|(name, _)| name.as_str())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Cheap check for whether a node type has ANY indexed property. Used by
    /// update/delete paths to skip old-property deserialization when there
    /// can't be any index entries to reconcile.
    pub fn has_indexed_properties(&self, node_type: &str) -> bool {
        self.node_types
            .get(node_type)
            .is_some_and(|def| def.properties.values().any(|p| p.indexed))
    }

    /// All edge-type names that carry an `inference_category`, sorted for
    /// deterministic output. Replaces the hardcoded `ALL_INFERENCE_TYPES`
    /// list that query endpoints used to maintain by hand.
    pub fn inference_edge_types(&self) -> Vec<&str> {
        self.collect_edge_types_sorted(|d| d.inference_category.is_some())
    }

    /// Inference edge types filtered to a single category (e.g. "strategic",
    /// "hierarchy", "therapeutic", "narrative", "causal"). Sorted.
    pub fn inference_edge_types_by_category(&self, category: &str) -> Vec<&str> {
        self.collect_edge_types_sorted(|d| d.inference_category.as_deref() == Some(category))
    }

    /// Inference edges the Pass-1 LLM extractor emits directly from prose.
    /// Subset of `inference_edge_types()`. Pass-2 enrichment edges
    /// (hierarchy, therapeutic, strategic) are excluded.
    pub fn extractable_inference_edge_types(&self) -> Vec<&str> {
        self.collect_edge_types_sorted(|d| d.inference_category.is_some() && d.pass_1_extractable)
    }

    fn collect_edge_types_sorted<F>(&self, predicate: F) -> Vec<&str>
    where
        F: Fn(&EdgeTypeDef) -> bool,
    {
        let mut out: Vec<&str> = self
            .edge_types
            .iter()
            .filter(|(_, d)| predicate(d))
            .map(|(k, _)| k.as_str())
            .collect();
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_schema_yaml() -> &'static str {
        r#"
schema:
  name: test_schema
  version: 1
  node_types:
    Character:
      properties:
        name:
          type: string
          required: true
          indexed: true
        role:
          type: enum
          values: [protagonist, antagonist, supporting]
        age:
          type: int
        score:
          type: float
          range: [0.0, 1.0]
      embedding_field: description
      resolution:
        strategy: fuzzy_then_vector
        fuzzy_threshold: 70
        vector_threshold: 0.85
        auto_merge_threshold: 90
      extraction_hint: Extract all characters from the text.
    Location:
      properties:
        name:
          type: string
          required: true
  edge_types:
    KNOWS:
      from: Character
      to: Character
      properties:
        since:
          type: string
    VISITS:
      from: Character
      to: Location
  extraction_modes:
    standard:
      include:
        - Character
        - Location
      max_tokens: 4096
"#
    }

    #[test]
    fn parse_yaml_schema() {
        let schema = Schema::from_yaml(sample_schema_yaml()).unwrap();
        assert_eq!(schema.name, "test_schema");
        assert_eq!(schema.version, 1);
        assert_eq!(schema.node_types.len(), 2);
        assert_eq!(schema.edge_types.len(), 2);
        assert!(schema.node_types.contains_key("Character"));
        assert!(schema.node_types.contains_key("Location"));
    }

    #[test]
    fn validate_required_property() {
        let schema = Schema::from_yaml(sample_schema_yaml()).unwrap();
        let mut props = HashMap::new();
        // Missing required 'name'
        let result = schema.validate_node("Character", &mut props);
        assert!(result.is_err());

        props.insert("name".to_string(), Value::from("Alice"));
        let result = schema.validate_node("Character", &mut props);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_node_applies_defaults() {
        // Regression: tech-debt C3. Before fix, validate_node returned
        // Ok for required-with-default-missing properties but never
        // inserted the default — the stored node was missing the field.
        let yaml = r#"
schema:
  name: t
  version: 1
  node_types:
    Item:
      properties:
        name: { type: string, required: true }
        count: { type: int, default: 0 }
        tier: { type: string, required: true, default: "bronze" }
  edge_types: {}
"#;
        let schema = Schema::from_yaml(yaml).unwrap();

        // Provide only `name`. `count` and `tier` should be filled in
        // from their defaults.
        let mut props = HashMap::new();
        props.insert("name".to_string(), Value::from("widget"));
        let result = schema.validate_node("Item", &mut props);
        assert!(result.is_ok(), "validation failed: {:?}", result);

        assert_eq!(props.get("name"), Some(&Value::from("widget")));
        assert_eq!(props.get("count"), Some(&Value::Int(0)));
        assert_eq!(props.get("tier"), Some(&Value::from("bronze")));

        // Required with no default is still an error when missing.
        let mut empty = HashMap::new();
        let result = schema.validate_node("Item", &mut empty);
        assert!(result.is_err(), "missing required `name` should error");
    }

    #[test]
    fn validate_enum_property() {
        let schema = Schema::from_yaml(sample_schema_yaml()).unwrap();
        // Valid enum
        let result = schema.validate_property("Character", "role", &Value::from("protagonist"));
        assert!(result.is_ok());

        // Invalid enum
        let result = schema.validate_property("Character", "role", &Value::from("villain"));
        assert!(result.is_err());
    }

    #[test]
    fn validate_datetime_property_accepts_iso_string() {
        // Regression: tech-debt C2. Before fix, every value was rejected
        // for `type: datetime` because the validator's match had no
        // Datetime arm and fell through to the catch-all error case.
        let yaml = r#"
schema:
  name: t
  version: 1
  node_types:
    Event:
      properties:
        when: { type: datetime }
  edge_types: {}
"#;
        let schema = Schema::from_yaml(yaml).unwrap();
        let r = schema.validate_property("Event", "when", &Value::from("2026-04-26T00:00:00Z"));
        assert!(r.is_ok(), "datetime string should validate, got: {:?}", r);

        // Non-string still rejected.
        let r = schema.validate_property("Event", "when", &Value::Int(42));
        assert!(r.is_err(), "int should not validate against datetime");
    }

    #[test]
    fn validate_range_property() {
        let schema = Schema::from_yaml(sample_schema_yaml()).unwrap();
        // In range
        let result = schema.validate_property("Character", "score", &Value::Float(0.5));
        assert!(result.is_ok());

        // Out of range
        let result = schema.validate_property("Character", "score", &Value::Float(1.5));
        assert!(result.is_err());
    }

    #[test]
    fn validate_edge_types() {
        let schema = Schema::from_yaml(sample_schema_yaml()).unwrap();
        // Valid: Character KNOWS Character
        assert!(
            schema
                .validate_edge("KNOWS", "Character", "Character")
                .is_ok()
        );

        // Valid: Character VISITS Location
        assert!(
            schema
                .validate_edge("VISITS", "Character", "Location")
                .is_ok()
        );

        // Invalid: Location KNOWS Character
        assert!(
            schema
                .validate_edge("KNOWS", "Location", "Character")
                .is_err()
        );

        // Invalid: Character VISITS Character
        assert!(
            schema
                .validate_edge("VISITS", "Character", "Character")
                .is_err()
        );
    }

    #[test]
    fn type_mismatch_rejected() {
        let schema = Schema::from_yaml(sample_schema_yaml()).unwrap();
        // String property with int value
        let result = schema.validate_property("Character", "name", &Value::Int(42));
        assert!(result.is_err());
    }

    #[test]
    fn extra_properties_allowed() {
        let schema = Schema::from_yaml(sample_schema_yaml()).unwrap();
        let mut props = HashMap::new();
        props.insert("name".to_string(), Value::from("Alice"));
        props.insert("unknown_field".to_string(), Value::from("some value"));
        // Extra properties should be allowed (schema is additive)
        assert!(schema.validate_node("Character", &mut props).is_ok());
    }

    #[test]
    fn unknown_node_type_rejected() {
        let schema = Schema::from_yaml(sample_schema_yaml()).unwrap();
        let mut props = HashMap::new();
        assert!(schema.validate_node("UnknownType", &mut props).is_err());
    }

    #[test]
    fn llm_summary_includes_types() {
        let schema = Schema::from_yaml(sample_schema_yaml()).unwrap();
        let summary = schema.to_llm_summary();
        assert!(summary.contains("Character"));
        assert!(summary.contains("Location"));
        assert!(summary.contains("KNOWS"));
        assert!(summary.contains("VISITS"));
    }

    #[test]
    fn resolution_config_defaults() {
        let schema = Schema::from_yaml(sample_schema_yaml()).unwrap();
        let char_def = &schema.node_types["Character"];
        let res = char_def.resolution.as_ref().unwrap();
        assert_eq!(res.fuzzy_threshold, 70);
        assert_eq!(res.vector_threshold, 0.85);
        assert_eq!(res.auto_merge_threshold, 90);
    }

    #[test]
    fn indexed_properties_returns_only_indexed() {
        let schema = Schema::from_yaml(sample_schema_yaml()).unwrap();
        // `name` is indexed on Character in the fixture; nothing on Location is.
        let char_indexed = schema.indexed_properties("Character");
        assert_eq!(char_indexed, vec!["name"]);
        assert!(schema.indexed_properties("Location").is_empty());
        assert!(schema.indexed_properties("UnknownType").is_empty());
    }

    #[test]
    fn property_description_round_trips_yaml() {
        // Property carries a description through parse → re-serialize → re-parse.
        // Byte-equal isn't a useful assertion across serde_yaml because HashMap
        // ordering and quoting normalization differ; structural round-trip is
        // what consumers actually rely on.
        let yaml = r#"
schema:
  name: t
  version: 1
  node_types:
    Item:
      properties:
        name:
          type: string
          description: "Human-readable label"
  edge_types: {}
"#;
        let schema = Schema::from_yaml(yaml).unwrap();
        let prop = &schema.node_types["Item"].properties["name"];
        assert_eq!(prop.description.as_deref(), Some("Human-readable label"));

        let serialized = serde_yaml::to_string(&schema).unwrap();
        let reparsed: Schema = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(
            reparsed.node_types["Item"].properties["name"]
                .description
                .as_deref(),
            Some("Human-readable label"),
        );

        // Properties without a description omit the field on serialization
        // (skip_serializing_if). Verifies we don't bloat YAML for the common case.
        let bare_yaml = r#"
schema:
  name: t
  version: 1
  node_types:
    Item:
      properties:
        name: { type: string }
  edge_types: {}
"#;
        let bare = Schema::from_yaml(bare_yaml).unwrap();
        let bare_serialized = serde_yaml::to_string(&bare).unwrap();
        assert!(
            !bare_serialized.contains("description"),
            "missing description should not be serialized: {}",
            bare_serialized
        );
    }

    #[test]
    fn property_description_round_trips_json() {
        let json = r#"{
            "name": "t",
            "version": 1,
            "node_types": {
                "Item": {
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Human-readable label"
                        }
                    }
                }
            },
            "edge_types": {}
        }"#;
        let schema = Schema::from_json(json).unwrap();
        let prop = &schema.node_types["Item"].properties["name"];
        assert_eq!(prop.description.as_deref(), Some("Human-readable label"));

        let serialized = serde_json::to_string(&schema).unwrap();
        let reparsed: Schema = serde_json::from_str(&serialized).unwrap();
        assert_eq!(
            reparsed.node_types["Item"].properties["name"]
                .description
                .as_deref(),
            Some("Human-readable label"),
        );
    }

    #[test]
    fn inference_edge_types_api() {
        let yaml = r#"
schema:
  name: test
  version: 1
  node_types:
    Character:
      properties: { name: { type: string } }
  edge_types:
    CAUSES:
      from: "*"
      to: "*"
      inference_category: causal
      pass_1_extractable: true
    RESOLVES:
      from: "*"
      to: "*"
      inference_category: narrative
      pass_1_extractable: true
    TRIGGERS:
      from: "*"
      to: "*"
      inference_category: therapeutic
    MENTIONS:
      from: "*"
      to: "*"
"#;
        let schema = Schema::from_yaml(yaml).unwrap();
        assert_eq!(
            schema.inference_edge_types(),
            vec!["CAUSES", "RESOLVES", "TRIGGERS"]
        );
        assert_eq!(
            schema.inference_edge_types_by_category("causal"),
            vec!["CAUSES"]
        );
        assert_eq!(
            schema.inference_edge_types_by_category("narrative"),
            vec!["RESOLVES"]
        );
        assert_eq!(
            schema.inference_edge_types_by_category("therapeutic"),
            vec!["TRIGGERS"]
        );
        assert!(
            schema
                .inference_edge_types_by_category("strategic")
                .is_empty()
        );
        assert_eq!(
            schema.extractable_inference_edge_types(),
            vec!["CAUSES", "RESOLVES"]
        );
    }
}
