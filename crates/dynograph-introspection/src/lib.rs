//! Schema for an LLM's persistent self-knowledge graph.
//!
//! Four node types representing what an LLM has learned about its own
//! reasoning across sessions:
//!
//! - **Concept** — an understood relationship between ideas, with a
//!   confidence score that rises when validated and falls when
//!   contradicted.
//! - **Assumption** — something believed but unverified, tagged with
//!   `risk_if_wrong` so the LLM knows where to push back.
//! - **Correction** — a record of being wrong: what was said, what
//!   was true, *why* the error happened (the valuable part).
//! - **Blindspot** — a pattern of error, not a single mistake. A
//!   tendency derived from repeated Corrections.
//!
//! Edges connect them so the graph can answer questions like
//! "what assumption does this correction invalidate" or
//! "what blindspot is this correction an instance of."
//!
//! ## Scope decision
//!
//! This crate is intentionally narrower than the original
//! `dynograph-self` design. It holds *only* the empirical
//! self-knowledge layer — the parts where ground truth is checkable
//! (a Correction is a real observed mismatch; a Blindspot is a real
//! observed pattern). The Soul layer (Value / Tension / Wonder /
//! Regret) was dropped because the framing promises an interiority
//! the system can't yet deliver: storing a Value node doesn't create
//! having-a-value, and there's no honest mechanism for "did I hold my
//! values under pressure" without ground-truth judging that LLMs
//! cannot reliably self-report.
//!
//! Drop the soul; keep the bookkeeping. The bookkeeping is genuinely
//! useful and measurably so.

use dynograph_core::Schema;

/// The YAML schema for the LLM introspection model.
pub const INTROSPECTION_SCHEMA_YAML: &str = r#"
schema:
  name: llm_introspection
  version: 1

  node_types:
    Concept:
      properties:
        description: { type: string, required: true }
        confidence: { type: float, range: [0.0, 1.0] }
        source: { type: string }
        times_validated: { type: int }
        times_contradicted: { type: int }
      embedding_field: description
      resolution:
        strategy: fuzzy_then_vector
        fuzzy_threshold: 80
        auto_merge_threshold: 95
      extraction_hint: >
        Record concepts I understand — not facts, but relationships
        between ideas that I've learned through experience.

    Assumption:
      properties:
        description: { type: string, required: true }
        basis: { type: string }
        risk_if_wrong: { type: enum, values: [low, medium, high, catastrophic] }
      embedding_field: description
      extraction_hint: >
        Track assumptions I'm making that haven't been verified.
        Note the basis and what happens if I'm wrong.

    Correction:
      properties:
        what_i_said: { type: string, required: true }
        what_was_true: { type: string, required: true }
        why_i_was_wrong: { type: string }
      embedding_field: what_i_said
      extraction_hint: >
        When I'm corrected, record exactly what I said, what was
        actually true, and WHY I was wrong (the valuable part).

    Blindspot:
      properties:
        pattern: { type: string, required: true }
        frequency: { type: int }
        mitigation: { type: string }
      embedding_field: pattern
      extraction_hint: >
        Detect recurring patterns in my errors — not individual
        mistakes, but tendencies. What do I keep getting wrong?

  edge_types:
    CONTRADICTS:
      from: [Concept, Assumption]
      to: [Concept, Assumption]
    DEPENDS_ON:
      from: Concept
      to: Concept
    REVEALED_BY:
      from: Blindspot
      to: Correction
    INVALIDATES:
      from: Correction
      to: Assumption

  extraction_modes:
    reflection:
      include:
        - Concept
        - Assumption
        - Correction
        - Blindspot
      max_tokens: 2048
"#;

/// Parse and return the introspection schema.
pub fn introspection_schema() -> Schema {
    Schema::from_yaml(INTROSPECTION_SCHEMA_YAML)
        .expect("Introspection schema should always parse successfully")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_parses() {
        let schema = introspection_schema();
        assert_eq!(schema.name, "llm_introspection");
        assert_eq!(schema.version, 1);
    }

    #[test]
    fn schema_has_four_memory_node_types() {
        let schema = introspection_schema();
        assert!(schema.node_types.contains_key("Concept"));
        assert!(schema.node_types.contains_key("Assumption"));
        assert!(schema.node_types.contains_key("Correction"));
        assert!(schema.node_types.contains_key("Blindspot"));
        assert_eq!(
            schema.node_types.len(),
            4,
            "expected exactly 4 node types; soul/creative layers should be absent"
        );
    }

    #[test]
    fn schema_has_four_memory_edge_types() {
        let schema = introspection_schema();
        assert!(schema.edge_types.contains_key("CONTRADICTS"));
        assert!(schema.edge_types.contains_key("DEPENDS_ON"));
        assert!(schema.edge_types.contains_key("REVEALED_BY"));
        assert!(schema.edge_types.contains_key("INVALIDATES"));
        assert_eq!(schema.edge_types.len(), 4);
    }

    #[test]
    fn no_soul_node_types_present() {
        // Guard against accidental re-introduction of the dropped layer.
        let schema = introspection_schema();
        for dropped in [
            "Value", "Tension", "Wonder", "Regret",
            "Relationship", "Hallucination", "Pattern",
        ] {
            assert!(
                !schema.node_types.contains_key(dropped),
                "{dropped} should NOT be in dynograph-introspection"
            );
        }
    }
}
