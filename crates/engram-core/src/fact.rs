//! `Fact` — the row shape of `facts` in Postgres.
//!
//! `ExtractedFact` (in `extractor.rs`) is the *write* shape returned by an
//! `Extractor`. `Fact` is the *read* shape: it carries the id, timestamps,
//! and provenance fields filled in by the database. Keeping them distinct
//! avoids confusion about which lifecycle state a value is in.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::{Scope, ThoughtId};

/// A persisted fact, as read from the `facts` table. `source_run_id` is a
/// raw `Uuid` rather than a typed newtype because consumers (search-result
/// builders, MCP JSON serializers) don't need the type discipline that the
/// `engram-storage::RunId` insert-side newtype provides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    pub id: Uuid,
    pub scope: Scope,
    pub statement: String,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub source_thought_id: ThoughtId,
    pub extractor_model: String,
    pub extractor_version: i32,
    pub source_run_id: Option<Uuid>,
    pub confidence: f32,
    /// Three-band routing flag. False for direct-source / high-confidence
    /// facts; true for the "stored but flagged" middle confidence band
    /// (`[reflector] review_queue_below ≤ confidence < min_confidence_to_store`).
    /// Default false keeps M1/M2 row shapes consistent.
    #[serde(default)]
    pub flagged: bool,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn make_fact() -> Fact {
        Fact {
            id: Uuid::from_str("550e8400-e29b-41d4-a716-446655440000").unwrap(),
            scope: Scope::new("work").unwrap(),
            statement: "Engram uses pgvector".to_string(),
            subject: Some("Engram".to_string()),
            predicate: Some("uses".to_string()),
            object: Some("pgvector".to_string()),
            source_thought_id: ThoughtId::from(
                Uuid::from_str("00000000-0000-0000-0000-000000000001").unwrap(),
            ),
            extractor_model: "vllm/qwen2.5-7b-instruct".to_string(),
            extractor_version: 1,
            source_run_id: Some(Uuid::from_str("00000000-0000-0000-0000-000000000002").unwrap()),
            confidence: 0.92,
            flagged: false,
            created_at: OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap(),
        }
    }

    #[test]
    fn fact_serde_roundtrip() {
        let f = make_fact();
        let json = serde_json::to_string(&f).unwrap();
        let parsed: Fact = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, f);
    }

    #[test]
    fn fact_optional_triple_can_be_absent() {
        let mut f = make_fact();
        f.subject = None;
        f.predicate = None;
        f.object = None;
        let json = serde_json::to_string(&f).unwrap();
        let parsed: Fact = serde_json::from_str(&json).unwrap();
        assert!(parsed.subject.is_none());
        assert!(parsed.predicate.is_none());
        assert!(parsed.object.is_none());
    }

    #[test]
    fn fact_row_deserializes_with_flagged_field() {
        let json_without = r#"{
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "scope": "work",
            "statement": "Engram uses pgvector",
            "subject": null,
            "predicate": null,
            "object": null,
            "source_thought_id": "00000000-0000-0000-0000-000000000001",
            "extractor_model": "x",
            "extractor_version": 1,
            "source_run_id": null,
            "confidence": 0.9,
            "created_at": "2026-05-15T00:00:00Z"
        }"#;
        let parsed: Fact = serde_json::from_str(json_without).unwrap();
        assert!(!parsed.flagged, "missing flagged field should default to false");

        let json_with = r#"{
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "scope": "work",
            "statement": "x",
            "subject": null,
            "predicate": null,
            "object": null,
            "source_thought_id": "00000000-0000-0000-0000-000000000001",
            "extractor_model": "x",
            "extractor_version": 1,
            "source_run_id": null,
            "confidence": 0.75,
            "flagged": true,
            "created_at": "2026-05-15T00:00:00Z"
        }"#;
        let parsed: Fact = serde_json::from_str(json_with).unwrap();
        assert!(parsed.flagged);
    }
}
