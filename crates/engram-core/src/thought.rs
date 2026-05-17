//! `ThoughtId` and `Thought` — the row-shape of `thoughts` in Postgres.

use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::{Metadata, Scope, Source, Tags};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ThoughtId(pub Uuid);

impl ThoughtId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }

    pub fn into_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for ThoughtId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Uuid> for ThoughtId {
    fn from(u: Uuid) -> Self {
        Self(u)
    }
}

impl fmt::Display for ThoughtId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl FromStr for ThoughtId {
    type Err = uuid::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Uuid::from_str(s).map(Self)
    }
}

/// Full thought row as read from or written to the database.
///
/// `content_fingerprint` is a SHA-256 of `content`, written at capture time
/// and used to dedup re-captures of identical content. Tag fields are
/// populated asynchronously by the tag drainer; until the first pass they
/// are `Tags::default()` / `None`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Thought {
    pub id: ThoughtId,
    pub scope: Scope,
    pub content: String,
    pub source: Source,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    pub metadata: Metadata,
    #[serde(with = "content_fingerprint_serde")]
    pub content_fingerprint: [u8; 32],
    pub tags: Tags,
    pub tags_extractor_model: Option<String>,
    pub tags_extractor_version: Option<i32>,
    #[serde(with = "time::serde::rfc3339::option")]
    pub tags_extracted_at: Option<OffsetDateTime>,
}

/// Serde helper for `content_fingerprint`.
///
/// Serializes as a lowercase 64-character hex string.
/// Deserializes from either hex (preferred) or standard padded base64
/// (fallback for clients that emit base64 byte arrays). The decoded length
/// is enforced to be exactly 32 bytes; anything else is a deserialization
/// error.
pub mod content_fingerprint_serde {
    use base64::{Engine, engine::general_purpose::STANDARD as B64};
    use serde::{Deserialize, Deserializer, Serializer, de::Error as _};

    pub fn serialize<S>(bytes: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;

        // Try hex first (the canonical encoding produced by `serialize`).
        if let Ok(decoded) = hex::decode(&s) {
            return decoded.try_into().map_err(|v: Vec<u8>| {
                D::Error::custom(format!(
                    "content_fingerprint must decode to 32 bytes, got {}",
                    v.len()
                ))
            });
        }

        // Fall back to standard padded base64 for callers emitting raw
        // byte arrays in JSON.
        let decoded = B64.decode(s.as_bytes()).map_err(|e| {
            D::Error::custom(format!(
                "content_fingerprint is neither hex nor base64: {e}"
            ))
        })?;
        decoded.try_into().map_err(|v: Vec<u8>| {
            D::Error::custom(format!(
                "content_fingerprint must decode to 32 bytes, got {}",
                v.len()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_thought() -> Thought {
        Thought {
            id: ThoughtId::from_str("550e8400-e29b-41d4-a716-446655440000").unwrap(),
            scope: Scope::new("work").unwrap(),
            content: "remember this".to_string(),
            source: Source::new("manual").unwrap(),
            created_at: OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap(),
            metadata: Metadata::from(json!({"client_name": "claude-code"})),
            content_fingerprint: [0xab; 32],
            tags: Tags::default(),
            tags_extractor_model: None,
            tags_extractor_version: None,
            tags_extracted_at: None,
        }
    }

    #[test]
    fn new_produces_v4_uuid() {
        let id = ThoughtId::new();
        assert_eq!(id.as_uuid().get_version_num(), 4);
    }

    #[test]
    fn fresh_ids_are_unique() {
        assert_ne!(ThoughtId::new(), ThoughtId::new());
    }

    #[test]
    fn parses_from_uuid_string() {
        let id: ThoughtId = "550e8400-e29b-41d4-a716-446655440000".parse().unwrap();
        assert_eq!(id.to_string(), "550e8400-e29b-41d4-a716-446655440000");
    }

    #[test]
    fn serde_roundtrip_is_transparent_uuid() {
        let id = ThoughtId::new();
        let json = serde_json::to_string(&id).unwrap();
        let parsed: ThoughtId = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, id);
    }

    #[test]
    fn thought_serde_roundtrip() {
        let t = make_thought();
        let s = serde_json::to_string(&t).unwrap();
        let parsed: Thought = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed, t);
    }

    #[test]
    fn content_fingerprint_serializes_as_lowercase_hex() {
        let t = make_thought();
        let v = serde_json::to_value(&t).unwrap();
        let hex_str = v["content_fingerprint"].as_str().unwrap();
        assert_eq!(hex_str.len(), 64);
        assert_eq!(hex_str, "ab".repeat(32));
        // Must be lowercase.
        assert_eq!(hex_str, hex_str.to_lowercase());
    }

    #[test]
    fn content_fingerprint_round_trips_via_base64() {
        // Confirm base64 deserialization is accepted for the fingerprint
        // field. Standard padded alphabet.
        use base64::{Engine, engine::general_purpose::STANDARD as B64};
        let bytes = [0x12u8; 32];
        let b64 = B64.encode(bytes);
        let mut v = serde_json::to_value(make_thought()).unwrap();
        v["content_fingerprint"] = serde_json::Value::String(b64);
        let parsed: Thought = serde_json::from_value(v).unwrap();
        assert_eq!(parsed.content_fingerprint, bytes);
    }

    #[test]
    fn content_fingerprint_rejects_wrong_length() {
        let mut v = serde_json::to_value(make_thought()).unwrap();
        // 30 bytes worth of hex (too short).
        v["content_fingerprint"] = serde_json::Value::String("ab".repeat(30));
        let err = serde_json::from_value::<Thought>(v).unwrap_err();
        assert!(
            err.to_string().contains("32 bytes"),
            "expected length error, got: {err}"
        );
    }

    #[test]
    fn content_fingerprint_rejects_garbage() {
        let mut v = serde_json::to_value(make_thought()).unwrap();
        v["content_fingerprint"] = serde_json::Value::String("not-hex-not-base64!@#$".to_string());
        let err = serde_json::from_value::<Thought>(v).unwrap_err();
        assert!(
            err.to_string().contains("neither hex nor base64")
                || err.to_string().contains("32 bytes"),
            "expected decode error, got: {err}"
        );
    }

    #[test]
    fn tags_extracted_at_round_trips_as_rfc3339_option() {
        let mut t = make_thought();
        let when = OffsetDateTime::from_unix_timestamp(1_710_000_000).unwrap();
        t.tags_extracted_at = Some(when);
        t.tags_extractor_model = Some("vllm/qwen2.5-7b-instruct".into());
        t.tags_extractor_version = Some(1);
        let s = serde_json::to_string(&t).unwrap();
        let parsed: Thought = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.tags_extracted_at, Some(when));
        assert_eq!(parsed.tags_extractor_version, Some(1));
    }
}
