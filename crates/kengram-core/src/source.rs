//! Provenance label for a captured thought. Required on every capture.
//! Convention: `"manual"`, `"agent:claude-code"`, `"reflector"`, etc. — the
//! prefix-before-colon names the kind of writer, the suffix names the
//! specific writer. Enforcement is "non-empty and reasonable length."

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Source(String);

#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("source must be non-empty")]
    Empty,
    #[error("source must be at most 128 characters (got {0})")]
    TooLong(usize),
}

impl Source {
    pub const MAX_LEN: usize = 128;

    pub fn new(s: impl Into<String>) -> Result<Self, SourceError> {
        let s = s.into();
        if s.is_empty() {
            return Err(SourceError::Empty);
        }
        if s.len() > Self::MAX_LEN {
            return Err(SourceError::TooLong(s.len()));
        }
        Ok(Self(s))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty() {
        assert!(matches!(Source::new(""), Err(SourceError::Empty)));
    }

    #[test]
    fn rejects_overlong() {
        let long = "x".repeat(Source::MAX_LEN + 1);
        assert!(matches!(Source::new(long), Err(SourceError::TooLong(_))));
    }

    #[test]
    fn accepts_manual() {
        let s = Source::new("manual").expect("valid source");
        assert_eq!(s.as_str(), "manual");
    }

    #[test]
    fn accepts_agent_identifier() {
        let s = Source::new("agent:claude-code").expect("valid source");
        assert_eq!(s.as_str(), "agent:claude-code");
    }

    #[test]
    fn serde_roundtrip_is_transparent_string() {
        let s = Source::new("manual").unwrap();
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, "\"manual\"");
        let parsed: Source = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);
    }
}
