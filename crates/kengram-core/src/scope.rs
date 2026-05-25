//! Free-form scope label, defaulting to `"global"`. Convention is dotted
//! notation (`work.tcgplayer`, `personal.medical`) but the only enforcement
//! is "non-empty and reasonable length." Hierarchy semantics (prefix-match)
//! are intentionally not implemented in M1 — see
//! `docs/milestones/m1-capture-and-search.md`.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Scope(String);

#[derive(Debug, thiserror::Error)]
pub enum ScopeError {
    #[error("scope must be non-empty")]
    Empty,
    #[error("scope must be at most 256 characters (got {0})")]
    TooLong(usize),
}

impl Scope {
    pub const GLOBAL: &'static str = "global";
    pub const MAX_LEN: usize = 256;

    pub fn new(s: impl Into<String>) -> Result<Self, ScopeError> {
        let s = s.into();
        if s.is_empty() {
            return Err(ScopeError::Empty);
        }
        if s.len() > Self::MAX_LEN {
            return Err(ScopeError::TooLong(s.len()));
        }
        Ok(Self(s))
    }

    pub fn global() -> Self {
        Self(Self::GLOBAL.to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl Default for Scope {
    fn default() -> Self {
        Self::global()
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_returns_global_string() {
        assert_eq!(Scope::global().as_str(), "global");
    }

    #[test]
    fn rejects_empty_string() {
        assert!(matches!(Scope::new(""), Err(ScopeError::Empty)));
    }

    #[test]
    fn rejects_overlong_string() {
        let long = "x".repeat(Scope::MAX_LEN + 1);
        assert!(matches!(Scope::new(long), Err(ScopeError::TooLong(_))));
    }

    #[test]
    fn accepts_simple_string() {
        let s = Scope::new("work").expect("valid scope");
        assert_eq!(s.as_str(), "work");
    }

    #[test]
    fn accepts_dotted_string() {
        let s = Scope::new("work.tcgplayer").expect("valid scope");
        assert_eq!(s.as_str(), "work.tcgplayer");
    }

    #[test]
    fn default_is_global() {
        assert_eq!(Scope::default(), Scope::global());
    }

    #[test]
    fn serde_roundtrip_is_transparent_string() {
        let s = Scope::new("personal").unwrap();
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, "\"personal\"");
        let parsed: Scope = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);
    }
}
