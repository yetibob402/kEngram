//! Free-form JSONB metadata attached to a captured thought. By convention,
//! callers populate a recommended set of keys (`client_name`, `session_id`,
//! `tool_name`, `agent_role`) but no validation is enforced. The JSON value
//! lives on the thought row in Postgres.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Metadata(Value);

impl Metadata {
    pub fn empty() -> Self {
        Self(Value::Object(Map::new()))
    }

    pub fn as_value(&self) -> &Value {
        &self.0
    }

    pub fn into_value(self) -> Value {
        self.0
    }
}

impl Default for Metadata {
    fn default() -> Self {
        Self::empty()
    }
}

impl From<Value> for Metadata {
    fn from(v: Value) -> Self {
        Self(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_is_empty_object() {
        assert_eq!(Metadata::empty().as_value(), &json!({}));
    }

    #[test]
    fn default_is_empty() {
        assert_eq!(Metadata::default(), Metadata::empty());
    }

    #[test]
    fn roundtrips_object() {
        let m: Metadata = json!({"session_id": "abc", "client_name": "claude-code"}).into();
        let serialized = serde_json::to_string(&m).unwrap();
        let parsed: Metadata = serde_json::from_str(&serialized).unwrap();
        assert_eq!(parsed, m);
    }
}
