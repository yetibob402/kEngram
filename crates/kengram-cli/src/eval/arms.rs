//! Model-arm configuration for `kengram eval tagger`.
//!
//! An "arm" is one tagger under test: a provider + endpoint + model. Arms
//! are declared in a small TOML file (committed example:
//! `eval/models.example.toml`) and constructed through the exact production
//! builders (`OpenAICompatibleTagger`, `HttpTagger`), so anything the
//! production tagger enforces — including the prompt/version provenance
//! binding — applies to eval arms for free.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, bail};
use kengram_core::Tagger;
use kengram_extract::{
    BUNDLED_TAGGER_VERSION, HttpTagger, HttpTaggerConfig,
    OpenAICompatibleConfig as TaggerBuilderConfig, OpenAICompatibleTagger,
};
use serde::Deserialize;

pub(crate) const DEFAULT_TEMPERATURE: f32 = 0.2;
pub(crate) const DEFAULT_TIMEOUT_SECONDS: u64 = 180;

#[derive(Debug, Deserialize)]
struct ArmsFile {
    #[serde(default)]
    arm: Vec<ArmSpec>,
}

/// One `[[arm]]` entry in the models TOML. `deny_unknown_fields` so a typo
/// (`temprature = 0.0`) is a load error instead of a silently-defaulted run.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArmSpec {
    pub name: String,
    /// `"openai-compatible"` (Ollama, vLLM, OpenRouter, ...) or `"http"`
    /// (the kengram tagger-sidecar protocol, e.g. the deterministic
    /// GLiNER sidecar).
    pub provider: String,
    pub endpoint: String,
    /// Backend-side model name. Required for `openai-compatible`; ignored
    /// for `http` (the sidecar serves whatever it serves).
    #[serde(default)]
    pub model_name: Option<String>,
    /// NAME of an environment variable holding the bearer token — never
    /// the token itself (this file may be committed).
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
    /// Optional system-prompt override for prompt A/B runs. Requires
    /// `model_version` (the production provenance binding rejects a custom
    /// prompt under the bundled version number).
    #[serde(default)]
    pub system_prompt_file: Option<PathBuf>,
    #[serde(default)]
    pub model_version: Option<i32>,
}

/// A constructed arm ready to run, plus the descriptive fields the report
/// records.
pub(crate) struct BuiltArm {
    pub name: String,
    pub provider: String,
    pub endpoint: String,
    pub model_name: String,
    pub model_version: i32,
    /// `"bundled"` or `"file:<path>"`.
    pub prompt: String,
    pub tagger: Arc<dyn Tagger>,
}

/// Load and validate the arms file. Usage errors (exit 2 at the CLI layer).
pub(crate) fn load_arm_specs(path: &Path) -> anyhow::Result<Vec<ArmSpec>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading models file {}", path.display()))?;
    let file: ArmsFile =
        toml::from_str(&raw).with_context(|| format!("parsing models file {}", path.display()))?;
    if file.arm.is_empty() {
        bail!("models file {} declares no [[arm]] entries", path.display());
    }
    let mut seen = std::collections::BTreeSet::new();
    for spec in &file.arm {
        if spec.name.is_empty() {
            bail!("models file {}: an arm has an empty name", path.display());
        }
        if !seen.insert(spec.name.as_str()) {
            bail!(
                "models file {}: duplicate arm name {:?}",
                path.display(),
                spec.name
            );
        }
        match spec.provider.as_str() {
            "openai-compatible" => {
                if spec.model_name.as_deref().unwrap_or("").is_empty() {
                    bail!(
                        "arm {:?}: provider openai-compatible requires model_name",
                        spec.name
                    );
                }
            }
            "http" => {}
            other => bail!(
                "arm {:?}: unknown provider {:?} (expected \"openai-compatible\" or \"http\")",
                spec.name,
                other
            ),
        }
    }
    Ok(file.arm)
}

/// Construct the tagger for one arm.
pub(crate) fn build_arm(spec: &ArmSpec) -> anyhow::Result<BuiltArm> {
    let api_key = match &spec.api_key_env {
        None => None,
        Some(var) => Some(std::env::var(var).with_context(|| {
            format!(
                "arm {:?}: api_key_env names {:?} but that environment variable is not set",
                spec.name, var
            )
        })?),
    };
    let timeout = Duration::from_secs(spec.timeout_seconds.unwrap_or(DEFAULT_TIMEOUT_SECONDS));

    match spec.provider.as_str() {
        "openai-compatible" => {
            let system_prompt = match &spec.system_prompt_file {
                None => None,
                Some(p) => Some(std::fs::read_to_string(p).with_context(|| {
                    format!(
                        "arm {:?}: reading system_prompt_file {}",
                        spec.name,
                        p.display()
                    )
                })?),
            };
            let prompt_label = spec
                .system_prompt_file
                .as_ref()
                .map(|p| format!("file:{}", p.display()))
                .unwrap_or_else(|| "bundled".to_string());
            let model_name = spec
                .model_name
                .clone()
                .expect("validated by load_arm_specs");
            let model_version = spec.model_version.unwrap_or(BUNDLED_TAGGER_VERSION);
            let config = TaggerBuilderConfig {
                endpoint: spec.endpoint.clone(),
                model_name: model_name.clone(),
                model_id: format!("eval/{}", spec.name),
                model_version,
                api_key,
                timeout,
                temperature: spec.temperature.unwrap_or(DEFAULT_TEMPERATURE),
                system_prompt,
            };
            let tagger = OpenAICompatibleTagger::new(config)
                .with_context(|| format!("constructing arm {:?}", spec.name))?;
            Ok(BuiltArm {
                name: spec.name.clone(),
                provider: spec.provider.clone(),
                endpoint: spec.endpoint.clone(),
                model_name,
                model_version,
                prompt: prompt_label,
                tagger: Arc::new(tagger),
            })
        }
        "http" => {
            let model_version = spec.model_version.unwrap_or(1);
            let config = HttpTaggerConfig {
                endpoint: spec.endpoint.clone(),
                model_id: format!("eval/{}", spec.name),
                model_version,
                api_key,
                timeout,
            };
            let tagger = HttpTagger::new(config)
                .with_context(|| format!("constructing arm {:?}", spec.name))?;
            Ok(BuiltArm {
                name: spec.name.clone(),
                provider: spec.provider.clone(),
                endpoint: spec.endpoint.clone(),
                model_name: spec
                    .model_name
                    .clone()
                    .unwrap_or_else(|| "sidecar".to_string()),
                model_version,
                prompt: "sidecar".to_string(),
                tagger: Arc::new(tagger),
            })
        }
        other => bail!("unknown provider {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn example_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval/models.example.toml")
    }

    fn write_temp(name: &str, contents: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("kengram-eval-arms-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn committed_example_arms_parse_and_build() {
        let specs = load_arm_specs(&example_path()).expect("example models file must parse");
        assert!(specs.len() >= 2);
        // Every example arm constructs without network I/O and without
        // requiring any env var to be set.
        for spec in &specs {
            assert!(
                spec.api_key_env.is_none(),
                "example arms must not require env vars"
            );
            let built = build_arm(spec).expect("example arm must construct");
            assert_eq!(built.name, spec.name);
            assert_eq!(
                built.prompt,
                if spec.provider == "http" {
                    "sidecar"
                } else {
                    "bundled"
                }
            );
        }
    }

    #[test]
    fn duplicate_arm_names_rejected() {
        let path = write_temp(
            "dup.toml",
            r#"
[[arm]]
name = "a"
provider = "openai-compatible"
endpoint = "http://localhost:11434/v1"
model_name = "m"

[[arm]]
name = "a"
provider = "http"
endpoint = "http://localhost:8082"
"#,
        );
        assert!(
            load_arm_specs(&path)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
    }

    #[test]
    fn unknown_provider_and_missing_model_name_rejected() {
        let bad_provider = write_temp(
            "bad-provider.toml",
            r#"
[[arm]]
name = "a"
provider = "grpc"
endpoint = "http://localhost:1"
"#,
        );
        assert!(
            load_arm_specs(&bad_provider)
                .unwrap_err()
                .to_string()
                .contains("unknown provider")
        );

        let no_model = write_temp(
            "no-model.toml",
            r#"
[[arm]]
name = "a"
provider = "openai-compatible"
endpoint = "http://localhost:11434/v1"
"#,
        );
        assert!(
            load_arm_specs(&no_model)
                .unwrap_err()
                .to_string()
                .contains("requires model_name")
        );
    }

    #[test]
    fn unknown_toml_key_rejected() {
        let typo = write_temp(
            "typo.toml",
            r#"
[[arm]]
name = "a"
provider = "openai-compatible"
endpoint = "http://localhost:11434/v1"
model_name = "m"
temprature = 0.0
"#,
        );
        assert!(load_arm_specs(&typo).is_err());
    }

    #[test]
    fn api_key_env_unset_is_a_build_error() {
        let spec = ArmSpec {
            name: "cloud".to_string(),
            provider: "openai-compatible".to_string(),
            endpoint: "https://openrouter.ai/api/v1".to_string(),
            model_name: Some("some/model".to_string()),
            api_key_env: Some("KENGRAM_EVAL_TEST_UNSET_VAR".to_string()),
            temperature: None,
            timeout_seconds: None,
            system_prompt_file: None,
            model_version: None,
        };
        let err = match build_arm(&spec) {
            Ok(_) => panic!("expected unset api_key_env to be a build error"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("KENGRAM_EVAL_TEST_UNSET_VAR"));
    }

    #[test]
    fn custom_prompt_without_version_bump_is_rejected_by_provenance_binding() {
        let prompt = write_temp("custom-prompt.txt", "You are a tagger.");
        let spec = ArmSpec {
            name: "prompt-ab".to_string(),
            provider: "openai-compatible".to_string(),
            endpoint: "http://localhost:11434/v1".to_string(),
            model_name: Some("m".to_string()),
            api_key_env: None,
            temperature: None,
            timeout_seconds: None,
            system_prompt_file: Some(prompt),
            model_version: None, // defaults to BUNDLED_TAGGER_VERSION -> must be rejected
        };
        assert!(
            build_arm(&spec).is_err(),
            "production provenance binding must apply"
        );
    }
}
