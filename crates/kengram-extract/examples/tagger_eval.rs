//! Tagger eval harness — local iteration tool for prompt changes.
//!
//! Calls a real Ollama (or any OpenAI-compatible /v1 endpoint) via the
//! same `OpenAICompatibleTagger` code path production uses, evaluates
//! per-fixture assertions declared in JSON, prints pass/fail. Designed
//! for the prompt-iteration loop documented in `docs/goals/README.md`.
//!
//! NOTE (M7.1): this is the fast smoke lane (partial-label assertions,
//! raw tagger output, seconds per run). For measured multi-model
//! comparison against full golden labels — per-field P/R/F1, kind
//! confusion matrix, stability, finalized-output scoring — use
//! `kengram eval tagger` (crates/kengram-cli/src/eval/), documented in
//! DEVELOPMENT.md "Tagger model evaluation".
//!
//! Defaults to `http://localhost:11434/v1` and `gemma3:12b`. Override
//! via env vars `OLLAMA_ENDPOINT` and `TAGGER_MODEL`. Iterate against a
//! remote Ollama (e.g., the iMac over Tailnet) by setting
//! `OLLAMA_ENDPOINT=http://100.110.75.74:11434/v1`.
//!
//! Usage:
//!
//!   cargo run --example tagger_eval -- path/to/fixtures.json [more.json ...]
//!   cargo run --example tagger_eval -- --json fixtures.json
//!
//! Exit codes:
//!   0  — all fixtures passed
//!   1  — at least one fixture failed an assertion
//!   2  — config/usage error (missing argv, malformed fixture file, tagger init failure)

use kengram_core::Tagger;
use kengram_extract::{BUNDLED_TAGGER_VERSION, OpenAICompatibleConfig, OpenAICompatibleTagger};
use serde::{Deserialize, Serialize};
use std::{env, process::ExitCode, time::Duration};

#[derive(Deserialize)]
struct FixtureFile {
    fixtures: Vec<Fixture>,
}

#[derive(Deserialize)]
struct Fixture {
    name: String,
    #[serde(default)]
    category: String,
    content: String,
    #[serde(default)]
    assertions: Assertions,
}

#[derive(Deserialize, Default)]
struct Assertions {
    #[serde(default)]
    people_must_contain: Vec<String>,
    #[serde(default)]
    people_must_not_contain: Vec<String>,
    #[serde(default)]
    entities_must_contain: Vec<String>,
    #[serde(default)]
    entities_must_not_contain: Vec<String>,
    #[serde(default)]
    action_items_must_contain_substring: Vec<String>,
    #[serde(default)]
    action_items_must_not_contain_substring: Vec<String>,
    #[serde(default)]
    kind_equals: Option<String>,
    #[serde(default)]
    topics_must_contain: Vec<String>,
    #[serde(default)]
    topics_must_not_contain: Vec<String>,
}

#[derive(Serialize)]
struct EvalResult {
    name: String,
    category: String,
    passed: bool,
    failures: Vec<String>,
}

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let json_output = args.iter().any(|a| a == "--json");
    let paths: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if paths.is_empty() {
        eprintln!("usage: tagger_eval [--json] <fixture.json> [more.json ...]");
        return ExitCode::from(2);
    }

    let endpoint =
        env::var("OLLAMA_ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let model = env::var("TAGGER_MODEL").unwrap_or_else(|_| "gemma3:12b".to_string());
    // Optional bearer token so the sweep's cloud arm (OpenRouter / ollama-cloud
    // / any authenticated OpenAI-compatible endpoint) can be exercised with the
    // same harness. Unset → no auth (the local-Ollama default).
    let api_key = env::var("TAGGER_API_KEY").ok().filter(|k| !k.is_empty());
    // Per-request timeout (default 180s). Bump it for slow models — a cold 27B
    // on modest hardware can exceed 180s and every fixture then "fails" as a
    // timeout, which is a latency artifact, not a tag-quality signal.
    let timeout_secs = env::var("TAGGER_TIMEOUT_SECONDS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(180);
    let cfg = OpenAICompatibleConfig {
        endpoint: endpoint.clone(),
        model_name: model.clone(),
        model_id: format!("ollama/{model}"),
        model_version: BUNDLED_TAGGER_VERSION,
        api_key,
        temperature: 0.2,
        timeout: Duration::from_secs(timeout_secs),
        system_prompt: None,
    };
    let tagger = match OpenAICompatibleTagger::new(cfg) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tagger init failed: {e}");
            return ExitCode::from(2);
        }
    };

    if !json_output {
        eprintln!(
            "# tagger_eval against {endpoint} (model={model}, prompt v{BUNDLED_TAGGER_VERSION})"
        );
    }

    let mut results = Vec::new();
    for path in paths {
        let raw = match std::fs::read_to_string(path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("read {path}: {e}");
                return ExitCode::from(2);
            }
        };
        let file: FixtureFile = match serde_json::from_str(&raw) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("parse {path}: {e}");
                return ExitCode::from(2);
            }
        };
        for fx in file.fixtures {
            results.push(eval_one(&tagger, &fx).await);
        }
    }

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    if json_output {
        let out = serde_json::json!({
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "results": results,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
    } else {
        for r in &results {
            if r.passed {
                println!("PASS  {:<14} {}", r.category, r.name);
            } else {
                println!("FAIL  {:<14} {}", r.category, r.name);
                for f in &r.failures {
                    println!("        {f}");
                }
            }
        }
        eprintln!("---");
        eprintln!("{passed}/{total} passed");
    }

    if passed == total {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

async fn eval_one(tagger: &OpenAICompatibleTagger, fx: &Fixture) -> EvalResult {
    let mut failures = Vec::new();
    let output = match tagger.tag(&fx.content, None).await {
        Ok(o) => o,
        Err(e) => {
            failures.push(format!("tagger error: {e}"));
            return EvalResult {
                name: fx.name.clone(),
                category: fx.category.clone(),
                passed: false,
                failures,
            };
        }
    };
    let tags = output.tags;
    let a = &fx.assertions;

    for name in &a.people_must_contain {
        let lc = name.to_lowercase();
        if !tags.people.iter().any(|p| p.to_lowercase() == lc) {
            failures.push(format!(
                "expected people to contain {name:?}, got {:?}",
                tags.people
            ));
        }
    }
    for name in &a.people_must_not_contain {
        let lc = name.to_lowercase();
        if tags.people.iter().any(|p| p.to_lowercase() == lc) {
            failures.push(format!(
                "expected people NOT to contain {name:?}, got {:?}",
                tags.people
            ));
        }
    }
    for ent in &a.entities_must_contain {
        let lc = ent.to_lowercase();
        if !tags.entities.iter().any(|e| e.to_lowercase() == lc) {
            failures.push(format!(
                "expected entities to contain {ent:?}, got {:?}",
                tags.entities
            ));
        }
    }
    for ent in &a.entities_must_not_contain {
        let lc = ent.to_lowercase();
        if tags.entities.iter().any(|e| e.to_lowercase() == lc) {
            failures.push(format!(
                "expected entities NOT to contain {ent:?}, got {:?}",
                tags.entities
            ));
        }
    }
    for sub in &a.action_items_must_contain_substring {
        let lc = sub.to_lowercase();
        if !tags
            .action_items
            .iter()
            .any(|ai| ai.to_lowercase().contains(&lc))
        {
            failures.push(format!(
                "expected some action_item to contain substring {sub:?}, got {:?}",
                tags.action_items
            ));
        }
    }
    for sub in &a.action_items_must_not_contain_substring {
        let lc = sub.to_lowercase();
        if tags
            .action_items
            .iter()
            .any(|ai| ai.to_lowercase().contains(&lc))
        {
            failures.push(format!(
                "expected no action_item to contain substring {sub:?}, got {:?}",
                tags.action_items
            ));
        }
    }
    if let Some(expected) = &a.kind_equals {
        let actual: String = tags
            .kind
            .as_ref()
            .and_then(|k| serde_json::to_value(k).ok())
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| "null".to_string());
        if &actual != expected {
            failures.push(format!("expected kind={expected:?}, got kind={actual:?}"));
        }
    }
    for t in &a.topics_must_contain {
        let lc = t.to_lowercase();
        if !tags.topics.iter().any(|x| x.to_lowercase() == lc) {
            failures.push(format!(
                "expected topics to contain {t:?}, got {:?}",
                tags.topics
            ));
        }
    }
    for t in &a.topics_must_not_contain {
        let lc = t.to_lowercase();
        if tags.topics.iter().any(|x| x.to_lowercase() == lc) {
            failures.push(format!(
                "expected topics NOT to contain {t:?}, got {:?}",
                tags.topics
            ));
        }
    }

    EvalResult {
        name: fx.name.clone(),
        category: fx.category.clone(),
        passed: failures.is_empty(),
        failures,
    }
}
