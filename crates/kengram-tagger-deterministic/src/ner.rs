//! NER stage of the deterministic tagger — wraps gline-rs.
//!
//! Single zero-shot call with labels:
//! `[person, product, organization, title, action item, task to do]`.
//!
//! From the Phase 0 spike (commit `de5da43`) we know:
//! - 91% person recall on real kengram thoughts
//! - 212ms/thought CPU inference (gliner_small-v2.1)
//! - Compound product names like "Claude Desktop" get tagged as
//!   product spans (not split), which lets us filter out person
//!   spans that overlap with them — the "Claude" → person v13
//!   failure becomes a non-issue here.
//!
//! Labels chosen to give gline-rs enough hints to do compound-name
//! disambiguation. `organization` is included so things like
//! "TCGplayer" or "Kengram Inc" don't get pulled into `person`
//! either. `task to do` is paired with `action item` because the
//! spike showed gline-rs is sensitive to which phrasing of an
//! action-shaped label is used; including both improves recall.

use gliner::model::GLiNER;
use gliner::model::input::text::TextInput;
use gliner::model::pipeline::span::SpanMode;

use kengram_core::TaggerError;

/// Labels passed to gline-rs on every NER call. Ordering doesn't
/// affect output but keeps the array stable for testing.
///
/// `title` (added 2026-05-24) gives gline-rs a destination for role
/// tokens like "CTO", "CEO", "VP", "PM". Before the label was
/// present, the model routed those tokens to `person` (most
/// problematic instance: `tcgplayer-cto-role-descriptor` fixture
/// extracting "CTO" as a person). Title spans don't flow onto
/// `NerOutput` — they're a routing destination, not a published
/// field. The shape mirrors how `product`/`organization` is used:
/// labels that absorb tokens we don't want as `person`.
pub const NER_LABELS: &[&str] = &[
    "person",
    "product",
    "organization",
    "title",
    "action item",
    "task to do",
];

/// Output of the NER stage. Only `people` and `action_items` flow
/// onto the persisted `Tags` struct — `product`/`organization` spans
/// are used internally for filtering person extractions and then
/// discarded (the 5-field schema drops `entities`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NerOutput {
    pub people: Vec<String>,
    pub action_items: Vec<String>,
}

/// Run the NER stage on a single cleaned thought.
///
/// `content` should be the output of `preprocess::clean_for_ner` —
/// running NER on the raw thought directly will reproduce the use-
/// mention pollution we're trying to defang.
///
/// The gline-rs model is borrowed (no clone); callers hold it on
/// `DeterministicTagger` and pass a reference in.
pub fn extract_ner(content: &str, gliner: &GLiNER<SpanMode>) -> Result<NerOutput, TaggerError> {
    let input = TextInput::from_str(&[content], NER_LABELS).map_err(|e| {
        TaggerError::MalformedResponse(format!("gline-rs TextInput::from_str: {e}"))
    })?;
    let output = gliner.inference(input).map_err(|e| TaggerError::Backend {
        status: 0,
        body: format!("gline-rs inference: {e}"),
    })?;

    // Collect spans by label. We use string-containment (not byte ranges)
    // for the person/product overlap filter — gline-rs's Span type
    // doesn't expose public start/end accessors, but containment is
    // equivalent for the kengram-corpus failure modes (Claude ⊆ Claude
    // Desktop, Frank ⊆ Frank.io).
    let mut people_raw: Vec<String> = Vec::new();
    let mut product_or_org_texts: Vec<String> = Vec::new();
    let mut action_items: Vec<String> = Vec::new();
    for spans in &output.spans {
        for span in spans {
            let text = span.text().to_string();
            match span.class() {
                "person" => people_raw.push(text),
                "product" | "organization" => product_or_org_texts.push(text),
                "action item" | "task to do" => action_items.push(text),
                // `title` spans aren't a published field — they exist
                // purely to give gline-rs a label-destination for role
                // tokens so they don't get routed to `person`. Drop on
                // collection (we don't need them again downstream).
                "title" => {}
                _ => {}
            }
        }
    }

    let people = strip_trailing_years(people_raw);
    let people = filter_person_in_product(people, &product_or_org_texts);
    let action_items = dedup_preserve_order(action_items);

    Ok(NerOutput {
        people,
        action_items,
    })
}

/// Strip a trailing 4-digit year (1900-2099) from each person span.
/// gline-rs sometimes greedily includes a following year token in a
/// person span (e.g. "Semon 1904" instead of just "Semon" when the
/// year follows the name in body prose). The year still surfaces
/// independently via `dates::extract_dates` so no information is lost.
fn strip_trailing_years(people: Vec<String>) -> Vec<String> {
    people
        .into_iter()
        .map(|p| {
            let trimmed = p.trim_end();
            let bytes = trimmed.as_bytes();
            if bytes.len() < 5 {
                return p;
            }
            // Look for a trailing space + 4 digits + plausible-year prefix (19/20).
            let last5 = &bytes[bytes.len() - 5..];
            let space_then_year = last5[0] == b' '
                && last5[1].is_ascii_digit()
                && last5[2].is_ascii_digit()
                && last5[3].is_ascii_digit()
                && last5[4].is_ascii_digit()
                && (last5[1] == b'1' || last5[1] == b'2')
                && (last5[2] == b'9' || last5[2] == b'0');
            if space_then_year {
                trimmed[..trimmed.len() - 5].to_string()
            } else {
                p
            }
        })
        .collect()
}

/// Drop person spans whose surface text is a substring of any
/// product/organization span. This is the compound-product-name
/// defense: "Claude Desktop" tagged as product makes "Claude"
/// extracted as person get dropped (because "Claude" is a substring
/// of "Claude Desktop").
///
/// Also dedupes the surviving person strings (case-insensitive on
/// the comparison; first-occurrence's casing is preserved).
fn filter_person_in_product(people: Vec<String>, product_or_org_texts: &[String]) -> Vec<String> {
    let mut kept: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for person in people {
        let person_lc = person.to_lowercase();
        // Skip if this person's text appears inside any product/org span.
        // We check that person != product (substring of self doesn't
        // count) and that person appears as a substring of the product.
        let is_compound = product_or_org_texts.iter().any(|po| {
            let po_lc = po.to_lowercase();
            po_lc != person_lc && po_lc.contains(&person_lc)
        });
        if is_compound {
            continue;
        }
        if seen.insert(person_lc) {
            kept.push(person);
        }
    }
    kept
}

/// Dedupe a list of strings, preserving the first occurrence order.
fn dedup_preserve_order(items: Vec<String>) -> Vec<String> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    items
        .into_iter()
        .filter(|s| seen.insert(s.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_person_in_product_drops_compound_product_collision() {
        // "Claude Desktop" tagged as product; "Claude" extracted as person.
        // Person is substring of product → drop the person extraction.
        let people = vec!["Claude".to_string()];
        let products = vec!["Claude Desktop".to_string()];
        let kept = filter_person_in_product(people, &products);
        assert!(kept.is_empty());
    }

    #[test]
    fn filter_person_in_product_case_insensitive() {
        // Casing differences in either side shouldn't defeat the filter.
        let people = vec!["claude".to_string()];
        let products = vec!["CLAUDE DESKTOP".to_string()];
        let kept = filter_person_in_product(people, &products);
        assert!(kept.is_empty());
    }

    #[test]
    fn filter_person_in_product_keeps_disjoint_persons() {
        let people = vec![
            "Claude".to_string(), // substring of product
            "Ron".to_string(),    // not in any product
        ];
        let products = vec!["Claude Desktop".to_string()];
        let kept = filter_person_in_product(people, &products);
        assert_eq!(kept, vec!["Ron".to_string()]);
    }

    #[test]
    fn filter_person_in_product_dedupes_repeated_names() {
        // gline-rs sometimes emits the same name twice if it appears
        // multiple times in the input. The filter dedupes case-
        // insensitively while preserving the first occurrence's casing.
        let people = vec![
            "Casey".to_string(),
            "casey".to_string(), // case-different duplicate
            "Ron".to_string(),
        ];
        let kept = filter_person_in_product(people, &[]);
        assert_eq!(kept, vec!["Casey".to_string(), "Ron".to_string()]);
    }

    #[test]
    fn filter_person_in_product_preserves_equal_person_and_product() {
        // Edge case: same string emitted as both person AND product.
        // The filter requires the product to be DIFFERENT from the
        // person (proper compound name, not exact equality), so this
        // person survives. The v12 disjointness validator downstream
        // handles the people↔entities collision separately.
        let people = vec!["Sarah".to_string()];
        let products = vec!["Sarah".to_string()];
        let kept = filter_person_in_product(people, &products);
        assert_eq!(kept, vec!["Sarah".to_string()]);
    }

    #[test]
    fn strip_trailing_years_basic_case() {
        // The Semon 1904 case from deterministic.json fixtures.
        let people = vec!["Semon 1904".to_string()];
        let stripped = strip_trailing_years(people);
        assert_eq!(stripped, vec!["Semon".to_string()]);
    }

    #[test]
    fn strip_trailing_years_handles_multiple_names() {
        let people = vec![
            "Maria".to_string(),       // no year — unchanged
            "Wilson 1985".to_string(), // 4-digit year — stripped
            "Ron".to_string(),         // no year — unchanged
        ];
        let stripped = strip_trailing_years(people);
        assert_eq!(
            stripped,
            vec!["Maria".to_string(), "Wilson".to_string(), "Ron".to_string(),]
        );
    }

    #[test]
    fn strip_trailing_years_only_matches_plausible_year_range() {
        // 1900-2099 range only. Numbers outside that range, or
        // non-year-shaped trailing tokens, should NOT be stripped.
        let people = vec![
            "Smith 1899".to_string(), // pre-1900 — unchanged
            "Jones 2100".to_string(), // post-2099 — unchanged
            "Bob 25".to_string(),     // not 4 digits — unchanged
            "Pat 12345".to_string(),  // 5 digits — unchanged
        ];
        let stripped = strip_trailing_years(people);
        assert_eq!(
            stripped,
            vec![
                "Smith 1899".to_string(),
                "Jones 2100".to_string(),
                "Bob 25".to_string(),
                "Pat 12345".to_string(),
            ]
        );
    }

    #[test]
    fn strip_trailing_years_preserves_names_with_internal_years() {
        // "Jane 1985 Smith" — internal year, not trailing. Leave alone.
        let people = vec!["Jane 1985 Smith".to_string()];
        let stripped = strip_trailing_years(people);
        assert_eq!(stripped, vec!["Jane 1985 Smith".to_string()]);
    }

    #[test]
    fn ner_labels_includes_title() {
        // Regression pin: the `title` label must be present so gline-rs
        // can route role/title tokens away from `person`.
        assert!(NER_LABELS.contains(&"title"));
    }

    #[test]
    fn dedup_preserve_order_keeps_first_occurrence() {
        let items = vec![
            "review the doc".to_string(),
            "ship the change".to_string(),
            "review the doc".to_string(), // duplicate
            "test it".to_string(),
        ];
        let deduped = dedup_preserve_order(items);
        assert_eq!(
            deduped,
            vec![
                "review the doc".to_string(),
                "ship the change".to_string(),
                "test it".to_string(),
            ]
        );
    }
}
