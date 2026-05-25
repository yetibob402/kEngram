//! Date extraction for the deterministic tagger.
//!
//! Returns the surface forms of date/temporal references that appear
//! in the thought's prose, suitable for storage as `Tags::dates_mentioned`.
//! No interpretation: "next Friday" stays "next Friday", "2026-05-22"
//! stays "2026-05-22". The deterministic part is that the regex
//! matches don't invent digits — the LLM tagger's "1904 → 2004"
//! transposition class of failure is impossible by construction.
//!
//! Phase 2a uses regex-only extraction. The `interim` crate is in
//! the dependency tree as a future-use hedge: if we find the
//! regex-extracted candidates are too noisy on the real corpus, we
//! can validate each candidate against `interim::parse_date_string`
//! and drop the ones that don't parse. Phase 2b will revisit if
//! needed.

use std::sync::OnceLock;

use regex::Regex;

/// Extract date / temporal references from a thought's content.
/// Returns surface forms in document order, deduplicated.
pub fn extract_dates(content: &str) -> Vec<String> {
    let mut results: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for re in patterns() {
        for m in re.find_iter(content) {
            let surface = m.as_str().trim().to_string();
            if surface.is_empty() {
                continue;
            }
            if seen.insert(surface.clone()) {
                results.push(surface);
            }
        }
    }
    results
}

/// Date-shape patterns covering what shows up in kengram thoughts.
/// Built once and reused — the regex crate is happy with shared
/// compiled patterns across calls.
fn patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        vec![
            // ISO date (yyyy-mm-dd).
            Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").expect("iso date"),
            // Year-only in plausible range (1900-2099).
            Regex::new(r"\b(?:19|20)\d{2}\b").expect("year only"),
            // Decade ("the 1990s", "1900s").
            Regex::new(r"\b(?:19|20)\d{2}s\b").expect("decade"),
            // Quarter ("Q1 2024", "Q3").
            Regex::new(r"\bQ[1-4](?:\s+\d{4})?\b").expect("quarter"),
            // Month name + day (with optional ordinal): "May 18", "May 18th",
            // "May 18, 2024".
            Regex::new(
                r"(?i)\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s+\d{4})?\b",
            )
            .expect("month-day"),
            // Day + month name: "18 May", "18th of May", "18 May 2024".
            Regex::new(
                r"(?i)\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+\d{4})?\b",
            )
            .expect("day-month"),
            // Month + year: "May 2026", "January 2024".
            Regex::new(
                r"(?i)\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2}\b",
            )
            .expect("month-year"),
            // Weekday references: "next Friday", "this Monday", "last Tuesday".
            Regex::new(
                r"(?i)\b(?:next|this|last)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
            )
            .expect("relative weekday"),
            // Standalone weekday — only when preceded by a temporal marker
            // like "on", "by", "before", or "after" to avoid false-positive
            // matches on names ("Casey was here on Friday" → "on Friday";
            // "Casey loves Friday" → not matched). Imperfect but better
            // than gobbling every weekday in prose.
            Regex::new(
                r"(?i)\b(?:on|by|before|after|until)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
            )
            .expect("anchored weekday"),
            // Relative temporal: "next sprint", "this week", "last month",
            // "next month", "this quarter".
            Regex::new(
                r"(?i)\b(?:next|this|last)\s+(?:week|month|year|quarter|sprint|standup)\b",
            )
            .expect("relative period"),
            // N-ago: "3 days ago", "2 weeks ago", "5 months ago".
            Regex::new(r"(?i)\b\d+\s+(?:day|week|month|year)s?\s+ago\b").expect("n-ago"),
            // In-N: "in 3 days", "in 2 weeks".
            Regex::new(r"(?i)\bin\s+\d+\s+(?:day|week|month|year)s?\b").expect("in-n"),
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_extracts(content: &str, expected: &[&str]) {
        let got = extract_dates(content);
        let got_lower: Vec<String> = got.iter().map(|s| s.to_lowercase()).collect();
        for want in expected {
            assert!(
                got_lower.contains(&want.to_lowercase()),
                "expected to extract {want:?}, got {got:?}",
            );
        }
    }

    fn assert_does_not_extract(content: &str, not_expected: &[&str]) {
        let got = extract_dates(content);
        let got_lower: Vec<String> = got.iter().map(|s| s.to_lowercase()).collect();
        for unwant in not_expected {
            assert!(
                !got_lower.contains(&unwant.to_lowercase()),
                "did NOT expect {unwant:?}, but extract returned {got:?}",
            );
        }
    }

    #[test]
    fn iso_date() {
        assert_extracts(
            "Decision on 2026-05-22 closed the iteration loop.",
            &["2026-05-22"],
        );
    }

    #[test]
    fn year_only() {
        assert_extracts("Founded in 2024 with Series A funding.", &["2024"]);
    }

    #[test]
    fn decade() {
        assert_extracts("Popular in the 1990s.", &["1990s"]);
    }

    #[test]
    fn quarter() {
        assert_extracts(
            "Roadmap targets Q3 2024 launch; cleanup follows in Q4.",
            &["Q3 2024", "Q4"],
        );
    }

    #[test]
    fn month_day() {
        assert_extracts(
            "Slot scheduled for May 18, 2024 with backup on May 18th.",
            &["May 18, 2024", "May 18th"],
        );
    }

    #[test]
    fn month_year() {
        assert_extracts(
            "Originally captured May 2026; revised in January 2024.",
            &["May 2026", "January 2024"],
        );
    }

    #[test]
    fn relative_weekday() {
        assert_extracts(
            "Will revisit next Friday and ship this Monday.",
            &["next Friday", "this Monday"],
        );
    }

    #[test]
    fn anchored_weekday_with_preposition() {
        assert_extracts(
            "Due by Friday, finalized on Monday.",
            &["by Friday", "on Monday"],
        );
    }

    #[test]
    fn weekday_in_a_name_context_is_not_extracted() {
        // "Casey loves Friday" — Friday isn't preceded by a temporal
        // marker, so the anchored-weekday pattern doesn't match. The
        // standalone weekday isn't its own pattern (deliberate, to
        // avoid name-context false positives).
        assert_does_not_extract("Casey loves Friday outings.", &["Friday"]);
    }

    #[test]
    fn relative_period() {
        assert_extracts(
            "Plan refresh next sprint, retro this week, prep last month.",
            &["next sprint", "this week", "last month"],
        );
    }

    #[test]
    fn n_ago() {
        assert_extracts(
            "Started 3 weeks ago, escalated 2 days ago.",
            &["3 weeks ago", "2 days ago"],
        );
    }

    #[test]
    fn in_n() {
        assert_extracts(
            "Ship target in 5 days; review in 2 weeks.",
            &["in 5 days", "in 2 weeks"],
        );
    }

    #[test]
    fn dedupes_repeated_dates() {
        // "Monday" appears twice via different patterns — but the
        // dedup pass keeps only the first surface form.
        let s = "Ship by Monday. We agreed on Monday last week.";
        let got = extract_dates(s);
        let monday_count = got
            .iter()
            .filter(|d| d.to_lowercase().contains("monday"))
            .count();
        // "by Monday" and "on Monday" are different surface forms — both
        // kept (they're not the same string).
        assert!(monday_count >= 1);
    }

    #[test]
    fn invents_no_digits() {
        // The whole point of regex-only date extraction: 1904 stays 1904.
        // (LLM tagger's failure was "Semon 1904" → "2004".)
        assert_extracts("Semon 1904 first used the term.", &["1904"]);
        assert_does_not_extract("Semon 1904 first used the term.", &["2004"]);
    }

    #[test]
    fn empty_input() {
        assert!(extract_dates("").is_empty());
    }

    #[test]
    fn no_dates_returns_empty() {
        assert!(extract_dates("Just prose with no temporal references at all.").is_empty());
    }

    #[test]
    fn extracts_multiple_distinct_dates() {
        let s = "On 2026-05-22 (a Friday), the team agreed to ship in Q3 2026.";
        let got = extract_dates(s);
        // Should contain ISO date AND quarter
        assert!(got.iter().any(|d| d == "2026-05-22"));
        assert!(got.iter().any(|d| d == "Q3 2026"));
    }
}
