//! Pre-processor for NER input — defangs the use-mention discourse
//! failures the LLM tagger couldn't solve.
//!
//! gline-rs is a zero-shot NER model: it extracts entity spans from
//! whatever text we feed it, with no concept of "this sentence is
//! quoting another sentence" or "this is a list of examples." That
//! shows up empirically as the Phase 0 spike's biggest residual:
//! meta-discussion thoughts ("the recommendation thought mentions
//! Bob, Mark, Rob...") get every mentioned name extracted as a person.
//!
//! Our defense is to strip those mention-shaped spans BEFORE handing
//! the text to gline-rs. This module is pure text manipulation — no
//! model calls, no I/O, deterministic.
//!
//! Important: the cleaned text is fed to NER *only*. The original
//! content is still what gets persisted on the thought row. We're
//! changing what the extractor sees, not what's stored.

use std::sync::OnceLock;

use regex::Regex;

/// Clean a thought's content for NER input: strip quoted spans,
/// parenthetical "e.g."/"such as"/"like" enumerations, and normalize
/// ALL-CAPS section headings to title case. Returns a new string;
/// caller is responsible for passing the cleaned version to NER and
/// keeping the original for storage.
pub fn clean_for_ner(content: &str) -> String {
    let after_quotes = strip_quoted_spans(content);
    let after_examples = strip_example_parentheticals(&after_quotes);
    normalize_allcaps_headings(&after_examples)
}

/// Strip text inside single- or double-quoted spans. Replaces the
/// quoted content with a single space so adjacent words don't collide.
///
/// Examples:
/// - `'The Bob-as-verb pattern (e.g., "Bob the index") needs work.'`
///   → `'The Bob-as-verb pattern (e.g.,  ) needs work.'`
/// - `'Foo says "extract this" and moves on.'`
///   → `'Foo says   and moves on.'`
///
/// Single and double quotes are both stripped. Curly/smart quotes
/// (`'…'`, `"…"`) also covered. Nested quotes aren't recursively
/// handled — the outermost match wins, which is fine for the
/// kengram-corpus shapes we care about.
pub fn strip_quoted_spans(content: &str) -> String {
    // Match (non-greedy) anything between matching pairs of:
    //   straight double quotes: "...",
    //   straight single quotes: '...',
    //   smart double quotes: "...",
    //   smart single quotes: '...'.
    // The non-greedy `.*?` plus single-line mode (no `(?s)`) ensures
    // we don't span line breaks — quotes that cross newlines are rare
    // in kengram thoughts and treating them as mismatched is safer.
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(concat!(
            r#""[^"\n]*?""#,
            "|",
            r#"'[^'\n]*?'"#,
            "|",
            "\u{201C}[^\u{201D}\n]*?\u{201D}",
            "|",
            "\u{2018}[^\u{2019}\n]*?\u{2019}",
        ))
        .expect("static regex compiles")
    });
    re.replace_all(content, " ").into_owned()
}

/// Strip parenthetical clauses introduced by "e.g.", "such as", or
/// "like" — the discourse markers that signal "what follows is a list
/// of examples." Both forms covered:
/// - Parenthetical: `(e.g., X, Y, Z)`, `(such as X)`, `(like X)`
/// - In-line: `... e.g., X, Y, Z ...` up to the next sentence end
///
/// Replaces matches with a single space.
pub fn strip_example_parentheticals(content: &str) -> String {
    // Parenthetical form first: matches a `(` followed by `e.g.`/`such as`/`like`
    // (case-insensitive), then any non-`)` chars, then `)`.
    static PAREN_RE: OnceLock<Regex> = OnceLock::new();
    let paren_re = PAREN_RE.get_or_init(|| {
        Regex::new(r"(?i)\(\s*(?:e\.g\.?|such as|like)\b[^)]*\)")
            .expect("paren example regex compiles")
    });
    let after_paren = paren_re.replace_all(content, " ").into_owned();

    // In-line form: matches `e.g., X, Y, Z` (or `such as ...`) up to
    // sentence-ending punctuation (`.`, `;`, or end of string), not
    // including the punctuation itself.
    static INLINE_RE: OnceLock<Regex> = OnceLock::new();
    let inline_re = INLINE_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:e\.g\.?|such as|like)\b[^.;\n]*")
            .expect("inline example regex compiles")
    });
    inline_re.replace_all(&after_paren, " ").into_owned()
}

/// Normalize ALL-CAPS section heading lines (e.g. `RERANKER NOW ON
/// SEARCH_THOUGHTS`) to title case before NER. The LLM tagger's
/// failure mode here was promoting these section labels to `entities`;
/// gline-rs has the same risk because the all-caps shape looks
/// proper-noun-y. Lowercasing them keeps real ALL-CAPS acronyms
/// (`MCP`, `HNSW`) intact since those are short.
///
/// Heuristic: a line is a heading if (a) it's a full line on its own,
/// (b) it has at least 4 ALL-CAPS characters in a row, (c) it
/// contains only uppercase letters, digits, underscores, and spaces.
/// Punctuation breaks the heuristic — body prose isn't matched.
pub fn normalize_allcaps_headings(content: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"(?m)^[A-Z][A-Z0-9_\s]{3,}$").expect("allcaps heading regex compiles")
    });
    re.replace_all(content, |caps: &regex::Captures<'_>| {
        let line = &caps[0];
        title_case_line(line)
    })
    .into_owned()
}

/// Convert a single ALL-CAPS line to title case: each whitespace-
/// separated token has its first character capitalized, the rest
/// lowercased. Underscores in compound identifiers (e.g.
/// `SEARCH_THOUGHTS`) are preserved as-is — gline-rs is more likely
/// to recognize them as a single technical identifier than as a name.
fn title_case_line(line: &str) -> String {
    line.split(' ')
        .map(|tok| {
            let mut chars = tok.chars();
            match chars.next() {
                Some(first) => {
                    let rest: String = chars.collect::<String>().to_lowercase();
                    format!("{first}{rest}")
                }
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_quoted_spans_double_quotes() {
        let s = r#"The pattern (e.g., "Bob the index") needs work."#;
        let cleaned = strip_quoted_spans(s);
        assert!(!cleaned.contains("Bob the index"));
        assert!(cleaned.contains("needs work"));
    }

    #[test]
    fn strip_quoted_spans_single_quotes() {
        let s = "Foo says 'extract this' and moves on.";
        let cleaned = strip_quoted_spans(s);
        assert!(!cleaned.contains("extract this"));
        assert!(cleaned.contains("Foo says"));
        assert!(cleaned.contains("moves on"));
    }

    #[test]
    fn strip_quoted_spans_smart_quotes() {
        let s = "She said \u{201C}hello there\u{201D} loudly.";
        let cleaned = strip_quoted_spans(s);
        assert!(!cleaned.contains("hello there"));
    }

    #[test]
    fn strip_quoted_spans_doesnt_cross_newlines() {
        // A stray unmatched quote on one line shouldn't gobble the next.
        let s = "First line with one ' quote.\nSecond line is safe.";
        let cleaned = strip_quoted_spans(s);
        assert!(cleaned.contains("Second line is safe"));
    }

    #[test]
    fn strip_example_parentheticals_parenthetical_form() {
        let s = "Common verb-as-names (e.g., Bob, Mark, Rob, Frank, Pat) cause failures.";
        let cleaned = strip_example_parentheticals(s);
        assert!(!cleaned.contains("Bob"));
        assert!(!cleaned.contains("Mark"));
        assert!(!cleaned.contains("Frank"));
        assert!(cleaned.contains("cause failures"));
    }

    #[test]
    fn strip_example_parentheticals_inline_form() {
        let s = "Verb-as-names include such as Bob, Mark, Rob. Other failures persist.";
        let cleaned = strip_example_parentheticals(s);
        assert!(!cleaned.contains("Bob"));
        assert!(!cleaned.contains("Mark"));
        assert!(cleaned.contains("Other failures persist"));
    }

    #[test]
    fn strip_example_parentheticals_handles_like_marker() {
        let s = "Use prompts like \"evaluate options\" and \"pick one\". Then go.";
        let cleaned = strip_example_parentheticals(s);
        // The "like" inline form should consume up to the period.
        assert!(!cleaned.contains("evaluate options"));
        assert!(cleaned.contains("Then go"));
    }

    #[test]
    fn strip_example_parentheticals_preserves_non_example_parens() {
        let s = "The result (verified locally) was unexpected.";
        let cleaned = strip_example_parentheticals(s);
        assert!(cleaned.contains("verified locally"));
        assert!(cleaned.contains("unexpected"));
    }

    #[test]
    fn normalize_allcaps_headings_basic() {
        let s = "RERANKER NOW ON SEARCH\nNext sentence is fine.";
        let cleaned = normalize_allcaps_headings(s);
        assert!(cleaned.contains("Reranker"));
        assert!(cleaned.contains("Now"));
        assert!(cleaned.contains("Search"));
        assert!(!cleaned.contains("RERANKER"));
        assert!(cleaned.contains("Next sentence is fine"));
    }

    #[test]
    fn normalize_allcaps_headings_preserves_underscored_identifiers() {
        let s = "SEARCH_THOUGHTS DEPLOYED\nbody text here";
        let cleaned = normalize_allcaps_headings(s);
        // SEARCH_THOUGHTS as a single token gets first-char-uppercase,
        // rest-lowercase treatment → "Search_thoughts". Not ideal but
        // better than promoting it to an entity.
        assert!(!cleaned.contains("SEARCH_THOUGHTS"));
        assert!(cleaned.contains("body text here"));
    }

    #[test]
    fn normalize_allcaps_headings_ignores_short_acronyms_in_prose() {
        let s = "The MCP server uses HNSW for retrieval.";
        let cleaned = normalize_allcaps_headings(s);
        // MCP and HNSW are inline in body prose, not standalone heading
        // lines — heuristic should leave them alone.
        assert!(cleaned.contains("MCP"));
        assert!(cleaned.contains("HNSW"));
    }

    #[test]
    fn normalize_allcaps_headings_ignores_punctuation_lines() {
        let s = "JUST A SENTENCE, NOT A HEADING.\nNext line";
        let cleaned = normalize_allcaps_headings(s);
        // The first line has a comma + period — not a pure heading shape.
        // Heuristic should leave it alone.
        assert!(cleaned.contains("JUST A SENTENCE"));
    }

    #[test]
    fn clean_for_ner_combines_all_steps() {
        let s = concat!(
            "RERANKER NOTES\n",
            "Use prompts like \"evaluate options\" and \"pick one\". ",
            "The (e.g., Bob, Mark) verb-as-name failure persists. ",
            "Real names: Sarah and David shipped the fix.",
        );
        let cleaned = clean_for_ner(s);
        // ALL-CAPS heading normalized
        assert!(!cleaned.contains("RERANKER NOTES"));
        // Quoted spans gone
        assert!(!cleaned.contains("evaluate options"));
        assert!(!cleaned.contains("pick one"));
        // e.g. enumeration gone
        assert!(!cleaned.contains("Bob"));
        assert!(!cleaned.contains("Mark"));
        // Real content preserved
        assert!(cleaned.contains("Sarah"));
        assert!(cleaned.contains("David"));
        assert!(cleaned.contains("shipped the fix"));
    }

    #[test]
    fn clean_for_ner_empty_input() {
        assert_eq!(clean_for_ner(""), "");
    }
}
