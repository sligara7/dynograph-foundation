//! Fuzzy string matching algorithms.
//!
//! Implements Jaro-Winkler similarity and token sort ratio for
//! entity name matching. Returns scores in [0, 100].

/// Jaro similarity between two strings. Returns [0.0, 1.0].
///
/// Note on very short strings: the match window is the standard
/// `⌊max(len1, len2) / 2⌋ − 1`, which is `0` for strings of length ≤ 3.
/// With a zero window two characters only match at the *same* index, so
/// a pure transposition of a short string (e.g. `"xy"` vs `"yx"`) scores
/// `0.0`. This is inherent to Jaro, not a bug; entity resolution leans
/// on [`token_sort_ratio`] to absorb *token*-order differences, where it
/// matters in practice.
fn jaro(s1: &str, s2: &str) -> f64 {
    if s1 == s2 {
        return 1.0;
    }
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }

    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    let match_distance = (len1.max(len2) / 2).saturating_sub(1);

    let mut s1_matches = vec![false; len1];
    let mut s2_matches = vec![false; len2];

    let mut matches = 0usize;
    let mut transpositions = 0usize;

    // Find matches
    for i in 0..len1 {
        let start = i.saturating_sub(match_distance);
        let end = (i + match_distance + 1).min(len2);

        for j in start..end {
            if s2_matches[j] || s1_chars[i] != s2_chars[j] {
                continue;
            }
            s1_matches[i] = true;
            s2_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions
    let mut k = 0;
    for i in 0..len1 {
        if !s1_matches[i] {
            continue;
        }
        while !s2_matches[k] {
            k += 1;
        }
        if s1_chars[i] != s2_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f64;
    (m / len1 as f64 + m / len2 as f64 + (m - transpositions as f64 / 2.0) / m) / 3.0
}

/// Jaro-Winkler similarity. Returns score in [0, 100].
///
/// Adds a prefix bonus for strings that match at the beginning
/// (up to 4 characters). This is useful for names where the
/// first few characters are often correct.
///
/// Case-insensitive: both inputs are lowercased first. Callers that
/// score one query against many candidates should use `PreparedName`
/// instead, which lowercases each name once.
pub fn jaro_winkler(s1: &str, s2: &str) -> u32 {
    jaro_winkler_prepared(&s1.to_lowercase(), &s2.to_lowercase())
}

/// Jaro-Winkler over inputs that are **already lowercased** — the body
/// of [`jaro_winkler`] minus the two `to_lowercase` allocations.
fn jaro_winkler_prepared(s1_lower: &str, s2_lower: &str) -> u32 {
    let jaro_score = jaro(s1_lower, s2_lower);

    // Winkler prefix bonus
    let s1_chars: Vec<char> = s1_lower.chars().collect();
    let s2_chars: Vec<char> = s2_lower.chars().collect();
    let prefix_len = s1_chars
        .iter()
        .zip(s2_chars.iter())
        .take(4) // max prefix length
        .take_while(|(a, b)| a == b)
        .count();

    let winkler = jaro_score + (prefix_len as f64 * 0.1 * (1.0 - jaro_score));

    (winkler * 100.0).round() as u32
}

/// Token sort ratio. Returns score in [0, 100].
///
/// Splits both strings into tokens, sorts them alphabetically,
/// rejoins, then computes Jaro-Winkler on the sorted versions.
/// This handles reordered names: "John Smith" matches "Smith, John".
pub fn token_sort_ratio(s1: &str, s2: &str) -> u32 {
    jaro_winkler(&sort_tokens(s1), &sort_tokens(s2))
}

/// Split on whitespace and sort the tokens alphabetically, rejoined with
/// single spaces — the order-insensitive normal form behind
/// [`token_sort_ratio`] and [`PreparedName`].
fn sort_tokens(s: &str) -> String {
    let mut tokens: Vec<&str> = s.split_whitespace().filter(|t| !t.is_empty()).collect();
    tokens.sort_unstable();
    tokens.join(" ")
}

/// A name pre-normalized into the two forms entity resolution scores
/// against: lowercased, and token-sorted-then-lowercased.
///
/// Building this once per name keeps the normalization contract — sort
/// tokens on the *original-case* string, then lowercase — in one place
/// rather than spread across call sites, and lets the resolver lowercase
/// the query once instead of re-normalizing it for every candidate.
pub(crate) struct PreparedName {
    lower: String,
    sorted_lower: String,
}

impl PreparedName {
    pub(crate) fn new(s: &str) -> Self {
        Self {
            lower: s.to_lowercase(),
            sorted_lower: sort_tokens(s).to_lowercase(),
        }
    }

    /// Fuzzy score in [0, 100]: the better of token-order-insensitive
    /// (`token_sort_ratio`) and direct Jaro-Winkler. Symmetric in the
    /// two names.
    pub(crate) fn score(&self, other: &PreparedName) -> u32 {
        jaro_winkler_prepared(&self.sorted_lower, &other.sorted_lower)
            .max(jaro_winkler_prepared(&self.lower, &other.lower))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_strings() {
        assert_eq!(jaro_winkler("hello", "hello"), 100);
    }

    #[test]
    fn empty_strings() {
        assert_eq!(jaro_winkler("", ""), 100);
        assert_eq!(jaro_winkler("hello", ""), 0);
        assert_eq!(jaro_winkler("", "hello"), 0);
    }

    #[test]
    fn similar_strings() {
        let score = jaro_winkler("martha", "marhta");
        assert!(score > 90, "Expected >90, got {}", score);
    }

    #[test]
    fn different_strings() {
        let score = jaro_winkler("abc", "xyz");
        assert!(score < 50, "Expected <50, got {}", score);
    }

    #[test]
    fn case_insensitive() {
        assert_eq!(jaro_winkler("Hello", "hello"), 100);
        assert_eq!(jaro_winkler("ALICE", "alice"), 100);
    }

    #[test]
    fn prefix_bonus() {
        // "dwayne" vs "duane" — same prefix "d" gives a Winkler boost
        let score = jaro_winkler("dwayne", "duane");
        assert!(score > 80, "Expected >80, got {}", score);
    }

    #[test]
    fn token_sort_handles_reorder() {
        let score = token_sort_ratio("John Smith", "Smith John");
        assert_eq!(score, 100);
    }

    #[test]
    fn token_sort_with_title() {
        let score = token_sort_ratio("Professor Whitfield", "Whitfield Professor");
        assert_eq!(score, 100);
    }

    #[test]
    fn token_sort_partial_match() {
        let score = token_sort_ratio("the old professor", "Professor Whitfield");
        assert!(score > 50, "Expected >50, got {}", score);
    }

    #[test]
    fn prepared_matches_public_jaro_winkler() {
        // The `_prepared` path (caller pre-lowercases) must produce the
        // exact same score as the public entry point that lowercases
        // internally — that equivalence is what lets the resolver hoist
        // the query's lowercase out of its candidate loop.
        for (a, b) in [
            ("Marcus Whitfield", "marcus whitfeld"),
            ("DWAYNE", "duane"),
            ("", "hello"),
            ("Alice", "Bob"),
        ] {
            assert_eq!(
                jaro_winkler(a, b),
                jaro_winkler_prepared(&a.to_lowercase(), &b.to_lowercase()),
                "prepared score must match public score for {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn short_pure_transposition_scores_zero() {
        // Documented Jaro limitation: a length-2 transposition has a
        // zero match window, so it scores 0. Locked here so the behavior
        // is intentional, not an accident. Token-order differences are
        // handled by `token_sort_ratio`, which is what resolution uses.
        assert_eq!(jaro_winkler("xy", "yx"), 0);
        assert_eq!(token_sort_ratio("Li Wei", "Wei Li"), 100);
    }

    #[test]
    fn realistic_entity_resolution() {
        // High confidence — should auto-merge
        assert!(jaro_winkler("Marcus Whitfield", "Marcus Whitfeld") > 90);

        // Medium confidence — tiebreaker zone
        let score = jaro_winkler("the professor", "Professor Whitfield");
        assert!(
            score > 50 && score < 95,
            "Expected tiebreaker zone, got {}",
            score
        );

        // Low confidence — different entities
        assert!(jaro_winkler("Alice", "Bob") < 50);
    }
}
