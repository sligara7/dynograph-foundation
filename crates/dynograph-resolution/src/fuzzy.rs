//! Fuzzy string matching algorithms.
//!
//! Implements Jaro-Winkler similarity and token sort ratio for
//! entity name matching. Returns scores in [0, 100].

/// Jaro similarity between two strings. Returns [0.0, 1.0].
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
pub fn jaro_winkler(s1: &str, s2: &str) -> u32 {
    let s1_lower = s1.to_lowercase();
    let s2_lower = s2.to_lowercase();

    let jaro_score = jaro(&s1_lower, &s2_lower);

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
    let sorted1 = sort_tokens(s1);
    let sorted2 = sort_tokens(s2);
    jaro_winkler(&sorted1, &sorted2)
}

fn sort_tokens(s: &str) -> String {
    let mut tokens: Vec<&str> = s.split_whitespace().filter(|t| !t.is_empty()).collect();
    tokens.sort_unstable();
    tokens.join(" ")
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
