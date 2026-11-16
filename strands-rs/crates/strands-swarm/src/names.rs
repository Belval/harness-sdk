//! Human-readable agent names: every agent has one, chosen by its spawner or derived from its task,
//! unique within the swarm, and usable wherever an agent id is accepted.

/// Longest name, in characters.
pub const MAX_LEN: usize = 24;

/// Words skipped when deriving a name from a task.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from", "help", "i", "in", "into",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "please", "so", "that", "the", "their", "then",
    "this", "to", "up", "us", "we", "with", "you", "your",
];

/// Words a derived name keeps at most.
const DERIVED_WORDS: usize = 3;

/// Lower-case words of `text`: runs of alphanumerics, `_` and `.`.
fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
        .map(|word| word.trim_matches('.').to_lowercase())
        .filter(|word| !word.is_empty())
}

/// Join words with `-`, stopping before [`MAX_LEN`] would be exceeded (a single over-long word is cut).
fn join(words: impl Iterator<Item = String>) -> String {
    let mut name = String::new();
    for word in words {
        let extra = usize::from(!name.is_empty()) + word.chars().count();
        if name.chars().count() + extra > MAX_LEN {
            if name.is_empty() {
                name = word.chars().take(MAX_LEN).collect();
            }
            break;
        }
        if !name.is_empty() {
            name.push('-');
        }
        name.push_str(&word);
    }
    name
}

/// Normalize a spawner-chosen name into a short, typeable slug: `Lazy MCP!` → `lazy-mcp`.
pub fn slug(raw: &str) -> String {
    join(words(raw))
}

/// A name for an agent whose spawner gave none: the first few meaningful words of its task.
pub fn derive(task: &str) -> String {
    let first_line = task
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or_default();
    let meaningful = words(first_line).filter(|word| !STOPWORDS.contains(&word.as_str()));
    join(meaningful.take(DERIVED_WORDS))
}

/// The name for a new agent: the requested one (normalized) or one derived from the task, made
/// unique against `taken` by a `-2`, `-3`, … suffix. Never empty, never `parent` or `user`, and
/// never shaped like an id (`a7`), so ids and the reserved receivers always resolve first.
pub fn choose<'a>(
    requested: Option<&str>,
    task: &str,
    taken: impl Iterator<Item = &'a str> + Clone,
) -> String {
    let base = requested
        .map(slug)
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| derive(task));
    let base = match base.as_str() {
        "" => "agent".to_owned(),
        "parent" | "user" => format!("{base}-agent"),
        _ if looks_like_id(&base) => format!("agent-{base}"),
        _ => base,
    };
    let free = |candidate: &str| !taken.clone().any(|name| name == candidate);
    if free(&base) {
        return base;
    }
    (2..)
        .map(|n| {
            let suffix = format!("-{n}");
            let stem: String = base.chars().take(MAX_LEN - suffix.len()).collect();
            format!("{}{suffix}", stem.trim_end_matches('-'))
        })
        .find(|candidate| free(candidate))
        .expect("an unused suffix exists")
}

/// Whether `text` has the shape of an agent id (`a` followed by digits).
pub fn looks_like_id(text: &str) -> bool {
    text.strip_prefix('a')
        .is_some_and(|digits| !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugs_spawner_names() {
        assert_eq!(slug("lazy-mcp"), "lazy-mcp");
        assert_eq!(slug("  Lazy MCP! "), "lazy-mcp");
        assert_eq!(slug("tui_tests v2.1"), "tui_tests-v2.1");
        assert_eq!(
            slug("a-really-long-name-that-goes-on-and-on"),
            "a-really-long-name-that"
        );
        assert_eq!(slug("x".repeat(40).as_str()).len(), MAX_LEN);
        assert_eq!(slug("!!!"), "");
    }

    #[test]
    fn derives_names_from_the_first_meaningful_words_of_the_task() {
        assert_eq!(derive("Fix the flaky TUI tests in view.rs"), "fix-flaky-tui");
        assert_eq!(derive("You are a reviewer. Read CR-123"), "reviewer-read-cr");
        assert_eq!(
            derive("\n\n  Give agents friendlier names, please"),
            "give-agents-friendlier"
        );
        assert_eq!(
            derive("internationalization-localization-infrastructure"),
            "internationalization"
        );
        assert_eq!(derive("the to of"), "");
    }

    #[test]
    fn chooses_unique_non_reserved_names() {
        let none: [&str; 0] = [];
        assert_eq!(choose(Some("Parser"), "whatever", none.iter().copied()), "parser");
        assert_eq!(choose(None, "count to 3", none.iter().copied()), "count-3");
        assert_eq!(choose(Some("  "), "", none.iter().copied()), "agent");
        assert_eq!(choose(Some("parent"), "", none.iter().copied()), "parent-agent");
        assert_eq!(choose(Some("a2"), "", none.iter().copied()), "agent-a2");

        let taken = ["parser", "parser-2"];
        assert_eq!(choose(Some("parser"), "", taken.iter().copied()), "parser-3");
        let long = "abcdefghijklmnopqrstuvwx";
        assert_eq!(
            choose(Some(long), "", [long].iter().copied()),
            "abcdefghijklmnopqrstuv-2"
        );
    }

    #[test]
    fn recognizes_id_shaped_text() {
        assert!(looks_like_id("a1"));
        assert!(looks_like_id("a42"));
        assert!(!looks_like_id("a"));
        assert!(!looks_like_id("ab"));
        assert!(!looks_like_id("parser"));
    }
}
