//! A user-level memory file shared by every agent and run: things the user asked to remember.
//!
//! It is plain Markdown, one dated bullet per entry, so the user can read and edit it by hand.
//! Every agent sees it at the end of its system prompt, refreshed from disk at the start of each
//! turn. Writes go through one swarm-wide lock, re-read the file first (another run or the user
//! may have changed it) and replace it atomically, so concurrent agents never clobber each other.

use std::path::{Path, PathBuf};

use strands_agents::types::{SystemContentBlock, SystemPrompt};
use strands_agents::{Error, Result};

/// Heading of the memory section in the system prompt.
pub const HEADING: &str = "# Memory (things the user asked to remember)";
/// Closes the section, so it can be replaced in place when the file changes.
const END: &str = "(end of memory)";
/// Largest share of the file put into the prompt, about 4k tokens; the newest entries win.
pub const MAX_PROMPT_CHARS: usize = 16_000;

/// The memory file and the lock that serializes writes to it.
#[derive(Debug)]
pub struct Memory {
    path: PathBuf,
    write: tokio::sync::Mutex<()>,
}

impl Memory {
    /// Memory backed by `path`; the file need not exist yet.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            write: tokio::sync::Mutex::new(()),
        }
    }

    /// `$XDG_CONFIG_HOME/strands-swarm/memory.md`, else `~/.config/strands-swarm/memory.md`.
    pub fn default_path() -> Option<PathBuf> {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
        Some(base.join("strands-swarm").join("memory.md"))
    }

    /// The file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The file's contents; empty when it does not exist.
    pub async fn read(&self) -> Result<String> {
        match tokio::fs::read_to_string(&self.path).await {
            Ok(text) => Ok(text),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
            Err(error) => Err(self.io_error("read", error)),
        }
    }

    /// Append `text` as a bullet dated `date` (`YYYY-MM-DD`). Returns the line written.
    pub async fn remember(&self, text: &str, date: &str) -> Result<String> {
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if text.is_empty() {
            return Err(Error::other("Nothing to remember: the text is empty."));
        }
        let line = format!("- {date}: {text}");
        let _guard = self.write.lock().await;
        let mut contents = self.read().await?;
        if !contents.is_empty() && !contents.ends_with('\n') {
            contents.push('\n');
        }
        contents.push_str(&line);
        contents.push('\n');
        self.replace(&contents).await?;
        Ok(line)
    }

    /// Remove every line containing `text` (case-insensitive). Returns the removed lines; an error
    /// when none matched.
    pub async fn forget(&self, text: &str) -> Result<Vec<String>> {
        let needle = text.trim().to_lowercase();
        if needle.is_empty() {
            return Err(Error::other("Nothing to forget: the text is empty."));
        }
        let _guard = self.write.lock().await;
        let contents = self.read().await?;
        let (removed, kept): (Vec<&str>, Vec<&str>) = contents
            .lines()
            .partition(|line| line.to_lowercase().contains(&needle));
        if removed.is_empty() {
            return Err(Error::other(format!("No memory line contains {text:?}.")));
        }
        let mut rest = kept.join("\n");
        if !rest.is_empty() {
            rest.push('\n');
        }
        self.replace(&rest).await?;
        Ok(removed.into_iter().map(str::to_owned).collect())
    }

    /// The system-prompt section for the current file, or `None` when it is empty.
    pub async fn section(&self) -> Option<String> {
        match self.read().await {
            Ok(text) => section(&text, &self.path),
            Err(error) => {
                tracing::warn!(%error, "memory file unreadable");
                None
            }
        }
    }

    async fn replace(&self, contents: &str) -> Result<()> {
        crate::files::replace_async(&self.path, contents.as_bytes().to_vec(), false)
            .await
            .map_err(|error| self.io_error("write", error))
    }

    fn io_error(&self, what: &str, error: std::io::Error) -> Error {
        Error::other(format!(
            "Failed to {what} memory file {}: {error}",
            self.path.display()
        ))
    }
}

/// The prompt section for memory `text` read from `path`: the newest [`MAX_PROMPT_CHARS`] at
/// line boundaries, with a note when older entries were left out.
pub fn section(text: &str, path: &Path) -> Option<String> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let mut kept = text;
    if text.len() > MAX_PROMPT_CHARS {
        let cut = text.len() - MAX_PROMPT_CHARS;
        let start = text[cut..].find('\n').map_or(text.len(), |at| cut + at + 1);
        kept = &text[start..];
    }
    let note = match kept.len() < text.len() {
        true => format!(
            "\n[{} older characters left out; the full file is {}.]",
            text.len() - kept.len(),
            path.display()
        ),
        false => String::new(),
    };
    Some(format!(
        "{HEADING}\nThe user asked the swarm to remember these (newest last). Use `remember` to add \
         to them and `forget` to drop outdated ones.{note}\n\n{kept}\n{END}"
    ))
}

/// `prompt` with its memory section replaced by `section` (removed when `None`, appended when
/// there was none). Structured prompts keep their other blocks, so cache points stay in place.
pub fn with_section(prompt: Option<SystemPrompt>, section: Option<&str>) -> Option<SystemPrompt> {
    match prompt {
        Some(SystemPrompt::Blocks(mut blocks)) => {
            let at = blocks
                .iter()
                .position(|block| matches!(block, SystemContentBlock::Text(t) if t.contains(HEADING)));
            match (at, section) {
                (Some(at), _) => {
                    let SystemContentBlock::Text(text) = &blocks[at] else {
                        unreachable!()
                    };
                    match replace_text(text, section) {
                        Some(text) if !text.is_empty() => blocks[at] = SystemContentBlock::Text(text),
                        _ => drop(blocks.remove(at)),
                    }
                }
                (None, Some(section)) => blocks.push(SystemContentBlock::Text(format!("\n\n{section}"))),
                (None, None) => {}
            }
            Some(SystemPrompt::Blocks(blocks))
        }
        other => {
            let text = other.map(|prompt| prompt.text()).unwrap_or_default();
            replace_text(&text, section).map(SystemPrompt::Text)
        }
    }
}

fn replace_text(text: &str, section: Option<&str>) -> Option<String> {
    let base = match (text.find(HEADING), text.find(END)) {
        (Some(start), Some(end)) if end > start => {
            format!("{}{}", text[..start].trim_end(), &text[end + END.len()..])
        }
        _ => text.to_owned(),
    };
    match section {
        Some(section) if base.is_empty() => Some(section.to_owned()),
        Some(section) => Some(format!("{}\n\n{section}", base.trim_end())),
        None if base.is_empty() => None,
        None => Some(base),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn remembers_dated_bullets_and_forgets_matching_lines() {
        let dir = tempfile::tempdir().unwrap();
        let memory = Memory::new(dir.path().join("sub/memory.md"));
        assert_eq!(memory.read().await.unwrap(), "");
        assert!(memory.section().await.is_none());

        let line = memory.remember("  use\nbun ", "2026-01-02").await.unwrap();
        assert_eq!(line, "- 2026-01-02: use bun");
        memory.remember("Prefer tabs", "2026-01-03").await.unwrap();
        assert_eq!(
            memory.read().await.unwrap(),
            "- 2026-01-02: use bun\n- 2026-01-03: Prefer tabs\n"
        );

        let removed = memory.forget("prefer TABS").await.unwrap();
        assert_eq!(removed, ["- 2026-01-03: Prefer tabs"]);
        assert_eq!(memory.read().await.unwrap(), "- 2026-01-02: use bun\n");
        assert!(memory.forget("nothing like it").await.is_err());
        assert!(memory.remember(" ", "2026-01-04").await.is_err());
    }

    #[tokio::test]
    async fn concurrent_writers_lose_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let memory = std::sync::Arc::new(Memory::new(dir.path().join("memory.md")));
        let writers = (0..20).map(|i| {
            let memory = memory.clone();
            tokio::spawn(async move { memory.remember(&format!("fact {i}"), "2026-01-01").await })
        });
        for writer in futures::future::join_all(writers).await {
            writer.unwrap().unwrap();
        }
        let text = memory.read().await.unwrap();
        assert_eq!(text.lines().count(), 20);
        for i in 0..20 {
            assert!(text.contains(&format!("fact {i}\n")), "{text}");
        }
    }

    #[test]
    fn the_section_keeps_the_newest_entries_and_says_when_it_cut() {
        let path = Path::new("/m.md");
        let small = section("- a\n", path).unwrap();
        assert!(small.starts_with(HEADING) && small.ends_with("- a\n(end of memory)"));
        assert!(!small.contains("left out"));

        let lines: Vec<String> = (0..2000).map(|i| format!("- entry {i:04}")).collect();
        let big = section(&lines.join("\n"), path).unwrap();
        assert!(big.contains("older characters left out; the full file is /m.md"));
        assert!(big.contains("- entry 1999\n(end of memory)"));
        assert!(!big.contains("- entry 0000"));
        assert!(big.len() < MAX_PROMPT_CHARS + 500);
    }

    #[test]
    fn the_section_is_replaced_in_place_or_appended() {
        let first = section("- a", Path::new("/m")).unwrap();
        let second = section("- b", Path::new("/m")).unwrap();

        let prompt = with_section(Some(SystemPrompt::Text("base".into())), Some(&first)).unwrap();
        let prompt = with_section(Some(prompt), Some(&second)).unwrap().text();
        assert_eq!(prompt, format!("base\n\n{second}"));
        let cleared = with_section(Some(SystemPrompt::Text(prompt)), None).unwrap();
        assert_eq!(cleared.text(), "base");

        let blocks = SystemPrompt::Blocks(vec![
            SystemContentBlock::Text("base".into()),
            SystemContentBlock::CachePoint(Default::default()),
        ]);
        let with = with_section(Some(blocks), Some(&first)).unwrap();
        let replaced = with_section(Some(with), Some(&second)).unwrap();
        let SystemPrompt::Blocks(blocks) = &replaced else {
            panic!("blocks")
        };
        assert_eq!(blocks.len(), 3);
        assert!(replaced.text().ends_with(&second) && !replaced.text().contains("- a"));
        let SystemPrompt::Blocks(blocks) = with_section(Some(replaced), None).unwrap() else {
            panic!("blocks")
        };
        assert_eq!(blocks.len(), 2);
    }
}
