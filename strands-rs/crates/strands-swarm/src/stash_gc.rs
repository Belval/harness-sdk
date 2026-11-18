//! Bounding the agents' context stashes, `sessions/<agent>/stash`.
//!
//! The SDK stashes every tool result and text block of a conversation there under
//! `<tool use or message id>_<index>`, so that `retrieve_context` can read back what the context
//! manager truncated or summarized away, and deletes none. An agent can only retrieve an entry
//! it can name, and it only knows the names its context holds, all of which its saved session
//! holds too. So a live agent's entries that nothing in its session directory names any more are
//! deleted ([`prune_unnamed`]): on resume, and between its turns while the swarm runs (see
//! `Swarm::prune_stash_between_turns`); a cancelled agent can retrieve nothing, so its whole stash
//! goes ([`drop_stash`]).

use std::collections::HashSet;
use std::path::Path;

use serde_json::Value;

/// The stash, inside an agent's session directory.
pub const STASH_DIR: &str = "stash";

/// What a pruning deleted.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Pruned {
    /// Entries.
    pub entries: usize,
    /// Their bytes.
    pub bytes: u64,
}

impl Pruned {
    fn add(&mut self, other: Pruned) {
        self.entries += other.entries;
        self.bytes += other.bytes;
    }
}

/// Delete the stash of the agent whose session directory is `dir`.
pub fn drop_stash(dir: &Path) -> Pruned {
    let stash = dir.join(STASH_DIR);
    let mut pruned = Pruned::default();
    for (_, path, bytes) in entries(&stash) {
        if std::fs::remove_file(path).is_ok() {
            pruned.add(Pruned { entries: 1, bytes });
        }
    }
    let _ = std::fs::remove_dir(&stash);
    pruned
}

/// Delete the entries of `dir`'s stash that no other file in `dir` (the saved conversation, its
/// summaries, an open tool batch) names by the id before its `_<index>`. Nothing is
/// deleted unless the directory holds a saved file and every one of them reads as JSON.
pub fn prune_unnamed(dir: &Path) -> Pruned {
    let mut pruned = Pruned::default();
    let Some(names) = Names::of_session(dir) else {
        return pruned;
    };
    for (key, path, bytes) in entries(&dir.join(STASH_DIR)) {
        if !names.has(&key) && std::fs::remove_file(path).is_ok() {
            pruned.add(Pruned { entries: 1, bytes });
        }
    }
    pruned
}

/// Each `(agent, cancelled)`'s stash under `sessions`, pruned by [`drop_stash`] or
/// [`prune_unnamed`].
pub fn prune_run<'a>(sessions: &Path, agents: impl IntoIterator<Item = (&'a str, bool)>) -> Pruned {
    let mut pruned = Pruned::default();
    for (id, cancelled) in agents {
        let dir = sessions.join(id);
        pruned.add(match cancelled {
            true => drop_stash(&dir),
            false => prune_unnamed(&dir),
        });
    }
    pruned
}

/// The stash's entries: key, path and size of each file directly in it.
fn entries(stash: &Path) -> Vec<(String, std::path::PathBuf, u64)> {
    let Ok(read) = std::fs::read_dir(stash) else {
        return Vec::new();
    };
    read.flatten()
        .filter_map(|entry| {
            let meta = entry.metadata().ok().filter(|meta| meta.is_file())?;
            let key = entry.file_name().into_string().ok()?;
            Some((key, entry.path(), meta.len()))
        })
        .collect()
}

/// Characters of the ids the SDK keys entries by (tool use ids, message UUIDs).
fn id_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// Every name a session's files hold: the id-like words of their strings, and the strings
/// themselves for ids of other characters.
struct Names {
    words: HashSet<String>,
    text: String,
}

impl Names {
    fn of_session(dir: &Path) -> Option<Self> {
        let mut names = Self {
            words: HashSet::new(),
            text: String::new(),
        };
        let mut files = 0;
        let stash = dir.join(STASH_DIR);
        let mut dirs = vec![dir.to_path_buf()];
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(&dir).ok()? {
                let entry = entry.ok()?;
                let path = entry.path();
                let name = entry.file_name();
                if entry.file_type().ok()?.is_dir() {
                    if path != stash {
                        dirs.push(path);
                    }
                    continue;
                }
                // Another process's half-written replacement.
                if name.to_string_lossy().contains(".tmp-") {
                    continue;
                }
                let json: Value = serde_json::from_slice(&std::fs::read(&path).ok()?).ok()?;
                names.add(&json);
                files += 1;
            }
        }
        (files > 0).then_some(names)
    }

    fn add(&mut self, json: &Value) {
        match json {
            Value::String(text) => self.add_text(text),
            Value::Array(items) => items.iter().for_each(|item| self.add(item)),
            Value::Object(map) => map.iter().for_each(|(key, value)| {
                self.add_text(key);
                self.add(value);
            }),
            _ => {}
        }
    }

    fn add_text(&mut self, text: &str) {
        for word in text.split(|c: char| !id_char(c)).filter(|word| !word.is_empty()) {
            // A key names its siblings too: `<id>_0` named, `<id>_1` is a fair guess.
            self.words.insert(id_of(word).to_owned());
            self.words.insert(word.to_owned());
        }
        self.text.push_str(text);
        self.text.push('\n');
    }

    /// Whether `key` (`<id>_<index>`) is named, or its id is.
    fn has(&self, key: &str) -> bool {
        let id = id_of(key);
        match id.chars().all(id_char) {
            true => self.words.contains(id),
            false => self.text.contains(id),
        }
    }
}

/// `key` without its `_<index>`.
fn id_of(key: &str) -> &str {
    key.rsplit_once('_')
        .filter(|(_, index)| index.parse::<usize>().is_ok())
        .map_or(key, |(id, _)| id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn session(files: &[(&str, Value)], stash: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, json) in files {
            let path = dir.path().join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, json.to_string()).unwrap();
        }
        std::fs::create_dir_all(dir.path().join(STASH_DIR)).unwrap();
        for key in stash {
            std::fs::write(dir.path().join(STASH_DIR).join(key), "12345").unwrap();
        }
        dir
    }

    fn left(dir: &Path) -> Vec<String> {
        let mut keys: Vec<_> = entries(&dir.join(STASH_DIR))
            .into_iter()
            .map(|(key, ..)| key)
            .collect();
        keys.sort();
        keys
    }

    const SNAPSHOT: &str = "scopes/agent/a1/snapshots/snapshot_latest.json";

    #[test]
    fn a_live_agent_keeps_the_entries_its_session_names() {
        let conversation = json!({"data": {"messages": [
            {"role": "user", "trackingId": "5f0c-77", "content": [{"text": "Summary: see [ref: tooluse_sum_1]."}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": "tooluse_a", "content": [{"text": "x"}]}}]},
            {"role": "user", "content": [{"text": "the odd one is call:weird.id"}]},
        ]}});
        let batch = json!({"results": [{"toolUseId": "tooluse_batch"}]});
        let dir = session(
            &[(SNAPSHOT, conversation), ("open-batch.json", batch)],
            &[
                "tooluse_a_0",
                "tooluse_a_1",
                "5f0c-77_0",
                "tooluse_sum_1",
                "tooluse_sum_0",
                "call:weird.id_0",
                "tooluse_batch_0",
                "tooluse_old_0",
                "call:gone.id_0",
                "tooluse_a",
            ],
        );
        let pruned = prune_unnamed(dir.path());
        assert_eq!(
            pruned,
            Pruned {
                entries: 2,
                bytes: 10
            }
        );
        assert_eq!(
            left(dir.path()),
            [
                "5f0c-77_0",
                "call:weird.id_0",
                "tooluse_a",
                "tooluse_a_0",
                "tooluse_a_1",
                "tooluse_batch_0",
                "tooluse_sum_0",
                "tooluse_sum_1",
            ]
        );
        assert_eq!(
            prune_unnamed(dir.path()),
            Pruned::default(),
            "nothing more to prune"
        );
    }

    #[test]
    fn a_session_that_cannot_be_read_in_full_keeps_its_stash() {
        let unsaved = session(&[], &["tooluse_old_0"]);
        assert_eq!(prune_unnamed(unsaved.path()), Pruned::default());
        let torn = session(&[(SNAPSHOT, json!({}))], &["tooluse_old_0"]);
        std::fs::write(torn.path().join("open-batch.json"), "{\"res").unwrap();
        assert_eq!(prune_unnamed(torn.path()), Pruned::default());
        assert_eq!(left(torn.path()), ["tooluse_old_0"]);
        // A temporary file is not part of the session.
        std::fs::remove_file(torn.path().join("open-batch.json")).unwrap();
        std::fs::write(torn.path().join("open-batch.json.tmp-1"), "{\"res").unwrap();
        assert_eq!(prune_unnamed(torn.path()).entries, 1);
    }

    #[test]
    fn a_cancelled_agent_loses_its_whole_stash_and_live_ones_their_unnamed_entries() {
        let sessions = tempfile::tempdir().unwrap();
        let snapshot = json!({"messages": [{"toolUseId": "tooluse_a"}]});
        for id in ["a1", "a2"] {
            let dir = sessions.path().join(id);
            std::fs::create_dir_all(dir.join("scopes")).unwrap();
            std::fs::write(dir.join("scopes/snapshot.json"), snapshot.to_string()).unwrap();
            std::fs::create_dir_all(dir.join(STASH_DIR)).unwrap();
            for key in ["tooluse_a_0", "tooluse_b_0"] {
                std::fs::write(dir.join(STASH_DIR).join(key), "123").unwrap();
            }
        }
        let pruned = prune_run(sessions.path(), [("a1", false), ("a2", true), ("a3", true)]);
        assert_eq!(pruned, Pruned { entries: 3, bytes: 9 });
        assert_eq!(left(&sessions.path().join("a1")), ["tooluse_a_0"]);
        assert!(!sessions.path().join("a2").join(STASH_DIR).exists());
        assert!(
            sessions.path().join("a2/scopes/snapshot.json").exists(),
            "the conversation stays"
        );
    }
}
