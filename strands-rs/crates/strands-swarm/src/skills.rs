//! Skills found on this machine: Claude Code and Codex `SKILL.md` directories, discovered once at
//! startup and offered to every agent through the SDK's progressive-disclosure skills plugin
//! (names and descriptions in the system prompt, the full `SKILL.md` on demand).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use strands_agents::plugins::vended::Skill;
use strands_agents::{Error, Result};

/// Longest description shown in the system prompt; the full text is in the loaded skill.
const MAX_DESCRIPTION_CHARS: usize = 300;

/// A directory of skills (or a single skill directory) and where it comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillDir {
    /// Label shown by `--list-skills`, e.g. `user`, `project`, `plugin:<name>`, `codex`.
    pub origin: String,
    /// The directory.
    pub path: PathBuf,
}

impl SkillDir {
    fn new(origin: impl Into<String>, path: PathBuf) -> Self {
        Self {
            origin: origin.into(),
            path,
        }
    }
}

/// A loaded skill and the label of the directory it came from.
#[derive(Debug, Clone)]
pub struct FoundSkill {
    /// The parsed skill.
    pub skill: Skill,
    /// Origin label of its [`SkillDir`].
    pub origin: String,
}

/// The skills to offer, first source wins on a name clash; `shadowed` holds the losers.
#[derive(Debug, Clone, Default)]
pub struct Skills {
    /// Skills in precedence order.
    pub skills: Vec<FoundSkill>,
    /// Skills hidden by an earlier one with the same name.
    pub shadowed: Vec<FoundSkill>,
}

impl Skills {
    /// The skills as given to agents: descriptions capped at [`MAX_DESCRIPTION_CHARS`] to keep
    /// the system prompt small (the full `SKILL.md` body is what matters, and loads on demand).
    pub fn for_agents(&self) -> Vec<Skill> {
        self.skills
            .iter()
            .map(|found| {
                let mut skill = found.skill.clone();
                skill.description = cap(&skill.description);
                skill
            })
            .collect()
    }
}

/// Discover skills: `extra` directories first (each a skill directory or a directory of skills;
/// a missing one is an error), then, unless `defaults` is off, [`default_skill_dirs`].
pub fn discover_skills(extra: &[PathBuf], defaults: bool, cwd: &Path, home: Option<&Path>) -> Result<Skills> {
    let mut dirs = Vec::new();
    for path in extra {
        if !path.is_dir() {
            return Err(Error::other(format!(
                "--skills-dir {} is not a directory",
                path.display()
            )));
        }
        dirs.push(SkillDir::new("--skills-dir", path.clone()));
    }
    if defaults {
        dirs.extend(default_skill_dirs(cwd, home));
    }
    Ok(load_skills(&dirs))
}

/// The standard skill locations in precedence order: project (`<cwd>/.claude/skills`,
/// `.agents/skills`, `.agent/skills`), user (`~/.claude/skills`), installed and enabled Claude
/// Code plugins, then Codex (`~/.codex/skills`, `~/.agents/skills`). Only existing directories.
pub fn default_skill_dirs(cwd: &Path, home: Option<&Path>) -> Vec<SkillDir> {
    let mut dirs: Vec<SkillDir> = [".claude/skills", ".agents/skills", ".agent/skills"]
        .iter()
        .map(|rel| SkillDir::new("project", cwd.join(rel)))
        .collect();
    if let Some(home) = home {
        dirs.push(SkillDir::new("user", home.join(".claude/skills")));
        dirs.extend(claude_plugin_skill_dirs(home, cwd));
        dirs.push(SkillDir::new("codex", home.join(".codex/skills")));
        dirs.push(SkillDir::new("codex", home.join(".agents/skills")));
    }
    let mut seen = Vec::new();
    dirs.retain(|dir| {
        let real = std::fs::canonicalize(&dir.path).ok();
        let keep = real.as_ref().is_some_and(|r| r.is_dir() && !seen.contains(r));
        seen.extend(real);
        keep
    });
    dirs
}

/// Skill directories of the Claude Code plugins in `~/.claude/plugins/installed_plugins.json`
/// that apply here (user scope, or project scope for exactly `cwd`) and are not disabled in
/// `enabledPlugins` of the user or project settings. A plugin's skills live in `<install>/skills`
/// plus any `skills` paths in its `.claude-plugin/plugin.json`.
pub fn claude_plugin_skill_dirs(home: &Path, cwd: &Path) -> Vec<SkillDir> {
    let Some(installed) = read_json(&home.join(".claude/plugins/installed_plugins.json")) else {
        return Vec::new();
    };
    let mut enabled: BTreeMap<String, bool> = BTreeMap::new();
    for settings in [
        home.join(".claude/settings.json"),
        cwd.join(".claude/settings.json"),
        cwd.join(".claude/settings.local.json"),
    ] {
        if let Some(Value::Object(map)) = read_json(&settings).and_then(|s| s.get("enabledPlugins").cloned())
        {
            for (key, value) in map {
                enabled.insert(key, value.as_bool().unwrap_or(true));
            }
        }
    }
    let Some(Value::Object(plugins)) = installed.get("plugins") else {
        return Vec::new();
    };
    let mut dirs = Vec::new();
    for (key, installs) in plugins {
        if enabled.get(key) == Some(&false) {
            continue;
        }
        let Some(install) = applicable_install(installs, cwd) else {
            continue;
        };
        let origin = format!("plugin:{key}");
        dirs.push(SkillDir::new(&origin, install.join("skills")));
        let manifest = read_json(&install.join(".claude-plugin/plugin.json"));
        let extra = match manifest.as_ref().and_then(|m| m.get("skills")) {
            Some(Value::String(path)) => vec![path.clone()],
            Some(Value::Array(paths)) => paths
                .iter()
                .filter_map(|p| p.as_str().map(str::to_owned))
                .collect(),
            _ => Vec::new(),
        };
        dirs.extend(
            extra
                .into_iter()
                .map(|rel| SkillDir::new(&origin, install.join(rel))),
        );
    }
    dirs
}

/// The install path to use: a project-scope install for `cwd`, else the user-scope one. Version 1
/// files have a single install object instead of a list.
fn applicable_install(installs: &Value, cwd: &Path) -> Option<PathBuf> {
    let list = match installs {
        Value::Array(list) => list.clone(),
        other => vec![other.clone()],
    };
    let path = |entry: &Value| entry.get("installPath")?.as_str().map(PathBuf::from);
    let for_cwd = list.iter().find(|entry| {
        entry
            .get("projectPath")
            .and_then(Value::as_str)
            .is_some_and(|project| Path::new(project) == cwd)
    });
    let user = list.iter().find(|entry| {
        let scope = entry.get("scope").and_then(Value::as_str);
        scope.is_none_or(|scope| scope == "user")
    });
    for_cwd.or(user).and_then(path)
}

fn read_json(path: &Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

/// Load every skill under `dirs`; on a name clash the earlier directory wins. Unparsable skills
/// are skipped with a warning.
pub fn load_skills(dirs: &[SkillDir]) -> Skills {
    let mut out = Skills::default();
    for dir in dirs {
        let loaded = match dir.path.join("SKILL.md").is_file() {
            true => Skill::from_file(&dir.path, false).map(|skill| vec![skill]),
            false => Skill::from_directory(&dir.path, false),
        };
        let skills = match loaded {
            Ok(skills) => skills,
            Err(error) => {
                tracing::warn!(path = %dir.path.display(), %error, "skipping skills directory");
                continue;
            }
        };
        for skill in skills {
            let found = FoundSkill {
                skill,
                origin: dir.origin.clone(),
            };
            match out.skills.iter().any(|s| s.skill.name == found.skill.name) {
                true => out.shadowed.push(found),
                false => out.skills.push(found),
            }
        }
    }
    out
}

fn cap(text: &str) -> String {
    match text.char_indices().nth(MAX_DESCRIPTION_CHARS) {
        Some((cut, _)) => format!("{}…", text[..cut].trim_end()),
        None => text.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn skill(dir: &Path, name: &str, description: &str) {
        fs::create_dir_all(dir.join(name)).unwrap();
        fs::write(
            dir.join(name).join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {description}\n---\nDo {name}."),
        )
        .unwrap();
    }

    fn names(skills: &Skills) -> Vec<(String, String)> {
        skills
            .skills
            .iter()
            .map(|s| (s.skill.name.clone(), s.origin.clone()))
            .collect()
    }

    fn write_json(path: &Path, value: serde_json::Value) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, value.to_string()).unwrap();
    }

    #[test]
    fn earlier_sources_win_and_shadowed_skills_are_reported() {
        let root = tempfile::tempdir().unwrap();
        let (cwd, home) = (root.path().join("repo"), root.path().join("home"));
        skill(&cwd.join(".claude/skills"), "review", "project review");
        skill(&home.join(".claude/skills"), "review", "user review");
        skill(&home.join(".claude/skills"), "deploy", "user deploy");
        skill(&home.join(".codex/skills"), "deploy", "codex deploy");
        skill(&home.join(".codex/skills"), "incident", "codex incident");
        fs::create_dir_all(home.join(".codex/skills/.system/hidden")).unwrap();

        let skills = discover_skills(&[], true, &cwd, Some(&home)).unwrap();
        assert_eq!(
            names(&skills),
            [
                ("review".into(), "project".into()),
                ("deploy".into(), "user".into()),
                ("incident".into(), "codex".into()),
            ]
        );
        assert_eq!(skills.skills[0].skill.description, "project review");
        assert_eq!(
            skills.skills[0].skill.path,
            Some(cwd.join(".claude/skills/review"))
        );
        let shadowed: Vec<&str> = skills
            .shadowed
            .iter()
            .map(|s| s.skill.description.as_str())
            .collect();
        assert_eq!(shadowed, ["user review", "codex deploy"]);
    }

    #[test]
    fn explicit_dirs_come_first_and_defaults_can_be_turned_off() {
        let root = tempfile::tempdir().unwrap();
        let home = root.path().join("home");
        skill(&home.join(".claude/skills"), "review", "user review");
        let extra = root.path().join("mine");
        skill(&extra, "review", "my review");
        let single = root.path().join("single");
        skill(root.path(), "single", "one skill directory");

        let skills = discover_skills(&[extra.clone(), single], true, root.path(), Some(&home)).unwrap();
        assert_eq!(skills.skills[0].skill.description, "my review");
        assert_eq!(skills.skills[1].skill.name, "single");
        assert_eq!(skills.shadowed.len(), 1);

        let only = discover_skills(&[extra], false, root.path(), Some(&home)).unwrap();
        assert_eq!(only.skills.len(), 1);
        assert!(only.shadowed.is_empty());
        assert!(discover_skills(&[root.path().join("missing")], true, root.path(), None).is_err());
    }

    #[test]
    fn plugin_skills_come_from_applicable_enabled_installs() {
        let root = tempfile::tempdir().unwrap();
        let (cwd, home) = (root.path().join("repo"), root.path().join("home"));
        let cache = home.join(".claude/plugins/cache");
        skill(&cache.join("core/2/skills"), "brazil", "user-scope install");
        skill(
            &cache.join("core/1/skills"),
            "brazil-old",
            "other project's install",
        );
        skill(
            &cache.join("local/1/skills"),
            "local-skill",
            "project install for cwd",
        );
        skill(
            &cache.join("local/1/extra"),
            "extra-skill",
            "manifest skills path",
        );
        write_json(
            &cache.join("local/1/.claude-plugin/plugin.json"),
            serde_json::json!({ "name": "local", "skills": "extra" }),
        );
        skill(&cache.join("off/1/skills"), "disabled-skill", "disabled plugin");
        let install = |scope: &str, project: Option<&Path>, path: PathBuf| serde_json::json!({ "scope": scope, "projectPath": project, "installPath": path });
        write_json(
            &home.join(".claude/plugins/installed_plugins.json"),
            serde_json::json!({ "version": 2, "plugins": {
                "core@m": [
                    install("project", Some(Path::new("/elsewhere")), cache.join("core/1")),
                    install("user", None, cache.join("core/2")),
                ],
                "local@m": [
                    install("user", None, cache.join("missing")),
                    install("local", Some(&cwd), cache.join("local/1")),
                ],
                "off@m": [install("user", None, cache.join("off/1"))],
            }}),
        );
        write_json(
            &home.join(".claude/settings.json"),
            serde_json::json!({ "enabledPlugins": { "core@m": true, "off@m": true } }),
        );
        write_json(
            &cwd.join(".claude/settings.local.json"),
            serde_json::json!({ "enabledPlugins": { "off@m": false } }),
        );

        let skills = discover_skills(&[], true, &cwd, Some(&home)).unwrap();
        assert_eq!(
            names(&skills),
            [
                ("brazil".into(), "plugin:core@m".into()),
                ("local-skill".into(), "plugin:local@m".into()),
                ("extra-skill".into(), "plugin:local@m".into()),
            ]
        );
    }

    #[test]
    fn agents_get_capped_descriptions_and_full_instructions() {
        let root = tempfile::tempdir().unwrap();
        skill(root.path(), "long", &"word ".repeat(100));
        let skills = discover_skills(&[root.path().to_path_buf()], false, root.path(), None).unwrap();
        let agent_skill = &skills.for_agents()[0];
        assert!(agent_skill.description.ends_with('…'));
        assert!(agent_skill.description.chars().count() <= MAX_DESCRIPTION_CHARS + 1);
        assert_eq!(agent_skill.instructions, "Do long.");
    }
}
