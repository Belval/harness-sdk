//! Commit rules (`--commit-rule 'PATH_GLOB=REGEX'`): commits in a repository whose path matches
//! the glob must carry a message with a line matching the regex (a ticket id, say).
//!
//! Enforced at commit time, whatever tool the agent commits or lands with: every agent shell, job
//! and land check gets git config through the environment (`GIT_CONFIG_COUNT`, …) that points
//! `core.hooksPath` at a hooks directory the swarm writes ([`CommitHooks::install`]). Its
//! `commit-msg` checks the rules; every hook then runs the repository's own one (in its
//! configured `core.hooksPath`, else `hooks/` of its common git dir), so a repository no rule
//! matches behaves as before. `land` checks the commits it lands too ([`check_range`]), the
//! backstop for a commit made with `--no-verify`.
//!
//! Regexes are POSIX extended regexes as `grep -E` reads them, in the hook and in `land` alike.
//! Merge commits are exempt (they record no work of their own); rebases, cherry-picks and `git
//! am` copy messages without running `commit-msg`, so landing tools keep working.

use std::collections::BTreeMap;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::land::{glob_match, sh_quote};
use crate::worktree::git;

/// One `--commit-rule`: commits in a repository at a path matching `glob` (`*`, `?`) need a
/// message line matching `regex`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitRule {
    /// Path glob of the repositories the rule covers.
    pub glob: String,
    /// `grep -E` regex some line of a commit message must match.
    pub regex: String,
}

impl std::str::FromStr for CommitRule {
    type Err = String;

    /// `PATH_GLOB=REGEX`, split at the first `=`.
    fn from_str(spec: &str) -> Result<Self, String> {
        match spec.split_once('=') {
            Some((glob, regex)) if !glob.is_empty() && !regex.is_empty() => Ok(Self {
                glob: glob.to_owned(),
                regex: regex.to_owned(),
            }),
            _ => Err(format!("expected PATH_GLOB=REGEX, got `{spec}`")),
        }
    }
}

impl CommitRule {
    /// Why `regex` is no `grep -E` pattern (an error or a warning, e.g. `\d`), if it is not one.
    fn invalid(&self) -> Option<String> {
        let output = std::process::Command::new("grep")
            .args(["-Eq", "-e", &self.regex])
            .stdin(std::process::Stdio::null())
            .output();
        match output {
            Ok(out) if out.status.code() == Some(1) && out.stderr.is_empty() => None,
            Ok(out) => Some(String::from_utf8_lossy(&out.stderr).trim().to_owned()),
            Err(error) => Some(format!("could not run grep: {error}")),
        }
    }

    fn refusal(&self) -> String {
        format!(
            "this repo needs a ticket id matching {} in the message (--commit-rule)",
            self.regex
        )
    }
}

/// Hooks [`CommitHooks::install`] writes: every one git runs client-side, so none of a
/// repository's own stops running, and the server-side ones a local push runs. Left out:
/// `push-to-checkout` and `proc-receive`, whose mere presence changes what git does.
const HOOKS: &[&str] = &[
    "applypatch-msg",
    "pre-applypatch",
    "post-applypatch",
    "pre-commit",
    "pre-merge-commit",
    "prepare-commit-msg",
    "commit-msg",
    "post-commit",
    "pre-rebase",
    "post-checkout",
    "post-merge",
    "pre-push",
    "pre-receive",
    "update",
    "post-receive",
    "post-update",
    "reference-transaction",
    "pre-auto-gc",
    "post-rewrite",
    "sendemail-validate",
    "post-index-change",
    "p4-changelist",
    "p4-prepare-changelist",
    "p4-post-changelist",
    "p4-pre-submit",
];

/// The swarm's hooks directory and the rules its `commit-msg` enforces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitHooks {
    /// The hooks directory (canonical), what `core.hooksPath` points at.
    pub dir: PathBuf,
    /// The rules its `commit-msg` enforces.
    pub rules: Vec<CommitRule>,
}

impl CommitHooks {
    /// Validate `rules` and (re)write the hooks in `dir`; a refusal appends a `commit_refused`
    /// line to `events` (the run's `events.jsonl`) when given.
    pub fn install(dir: &Path, rules: Vec<CommitRule>, events: Option<&Path>) -> std::io::Result<Self> {
        for rule in &rules {
            if let Some(why) = rule.invalid() {
                return Err(std::io::Error::other(format!(
                    "--commit-rule '{}={}': not a grep -E regex: {why}",
                    rule.glob, rule.regex
                )));
            }
        }
        std::fs::create_dir_all(dir)?;
        let dir = std::fs::canonicalize(dir)?;
        let script = script(&dir, &rules, events);
        for hook in HOOKS {
            let path = dir.join(hook);
            let tmp = dir.join(format!(".{hook}.tmp-{}", std::process::id()));
            let mut file = std::fs::File::create(&tmp)?;
            file.write_all(script.as_bytes())?;
            file.set_permissions(std::fs::Permissions::from_mode(0o755))?;
            drop(file);
            std::fs::rename(&tmp, &path)?;
        }
        Ok(Self { dir, rules })
    }

    /// Point `core.hooksPath` at the hooks through `env`, after any config entries `env` (or, if
    /// it has none, this process's environment) passes already.
    pub fn add_env(&self, env: &mut BTreeMap<String, String>) {
        let count = env
            .get("GIT_CONFIG_COUNT")
            .cloned()
            .or_else(|| std::env::var("GIT_CONFIG_COUNT").ok())
            .and_then(|count| count.trim().parse::<usize>().ok())
            .unwrap_or(0);
        env.insert(format!("GIT_CONFIG_KEY_{count}"), "core.hooksPath".into());
        env.insert(
            format!("GIT_CONFIG_VALUE_{count}"),
            self.dir.to_string_lossy().into_owned(),
        );
        env.insert("GIT_CONFIG_COUNT".into(), (count + 1).to_string());
    }
}

/// The hook script: every hook is a copy, telling which one it is by its name.
fn script(dir: &Path, rules: &[CommitRule], events: Option<&Path>) -> String {
    let checks: String = rules
        .iter()
        .map(|rule| {
            format!(
                "  rule {} {} || exit 1\n",
                sh_quote(&rule.glob),
                sh_quote(&rule.regex)
            )
        })
        .collect();
    let events = events.map_or_else(|| "''".to_owned(), |e| sh_quote(&e.to_string_lossy()));
    format!(
        r#"#!/bin/sh
# Written by strands-swarm for --commit-rule (rewritten at every start): enforces the commit
# rules, then runs the repository's own hook of the same name.
hook=${{0##*/}}
self={dir}
events={events}
json() {{ printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }}
# Merge commits are exempt: they record no work of their own.
if [ "$hook" = commit-msg ] && ! git rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1; then
  top=$(git rev-parse --show-toplevel 2>/dev/null)
  common=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
  main=${{common%/.git}}
  # What git keeps of the message: no comment lines, nothing below a `commit -v` scissors line.
  msg=$(sed -e '/^# -* >8 -*$/,$d' -e '/^#/d' "$1")
  rule() {{
    case $top in $1) ;; *) case $main in $1) ;; *) return 0 ;; esac ;; esac
    printf '%s\n' "$msg" | grep -Eq -e "$2" && return 0
    echo "commit refused: this repo needs a ticket id matching $2 in the message (--commit-rule)" >&2
    if [ -n "$events" ] && [ -f "$events" ]; then
      printf '{{"ts":"%s","event":"commit_refused","repo":"%s","agent":"%s","regex":"%s"}}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%S.000Z)" "$(json "$main")" \
        "$(json "${{STRANDS_SWARM_SHELL##*#}}")" "$(json "$2")" >>"$events"
    fi
    return 1
  }}
{checks}fi
# The repository's own hook: its configured core.hooksPath (not the one the swarm passes), else
# hooks/ in its common git dir.
own=$(git config --show-scope --get-all core.hooksPath 2>/dev/null | grep -v '^command' | tail -n 1 | cut -f 2-)
case $own in
  '') own=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)/hooks ;;
  '~/'*) own=$HOME/${{own#'~/'}} ;;
esac
if [ "$own" != "$self" ] && [ -f "$own/$hook" ] && [ -x "$own/$hook" ]; then
  exec "$own/$hook" "$@"
fi
case $hook in
  pre-push|pre-receive|post-receive|reference-transaction|post-rewrite) cat >/dev/null ;;
esac
exit 0
"#,
        dir = sh_quote(&dir.to_string_lossy()),
        checks = if checks.is_empty() {
            "  :\n".to_owned()
        } else {
            checks
        },
    )
}

/// The rules that apply to the repository `repo` is in: those whose glob matches its top level
/// or its main checkout (a worktree's main repository).
pub async fn rules_for<'a>(repo: &Path, rules: &'a [CommitRule]) -> Vec<&'a CommitRule> {
    if rules.is_empty() {
        return Vec::new();
    }
    let top = git(repo, &["rev-parse", "--show-toplevel"])
        .await
        .unwrap_or_default();
    let common = git(repo, &["rev-parse", "--path-format=absolute", "--git-common-dir"])
        .await
        .unwrap_or_default();
    let main = common.strip_suffix("/.git").unwrap_or(&common);
    rules
        .iter()
        .filter(|rule| glob_match(&rule.glob, &top) || glob_match(&rule.glob, main))
        .collect()
}

/// Whether some line of `message` matches the `grep -E` `regex`.
async fn matches(regex: &str, message: &str) -> bool {
    use tokio::io::AsyncWriteExt;
    let child = tokio::process::Command::new("grep")
        .args(["-Eq", "-e", regex])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
    let Ok(mut child) = child else {
        return false;
    };
    if let Some(mut stdin) = child.stdin.take() {
        // grep -q may stop reading at the first match.
        let _ = stdin.write_all(message.as_bytes()).await;
    }
    matches!(child.wait().await, Ok(status) if status.success())
}

/// `Err` naming the first commit in `range` (merges aside) whose message breaks a rule that
/// applies to `repo`'s repository.
pub async fn check_range(repo: &Path, rules: &[CommitRule], range: &str) -> Result<(), String> {
    let rules = rules_for(repo, rules).await;
    if rules.is_empty() {
        return Ok(());
    }
    let log = git(repo, &["log", "--no-merges", "-z", "--format=%h %s%n%B", range]).await?;
    for entry in log.split('\0').filter(|entry| !entry.trim().is_empty()) {
        let (commit, message) = entry.trim_start().split_once('\n').unwrap_or((entry, ""));
        for rule in &rules {
            if !matches(&rule.regex, message).await {
                return Err(format!("commit {commit}: {}", rule.refusal()));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `sh -c script` in `dir` with `env` (git identity included): success and stderr.
    fn sh(dir: &Path, env: &BTreeMap<String, String>, script: &str) -> (bool, String) {
        let mut command = std::process::Command::new("sh");
        command.arg("-c").arg(script).current_dir(dir).envs(env);
        for var in crate::worktree::GIT_LOCATION_VARS {
            command.env_remove(var);
        }
        for (key, value) in [
            ("GIT_AUTHOR_NAME", "t"),
            ("GIT_AUTHOR_EMAIL", "t@t"),
            ("GIT_COMMITTER_NAME", "t"),
            ("GIT_COMMITTER_EMAIL", "t@t"),
            ("GIT_CONFIG_NOSYSTEM", "1"),
            ("HOME", &dir.to_string_lossy()),
        ] {
            command.env(key, value);
        }
        let output = command.output().unwrap();
        (
            output.status.success(),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        )
    }

    struct Fixture {
        dir: tempfile::TempDir,
        env: BTreeMap<String, String>,
        events: PathBuf,
    }

    impl Fixture {
        fn path(&self, name: &str) -> PathBuf {
            self.dir.path().join(name)
        }

        fn sh(&self, dir: &str, script: &str) -> (bool, String) {
            sh(&self.path(dir), &self.env, script)
        }

        fn ok(&self, dir: &str, script: &str) {
            let (ok, stderr) = self.sh(dir, script);
            assert!(ok, "{script}: {stderr}");
        }

        /// A repository at `name` with one ticketed commit on `main`.
        fn repo(&self, name: &str) {
            std::fs::create_dir_all(self.path(name)).unwrap();
            self.ok(
                name,
                "git init -q -b main && git config commit.gpgsign false && echo a > a && \
                 git add a && git commit -qm 'init (V24000001)'",
            );
        }
    }

    const TICKET: &str = r"\(V24[0-9]{6,}\)";

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let events = dir.path().join("run/events.jsonl");
        std::fs::create_dir_all(events.parent().unwrap()).unwrap();
        std::fs::write(&events, "").unwrap();
        let glob = format!("{}/quickpdf*", dir.path().canonicalize().unwrap().display());
        let rules = vec![CommitRule {
            glob,
            regex: TICKET.into(),
        }];
        let hooks = CommitHooks::install(&dir.path().join("run/git-hooks"), rules, Some(&events)).unwrap();
        let mut env = BTreeMap::new();
        hooks.add_env(&mut env);
        Fixture { dir, env, events }
    }

    #[test]
    fn a_rule_is_a_glob_and_a_regex_split_at_the_first_equals() {
        let rule: CommitRule = "/w/q*=a=(b)".parse().unwrap();
        assert_eq!((rule.glob.as_str(), rule.regex.as_str()), ("/w/q*", "a=(b)"));
        assert!("/w/q*".parse::<CommitRule>().is_err());
        assert!("=x".parse::<CommitRule>().is_err());
        let bad = CommitRule {
            glob: "*".into(),
            regex: "(".into(),
        };
        assert!(CommitHooks::install(Path::new("/nonexistent/x"), vec![bad], None).is_err());
    }

    #[test]
    fn config_entries_already_passed_are_kept() {
        let hooks = CommitHooks {
            dir: "/run/git-hooks".into(),
            rules: Vec::new(),
        };
        let mut env = BTreeMap::from([
            ("GIT_CONFIG_COUNT".to_owned(), "1".to_owned()),
            ("GIT_CONFIG_KEY_0".to_owned(), "user.name".to_owned()),
        ]);
        hooks.add_env(&mut env);
        assert_eq!(env["GIT_CONFIG_COUNT"], "2");
        assert_eq!(env["GIT_CONFIG_KEY_1"], "core.hooksPath");
        assert_eq!(env["GIT_CONFIG_VALUE_1"], "/run/git-hooks");
        assert_eq!(env["GIT_CONFIG_KEY_0"], "user.name");
    }

    #[test]
    fn a_matching_repo_refuses_a_commit_without_a_ticket_and_takes_one_with_it() {
        let fx = fixture();
        fx.repo("quickpdf");
        let (ok, stderr) = fx.sh(
            "quickpdf",
            "echo b > b && git add b && git commit -qm 'no ticket'",
        );
        assert!(!ok, "refused");
        assert!(
            stderr.contains(&format!(
                "commit refused: this repo needs a ticket id matching {TICKET} in the message \
                 (--commit-rule)"
            )),
            "{stderr}"
        );
        let events = std::fs::read_to_string(&fx.events).unwrap();
        let event: serde_json::Value = serde_json::from_str(events.trim()).unwrap();
        assert_eq!(event["event"], "commit_refused");
        assert_eq!(event["regex"], TICKET);
        assert!(event["repo"].as_str().unwrap().ends_with("/quickpdf"), "{event}");
        // The ticket may be on any line; comment lines do not count.
        fx.ok("quickpdf", "git commit -qm 'fix: thing' -m 'Ticket (V24123456)'");
        let (ok, _) = fx.sh(
            "quickpdf",
            "echo c > c && git add c && printf 'x\\n# (V24123456)\\n' > m && \
             git commit -q --cleanup=strip -F m",
        );
        assert!(!ok, "a comment line is not the message");
    }

    #[test]
    fn a_repo_no_rule_matches_commits_as_before_and_its_own_hooks_still_run() {
        let fx = fixture();
        fx.repo("other");
        fx.ok("other", "echo b > b && git add b && git commit -qm 'no ticket'");
        // The repository's own hooks run: in hooks/ of its git dir, and in a configured
        // core.hooksPath (relative to the top level).
        fx.ok(
            "other",
            "printf '#!/bin/sh\\necho \"$1\" > ran-commit-msg\\n' > .git/hooks/commit-msg && \
             chmod +x .git/hooks/commit-msg && git commit -q --allow-empty -m two && \
             test -f ran-commit-msg",
        );
        fx.ok(
            "other",
            "mkdir own && printf '#!/bin/sh\\necho no >&2; exit 1\\n' > own/pre-commit && \
             chmod +x own/pre-commit && git config core.hooksPath own",
        );
        let (ok, stderr) = fx.sh("other", "git commit -q --allow-empty -m three");
        assert!(
            !ok && stderr.contains("no"),
            "the configured hook refused: {stderr}"
        );
    }

    #[test]
    fn a_matching_repo_still_runs_its_own_hooks_after_the_rule() {
        let fx = fixture();
        fx.repo("quickpdf");
        fx.ok(
            "quickpdf",
            "printf '#!/bin/sh\\ngrep -q nope \"$1\" && exit 1; exit 0\\n' > .git/hooks/commit-msg && \
             chmod +x .git/hooks/commit-msg",
        );
        let (ok, _) = fx.sh("quickpdf", "git commit -q --allow-empty -m 'nope (V24123456)'");
        assert!(!ok, "the repository's own commit-msg refused");
        fx.ok("quickpdf", "git commit -q --allow-empty -m 'yes (V24123456)'");
    }

    #[test]
    fn a_worktree_of_a_matching_repo_is_covered_wherever_it_is() {
        let fx = fixture();
        fx.repo("quickpdf");
        fx.ok("quickpdf", "git worktree add -q -b w ../elsewhere");
        let (ok, stderr) = fx.sh("elsewhere", "git commit -q --allow-empty -m 'no ticket'");
        assert!(!ok && stderr.contains("commit refused"), "{stderr}");
        fx.ok("elsewhere", "git commit -q --allow-empty -m 'ok (V24123456)'");
    }

    #[test]
    fn rebases_merges_and_ref_hooks_keep_working() {
        let fx = fixture();
        fx.repo("quickpdf");
        // A reference-transaction hook (like a guard of main) still sees ref updates.
        fx.ok(
            "quickpdf",
            "printf '#!/bin/sh\\ncat >> \"$(git rev-parse --git-common-dir)/refs.log\"\\n' \
             > .git/hooks/reference-transaction && chmod +x .git/hooks/reference-transaction && \
             git checkout -q -b side && echo s > s && git add s && git commit -qm 's (V24000002)' && \
             git checkout -q main && echo m > m && git add m && git commit -qm 'm (V24000003)' && \
             git checkout -q side && git rebase -q main && \
             git commit -q --allow-empty -m 'fixup! s (V24000002)' && \
             GIT_SEQUENCE_EDITOR=true git rebase -q -i --autosquash main && \
             git checkout -q main && git merge -q --no-ff --no-edit side && \
             git revert --no-edit HEAD~1 >/dev/null 2>&1; \
             test -s .git/refs.log",
        );
        let (ok, stderr) = fx.sh(
            "quickpdf",
            "git log --format=%s -1 | grep -q 'Revert \"m (V24000003)\"'",
        );
        assert!(ok, "a revert of a ticketed commit keeps the ticket: {stderr}");
    }

    #[tokio::test]
    async fn land_ranges_are_checked_against_the_rules_of_their_repo() {
        let fx = fixture();
        fx.repo("quickpdf");
        fx.ok(
            "quickpdf",
            "git checkout -q -b f && git commit -q --no-verify --allow-empty -m 'sneaky' && \
             git checkout -q -b g main && git commit -q --allow-empty -m 'good (V24000004)'",
        );
        let rules = [CommitRule {
            glob: format!("{}/quickpdf*", fx.dir.path().canonicalize().unwrap().display()),
            regex: TICKET.into(),
        }];
        let repo = fx.path("quickpdf");
        let error = check_range(&repo, &rules, "main..f").await.unwrap_err();
        assert!(
            error.contains("sneaky") && error.contains("--commit-rule"),
            "{error}"
        );
        check_range(&repo, &rules, "main..g").await.unwrap();
        let other = [CommitRule {
            glob: "/elsewhere/*".into(),
            regex: TICKET.into(),
        }];
        check_range(&repo, &other, "main..f").await.unwrap();
    }
}
