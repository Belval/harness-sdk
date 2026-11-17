//! `scripts/swarmctl`: its logs stay bounded, and its supervisor rolls a failing new binary back,
//! backs off a crash loop and leaves a notice for the swarm.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const SCRIPT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/swarmctl");

#[test]
fn swarmctl_rotates_its_logs_past_the_cap() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("stderr.log");
    let script = SCRIPT;
    let append = |input: &str| {
        let mut child = std::process::Command::new("bash")
            .arg(script)
            .arg("log")
            .arg(&log)
            .env("SWARMCTL_LOG_MAX", "100")
            .stdin(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        use std::io::Write;
        child.stdin.take().unwrap().write_all(input.as_bytes()).unwrap();
        assert!(child.wait().unwrap().success());
    };
    let lines: String = (1..=100).map(|n| format!("line {n}\n")).collect();
    append(&lines);
    append("tail without newline");
    let current = std::fs::read_to_string(&log).unwrap();
    let previous = std::fs::read_to_string(dir.path().join("stderr.log.1")).unwrap();
    assert!(
        current.len() <= 120 && previous.len() <= 120,
        "{current:?} {previous:?}"
    );
    assert!(
        current.ends_with("line 100\ntail without newline\n"),
        "{current:?}"
    );
}

/// A run directory for run `r` under a fake state home, without binaries yet.
struct Run {
    home: tempfile::TempDir,
}

impl Run {
    fn new() -> Self {
        let run = Self {
            home: tempfile::tempdir().unwrap(),
        };
        std::fs::create_dir_all(run.dir().join("bin")).unwrap();
        std::fs::write(run.dir().join("args"), "").unwrap();
        std::fs::write(run.dir().join("env"), "").unwrap();
        run
    }

    fn dir(&self) -> PathBuf {
        self.home.path().join("strands-swarm/r")
    }

    fn bin(&self, name: &str) -> PathBuf {
        self.dir().join("bin").join(name)
    }

    /// Install a fake binary at `path` running `body` (with `$RUN` the run directory), installed
    /// `age` ago (its mtime). Written by a child `sh`, so no test thread holds it open for writing
    /// when another forks ("Text file busy").
    fn binary(&self, path: &Path, body: &str, age: &str) {
        let script = format!("#!/bin/sh\nRUN='{}'\n{body}\n", self.dir().display());
        let status = std::process::Command::new("sh")
            .arg("-c")
            .arg(r#"printf '%s' "$1" > "$2" && chmod +x "$2" && touch -d "$3" "$2""#)
            .arg("sh")
            .arg(script)
            .arg(path)
            .arg(age)
            .status()
            .unwrap();
        assert!(status.success());
    }

    /// `swarmctl supervise r` until it returns (a fake binary creates the stop file): its stdout.
    fn supervise(&self) -> String {
        let mut child = std::process::Command::new("bash")
            .arg(SCRIPT)
            .arg("supervise")
            .arg("r")
            .env("XDG_STATE_HOME", self.home.path())
            .env("HOME", self.home.path())
            .env("SWARMCTL_RESTART_DELAY", "0")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        while child.try_wait().unwrap().is_none() {
            if Instant::now() > deadline {
                child.kill().unwrap();
                panic!("the supervisor did not stop: {:?}", self.read("supervisor.log"));
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success(), "{output:?}");
        String::from_utf8(output.stdout).unwrap()
    }

    fn read(&self, name: &str) -> String {
        std::fs::read_to_string(self.dir().join(name)).unwrap_or_default()
    }
}

#[test]
fn a_new_binary_that_fails_is_rolled_back_once_and_the_swarm_is_told() {
    let run = Run::new();
    let crash = r#"echo new >> "$RUN/calls"; echo "deadlock cycle 0" >&2; exit 70"#;
    run.binary(&run.bin("current"), crash, "now");
    let stop = r#"echo old >> "$RUN/calls"; touch "$RUN/stop""#;
    run.binary(&run.bin("previous"), stop, "2 days ago");

    let shown = run.supervise();

    assert_eq!(run.read("calls"), "new\nold\n");
    assert!(!run.bin("previous").exists());
    let failed = std::fs::read_to_string(run.bin("failed")).unwrap();
    assert!(failed.contains("exit 70"), "{failed}");
    let notice = run.read("supervisor.notice");
    assert!(
        notice.contains("The new binary exited with code 70 (deadlock detected)")
            && notice.contains("rolled back to the previous binary")
            && notice.contains("deadlock cycle 0"),
        "{notice}"
    );
    assert!(run.read("supervisor.log").contains("rolled back"));
    assert!(
        shown.contains("deadlock detected") && shown.contains("deadlock cycle 0"),
        "{shown}"
    );
}

#[test]
fn a_proven_binary_in_a_crash_loop_is_restarted_not_rolled_back() {
    let run = Run::new();
    let flaky = r#"echo run >> "$RUN/calls"
[ "$(wc -l < "$RUN/calls")" -ge 4 ] && { touch "$RUN/stop"; exit 0; }
exit 1"#;
    run.binary(&run.bin("current"), flaky, "2 days ago");
    run.binary(&run.bin("previous"), "exit 0", "3 days ago");

    let shown = run.supervise();

    assert_eq!(run.read("calls"), "run\n".repeat(4));
    assert!(run.bin("previous").exists() && !run.bin("failed").exists());
    let notice = run.read("supervisor.notice");
    assert!(notice.contains("The swarm failed 3 times in a row"), "{notice}");
    assert_eq!(notice.matches("in a row").count(), 1, "{notice}");
    assert!(shown.contains("restarting in"), "{shown}");
}

#[test]
fn a_reload_starts_the_new_binary_s_trial_even_if_it_was_installed_long_ago() {
    let run = Run::new();
    let crasher = run.dir().join("crasher");
    run.binary(&crasher, r#"echo new >> "$RUN/calls"; exit 101"#, "now");
    // What `reload_swarm` does: current becomes previous, the new binary current; then exit 75.
    let reload = r#"echo old >> "$RUN/calls"
[ -f "$RUN/reloaded" ] && { touch "$RUN/stop"; exit 0; }
touch "$RUN/reloaded"
mv "$RUN/bin/current" "$RUN/bin/previous"
cp -p "$RUN/crasher" "$RUN/bin/current"
touch -d '1 day ago' "$RUN/bin/current"
exit 75"#;
    run.binary(&run.bin("current"), reload, "2 days ago");

    run.supervise();

    assert_eq!(run.read("calls"), "old\nnew\nold\n");
    assert!(
        run.read("supervisor.notice")
            .contains("The new binary panicked (exit code 101)"),
        "{}",
        run.read("supervisor.notice")
    );
}
