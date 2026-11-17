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
            .env("SWARMCTL_MEMORY_EVENTS", self.home.path().join("memory.events"))
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
fn a_new_binary_killed_from_outside_is_restarted_on_trial_not_rolled_back() {
    let run = Run::new();
    std::fs::write(run.home.path().join("memory.events"), "oom 0\noom_kill 3\n").unwrap();
    // SIGKILL, then SIGTERM, then a crash: still on trial, so that one is rolled back.
    let killed = r#"echo new >> "$RUN/calls"
case "$(wc -l < "$RUN/calls")" in 1) kill -9 $$ ;; 2) kill -15 $$ ;; esac
exit 1"#;
    run.binary(&run.bin("current"), killed, "now");
    run.binary(
        &run.bin("previous"),
        r#"echo old >> "$RUN/calls"; touch "$RUN/stop""#,
        "2 days ago",
    );

    let shown = run.supervise();

    assert_eq!(run.read("calls"), "new\nnew\nnew\nold\n");
    assert!(shown.contains("killed from outside (signal 9)"), "{shown}");
    let notice = run.read("supervisor.notice");
    assert!(notice.contains("The new binary exited with code 1"), "{notice}");
    assert_eq!(notice.matches("rolled back").count(), 1, "{notice}");
}

#[test]
fn a_new_binary_killed_by_the_oom_killer_is_rolled_back() {
    let run = Run::new();
    let events = run.home.path().join("memory.events");
    std::fs::write(&events, "oom 0\noom_kill 3\n").unwrap();
    let oom = format!(
        r#"echo new >> "$RUN/calls"; echo 'oom_kill 4' > '{}'; kill -9 $$"#,
        events.display()
    );
    run.binary(&run.bin("current"), &oom, "now");
    run.binary(
        &run.bin("previous"),
        r#"echo old >> "$RUN/calls"; touch "$RUN/stop""#,
        "2 days ago",
    );

    run.supervise();

    assert_eq!(run.read("calls"), "new\nold\n");
    let notice = run.read("supervisor.notice");
    assert!(
        notice.contains("The new binary was killed by the OOM killer"),
        "{notice}"
    );
}

#[test]
fn a_new_binary_killed_when_oom_kills_cannot_be_counted_is_rolled_back() {
    let run = Run::new();
    run.binary(
        &run.bin("current"),
        r#"echo new >> "$RUN/calls"; kill -9 $$"#,
        "now",
    );
    run.binary(
        &run.bin("previous"),
        r#"echo old >> "$RUN/calls"; touch "$RUN/stop""#,
        "2 days ago",
    );

    run.supervise();

    assert_eq!(run.read("calls"), "new\nold\n");
    assert!(run.read("supervisor.notice").contains("killed by signal 9"));
}

#[test]
fn the_supervisor_gives_the_terminal_back_after_a_swarm_that_could_not() {
    let run = Run::new();
    let raw = r#"echo run >> "$RUN/calls"
[ "$(wc -l < "$RUN/calls")" -ge 2 ] && { stty -a > "$RUN/stty"; touch "$RUN/stop"; exit 0; }
stty raw -echo; kill -15 $$"#;
    run.binary(&run.bin("current"), raw, "2 days ago");
    // `script` gives the supervisor a terminal, as tmux does.
    let status = std::process::Command::new("script")
        .args(["-qec", &format!("bash {SCRIPT} supervise r"), "/dev/null"])
        .env("XDG_STATE_HOME", run.home.path())
        .env("HOME", run.home.path())
        .env("SWARMCTL_RESTART_DELAY", "0")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .status();
    if status.is_err() {
        eprintln!("skipped: no script(1)");
        return;
    }
    let stty = run.read("stty");
    assert!(stty.contains(" icanon") && stty.contains(" echo "), "{stty}");
}

#[test]
fn a_binary_installed_long_ago_but_never_started_is_on_trial_when_it_first_runs() {
    let run = Run::new();
    // A stop came between an idle reload's install and the restart; the run resumes a day later.
    run.binary(
        &run.bin("current"),
        r#"echo new >> "$RUN/calls"; exit 3"#,
        "1 day ago",
    );
    std::fs::write(run.bin("started"), "1 1\n").unwrap();
    run.binary(
        &run.bin("previous"),
        r#"echo old >> "$RUN/calls"; touch "$RUN/stop""#,
        "2 days ago",
    );

    run.supervise();

    assert_eq!(run.read("calls"), "new\nold\n");
    assert!(run.read("supervisor.notice").contains("rolled back"));
}

#[test]
fn boot_refuses_a_run_that_does_not_exist() {
    let tmp = tempfile::tempdir().unwrap();
    // A fake crontab, so that no test ever edits the real one.
    let shims = tmp.path().join("shims");
    std::fs::create_dir_all(&shims).unwrap();
    let crontab = shims.join("crontab");
    let body = format!(
        "#!/bin/sh\necho \"$@\" >> '{}/crontab.calls'\n",
        tmp.path().display()
    );
    std::fs::write(&crontab, body).unwrap();
    std::process::Command::new("chmod")
        .arg("+x")
        .arg(&crontab)
        .status()
        .unwrap();
    let path = format!("{}:{}", shims.display(), std::env::var("PATH").unwrap());

    let output = std::process::Command::new("bash")
        .args([SCRIPT, "boot", "nosuch"])
        .env("XDG_STATE_HOME", tmp.path())
        .env("PATH", path)
        .output()
        .unwrap();

    assert!(!output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("no run nosuch"));
    assert!(!tmp.path().join("crontab.calls").exists());
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

/// `swarmctl start` under a custom `XDG_STATE_HOME` while a tmux server without it is already
/// running: the supervisor and the swarm still find the run there.
#[test]
fn a_custom_state_home_reaches_the_supervisor_through_a_running_tmux_server() {
    let Some(shown) = start_in_tmux(&["--headless"], &[]) else {
        return;
    };
    assert_eq!(
        shown, "$STATE 3 --run r --headless",
        "the supervisor never ran the binary under the custom state home"
    );
}

/// With no arguments for the swarm, the supervisor passes none (not one empty TASK).
#[test]
fn a_start_without_arguments_replays_none() {
    let Some(shown) = start_in_tmux(&[], &[]) else {
        return;
    };
    assert_eq!(shown, "$STATE 2 --run r");
}

/// `start` keeps the swarm's own AWS settings (`STRANDS_SWARM_AWS_*`) next to the user's
/// `AWS_*`, both reaching the swarm unchanged: the swarm calls Bedrock under the first, its agents
/// inherit the second. The tmux server's own `AWS_PROFILE` does not reach it (see
/// [`start_in_tmux`]).
#[test]
fn start_keeps_the_swarm_s_aws_settings_apart_from_the_user_s() {
    let env = [("STRANDS_SWARM_AWS_PROFILE", "swarm"), ("AWS_PROFILE", "mine")];
    let Some(shown) = start_in_tmux(&[], &env) else {
        return;
    };
    assert_eq!(shown, "$STATE 2 --run r swarm-profile=swarm profile=mine");
}

/// `swarmctl start r <fake binary> args` with `env` in a private tmux server that already has a
/// session: what the fake binary saw (`$XDG_STATE_HOME $# $*`, the state home as `$STATE`, then
/// any AWS profiles), or None without tmux. The tmux server has an `AWS_PROFILE` of its own.
fn start_in_tmux(args: &[&str], env: &[(&str, &str)]) -> Option<String> {
    let tmux_works = std::process::Command::new("tmux").arg("-V").output();
    if !tmux_works.is_ok_and(|output| output.status.success()) {
        eprintln!("skipped: no tmux");
        return None;
    }
    let tmp = tempfile::tempdir().unwrap();
    let state = tmp.path().join("state");
    let sockets = tmp.path().join("tmux");
    std::fs::create_dir_all(&sockets).unwrap();
    let tmux = |args: &[&str]| {
        std::process::Command::new("tmux")
            .args(args)
            .env("TMUX_TMPDIR", &sockets)
            .env("HOME", tmp.path())
            // Not in the env file, so the swarm must not see it.
            .env("AWS_PROFILE", "from-the-tmux-server")
            .env_remove("TMUX")
            .env_remove("XDG_STATE_HOME")
            .status()
            .unwrap()
    };
    assert!(tmux(&["new-session", "-d", "-s", "other", "sleep 60"]).success());
    let seen = tmp.path().join("seen");
    let binary = tmp.path().join("fake-swarm");
    let body = format!(
        "#!/bin/sh\necho \"$XDG_STATE_HOME $# $*${{STRANDS_SWARM_AWS_PROFILE:+ swarm-profile=$STRANDS_SWARM_AWS_PROFILE}}${{AWS_PROFILE:+ profile=$AWS_PROFILE}}\" > '{}'\ntouch \"$XDG_STATE_HOME/strands-swarm/r/stop\"\n",
        seen.display()
    );
    std::fs::write(&binary, body).unwrap();
    std::process::Command::new("chmod")
        .arg("+x")
        .arg(&binary)
        .status()
        .unwrap();

    let started = std::process::Command::new("bash")
        .arg(SCRIPT)
        .args(["start", "r"])
        .arg(&binary)
        .args(args)
        .env("XDG_STATE_HOME", &state)
        .env("TMUX_TMPDIR", &sockets)
        .env("HOME", tmp.path())
        .env_remove("TMUX")
        .env_remove("AWS_PROFILE")
        .env_remove("STRANDS_SWARM_AWS_PROFILE")
        .envs(env.iter().copied())
        .output()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    while !seen.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    let shown = std::fs::read_to_string(&seen).unwrap_or_default();
    tmux(&["kill-server"]);
    assert!(started.status.success(), "{started:?}");
    Some(shown.trim().replace(&state.display().to_string(), "$STATE"))
}

/// A private tmux server for `swarmctl` with a fake swarm that records each start in `starts`
/// (`pid args`), stays up until Ctrl-C and, on its first start, runs `first_start_extra`.
struct Tmux {
    tmp: tempfile::TempDir,
}

impl Tmux {
    fn new() -> Option<Self> {
        let works = std::process::Command::new("tmux").arg("-V").output();
        if !works.is_ok_and(|output| output.status.success()) {
            eprintln!("skipped: no tmux");
            return None;
        }
        let tmux = Self {
            tmp: tempfile::tempdir().unwrap(),
        };
        std::fs::create_dir_all(tmux.tmp.path().join("tmux")).unwrap();
        Some(tmux)
    }

    fn state(&self) -> PathBuf {
        self.tmp.path().join("state")
    }

    fn run_dir(&self) -> PathBuf {
        self.state().join("strands-swarm/r")
    }

    fn command(&self, program: &str) -> std::process::Command {
        let mut command = std::process::Command::new(program);
        command
            .env("XDG_STATE_HOME", self.state())
            .env("TMUX_TMPDIR", self.tmp.path().join("tmux"))
            .env("HOME", self.tmp.path())
            .env_remove("TMUX");
        command
    }

    fn tmux(&self, args: &[&str]) -> String {
        let output = self.command("tmux").args(args).output().unwrap();
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn swarmctl(&self, args: &[&str]) -> std::process::Output {
        self.command("bash")
            .arg(SCRIPT)
            .args(args)
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap()
    }

    /// The fake swarm: appends `pid args` to `starts`, ignores everything but Ctrl-C, and on its
    /// first start launches `first_start_extra` as a job that is killed with the swarm.
    fn fake_swarm(&self, first_start_extra: &str) -> PathBuf {
        let binary = self.tmp.path().join("fake-swarm");
        let body = format!(
            r#"#!/bin/sh
RUN='{run}'
echo "$$ $*" >> "$RUN/starts"
trap 'kill $job 2>/dev/null; exit 0' INT TERM
if [ ! -f "$RUN/first" ]; then
  touch "$RUN/first"
  {extra}
fi
while :; do sleep 0.1; done
"#,
            run = self.run_dir().display(),
            extra = first_start_extra,
        );
        std::fs::write(&binary, body).unwrap();
        std::process::Command::new("chmod")
            .arg("+x")
            .arg(&binary)
            .status()
            .unwrap();
        binary
    }

    fn starts(&self) -> Vec<String> {
        std::fs::read_to_string(self.run_dir().join("starts"))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    fn wait_for<T>(&self, what: &str, mut check: impl FnMut() -> Option<T>) -> T {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(found) = check() {
                return found;
            }
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn wait_for_starts(&self, count: usize) {
        self.wait_for("the swarm to start", || {
            (self.starts().len() >= count).then_some(())
        });
    }

    /// `session_id pane_id pane_pid` of the run's session.
    fn pane(&self) -> String {
        self.tmux(&[
            "display-message",
            "-p",
            "-t",
            "swarm-r",
            "#{session_id} #{pane_id} #{pane_pid}",
        ])
    }
}

impl Drop for Tmux {
    fn drop(&mut self) {
        self.tmux(&["kill-server"]);
    }
}

/// An attached client of the session (a control-mode client needs no terminal).
fn attach(tmux: &Tmux) -> std::process::Child {
    let client = tmux
        .command("tmux")
        .args(["-C", "attach", "-t", "swarm-r"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .spawn()
        .unwrap();
    tmux.wait_for("the client to attach", || {
        (!tmux.tmux(&["list-clients", "-t", "swarm-r"]).is_empty()).then_some(())
    });
    client
}

/// `restart` stops the swarm and starts the supervisor again in the same session and pane: an
/// attached client stays attached, the new supervisor and swarm are new processes, and they
/// re-read the arguments.
#[test]
fn restart_keeps_the_session_and_its_clients_and_rereads_the_arguments() {
    let Some(tmux) = Tmux::new() else { return };
    let binary = tmux.fake_swarm("");
    let started = tmux.swarmctl(&["start", "r", binary.to_str().unwrap(), "--one"]);
    assert!(started.status.success(), "{started:?}");
    tmux.wait_for_starts(1);
    let mut client = attach(&tmux);
    let before = tmux.pane();
    std::fs::write(tmux.run_dir().join("args"), "--two\0").unwrap();

    let restarted = tmux.swarmctl(&["restart", "r"]);

    let shown = String::from_utf8_lossy(&restarted.stdout);
    assert!(restarted.status.success(), "{restarted:?}");
    assert!(
        shown.contains("1 client(s) stay attached") && shown.contains("restarted"),
        "{shown}"
    );
    tmux.wait_for_starts(2);
    let starts = tmux.starts();
    assert!(starts[0].ends_with("--run r --one"), "{starts:?}");
    assert!(starts[1].ends_with("--run r --two"), "{starts:?}");
    assert_ne!(starts[0].split(' ').next(), starts[1].split(' ').next());
    let after = tmux.pane();
    let (b, a): (Vec<_>, Vec<_>) = (before.split(' ').collect(), after.split(' ').collect());
    assert_eq!((a[0], a[1]), (b[0], b[1]), "session and pane are kept");
    assert_ne!(a[2], b[2], "a new supervisor process: {before} / {after}");
    assert!(client.try_wait().unwrap().is_none(), "the client was detached");
    assert!(!tmux.tmux(&["list-clients", "-t", "swarm-r"]).is_empty());
    assert!(!tmux.run_dir().join("stop").exists());
    // The remain-on-exit used for the restart is not left on.
    assert_eq!(
        tmux.tmux(&["show-options", "-wv", "-t", "swarm-r", "remain-on-exit"]),
        ""
    );
    // The new supervisor keeps supervising: it is a live pane, not a dead one.
    assert_eq!(
        tmux.tmux(&["display-message", "-p", "-t", "swarm-r", "#{pane_dead}"]),
        "0"
    );
    client.kill().unwrap();
    client.wait().unwrap();
}

/// A restart started by a job of the swarm completes although the job dies with the swarm.
#[test]
fn restart_from_a_job_of_the_swarm_survives_the_job() {
    let Some(tmux) = Tmux::new() else { return };
    let script = format!(
        r#"( sleep 1; bash '{SCRIPT}' restart r > "$RUN/job.out" 2>&1; touch "$RUN/job.survived" ) &
  job=$!"#
    );
    let binary = tmux.fake_swarm(&script);
    let started = tmux.swarmctl(&["start", "r", binary.to_str().unwrap()]);
    assert!(started.status.success(), "{started:?}");
    tmux.wait_for_starts(1);
    let before = tmux.pane();

    tmux.wait_for_starts(2);

    let after = tmux.pane();
    assert_eq!(
        before.split(' ').take(2).collect::<Vec<_>>(),
        after.split(' ').take(2).collect::<Vec<_>>()
    );
    assert_ne!(before, after);
    assert!(!tmux.run_dir().join("stop").exists());
    assert!(
        !tmux.run_dir().join("job.survived").exists(),
        "the job was meant to die with the swarm"
    );
    assert_eq!(
        tmux.tmux(&["display-message", "-p", "-t", "swarm-r", "#{pane_dead}"]),
        "0"
    );
}

/// A run that is not running is resumed by `restart`; `stop` says who it detaches.
#[test]
fn restart_resumes_a_stopped_run_and_stop_tells_clients_they_are_detached() {
    let Some(tmux) = Tmux::new() else { return };
    let binary = tmux.fake_swarm("");
    let started = tmux.swarmctl(&["start", "r", binary.to_str().unwrap()]);
    assert!(started.status.success(), "{started:?}");
    tmux.wait_for_starts(1);
    let mut client = attach(&tmux);

    let stopped = tmux.swarmctl(&["stop", "r"]);

    let shown = String::from_utf8_lossy(&stopped.stdout);
    assert!(
        shown.contains("1 client(s) attached to swarm-r will be detached"),
        "{shown}"
    );
    assert!(shown.contains("stopped"), "{shown}");
    client.wait().unwrap();

    let restarted = tmux.swarmctl(&["restart", "r"]);

    let shown = String::from_utf8_lossy(&restarted.stdout);
    assert!(
        restarted.status.success() && shown.contains("not running; resuming"),
        "{restarted:?}"
    );
    tmux.wait_for_starts(2);
    assert_eq!(
        tmux.tmux(&["display-message", "-p", "-t", "swarm-r", "#{pane_dead}"]),
        "0"
    );
}
