//! The real binary, started the way a script or an agent's shell would.

use std::process::{Command, Stdio};

/// Without a terminal the TUI refuses to start with a clear error (naming `--headless`) instead
/// of panicking, and before it resumes or saves anything.
#[test]
fn the_tui_without_a_terminal_exits_with_an_error_naming_headless() {
    let state = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_strands-swarm"))
        .args(["--run", "notty", "--cwd"])
        .arg(state.path())
        .arg("say hi")
        .env("XDG_STATE_HOME", state.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("needs a terminal"), "{stderr}");
    assert!(stderr.contains("--headless"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    let run = state.path().join("strands-swarm/notty");
    assert!(!run.join("swarm.json").exists(), "nothing was started");
}
