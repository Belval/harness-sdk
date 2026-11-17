//! `scripts/swarmctl`'s own logs stay bounded.

#[test]
fn swarmctl_rotates_its_logs_past_the_cap() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("stderr.log");
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/swarmctl");
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
