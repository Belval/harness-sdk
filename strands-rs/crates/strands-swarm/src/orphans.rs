//! Shell commands a swarm that died hard left running.
//!
//! A shell call runs in a session of its own, which the SDK kills when the call ends or is cut,
//! and a clean stop (quit, reload) cuts every call in flight. A swarm killed with `kill -9` or by
//! the OOM killer runs no such cleanup, so the commands its calls had in flight (a test run, a
//! build) keep running, holding CPU and cargo's locks while the resumed agents start the same
//! work again.
//!
//! So every agent's shell carries [`SHELL_MARKER`] naming the run and the agent, a clean stop
//! leaves [`CLEAN_STOP_FILE`] in the run directory, and a resume without it kills, for each agent
//! whose saved batch had a shell call cut ([`crate::resume::OpenBatch::cut_shell`]), the process
//! groups led by a process with that agent's marker: what cutting the call would have killed.
//! Processes an earlier call left behind on purpose (`nohup … &`) are not group leaders and run
//! on, as do background jobs, which carry no marker.

use std::path::Path;

use nix::sys::signal::Signal;

/// Environment variable set on agents' shell commands: `<run dir>#<agent id>`.
pub const SHELL_MARKER: &str = "STRANDS_SWARM_SHELL";
/// Left in the run directory by a clean stop; a resume takes it.
pub const CLEAN_STOP_FILE: &str = "clean-stop";

/// The value of [`SHELL_MARKER`] for `agent`'s shell commands in the run in `run_dir`.
pub fn marker(run_dir: &Path, agent: &str) -> String {
    format!("{}#{agent}", run_dir.display())
}

/// Mark the run in `run_dir` as stopped cleanly.
pub fn mark_clean_stop(run_dir: &Path) -> std::io::Result<()> {
    std::fs::write(run_dir.join(CLEAN_STOP_FILE), "")
}

/// Whether the last process of the run in `run_dir` stopped cleanly; removes the mark, so a crash
/// of this process is not taken for a clean stop.
pub fn take_clean_stop(run_dir: &Path) -> bool {
    std::fs::remove_file(run_dir.join(CLEAN_STOP_FILE)).is_ok()
}

/// Whether a NUL-separated environment block (`/proc/<pid>/environ`) holds `entry`.
fn carries(environ: &[u8], entry: &[u8]) -> bool {
    environ.split(|byte| *byte == 0).any(|var| var == entry)
}

/// The process group of a `/proc/<pid>/stat` line (the field after state and parent).
fn group_of(stat: &str) -> Option<u32> {
    let rest = &stat[stat.rfind(')')? + 1..];
    rest.split_whitespace().nth(2)?.parse().ok()
}

/// The process groups (other than this process's) led by a process carrying `marker`. Empty
/// where there is no `/proc`.
pub fn groups(marker: &str) -> Vec<u32> {
    let entry = format!("{SHELL_MARKER}={marker}").into_bytes();
    let Ok(procs) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    let own = std::process::id();
    procs
        .flatten()
        .filter_map(|proc| proc.file_name().to_str()?.parse::<u32>().ok())
        .filter(|pid| *pid != own)
        .filter(|pid| {
            let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
            group_of(&stat) == Some(*pid)
        })
        .filter(|pid| {
            std::fs::read(format!("/proc/{pid}/environ")).is_ok_and(|environ| carries(&environ, &entry))
        })
        .collect()
}

/// Kill the process groups of `agents`' cut shell calls ([`groups`]); returns how many.
pub fn kill_cut_calls(run_dir: &Path, agents: &[String]) -> usize {
    let groups: Vec<u32> = agents
        .iter()
        .flat_map(|agent| groups(&marker(run_dir, agent)))
        .collect();
    for group in &groups {
        // It may have ended since the scan; it is gone either way.
        crate::jobs::signal_group(*group, Signal::SIGKILL);
    }
    groups.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_marker_must_match_the_whole_entry() {
        let entry = format!("{SHELL_MARKER}={}", marker(Path::new("/runs/a"), "a2")).into_bytes();
        assert!(carries(b"PATH=/bin\0STRANDS_SWARM_SHELL=/runs/a#a2\0", &entry));
        assert!(!carries(b"STRANDS_SWARM_SHELL=/runs/a#a22\0", &entry));
        assert!(!carries(b"X=STRANDS_SWARM_SHELL=/runs/a#a2", &entry));
    }

    #[test]
    fn the_group_is_read_after_the_command_name() {
        assert_eq!(group_of("42 (sh) S 1 42 42 0 -1"), Some(42));
        assert_eq!(group_of("43 (a) b) (c) S 42 41 41 0"), Some(41));
        assert_eq!(group_of(""), None);
    }

    #[test]
    fn a_resume_takes_the_clean_stop_mark() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!take_clean_stop(dir.path()));
        mark_clean_stop(dir.path()).unwrap();
        assert!(take_clean_stop(dir.path()));
        assert!(
            !take_clean_stop(dir.path()),
            "a crash after this resume is no clean stop"
        );
    }

    /// `sh -c script` in a session of its own, as a shell call runs, with `agent`'s marker.
    #[cfg(target_os = "linux")]
    fn shell_call(run: &Path, agent: &str, script: &str) -> tokio::process::Child {
        let mut command = tokio::process::Command::new("sh");
        command.args(["-c", script]).env(SHELL_MARKER, marker(run, agent));
        strands_agents::sandbox::detach_from_terminal(&mut command);
        command.spawn().unwrap()
    }

    #[cfg(target_os = "linux")]
    fn alive(pid: u32) -> bool {
        std::fs::read_to_string(format!("/proc/{pid}/stat"))
            .is_ok_and(|stat| !stat[stat.rfind(')').unwrap() + 1..].trim_start().starts_with('Z'))
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn only_the_cut_calls_of_the_agents_named_are_killed() {
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path().join("run");
        let pid_file = dir.path().join("nohup.pid");
        // An earlier call of a2 that left a process running on purpose, then ended.
        let mut earlier = shell_call(
            &run,
            "a2",
            &format!(
                "nohup sleep 30 >/dev/null 2>&1 & echo $! > {}",
                pid_file.display()
            ),
        );
        earlier.wait().await.unwrap();
        let left: u32 = std::fs::read_to_string(&pid_file)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let mut cut = shell_call(&run, "a2", "sleep 30; true");
        let mut other_agent = shell_call(&run, "a3", "sleep 30; true");
        let mut other_run = shell_call(&dir.path().join("other"), "a2", "sleep 30; true");
        assert_eq!(groups(&marker(&run, "a2")), [cut.id().unwrap()]);

        assert_eq!(kill_cut_calls(&run, &["a2".to_owned()]), 1);
        assert!(!cut.wait().await.unwrap().success());
        assert!(alive(left), "a process left on purpose runs on");
        for child in [&mut other_agent, &mut other_run] {
            assert!(
                child.try_wait().unwrap().is_none(),
                "another agent's or run's call runs on"
            );
            child.kill().await.unwrap();
        }
        let _ = std::process::Command::new("kill").arg(left.to_string()).status();
        assert_eq!(kill_cut_calls(&run, &["a2".to_owned()]), 0);
    }
}
