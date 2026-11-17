//! Spawn a local process and stream its output (`strands-ts/src/sandbox/stream-process.ts`).
//!
//! Shared by every process-backed sandbox: [`LocalSandbox`](super::LocalSandbox) spawns `sh -c`,
//! [`DockerSandbox`](super::DockerSandbox) spawns `docker exec`, [`SshSandbox`](super::SshSandbox)
//! spawns `ssh`. The process runs in a session (and group) of its own, so a timeout or a cancelled
//! call kills the whole *local* tree, and nothing it starts can reach the controlling terminal; a
//! command already running in a container or on a remote host keeps running, exactly as in TS.

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::stream::BoxStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::mpsc::{self, UnboundedSender};

use super::{
    ExecuteOptions, ExecutionResult, SandboxError, SandboxEvent, SandboxResult, StreamChunk, StreamType,
};

/// How long the pipe collectors get to finish after the command ended, before their buffers are
/// taken as-is. Bounds the call when a grandchild still holds the inherited pipes open.
const DRAIN_GRACE: Duration = Duration::from_millis(250);

/// Exit code reported when the program itself is missing (TS yields the same for `ENOENT`).
const NOT_FOUND_EXIT_CODE: i32 = 127;

/// A local process to run on behalf of a sandbox backend.
pub(super) struct Spawn {
    program: OsString,
    args: Vec<OsString>,
    cwd: Option<PathBuf>,
    env: Option<HashMap<String, String>>,
    enoent_message: Option<&'static str>,
}

impl Spawn {
    /// Run `program` with `args`, inheriting this process's directory and environment.
    pub(super) fn new<A>(program: impl Into<OsString>, args: impl IntoIterator<Item = A>) -> Self
    where
        A: Into<OsString>,
    {
        Self {
            program: program.into(),
            args: args.into_iter().map(Into::into).collect(),
            cwd: None,
            env: None,
            enoent_message: None,
        }
    }

    /// Working directory of the local process. Remote backends pass the directory to the remote
    /// end instead (`docker exec -w`, `cd` over ssh) and leave this unset.
    pub(super) fn cwd(mut self, cwd: Option<PathBuf>) -> Self {
        self.cwd = cwd;
        self
    }

    /// Variables added to the local process's environment. Remote backends pass environment
    /// variables to the remote end instead.
    pub(super) fn env(mut self, env: Option<HashMap<String, String>>) -> Self {
        self.env = env;
        self
    }

    /// Report a missing program as exit 127 with `message` on stderr instead of an error, the way
    /// TS's `enoentMessage` does.
    pub(super) fn enoent_message(mut self, message: &'static str) -> Self {
        self.enoent_message = Some(message);
        self
    }
}

/// Run the process to completion and return its terminal result.
pub(super) async fn run(spawn: Spawn, mut options: ExecuteOptions) -> SandboxResult<ExecutionResult> {
    match launch(spawn, &mut options, None)? {
        Launched::Missing(result) => Ok(result),
        Launched::Running(running) => drive(running, &options).await,
    }
}

/// Run the process, yielding output as it arrives and the terminal result last.
pub(super) fn stream(
    spawn: Spawn,
    mut options: ExecuteOptions,
) -> BoxStream<'static, SandboxResult<SandboxEvent>> {
    Box::pin(async_stream::stream! {
        let (chunks, mut rx) = mpsc::unbounded_channel();
        let running = match launch(spawn, &mut options, Some(chunks)) {
            Err(error) => {
                yield Err(error);
                return;
            }
            Ok(Launched::Missing(result)) => {
                yield Ok(SandboxEvent::Result(result));
                return;
            }
            Ok(Launched::Running(running)) => running,
        };

        // `drive` is created once and only ever polled, so re-arming a cancel-unsafe future
        // across loop iterations cannot happen.
        let mut driver = std::pin::pin!(drive(running, &options));
        loop {
            tokio::select! {
                biased;
                Some(event) = rx.recv() => yield Ok(event),
                outcome = &mut driver => {
                    // `drive` awaits the collectors before returning, so everything the process
                    // wrote is already queued.
                    while let Ok(event) = rx.try_recv() {
                        yield Ok(event);
                    }
                    yield outcome.map(SandboxEvent::Result);
                    return;
                }
            }
        }
    })
}

/// A spawned process, or the stand-in result for a program that is not installed.
enum Launched {
    Running(Running),
    Missing(ExecutionResult),
}

struct Running {
    child: Child,
    pid: Option<u32>,
    stdin: Option<(Option<ChildStdin>, String)>,
    stdout: Collected,
    stderr: Collected,
}

fn launch(
    spawn: Spawn,
    options: &mut ExecuteOptions,
    chunks: Option<UnboundedSender<SandboxEvent>>,
) -> SandboxResult<Launched> {
    if options
        .cancel
        .as_ref()
        .is_some_and(|cancel| cancel.is_cancelled())
    {
        return Err(SandboxError::Aborted {
            stdout: String::new(),
            stderr: String::new(),
        });
    }

    // Through `setsid`, a missing program would only show as the child's exit 127.
    let path = spawn.env.as_ref().and_then(|env| env.get("PATH")).map(OsStr::new);
    if !program_exists(&spawn.program, path, spawn.cwd.as_deref()) {
        return match spawn.enoent_message {
            Some(message) => Ok(Launched::Missing(ExecutionResult {
                stderr: message.to_owned(),
                exit_code: NOT_FOUND_EXIT_CODE,
                ..ExecutionResult::default()
            })),
            None => Err(SandboxError::Other(format!(
                "{}: {}",
                spawn.program.to_string_lossy(),
                std::io::Error::from(std::io::ErrorKind::NotFound)
            ))),
        };
    }
    // Own session and group, so the whole tree (not just the program) can be signalled.
    let mut cmd = session_command(&spawn.program);
    cmd.args(&spawn.args)
        .stdin(if options.stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(cwd) = &spawn.cwd {
        cmd.current_dir(cwd);
    }
    if let Some(env) = &spawn.env {
        cmd.envs(env);
    }

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(error) => {
            return match (spawn.enoent_message, error.kind()) {
                (Some(message), std::io::ErrorKind::NotFound) => Ok(Launched::Missing(ExecutionResult {
                    stderr: message.to_owned(),
                    exit_code: NOT_FOUND_EXIT_CODE,
                    ..ExecutionResult::default()
                })),
                _ => Err(SandboxError::Other(error.to_string())),
            };
        }
    };

    let pid = child.id();
    let stdin = options.stdin.take().map(|stdin| (child.stdin.take(), stdin));
    let stdout = collect(child.stdout.take(), StreamType::Stdout, chunks.clone());
    let stderr = collect(child.stderr.take(), StreamType::Stderr, chunks);
    Ok(Launched::Running(Running {
        child,
        pid,
        stdin,
        stdout,
        stderr,
    }))
}

async fn drive(running: Running, options: &ExecuteOptions) -> SandboxResult<ExecutionResult> {
    let Running {
        mut child,
        pid,
        stdin,
        stdout,
        stderr,
    } = running;
    // Kills the group on every exit path, including when this future is dropped.
    let _group = GroupKill(pid);

    // Raced below, never awaited to completion on its own: a child that fills its stdout pipe
    // while we are still writing stdin would otherwise deadlock with no timeout armed.
    let feed_stdin = async {
        if let Some((Some(mut pipe), stdin)) = stdin {
            let _ = pipe.write_all(stdin.as_bytes()).await;
            let _ = pipe.shutdown().await;
        }
        std::future::pending::<()>().await
    };

    let cancelled = async {
        match &options.cancel {
            Some(token) => token.cancelled().await,
            None => std::future::pending().await,
        }
    };
    let waited = async {
        match options.timeout {
            Some(timeout) => tokio::time::timeout(timeout, child.wait()).await.map_err(|_| ()),
            None => Ok(child.wait().await),
        }
    };

    tokio::select! {
        biased;
        () = feed_stdin => unreachable!("stdin feed never completes"),
        () = cancelled => {
            kill_group(pid);
            let _ = child.kill().await;
            Err(SandboxError::Aborted {
                stdout: stdout.read(DRAIN_GRACE).await,
                stderr: stderr.read(DRAIN_GRACE).await,
            })
        }
        status = waited => match status {
            Ok(status) => {
                let status = status.map_err(|e| SandboxError::Other(e.to_string()))?;
                kill_group(pid);
                Ok(ExecutionResult {
                    stdout: stdout.read(DRAIN_GRACE).await,
                    stderr: stderr.read(DRAIN_GRACE).await,
                    exit_code: exit_code_of(&status),
                    output_files: Vec::new(),
                })
            }
            Err(()) => {
                kill_group(pid);
                let _ = child.kill().await;
                Err(SandboxError::Timeout {
                    seconds: options.timeout.unwrap_or_default().as_secs_f64(),
                    stdout: stdout.read(DRAIN_GRACE).await,
                    stderr: stderr.read(DRAIN_GRACE).await,
                })
            }
        },
    }
}

#[cfg(unix)]
fn exit_code_of(status: &std::process::ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    status
        .code()
        .or_else(|| status.signal().map(|signal| 128 + signal))
        .unwrap_or(-1)
}

#[cfg(not(unix))]
fn exit_code_of(status: &std::process::ExitStatus) -> i32 {
    status.code().unwrap_or(-1)
}

struct Collected {
    buffer: Arc<Mutex<Vec<u8>>>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl Collected {
    /// Wait up to `grace` for EOF, then take whatever has been buffered so far.
    async fn read(mut self, grace: Duration) -> String {
        if let Some(mut task) = self.task.take() {
            if tokio::time::timeout(grace, &mut task).await.is_err() {
                task.abort();
            }
        }
        let bytes = self.buffer.lock().expect("output buffer poisoned").clone();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl Drop for Collected {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

fn collect<R>(
    reader: Option<R>,
    stream_type: StreamType,
    chunks: Option<UnboundedSender<SandboxEvent>>,
) -> Collected
where
    R: AsyncReadExt + Unpin + Send + 'static,
{
    let buffer = Arc::new(Mutex::new(Vec::new()));
    let task = reader.map(|mut reader| {
        let buffer = Arc::clone(&buffer);
        tokio::spawn(async move {
            let mut chunk = [0u8; 8192];
            let mut pending = Vec::new();
            while let Ok(read) = reader.read(&mut chunk).await {
                if read == 0 {
                    break;
                }
                buffer
                    .lock()
                    .expect("output buffer poisoned")
                    .extend_from_slice(&chunk[..read]);
                if let Some(chunks) = &chunks {
                    pending.extend_from_slice(&chunk[..read]);
                    let data = take_decodable(&mut pending);
                    if !data.is_empty()
                        && chunks
                            .send(SandboxEvent::Chunk(StreamChunk { data, stream_type }))
                            .is_err()
                    {
                        break;
                    }
                }
            }
            if let Some(chunks) = &chunks {
                if !pending.is_empty() {
                    let data = String::from_utf8_lossy(&pending).into_owned();
                    let _ = chunks.send(SandboxEvent::Chunk(StreamChunk { data, stream_type }));
                }
            }
        })
    });
    Collected { buffer, task }
}

/// Take the longest decodable prefix of `pending`, leaving a trailing partial character behind so
/// a multibyte sequence split across two pipe reads is not streamed as two invalid halves.
fn take_decodable(pending: &mut Vec<u8>) -> String {
    let valid = std::str::from_utf8(pending).map_or_else(|error| error.valid_up_to(), |_| pending.len());
    // A truncated character is at most 3 bytes; a longer tail is genuinely invalid, so decode it
    // now rather than stalling the stream until EOF.
    let split = if pending.len() - valid > 3 {
        pending.len()
    } else {
        valid
    };
    let rest = pending.split_off(split);
    String::from_utf8_lossy(&std::mem::replace(pending, rest)).into_owned()
}

/// SIGKILLs the process group of `pid` when dropped (timeout, error, or a cancelled call).
struct GroupKill(Option<u32>);

impl Drop for GroupKill {
    fn drop(&mut self) {
        kill_group(self.0);
    }
}

/// A command that runs `program` in a new session, like [`detach_from_terminal`], but without
/// forking this process: on Linux it execs through util-linux's (or busybox's) `setsid`, so std can
/// still spawn it with `posix_spawn`.
///
/// A `pre_exec` hook (what [`detach_from_terminal`] installs) makes std `fork()` the whole
/// process, which copies its page tables while holding its memory map lock: from a process with
/// 2 GB resident that took ~160 ms per spawn and stalled every thread meanwhile, against ~2 ms for
/// this command. `setsid` is spawned by absolute path, so an environment that sets `PATH` does not
/// force the fork path either; the child is not a group leader, so `setsid` calls `setsid()` and
/// execs `program` in place: the child's pid stays the session's and group's id.
///
/// A missing `program` makes the child exit 127 instead of failing the spawn; check with
/// [`program_exists`] first where that matters. Where no `setsid` binary is found (and off
/// Linux) this falls back to [`detach_from_terminal`].
pub fn session_command(program: impl AsRef<OsStr>) -> Command {
    #[cfg(target_os = "linux")]
    if let Some(setsid) = setsid_binary() {
        let mut command = Command::new(setsid);
        command.arg("--").arg(program);
        return command;
    }
    let mut command = Command::new(program);
    detach_from_terminal(&mut command);
    command
}

/// The `setsid` binary, looked up once.
#[cfg(target_os = "linux")]
fn setsid_binary() -> Option<&'static std::path::Path> {
    static SETSID: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    SETSID
        .get_or_init(|| {
            ["/usr/bin/setsid", "/bin/setsid"]
                .into_iter()
                .map(PathBuf::from)
                .find(|path| is_executable(path))
        })
        .as_deref()
}

/// Whether `program` would be found by `execvp` with `path` as `PATH` (this process's `PATH`
/// when `None`) in the directory `cwd` (this process's when `None`): a name with a `/` is checked
/// as a path, a bare name in each `PATH` directory.
pub fn program_exists(program: &OsStr, path: Option<&OsStr>, cwd: Option<&std::path::Path>) -> bool {
    let name = std::path::Path::new(program);
    if program.as_encoded_bytes().contains(&b'/') {
        return is_executable(&cwd.map_or_else(|| name.to_path_buf(), |cwd| cwd.join(name)));
    }
    let inherited = std::env::var_os("PATH");
    let path = path
        .or(inherited.as_deref())
        .unwrap_or(OsStr::new("/bin:/usr/bin"));
    std::env::split_paths(path).any(|dir| is_executable(&dir.join(name)))
}

fn is_executable(path: &std::path::Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

/// Start `command` in a new session (`setsid`): its own process group, led by the child, and no
/// controlling terminal.
///
/// Redirecting stdio is not enough to keep a child off the terminal: anything it starts can still
/// open `/dev/tty`, which is how `git` and `ssh` prompt for credentials and how some tools print
/// progress. Under a full-screen UI that output corrupts the screen and the prompt hangs the
/// call; in a session of its own, opening `/dev/tty` fails, so such a program errors out at once.
/// Signal the whole tree with `killpg(child pid)`. Do not also set `process_group`: a group
/// leader cannot start a session. A no-op off Unix.
///
/// This installs a `pre_exec` hook, which forces std to `fork()` the whole process; prefer
/// [`session_command`], which does not.
pub fn detach_from_terminal(command: &mut Command) -> &mut Command {
    #[cfg(unix)]
    {
        use process_wrap::tokio::{CommandWrap, CommandWrapper, ProcessSession};
        // `pre_spawn` only installs the `setsid` hook on `command`; the wrapper is not used.
        let unused = CommandWrap::from(Command::new("sh"));
        let _ = ProcessSession.pre_spawn(command, &unused);
    }
    command
}

/// SIGKILL the whole process group led by `pid`; children of the command die with it.
#[cfg(unix)]
fn kill_group(pid: Option<u32>) {
    if let Some(pid) = pid {
        signal_session(pid, nix::sys::signal::SIGKILL);
    }
}

/// Send `signal` to the session (process group) led by `leader`, a child started by
/// [`session_command`]; whether it reached any process.
///
/// Right after the spawn the child may not have called `setsid()` yet, so there is no group to
/// signal: it is then still in this process's group and is signalled alone (nothing it starts
/// exists yet), and the group once more in case it formed meanwhile. A `leader` that is neither a
/// group leader nor in this process's group (a reused pid) is left alone.
#[cfg(unix)]
pub fn signal_session(leader: u32, signal: nix::sys::signal::Signal) -> bool {
    use nix::sys::signal::{kill, killpg};
    use nix::unistd::{getpgid, getpgrp, Pid};
    let Some(pid) = i32::try_from(leader)
        .ok()
        .filter(|pid| *pid > 0)
        .map(Pid::from_raw)
    else {
        return false;
    };
    if killpg(pid, signal).is_ok() {
        return true;
    }
    if getpgid(Some(pid)) != Ok(getpgrp()) {
        return false;
    }
    let alone = kill(pid, signal).is_ok();
    killpg(pid, signal).is_ok() || alone
}

#[cfg(not(unix))]
fn kill_group(_pid: Option<u32>) {}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;

    fn sh(script: &str) -> Spawn {
        Spawn::new("sh", ["-c", script])
    }

    #[tokio::test]
    async fn yields_stream_chunks_then_execution_result() {
        let events: Vec<_> = stream(sh("echo out; echo err >&2"), ExecuteOptions::default())
            .collect()
            .await;
        let mut chunks = Vec::new();
        let mut result = None;
        for event in events {
            match event.unwrap() {
                SandboxEvent::Chunk(chunk) => chunks.push(chunk),
                SandboxEvent::Result(terminal) => result = Some(terminal),
            }
        }
        assert!(!chunks.is_empty(), "expected at least one chunk");
        assert!(chunks
            .iter()
            .any(|c| c.stream_type == StreamType::Stdout && c.data.contains("out")));
        assert!(chunks
            .iter()
            .any(|c| c.stream_type == StreamType::Stderr && c.data.contains("err")));
        let result = result.expect("terminal result");
        assert_eq!(result.stdout, "out\n");
        assert_eq!(result.stderr, "err\n");
        assert_eq!(result.exit_code, 0);
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn runs_in_a_session_of_its_own_without_a_terminal() {
        let script = "read -r pid comm state ppid group session rest < /proc/$$/stat
            echo \"$pid $group $session\"
            if (: > /dev/tty) 2>/dev/null; then echo tty; else echo no tty; fi";
        let result = run(sh(script), ExecuteOptions::default()).await.unwrap();
        let mut lines = result.stdout.lines();
        let ids: Vec<&str> = lines.next().unwrap().split(' ').collect();
        assert_eq!(ids, [ids[0]; 3], "pid, group and session: {ids:?}");
        assert_ne!(ids[0], std::process::id().to_string());
        assert_eq!(lines.next(), Some("no tty"));
    }

    /// A program named by absolute path keeps `posix_spawn` even when the command's environment
    /// sets `PATH` (std forks for a bare name then), and the session holds with a `PATH` override.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn a_session_command_needs_no_fork() {
        let command = session_command("sh");
        let program = std::path::Path::new(command.as_std().get_program());
        assert!(program.is_absolute(), "{}", program.display());
        let env = HashMap::from([("PATH".to_owned(), "/usr/bin:/bin".to_owned())]);
        let script =
            "read -r pid comm state ppid group session rest < /proc/$$/stat; echo \"$pid $group $session\"";
        let result = run(sh(script).env(Some(env)), ExecuteOptions::default())
            .await
            .unwrap();
        let ids: Vec<&str> = result.stdout.trim().split(' ').collect();
        assert_eq!(ids, [ids[0]; 3], "pid, group and session: {ids:?}");
    }

    #[test]
    fn program_exists_searches_path_and_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let tool = dir.path().join("tool");
        std::fs::write(&tool, "#!/bin/sh\n").unwrap();
        let path = Some(dir.path().as_os_str());
        assert!(!program_exists(OsStr::new("tool"), path, None), "not executable");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tool, std::fs::Permissions::from_mode(0o755)).unwrap();
            assert!(program_exists(OsStr::new("tool"), path, None));
            assert!(!program_exists(
                OsStr::new("tool"),
                Some(OsStr::new("/nonexistent")),
                None
            ));
            assert!(program_exists(OsStr::new("./tool"), None, Some(dir.path())));
            assert!(program_exists(tool.as_os_str(), Some(OsStr::new("")), None));
        }
    }

    /// A session signalled right after the spawn, before the child may have called `setsid()`,
    /// still dies with everything it started.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_session_signalled_at_once_dies() {
        for _ in 0..20 {
            let mut child = session_command("sh")
                .args(["-c", "sleep 30 & wait"])
                .spawn()
                .unwrap();
            assert!(signal_session(child.id().unwrap(), nix::sys::signal::SIGKILL));
            let status = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
            assert!(status.unwrap().unwrap().code().is_none(), "killed by the signal");
        }
    }

    #[tokio::test]
    async fn a_missing_program_fails_by_name() {
        let error = run(
            Spawn::new("strands_nonexistent_binary_xyz", [""; 0]),
            ExecuteOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("strands_nonexistent_binary_xyz"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn returns_enoent_message_when_the_program_does_not_exist() {
        let spawn =
            Spawn::new("strands_nonexistent_binary_xyz", ["--version"]).enoent_message("docker is missing");
        let result = run(spawn, ExecuteOptions::default()).await.unwrap();
        assert_eq!(result.exit_code, 127);
        assert_eq!(result.stderr, "docker is missing");
        assert_eq!(result.stdout, "");
    }

    #[tokio::test]
    async fn errors_when_the_program_does_not_exist_and_no_enoent_message_is_set() {
        let error = run(
            Spawn::new("strands_nonexistent_binary_xyz", [""; 0]),
            ExecuteOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, SandboxError::Other(_)), "{error}");
    }

    #[tokio::test]
    async fn maps_signal_termination_to_128_plus_signal_number() {
        let result = run(sh("kill -9 $$"), ExecuteOptions::default()).await.unwrap();
        assert_eq!(result.exit_code, 137);
    }

    #[tokio::test]
    async fn decodes_multibyte_utf8_sequences_split_across_chunk_boundaries() {
        // `é` is 0xC3 0xA9; the two halves land in separate pipe reads.
        let events: Vec<_> = stream(
            sh("printf 'caf\\303'; sleep 0.1; printf '\\251'"),
            ExecuteOptions::default(),
        )
        .collect()
        .await;
        let mut streamed = String::new();
        let mut result = None;
        for event in events {
            match event.unwrap() {
                SandboxEvent::Chunk(chunk) => streamed.push_str(&chunk.data),
                SandboxEvent::Result(terminal) => result = Some(terminal),
            }
        }
        assert_eq!(streamed, "café");
        assert_eq!(result.unwrap().stdout, "café");
    }

    #[test]
    fn keeps_a_partial_character_pending_and_flushes_a_genuinely_invalid_tail() {
        let mut pending = b"caf\xc3".to_vec();
        assert_eq!(take_decodable(&mut pending), "caf");
        assert_eq!(pending, b"\xc3");

        let mut invalid = b"ok\xff\xff\xff\xff".to_vec();
        assert_eq!(take_decodable(&mut invalid), "ok\u{fffd}\u{fffd}\u{fffd}\u{fffd}");
        assert!(invalid.is_empty());
    }

    #[tokio::test]
    async fn streaming_reports_a_timeout_with_the_output_captured_before_the_kill() {
        let events: Vec<_> = stream(
            sh("echo partial; sleep 5"),
            ExecuteOptions::with_timeout(Duration::from_millis(300)),
        )
        .collect()
        .await;
        let error = events
            .into_iter()
            .filter_map(Result::err)
            .next()
            .expect("timeout error");
        match error {
            SandboxError::Timeout { stdout, .. } => assert_eq!(stdout, "partial\n"),
            other => panic!("expected a timeout, got {other}"),
        }
    }

    #[tokio::test]
    async fn streaming_aborts_when_the_token_is_already_cancelled() {
        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel();
        let events: Vec<_> = stream(
            sh("echo hi"),
            ExecuteOptions {
                cancel: Some(cancel),
                ..ExecuteOptions::default()
            },
        )
        .collect()
        .await;
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], Err(SandboxError::Aborted { .. })));
    }
}
