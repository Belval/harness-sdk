//! An agent's working directory, which the agent itself can move: a `shell` call that is only
//! `cd DIR` changes it for every later `shell`, `run_background` and `spawn_agent` call, and the
//! swarm keeps it across restarts.
//!
//! [`Workdir`] is the directory, shared between the runtime (which moves it and saves it) and the
//! agent the factory built ([`WorkdirSandbox`] runs commands in it, [`environment`] describes it).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use strands_agents::plugins::vended::ContextInjector;
use strands_agents::sandbox::{
    ExecuteOptions, ExecutionResult, FileInfo, LocalSandbox, Sandbox, SandboxEvent, SandboxResult,
};
use strands_agents::types::{SystemContentBlock, SystemPrompt};

/// An agent's current working directory; clones share it.
#[derive(Clone, Debug, Default)]
pub struct Workdir(Arc<Mutex<PathBuf>>);

impl Workdir {
    /// A working directory starting at `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self(Arc::new(Mutex::new(dir.into())))
    }

    /// The directory now.
    pub fn get(&self) -> PathBuf {
        self.0.lock().clone()
    }

    /// Move to `dir`.
    pub fn set(&self, dir: PathBuf) {
        *self.0.lock() = dir;
    }
}

/// The directory a `shell` command moves to when it is only `cd DIR` (one word, optionally
/// quoted; `~` is `$HOME`), resolved against `from`; `None` for any other command, and for a
/// target that is not a directory, so the shell reports that the usual way.
pub fn lone_cd(command: &str, from: &Path) -> Option<PathBuf> {
    let arg = command.trim().strip_prefix("cd")?;
    if !arg.starts_with(char::is_whitespace) {
        return None;
    }
    let arg = arg.trim();
    let arg = match arg.chars().next() {
        Some(q @ ('\'' | '"')) => {
            let inner = arg[1..].strip_suffix(q)?;
            match inner.contains(['\'', '"', '\\', '$', '`']) {
                true => return None,
                false => inner,
            }
        }
        _ if arg.contains(|c: char| c.is_whitespace() || "'\"\\$`;&|<>(){}*?[]#".contains(c)) => return None,
        _ => arg,
    };
    if arg.is_empty() || arg == "-" {
        return None;
    }
    let path = match arg.strip_prefix('~') {
        Some(rest) if rest.is_empty() || rest.starts_with('/') => {
            PathBuf::from(std::env::var_os("HOME")?).join(rest.trim_start_matches('/'))
        }
        Some(_) => return None,
        None => from.join(arg),
    };
    let dir = crate::runtime::lexically_normal(&path);
    dir.is_dir().then_some(dir)
}

/// The text a `cd` call returns.
pub fn moved_text(dir: &Path) -> String {
    format!(
        "Your working directory is now `{}`: later `shell`, `run_background` and `spawn_agent` calls \
         start there, also after a restart.",
        dir.display()
    )
}

/// `prompt` with the swarm prompt's working directory sentence naming `to` instead of `from`.
pub fn moved_prompt(prompt: SystemPrompt, from: &Path, to: &Path) -> SystemPrompt {
    let sentence = |dir: &Path| format!("Your working directory is `{}`", dir.display());
    let (from, to) = (sentence(from), sentence(to));
    match prompt {
        SystemPrompt::Text(text) => SystemPrompt::Text(text.replacen(&from, &to, 1)),
        SystemPrompt::Blocks(blocks) => SystemPrompt::Blocks(
            blocks
                .into_iter()
                .map(|block| match block {
                    SystemContentBlock::Text(text) => SystemContentBlock::Text(text.replacen(&from, &to, 1)),
                    other => other,
                })
                .collect(),
        ),
    }
}

/// A [`LocalSandbox`] that runs commands in the [`Workdir`] as it is at each call, and resolves
/// relative file paths against it.
#[derive(Debug)]
pub struct WorkdirSandbox {
    dir: Workdir,
    inner: LocalSandbox,
}

impl WorkdirSandbox {
    /// Commands run in `dir` with `env` added to the swarm's environment.
    pub fn new(dir: Workdir, env: std::collections::HashMap<String, String>) -> Self {
        let inner = match env.is_empty() {
            true => LocalSandbox::new(),
            false => LocalSandbox::new().with_env(env),
        };
        Self { dir, inner }
    }

    fn options(&self, mut options: ExecuteOptions) -> ExecuteOptions {
        options.cwd.get_or_insert_with(|| self.dir.get());
        options
    }

    fn path(&self, path: &str) -> String {
        self.dir.get().join(path).to_string_lossy().into_owned()
    }
}

#[async_trait]
impl Sandbox for WorkdirSandbox {
    fn kind(&self) -> &str {
        self.inner.kind()
    }

    async fn execute(&self, command: &str, options: ExecuteOptions) -> SandboxResult<ExecutionResult> {
        self.inner.execute(command, self.options(options)).await
    }

    fn execute_streaming<'a>(
        &'a self,
        command: &'a str,
        options: ExecuteOptions,
    ) -> BoxStream<'a, SandboxResult<SandboxEvent>> {
        self.inner.execute_streaming(command, self.options(options))
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        self.inner.read_file(&self.path(path)).await
    }

    async fn write_file(&self, path: &str, content: &[u8]) -> SandboxResult<()> {
        self.inner.write_file(&self.path(path), content).await
    }

    async fn remove_file(&self, path: &str) -> SandboxResult<()> {
        self.inner.remove_file(&self.path(path)).await
    }

    async fn list_files(&self, path: &str) -> SandboxResult<Vec<FileInfo>> {
        self.inner.list_files(&self.path(path)).await
    }
}

/// How long a gathered environment block is reused while the directory stays put (it carries
/// the date and the project docs).
const ENVIRONMENT_TTL: Duration = Duration::from_secs(600);

/// The harness's environment block (platform, date, working directory, nearby `AGENTS.md` and
/// READMEs) for the [`Workdir`] as it is at each model call, so it follows a `cd`.
#[derive(Clone)]
pub struct Environment {
    dir: Workdir,
    memo: Arc<Mutex<Option<(PathBuf, Instant, String)>>>,
}

impl Environment {
    /// The block for `dir`.
    pub fn new(dir: Workdir) -> Self {
        Self {
            dir,
            memo: Arc::default(),
        }
    }

    /// The block now: gathered again once the directory moved or the last one is old.
    pub async fn text(&self) -> String {
        let dir = self.dir.get();
        if let Some((at, when, text)) = self.memo.lock().as_ref() {
            if *at == dir && when.elapsed() < ENVIRONMENT_TTL {
                return text.clone();
            }
        }
        let text = strands_harness::plugins::environment::render_environment(Some(dir.clone())).await;
        *self.memo.lock() = Some((dir, Instant::now(), text.clone()));
        text
    }

    /// The injector that adds [`Environment::text`] to every model call.
    pub fn injector(self) -> ContextInjector {
        ContextInjector::new(move |_ctx| {
            let this = self.clone();
            Box::pin(async move { Ok(Some(this.text().await)) })
        })
        .with_name("swarm:environment")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_lone_cd_to_a_directory_moves() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("a b/c")).unwrap();
        std::fs::write(root.join("file"), "").unwrap();
        let sub = root.join("a b");
        assert_eq!(lone_cd("cd 'a b'", root), Some(sub.clone()));
        assert_eq!(lone_cd("  cd \"a b\"/c\n", root), None, "quotes inside a word");
        assert_eq!(lone_cd("cd c", &sub), Some(sub.join("c")));
        assert_eq!(lone_cd("cd ..", &sub), Some(root.to_path_buf()));
        let abs = format!("cd '{}/'", sub.join("c").display());
        assert_eq!(lone_cd(&abs, Path::new("/")), Some(sub.join("c")));
        for other in [
            "cd",
            "cd -",
            "cd file",
            "cd missing",
            "cd a b",
            "cd 'a b' && ls",
            "cd 'a b'; ls",
            "cd $HOME",
            "cdx",
            "ls",
            "echo cd 'a b'",
        ] {
            assert_eq!(lone_cd(other, root), None, "{other}");
        }
        let home = std::env::var("HOME").unwrap();
        assert_eq!(lone_cd("cd ~", root), Some(PathBuf::from(&home)));
    }

    #[tokio::test]
    async fn the_sandbox_follows_the_workdir() {
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let dir = Workdir::new(a.path());
        let env = [("SWARM_X".to_owned(), "1".to_owned())].into();
        let sandbox = WorkdirSandbox::new(dir.clone(), env);
        let command = "pwd -P; echo $SWARM_X";
        let pwd = || sandbox.execute(command, ExecuteOptions::default());
        let canonical = |path: &Path| std::fs::canonicalize(path).unwrap().display().to_string();
        assert_eq!(
            pwd().await.unwrap().stdout,
            format!("{}\n1\n", canonical(a.path()))
        );
        dir.set(b.path().to_path_buf());
        assert_eq!(
            pwd().await.unwrap().stdout,
            format!("{}\n1\n", canonical(b.path()))
        );
        sandbox.write_text("f", "x").await.unwrap();
        assert!(b.path().join("f").exists());
        assert_eq!(sandbox.read_text("f").await.unwrap(), "x");
    }

    #[tokio::test]
    async fn the_environment_block_follows_the_workdir() {
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        std::fs::write(b.path().join("AGENTS.md"), "B's rules").unwrap();
        let dir = Workdir::new(a.path());
        let environment = Environment::new(dir.clone());
        let first = environment.text().await;
        assert!(first.contains(&a.path().display().to_string()), "{first}");
        assert!(!first.contains("B's rules"));
        dir.set(b.path().to_path_buf());
        let moved = environment.text().await;
        assert!(moved.contains(&b.path().display().to_string()), "{moved}");
        assert!(moved.contains("B's rules"), "{moved}");
    }
}
