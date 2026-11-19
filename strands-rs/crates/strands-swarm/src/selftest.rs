//! Reload self-test: before a run restarts into a new binary, that binary resumes a throwaway
//! copy of the run with a stub model and proves it can restore every agent, deliver mail, get
//! replies, drive the TUI and stay free of deadlocks. A binary that fails is never installed.
//!
//! Nothing outside the copy may be touched: its saved paths (worktrees, private target dirs) are
//! the live run's. So the copy's agents lose their branch and env and work in the copy itself,
//! disk reclamation is off, state, sessions and journal live in the copy, the stub model answers
//! every call with a plain `ok` and never calls a tool, and agents get no sandbox.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use strands_agents::models::{ContentBlockDelta, Model, ModelStreamEvent, StreamOptions};
use strands_agents::sandbox::session_command;
use strands_agents::types::{Message, Role, StopReason};
use strands_agents::{Agent, Error, Result};

use crate::factory::agent_session_manager;
use crate::runtime::{AgentFactory, AgentSpec, AgentStatus, Swarm, SwarmConfig, SwarmEvent, USER};
use crate::state::{self, SwarmState, SESSIONS_DIR};
use crate::tui::{view, App};
use crate::watchdog;

/// How long every live agent has to reply.
pub const REPLY_TIMEOUT: Duration = Duration::from_secs(20);

/// Top-level entries of a run directory the copy leaves out: binaries, logs, the audit trail,
/// background jobs (their processes belong to the live run) and the 1 GB disk reserve.
const SKIPPED: &[&str] = &[
    "bin",
    "jobs",
    crate::reserve::RESERVE_FILE,
    "stderr.log",
    "supervisor.log",
    "journal.jsonl",
    "journal.jsonl.1",
    "events.jsonl",
    "events.jsonl.1",
];

const PROBE: &str = "[Self-test of a new strands-swarm binary: reply with a short acknowledgement.]";

/// A model that answers `ok` without network access.
struct StubModel;

impl Model for StubModel {
    fn model_id(&self) -> Option<String> {
        Some("self-test-stub".into())
    }

    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        _options: &'a StreamOptions,
    ) -> BoxStream<'a, Result<ModelStreamEvent>> {
        // Refuse what the providers refuse, so a binary that sends a broken conversation fails
        // here instead of on every turn of the live run.
        if let Some(error) = unpaired_tool_call(messages) {
            return Box::pin(futures::stream::iter([Err(Error::model(error))]));
        }
        let events = vec![
            ModelStreamEvent::MessageStart {
                role: Role::Assistant,
            },
            ModelStreamEvent::ContentBlockStart { start: None },
            ModelStreamEvent::ContentBlockDelta {
                delta: ContentBlockDelta::Text { text: "ok".into() },
            },
            ModelStreamEvent::ContentBlockStop {},
            ModelStreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
                additional_model_response_fields: None,
            },
        ];
        Box::pin(futures::stream::iter(events.into_iter().map(Ok)))
    }
}

/// Where `messages` breaks the providers' pairing rule, worded like Bedrock: each tool call is
/// answered in the next message, and each result answers a call of the message before it.
fn unpaired_tool_call(messages: &[Message]) -> Option<String> {
    for (at, message) in messages.iter().enumerate() {
        let calls = |m: &Message| {
            m.tool_uses()
                .map(|c| c.tool_use_id.clone())
                .collect::<BTreeSet<_>>()
        };
        let results = |m: &Message| {
            m.tool_results()
                .map(|r| r.tool_use_id.clone())
                .collect::<BTreeSet<_>>()
        };
        if message.role == Role::Assistant {
            let answered = messages
                .get(at + 1)
                .filter(|next| next.role == Role::User)
                .map(results);
            let missing: Vec<String> = calls(message)
                .into_iter()
                .filter(|id| !answered.as_ref().is_some_and(|answered| answered.contains(id)))
                .collect();
            if !missing.is_empty() {
                return Some(format!(
                    "messages.{at}: `tool_use` ids were found without `tool_result` blocks immediately after: {}",
                    missing.join(", ")
                ));
            }
        } else {
            let before = at
                .checked_sub(1)
                .map(|b| &messages[b])
                .filter(|b| b.role == Role::Assistant);
            let called = before.map(calls).unwrap_or_default();
            if let Some(stray) = results(message).into_iter().find(|id| !called.contains(id)) {
                return Some(format!(
                    "messages.{at}: unexpected `tool_use_id` found in `tool_result` blocks: {stray}"
                ));
            }
        }
    }
    None
}

/// Agents on the stub model, restoring their conversations from `sessions`.
fn stub_factory(sessions: PathBuf) -> AgentFactory {
    Arc::new(move |spec: AgentSpec| {
        let sessions = sessions.clone();
        Box::pin(async move {
            Agent::builder()
                .id(spec.id.clone())
                .model(StubModel)
                .system_prompt(spec.instructions)
                .tools(spec.tools)
                .session_manager(agent_session_manager(&sessions, &spec.id)?)
                .no_sandbox()
                .build()
        })
    })
}

/// Resume the run copy in `dir` on a stub model, message every live agent and check that each
/// replies within [`REPLY_TIMEOUT`], that the TUI renders the result, and that no lock deadlocked.
/// Returns a one-line summary. `dir` is modified (it is a copy); a run that was never saved
/// resumes as an empty swarm.
pub async fn run(dir: &Path) -> Result<String> {
    let saved = state::load(dir).await?.map(|saved| detach(saved, dir));
    let mut config = SwarmConfig::new(stub_factory(dir.join(SESSIONS_DIR)), dir);
    if let Some(saved) = &saved {
        config.run_id = saved.run_id.clone();
    }
    config.state_dir = Some(dir.to_path_buf());
    config.max_agents = usize::MAX;
    config.reclaim_disk = false;
    // Cancelled agents take no part in the probe: their histories would only fill the channel.
    config.cancelled_history = false;
    config.cargo_target_dir = None;
    let run_id = config.run_id.clone();
    let (swarm, mut events) = match saved {
        Some(saved) => Swarm::resume(config, saved).await?,
        None => Swarm::new(config),
    };
    // The stub spends no tokens, but a spent budget would refuse its calls.
    swarm.set_budget(None);

    let mut pending: BTreeSet<String> = swarm
        .agents()
        .into_iter()
        .filter(|info| info.status.is_live())
        .map(|info| info.id)
        .collect();
    let total = pending.len();
    // Agents with a saved conversation must get it back (a resumed agent publishes it as History).
    let mut unrestored: BTreeSet<String> = pending
        .iter()
        .filter(|id| dir.join(SESSIONS_DIR).join(id).is_dir())
        .cloned()
        .collect();
    let mut failure = None;
    for id in &pending {
        if let Err(error) = swarm.send(USER, id, PROBE) {
            failure = Some(format!("cannot message {}: {error}", swarm.label(id)));
        }
    }

    let mut app = App::default();
    let deadline = tokio::time::sleep(REPLY_TIMEOUT);
    tokio::pin!(deadline);
    while !pending.is_empty() && failure.is_none() {
        let event = tokio::select! {
            event = events.recv() => event,
            _ = &mut deadline => break,
        };
        let Some(event) = event else { break };
        match &event {
            SwarmEvent::Mail { mail, .. } if mail.kind == "reply" => {
                pending.remove(&mail.from);
            }
            SwarmEvent::Mail { mail, .. } if mail.kind == "error" => {
                failure = Some(format!("{} reported: {}", mail.from_label, mail.text));
            }
            // Restoring an agent whose turn had failed shows that old failure; the probe it then
            // answers (or not) is what tests this binary.
            SwarmEvent::Status {
                id,
                status: AgentStatus::Errored(error),
            } if error != crate::runtime::RESTORED_ERROR => {
                failure = Some(format!("{} errored: {error}", swarm.label(id)))
            }
            SwarmEvent::History { id, .. } => {
                unrestored.remove(id);
            }
            _ => {}
        }
        app.apply(event);
    }
    while let Ok(event) = events.try_recv() {
        app.apply(event);
    }
    draw_every_agent(&mut app, &run_id);
    let deadlock = watchdog::check();
    swarm.shutdown().await;

    if let Some(report) = deadlock {
        return Err(Error::other(format!("deadlock: {report}")));
    }
    if let Some(failure) = failure {
        return Err(Error::other(failure));
    }
    if !unrestored.is_empty() {
        let lost: Vec<String> = unrestored.iter().map(|id| swarm.label(id)).collect();
        return Err(Error::other(format!(
            "saved conversations were not restored for {}",
            lost.join(", ")
        )));
    }
    if !pending.is_empty() {
        let missing: Vec<String> = pending.iter().map(|id| swarm.label(id)).collect();
        return Err(Error::other(format!(
            "no reply within {}s from {}",
            REPLY_TIMEOUT.as_secs(),
            missing.join(", ")
        )));
    }
    Ok(format!("self-test passed: {total} live agents replied"))
}

/// Cut a copied run loose from the live run's directories: no agent keeps a branch (which marks
/// a worktree it owns), a private env, or a working directory outside `dir`.
fn detach(mut saved: SwarmState, dir: &Path) -> SwarmState {
    saved.cwd = dir.to_path_buf();
    // The stub costs nothing, and a spent budget would refuse the probe it must answer.
    saved.budget_tokens = None;
    for agent in &mut saved.agents {
        agent.branch = None;
        agent.env.clear();
        agent.cwd = dir.to_path_buf();
        agent.budget_tokens = None;
    }
    saved
}

/// Render the TUI with each agent selected in turn. A rendering bug panics, which fails the
/// self-test process.
fn draw_every_agent(app: &mut App, run_id: &str) {
    let Ok(mut terminal) = Terminal::new(TestBackend::new(120, 40));
    for tick in 0..app.tree().len().max(1) {
        let Ok(_) = terminal.draw(|frame| view::draw(frame, app, run_id, tick));
        app.select_by(1);
    }
}

/// Copy run directory `from` to `to`, leaving out [`SKIPPED`] entries. A run that has not
/// written its directory yet copies as an empty one.
pub fn copy_run(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    if !from.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        if SKIPPED.iter().any(|skip| entry.file_name() == *skip) {
            continue;
        }
        copy_tree(&entry.path(), &to.join(entry.file_name()))?;
    }
    Ok(())
}

/// Files that vanish mid-copy (temporaries renamed away by a concurrent atomic write) are skipped.
fn copy_tree(from: &Path, to: &Path) -> std::io::Result<()> {
    let gone = |error: &std::io::Error| error.kind() == std::io::ErrorKind::NotFound;
    let kind = match std::fs::symlink_metadata(from) {
        Ok(metadata) => metadata.file_type(),
        Err(error) if gone(&error) => return Ok(()),
        Err(error) => return Err(error),
    };
    if kind.is_dir() {
        std::fs::create_dir_all(to)?;
        for entry in std::fs::read_dir(from)? {
            let entry = entry?;
            copy_tree(&entry.path(), &to.join(entry.file_name()))?;
        }
    } else if kind.is_file() {
        match std::fs::copy(from, to) {
            Err(error) if gone(&error) => {}
            other => {
                other?;
            }
        }
    }
    Ok(())
}

/// Run `binary --self-test` on a temporary copy of the run in `state_dir`, and remove the copy.
/// Fails with the tail of the binary's stderr unless it exits successfully within `timeout`.
pub async fn check_binary(binary: &Path, state_dir: &Path, timeout: Duration) -> Result<()> {
    let copy = std::env::temp_dir().join(format!(
        "strands-swarm-self-test-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let (from, to) = (state_dir.to_path_buf(), copy.clone());
    let copied = tokio::task::spawn_blocking(move || copy_run(&from, &to))
        .await
        .map_err(|e| Error::other(format!("copying the run for the self-test failed: {e}")))?;
    let outcome = match copied {
        Ok(()) => run_self_test(binary, &copy, timeout).await,
        Err(e) => Err(Error::other(format!(
            "copying the run for the self-test failed: {e}"
        ))),
    };
    let _ = tokio::fs::remove_dir_all(&copy).await;
    outcome
}

async fn run_self_test(binary: &Path, copy: &Path, timeout: Duration) -> Result<()> {
    let child = session_command(binary)
        .arg("--self-test")
        .arg(copy)
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true)
        .output();
    let output = tokio::time::timeout(timeout, child)
        .await
        .map_err(|_| {
            Error::other(format!(
                "{} --self-test timed out after {}s; the swarm keeps its current binary.",
                binary.display(),
                timeout.as_secs()
            ))
        })?
        .map_err(|e| Error::other(format!("Cannot run {}: {e}", binary.display())))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let tail: Vec<&str> = stderr.lines().rev().take(20).collect();
    let tail: Vec<&str> = tail.into_iter().rev().collect();
    Err(Error::other(format!(
        "{} --self-test failed ({}); the swarm keeps its current binary.\n{}",
        binary.display(),
        output.status,
        tail.join("\n")
    )))
}

#[cfg(test)]
mod pairing_tests {
    use super::*;
    use futures::StreamExt;
    use serde_json::json;
    use strands_agents::types::{ContentBlock, ToolResultBlock, ToolUseBlock};

    fn call(id: &str) -> Message {
        Message::new(
            Role::Assistant,
            vec![ContentBlock::ToolUse(ToolUseBlock::new("shell", id, json!({})))],
        )
    }

    async fn first(messages: &[Message]) -> Result<ModelStreamEvent> {
        let options = StreamOptions::default();
        let event = StubModel.stream(messages, &options).next().await.unwrap();
        event
    }

    #[tokio::test]
    async fn the_stub_refuses_a_call_without_its_result_like_the_providers() {
        let broken = [Message::user("go"), call("t1"), Message::user("mail")];
        let error = first(&broken).await.unwrap_err().to_string();
        assert!(
            error.contains("messages.1: `tool_use` ids were found without `tool_result`"),
            "{error}"
        );
        assert!(error.contains("t1"));

        let stray = [
            Message::user("go"),
            Message::assistant("hi"),
            Message::new(
                Role::User,
                vec![ContentBlock::ToolResult(ToolResultBlock::error("t9", "x"))],
            ),
        ];
        assert!(first(&stray).await.unwrap_err().to_string().contains("t9"));

        let mut repaired = broken.to_vec();
        crate::resume::pair_tool_calls(&mut repaired);
        assert!(first(&repaired).await.is_ok(), "a repaired conversation passes");
    }
}
