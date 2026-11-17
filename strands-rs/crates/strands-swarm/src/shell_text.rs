//! `shell` results as plain text.
//!
//! The SDK's shell tool returns a JSON block (`{"output": …, "error": …, "exit_code": …}`), which
//! reaches the model JSON-encoded: every quote of a file it `cat`s arrives as `\"` and every line
//! break as `\n`. Models copy that into `edit`'s `old_str`, which then matches nothing (in a test
//! run, four of seven agents lost a turn to it). [`ShellText`] turns the block into the text a
//! terminal would show, with stderr and a failing exit code marked and without the escape
//! sequences and `\r` overwrites a terminal would interpret ([`crate::text::plain`]). A timed-out
//! call's partial output, which the SDK appends to its error as a JSON line, gets the same text.

use async_trait::async_trait;
use strands_agents::hooks::AfterToolCallEvent;
use strands_agents::tools::vended::ShellOutput;
use strands_agents::types::ToolResultContent;
use strands_agents::{Agent, Plugin, Result};

/// Rewrites each `shell` result's JSON block into [`render`]'s text.
pub struct ShellText;

#[async_trait]
impl Plugin for ShellText {
    fn name(&self) -> &str {
        "swarm:shell-text"
    }

    async fn init_agent(&self, agent: &Agent) -> Result<()> {
        agent.hooks().add_sync_callback(|event: &mut AfterToolCallEvent| {
            if event.tool_use.name == "shell" {
                for item in &mut event.result.content {
                    let text = match item {
                        ToolResultContent::Json(json) => serde_json::from_value::<ShellOutput>(json.clone())
                            .ok()
                            .map(|output| render(&output)),
                        ToolResultContent::Text(text) => render_timeout(text),
                        _ => None,
                    };
                    if let Some(text) = text {
                        *item = ToolResultContent::Text(text);
                    }
                }
            }
            Ok(())
        });
        Ok(())
    }
}

/// The SDK's timeout error (`Error: Execution timed out after S seconds`, then the partial
/// output as a JSON line) as [`render`]'s text ending in `[timed out after Ss]`; `None` for any
/// other text. A cancelled call has no partial output to show: the SDK drops it (`ShellTool`
/// returns `Error::Cancelled` and `SandboxError::Aborted` carries no output).
fn render_timeout(text: &str) -> Option<String> {
    let (first, json) = text.split_once('\n')?;
    let seconds = first
        .strip_prefix("Error: Execution timed out after ")?
        .strip_suffix(" seconds")?;
    let partial = serde_json::from_str::<ShellOutput>(json).ok()?;
    Some(render_ending(&partial, &format!("[timed out after {seconds}s]")))
}

/// stdout as is, then `[stderr]` and stderr, then `[exit code N]` unless it is 0;
/// `(no output)` when all three are empty.
pub fn render(output: &ShellOutput) -> String {
    let end = match output.exit_code {
        0 => String::new(),
        code => format!("[exit code {code}]"),
    };
    render_ending(output, &end)
}

/// stdout, then `[stderr]` and stderr, then `end` unless it is empty.
fn render_ending(output: &ShellOutput, end: &str) -> String {
    let mut text = crate::text::plain(&output.output)
        .trim_end_matches('\n')
        .to_owned();
    let section = |text: &mut String, part: &str| {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(part);
    };
    let error = crate::text::plain(&output.error);
    let error = error.trim_end_matches('\n');
    if !error.is_empty() {
        section(&mut text, &format!("[stderr]\n{error}"));
    }
    if !end.is_empty() {
        section(&mut text, end);
    }
    if text.is_empty() {
        text = "(no output)".to_owned();
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn output(output: &str, error: &str, exit_code: i32) -> ShellOutput {
        serde_json::from_value(json!({ "output": output, "error": error, "exit_code": exit_code })).unwrap()
    }

    #[test]
    fn stdout_is_shown_verbatim_without_escapes() {
        let text = render(&output("x = \"a\"\n\ty\n", "", 0));
        assert_eq!(text, "x = \"a\"\n\ty");
    }

    #[test]
    fn stderr_and_a_failing_exit_code_are_marked() {
        assert_eq!(
            render(&output("out\n", "boom\n", 2)),
            "out\n[stderr]\nboom\n[exit code 2]"
        );
        assert_eq!(render(&output("", "", 1)), "[exit code 1]");
        assert_eq!(render(&output("", "warn", 0)), "[stderr]\nwarn");
    }

    #[test]
    fn colours_and_progress_overwrites_are_dropped() {
        assert_eq!(
            render(&output(
                "\x1b[32m✓\x1b[39m 3 passed\n",
                " 1/3\r 3/3\n\x1b[31mFAIL\x1b[0m\n",
                1
            )),
            "✓ 3 passed\n[stderr]\n 3/3\nFAIL\n[exit code 1]"
        );
    }

    #[test]
    fn nothing_at_all_says_so() {
        assert_eq!(render(&output("", "", 0)), "(no output)");
    }

    #[test]
    fn a_timeout_ends_with_how_long_it_ran_instead_of_its_exit_code() {
        let partial = serde_json::to_string(&output("half\n", "", 124)).unwrap();
        let text = format!("Error: Execution timed out after 120 seconds\n{partial}");
        assert_eq!(render_timeout(&text).unwrap(), "half\n[timed out after 120s]");
        let silent = serde_json::to_string(&output("", "", 124)).unwrap();
        let text = format!("Error: Execution timed out after 2.5 seconds\n{silent}");
        assert_eq!(render_timeout(&text).unwrap(), "[timed out after 2.5s]");
        assert_eq!(render_timeout("Error: Execution aborted"), None);
        assert_eq!(render_timeout("Error: boom\n{\"output\":\"\"}"), None);
    }
}
