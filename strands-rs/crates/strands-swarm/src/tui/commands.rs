//! Slash commands and parsing of the input line into a [`Command`].

/// A slash command, for the popup that opens when the input starts with `/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlashCommand {
    /// The command, e.g. `/new`.
    pub name: &'static str,
    /// Argument synopsis; empty when it takes none.
    pub args: &'static str,
    /// One-line help.
    pub help: &'static str,
}

/// Every slash command, in popup order.
pub const SLASH_COMMANDS: &[SlashCommand] = &[
    SlashCommand {
        name: "/new",
        args: "<task>",
        help: "start a new root agent",
    },
    SlashCommand {
        name: "/goal",
        args: "<goal>",
        help: "start a root agent that works until the goal is met",
    },
    SlashCommand {
        name: "/to",
        args: "<agent> <message>",
        help: "message an agent by name or id without selecting it",
    },
    SlashCommand {
        name: "/model",
        args: "[<id>|default]",
        help: "show or switch the selected agent's model",
    },
    SlashCommand {
        name: "/max-agents",
        args: "[N|default]",
        help: "show or change how many agents may run at once",
    },
    SlashCommand {
        name: "/max-calls",
        args: "[N|default]",
        help: "show or change how many model calls may run at once (0: no limit)",
    },
    SlashCommand {
        name: "/max-load",
        args: "[N|off|default]",
        help: "show or change the host load above which new model calls and jobs wait",
    },
    SlashCommand {
        name: "/min-free-mem",
        args: "[SIZE|off|default]",
        help: "show or change the available memory below which new model calls and jobs wait",
    },
    SlashCommand {
        name: "/priority",
        args: "[AGENTS|off|default]",
        help: "show or set who skips host holds and caps besides root agents",
    },
    SlashCommand {
        name: "/budget",
        args: "[N|off]",
        help: "show, set or remove the swarm-wide token budget",
    },
    SlashCommand {
        name: "/remember",
        args: "[<text>]",
        help: "add to the memory every agent sees, or show where it is",
    },
    SlashCommand {
        name: "/forget",
        args: "<text>",
        help: "drop memory lines containing the text",
    },
    SlashCommand {
        name: "/stop",
        args: "",
        help: "interrupt the selected agent's turn; it stays alive (esc esc)",
    },
    SlashCommand {
        name: "/image",
        args: "<path> [text]",
        help: "message the selected agent with an image attached",
    },
    SlashCommand {
        name: "/cancel",
        args: "",
        help: "cancel the selected agent and its children",
    },
    SlashCommand {
        name: "/schedules",
        args: "",
        help: "list scheduled messages",
    },
    SlashCommand {
        name: "/status",
        args: "",
        help: "every agent's state, todos, unmerged commits and jobs (no model call)",
    },
    SlashCommand {
        name: "/cost",
        args: "",
        help: "estimated cost per agent: tokens, cache use, $ from list prices (no model call)",
    },
    SlashCommand {
        name: "/gc",
        args: "",
        help: "reclaim the disk of cancelled worktree agents",
    },
    SlashCommand {
        name: "/quit",
        args: "",
        help: "quit (the run is saved)",
    },
];

/// What the input line asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Start a new root agent with this task.
    NewRoot(String),
    /// Start a new root agent that keeps working until this goal is met.
    Goal(String),
    /// Send a message to the selected agent.
    Message(String),
    /// Send a message to the agent with this id or name, whatever is selected.
    To {
        /// Agent id or name.
        who: String,
        /// Message text (empty when the user left it out).
        text: String,
    },
    /// Cancel the selected agent.
    Cancel,
    /// Interrupt the selected agent's current turn; it stays alive and idle.
    Stop,
    /// Show the selected agent's model.
    ShowModel,
    /// Switch the selected agent's model; `None` returns it to the default.
    SetModel(Option<String>),
    /// Show the token budget, or set it (`off` lifts it).
    Budget(Option<String>),
    /// `/max-agents [N|default]`: show or change the live-agent limit.
    MaxAgents(Option<String>),
    /// `/max-calls [N|default]`: show or change the model-call limit (`0`: none).
    MaxCalls(Option<String>),
    /// `/max-load [N|off|default]`: show or change the host load limit.
    MaxLoad(Option<String>),
    /// `/min-free-mem [SIZE|off|default]`: show or change the free-memory floor.
    MinFreeMem(Option<String>),
    /// `/priority [AGENTS|off|default]`: show or replace the priority lane.
    Priority(Option<String>),
    /// Add to the user's memory, or show it (`None`).
    Remember(Option<String>),
    /// Drop memory lines containing the text.
    Forget(String),
    /// Send the selected agent a message with an image attached.
    Image {
        /// Image file.
        path: std::path::PathBuf,
        /// Message text (may be empty).
        text: String,
    },
    /// Reclaim the disk of cancelled worktree agents.
    Gc,
    /// List scheduled messages.
    Schedules,
    /// Open the status popup.
    Status,
    /// Open the cost popup.
    Cost,
    /// Quit.
    Quit,
}

/// A command that shows a limit (`None`) or changes it.
type LimitCommand = fn(Option<String>) -> Command;

/// Parse a submitted input line. `has_agents` decides whether plain text starts a root agent.
pub fn parse_command(line: &str, has_agents: bool) -> Option<Command> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    if let Some(goal) = line.strip_prefix("/goal ") {
        return Some(Command::Goal(goal.trim().to_owned()));
    }
    if let Some(task) = line.strip_prefix("/new ") {
        return Some(Command::NewRoot(task.trim().to_owned()));
    }
    if let Some(text) = line.strip_prefix("/remember ") {
        return Some(Command::Remember(Some(text.trim().to_owned())));
    }
    if let Some(text) = line.strip_prefix("/forget ") {
        return Some(Command::Forget(text.trim().to_owned()));
    }
    if let Some(rest) = line.strip_prefix("/image ") {
        let rest = rest.trim();
        let (path, text) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
        return Some(Command::Image {
            path: path.into(),
            text: text.trim().to_owned(),
        });
    }
    if let Some(rest) = line.strip_prefix("/to ") {
        let rest = rest.trim();
        let (who, text) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
        return Some(Command::To {
            who: who.to_owned(),
            text: text.trim().to_owned(),
        });
    }
    // The limits: alone they show the limit, with an argument they set it.
    let limits: [(&str, LimitCommand); 6] = [
        ("/max-agents", Command::MaxAgents),
        ("/max-calls", Command::MaxCalls),
        ("/max-load", Command::MaxLoad),
        ("/min-free-mem", Command::MinFreeMem),
        ("/priority", Command::Priority),
        ("/budget", Command::Budget),
    ];
    for (name, command) in limits {
        match line.strip_prefix(name) {
            Some("") => return Some(command(None)),
            Some(arg) if arg.starts_with(' ') => return Some(command(Some(arg.trim().to_owned()))),
            _ => {}
        }
    }
    if let Some(model) = line.strip_prefix("/model ") {
        return Some(Command::SetModel(match model.trim() {
            "default" => None,
            id => Some(id.to_owned()),
        }));
    }
    match line {
        "/cancel" => return Some(Command::Cancel),
        "/stop" => return Some(Command::Stop),
        "/model" => return Some(Command::ShowModel),
        "/remember" => return Some(Command::Remember(None)),
        "/gc" => return Some(Command::Gc),
        "/schedules" => return Some(Command::Schedules),
        "/status" => return Some(Command::Status),
        "/cost" => return Some(Command::Cost),
        "/quit" | "/q" => return Some(Command::Quit),
        _ => {}
    }
    match has_agents {
        true => Some(Command::Message(line.to_owned())),
        false => Some(Command::NewRoot(line.to_owned())),
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_image_command() {
        assert_eq!(
            parse_command("/image shots/a.png what is wrong here?", true),
            Some(Command::Image {
                path: "shots/a.png".into(),
                text: "what is wrong here?".into()
            })
        );
        assert_eq!(
            parse_command("/image a.png", true),
            Some(Command::Image {
                path: "a.png".into(),
                text: String::new()
            })
        );
    }

    #[test]
    fn parses_input_lines() {
        assert_eq!(
            parse_command("fix it", false),
            Some(Command::NewRoot("fix it".into()))
        );
        assert_eq!(
            parse_command("fix it", true),
            Some(Command::Message("fix it".into()))
        );
        assert_eq!(
            parse_command("/new other", true),
            Some(Command::NewRoot("other".into()))
        );
        assert_eq!(parse_command("/cancel", true), Some(Command::Cancel));
        assert_eq!(parse_command("/stop", true), Some(Command::Stop));
        assert_eq!(parse_command("/budget", true), Some(Command::Budget(None)));
        assert_eq!(parse_command("/remember", true), Some(Command::Remember(None)));
        assert_eq!(
            parse_command("/remember use bun ", false),
            Some(Command::Remember(Some("use bun".into())))
        );
        assert_eq!(
            parse_command("/forget bun", true),
            Some(Command::Forget("bun".into()))
        );
        assert_eq!(
            parse_command("/budget 2M", false),
            Some(Command::Budget(Some("2M".into())))
        );
        assert_eq!(
            parse_command("/goal ship it", true),
            Some(Command::Goal("ship it".into()))
        );
        assert_eq!(parse_command("/model", true), Some(Command::ShowModel));
        assert_eq!(
            parse_command("/to a10  monitor my Slack ", false),
            Some(Command::To {
                who: "a10".into(),
                text: "monitor my Slack".into()
            })
        );
        assert_eq!(
            parse_command("/to a10", true),
            Some(Command::To {
                who: "a10".into(),
                text: String::new()
            })
        );
        assert_eq!(parse_command("/max-load", true), Some(Command::MaxLoad(None)));
        assert_eq!(
            parse_command("/min-free-mem", true),
            Some(Command::MinFreeMem(None))
        );
        assert_eq!(
            parse_command("/min-free-mem 4G", true),
            Some(Command::MinFreeMem(Some("4G".into())))
        );
        assert_eq!(parse_command("/priority", true), Some(Command::Priority(None)));
        assert_eq!(
            parse_command("/priority slack-bridge a7", true),
            Some(Command::Priority(Some("slack-bridge a7".into())))
        );
        assert_eq!(
            parse_command("/max-load 96", true),
            Some(Command::MaxLoad(Some("96".into())))
        );
        assert_eq!(parse_command("/gc", false), Some(Command::Gc));
        assert_eq!(parse_command("/schedules", false), Some(Command::Schedules));
        assert_eq!(parse_command("/status", false), Some(Command::Status));
        assert_eq!(parse_command("/cost", false), Some(Command::Cost));
        assert_eq!(
            parse_command("/model  anthropic/claude-x ", true),
            Some(Command::SetModel(Some("anthropic/claude-x".into())))
        );
        assert_eq!(
            parse_command("/model default", true),
            Some(Command::SetModel(None))
        );
        assert_eq!(parse_command("  ", true), None);
    }
}
