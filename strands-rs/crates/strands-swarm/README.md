# strands-swarm

A model-driven agent swarm on `strands-harness`. No planner, no scheduler: every agent is a harness
agent with six extra tools, and decides for itself whether and how to split the work.

| Tool | What it does |
| --- | --- |
| `spawn_agent` | Start an agent on a task (optionally in its own git worktree, with a model or a goal). Non-blocking. |
| `send_message` | Message any agent (by id or name), or `parent`. Busy agents see it after their current tool call; idle ones wake up. |
| `wait` | Block until messages arrive (optionally from specific agents). |
| `list_agents` | The agent tree with statuses, directories and branches. |
| `cancel_agent` | Stop an agent and everything it spawned. |
| `reload_swarm` | Restart the whole swarm into a newly built binary; every agent resumes. |
| `find_tools` | Only with MCP servers: search the shared MCP tools by keyword and enable some for the calling agent. |
| `skills` | Only with skills: load a skill's full `SKILL.md` instructions (names and descriptions are in the system prompt). |

Every agent has a short, unique name (`lazy-mcp (a2)`): the one its spawner gave, else one derived
from its task; ids and names work interchangeably. Each agent's final reply per turn goes to its parent. A `goal` (TUI `/goal`, CLI `--goal`, or on
`spawn_agent`) puts a judge in the loop that sends the agent back to work until the goal is met.

## Run

```sh
cargo build --release -p strands-swarm
strands-swarm [--run NAME] [--goal GOAL] [--model ID] [--mcp-config mcp.json]... [--mcp-deny REGEX]...
              [--mcp-eager] [--list-tools] [--skills-dir DIR]... [--no-default-skills] [--list-skills]
              [--headless] [TASK]
```

TUI: ↑/↓ select an agent, type to message it, `/new <task>`, `/goal <goal>`, `/cancel`, Ctrl-C quits.
`/model <id>` switches the selected agent's model from its next model call on (`/model default`
returns it to `--model`; `/model` alone shows the current one). The id is resolved like `--model`
(`provider/name` or a Bedrock id, same effort), an invalid one is rejected without touching the
agent, and the choice is saved with the run. Programmatically: `Swarm::set_model`.
Ctrl-←/→ (or Alt-←/→) narrows or widens the agent tree; narrow it past its minimum to hide it.
Cancelled agents are hidden from the tree (the header counts them); F3 shows them.
F2 prints the selected agent's log as plain text into the terminal's own scrollback (native
scrolling, search and copy-paste); any key returns. A resumed run shows each agent's restored
conversation.

`--run NAME` makes the run persistent under `~/.local/state/strands-swarm/NAME`: the agent tree,
unread messages and every conversation (saved after each message). Starting again with the same
name resumes it. `--mcp-config` accepts any `mcp.json` (or `~/.claude.json`); the servers are
connected once and shared by all agents. Agents start without MCP tools, so their specs are not sent
on every model call: `find_tools` (whose description lists the servers and their tool counts)
searches the catalog and enables tools by name for the calling agent, from its next model call on.
Enabled tools are kept in the agent's session state and come back on resume. `--mcp-eager` registers
every MCP tool on every agent instead; `--mcp-deny` hides tools by name, and `--list-tools` prints
what is allowed and denied.

A persistent run also keeps an audit trail next to its state: `journal.jsonl` records every message
to and from the user (tasks, messages, replies, errors) and `events.jsonl` compact lifecycle events
(spawns, status changes, tool calls, model switches, resumes, reloads, shutdowns). Both are JSON
lines with a UTC `ts`, written by a background thread and rotated to `<file>.1` at 8 MB.
`strands-swarm --run NAME --journal` prints the journal and exits.

Skills (`<name>/SKILL.md` with `name` and `description` frontmatter, as in Claude Code and Codex)
are discovered once at startup and offered to every agent with progressive disclosure: the system
prompt lists each skill's name, description (capped at 300 characters) and location, and the
`skills` tool loads the full instructions on demand. On a name clash the first source wins:
`--skills-dir` (repeatable), then the project's `.claude/skills`, `.agents/skills` and
`.agent/skills`, then `~/.claude/skills`, then the skills of Claude Code plugins that are installed
(user scope, or project scope for this directory) and not disabled in `enabledPlugins`, then
`~/.codex/skills` and `~/.agents/skills`. `--no-default-skills` keeps only `--skills-dir`;
`--list-skills` prints what agents get and what is shadowed.

## Operate 24/7

`scripts/swarmctl` keeps a run alive in tmux: restarts on exit, restarts into the new binary on
`reload_swarm`, and rolls back if a new binary crashes within 30 s.

```sh
swarmctl start NAME ./target/release/strands-swarm --cwd REPO --goal "..."   # start or resume
swarmctl attach NAME        # open the TUI (detach: C-b d)
swarmctl stop NAME          # graceful stop; state is kept
swarmctl resume NAME        # start again with the saved binary and arguments
swarmctl boot NAME          # resume after every reboot (crontab @reboot)
swarmctl status
```
