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
| `skills` | Only with skills: load a skill's full `SKILL.md` instructions, its location and its bundled files. |
| `remember` / `forget` | Only with memory: add a dated entry to the user's memory file, or drop the lines matching a text. |

Every agent has a short, unique name (`lazy-mcp (a2)`): the one its spawner gave, else one derived
from its task; ids and names work interchangeably. Each agent's final reply per turn goes to its parent. A `goal` (TUI `/goal`, CLI `--goal`, or on
`spawn_agent`) puts a judge in the loop that sends the agent back to work until the goal is met.
The judge runs after every turn the agent ends, on the agent's model or `--judge-model ID`, and sees
the goal, the task, and the latest ~40k characters of the conversation. Its calls count toward the
agent's tokens and the budget and queue under `--max-concurrent-calls`, like the agent's own.

## Run

```sh
cargo build --release -p strands-swarm
strands-swarm [--run NAME] [--goal GOAL] [--model ID] [--judge-model ID]
              [--mcp-config mcp.json]... [--mcp-deny REGEX]... [--mcp-eager] [--list-tools]
              [--skills-dir DIR]... [--no-default-skills] [--list-skills]
              [--max-agents N] [--max-concurrent-calls N] [--memory FILE | --no-memory]
              [--headless] [--journal | --gc] [TASK]
```

Memory: one Markdown file shared by every agent and every run, for things you ask the swarm to
remember (default `~/.config/strands-swarm/memory.md`, `--memory FILE` elsewhere, `--no-memory` to
turn it off). Each entry is a dated bullet; edit the file by hand as you like. Its newest ~4k tokens
are appended to every agent's system prompt under "# Memory (things the user asked to remember)",
re-read from disk at the start of each turn, so a change reaches running agents on their next turn
(not mid-turn). Agents add entries with `remember` and drop them with `forget`; you use
`/remember <text>`, `/forget <text>` and `/remember` (where the file is). Writes are serialized
across the swarm and replace the file atomically.

`--max-agents` (16) caps live agents; `--max-concurrent-calls` (8, `0` = unlimited) caps model calls in
flight across the whole swarm, so a large swarm queues ("queued for a model slot…" in the log)
instead of tripping provider throttling. `--budget-tokens N` (e.g. `5M`) caps the tokens (input +
output) the whole swarm may use: once spent, model calls fail with a "budget exhausted" error that
travels up to you, and `/budget N` in the TUI raises it (`/budget off` lifts it, `/budget` shows usage),
which tells every stopped agent to continue. The budget is saved with the run.

TUI: ↑/↓ select an agent, type to message it, `/new <task>`, `/goal <goal>`, `/cancel`, `/gc`,
Ctrl-C quits.
Typing `/` opens a popup of the commands with one-line help, filtered as you type; ↑/↓ and
Tab or Enter pick one; F1 lists the keys. Each agent is one line in the tree: status, name, model,
input/output tokens and how full its context window was on its last model call, e.g.
`● parser opus-5-5 1.2M/34.0k 34% a3`.
`/model <id>` switches the selected agent's model from its next model call on (`/model default`
returns it to `--model`; `/model` alone shows the current one). The id is resolved like `--model`
(`provider/name` or a Bedrock id, same effort), an invalid one is rejected without touching the
agent, and the choice is saved with the run. Programmatically: `Swarm::set_model`.
Ctrl-←/→ (or Alt-←/→) narrows or widens the agent tree; narrow it past its minimum to hide it.
Cancelled agents are hidden from the tree (the header counts them); F3 shows them.
F2 prints the selected agent's log as plain text into the terminal's own scrollback (native
scrolling, search and copy-paste); any key returns. A resumed run shows each agent's restored
conversation.

Mouse: drag to select text inside the log or the agent tree (the selection never crosses into the
other pane); releasing copies it to the system clipboard with an OSC 52 escape, which works over
ssh too. Under tmux this needs `set -g set-clipboard on`. The wheel scrolls the log, Esc clears the
highlight. For the terminal's own selection, hold Shift while dragging (Option/Alt in iTerm2), or
use F2.

`--run NAME` makes the run persistent under `~/.local/state/strands-swarm/NAME`: the agent tree,
unread messages and every conversation (saved after each message). Starting again with the same
name resumes it. `--mcp-config` accepts any `mcp.json` (or `~/.claude.json`); the servers are
connected once and shared by all agents. Under the TUI each server's stderr goes to
`<run dir>/logs/mcp-<server>.log` (a temp directory without `--run`); `--headless` keeps it on the
terminal. Agents start without MCP tools, so their specs are not sent
on every model call: `find_tools` (whose description lists the servers and their tool counts)
searches the catalog and enables tools by name for the calling agent, from its next model call on.
Enabled tools are kept in the agent's session state and come back on resume. `--mcp-eager` registers
every MCP tool on every agent instead; `--mcp-deny` hides tools by name, and `--list-tools` prints
what is allowed and denied.

A persistent run also keeps an audit trail next to its state: `journal.jsonl` records every message
to and from the user (tasks, messages, replies, errors) and `events.jsonl` compact lifecycle events
(spawns, status changes, tool calls, model switches, resumes, reloads, shutdowns, disk reclaimed). Both are JSON
lines with a UTC `ts`, written by a background thread and rotated to `<file>.1` at 8 MB.
`strands-swarm --run NAME --journal` prints the journal and exits.

A worktree agent owns its worktree at `<repo>.swarm/<run>/<agent>` and, when `CARGO_TARGET_DIR` is
set, a private target dir at `$CARGO_TARGET_DIR-worktrees/<run>/<agent>` (several GB). Once it and
everything it spawned have stopped after a cancel (or on resume, for agents cancelled earlier), the
target dir is deleted and the worktree removed with `git worktree remove` if it has no uncommitted
changes; its branch is never deleted. Nothing outside those two exact paths is touched, and each
removal or kept directory is logged as a `gc` event. `/gc` in the TUI retries (e.g. after you commit
in a kept worktree) and reports the space reclaimed; `strands-swarm --run NAME --gc` does the same
for a saved run and exits.

Skills (`<name>/SKILL.md` with `name` and `description` frontmatter, as in Claude Code and Codex)
are discovered once at startup and offered to every agent with progressive disclosure: the system
prompt lists each skill as `name: description` cut to 200 characters (keeping a later "Use when…"
sentence, which is what picks the skill), and the `skills` tool returns the full instructions, the
skill's location and its bundled files. On a name clash the first source wins:
`--skills-dir` (repeatable), then the project's `.claude/skills`, `.agents/skills` and
`.agent/skills`, then `~/.claude/skills`, then the skills of Claude Code plugins that are installed
(user scope, or project scope for this directory) and not disabled in `enabledPlugins`, then
`~/.codex/skills` and `~/.agents/skills`. `--no-default-skills` keeps only `--skills-dir`;
`--list-skills` prints what agents get and what is shadowed.

## Operate 24/7

`scripts/swarmctl` keeps a run alive in tmux: restarts on exit, restarts into the new binary on
`reload_swarm`, and rolls back if a new binary crashes within 30 s.

Before a reload installs anything, the running swarm has the new binary prove itself:
`NEW --self-test <copy>` resumes a temporary copy of the run directory (state and sessions, no
`bin/` or logs) on a stub model that answers `ok`, messages every live agent, requires a reply from
each and every saved conversation restored within 20 s, renders the TUI into a test backend and
checks the deadlock watchdog. If it fails or takes over 60 s, `reload_swarm` returns the error with
the tail of its stderr and the swarm keeps its binary. The copy's saved paths are the live run's, so
its agents lose their branches, private env and working directories, disk reclamation is off
(`SwarmConfig::reclaim_disk`), the stub never calls tools and agents get no sandbox: the real run,
its worktrees, target dirs and sessions are never touched. The copy is deleted after.

```sh
swarmctl start NAME ./target/release/strands-swarm --cwd REPO --goal "..."   # start or resume
swarmctl attach NAME        # open the TUI (detach: C-b d)
swarmctl stop NAME          # graceful stop; state is kept
swarmctl resume NAME        # start again with the saved binary and arguments
swarmctl boot NAME          # resume after every reboot (crontab @reboot)
swarmctl status
```
