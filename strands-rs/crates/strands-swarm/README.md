# strands-swarm

A model-driven agent swarm on `strands-harness`. No planner, no scheduler: every agent is a harness
agent with nine extra tools, and decides for itself whether and how to split the work.

| Tool | What it does |
| --- | --- |
| `spawn_agent` | Start an agent on a task (optionally in its own git worktree, with a model or a goal). Non-blocking. |
| `send_message` | Message any agent (by id or name), or `parent`. Busy agents see it after their current tool call; idle ones wake up. |
| `wait` | Block until messages arrive (optionally from specific agents). |
| `list_agents` | The agent tree with statuses, directories, branches and running background jobs. |
| `cancel_agent` | Stop an agent and everything it spawned, background jobs included. |
| `reload_swarm` | Restart the whole swarm into a newly built binary; every agent resumes. |
| `schedule` | Post a message to an agent's mailbox (default: your own) later and/or every N minutes; list or cancel schedules. |
| `run_background` | Start a shell command as a background job and return its id at once; the agent gets a message with the exit code and the output's last 40 lines when it ends. |
| `jobs` | List background jobs (status, log path), or kill one with everything it started. |
| `land` | Only with `--allow-land`: rebase the caller's (or a descendant's) branch onto the root's checked-out branch, run the checks, and fast-forward it (see below). |
| `find_tools` | Only with MCP servers: search the shared MCP tools by keyword and enable some for the calling agent. |
| `skills` | Only with skills: load a skill's full `SKILL.md` instructions, its location and its bundled files. |
| `remember` / `forget` | Only with memory: add a dated entry to the user's memory file, or drop the lines matching a text. |

Background jobs run with `sh -c` in the agent's directory and environment (its private
`CARGO_TARGET_DIR` included), in their own process group, so neither a reload nor a Ctrl-C stops
them. Each one is a few files in `<run>/jobs` (a temp directory without `--run`): `<id>.json`,
the output in `<id>.log` and, from a small `sh` wrapper, the exit code in `<id>.exit`. The swarm
polls them and mails the owner when one ends, also for jobs started before a restart; a job
whose process vanished without an exit code (a reboot) is reported as such. A killed job is not
reported, and cancelling an agent kills the jobs of its whole subtree.

`land` (opt-in with `--allow-land`) lets a worker merge its own work instead of handing it to an
integrator: `{branch?, into?, check?: [commands]}` lands `branch` on `into`. `branch` defaults to
the caller's own and may only be that or the branch of an agent below it, so nobody lands another
agent's unfinished work. `into` defaults to the branch the caller's root agent has checked out;
any other target must match a `--land-into GLOB` the user gave (repeatable, `*` and `?`), else it
is refused, so `main` or someone else's branch stays out of reach. As a background job it
creates a scratch worktree at the branch's tip, rebases it onto `into`, runs each check with `sh`
from the same subdirectory as the caller's working directory (default checks: the `--gate CMD`
commands, repeatable) with a private `CARGO_TARGET_DIR` (`<shared>-land/<run>`, or under the jobs
directory), and only then fast-forwards `into`: `git merge --ff-only` in the worktree that has it
checked out, else a compare-and-swap `update-ref`. It refuses when that worktree has uncommitted
changes to tracked files, and gives up if `into` moved meanwhile. A conflict (with the conflicted
files), a failing check (with its last 30 lines) or a refusal changes nothing; the scratch worktree
is always removed, and the caller's own branch is never modified (`git rebase <into>` afterwards
drops the landed commits from it). Programmatically: `Swarm::land`.

Every agent has a short, unique name (`lazy-mcp (a2)`): the one its spawner gave, else one derived
from its task; ids and names work interchangeably. Each agent's final reply per turn goes to its parent. A `goal` (TUI `/goal`, CLI `--goal`, or on
`spawn_agent`) puts a judge in the loop that sends the agent back to work until the goal is met.
The judge runs after every turn the agent ends and sees the goal, the task, and the latest ~40k
characters of the conversation. Side calls (the judge, context summarization) run on the agent's
model or `--side-model ID`, count toward the agent's tokens and the budget, and queue under
`--max-concurrent-calls`, like the agent's own calls.

## Run

```sh
cargo build --release -p strands-swarm
strands-swarm [--run NAME] [--goal GOAL] [--model ID] [--side-model ID]
              [--mcp-config mcp.json]... [--mcp-deny REGEX]... [--mcp-eager] [--list-tools]
              [--skills-dir DIR]... [--no-default-skills] [--list-skills]
              [--max-agents N] [--max-concurrent-calls N] [--compact-at TOKENS] [--memory FILE | --no-memory]
              [--notify] [--status-base BRANCH] [--allow-land] [--land-into GLOB]... [--gate CMD]... [--headless] [--journal | --gc] [TASK]
```

Memory: one Markdown file shared by every agent and every run, for things you ask the swarm to
remember (default `~/.config/strands-swarm/memory.md`, `--memory FILE` elsewhere, `--no-memory` to
turn it off). Each entry is a dated bullet; edit the file by hand as you like. Its newest ~4k tokens
are appended to every agent's system prompt under "# Memory (things the user asked to remember)",
re-read from disk at the start of each turn, so a change reaches running agents on their next turn
(not mid-turn). Agents add entries with `remember` and drop them with `forget`; you use
`/remember <text>`, `/forget <text>` and `/remember` (where the file is). Writes are serialized
across the swarm and replace the file atomically.

`--max-agents` (16) caps live agents (spawns beyond it are `queued` and start, oldest first, as agents
are cancelled; a queued agent can already be messaged, and cancelling it drops it); `--max-concurrent-calls` (8, `0` = unlimited) caps model calls in
flight across the whole swarm, so a large swarm queues ("queued for a model slot…" in the log)
instead of tripping provider throttling. `--budget-tokens N` (e.g. `5M`) caps the tokens (input +
output) the whole swarm may use: once spent, model calls fail with a "budget exhausted" error that
travels up to you, and `/budget N` in the TUI raises it (`/budget off` lifts it, `/budget` shows usage),
which tells every stopped agent to continue. The budget is saved with the run.

`--max-agents N` caps live agents (spawns beyond it queue). `/max-agents N` changes the cap for the
running swarm, starts queued agents at once and is kept across restarts ahead of the flag;
`/max-agents default` returns to the flag's value.

`schedule {message, to?, every?: "30m", at?: "in 2h" | "2026-11-17T09:00Z", id?}` posts
`[schedule sN] message` (kind `schedule`) to the receiver at the given times; `{cancel: id}` removes
one and `{}` lists them (`list_agents` shows each agent's, the TUI `/schedules` all). Repeats are at
least a minute apart. An agent that polls something schedules itself and ends its turn instead of
holding a long `wait`; while it is idle with a schedule pending it does not count toward
`--max-agents`. Schedules are saved with the run and survive reloads; a schedule whose time passed
while the swarm was down fires once on resume, then keeps its interval. Schedules of a cancelled
agent (as receiver or creator) are dropped.

TUI: ↑/↓ select an agent, type to message it, `/to <agent> <message>` (by name or id, without selecting it), `/new <task>`, `/goal <goal>`, `/stop`, `/cancel`,
`/gc`, `/schedules`, `/status`, Ctrl-C quits.
`/status` answers "what is everyone doing?" without a model call: a popup with each live agent's
state and current activity, its todo list, its branch's commits not yet on the base
(`git log <base>..<branch>`; base is the branch checked out where the swarm started, or
`--status-base BRANCH`), running background jobs and schedules. It refreshes while open (git runs in
the background, cached for 5 s); ↑/↓ scroll, any other key closes it. Programmatically:
`Swarm::status`. `--notify` rings the terminal bell when an agent messages you (tmux and most
terminals turn it into a window flag or a desktop notification).
`/stop` (or Esc twice within a second on an empty input) interrupts the selected agent's turn without
cancelling it, like Esc in Claude Code: an open tool call is closed as cancelled, the agent goes idle
with its conversation intact, its parent gets an `interrupted` message, its children keep running,
and a goal agent waits for its next message instead of resuming (restarts included). The next
message continues the same conversation. Programmatically: `Swarm::interrupt`. `/cancel` stops the
agent and its whole subtree for good.
Typing `/` opens a popup of the commands with one-line help, filtered as you type; ↑/↓ and
Tab or Enter pick one; F1 lists the keys. Each agent is one line in the tree: status, name, model,
input/output tokens and how close its last model call came to the compaction point, e.g.
`● parser opus-5-5 1.2M/34.0k 34% a3`: at 100% its older turns are summarized before the next call
(the log notes it at 90%). The compaction point is `--compact-at` (default `200k`) or 85% of the
model's context window, whichever is smaller, so agents on 1M-token models do not carry ~850k
tokens per call; it follows the window on a `/model` switch.
Reasoning (thinking text and signatures) is dropped from turns before the current one, which
models that keep prior thinking would otherwise bill as input on every call; the turn in progress
keeps it, as Anthropic requires within a tool-use loop.
`shell` and `read` results over 2,000 characters are cut to their head and tail (~800 characters)
once six newer ones exist, in batches to spare the prompt cache; `retrieve_context` with the
reference at the end returns the full output.
The header totals the swarm's tokens as `in 84.8M (97% cached) · out 291k`: input counts cache reads
and writes (the budget does too), and the share read from the prompt cache is shown beside it, as
in each agent's log title.
`/model <id>` switches the selected agent's model from its next model call on (`/model default`
returns it to `--model`; `/model` alone shows the current one). The id is resolved like `--model`
(`provider/name` or a Bedrock id, same effort), an invalid one is rejected without touching the
agent, and the choice is saved with the run. Programmatically: `Swarm::set_model`.
Images show inline in the log: a tool result with an image (e.g. `read` on a png) and a message
sent with `/image <path> [text]` (the selected agent gets the image as a multimodal input; the
mailbox and journal keep only its path) appear as an `[image: path WxH]` line with a thumbnail of
at most 12 rows below it, drawn with kitty, iTerm2 or sixel graphics when the terminal answers the
query, and with unicode halfblocks otherwise and always inside tmux. Only the newest 4 thumbnails
stay decoded; older images keep just their placeholder, which is also what F2 and mouse copies
show.
The input box edits like a shell line: ←/→, Home/End (Ctrl-A/Ctrl-E), Ctrl-←/→ by word, Delete,
Ctrl-W/Ctrl-U (Alt-Backspace too), Alt-Enter (or Shift-Enter where the terminal reports it) for a
new line, and ↑/↓ through sent messages while it holds text (Ctrl-P/Ctrl-N from an empty one; on an
empty input ↑/↓ select agents). Alt-←/→ narrows or widens the agent tree; narrow it past its
minimum to hide it.
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
shared by all agents and start lazily: with `--run`, each server's tool list (name, description,
schema) is cached in `<run dir>/mcp-tools.json` under a hash of its config, so a start or reload with an
unchanged config connects no server (about 6.5 s → 0.03 s for 5 servers / 274 tools). A server
connects in the background when an agent enables one of its tools, or on the first call (60 s
timeout), and reconnects on the next call if it died. A new or changed server is connected at
startup to list its tools; `--mcp-eager` and `--list-tools` connect every server up front. Under the TUI each server's stderr goes to
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

The new binary is installed as soon as it passes; `reload_swarm` restarts into it right away with
`when: "now"`, or by default (`"idle"`) once no agent has a tool call in flight, so a running test
or build is not cut short. `wait` calls do not count, and after 10 minutes
(`SwarmConfig::reload_idle_timeout`) it restarts anyway. Meanwhile the TUI header shows
`reload pending (waiting for N tool calls)`; a newer reload replaces the pending one.

Each binary embeds the git commit it was built from (`build.rs`) and saves it with the run. When
a run resumes under another commit, the TUI shows a notice with the subjects of the commits in
between (`git log old..new` in the checkout the binary was built from) and `events.jsonl` gets a
`changes` entry with the full list.

```sh
swarmctl start NAME ./target/release/strands-swarm --cwd REPO --goal "..."   # start or resume
swarmctl attach NAME        # open the TUI (detach: C-b d)
swarmctl stop NAME          # graceful stop; state is kept
swarmctl resume NAME        # start again with the saved binary and arguments
swarmctl boot NAME          # resume after every reboot (crontab @reboot)
swarmctl status
```
