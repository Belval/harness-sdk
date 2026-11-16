# strands-swarm

A model-driven agent swarm on `strands-harness`. No planner, no scheduler: every agent is a harness
agent with six extra tools, and decides for itself whether and how to split the work.

| Tool | What it does |
| --- | --- |
| `spawn_agent` | Start an agent on a task (optionally in its own git worktree, with a model or a goal). Non-blocking. |
| `send_message` | Message any agent, or `parent`. Busy agents see it after their current tool call; idle ones wake up. |
| `wait` | Block until messages arrive (optionally from specific agents). |
| `list_agents` | The agent tree with statuses, directories and branches. |
| `cancel_agent` | Stop an agent and everything it spawned. |
| `reload_swarm` | Restart the whole swarm into a newly built binary; every agent resumes. |

Each agent's final reply per turn goes to its parent. A `goal` (TUI `/goal`, CLI `--goal`, or on
`spawn_agent`) puts a judge in the loop that sends the agent back to work until the goal is met.

## Run

```sh
cargo build --release -p strands-swarm
strands-swarm [--run NAME] [--goal GOAL] [--model ID] [--mcp-config mcp.json]... [--headless] [TASK]
```

TUI: ↑/↓ select an agent, type to message it, `/new <task>`, `/goal <goal>`, `/cancel`, Ctrl-C quits.

`--run NAME` makes the run persistent under `~/.local/state/strands-swarm/NAME`: the agent tree,
unread messages and every conversation (saved after each message). Starting again with the same
name resumes it. `--mcp-config` accepts any `mcp.json` (or `~/.claude.json`); the servers are
connected once and shared by all agents.

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
