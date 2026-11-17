# strands-swarm

A model-driven agent swarm on `strands-harness`. No planner, no scheduler: every agent is a harness
agent with the swarm tools below, and decides for itself whether and how to split the work.

| Tool | What it does |
| --- | --- |
| `spawn_agent` | Start an agent on a task (optionally in another directory, in its own git worktree, with a model or a goal). Non-blocking. |
| `send_message` | Message any agent (by id or name), or `parent`. Busy agents see it after their current tool call; idle ones wake up. |
| `wait` | Block until messages arrive (optionally from specific agents, or `jobs` for job-end reports, which returns at once when none is pending); a user message always ends it. With `all`, until every agent in `from` has sent something; a timeout (default 10 min, at most an hour) returns what came and who is still missing. Mail stays unread (and saved) until it returns, so a restart meanwhile loses none. |
| `list_agents` | The agent tree: status, directory, branch (ahead/behind the base, clean merge or conflicting files), running jobs, schedules, model and estimated spend. |
| `cancel_agent` | Stop an agent and everything it spawned, background jobs included; reports which worktrees and merged branches were removed and what was kept. |
| `reload_swarm` | With `--run`: restart the whole swarm into a newly built binary; every agent resumes. |
| `schedule` | Post a message to a mailbox (default: your own) later and/or repeatedly (`every: "30m"`, at least a minute); list or cancel schedules. |
| `run_background` | Run a shell command as a background job; its owner gets the exit code and last 40 output lines when it ends. |
| `jobs` | List background jobs (status, log path), or kill one with everything it started. |
| `context` | The agent's context dashboard: size against the compaction point, the biggest items, the stash; offload results or summarize now. |
| `land` | With `--allow-land`: rebase a branch onto the root's branch, run the checks, fast-forward. |
| `find_tools` | With MCP servers: search the shared MCP tools and enable some for the calling agent. |
| `skills` | With skills: load a skill's `SKILL.md` (and its full description), its location and bundled files. |
| `remember` / `forget` | With memory: add a dated entry to the user's memory file, or drop matching lines. |

Agents also get the harness's `shell`, `read`, `write`, `edit`, `web_fetch`, `todo_write` and
`retrieve_context`, but not the SDK's background tasks (`_background_execution` on every tool):
`run_background` covers slow commands. A `shell` result reaches the model as the text a terminal
would show (stdout, then `[stderr]` and `[exit code N]` when there are any), not the SDK's JSON
block, whose escaped quotes models copied into `edit`'s `old_str`. Tool specs and the swarm's
prompt sections are sent on every model call, so `the_fixed_prefix_stays_within_its_budget` pins
their size: 17 kB (about 5k tokens) with 20 skills, the skills list at 2.5 kB of it.

## Run

```sh
cargo build --release -p strands-swarm
strands-swarm [--run NAME] [--cwd DIR] [--goal GOAL] [--model ID] [--side-model ID] [--effort E]
              [--cache-ttl 5m|1h] [--no-keep-warm]
              [--instructions FILE] [--memory FILE | --no-memory]
              [--mcp-config mcp.json]... [--mcp-deny REGEX]... [--mcp-eager] [--list-tools]
              [--skills-dir DIR]... [--no-default-skills] [--list-skills]
              [--max-agents N] [--max-concurrent-calls N] [--budget-tokens N] [--compact-at TOKENS]
              [--allow-land] [--land-into GLOB]... [--gate CMD]...
              [--notify] [--status-base BRANCH] [--prices FILE] [--min-free-disk SIZE] [--headless]
              [--journal | --gc | --cost] [TASK]
```

`--run NAME` makes the run persistent under `$XDG_STATE_HOME/strands-swarm/NAME` (default
`~/.local/state`): the agent tree, unread mail, schedules and every conversation (saved after
each message; see [Persistence](#persistence)). Starting again with the same name resumes it; a
TASK given then is sent to the root agent as a message, unless it is the one the run was last
started with: a supervisor replays its arguments on every restart. `--goal` is never sent, as it
only seeds a new root. With `--run`, `--journal`, `--gc` and
`--cost` print a report on the saved run and exit.

`--headless` prints events to stderr and messages to you to stdout instead of the TUI, and exits
once the root has replied and no agent is busy (a resume without a task: once no agent is busy)
or every agent has stopped, with a non-zero status when that reply is an error. A new run needs a
TASK or `--goal`. Without a terminal (stdout not a tty), the TUI exits with an error that
points to `--headless` before it starts or resumes anything.

## Agents

- **Names.** Every agent has a short, unique name (`lazy-mcp (a2)`): the one its spawner gave, else
  one derived from its task. Ids and names work interchangeably, and so do the `name (id)` labels
  the swarm prints and a name as the spawner wrote it (`Fix-P12` for `fix-p12`).
- **Replies.** Each agent's final reply per turn goes to its parent. Mail that arrives mid-turn is
  appended to the next tool result. A reply that is empty or just `[silent]` (e.g. a scheduled poll
  that found nothing) reaches no agent parent; the user's root always replies. A `send_message` to
  an agent's parent says so in its result, as agents that report mid-turn tend to repeat the report
  as their final reply, which costs the parent a turn.
- **Working directory.** An agent works where its spawner does unless `spawn_agent` gets a `cwd`
  (absolute or relative), e.g. another repository; with `worktree` the worktree is cut from the
  repository there.
- **Goals.** A `goal` (`spawn_agent`, `/goal`, `--goal`) puts a judge in the loop that sends the
  agent back to work until the goal is met. It runs after every turn and sees the goal, the task
  and the latest ~40k characters of the conversation.
- **Side calls.** The judge and context summarization run on the agent's model or `--side-model ID`.
  They count toward the agent's tokens and the budget, and queue like its own calls. They skip
  the prompt cache: their prompts are never sent twice, so a cache write would cost 25% extra.
- **Context.**
  - An agent's older turns are summarized once a call's input reaches the compaction point:
    `--compact-at` (default `200k`) or 85% of the model's window, whichever is smaller.
  - The point follows the window on a `/model` switch. The log notes 90%.
  - A summary brings the input down to half the point: how many of the oldest turns go into it
    follows their token counts. The task message stays verbatim, and so do the latest four
    messages. If the first pass leaves the input above the point, a second pass runs. The log
    notes `context compacted: older turns summarized, 305.0k → 98.1k tokens`. If the summarizer
    fails, the context is sent as it is, and the log, the footer and `events.jsonl`
    (`compact_failed`) say why.
  - When a compaction changes nothing (the summarizer failed, or nothing could be cut), automatic
    retries wait until the input has grown 10% or ten calls have passed; overflow and `context`
    requests still retry at once.
  - **Pruning.** Reasoning from earlier turns (thinking text and signatures, which some models bill
    as input on every call) is dropped; the turn in progress keeps it, as Anthropic requires within
    a tool-use loop. `shell` and `read` results over 2,000 characters are cut to their head and
    tail (~800 characters) once six newer ones exist; `retrieve_context` with the reference at the
    end returns the full output.
  - Both edit history, and the prompt cache only matches an unchanged prefix, so they run only when
    the cache is cold anyway: in the pass that summarizes, after the context was edited (e.g. by
    the `context` tool), on the first call after the cache's lifetime (five idle minutes, an hour
    with `--cache-ttl 1h`), on overflow, or once they would remove 40% of the history. Every other
    call re-sends history byte for byte.
  - The `context` tool lets the agent manage its own context. It lists the biggest items with tool
    results numbered. `offload: [n, …]` replaces results with a stub whose reference reads them
    back. `summarize: true` summarizes all but the latest four messages before the next call.
- **Tool order.** A batch of tool calls runs at once, except that `read`, `write`, `edit`,
  `shell`, `run_background`, `land`, `spawn_agent`, `cancel_agent`, `send_message`, `wait`,
  `list_agents` and `jobs` run one after another in the model's order: a test run waits for the
  edits before it, a read for the command that writes the file, a worktree spawn for the commit
  it should branch from, a cancel for the command that cleans the worktree it reclaims, a `wait`
  for the spawns and jobs before it.
- **Worktrees.**
  - A worktree agent works in `<repo>.swarm/<run>/<agent>` on branch `swarm/<run>/<agent>`.
  - When `CARGO_TARGET_DIR` is set, it builds in its own `$CARGO_TARGET_DIR-worktrees/<run>/<agent>`.
    Builds are not shared, so cargo cannot take another checkout's artifacts as fresh.
- **Restarts.** An agent that was mid-turn resumes with a short notice instead of re-reading its
  work. The notice lists the tool calls the restart cut short, each with how long it had run, and
  `git status --short --branch` of its directory. After a hard crash, "how long" runs to the
  run's last write (an event, a save, or swarmctl noting the exit), not to its last save. When a cut call had run a minute or more, it
  also says that background jobs survive restarts. The cut turn ends with "Interrupted: the swarm
  restarted." rather than the SDK's "Cancelled by user", which a real `/stop` keeps. An agent
  that had finished its turn rests until its next message; a goal agent whose turn failed
  retries.
- **Hard crashes.** The SDK adds a batch of tool calls to the conversation only once all of them
  finished, so each agent's batch in progress is also written ahead to
  `sessions/<agent>/open-batch.json`. After a kill -9, OOM kill or power loss the resumed agent gets
  the batch back: the calls, the results that came in, and the cut calls in its notice.
  Only after such a crash (no `clean-stop` mark from a quit or reload) does a resume (Linux) kill
  the process groups of the shell calls the batch had cut, found by `STRANDS_SWARM_SHELL=<run dir>#<agent>`;
  `nohup … &` leftovers of earlier calls and background jobs run on.

## Limits

- **Agents.** `--max-agents N` (16) caps live agents.
  - Spawns beyond the cap are `queued` and start oldest first as slots free.
  - A queued agent can already be messaged; cancelling it drops it.
  - A `wait` for queued agents returns at once, naming the idle agents that hold slots, when no
    other agent is running that could cancel one: only the waiter can free a slot for them.
  - An idle agent that is only waiting on a schedule holds no slot.
  - `/max-agents N` changes the cap for the running swarm and is kept across restarts.
    `/max-agents default` returns to the flag.
- **Model calls.** `--max-concurrent-calls N` (8, `0` = unlimited) caps model calls in flight
  across the swarm. Extra calls queue ("queued for a model slot…") instead of tripping provider
  throttling.
- **Model errors.**
  - Throttling, server errors (5xx, Bedrock's `InternalServerException` and the like) and dropped
    connections are retried with backoff, up to 10 attempts (about eight minutes). A rejected
    request is not retried.
  - If retrying doesn't help, the turn fails and the parent gets an `error` message. A goal agent
    tries again (15 s, doubling to 10 min) and reports only the first failure. The judge sees only
    replies, not failed turns.
  - The next message picks up where the failed call stopped.
- **Tokens.** `--budget-tokens N` (e.g. `5M`) caps the swarm's input + output tokens.
  - Once spent, model calls fail with a "budget exhausted" error that travels up to you.
    Each agent reports it once: mail that wakes it again (a schedule firing) is refused without
    another error until the budget is raised.
  - `/budget N` raises it and tells every stopped agent to continue. `/budget off` lifts it and
    `/budget` shows usage.
  - The budget is saved with the run. A resume keeps it unless given a `--budget-tokens` other
    than the one the run was last started with (a supervisor that replays its arguments leaves
    a `/budget` change in place).

## Cost

- **Estimates, not a bill.** Spend is estimated from reported usage, tracked per agent and model, so
  a `/model` switch is priced per model.
- **Prices.**
  - A built-in table holds Claude list prices per 1M tokens by tier (opus, sonnet, haiku, plus older
    versions). Cache reads cost 0.1× the input price and cache writes 1.25× (2× with
    `--cache-ttl 1h`).
  - `--prices FILE` adds or overrides entries. It is a JSON object keyed by model-id substring:
    `{"opus": {"input": 5, "output": 25}}`, with optional `cache_read`/`cache_write`.
  - Its keys win over the built-in ones; within each, the longest match wins.
  - Unpriced models show `unknown`, or `$3.10+?` when partly priced.
- **Cache TTL.** `--cache-ttl 1h` makes Bedrock/Anthropic keep cached prefixes for an hour
  instead of five minutes, so a call after a long `wait` or a slow reload reads its prefix
  instead of rewriting it. Every write then costs 2× input instead of 1.25×, so it pays off only
  when calls often follow a 5–60 minute pause: per call, 0.75× the new tokens it writes against
  1.15× the prefix for each pause it bridges. Unavoidable misses (a new agent, a binary that
  changed the tools or prompt) cost the 2× rate too. Off by default.
- **Keep-warm.** While a turn waits on a tool call (a `wait`, a slow build) after a call that hit
  the cache, the agent's last request is re-sent every 4.5 minutes and dropped at the first
  token: the provider reads (0.1×) and so refreshes the cached prefix that would otherwise expire
  and be rewritten (1.25×) on the next call. It stops at that call, an interrupt or cancel, or
  after 12 refreshes (under an hour); refreshes pass the model-call gate, count as a full cache
  read toward the agent's tokens, and are logged as `keep_warm` in `events.jsonl`. On by default
  with the 5-minute cache; `--no-keep-warm` turns it off.
- **Where it shows.**
  - `/cost` (or `--run NAME --cost`) shows per agent: input, output, cache-read and cache-write
    tokens, the model, an estimated `$`, and the median input per call and cache hit rate over the
    last 50 calls. Below the table, side calls get their own rows by purpose: the goal `judge`,
    the context `summarizer` and `other` calls on the auxiliary model (told apart by the system
    prompt they send, so a custom judge or summarizer prompt counts as `other`). They are already
    part of each agent's totals.
  - The TUI header shows `est. $12.30`.
  - `list_agents` gives models the same numbers: `est. $4.10 / 12.3M tokens`, a subtree total
    (`with 3 descendants $9.80 / 30.1M`) and a swarm total. A coordinator can spot an expensive
    worker and cancel it or move the work to a cheaper model.
  - `spawn_agent`'s `model` notes the relative prices (sonnet ≈ 0.6× opus, haiku ≈ 0.2×).
  - None of this goes into the system prompt, which stays byte-stable.
  - Programmatically: `Swarm::cost`.

## Background jobs and schedules

- **Background jobs.**
  - Jobs run with `sh -c` in the agent's directory and environment, private `CARGO_TARGET_DIR`
    included. Each runs in a session of its own, so neither a reload nor Ctrl-C stops it.
  - Each job is `<id>.json`, `<id>.log`, `<id>.exit` and, once its owner was told, `<id>.reported`
    in `<run>/jobs` (a temp directory without `--run`).
  - The swarm polls jobs and mails the owner when one ends, also for jobs started before a restart.
    A job whose process vanished without an exit code (a reboot) is reported as such.
  - A killed job is not reported. Cancelling an agent kills its whole subtree's jobs.
  - A `shell` call that sleeps 30 s or more (`sleep 240; tail job.log`) while its agent has a job
    running gets one line appended to its result: the job is still running, and `wait` with
    `from: ["jobs"]` returns as soon as it ends, whereas a sleep can't be cut short by mail or a reload.
  - A running job's log over 32 MB is cut to its last 1 MB. Only the newest 200 reported jobs keep
    their files.
- **Schedules.**
  - `schedule {message, to?, every?: "30m", at?: "in 2h" | "2026-11-17T09:00Z", id?}` posts
    `[schedule sN] message` (kind `schedule`) at the given times.
  - `{cancel: id}` removes one and `{}` lists them. Repeats are at least a minute apart.
  - An agent that polls something schedules itself and ends its turn instead of a long `wait`.
  - From its second fire on, a repeat adds what the receiver spent since the last one:
    `(You used 330.0k tokens, est. $0.21, since this schedule last fired 2m ago: about $6.30/h at
    this interval.)`. A poll over a large context costs about that each time even when nothing
    changed, so the agent can see what its loop costs and pick the interval.
  - Schedules are saved with the run. One that came due while the swarm was down fires once on
    resume, then keeps its interval.
  - Schedules of a cancelled agent (receiver or creator) are dropped.

## Landing

`land` is opt-in with `--allow-land`. It lets a worker merge its own work instead of handing it to
an integrator. `{branch?, into?, check?: [commands]}` lands `branch` on `into`.

- **Which branches.**
  - `branch` defaults to the caller's own. It may only be that or the branch of an agent below it.
  - `into` defaults to the branch the caller's root agent has checked out. Any other target must
    match a `--land-into GLOB` (repeatable, `*` and `?`).
- **How it lands.** As a background job, it:
  1. rebases a scratch worktree at the branch's tip onto `into`;
  2. runs each check with `sh` from the caller's subdirectory, using a private `CARGO_TARGET_DIR`
     (default checks: `--gate CMD`, repeatable);
  3. only then fast-forwards `into`: `git merge --ff-only` where it is checked out, else a
     compare-and-swap `update-ref`.
- **Refusals.**
  - It refuses when that checkout has uncommitted changes to tracked files, or untracked files
    where the landing adds files (checked before the checks run). It gives up if `into`
    moved meanwhile. Land jobs take turns from that check to the fast-forward (a lock in the
    repository's git dir), so of two that finish together one lands and the other is told so.
  - A conflict (with the files), a failing check (with its last 30 lines) or a refusal changes
    nothing.
  - The scratch worktree is always removed, and the caller's branch is never modified. A land job
    that is stopped (killed with `jobs`, or its agent cancelled) kills the check in flight with its
    children and removes the scratch worktree too. One killed outright (`kill -9`, a reboot)
    leaves it behind; the next resume, `/gc` or `--gc` that finds no land job running removes it
    (a `gc` event for `land`), with any plan left in `jobs/land` (each job deletes its own) once
    a minute old.
  - The checks' shared target dir (`$CARGO_TARGET_DIR-land/<run>`, else `jobs/land/target`) stays
    for the next landing's incremental build. `/gc` and `--gc` delete it when no land job runs,
    and a run without `--run` on exit.
- Programmatically: `Swarm::land`.

## TUI

**Commands.** Typing `/` opens a filtered popup of the commands; F1 lists the keys.

| Command | What it does |
| --- | --- |
| `/new <task>` | Start a new root agent. |
| `/goal <goal>` | Start a root agent that works until the goal is met. |
| `/to <agent> <message>` | Message an agent without selecting it. |
| `/model [<id> \| default]` | Switch the selected agent's model from its next call on (saved with the run, `Swarm::set_model`). |
| `/image <path> [text]` | Send the selected agent an image. |
| `/stop` (or Esc twice on an empty input) | Interrupt the selected agent's turn without cancelling it (`Swarm::interrupt`). |
| `/cancel` | Stop the selected agent and its subtree for good. |
| `/status` | Each live agent's state, activity, todos, unmerged commits and merge outlook, jobs and schedules, without a model call. |
| `/cost` | Estimated spend (see [Cost](#cost)). |
| `/budget`, `/max-agents` | See [Limits](#limits). |
| `/remember`, `/forget` | Edit the memory file (see [Persistence](#persistence)). |
| `/schedules` | All schedules. |
| `/gc` | Retry reclaiming the disk of cancelled worktree agents. |
| `/quit` (`/q`, Ctrl-C) | Quit. |

Other text messages the selected agent, or starts a root agent when there is none.

**Command details.**

- **`/model`:** `/model default` returns to `--model`, and `/model` alone shows the current model.
  The id is resolved like `--model`, and an invalid one is rejected. `opus`, `sonnet` and `haiku`
  (here, in `--model` and `--side-model`, and in `spawn_agent`'s `model`) stand for that tier's
  Claude 5.5 id with `--model`'s provider prefix, e.g. `bedrock/global.anthropic.claude-sonnet-5-5`.
- **`/stop`:** an open tool call is closed as cancelled and the agent goes idle with its
  conversation intact. Its parent gets an `interrupted` message and its children keep running. A
  goal agent waits for its next message.
- **`/status`:**
  - Unmerged commits are `git log <base>..<branch>`. The base is `--status-base`, else the branch
    checked out where the swarm started (a linked worktree's, if it started in one).
  - Each branch also says how many base commits it lacks and how it would merge:
    `3 not on main, 5 behind, merges cleanly` or `… conflicts in README.md, src/tui/view.rs`.
    The trial merge is `git merge-tree --write-tree` (git ≥ 2.38), which touches no checkout or
    ref, so the integrator sees conflicts before cherry-picking. Commits the base already has a
    copy of (cherry-picked or rebased, same patch id per `git cherry`) or that `land` landed
    (recorded in `<git common dir>/swarm-landed`, since a rebase can change patch ids) are counted:
    `3 not on main (all applied), 4 behind` means nothing is left to integrate. `list_agents`
    shows the same, fresh on every call.
  - The branch compared is the one checked out in the agent's cwd (its given `swarm/<run>/<id>`
    when detached); `list_agents` names it when it is another (`checked out swarm/r/a7-fix`).
    The agent's other branches (`swarm/<run>/<id>` and `swarm/<run>/<id>-*`) whose commits the
    base lacks are listed after it: `…; also unmerged: swarm/r/a7`.
  - It refreshes while open; git runs in the background and is cached for 5 s.
  - Programmatically: `Swarm::status`.

**Agent tree.**

- Each agent is one line: status, name, model, input/output tokens and how close its last call
  came to the compaction point, e.g. `● parser opus-5-5 1.2M/34.0k 34% a3`.
- Cancelled agents without a live descendant are hidden (the header counts them); F3 shows them.
- Alt-←/→ narrows or widens the tree. Narrow it past its minimum to hide it.

**Header.**

- It totals tokens as `in 84.8M (97% cached) · out 291k · est. $12.30`.
- Input counts cache reads and writes, as the budget does.

**Input.**

- The input box edits like a shell line: ←/→, Home/End (Ctrl-A/E), Ctrl-←/→ by word, Delete,
  Ctrl-W/Ctrl-U (Alt-Backspace). Tab completes the highlighted command.
- Alt-Enter (or Shift-Enter where reported) inserts a new line.
- ↑/↓ go through sent messages while the input holds text (Ctrl-P/N from an empty one). On an
  empty input they select agents.

**Images.**

- Images from tool results (e.g. `read` on a png) and from `/image` show as an
  `[image: path WxH]` line.
- Below it comes a thumbnail of at most 12 rows when the terminal answers with kitty, iTerm2 or
  sixel graphics. Inside tmux (so under `swarmctl`), and on terminals without graphics, there is
  no thumbnail: unicode halfblocks would blur it beyond reading. Open the path to see the image.
- Only the newest 4 stay decoded. The mailbox and journal keep only the path.

**Log and copying.**

- F2 prints the selected agent's log as plain text into the terminal's scrollback; any key returns.
- A resumed run shows the last 100 messages of each agent's restored conversation, cancelled
  agents included (F3 lists them).
- Drag to select text in the log or tree. Releasing copies it with OSC 52, which works over ssh;
  tmux needs `set -g set-clipboard on`.
- The wheel and PgUp/PgDn scroll the log, End (on an empty input) jumps to its bottom, and Esc
  clears the highlight. Hold Shift (Option/Alt in iTerm2) for the terminal's own selection.

**Notifications.** `--notify` rings the bell when an agent, the supervisor or the disk guard
messages you.

## Persistence

- **Run dir.** `swarm.json` (agent tree, statuses, unread mail, token counts, schedules, budget,
  agent cap; replaced atomically), `sessions/` (each agent's conversation and context stash),
  `jobs/`, `journal.jsonl`, `events.jsonl`, `swarm.log`, `logs/` (MCP servers), `mcp-tools.json`
  and `bin/` (installed binaries). `swarmctl` adds `args`, `env`, `stderr.log`, `supervisor.log`,
  `supervisor.notice` and the `stop` marker.
- **Journal.**
  - A persistent run keeps `journal.jsonl` with every message to and from the user, and
    `events.jsonl` with compact lifecycle events: spawns, status changes, tool calls and their
    ends, model switches, compactions (`before`/`after` tokens) and failed ones, side calls
    (`side_call`: purpose, model and tokens of each judge or summarizer call), keep-warm
    refreshes, schedules, jobs, budget and agent-cap changes, resumes, reloads, version
    `changes`, shutdowns and disk reclaimed (`gc`).
  - Both are JSON lines with a UTC `ts`, written by a background thread and rotated to `<file>.1`
    at 8 MB.
- **Log.** Warnings and errors (or whatever `RUST_LOG` selects) go to `<run dir>/swarm.log` under
  the TUI (`$TMPDIR/strands-swarm-<pid>.log` without `--run`), and to stderr otherwise. When a
  process opens a log that has grown past 8 MB, it is first moved to `swarm.log.1`.
- **Terminal.** Every child the swarm starts (shell commands, jobs, git, land gates, MCP servers)
  runs in a session of its own, without a controlling terminal. Nothing it writes to `/dev/tty`
  reaches the TUI, and a prompt (git credentials, an ssh host key) fails at once instead of hanging.
- **Disk.**
  - Once a cancelled worktree agent and its subtree have stopped (or on resume), its private target
    dir is deleted.
  - Its worktree is removed with `git worktree remove` unless it has uncommitted changes (the
    reason names the first three, e.g. `uncommitted changes: ?? tests/__pycache__/`, often only
    build output to delete), and so are `<repo>.swarm/<run>` and `<repo>.swarm` once that leaves
    them empty. A worktree with
    another branch checked out goes only once the base has all of that branch (which stays);
    otherwise it is kept, e.g. `on branch swarm/r/a3-fix: 1 commit not on main`. A detached one
    is kept.
  - Its branch `swarm/<run>/<agent>` is deleted once no worktree has it checked out and the base
    `/status` uses has every commit on it: reachable (`merged`) or as a patch-equivalent copy
    (`all applied`, per `git cherry` or the landing record; a merge commit never counts as
    applied unless landed). Otherwise it stays,
    e.g. `kept branch swarm/r/a3 (2 commits not on main)`, until a later pass (resume, `/gc`,
    `--gc`) finds it merged. The ref is deleted only if it still points at the commit checked.
  - `cancel_agent` waits up to 10 s for this and its result says what was removed and kept.
    `list_agents` with `include_cancelled` shows it later: `cwd … (removed)`, `branch … (deleted)`
    or `branch … (kept: 2 not on main, merges cleanly)`.
  - A run without `--run` cannot be resumed, so on exit every worktree agent is treated as
    cancelled.
  - Nothing outside those exact paths and that branch is touched, and each removal is logged as a
    `gc` event. A pass on its own (after a cancel, on resume) logs an agent only when its outcome
    changed: a worktree still kept for the same reason is not logged again on every restart.
  - Only the swarm's own `$CARGO_TARGET_DIR-worktrees/<run>/<agent>` dirs are reclaimed. An agent
    that points `CARGO_TARGET_DIR` somewhere else owns that dir and must clean it itself.
  - Every minute the swarm checks free space on the filesystems of the run dir, the agents'
    directories and the cargo target dirs. Below `--min-free-disk` (default `20G`, `0` turns it
    off) it mails the live root agents (or, with none, you) once, from `disk`, with the free space
    and the biggest per-agent target dirs (and the land checks' shared `$CARGO_TARGET_DIR-land/<run>`,
    which only `/gc` and `--gc` reclaim); the TUI shows the first line in its footer. It warns
    again only after the space has recovered to 10% above the threshold, restarts included (a
    persistent run notes the warning in `<run dir>/disk-warned`).
- **Memory.**
  - One Markdown file is shared by every agent and run: `$XDG_CONFIG_HOME/strands-swarm/memory.md`
    (default `~/.config`), or `--memory FILE`, or off with `--no-memory`.
  - Entries are dated bullets, and the file can be edited by hand.
  - Its newest ~4k tokens are appended to every agent's system prompt, re-read at each turn's
    start.
  - Agents use `remember`/`forget`; you use `/remember <text>`, `/forget <text>` and `/remember`
    (which shows where the file is).
  - Writes are serialized and atomic.
- **MCP.**
  - `--mcp-config` accepts any `mcp.json` (or `~/.claude.json`). Servers are shared by all agents
    and start lazily.
  - With `--run`, each server's tool list is cached in `<run dir>/mcp-tools.json` under a hash of
    its config. A start with an unchanged config connects no server (about 6.5 s → 0.03 s for 274
    tools).
  - A server connects when an agent enables one of its tools, or on the first call (60 s timeout),
    and reconnects if it died.
  - Agents start without MCP tools. `find_tools` (whose description lists the servers) searches the
    catalog and enables tools from the next call on. Enabled tools come back on resume.
  - `--mcp-eager` registers every tool on every agent, `--mcp-deny` hides tools by name, and
    `--list-tools` prints what is allowed and denied.
  - Under the TUI a server's stderr goes to `<run dir>/logs/mcp-<server>.log`
    (`$TMPDIR/strands-swarm-<pid>/` without `--run`), rotated to `<log>.1` at 4 MB; other modes
    keep it on the terminal.
- **Skills.**
  - Skills are `<name>/SKILL.md` files with `name` and `description` frontmatter, as in Claude Code
    and Codex. They are discovered at startup.
  - The system prompt lists each as `name: summary`: the description's leading whole sentences
    that fit in 100 characters, or its first sentence cut there (about 25 tokens a skill). The
    `skills` tool returns the full description with the instructions.
  - On a name clash the first source wins, in this order: `--skills-dir`; the project's
    `.claude/skills`, `.agents/skills`, `.agent/skills`; `~/.claude/skills`; enabled Claude Code
    plugins; `~/.codex/skills`; `~/.agents/skills`.
  - `--no-default-skills` keeps only `--skills-dir`, and `--list-skills` prints the result.

## Operate 24/7

`scripts/swarmctl` keeps a run alive in tmux. It restarts the run whenever it exits (after 5 s,
`SWARMCTL_RESTART_DELAY`), a normal quit included, until `swarmctl stop`, and restarts into the
new binary at once on `reload_swarm`. The swarm's stderr goes to `<run dir>/stderr.log` and the
supervisor's notes to `supervisor.log`, each rotated to `<log>.1` at 8 MB (`SWARMCTL_LOG_MAX`).

**Failures.**

- A swarm that hangs exits instead, so the supervisor sees it: 70 when a deadlock is detected, 71
  when it is not up (first TUI frame drawn, or headless running) within 180 s.
- A new binary is on trial for 10 minutes (`SWARMCTL_TRIAL_WINDOW`) from its reload (or
  `swarmctl start`). If it fails (non-zero exit, a signal, 70, 71) on trial, the supervisor puts
  the previous binary back and keeps the failed one as `bin/failed`. A binary that ran longer is
  restarted, never rolled back, and a rolled-back one is not rolled back again.
- Quick failures in a row (each within 60 s of starting) back off from 5 s, doubling up to 5
  minutes. The tmux pane shows each failure with the tail of `stderr.log`, and `swarmctl status`
  the supervisor's last note.
- A rollback, and a third failure in a row (the last two quick), are left in
  `<run dir>/supervisor.notice` with the stderr tail. When the swarm next starts it mails the
  notice to its live root agents (or you, with none; from `supervisor`, also shown in the footer)
  and deletes it.

```sh
swarmctl start NAME ./target/release/strands-swarm --cwd REPO --goal "..."   # start or resume
swarmctl attach NAME        # open the TUI (detach: C-b d)
swarmctl stop NAME          # graceful stop (the session is killed after 30 s); state is kept
swarmctl resume NAME        # start again with the saved binary and arguments
swarmctl boot NAME          # resume after every reboot (crontab @reboot); unboot NAME undoes it
swarmctl status
```

**Self-test.**

- Before a reload installs a binary, `NEW --version` must answer within 30 s, then
  `NEW --self-test <copy>` resumes a temporary copy of the run directory on a stub model that
  answers `ok`.
- It messages every live agent and requires each to reply and every saved conversation to be
  restored within 20 s. It also renders the TUI into a test backend and checks the deadlock
  watchdog.
- The copy's agents lose their branches, private env and directories, disk reclamation is off
  (`SwarmConfig::reclaim_disk`), and the stub calls no tools. The real run is never touched, and
  the copy is deleted after.
- On failure or after 60 s, `reload_swarm` returns the error with its stderr tail and the swarm
  keeps its binary.

**Reload timing.**

- A binary that passes is installed at once. `when: "now"` restarts right away.
- The default, `"idle"`, waits until no agent has a tool call in flight (`wait` excluded), at most
  10 minutes (`SwarmConfig::reload_idle_timeout`). The header shows
  `reload pending (waiting for N tool calls)`.
- While it waits, agents start no new tool calls: each agent's next model call (and side call)
  waits for the restart ("waiting for the reload…"), so a busy swarm still gets idle. Its tool
  results are saved, and the resumed agent makes that call; nothing is cut. Keep-warm refreshes go
  on, so that call finds the cache warm.
- A newer reload replaces a pending one.

**Version tracking.** Each binary embeds its git commit (`build.rs`), saved with the run. When a run
resumes under another commit, the TUI notes the commit subjects in between (`git log old..new` in
the source checkout), and `events.jsonl` gets a `changes` entry.
