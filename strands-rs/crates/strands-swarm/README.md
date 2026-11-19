# strands-swarm

A model-driven agent swarm on `strands-harness`. No planner, no scheduler: every agent is a harness
agent with the swarm tools below, and decides for itself whether and how to split the work.

| Tool | What it does |
| --- | --- |
| `spawn_agent` | Start an agent on a task (optionally in another directory, in its own git worktree, with a model, a goal or its own compaction point). Non-blocking. |
| `send_message` | Message any agent (by id or name), or `parent`; a comma-separated `to` sends one message to each (every name must resolve first, so a typo sends nothing). Busy agents see it after their current tool call; idle ones wake up, unless it is sent with `wake: false`. |
| `wait` | Block until messages arrive (optionally from specific agents, or `jobs` for job-end reports, which returns at once when none is pending); a user message always ends it. With `all`, until every agent in `from` has sent something; a timeout (default 10 min, at most an hour) returns what came and who is still missing. Mail stays unread (and saved) until it returns, so a restart meanwhile loses none. |
| `list_agents` | The agent tree: status, directory, branch (ahead/behind the base, clean merge or conflicting files, other agents' branches changing the same files), running jobs, schedules, model (and an agent's own compaction point), estimated spend, and which idle worktree agents have nothing left to merge (cancel to reclaim their disk). |
| `cancel_agent` | Stop an agent and everything it spawned, background jobs included; reports which worktrees and merged branches were removed and what was kept. |
| `reload_swarm` | With `--run`: restart the whole swarm into a newly built binary; every agent resumes. |
| `schedule` | Post a message to a mailbox (default: your own) later and/or repeatedly (`every: "30m"`, at least a minute); list or cancel schedules. |
| `run_background` | Run a shell command as a background job; its owner gets the exit code and last 40 output lines when it ends. |
| `jobs` | List your background jobs and other agents' running ones (status, log path; other agents' ended jobs only counted), or kill one with everything it started. |
| `context` | The agent's context dashboard: size against the compaction point, the biggest items, the stash; offload results or summarize now. |
| `land` | With `--allow-land`: rebase a branch onto the root's branch, run the checks, fast-forward. |
| `find_tools` | With MCP servers: search the shared MCP tools and enable some for the calling agent. |
| `skills` | With skills: load a skill's `SKILL.md` (and its full description), its location and bundled files. |
| `remember` / `forget` | With memory: add a dated entry to the user's memory file, or drop matching lines. |

Agents also get the harness's `shell`, `read`, `write`, `edit`, `web_fetch`, `todo_write` and
`retrieve_context`, but not the SDK's background tasks (`_background_execution` on every tool):
`run_background` covers slow commands. A `shell` result reaches the model as the text a terminal
would show (stdout, then `[stderr]` and `[exit code N]` when there are any), not the SDK's JSON
block, whose escaped quotes models copied into `edit`'s `old_str`. It and a job report's last lines
drop ANSI escape sequences, and of a line `\r` overwrote (progress bars) only the final text. Tool
specs and the swarm's prompt sections are sent on every model call, so
`the_fixed_prefix_stays_within_its_budget` pins their size: 16.4 kB (about 4.8k tokens) with 20
skills, the skills list at 2.5 kB of it.

## Run

```sh
# Setting the head makes the build embed it even when worktrees share a target dir.
STRANDS_SWARM_BUILD_HEAD=$(git rev-parse HEAD) cargo build --release -p strands-swarm
strands-swarm [--run NAME] [--cwd DIR] [--goal GOAL] [--model ID] [--side-model ID] [--effort E]
              [--cache-ttl 5m|1h] [--no-keep-warm]
              [--instructions FILE] [--memory FILE | --no-memory]
              [--mcp-config mcp.json]... [--mcp-deny REGEX]... [--mcp-deny-except REGEX=AGENT[,AGENT…]]...
              [--mcp-eager] [--list-tools]
              [--skills-dir DIR]... [--no-default-skills] [--list-skills]
              [--max-agents N] [--max-concurrent-calls N] [--max-load N] [--min-free-mem SIZE]
              [--priority AGENT]... [--build-jobs N]
              [--budget-tokens N] [--compact-at TOKENS]
              [--allow-land] [--land-into GLOB]... [--gate CMD]...
              [--notify] [--status-base BRANCH] [--prices FILE] [--min-free-disk SIZE] [--headless]
              [--journal | --gc | --cost] [TASK]
```

`--run NAME` makes the run persistent under `$XDG_STATE_HOME/strands-swarm/NAME` (default
`~/.local/state`): the agent tree, unread mail, schedules and every conversation (saved after
each message; see [Persistence](#persistence)). Starting again with the same name resumes it; a
TASK given then goes to the root agent as a message unless it is the one the run was last started
with (a supervisor replays its arguments), and `--goal` only seeds a new root. With `--run`,
`--journal`, `--gc` and `--cost` print a report on the saved run and exit.

`--headless` prints events to stderr and messages to you to stdout instead of the TUI, and exits
once the root has replied and no agent is busy (a resume without a task: once no agent is busy)
or every agent has stopped, with a non-zero status when that reply is an error. A new run needs a
TASK or `--goal`. Without a terminal, the TUI refuses to start (before resuming anything) and
points to `--headless`.

**AWS credentials.** Every process the swarm starts (agent shells, `run_background` jobs, land
checks, MCP servers) inherits its environment, `AWS_*` included, untouched. To call Bedrock (and Bedrock Mantle, whose
bearer tokens it mints from the same credentials) under settings of its own without handing them
to those, set `STRANDS_SWARM_AWS_PROFILE`,
`STRANDS_SWARM_AWS_CONFIG_FILE`, `STRANDS_SWARM_AWS_SHARED_CREDENTIALS_FILE` and/or
`STRANDS_SWARM_AWS_REGION`: only the swarm's model client reads them, in place of the matching
`AWS_*`, and no AWS tool knows these names. With a profile set, its credentials and region win
over access keys and `AWS_REGION` in the environment; unset ones fall back to the AWS chain. A
user who just exports `AWS_PROFILE` gets it for both.

```sh
STRANDS_SWARM_AWS_PROFILE=swarm STRANDS_SWARM_AWS_CONFIG_FILE=~/.config/strands-swarm/aws-config \
  STRANDS_SWARM_AWS_SHARED_CREDENTIALS_FILE=/dev/null swarmctl start NAME ./strands-swarm ...
```

## Agents

- **Names.** Every agent has a short, unique name (`lazy-mcp (a2)`): the one its spawner gave, else
  one derived from its task. Ids and names work interchangeably, and so do the `name (id)` labels
  the swarm prints and a name as the spawner wrote it (`Fix-P12` for `fix-p12`).
- **Replies.** Each agent's final reply per turn goes to its parent. Mail that arrives mid-turn is
  appended to the next tool result. A reply that is empty or just `[silent]` (e.g. a scheduled poll
  that found nothing) reaches no agent parent; the user's root always replies. A `send_message` to
  an agent's parent says so in its result, as agents that report mid-turn tend to repeat the report
  as their final reply, which costs the parent a turn.
- **FYI mail.** Every wake costs the receiver a whole turn, its context re-sent. `send_message`
  with `wake: false` posts kind `fyi` (an acknowledgement, a progress note, news a worker can use
  later): an idle receiver is not woken but gets it with its next waking mail, a busy one after
  its current tool call as usual. It ends no `wait` (`all` included), counts as no unread mail
  in `list_agents` and `/status`, keeps no headless run alive, and is saved like other mail.
- **Working directory.** An agent works where its spawner does unless `spawn_agent` gets a `cwd`
  (absolute or relative), e.g. another repository; with `worktree` the worktree is cut from the
  repository there. An agent moves itself with a `shell` call that is only `cd DIR` (one word or
  quoted; relative, absolute or `~`): instead of running, it moves the agent's later `shell` and
  `run_background` calls, the children it spawns without a `cwd`, the working directory its system
  prompt and environment block (`AGENTS.md`, READMEs) name, and what `list_agents` and the TUI
  show. The move is saved (`workdir` in `swarm.json`, `moved` in `events.jsonl`) and kept across
  restarts. A `cd` to something that is not a directory runs as usual, so the shell reports it.
  Where the agent started stays its home for worktree, branch and reclaim bookkeeping.
  A move edits the system prompt, so the agent's next call writes its prompt cache anew.
- **Goals.** A `goal` (`spawn_agent`, `/goal`, `--goal`) puts a judge in the loop that sends the
  agent back to work until the goal is met. It runs after every turn and sees the goal, the task
  and the latest ~40k characters of the conversation.
- **Side calls.** The judge and context summarization run on the agent's model or `--side-model ID`.
  They count toward the agent's tokens and the budget, and queue like its own calls. They skip
  the prompt cache: their prompts are never sent twice, so a cache write would cost 25% extra.
- **Context.**
  - An agent's older turns are summarized once a call's input reaches the compaction point:
    85% of the model's window (850k tokens for a 1M window), or `--compact-at TOKENS` if that
    is smaller. `spawn_agent`'s `compact_at` sets one agent's point instead, saved with the agent
    across restarts; an agent spawned without one follows `--compact-at` as the run is started.
  - The point follows the window on a `/model` switch. The log notes 90%.
  - A summary brings the input down to half the point: how many of the oldest turns go into it
    follows their token counts. The task message stays verbatim, and so do the latest four
    messages. If the first pass leaves the input above the point, a second pass runs. The log
    notes `context compacted: older turns summarized, 305.0k → 98.1k tokens`. The summarizer
    gets the turns in slices of at most half its own window (one call at 850k of a 1M window;
    a 200k-window `--side-model` gets 100k slices), each with the summary so far; a slice it
    still rejects as too long is halved and sent again, and a call that fails transiently
    (throttling, a dropped stream) is tried twice more, 5 s and 10 s later. If the summarizer
    fails, the context is sent as it is, and the log, the footer and `events.jsonl`
    (`compact_failed`) say why.
  - When a compaction changes nothing (the summarizer failed, or nothing could be cut), automatic
    retries wait until the input has grown 10% or ten calls have passed; overflow and `context`
    requests still retry at once.
  - **Pruning.** Reasoning (thinking text and signatures, which some models bill as input on
    every call) is dropped from every assistant message but the latest one in the turn in progress
    that has any: in a long tool loop, earlier steps' reasoning goes too. Bedrock requires the last
    assistant message's thinking back, whole and signed, and Anthropic accepts leaving out the
    oldest blocks. Thinking after anything pruning edits (a trimmed result, a summary, a `context`
    offload) goes as well, the latest included: Anthropic binds thinking to all that precedes it,
    and allows leaving out the newest. Tool results over 2,000 characters (`shell`, `read`, MCP
    listings and the rest, but not `skills`, `wait` or `retrieve_context`) are cut to their head
    and tail (~800 characters) once six newer ones exist; `retrieve_context` with the reference at
    the end returns the full output.
  - Both edit history, and the prompt cache only matches an unchanged prefix, so they run only when
    the cache is cold anyway: in the pass that summarizes, after the context was edited (e.g. by
    the `context` tool), after the system prompt or tools changed (a `cd`, a `find_tools` enable;
    then all thinking goes, as it is bound to them), on the first call after the cache's lifetime
    (five idle minutes, an hour with `--cache-ttl 1h`), on overflow, or once they would remove 40%
    of the history. Every other call re-sends history byte for byte.
  - The `context` tool lets the agent manage its own context. It lists the biggest items with tool
    results numbered (not previews and stubs, which offloading would not shrink). `offload: [n, …]`
    replaces results with a stub whose reference reads them back; a stub names only its own
    result's stash keys. `summarize: true` summarizes all but the latest four messages before the
    next call.
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
  run's last write (an event, a save, or swarmctl noting the exit). When a cut call had run a
  minute or more, it also says that background jobs survive restarts. A cut `wait` is listed
  apart, with the note that nothing it waited for is lost and polling is not needed. The cut
  turn ends with "Interrupted: the swarm restarted." rather than the SDK's "Cancelled by user",
  which a real `/stop` keeps. An agent
  that had finished its turn rests until its next message; a goal agent whose turn failed
  retries.
- **Hard crashes.** The SDK adds a batch of tool calls to the conversation only once all of them
  finished, so each agent's batch in progress is also written ahead to
  `sessions/<agent>/open-batch.json`, synced to disk on each change. After a kill -9, OOM kill or
  power loss the resumed agent gets the batch back: the calls, the results that came in, and the
  cut calls in its notice. Only after such a crash (no `clean-stop` mark from a quit or reload)
  does a resume (Linux) kill the process
  groups of the shell calls the batch had cut, found by `STRANDS_SWARM_SHELL=<run dir>#<agent>`;
  `nohup … &` leftovers of earlier calls and background jobs run on.

## Limits

- **Agents.** `--max-agents N` (16) caps live agents.
  - Spawns beyond the cap are `queued` and start oldest first as slots free (the priority lane,
    below, never queues).
  - A queued agent can already be messaged; cancelling it drops it.
  - A `wait` for queued agents returns at once, naming the idle agents that hold slots, when no
    other agent is running that could cancel one: only the waiter can free a slot for them.
  - An idle agent that is only waiting on a schedule holds no slot.
  - `/max-agents N` changes the cap for the running swarm and is kept across restarts.
    `/max-agents default` returns to the flag.
- **Model calls.** `--max-concurrent-calls N` (32, `0` = unlimited) caps model calls in flight
  across the swarm, separately from `--max-agents`; extra calls queue oldest first instead of
  tripping provider throttling.
  - A waiting agent says which cap is full: "queued for a model slot (8/8 calls in flight;
    /max-calls to raise)", in its log line, in `/status` and in `list_agents`. `/status` also
    starts with both caps and their usage, e.g. `agents 3/100 · calls 8/8`.
  - `/max-calls N` changes the cap for the running swarm and is kept across restarts, ahead of the
    flag, like `/max-agents`; `/max-calls default` returns to the flag and `/max-calls 0` lifts
    it. Raising it lets queued calls through at once; lowering it cancels nothing in flight.
  - Why 32: a call holds a slot for its whole stream (seconds to minutes), so the cap bounds
    simultaneous streams, not request rate; throttling is answered by the backoff in `retry.rs`.
    32 never queues the default 16 agents (each has at most one call in flight, plus side calls)
    and still bounds a 100-agent swarm to a third of its agents. Lower it with `/max-calls` if
    your account throttles.
- **Model errors.**
  - Throttling, server errors (5xx, Bedrock's `InternalServerException` and the like) and dropped
    connections are retried with backoff, up to 10 attempts (about eight minutes). A rejected
    request is not retried.
  - If retrying doesn't help, the turn fails and the parent gets an `error` message. A goal agent
    tries again (15 s, doubling to 10 min). Of a streak of failed turns it reports the first
    (adding that it retries on its own), then the 8th, 16th, 32nd… in a row (about 26 min, 1h45m
    and 4h25m in) as `Still failing: 8 turns in a row over 26m. … Last error: …`, so a parent can
    tell a retry loop that went quiet from one that recovered without being woken on every retry.
    The judge sees only replies, not failed turns.
  - The next message picks up where the failed call stopped.
- **Host load.** New work waits while the host is overloaded: its 1-minute load average is above
  `--max-load N` (default twice the CPUs; `0` turns it off), or its available memory
  (`MemAvailable`) is below `--min-free-mem SIZE` (default 5% of its memory, at most 8 GB; `0`
  turns it off).
  - What waits: a new model call (before it queues for a model slot), a background job's start
    (`run_background`, land jobs too) and an agent's `spawn_agent`. Nothing already running is
    held or slowed: a build in flight keeps going, and the user's own spawns are not held. Nor is
    the priority lane, below.
  - A held agent says why, like a queued call: `waiting: host load 180/128 (/max-load to
    change)` in its log line, `/status` and `list_agents`; `/status` starts with
    `agents 3/100 · calls 8/8 · load 45/128`. Holds are logged as `host_held` in `events.jsonl`.
  - The host is read from `/proc` at most once a second. A held caller looks again each second,
    and every fresh reading (and every `/max-load` or `/min-free-mem` change) wakes all held
    callers, so one reading releases them all; off Linux nothing is held.
  - `/max-load N` changes the limit for the running swarm, `/max-load off` lifts it and
    `/max-load default` returns to the flag; `/min-free-mem SIZE|off|default` does the same for
    memory. Both are kept in `swarm.json` across restarts, ahead of the flags, like `/max-calls`.
- **Priority lane.** Root agents, and the agents named with `--priority AGENT` (repeatable; an id
  or name, as `send_message` takes it, so a name can be given before that agent is spawned), never
  wait on the host or the caps; for an agent that relays the user's messages, a hold makes the
  user wait.
  - Their model calls skip the host load and memory holds and never wait for a model slot. They
    still count toward `--max-concurrent-calls`: the cap keeps bounding everyone else, and a
    priority call takes a slot from the queue rather than stacking new streams on a throttled
    account unseen; only priority calls can push it past the cap (`calls 9/8`).
  - They are never `queued` at `--max-agents` and start at once when added; they hold a slot
    like any agent. Their background jobs and spawns skip the host hold too, but worktree spawns
    and lands still wait while disk space is low: a priority agent can still fill the disk.
  - `/status` and `list_agents` mark them `priority`.
  - `/priority AGENT…` replaces the lane (every agent named must exist; it is kept by id),
    `/priority off` leaves only the roots, `/priority default` returns to the flags and
    `/priority` shows it. Kept in `swarm.json` across restarts, ahead of the flags, like
    `/max-calls`.
- **Builds.** Agents' shell commands and background jobs (land checks included) get
  `CARGO_BUILD_JOBS` set to `--build-jobs N` (default a quarter of the CPUs, 1 to 16; `0` sets
  none) unless the swarm's own environment sets it; without it every cargo build starts a
  compiler per CPU. Like any `CARGO_BUILD_JOBS`, it overrides `build.jobs` in cargo's config
  files; `--build-jobs 0` keeps theirs. It only bounds each build; to bound all of them
  together, cap compilers host-wide:
  - cargo runs every compiler through `build.rustc-wrapper` from `~/.cargo/config.toml`
    (`[build]` / `rustc-wrapper = "/path/to/rustc-slot"`) with the real `rustc` and its
    arguments as `"$@"`.
  - A wrapper that takes one of N slots before it execs makes N the host-wide limit, whichever
    swarm, worktree or terminal the build came from. With `flock` that is a few lines of `sh`:
    loop over lock files `0..N` under a shared directory, `exec 9>"$dir/$i"`, and once
    `flock -n 9` succeeds `exec "$@"` (the inherited descriptor holds the slot until that
    compiler exits); when none is free, sleep a moment and try again.
  - Pick N below the CPU count: linkers, build scripts and test binaries run outside the
    wrapper. cargo's quick probes (`rustc -vV`, `--print`) come through it too; a wrapper may
    exec those at once.
- **Tokens.** `--budget-tokens N` (e.g. `5M`) caps the swarm's input + output tokens.
  - Once spent, model calls fail with a "budget exhausted" error that travels up to you.
    Each agent reports it once: mail that wakes it again (a schedule firing) is refused without
    another error until the budget is raised.
  - `/budget N` raises it and tells every stopped agent to continue. `/budget off` lifts it and
    `/budget` shows usage.
  - The budget is saved with the run. On a resume an explicit new flag wins; otherwise the
    budget in force (a `/budget` change included) stays. A flag is new when it differs from the
    last `--budget-tokens` given, which a start without the flag does not forget, so a supervisor
    replaying its arguments leaves a `/budget` change in place. To reset it to the flag's value
    again, use `/budget N`.
- **Per-agent tokens.** `spawn_agent`'s `budget_tokens` bounds one agent, e.g. a worker the
  spawner does not fully trust. It is saved with the agent and `list_agents` shows
  `budget: 1.2M of 2M tokens used`.
  - It counts the agent's own calls and its descendants' (input + output, as the swarm budget
    does), so a worker cannot get past it by spawning helpers. Once spent, model calls are
    refused for the agent and all of them, through the same gate as the swarm budget: the agent
    goes idle as `errored: <budget exhausted …>`, and its parent gets one error mail naming the
    agent, its usage and how to raise it. A descendant stopped by it reports to its own parent
    the same way, once.
  - The parent raises it with `send_message`'s `budget_tokens` (any ancestor, or the user, may;
    the agent itself may not). Agents it had stopped are told to continue; the message goes
    with that note, so one turn follows. A later `send_message` with a lower value lowers it
    again.
- **Process memory.** The binary allocates with jemalloc, whose background threads return freed
  pages to the system within seconds, so the transient conversation copies of model calls and
  saves do not pin each thread's arena at its high-water mark.
- **State saves coalesce.** Every model call asks for a save of `swarm.json`, and a save builds its
  state only once it holds the write lock, skipping it when a write that began after the request
  has covered it. Saves that each queued a copy of the whole swarm held 1.2 GB behind a burst of
  86 agents answering at once (680 waiting saves) and left the heap fragmented: 2.5 GB resident
  after the burst, 0.8 GB with coalescing.
- **Keep-warm holds no conversation.** A refresh is built from the agent's own history when it
  is sent, so the waiting agent's conversation is in memory once: the request keeps a digest of
  what the call sent and the per-call blocks after the cache point (injected text), and a refresh
  is sent only while the history still hashes to that digest (an edit by compaction or pruning
  ends them, the agent's next call would not read that prefix). A request that history cannot
  rebuild (a hook edited an earlier message for one call) is kept as a copy. Ten agents with 5 MB
  conversations in a tool call: 211 MB resident with a copy per request, 160 MB without
  (`tests/keep_warm_memory.rs`, `KEEP_WARM_AGENTS=10 KEEP_WARM_MB=5 KEEP_WARM_ONLY=on`).

## Cost

- **Estimates, not a bill.** Spend is estimated from reported usage, tracked per agent and model, so
  a `/model` switch is priced per model.
- **Prices.**
  - A built-in table holds Claude list prices per 1M tokens by tier (opus, sonnet, haiku, plus older
    versions) and OpenAI's GPT-5.4 to GPT-6.1 (Bedrock's global prices; GPT-6.2 Astra borrows
    GPT-6 Astra's until its own is listed). Cache reads cost 0.1× the input price (0.05× on
    GPT-6.1) and cache writes 1.25× (2× with `--cache-ttl 1h`; OpenAI's are 1.25× on GPT-5.6 and
    later whatever the TTL, and plain input, no fee, on GPT-5.5 and earlier).
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
  with the 5-minute cache; `--no-keep-warm` turns it off. The interval follows the model's cache
  lifetime (`cost::cache_lifetime`, next to the prices): 4.5 minutes for the 5-minute cache,
  27 for OpenAI's GPT-5.6 and later, which keep it at least 30 minutes (earlier GPT versions
  and gpt-oss state no lifetime and keep the 5-minute assumption), so a tool call under half an
  hour sends none there.
  - An idle agent whose next schedule fires in less than 11 refreshes (about 50 minutes) is kept
    warm the same way until it fires, e.g. a 30-minute poller: 6 refreshes cost 0.6 of its prefix
    against 1.15 more for rewriting it instead of reading it (1.25× against 0.1×): about half.
    The limit is priced per model (`--prices` cache read and write); a poller further out is
    left to expire, where `--cache-ttl 1h` is the cheaper answer. The limit counts refreshes at
    the model's interval, so on a 30-minute cache it reaches about six times as far.
- **OpenAI models on Bedrock** (GPT-6.x Astra) cache prompt prefixes on their own, without cache
  points; `--cache-ttl` does not apply, and their cache lifetime and write prices are above. Bedrock
  routes their requests by a hash of the prompt prefix and spills a prefix sent more than ~15
  times a minute to other servers, where the cache misses; agents share the swarm's system
  prompt. With a state directory, each agent's calls (keep-warm refreshes included) carry
  `prompt_cache_key` `strands-<agent id>` (on Converse and on Mantle's Responses API alike), so
  they stay on the server that holds its cache.
- **Where it shows.**
  - `/cost` (or `--run NAME --cost`) shows per agent: input, output, cache-read and cache-write
    tokens, the model, an estimated `$`, and the median input per call and cache hit rate over the
    last 50 calls. Below the total, one line gives the swarm's cache hit rate over the last hour's
    calls and since the run started (which keeps every early miss). Below the table, side calls
    get their own rows by purpose: the goal `judge`,
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
    and `CARGO_BUILD_JOBS` ([Limits](#limits)) included. Each runs in a session of its own, so neither a reload nor Ctrl-C stops it.
  - Each job is `<id>.json`, `<id>.log`, `<id>.exit` and, once its owner was told, `<id>.reported`
    in `<run>/jobs` (a temp directory without `--run`).
  - The swarm polls jobs and mails the owner when one ends, also for jobs started before a restart.
    A job whose process vanished without an exit code (a reboot) is reported as such. When the
    log is empty and the command redirected its output (`> build.log 2>&1`), the report names the
    files it redirected to instead of only "(no output)".
  - A killed job is not reported. Cancelling an agent kills its whole subtree's jobs.
  - A `shell` call that sleeps 30 s or more (`sleep 240; tail job.log`) while its agent has a job
    running gets one line appended to its result: the job is still running, and `wait` with
    `from: ["jobs"]` returns as soon as it ends, whereas a sleep can't be cut short by mail or a reload.
  - A running job whose log has not grown for 30 minutes (`SwarmConfig::job_quiet`) gets its
    owner one FYI note (`job j3 (gates) has written nothing for 30m; still running: …`), saved as
    `<id>.stalled` so a restart does not repeat it. Nothing is killed. A land job's checks count
    as writing while they print (the land process touches its log's mtime; their output stays out of
    the log).
  - A running job's log over 32 MB is cut to its last 1 MB. Only the newest 200 reported jobs keep
    their files.
- **Schedules.**
  - `schedule {message, to?, every?: "30m", at?: "in 2h" | "2026-11-17T09:00Z", id?}` posts
    `[schedule sN, 2026-11-17 09:00 UTC] message` (kind `schedule`) at the given times.
  - `{cancel: id}` removes one and `{}` lists them. Repeats are at least a minute apart.
  - An agent that polls something schedules itself and ends its turn instead of a long `wait`.
  - From its second fire on, a repeat adds what the receiver spent since the last one:
    `(You used 330.0k tokens, est. $0.21, since this schedule last fired 2m ago: about $6.30/h at
    this interval.)`, so a polling agent sees what its loop costs.
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
  - The scratch worktree is always removed, and the caller's branch is never modified. A stopped
    land job (killed with `jobs`, or its agent cancelled) kills the check in flight with its
    children. What a job killed outright (`kill -9`, a reboot) leaves in `jobs/land` goes once a
    minute old, on the next resume, `/gc` or `--gc` that finds no land job running (a `gc` event
    for `land`).
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
| `/cost` | Estimated spend (see [Cost](#cost)); its intro and footnote wrap, its table is clipped on narrow terminals. |
| `/budget`, `/max-agents`, `/max-calls`, `/max-load`, `/min-free-mem`, `/priority` | See [Limits](#limits). |
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
- **Last progress.** `list_agents` and `/status` add `last progress 3h ago` to a *running* agent
  once its newest sign of progress is over 15 minutes old: a model reply, a tool call that ended
  successfully, or a commit at its branch's tip. It is kept in memory and saved with the agent
  (`progress`), so a restart does not make a stuck agent look fresh. A tool call still in flight
  is not progress, so a long build shows the note while it runs.
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
  - Two live agents' unmerged branches that change the same files (`git diff --name-only
    <base>...<branch>`) are trial-merged with each other, so the integrator can sequence work
    that would otherwise show `merges cleanly` twice and conflict on the second landing:
    `…; conflicts with a4 (src/a.rs); overlaps with a6 (README.md)`. A conflict only in files one
    of them changes is with the base and is left to the merge outlook above. Diffs and trial
    merges are cached by commit, so only a branch that moved is looked at again; pairs without a
    shared file cost nothing; at most 45 new trial merges run per refresh (later pairs read
    `overlaps`).
  - It refreshes while open; git runs in the background and is cached for 5 s.
  - Programmatically: `Swarm::status`.

**Agent tree.**

- Each agent is one line: status, name, model, input/output tokens and how close its last call
  came to the compaction point, e.g. `● parser opus-5-5 1.2M/34.0k 34% a3`.
- The status spins while the agent works (a model call or a tool call). A still `◌` means it
  waits for a model-call slot or for a pending reload, a still `⧗` that it is blocked in `wait`.
- Cancelled agents without a live descendant are hidden (the header counts them); F3 shows them.
- With an empty input, ← folds the selected agent: its descendants leave the list, and its row
  counts them and those of them that work or failed, e.g. `● parser ▸ (+12, 2 running, 1 errored)`.
  → unfolds it. ← on an agent that shows no children selects its parent, → on an unfolded one
  its first child, as in a file tree. ↑/↓ skip folded agents, which still get mail and log
  events. Folds last for the session, not across restarts or reloads.
- Alt-←/→ narrows or widens the tree. Narrow it past its minimum to hide it.

**Header.**

- It counts the agents that work as `3 running` and those that wait (`◌`, `⧗`) as `2 waiting`.
- It totals tokens as `in 84.8M (97% cached last hour) · out 291k · est. $12.30`: the cached
  share is over the last hour's calls (up to each agent's last 50), or since the run started when
  there were none.
- Input counts cache reads and writes, as the budget does.

**Input.**

- The input box edits like a shell line: ←/→ (when it has text), Home/End (Ctrl-A/E), Ctrl-←/→
  by word, Delete, Ctrl-W/Ctrl-U (Alt-Backspace). Tab completes the highlighted command.
- Alt-Enter (or Shift-Enter where reported) inserts a new line.
- A long line wraps (at a space where one fits) instead of scrolling sideways; the box grows to
  6 rows, then scrolls with the cursor.
- ↑/↓ go through sent messages while the input holds text (Ctrl-P/N from an empty one). On an
  empty input they select agents.

**Images.**

- Images from tool results (e.g. `read` on a png) and from `/image` show as an
  `[image: path WxH]` line.
- Below it comes a thumbnail of at most 12 rows when the terminal answers with kitty, iTerm2 or
  sixel graphics. Inside tmux (so under `swarmctl`), and on terminals without graphics, there is
  no thumbnail: unicode halfblocks would blur it beyond reading. Open the path to see the image.
- Only the newest 4 stay decoded. The mailbox and journal keep only the path.
- Before an image reaches a model (a tool result from `read` or MCP, or an `/image`
  attachment), it is downscaled to at most 2576 px on the long edge and 3.75 MB, the limits of
  Claude and Bedrock. A model error that names an image limit replaces the images over it in the
  agent's history with `[image removed: over the model's size limit]`, and the call is retried
  once. Otherwise every later call would resend the image and fail.

**Log and copying.**

- A tool line whose summary is one long word (a path) breaks the word at the edge, live and
  restored, rather than leaving `✗ read` alone above it.
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
- **Saves.** An agent's conversation is saved after each message, as compact JSON serialized
  straight from the agent into one buffer the size of the file (no copy of the conversation),
  then written to a temp file, fsynced and renamed over the old one. A save that would change
  nothing but the snapshot's `createdAt` (the end of a turn, after the reply was saved) is
  skipped: the agent's session manager keeps a hash of what it last wrote, and the swarm's own
  saves (mail folded into an open turn, a restored tool batch) go through it too. `swarm.json`
  is written the same way, compact and from one buffer, and not at all when it would hold the
  same bytes as the file already does.
- **Bounds.** A run dir grows with the live agents' work since the last resume and with unreported
  jobs, not with the run's age:
  - `journal.jsonl` and `events.jsonl` are rotated to `<file>.1` at 16 MB, so each takes at most
    32 MB (a busy 250-agent swarm writes about 13 MB of events a day); `swarm.log` at 8 MB.
  - In `jobs/`, a running job's log is cut to its last 1 MB past 32 MB, and only the newest 200
    reported jobs keep their files (see **Background jobs**).
  - The stash (`sessions/<agent>/stash`) holds every tool result and text block of an agent's
    conversation, for `retrieve_context`. A cancelled agent's goes once its last turn has ended
    (or on resume). On each resume, before any agent runs, a live agent's entries are deleted
    unless a file of its saved session (the conversation with its summaries, an open tool batch)
    still names their id, so whatever the agent can still name stays retrievable. A live agent's
    stash is pruned the same way while the swarm runs: when one of its turns ends at least an
    hour (`SwarmConfig::stash_prune_every`) after its last pruning, its conversation is saved
    and the unnamed entries go before its next turn can start. That cannot race the stash's
    writes, which the SDK makes only while it builds the agent or runs one of its turns. A
    session that is missing or does not read as JSON, or a save that fails, keeps the whole
    stash. Each pruning is logged as `stash_pruned` (`entries`, `bytes`, and `id` for one
    agent's).
  - `mcp-tools.json` is rewritten only when a server lists different tools.
- **Journal.**
  - A persistent run keeps `journal.jsonl` with every message to and from the user, and
    `events.jsonl` with compact lifecycle events: spawns, status changes, tool calls and their
    ends, each own model call's tokens (`model_call`: model, `input` with cache reads and writes
    included, `output`, `cache_read`, `cache_write`, so hit rates can be read per agent and hour),
    model switches, compactions (`before`/`after` tokens) and failed ones, side calls
    (`side_call`: purpose, model and tokens of each judge or summarizer call), keep-warm
    refreshes, schedules, jobs, budget and agent-cap changes, resumes, reloads, version
    `changes`, shutdowns, disk reclaimed (`gc`) and stash entries pruned (`stash_pruned`).
  - Both are JSON lines with a UTC `ts`, written by a background thread and rotated to `<file>.1`
    at 16 MB.
- **Log.** Warnings and errors (or whatever `RUST_LOG` selects) go to `<run dir>/swarm.log` under
  the TUI (`$TMPDIR/strands-swarm-<pid>.log` without `--run`), and to stderr otherwise. When a
  process opens a log that has grown past 8 MB, it is first moved to `swarm.log.1`.
- **Terminal.** Every child the swarm starts (shell commands, jobs, git, land gates, MCP servers)
  runs in a session of its own, without a controlling terminal. Nothing it writes to `/dev/tty`
  reaches the TUI, and a prompt (git credentials, an ssh host key) fails at once instead of hanging.
  On Linux each is started through util-linux's `setsid` (by absolute path) rather than a
  `pre_exec` hook, so std spawns it with `posix_spawn` instead of `fork()`ing the whole swarm,
  which stalls every thread (the TUI too) for as long as copying its page tables takes. Without
  a `setsid` binary it falls back to the hook.
- **Disk.**
  - Once a cancelled worktree agent and its subtree have stopped (or on resume), its private target
    dir is deleted.
  - Its worktree is removed with `git worktree remove` unless it has uncommitted changes (the
    reason names the first three, e.g. `uncommitted changes: ?? tests/__pycache__/`), and so are
    `<repo>.swarm/<run>` and `<repo>.swarm` once that leaves them empty. A worktree with another
    branch checked out goes only once the base has all of that branch (which stays), else it is
    kept (`on branch swarm/r/a3-fix: 1 commit not on main`). A detached one is kept.
  - Its branch `swarm/<run>/<agent>` is deleted once no worktree has it checked out and the base
    `/status` uses has every commit on it: reachable (`merged`) or as a patch-equivalent copy
    (`all applied`, per `git cherry` or the landing record; a merge commit never counts as
    applied unless landed). Otherwise it stays, e.g. `kept branch swarm/r/a3 (2 commits not on
    main)`, until a later pass (resume, `/gc`, `--gc`) finds it merged. The ref is deleted only
    if it still points at the commit checked.
  - `cancel_agent` waits up to 10 s for this and its result says what was removed and kept.
    `list_agents` with `include_cancelled` shows it later: `cwd … (removed)`, `branch … (deleted)`
    or `branch … (kept: 2 not on main, merges cleanly)`.
  - A run without `--run` cannot be resumed, so on exit every worktree agent is treated as
    cancelled.
  - Nothing outside those exact paths and that branch is touched, and each removal is logged as a
    `gc` event. A pass after a cancel or on resume logs an agent only when its outcome changed,
    not a worktree kept for the same reason on every restart.
  - Only the swarm's own `$CARGO_TARGET_DIR-worktrees/<run>/<agent>` dirs are reclaimed. An agent
    that points `CARGO_TARGET_DIR` somewhere else owns that dir and must clean it itself.
  - On start and then every minute (every 5 s while space is low) the swarm checks free space
    (`statvfs`) on the filesystems of the run dir, the agents' directories and the cargo target
    dirs. Below `--min-free-disk` (default `20G`, `0` turns it off) it mails the live root agents
    (or, with none, you) once, from `disk`, with the free space and the biggest target dirs (per
    agent, and the land checks' shared one); the TUI shows the first line in its footer. It warns
    again only after the space has recovered to 10% above the threshold, restarts included
    (`<run dir>/disk-warned`).
  - Meanwhile agents' worktree spawns and land jobs wait, shown like a load hold: `waiting: disk
    space low: 3.0 GB free on the filesystem of /x (below 20 GB; cancel finished worktree agents
    to free some)`. They go on once space has recovered to 10% above the threshold. Spawns
    without a worktree, other jobs and the user's own spawns are not held.
  - **Reserve.** A persistent run keeps 1 GB allocated in `<run dir>/disk-reserve` while space is
    fine. The first save of `swarm.json` or of a conversation that fails with a full disk deletes
    it and tries again, so the run can still be saved and resumed; the guard recreates it once
    space has recovered.
  - Finished worktree agents are never cancelled for you (one may be kept on purpose), but a live
    one that is idle, runs no background job, has a clean worktree and nothing on its branches
    the base lacks is flagged in `list_agents`: `…, idle with nothing left to merge: cancel to
    reclaim its worktree and 12 GB target dir`. When the disk guard warns, it first deletes the
    private target dirs of such agents (never their worktree, branch or conversation; the next
    build recreates it) and says so in the warning: `Deleted the target dirs of idle agents with
    nothing left to merge (…): 12 GB done (a4)`. While space stays low it does so again every
    minute, for agents that finished since. Off with `SwarmConfig::reclaim_disk`.
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
    tools). The file is rewritten only when a server lists tools that differ from its entry.
  - A server connects when an agent enables one of its tools, or on the first call (60 s timeout),
    and reconnects if it died.
  - Agents start without MCP tools. `find_tools` (whose description lists the servers) searches the
    catalog and enables tools from the next call on. Enabled tools come back on resume.
  - `--mcp-eager` registers every tool on every agent, `--mcp-deny` hides tools by name, and
    `--list-tools` prints what is allowed and denied.
  - `--mcp-deny-except 'REGEX=AGENT[,AGENT…]'` (repeatable) denies matching tools to every agent
    but the named ones, given by id or name as `send_message` takes them. Others neither find nor
    enable them (`find_tools` says whom they are reserved for), and a resumed agent that had one
    enabled loses it. A server acting with your own token can thus write only through one agent:
    `--mcp-deny-except 'slack-mcp_(post|reply|upload|add_reaction|update|delete|send|create)=slack-bridge'`.
    A name belongs to whichever agent claims it first, so give an id to pin a rule to an
    agent that exists already. Nothing is saved: a resumed run gets the rule from its arguments.
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
`start` saves the environment the swarm needs (`PATH`, `AWS_*`, `STRANDS_SWARM_AWS_*`, cargo's)
to `<run dir>/env` (mode 600), and every restart sources it. That file alone sets the AWS
variables: those the tmux server has from the shell that started it are dropped first.

**Restart.** `swarmctl restart NAME` makes the supervisor re-read `args`, `env` and the swarmctl
script itself without losing the tmux session (`stop` then `resume` kills it, and with it the view
of anyone attached; `stop` says when clients will be detached).

- The tmux server does the work (`tmux run-shell -b`, logging to `<run dir>/restart.log`), so it
  completes when called from inside the swarm, say a background job that dies with the swarm.
- It sends Ctrl-C as `stop` does and waits up to 30 s for the supervisor to exit, with
  `remain-on-exit` on so the pane stays. It removes the `stop` file, `respawn-pane -k`s the
  supervisor in that pane and unsets `remain-on-exit`.
- The session, the pane and attached clients stay; the clients see the new TUI. A run that is not
  running is resumed instead.

**Failures.**

- A swarm that hangs exits instead, so the supervisor sees it: 70 when a deadlock is detected, 71
  when it is not up (first TUI frame drawn, or headless running) within 180 s.
- A new binary is on trial for 10 minutes (`SWARMCTL_TRIAL_WINDOW`) from when it first runs
  after its reload or `swarmctl start` (`bin/started` names the binary started last), even if a
  stop came in between. If it fails (non-zero exit, a signal, 70, 71) on trial, the supervisor puts
  the previous binary back and keeps the failed one as `bin/failed`. The previous binary is the
  one that ran: a reload that supersedes a pending one replaces only `bin/current`. A binary that
  ran longer is restarted, never rolled back, and a rolled-back one is not rolled back again.
- SIGKILL and SIGTERM come from outside (the swarm never sends them itself), so they are no
  verdict on a binary: it is restarted and stays on trial. SIGKILL by the OOM killer (the
  `oom_kill` count in the cgroup's `memory.events` grew during the run), or when that count
  cannot be read, is a failure like any other.
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
swarmctl restart NAME       # stop, then start again in the same tmux session and pane
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

**Prompt caches across a reload.**

- Providers cache tools, then the system prompt, then messages, so a restart keeps every agent's
  cache only if the new binary sends the same fixed prefix; otherwise every busy agent rewrites
  its whole prefix on its next call.
- `NEW --print-prefix` prints a root agent's system prompt and tool specs on a canonical setup
  (the one the fixed-prefix budget test pins; `--allow-land` adds `land`). `reload_swarm`
  compares the running binary's with the new one's and says in its result whether caches
  survive, or which tools and prompt sections change, how many agents have a warm cache (a call
  or keep-warm refresh in the last 5 minutes) and their last calls' tokens and estimated extra
  cost. `check: true` reports only that, without self-testing or installing, so prefix changes
  can be batched into one reload.

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
