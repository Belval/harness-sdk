//! Strands swarm: a model-driven agent swarm on top of [`strands_harness`].
//!
//! There is no planner or fixed topology. Every agent is a harness agent that also carries the
//! swarm tools ([`tools`]) and decides for itself whether to split work, how deep to go, and how
//! to coordinate.
//!
//! Each agent is an actor: it runs a turn, reports its final reply to its parent, then sleeps until
//! a message arrives. Messages that arrive mid-turn are appended to the agent's next tool result,
//! so nothing is interrupted and nothing is lost. An agent can ask for its own git worktree, which
//! keeps parallel edits apart; merging is left to the agents, using plain git (or `land`).
//!
//! The runtime enforces only resource limits (live agents, concurrent model calls, a token budget)
//! and cancellation, which cascades down the spawn tree.

pub mod catalog;
pub mod compaction;
pub mod cost;
pub mod dashboard;
pub mod factory;
pub mod files;
pub mod gc;
pub mod images;
pub mod jobs;
pub mod journal;
pub mod land;
pub mod logging;
pub mod mcp;
pub mod memory;
pub mod names;
pub mod orphans;
pub mod pruning;
pub mod reserve;
pub mod resume;
pub mod retry;
pub mod runtime;
pub mod schedule;
pub mod selftest;
pub mod shell_text;
pub mod skills;
pub mod stash_gc;
pub mod state;
pub mod status;
pub mod text;
pub mod tools;
pub mod tui;
pub mod version;
pub mod watchdog;
pub mod workdir;
pub mod worktree;

pub use catalog::ToolCatalog;
pub use factory::{
    agent_session_manager, bounded_goal, harness_factory, harness_model_resolver, swarm_aws_settings,
    HarnessFactoryOptions,
};
pub use mcp::{load_shared_mcp, McpOptions, SharedMcp};
pub use memory::Memory;
pub use runtime::{
    AgentFactory, AgentInfo, AgentSpec, AgentStatus, LandRequest, Mail, ModelResolver, ReloadWhen,
    ScheduleRequest, SpawnRequest, Swarm, SwarmConfig, SwarmEvent, WaitOutcome, FYI, GOAL_MAX_ATTEMPTS,
    RELOAD_EXIT_CODE, SILENT_REPLY, SUPERVISOR_NOTICE, SUPERVISOR_SENDER, USER,
};
pub use skills::{discover_skills, Skills};
