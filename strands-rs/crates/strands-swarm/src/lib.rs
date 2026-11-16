//! Strands swarm: a model-driven agent swarm on top of [`strands_harness`].
//!
//! There is no planner, scheduler or fixed topology. Every agent is a harness agent that also
//! carries five tools — `spawn_agent`, `send_message`, `wait`, `list_agents`, `cancel_agent` — and
//! decides for itself whether to split work, how deep to go, and how to coordinate.
//!
//! Each agent is an actor: it runs a turn, reports its final reply to its parent, then sleeps until
//! a message arrives. Messages that arrive mid-turn are appended to the agent's next tool result,
//! so nothing is interrupted and nothing is lost. An agent can ask for its own git worktree, which
//! keeps parallel edits apart; merging is left to the agents, using plain git.
//!
//! The runtime enforces only resource limits (live agent count) and cancellation, which cascades
//! down the spawn tree.

pub mod budget;
pub mod catalog;
pub mod compaction;
pub mod factory;
pub mod gc;
pub mod images;
pub mod journal;
pub mod mcp;
pub mod memory;
pub mod names;
pub mod pruning;
pub mod runtime;
pub mod schedule;
pub mod selftest;
pub mod skills;
pub mod state;
pub mod tools;
pub mod tui;
pub mod version;
pub mod watchdog;
pub mod worktree;

pub use catalog::ToolCatalog;
pub use factory::{
    agent_session_manager, bounded_goal, harness_factory, harness_model_resolver, HarnessFactoryOptions,
};
pub use mcp::{load_shared_mcp, SharedMcp};
pub use memory::Memory;
pub use runtime::{
    AgentFactory, AgentInfo, AgentSpec, AgentStatus, Mail, ModelResolver, ReloadWhen, ScheduleRequest,
    SpawnRequest, Swarm, SwarmConfig, SwarmEvent, WaitOutcome, GOAL_MAX_ATTEMPTS, RELOAD_EXIT_CODE, USER,
};
pub use skills::{discover_skills, Skills};
