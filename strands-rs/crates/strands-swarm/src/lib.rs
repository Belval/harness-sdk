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

pub mod catalog;
pub mod factory;
pub mod mcp;
pub mod names;
pub mod runtime;
pub mod state;
pub mod tools;
pub mod tui;
pub mod worktree;

pub use catalog::ToolCatalog;
pub use factory::{agent_session_manager, harness_factory, HarnessFactoryOptions};
pub use mcp::{load_shared_mcp, SharedMcp};
pub use runtime::{
    AgentFactory, AgentInfo, AgentSpec, AgentStatus, Mail, SpawnRequest, Swarm, SwarmConfig, SwarmEvent,
    RELOAD_EXIT_CODE, USER,
};
