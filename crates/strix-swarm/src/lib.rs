//! # strix-swarm — Integration Orchestrator
//!
//! The swarm crate chains all 5 STRIX layers into a unified tick loop:
//!
//! ```text
//! telemetry → particle_filter → regime_detect → auction → gossip → pheromone → trace
//! ```
//!
//! [`SwarmOrchestrator`] is the central struct that owns instances of every
//! subsystem and runs them in the correct sequence each tick.

pub mod convert;
pub mod tick;

pub use tick::{SwarmConfig, SwarmDecision, SwarmOrchestrator};
