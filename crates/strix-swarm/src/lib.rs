//! # strix-swarm - Integration Orchestrator
//!
//! The swarm crate chains all 5 STRIX layers into a unified tick loop:
//!
//! ```text
//! telemetry -> particle_filter -> regime_detect -> auction -> gossip -> pheromone -> trace
//! ```
//!
//! [`SwarmOrchestrator`] is the central struct that owns instances of every
//! subsystem and runs them in the correct sequence each tick.

pub mod convert;
pub mod criticality;
pub mod fear_adapter;
pub mod tick;

pub use criticality::{
    CriticalityAdjustment, CriticalityConfig, CriticalityScheduler, CriticalitySignals,
};
#[cfg(feature = "phi-sim")]
pub use fear_adapter::{DroneFearInputs, SwarmFearAdapter};
pub use tick::{SwarmConfig, SwarmDecision, SwarmOrchestrator};