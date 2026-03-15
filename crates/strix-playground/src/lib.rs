//! Battlefield simulation playground for Strix drone swarm testing.
//!
//! Provides a fluent DSL for defining combat scenarios, a closed-loop
//! simulation engine, and structured diagnostic output.

pub mod engine;
pub mod playground;
pub mod presets;
pub mod report;
pub mod scenario;
pub mod threats;

pub use engine::Engine;
pub use playground::Playground;
pub use report::BattleReport;
pub use scenario::{Event, ScheduledEvent, ThreatBehavior, ThreatSpec};
pub use threats::ThreatActor;
