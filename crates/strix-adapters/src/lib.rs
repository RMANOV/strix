//! # strix-adapters — Puppet Master Layer
//!
//! Platform adapter abstraction for the STRIX drone swarm orchestrator.
//!
//! Provides a unified [`PlatformAdapter`](traits::PlatformAdapter) trait that
//! lets the rest of the system control any drone platform — MAVLink autopilots,
//! ROS2 robots, or a built-in physics simulator — through a single interface.
//!
//! ## Modules
//!
//! - [`traits`]: Core trait + shared types (Waypoint, Telemetry, RichTelemetry, Action, …)
//! - [`command`]: Command lifecycle types (CommandId, CommandAcceptance, CommandOutcome)
//! - [`mavlink`]: MAVLink adapter stub (PX4 / ArduPilot)
//! - [`ros2`]: ROS2 adapter stub (UGV / USV)
//! - [`simulator`]: Fully working simulator for testing

pub mod command;
pub mod mavlink;
pub mod ros2;
pub mod simulator;
pub mod traits;
