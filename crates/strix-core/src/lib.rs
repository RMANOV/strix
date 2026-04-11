//! **strix-core** — Particle filter + Kalman + CUSUM + Regime detection
//! for GPS-denied drone navigation.
//!
//! This crate is the computational heart of STRIX, a drone swarm
//! orchestrator.  It adapts the 2D trading particle filter
//! (`particle-filter-rs`) to a 6D navigation problem
//! `[x, y, z, vx, vy, vz]` with three battlespace regimes:
//!
//! | Regime  | Trading Analogue | Dynamics                                  |
//! |---------|------------------|-------------------------------------------|
//! | PATROL  | RANGE            | Mean-reverting velocity, hold pattern      |
//! | ENGAGE  | TREND            | Velocity tracks threat bearing, beta=0.3   |
//! | EVADE   | PANIC            | High-noise random walk, rapid evasion      |
//!
//! Key innovations:
//! - **Dual Particle Filter**: friendly navigation + enemy tracking
//! - **Multi-Horizon Temporal Manager**: H1(0.1s), H2(5s), H3(60s)
//! - **CUSUM anomaly detection**: jamming, threat shifts, environment
//! - **Uncertainty quantification**: Hurst, volatility, kurtosis

pub mod anomaly;
pub mod cbf;
pub mod ew_response;
pub mod fleet_metrics;
pub mod formation;
pub mod gcbf;
pub mod hysteresis;
pub mod intent;
pub mod particle_common;
pub mod particle_nav;
pub mod regime;
pub mod roe;
pub mod state;
pub mod temporal;
pub mod threat_tracker;
pub mod uncertainty;
pub mod units;

// Re-export primary types at crate root for convenience.
pub use state::{
    DroneState, FleetState, Observation, Regime, SensorConfig, StateError, ThreatRegime,
    ThreatState,
};

pub use particle_nav::ParticleNavFilter;
pub use temporal::TemporalManager;
pub use threat_tracker::ThreatTracker;

// Note: PyO3 wrappers live in the `strix-python` crate (cdylib target).
