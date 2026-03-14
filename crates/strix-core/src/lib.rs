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
pub mod particle_nav;
pub mod regime;
pub mod state;
pub mod temporal;
pub mod threat_tracker;
pub mod uncertainty;

// Re-export primary types at crate root for convenience.
pub use state::{
    DroneState, FleetState, Observation, Regime, SensorConfig, StateError, ThreatRegime,
    ThreatState,
};

pub use particle_nav::ParticleNavFilter;
pub use temporal::TemporalManager;
pub use threat_tracker::ThreatTracker;

// ---------------------------------------------------------------------------
// PyO3 module registration (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
pub mod python;

// ---------------------------------------------------------------------------
// PyO3 module registration (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
mod _pymodule {
    use pyo3::prelude::*;

    /// Python module entry point for `strix_core`.
    #[pymodule]
    fn _strix_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;

        // Types
        m.add_class::<super::python::PyRegime>()?;
        m.add_class::<super::python::PySensorConfig>()?;
        m.add_class::<super::python::PyDroneState>()?;
        m.add_class::<super::python::PyParticleNavFilter>()?;

        // Functions
        m.add_function(wrap_pyfunction!(super::python::py_detect_jamming, m)?)?;
        m.add_function(wrap_pyfunction!(super::python::py_detect_regime, m)?)?;

        Ok(())
    }
}
