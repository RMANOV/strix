//! PyO3 wrapper types for the strix-core API.
//!
//! These types bridge the Rust API (which uses nalgebra/ndarray) to Python
//! using simple lists and tuples. Each wrapper holds the inner Rust type
//! and exposes `#[pymethods]` for Python consumption.
//!
//! ```python
//! from strix._strix_core import ParticleNavFilter, DroneState, Regime
//!
//! f = ParticleNavFilter(n_particles=200, position=[0.0, 0.0, -50.0])
//! pos, vel, probs = f.step(observations=[], threat_bearing=[1.0, 0.0, 0.0], vel_gain=1.0, dt=0.1)
//! ```

use pyo3::prelude::*;

use nalgebra::Vector3;

use crate::anomaly::{self, CusumConfig};
use crate::particle_nav::ParticleNavFilter;
use crate::regime::{self, DetectionConfig, RegimeSignals};
use crate::state::{Observation, Regime, SensorConfig};

// ---------------------------------------------------------------------------
// PyRegime — enum wrapper
// ---------------------------------------------------------------------------

/// Battlespace operating regime: Patrol, Engage, or Evade.
#[pyclass(name = "Regime")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyRegime {
    inner: Regime,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyRegime {
    /// Patrol regime (normal operations).
    #[classattr]
    fn Patrol() -> Self {
        Self {
            inner: Regime::Patrol,
        }
    }

    /// Engage regime (active threat engagement).
    #[classattr]
    fn Engage() -> Self {
        Self {
            inner: Regime::Engage,
        }
    }

    /// Evade regime (threat avoidance).
    #[classattr]
    fn Evade() -> Self {
        Self {
            inner: Regime::Evade,
        }
    }

    /// Create from integer index (0=Patrol, 1=Engage, 2=Evade).
    #[staticmethod]
    fn from_index(idx: u8) -> PyResult<Self> {
        Regime::from_index(idx)
            .map(|r| Self { inner: r })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Get the integer index.
    fn as_index(&self) -> usize {
        self.inner.as_index()
    }

    fn __repr__(&self) -> String {
        format!("Regime.{:?}", self.inner)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl From<Regime> for PyRegime {
    fn from(r: Regime) -> Self {
        Self { inner: r }
    }
}

impl From<PyRegime> for Regime {
    fn from(pr: PyRegime) -> Self {
        pr.inner
    }
}

// ---------------------------------------------------------------------------
// PySensorConfig
// ---------------------------------------------------------------------------

/// Sensor noise configuration for the particle filter.
#[pyclass(name = "SensorConfig")]
#[derive(Debug, Clone)]
pub struct PySensorConfig {
    inner: SensorConfig,
}

#[pymethods]
impl PySensorConfig {
    /// Create with default noise parameters.
    #[new]
    fn new() -> Self {
        Self {
            inner: SensorConfig::default(),
        }
    }

    /// IMU acceleration noise std-dev (m/s^2).
    #[getter]
    fn imu_accel_noise(&self) -> f64 {
        self.inner.imu_accel_noise
    }

    /// Barometer altitude noise std-dev (meters).
    #[getter]
    fn baro_noise(&self) -> f64 {
        self.inner.baro_noise
    }

    fn __repr__(&self) -> String {
        format!(
            "SensorConfig(imu={:.3}, baro={:.3}, mag={:.3})",
            self.inner.imu_accel_noise, self.inner.baro_noise, self.inner.mag_noise
        )
    }
}

// ---------------------------------------------------------------------------
// PyDroneState
// ---------------------------------------------------------------------------

/// 6-DOF drone state: position + velocity + regime.
#[pyclass(name = "DroneState")]
#[derive(Debug, Clone)]
pub struct PyDroneState {
    /// Position [x, y, z] in meters.
    #[pyo3(get)]
    pub position: [f64; 3],
    /// Velocity [vx, vy, vz] in m/s.
    #[pyo3(get)]
    pub velocity: [f64; 3],
    /// Operating regime.
    #[pyo3(get)]
    pub regime: PyRegime,
    /// Drone ID.
    #[pyo3(get)]
    pub drone_id: u32,
}

#[pymethods]
impl PyDroneState {
    #[new]
    fn new(drone_id: u32, position: [f64; 3]) -> Self {
        Self {
            position,
            velocity: [0.0; 3],
            regime: PyRegime::from(Regime::Patrol),
            drone_id,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DroneState(id={}, pos=[{:.1},{:.1},{:.1}], regime={:?})",
            self.drone_id, self.position[0], self.position[1], self.position[2], self.regime.inner
        )
    }
}

// ---------------------------------------------------------------------------
// PyParticleNavFilter
// ---------------------------------------------------------------------------

/// 6D particle navigation filter for GPS-denied navigation.
///
/// Wraps the Rust ParticleNavFilter with a Python-friendly API.
#[pyclass(name = "ParticleNavFilter")]
pub struct PyParticleNavFilter {
    inner: ParticleNavFilter,
    drone_id: u32,
}

#[pymethods]
impl PyParticleNavFilter {
    /// Create a new filter.
    ///
    /// Args:
    ///     n_particles: Number of particles (default 200).
    ///     position: Initial position [x, y, z].
    ///     drone_id: Drone identifier.
    #[new]
    #[pyo3(signature = (n_particles=200, position=[0.0, 0.0, 0.0], drone_id=0))]
    fn new(n_particles: usize, position: [f64; 3], drone_id: u32) -> Self {
        let pos = Vector3::new(position[0], position[1], position[2]);
        Self {
            inner: ParticleNavFilter::new(n_particles, pos),
            drone_id,
        }
    }

    /// Run one predict-update-resample cycle.
    ///
    /// Args:
    ///     observations: List of (type, value) observation tuples.
    ///         Supported types: "barometer" (altitude in m).
    ///     threat_bearing: Unit vector [x,y,z] toward nearest threat.
    ///     vel_gain: Velocity gain for ENGAGE regime (default 1.0).
    ///     dt: Time step in seconds.
    ///
    /// Returns:
    ///     Tuple of (position [x,y,z], velocity [vx,vy,vz], regime_probs [P,E,V]).
    #[pyo3(signature = (observations=vec![], threat_bearing=[0.0, 0.0, 0.0], vel_gain=1.0, dt=0.1))]
    fn step(
        &mut self,
        observations: Vec<(String, f64)>,
        threat_bearing: [f64; 3],
        vel_gain: f64,
        dt: f64,
    ) -> ([f64; 3], [f64; 3], [f64; 3]) {
        // Convert observations
        let obs: Vec<Observation> = observations
            .iter()
            .filter_map(|(kind, value)| match kind.as_str() {
                "barometer" => Some(Observation::Barometer {
                    altitude: *value,
                    timestamp: 0.0,
                }),
                _ => None,
            })
            .collect();

        let tb = Vector3::new(threat_bearing[0], threat_bearing[1], threat_bearing[2]);
        let (pos, vel, probs) = self.inner.step(&obs, &tb, vel_gain, dt);

        ([pos.x, pos.y, pos.z], [vel.x, vel.y, vel.z], probs)
    }

    /// Get current best-estimate drone state.
    fn to_drone_state(&self) -> PyDroneState {
        let ds = self.inner.to_drone_state(self.drone_id);
        PyDroneState {
            position: [ds.position.x, ds.position.y, ds.position.z],
            velocity: [ds.velocity.x, ds.velocity.y, ds.velocity.z],
            regime: PyRegime::from(ds.regime),
            drone_id: ds.drone_id,
        }
    }

    /// Number of particles in the filter.
    #[getter]
    fn n_particles(&self) -> usize {
        self.inner.weights.len()
    }

    /// Current effective sample size.
    #[getter]
    fn ess(&self) -> f64 {
        crate::particle_nav::effective_sample_size(&self.inner.weights)
    }

    fn __repr__(&self) -> String {
        let ds = self.inner.to_drone_state(self.drone_id);
        format!(
            "ParticleNavFilter(n={}, pos=[{:.1},{:.1},{:.1}], regime={:?})",
            self.inner.weights.len(),
            ds.position.x,
            ds.position.y,
            ds.position.z,
            ds.regime
        )
    }
}

// ---------------------------------------------------------------------------
// Python functions
// ---------------------------------------------------------------------------

/// Detect GPS/radio jamming via CUSUM on signal quality metrics.
///
/// Args:
///     signal_metrics: Time series of signal quality values (e.g. SNR).
///     threshold: CUSUM threshold multiplier (default 0.5).
///     min_samples: Minimum samples before CUSUM is active (default 10).
///
/// Returns:
///     Tuple of (is_jamming: bool, direction: int, cusum_value: float).
#[pyfunction]
#[pyo3(name = "detect_jamming", signature = (signal_metrics, threshold=0.5, min_samples=10))]
pub fn py_detect_jamming(
    signal_metrics: Vec<f64>,
    threshold: f64,
    min_samples: usize,
) -> (bool, i32, f64) {
    let config = CusumConfig {
        threshold_h: threshold,
        min_samples,
    };
    anomaly::detect_jamming(&signal_metrics, &config)
}

/// Detect the current battlespace regime from tactical signals.
///
/// Args:
///     cusum_triggered: Whether CUSUM detected a break.
///     threat_distance: Distance to nearest threat (meters).
///     closing_rate: Threat closing speed (m/s, negative = approaching).
///     hurst: Hurst exponent of threat movement (default 0.5).
///     current_regime: Current regime index (0=Patrol, 1=Engage, 2=Evade).
///
/// Returns:
///     PyRegime — the detected regime.
#[pyfunction]
#[pyo3(name = "detect_regime", signature = (cusum_triggered, threat_distance, closing_rate, hurst=0.5, current_regime=0))]
pub fn py_detect_regime(
    cusum_triggered: bool,
    threat_distance: f64,
    closing_rate: f64,
    hurst: f64,
    current_regime: u8,
) -> PyResult<PyRegime> {
    let current = Regime::from_index(current_regime)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let signals = RegimeSignals {
        cusum_triggered,
        cusum_direction: if cusum_triggered { -1 } else { 0 },
        hurst,
        volatility_ratio: 1.0,
        threat_distance,
        closing_rate,
    };

    let result = regime::detect_regime(&signals, current, &DetectionConfig::default());
    Ok(PyRegime::from(result))
}
