//! PyO3 wrapper types for the strix-auction API.
//!
//! Exposes auction types (`Auctioneer`, `Task`, `DroneState`, etc.) to Python
//! so that `brain.py` can call `Auctioneer.run_auction()` directly.
//!
//! ```python
//! from strix._strix_core import Auctioneer, AuctionDroneState, Task, ThreatState
//!
//! auctioneer = Auctioneer()
//! drones = [AuctionDroneState(id=1, position=[0.0, 0.0, 100.0])]
//! tasks  = [Task(id=10, location=[50.0, 50.0, 50.0])]
//! result = auctioneer.run_auction(drones, tasks, [])
//! ```

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// PyPosition
// ---------------------------------------------------------------------------

/// Minimal 3-D position used by the auction engine.
#[pyclass(name = "Position")]
#[derive(Debug, Clone, Copy)]
pub struct PyPosition {
    inner: strix_auction::Position,
}

#[pymethods]
impl PyPosition {
    #[new]
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            inner: strix_auction::Position::new(x, y, z),
        }
    }

    #[getter]
    fn x(&self) -> f64 {
        self.inner.x
    }

    #[getter]
    fn y(&self) -> f64 {
        self.inner.y
    }

    #[getter]
    fn z(&self) -> f64 {
        self.inner.z
    }

    fn distance_to(&self, other: &PyPosition) -> f64 {
        self.inner.distance_to(&other.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Position({:.1}, {:.1}, {:.1})",
            self.inner.x, self.inner.y, self.inner.z
        )
    }
}

// ---------------------------------------------------------------------------
// PyCapabilities
// ---------------------------------------------------------------------------

/// Capability flags a drone may carry.
#[pyclass(name = "Capabilities")]
#[derive(Debug, Clone)]
pub struct PyCapabilities {
    pub(crate) inner: strix_auction::Capabilities,
}

#[pymethods]
impl PyCapabilities {
    #[new]
    #[pyo3(signature = (has_sensor=true, has_weapon=false, has_ew=false, has_relay=false))]
    fn new(has_sensor: bool, has_weapon: bool, has_ew: bool, has_relay: bool) -> Self {
        Self {
            inner: strix_auction::Capabilities {
                has_sensor,
                has_weapon,
                has_ew,
                has_relay,
            },
        }
    }

    #[getter]
    fn has_sensor(&self) -> bool {
        self.inner.has_sensor
    }

    #[getter]
    fn has_weapon(&self) -> bool {
        self.inner.has_weapon
    }

    #[getter]
    fn has_ew(&self) -> bool {
        self.inner.has_ew
    }

    #[getter]
    fn has_relay(&self) -> bool {
        self.inner.has_relay
    }

    fn __repr__(&self) -> String {
        format!(
            "Capabilities(sensor={}, weapon={}, ew={}, relay={})",
            self.inner.has_sensor, self.inner.has_weapon, self.inner.has_ew, self.inner.has_relay
        )
    }
}

// ---------------------------------------------------------------------------
// PyAuctionDroneState
// ---------------------------------------------------------------------------

/// Lightweight drone state used by the auction system.
///
/// Named `AuctionDroneState` to avoid clash with the existing `DroneState`
/// wrapper in `python.rs`.
#[pyclass(name = "AuctionDroneState")]
#[derive(Debug, Clone)]
pub struct PyAuctionDroneState {
    pub(crate) inner: strix_auction::DroneState,
}

#[pymethods]
impl PyAuctionDroneState {
    #[new]
    #[pyo3(signature = (id, position, velocity=[0.0, 0.0, 0.0], regime_index=0, capabilities=None, energy=1.0, alive=true))]
    fn new(
        id: u32,
        position: [f64; 3],
        velocity: [f64; 3],
        regime_index: u8,
        capabilities: Option<&PyCapabilities>,
        energy: f64,
        alive: bool,
    ) -> PyResult<Self> {
        let regime = strix_auction::Regime::from_index(regime_index)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let caps = capabilities.map(|c| c.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: strix_auction::DroneState {
                id,
                position: strix_auction::Position::new(position[0], position[1], position[2]),
                velocity,
                regime,
                capabilities: caps,
                energy,
                alive,
            },
        })
    }

    #[getter]
    fn id(&self) -> u32 {
        self.inner.id
    }

    #[getter]
    fn energy(&self) -> f64 {
        self.inner.energy
    }

    #[getter]
    fn alive(&self) -> bool {
        self.inner.alive
    }

    fn __repr__(&self) -> String {
        format!(
            "AuctionDroneState(id={}, energy={:.2})",
            self.inner.id, self.inner.energy
        )
    }
}

// ---------------------------------------------------------------------------
// PyTask
// ---------------------------------------------------------------------------

/// A task to be auctioned among drones.
#[pyclass(name = "Task")]
#[derive(Debug, Clone)]
pub struct PyTask {
    pub(crate) inner: strix_auction::Task,
}

#[pymethods]
impl PyTask {
    #[new]
    #[pyo3(signature = (id, location, required_capabilities=None, priority=0.5, urgency=0.5, bundle_id=None, dark_pool=None))]
    fn new(
        id: u32,
        location: [f64; 3],
        required_capabilities: Option<&PyCapabilities>,
        priority: f64,
        urgency: f64,
        bundle_id: Option<u32>,
        dark_pool: Option<u32>,
    ) -> Self {
        let caps = required_capabilities
            .map(|c| c.inner.clone())
            .unwrap_or_default();
        Self {
            inner: strix_auction::Task {
                id,
                location: strix_auction::Position::new(location[0], location[1], location[2]),
                required_capabilities: caps,
                priority,
                urgency,
                bundle_id,
                dark_pool,
            },
        }
    }

    #[getter]
    fn id(&self) -> u32 {
        self.inner.id
    }

    #[getter]
    fn priority(&self) -> f64 {
        self.inner.priority
    }

    #[getter]
    fn urgency(&self) -> f64 {
        self.inner.urgency
    }

    fn __repr__(&self) -> String {
        format!(
            "Task(id={}, priority={:.2})",
            self.inner.id, self.inner.priority
        )
    }
}

// ---------------------------------------------------------------------------
// PyThreatType
// ---------------------------------------------------------------------------

/// Classification of threat types.
#[pyclass(name = "ThreatType")]
#[derive(Debug, Clone, Copy)]
pub struct PyThreatType {
    pub(crate) inner: strix_auction::ThreatType,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyThreatType {
    /// Surface-to-air missile system.
    #[classattr]
    fn Sam() -> Self {
        Self {
            inner: strix_auction::ThreatType::Sam,
        }
    }

    /// Small arms / anti-aircraft artillery.
    #[classattr]
    fn SmallArms() -> Self {
        Self {
            inner: strix_auction::ThreatType::SmallArms,
        }
    }

    /// Electronic warfare / jamming.
    #[classattr]
    fn ElectronicWarfare() -> Self {
        Self {
            inner: strix_auction::ThreatType::ElectronicWarfare,
        }
    }

    /// Unknown threat.
    #[classattr]
    fn Unknown() -> Self {
        Self {
            inner: strix_auction::ThreatType::Unknown,
        }
    }

    fn __repr__(&self) -> String {
        format!("ThreatType.{:?}", self.inner)
    }
}

// ---------------------------------------------------------------------------
// PyThreatState
// ---------------------------------------------------------------------------

/// Threat information used for risk calculations.
#[pyclass(name = "ThreatState")]
#[derive(Debug, Clone)]
pub struct PyThreatState {
    pub(crate) inner: strix_auction::ThreatState,
}

#[pymethods]
impl PyThreatState {
    #[new]
    #[pyo3(signature = (id, position, lethal_radius=200.0, threat_type=None))]
    fn new(
        id: u32,
        position: [f64; 3],
        lethal_radius: f64,
        threat_type: Option<&PyThreatType>,
    ) -> Self {
        Self {
            inner: strix_auction::ThreatState {
                id,
                position: strix_auction::Position::new(position[0], position[1], position[2]),
                lethal_radius,
                threat_type: threat_type
                    .map(|t| t.inner)
                    .unwrap_or(strix_auction::ThreatType::Unknown),
            },
        }
    }

    #[getter]
    fn id(&self) -> u32 {
        self.inner.id
    }

    #[getter]
    fn lethal_radius(&self) -> f64 {
        self.inner.lethal_radius
    }

    fn __repr__(&self) -> String {
        format!(
            "ThreatState(id={}, radius={:.0})",
            self.inner.id, self.inner.lethal_radius
        )
    }
}

// ---------------------------------------------------------------------------
// PyAssignment  (read-only output)
// ---------------------------------------------------------------------------

/// Drone-to-task assignment produced by the auction.
#[pyclass(name = "Assignment")]
#[derive(Debug, Clone)]
pub struct PyAssignment {
    inner: strix_auction::Assignment,
}

#[pymethods]
impl PyAssignment {
    #[getter]
    fn drone_id(&self) -> u32 {
        self.inner.drone_id
    }

    #[getter]
    fn task_id(&self) -> u32 {
        self.inner.task_id
    }

    #[getter]
    fn bid_score(&self) -> f64 {
        self.inner.bid_score
    }

    fn __repr__(&self) -> String {
        format!(
            "Assignment(drone={}, task={}, score={:.2})",
            self.inner.drone_id, self.inner.task_id, self.inner.bid_score
        )
    }
}

// ---------------------------------------------------------------------------
// PyAuctionResult  (read-only output)
// ---------------------------------------------------------------------------

/// Outcome of a single auction round.
#[pyclass(name = "AuctionResult")]
#[derive(Debug, Clone)]
pub struct PyAuctionResult {
    inner: strix_auction::AuctionResult,
}

#[pymethods]
impl PyAuctionResult {
    /// Finalized drone-to-task assignments.
    #[getter]
    fn assignments(&self) -> Vec<PyAssignment> {
        self.inner
            .assignments
            .iter()
            .map(|a| PyAssignment { inner: a.clone() })
            .collect()
    }

    /// Task IDs that received no valid bid.
    #[getter]
    fn unassigned_tasks(&self) -> Vec<u32> {
        self.inner.unassigned_tasks.clone()
    }

    /// Total market welfare (sum of winning bid scores).
    #[getter]
    fn total_welfare(&self) -> f64 {
        self.inner.total_welfare
    }

    fn __repr__(&self) -> String {
        format!(
            "AuctionResult(assignments={}, welfare={:.2})",
            self.inner.assignments.len(),
            self.inner.total_welfare
        )
    }
}

// ---------------------------------------------------------------------------
// PyAuctioneer
// ---------------------------------------------------------------------------

/// Orchestrates the combinatorial task auction.
#[pyclass(name = "Auctioneer")]
pub struct PyAuctioneer {
    inner: strix_auction::Auctioneer,
}

#[pymethods]
impl PyAuctioneer {
    #[new]
    #[pyo3(signature = (min_bid_threshold=0.0))]
    fn new(min_bid_threshold: f64) -> Self {
        Self {
            inner: strix_auction::Auctioneer::new().with_min_bid(min_bid_threshold),
        }
    }

    /// Run a full auction round.
    ///
    /// Args:
    ///     drones: Fleet state — each drone bids independently.
    ///     tasks: Tasks available for this round.
    ///     threats: Current threat picture (affects bid risk scoring).
    ///
    /// Returns:
    ///     AuctionResult with assignments, unassigned tasks, and total welfare.
    fn run_auction(
        &mut self,
        drones: Vec<PyRef<PyAuctionDroneState>>,
        tasks: Vec<PyRef<PyTask>>,
        threats: Vec<PyRef<PyThreatState>>,
    ) -> PyAuctionResult {
        let d: Vec<strix_auction::DroneState> = drones.iter().map(|d| d.inner.clone()).collect();
        let t: Vec<strix_auction::Task> = tasks.iter().map(|t| t.inner.clone()).collect();
        let th: Vec<strix_auction::ThreatState> = threats.iter().map(|t| t.inner.clone()).collect();
        let result = self
            .inner
            .run_auction(&d, &t, &th, &std::collections::HashMap::new(), &[]);
        PyAuctionResult { inner: result }
    }

    /// Signal that conditions have changed and a re-auction is warranted.
    fn trigger_reauction(&mut self) {
        self.inner.trigger_reauction();
    }

    /// Whether a re-auction has been requested.
    #[getter]
    fn needs_reauction(&self) -> bool {
        self.inner.needs_reauction
    }

    fn __repr__(&self) -> String {
        format!("Auctioneer(min_bid={:.2})", self.inner.min_bid_threshold)
    }
}

// ---------------------------------------------------------------------------
// PyLossAnalyzer
// ---------------------------------------------------------------------------

/// Anti-fragile loss recovery — the swarm gets stronger after losses.
#[pyclass(name = "LossAnalyzer")]
pub struct PyLossAnalyzer {
    inner: strix_auction::LossAnalyzer,
}

#[pymethods]
impl PyLossAnalyzer {
    #[new]
    fn new() -> Self {
        Self {
            inner: strix_auction::LossAnalyzer::new(),
        }
    }

    /// Anti-fragility score: how much the swarm has learned from losses.
    fn antifragile_score(&self) -> f64 {
        self.inner.antifragile_score()
    }

    /// Number of active kill zones.
    fn active_kill_zones(&self) -> usize {
        self.inner.active_kill_zones()
    }

    fn __repr__(&self) -> String {
        format!(
            "LossAnalyzer(kill_zones={}, score={:.2})",
            self.inner.active_kill_zones(),
            self.inner.antifragile_score()
        )
    }
}
