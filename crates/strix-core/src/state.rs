//! Core state types for GPS-denied drone navigation.
//!
//! Defines the 6D state space (position + velocity), drone/fleet/threat
//! representations, sensor observations, and regime classification.
//!
//! # Coordinate Convention — NED (North-East-Down)
//!
//! All positions and velocities in this crate use the **NED frame**:
//! - **X = North** (positive = North)
//! - **Y = East** (positive = East)
//! - **Z = Down** (positive = down; *negative Z = altitude above ground*)
//!
//! Example: a drone at 50 m altitude has `position.z = -50.0`.
//! Altitude floor/ceiling in [`CbfConfig`] use the same sign convention.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can arise within the strix-core state layer.
#[derive(Debug, Error)]
pub enum StateError {
    /// Particle count must be > 0.
    #[error("particle count must be > 0, got {0}")]
    InvalidParticleCount(usize),

    /// Regime index out of range.
    #[error("invalid regime index: {0}")]
    InvalidRegime(u8),

    /// Sensor data is malformed or out of range.
    #[error("sensor data error: {0}")]
    SensorError(String),
}

// ---------------------------------------------------------------------------
// Regime
// ---------------------------------------------------------------------------

/// Battlespace operating regime — analogous to the trading
/// RANGE / TREND / PANIC regimes but mapped to drone tactics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Regime {
    /// Normal operations — area monitoring, formation holding.
    Patrol = 0,
    /// Active threat engagement — velocity tracks threat bearing.
    Engage = 1,
    /// Threat avoidance — high-noise escape maneuvers.
    Evade = 2,
}

impl Regime {
    /// Total number of regimes.
    pub const COUNT: usize = 3;

    /// Try to construct from a numeric index.
    pub fn from_index(idx: u8) -> Result<Self, StateError> {
        match idx {
            0 => Ok(Self::Patrol),
            1 => Ok(Self::Engage),
            2 => Ok(Self::Evade),
            other => Err(StateError::InvalidRegime(other)),
        }
    }

    /// Convert to usize index for matrix lookup.
    pub fn as_index(self) -> usize {
        self as usize
    }
}

/// Default 3x3 Markov transition matrix for regime switching.
///
/// Row = current regime, column = next regime.
/// Diagonal-dominant: regimes tend to persist.
pub fn default_transition_matrix() -> [[f64; 3]; 3] {
    [
        // From PATROL  → [PATROL, ENGAGE, EVADE]
        [0.90, 0.07, 0.03],
        // From ENGAGE  → [PATROL, ENGAGE, EVADE]
        [0.10, 0.80, 0.10],
        // From EVADE   → [PATROL, ENGAGE, EVADE]
        [0.15, 0.10, 0.75],
    ]
}

// ---------------------------------------------------------------------------
// Threat regime (enemy-side)
// ---------------------------------------------------------------------------

/// Enemy intent classification for the threat particle filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ThreatRegime {
    /// Stationary / defensive posture — mean-reverting around position.
    Defend = 0,
    /// Advancing towards our fleet centroid.
    CounterAttack = 1,
    /// Moving away / disengaging.
    Retreat = 2,
}

impl ThreatRegime {
    /// Total number of threat regimes.
    pub const COUNT: usize = 3;

    /// Try to construct from a numeric index.
    pub fn from_index(idx: u8) -> Result<Self, StateError> {
        match idx {
            0 => Ok(Self::Defend),
            1 => Ok(Self::CounterAttack),
            2 => Ok(Self::Retreat),
            other => Err(StateError::InvalidRegime(other)),
        }
    }

    /// Convert to usize index.
    pub fn as_index(self) -> usize {
        self as usize
    }
}

/// Default transition matrix for threat regimes.
pub fn default_threat_transition_matrix() -> [[f64; 3]; 3] {
    [
        // DEFEND      → [DEFEND, COUNTER_ATTACK, RETREAT]
        [0.85, 0.10, 0.05],
        // COUNTER_ATK → [DEFEND, COUNTER_ATTACK, RETREAT]
        [0.08, 0.82, 0.10],
        // RETREAT      → [DEFEND, COUNTER_ATTACK, RETREAT]
        [0.12, 0.08, 0.80],
    ]
}

// ---------------------------------------------------------------------------
// 6D Drone State
// ---------------------------------------------------------------------------

/// Full 6-DOF state of a single drone particle.
///
/// Replaces the original 2D `[log_price, velocity]` with
/// `[x, y, z, vx, vy, vz]` plus metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneState {
    /// 3D position in local NED frame (meters).
    pub position: Vector3<f64>,
    /// 3D velocity in local NED frame (m/s).
    pub velocity: Vector3<f64>,
    /// Current operating regime.
    pub regime: Regime,
    /// Particle weight (importance weight, sums to 1 across the filter).
    pub weight: f64,
    /// Unique drone identifier within the fleet.
    pub drone_id: u32,
    /// Capability bit-flags (sensor payload, weapon type, relay, etc.).
    pub capabilities: u64,
}

impl DroneState {
    /// Create a new drone state at the given position with zero velocity.
    pub fn new(drone_id: u32, position: Vector3<f64>) -> Self {
        Self {
            position,
            velocity: Vector3::zeros(),
            regime: Regime::Patrol,
            weight: 1.0,
            drone_id,
            capabilities: 0,
        }
    }

    /// Pack state into a flat `[f64; 6]` array: `[x, y, z, vx, vy, vz]`.
    pub fn to_array(&self) -> [f64; 6] {
        [
            self.position.x,
            self.position.y,
            self.position.z,
            self.velocity.x,
            self.velocity.y,
            self.velocity.z,
        ]
    }

    /// Unpack from a flat `[f64; 6]` array, preserving regime/weight/id.
    pub fn from_array(&mut self, arr: &[f64; 6]) {
        self.position = Vector3::new(arr[0], arr[1], arr[2]);
        self.velocity = Vector3::new(arr[3], arr[4], arr[5]);
    }
}

// ---------------------------------------------------------------------------
// Fleet State
// ---------------------------------------------------------------------------

/// Aggregate fleet state — a collection of drones plus fleet-level metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetState {
    /// Individual drone states.
    pub drones: Vec<DroneState>,
    /// Fleet centroid position (weighted mean of drone positions).
    pub centroid: Vector3<f64>,
    /// Fleet dispersion — RMS distance from centroid (meters).
    pub dispersion: f64,
    /// Number of active drones.
    pub active_count: usize,
}

impl FleetState {
    /// Compute centroid and dispersion from the current drone states.
    pub fn recompute_metrics(&mut self) {
        let n = self.drones.len();
        if n == 0 {
            self.centroid = Vector3::zeros();
            self.dispersion = 0.0;
            self.active_count = 0;
            return;
        }
        self.active_count = n;

        let mut sum = Vector3::zeros();
        for d in &self.drones {
            sum += d.position;
        }
        self.centroid = sum / n as f64;

        let mut sq_sum = 0.0_f64;
        for d in &self.drones {
            sq_sum += (d.position - self.centroid).norm_squared();
        }
        self.dispersion = (sq_sum / n as f64).sqrt();
    }

    /// Create a new fleet from drone states and immediately compute metrics.
    pub fn from_drones(drones: Vec<DroneState>) -> Self {
        let mut fleet = Self {
            drones,
            centroid: Vector3::zeros(),
            dispersion: 0.0,
            active_count: 0,
        };
        fleet.recompute_metrics();
        fleet
    }
}

// ---------------------------------------------------------------------------
// Threat State
// ---------------------------------------------------------------------------

/// A single threat particle — one hypothesis about an enemy entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatState {
    /// Hypothesised enemy position (NED, meters).
    pub position: Vector3<f64>,
    /// Hypothesised enemy velocity (NED, m/s).
    pub velocity: Vector3<f64>,
    /// Enemy regime hypothesis.
    pub regime: ThreatRegime,
    /// Particle weight.
    pub weight: f64,
    /// Threat identifier (track number).
    pub threat_id: u32,
}

impl ThreatState {
    /// Create a new threat particle at the given position.
    pub fn new(threat_id: u32, position: Vector3<f64>) -> Self {
        Self {
            position,
            velocity: Vector3::zeros(),
            regime: ThreatRegime::Defend,
            weight: 1.0,
            threat_id,
        }
    }

    /// Pack into flat array.
    pub fn to_array(&self) -> [f64; 6] {
        [
            self.position.x,
            self.position.y,
            self.position.z,
            self.velocity.x,
            self.velocity.y,
            self.velocity.z,
        ]
    }

    /// Unpack from flat array.
    pub fn from_array(&mut self, arr: &[f64; 6]) {
        self.position = Vector3::new(arr[0], arr[1], arr[2]);
        self.velocity = Vector3::new(arr[3], arr[4], arr[5]);
    }
}

// ---------------------------------------------------------------------------
// Sensor Observations
// ---------------------------------------------------------------------------

/// A single sensor reading from one of the on-board instruments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Observation {
    /// IMU acceleration reading (body-frame, m/s^2).
    Imu {
        /// Acceleration vector in body frame.
        acceleration: Vector3<f64>,
        /// Angular velocity (rad/s), optional.
        gyro: Option<Vector3<f64>>,
        /// Timestamp in seconds since epoch.
        timestamp: f64,
    },
    /// Barometric altitude (meters above sea level).
    Barometer { altitude: f64, timestamp: f64 },
    /// Magnetometer heading (radians from magnetic north, NED).
    Magnetometer {
        heading: Vector3<f64>,
        timestamp: f64,
    },
    /// Rangefinder distance-to-ground (meters).
    Rangefinder {
        distance: f64,
        /// Sensor pointing direction in body frame (unit vector).
        direction: Vector3<f64>,
        timestamp: f64,
    },
    /// Visual odometry — estimated delta-position since last frame.
    VisualOdometry {
        delta_position: Vector3<f64>,
        /// Confidence in [0, 1].
        confidence: f64,
        timestamp: f64,
    },
    /// Radio bearing — direction to a known emitter or another drone.
    RadioBearing {
        /// Unit bearing vector in NED.
        bearing: Vector3<f64>,
        /// Signal strength (dBm).
        signal_strength: f64,
        /// Emitter ID if known.
        emitter_id: Option<u32>,
        timestamp: f64,
    },
}

impl Observation {
    /// Extract the timestamp regardless of sensor type.
    pub fn timestamp(&self) -> f64 {
        match self {
            Self::Imu { timestamp, .. }
            | Self::Barometer { timestamp, .. }
            | Self::Magnetometer { timestamp, .. }
            | Self::Rangefinder { timestamp, .. }
            | Self::VisualOdometry { timestamp, .. }
            | Self::RadioBearing { timestamp, .. } => *timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// Sensor Configuration
// ---------------------------------------------------------------------------

/// Per-sensor-type noise parameters used by the measurement update step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorConfig {
    /// IMU acceleration noise std-dev (m/s^2).
    pub imu_accel_noise: f64,
    /// IMU gyro noise std-dev (rad/s).
    pub imu_gyro_noise: f64,
    /// Barometer altitude noise std-dev (meters).
    pub baro_noise: f64,
    /// Magnetometer heading noise std-dev (radians).
    pub mag_noise: f64,
    /// Rangefinder distance noise std-dev (meters).
    pub rangefinder_noise: f64,
    /// Visual odometry position noise std-dev (meters).
    pub vo_noise: f64,
    /// Radio bearing angular noise std-dev (radians).
    pub radio_bearing_noise: f64,
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            imu_accel_noise: 0.1,
            imu_gyro_noise: 0.01,
            baro_noise: 0.5,
            mag_noise: 0.05,
            rangefinder_noise: 0.1,
            vo_noise: 0.02,
            radio_bearing_noise: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regime_round_trip() {
        for idx in 0..3u8 {
            let r = Regime::from_index(idx).unwrap();
            assert_eq!(r.as_index(), idx as usize);
        }
        assert!(Regime::from_index(5).is_err());
    }

    #[test]
    fn threat_regime_round_trip() {
        for idx in 0..3u8 {
            let r = ThreatRegime::from_index(idx).unwrap();
            assert_eq!(r.as_index(), idx as usize);
        }
    }

    #[test]
    fn drone_state_array_round_trip() {
        let mut ds = DroneState::new(1, Vector3::new(1.0, 2.0, 3.0));
        ds.velocity = Vector3::new(4.0, 5.0, 6.0);
        let arr = ds.to_array();
        assert_eq!(arr, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let new_arr = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        ds.from_array(&new_arr);
        assert_eq!(ds.position.x, 10.0);
        assert_eq!(ds.velocity.z, 60.0);
    }

    #[test]
    fn fleet_metrics() {
        let d1 = DroneState::new(1, Vector3::new(0.0, 0.0, 0.0));
        let d2 = DroneState::new(2, Vector3::new(10.0, 0.0, 0.0));
        let fleet = FleetState::from_drones(vec![d1, d2]);
        assert!((fleet.centroid.x - 5.0).abs() < 1e-12);
        assert!(fleet.dispersion > 0.0);
        assert_eq!(fleet.active_count, 2);
    }

    #[test]
    fn transition_matrix_rows_sum_to_one() {
        let tm = default_transition_matrix();
        for row in &tm {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn observation_timestamp() {
        let obs = Observation::Barometer {
            altitude: 100.0,
            timestamp: 42.0,
        };
        assert_eq!(obs.timestamp(), 42.0);
    }
}
