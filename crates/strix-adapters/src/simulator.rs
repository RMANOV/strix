//! Fully working drone simulator adapter for testing.
//!
//! Unlike the MAVLink and ROS2 stubs, `SimulatorAdapter` maintains real
//! internal state and applies a simple physics model so the rest of the
//! STRIX stack can be tested end-to-end without hardware.
//!
//! ## Physics model
//!
//! - Position integrates velocity: `pos += vel * dt`
//! - Velocity integrates acceleration: `vel += accel * dt`
//! - Acceleration is computed from the desired waypoint (P-controller)
//! - Battery drains linearly with time and thrust
//! - Optional GPS noise and wind disturbance
//!
//! ## Fleet management
//!
//! [`SimulatorFleet`] creates and manages multiple [`SimulatorAdapter`]s
//! for swarm-level testing.

use std::sync::{Arc, Mutex};

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use strix_core::cbf::{self, CbfConfig, NeighborState, NoFlyZone};

use crate::traits::*;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Simulator configuration — physics tuning and fault injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatorConfig {
    /// Constant wind vector `[wx, wy, wz]` in m/s (NED).
    pub wind: [f64; 3],
    /// GPS position noise standard deviation (meters).
    pub gps_noise_std: f64,
    /// Battery drain rate (fraction per second at hover).
    pub battery_drain_rate: f64,
    /// Proportional gain for the waypoint tracker.
    pub p_gain: f64,
    /// Maximum acceleration magnitude (m/s^2).
    pub max_accel: f64,
    /// Physics time-step (seconds).
    pub dt: f64,
    /// Whether to inject random failures.
    pub failure_injection: bool,
    /// Probability of a random failure per `step()` call.
    pub failure_probability: f64,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            wind: [0.0, 0.0, 0.0],
            gps_noise_std: 0.0,
            battery_drain_rate: 0.001, // ~16 minutes at hover
            p_gain: 1.5,
            max_accel: 8.0,
            dt: 0.1,
            failure_injection: false,
            failure_probability: 0.0001,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

/// Mutable internal state for a single simulated drone.
#[derive(Debug, Clone)]
struct SimState {
    /// Position in local NED (meters).
    position: [f64; 3],
    /// Velocity in local NED (m/s).
    velocity: [f64; 3],
    /// Attitude (roll, pitch, yaw) in radians.
    attitude: [f64; 3],
    /// Battery state of charge [0.0, 1.0].
    battery: f64,
    /// Whether the motors are armed.
    armed: bool,
    /// Whether the drone is airborne.
    airborne: bool,
    /// Current flight mode.
    mode: FlightMode,
    /// Active waypoint target, if any.
    target: Option<[f64; 3]>,
    /// Commanded speed (m/s).
    speed_setpoint: f64,
    /// Monotonic simulation clock (seconds).
    clock: f64,
    /// Whether a simulated failure has occurred.
    failed: bool,
}

impl SimState {
    fn new(position: [f64; 3]) -> Self {
        Self {
            position,
            velocity: [0.0; 3],
            attitude: [0.0; 3],
            battery: 1.0,
            armed: false,
            airborne: false,
            mode: FlightMode::Stabilize,
            target: None,
            speed_setpoint: 10.0,
            clock: 0.0,
            failed: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Simulator adapter
// ---------------------------------------------------------------------------

/// A fully-functional simulated drone for testing.
///
/// Thread-safe: the internal state is wrapped in `Arc<Mutex<_>>` so the
/// adapter can be shared across the swarm orchestrator's threads.
pub struct SimulatorAdapter {
    /// Unique drone ID.
    drone_id: u32,
    /// Simulator configuration.
    config: SimulatorConfig,
    /// Platform capabilities.
    caps: Capabilities,
    /// Thread-safe mutable state.
    state: Arc<Mutex<SimState>>,
}

impl SimulatorAdapter {
    /// Create a new simulator drone at the given starting position.
    pub fn new(drone_id: u32, start_pos: [f64; 3], config: SimulatorConfig) -> Self {
        Self {
            drone_id,
            config,
            caps: Capabilities {
                max_speed: 20.0,
                max_altitude: 500.0,
                endurance: 1800.0,
                sensors: vec![SensorType::Camera, SensorType::Thermal],
                payload_kg: 1.5,
                comms_range: 5000.0,
            },
            state: Arc::new(Mutex::new(SimState::new(start_pos))),
        }
    }

    /// Create with default config at the origin.
    pub fn new_default(drone_id: u32) -> Self {
        Self::new(drone_id, [0.0, 0.0, 0.0], SimulatorConfig::default())
    }

    /// Advance the simulation by one time-step.
    ///
    /// This is the core physics loop:
    /// 1. Compute acceleration towards the target waypoint (P-controller).
    /// 2. Apply wind disturbance.
    /// 3. Integrate velocity and position.
    /// 4. Drain the battery.
    /// 5. (Optionally) inject failures.
    pub fn step(&self) {
        let mut s = self.state.lock().expect("simulator state mutex poisoned");
        if s.failed || !s.armed {
            return;
        }

        let dt = self.config.dt;
        s.clock += dt;

        // --- acceleration towards target (critically-damped PD controller) ---
        let mut accel = [0.0_f64; 3];
        let has_target = s.target.is_some();
        if let Some(target) = s.target {
            let kp = self.config.p_gain;
            // Critical damping: kd = 2*sqrt(kp)
            let kd = 2.0 * kp.sqrt();
            for i in 0..3 {
                let error = target[i] - s.position[i];
                accel[i] = kp * error - kd * s.velocity[i];
            }
            // Clamp acceleration magnitude — but allow 2x for deceleration
            // (braking is more important than max thrust for convergence)
            let mag = (accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]).sqrt();
            let dist = ((target[0] - s.position[0]).powi(2)
                + (target[1] - s.position[1]).powi(2)
                + (target[2] - s.position[2]).powi(2))
            .sqrt();
            // Near the target, allow full PD authority for precise convergence
            let effective_max = if dist < 20.0 {
                self.config.max_accel * 3.0
            } else {
                self.config.max_accel
            };
            if mag > effective_max {
                let scale = effective_max / mag;
                for a in &mut accel {
                    *a *= scale;
                }
            }
        }

        // --- wind disturbance ---
        for (a, w) in accel.iter_mut().zip(self.config.wind.iter()) {
            *a += w * 0.1; // wind as a gentle force
        }

        // --- drag when loitering (no target) ---
        if !has_target {
            for v in &mut s.velocity {
                *v *= 1.0 - 2.0 * dt; // exponential drag
            }
        }

        // --- integrate velocity ---
        for (v, a) in s.velocity.iter_mut().zip(accel.iter()) {
            *v += a * dt;
        }

        // Clamp speed to setpoint only during cruise (not near target)
        let speed = (s.velocity[0].powi(2) + s.velocity[1].powi(2) + s.velocity[2].powi(2)).sqrt();
        let near_target = s.target.is_none_or(|t| {
            let d = ((t[0] - s.position[0]).powi(2)
                + (t[1] - s.position[1]).powi(2)
                + (t[2] - s.position[2]).powi(2))
            .sqrt();
            d < s.speed_setpoint * 3.0 // within ~3 seconds of braking distance
        });
        if speed > s.speed_setpoint && speed > 0.0 && !near_target {
            let scale = s.speed_setpoint / speed;
            for v in &mut s.velocity {
                *v *= scale;
            }
        }

        // --- integrate position ---
        for i in 0..3 {
            s.position[i] += s.velocity[i] * dt;
        }

        // --- attitude from velocity (simplified) ---
        let ground_speed = (s.velocity[0].powi(2) + s.velocity[1].powi(2)).sqrt();
        if ground_speed > 0.1 {
            s.attitude[2] = s.velocity[1].atan2(s.velocity[0]); // yaw
            s.attitude[1] = (-s.velocity[2]).atan2(ground_speed); // pitch
        }

        // --- battery drain ---
        let thrust_factor = 1.0 + speed * 0.02; // more drain at higher speed
        s.battery -= self.config.battery_drain_rate * dt * thrust_factor;
        s.battery = s.battery.max(0.0);

        if s.battery <= 0.0 {
            s.failed = true;
        }

        // --- check arrival ---
        if let Some(target) = s.target {
            let dist_sq = (0..3)
                .map(|i| (s.position[i] - target[i]).powi(2))
                .sum::<f64>();
            if dist_sq < 1.0 {
                // within 1 meter — arrived
                s.target = None;
                for v in &mut s.velocity {
                    *v *= 0.1; // decelerate
                }
            }
        }

        // --- optional failure injection ---
        if self.config.failure_injection {
            // Deterministic pseudo-random: use clock as seed
            let pseudo = ((s.clock * 1000.0) as u64 % 10000) as f64 / 10000.0;
            if pseudo < self.config.failure_probability {
                s.failed = true;
            }
        }
    }

    /// Advance the simulation by `n` time-steps.
    pub fn step_n(&self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Reset the simulator to initial conditions.
    pub fn reset(&self, position: [f64; 3]) {
        let mut s = self.state.lock().expect("simulator state mutex poisoned");
        *s = SimState::new(position);
    }

    /// Get direct access to the simulation clock (seconds).
    pub fn clock(&self) -> f64 {
        self.state
            .lock()
            .expect("simulator state mutex poisoned")
            .clock
    }

    /// Get the current battery level.
    pub fn battery(&self) -> f64 {
        self.state
            .lock()
            .expect("simulator state mutex poisoned")
            .battery
    }

    /// Check whether the simulated drone has failed.
    pub fn has_failed(&self) -> bool {
        self.state
            .lock()
            .expect("simulator state mutex poisoned")
            .failed
    }

    /// Get the Euclidean distance from the current position to a point.
    pub fn distance_to(&self, target: [f64; 3]) -> f64 {
        let s = self.state.lock().expect("simulator state mutex poisoned");
        ((s.position[0] - target[0]).powi(2)
            + (s.position[1] - target[1]).powi(2)
            + (s.position[2] - target[2]).powi(2))
        .sqrt()
    }
}

impl PlatformAdapter for SimulatorAdapter {
    fn id(&self) -> u32 {
        self.drone_id
    }

    fn platform_name(&self) -> &str {
        "STRIX Simulator"
    }

    fn send_waypoint(&self, wp: &Waypoint) -> Result<(), AdapterError> {
        let mut s = self.state.lock().expect("simulator state mutex poisoned");
        if !s.armed {
            return Err(AdapterError::CommandRejected("not armed".into()));
        }
        s.target = Some([wp.lat, wp.lon, wp.alt]);
        s.speed_setpoint = wp.speed.min(self.caps.max_speed);
        s.mode = FlightMode::Guided;
        Ok(())
    }

    fn get_telemetry(&self) -> Result<Telemetry, AdapterError> {
        let s = self.state.lock().expect("simulator state mutex poisoned");
        if s.failed {
            return Err(AdapterError::TelemetryUnavailable(
                "drone has failed".into(),
            ));
        }
        Ok(Telemetry {
            position: s.position,
            velocity: s.velocity,
            attitude: s.attitude,
            battery: s.battery,
            gps_fix: if self.config.gps_noise_std > 0.0 {
                GpsFix::DGps
            } else {
                GpsFix::Fix3D
            },
            armed: s.armed,
            mode: s.mode,
            timestamp: s.clock,
        })
    }

    fn execute_action(&self, action: &Action) -> Result<(), AdapterError> {
        let mut s = self.state.lock().expect("simulator state mutex poisoned");
        if s.failed {
            return Err(AdapterError::CommandRejected("drone has failed".into()));
        }
        match action {
            Action::Arm => {
                s.armed = true;
            }
            Action::Disarm => {
                s.armed = false;
                s.velocity = [0.0; 3];
                s.target = None;
            }
            Action::Takeoff(alt) => {
                if !s.armed {
                    return Err(AdapterError::CommandRejected("not armed".into()));
                }
                s.target = Some([s.position[0], s.position[1], *alt]);
                s.airborne = true;
                s.mode = FlightMode::Guided;
            }
            Action::Land => {
                s.target = Some([s.position[0], s.position[1], 0.0]);
                s.mode = FlightMode::Land;
            }
            Action::RTL => {
                s.target = Some([0.0, 0.0, 0.0]);
                s.mode = FlightMode::RTL;
            }
            Action::GoTo(wp) => {
                if !s.armed {
                    return Err(AdapterError::CommandRejected("not armed".into()));
                }
                drop(s); // release lock for send_waypoint
                return self.send_waypoint(wp);
            }
            Action::SetSpeed(spd) => {
                s.speed_setpoint = spd.min(self.caps.max_speed);
            }
            Action::SetMode(mode) => {
                s.mode = *mode;
            }
            Action::PayloadDrop => {
                // Simulated — just log it
            }
            Action::CameraCapture => {
                // Simulated — just log it
            }
        }
        Ok(())
    }

    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    fn is_connected(&self) -> bool {
        // Simulator is always "connected"
        !self
            .state
            .lock()
            .expect("simulator state mutex poisoned")
            .failed
    }

    fn health_check(&self) -> Result<HealthStatus, AdapterError> {
        let s = self.state.lock().expect("simulator state mutex poisoned");
        if s.failed {
            return Ok(HealthStatus::Critical("simulated failure".into()));
        }
        if s.battery < 0.1 {
            return Ok(HealthStatus::Critical(format!(
                "battery critical: {:.0}%",
                s.battery * 100.0
            )));
        }
        if s.battery < 0.3 {
            return Ok(HealthStatus::Degraded(format!(
                "battery low: {:.0}%",
                s.battery * 100.0
            )));
        }
        Ok(HealthStatus::Healthy)
    }
}

// ---------------------------------------------------------------------------
// Simulator Fleet
// ---------------------------------------------------------------------------

/// Manages a fleet of simulated drones for swarm testing.
pub struct SimulatorFleet {
    /// All drones in the fleet.
    pub drones: Vec<SimulatorAdapter>,
    /// Optional CBF safety config. When set, `step_all_safe` applies
    /// barrier function corrections after each physics step.
    pub cbf_config: Option<CbfConfig>,
    /// No-fly zones for CBF avoidance.
    pub no_fly_zones: Vec<NoFlyZone>,
}

impl SimulatorFleet {
    /// Create a fleet of `n` drones in a grid formation.
    ///
    /// Drones are placed at (i*spacing, j*spacing, 0) in a roughly square grid.
    pub fn new_grid(n: usize, spacing: f64, config: SimulatorConfig) -> Self {
        let cols = (n as f64).sqrt().ceil() as usize;
        let drones = (0..n)
            .map(|i| {
                let row = i / cols;
                let col = i % cols;
                let pos = [col as f64 * spacing, row as f64 * spacing, 0.0];
                SimulatorAdapter::new(i as u32, pos, config.clone())
            })
            .collect();
        Self {
            drones,
            cbf_config: None,
            no_fly_zones: Vec::new(),
        }
    }

    /// Create a fleet of `n` drones all at the origin.
    pub fn new_at_origin(n: usize) -> Self {
        let drones = (0..n)
            .map(|i| SimulatorAdapter::new_default(i as u32))
            .collect();
        Self {
            drones,
            cbf_config: None,
            no_fly_zones: Vec::new(),
        }
    }

    /// Enable CBF safety layer for this fleet.
    pub fn with_cbf(mut self, config: CbfConfig) -> Self {
        self.cbf_config = Some(config);
        self
    }

    /// Add a no-fly zone.
    pub fn add_no_fly_zone(&mut self, zone: NoFlyZone) {
        self.no_fly_zones.push(zone);
    }

    /// Advance all drones by one time-step.
    pub fn step_all(&self) {
        for drone in &self.drones {
            drone.step();
        }
    }

    /// Advance all drones by `n` time-steps.
    pub fn step_all_n(&self, n: usize) {
        for _ in 0..n {
            self.step_all();
        }
    }

    /// Advance all drones by one step with CBF safety corrections.
    ///
    /// After normal physics, applies control barrier functions to correct
    /// velocities that would violate safety constraints (inter-drone
    /// separation, altitude bounds, no-fly zones).
    pub fn step_all_safe(&self) {
        // 1. Normal physics step.
        self.step_all();

        // 2. Apply CBF corrections if configured.
        let cbf_config = match &self.cbf_config {
            Some(cfg) => cfg,
            None => return,
        };

        // Gather all positions.
        let neighbor_states: Vec<(u32, NeighborState)> = self
            .drones
            .iter()
            .filter(|d| !d.has_failed())
            .filter_map(|d| {
                d.get_telemetry().ok().map(|t| {
                    (
                        d.id(),
                        NeighborState {
                            position: Vector3::new(t.position[0], t.position[1], t.position[2]),
                            velocity: Vector3::new(t.velocity[0], t.velocity[1], t.velocity[2]),
                        },
                    )
                })
            })
            .collect();

        // For each drone, compute CBF-filtered velocity.
        for drone in &self.drones {
            if drone.has_failed() {
                continue;
            }
            let telem = match drone.get_telemetry() {
                Ok(t) => t,
                Err(_) => continue,
            };

            let my_pos = Vector3::new(telem.position[0], telem.position[1], telem.position[2]);
            let my_vel = Vector3::new(telem.velocity[0], telem.velocity[1], telem.velocity[2]);

            // Collect neighbor states (exclude self).
            let neighbors: Vec<NeighborState> = neighbor_states
                .iter()
                .filter(|(id, _)| *id != drone.id())
                .map(|(_, state)| state.clone())
                .collect();

            let result = cbf::cbf_filter_with_neighbor_states(
                &my_pos,
                &my_vel,
                &neighbors,
                &self.no_fly_zones,
                cbf_config,
            );

            if result.any_active {
                // Apply corrected velocity directly to internal state.
                let mut s = drone.state.lock().expect("drone state mutex poisoned");
                s.velocity = [
                    result.safe_velocity.x,
                    result.safe_velocity.y,
                    result.safe_velocity.z,
                ];
            }
        }
    }

    /// Advance all drones by `n` steps with CBF safety corrections.
    pub fn step_all_safe_n(&self, n: usize) {
        for _ in 0..n {
            self.step_all_safe();
        }
    }

    /// Arm all drones in the fleet.
    pub fn arm_all(&self) -> Result<(), AdapterError> {
        for drone in &self.drones {
            drone.execute_action(&Action::Arm)?;
        }
        Ok(())
    }

    /// Get telemetry from all drones.
    pub fn get_all_telemetry(&self) -> Vec<Result<Telemetry, AdapterError>> {
        self.drones.iter().map(|d| d.get_telemetry()).collect()
    }

    /// Number of drones still operational.
    pub fn active_count(&self) -> usize {
        self.drones.iter().filter(|d| !d.has_failed()).count()
    }

    /// Get a reference to a drone by ID.
    pub fn get(&self, id: u32) -> Option<&SimulatorAdapter> {
        self.drones.iter().find(|d| d.id() == id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_takeoff_fly() {
        let sim = SimulatorAdapter::new_default(1);

        // Must arm first
        assert!(sim
            .send_waypoint(&Waypoint {
                lat: 10.0,
                lon: 0.0,
                alt: 0.0,
                speed: 5.0,
                heading: None,
            })
            .is_err());

        sim.execute_action(&Action::Arm).unwrap();
        let telem = sim.get_telemetry().unwrap();
        assert!(telem.armed);

        // Take off to 50m
        sim.execute_action(&Action::Takeoff(50.0)).unwrap();
        sim.step_n(500); // 50 seconds

        let telem = sim.get_telemetry().unwrap();
        // Should have climbed significantly
        assert!(telem.position[2] > 10.0, "alt = {}", telem.position[2]);
    }

    #[test]
    fn fly_to_waypoint() {
        let sim = SimulatorAdapter::new_default(1);
        sim.execute_action(&Action::Arm).unwrap();

        let wp = Waypoint {
            lat: 100.0,
            lon: 0.0,
            alt: 50.0,
            speed: 10.0,
            heading: None,
        };
        sim.send_waypoint(&wp).unwrap();

        // Run for a long time to converge
        sim.step_n(2000); // 200 seconds

        let dist = sim.distance_to([100.0, 0.0, 50.0]);
        assert!(dist < 5.0, "distance to target = {dist:.1}");
    }

    #[test]
    fn battery_drain() {
        let config = SimulatorConfig {
            battery_drain_rate: 0.01, // fast drain for testing
            ..Default::default()
        };
        let sim = SimulatorAdapter::new(1, [0.0, 0.0, 0.0], config);
        sim.execute_action(&Action::Arm).unwrap();
        sim.step_n(100); // 10 seconds

        let batt = sim.battery();
        assert!(batt < 1.0, "battery should drain: {batt}");
        assert!(batt > 0.0, "shouldn't be dead yet: {batt}");
    }

    #[test]
    fn health_check_battery_levels() {
        let config = SimulatorConfig {
            battery_drain_rate: 0.1, // very fast drain
            ..Default::default()
        };
        let sim = SimulatorAdapter::new(1, [0.0, 0.0, 0.0], config);
        sim.execute_action(&Action::Arm).unwrap();

        assert_eq!(sim.health_check().unwrap(), HealthStatus::Healthy);

        // Drain to low
        sim.step_n(70);
        let health = sim.health_check().unwrap();
        assert!(
            matches!(
                health,
                HealthStatus::Degraded(_) | HealthStatus::Critical(_)
            ),
            "health = {health:?}"
        );
    }

    #[test]
    fn fleet_grid() {
        let fleet = SimulatorFleet::new_grid(9, 10.0, SimulatorConfig::default());
        assert_eq!(fleet.drones.len(), 9);
        assert_eq!(fleet.active_count(), 9);

        // Check grid positions
        let t0 = fleet.drones[0].get_telemetry().unwrap();
        let t8 = fleet.drones[8].get_telemetry().unwrap();
        assert_eq!(t0.position, [0.0, 0.0, 0.0]);
        assert_eq!(t8.position, [20.0, 20.0, 0.0]);
    }

    #[test]
    fn fleet_arm_and_step() {
        let fleet = SimulatorFleet::new_at_origin(4);
        fleet.arm_all().unwrap();
        fleet.step_all_n(10);

        let telems = fleet.get_all_telemetry();
        assert_eq!(telems.len(), 4);
        for t in &telems {
            assert!(t.is_ok());
        }
    }

    #[test]
    fn wind_affects_position() {
        let config = SimulatorConfig {
            wind: [5.0, 0.0, 0.0],
            ..Default::default()
        };
        let sim = SimulatorAdapter::new(1, [0.0, 0.0, 0.0], config);
        sim.execute_action(&Action::Arm).unwrap();
        sim.step_n(100);

        let telem = sim.get_telemetry().unwrap();
        // Wind pushes in +x direction
        assert!(telem.position[0] > 0.0, "wind should push +x");
    }

    #[test]
    fn disarm_stops_motion() {
        let sim = SimulatorAdapter::new_default(1);
        sim.execute_action(&Action::Arm).unwrap();
        sim.execute_action(&Action::Takeoff(50.0)).unwrap();
        sim.step_n(50);

        sim.execute_action(&Action::Disarm).unwrap();
        let pos_before = sim.get_telemetry().unwrap().position;
        sim.step_n(100);
        let pos_after = sim.get_telemetry().unwrap().position;
        assert_eq!(pos_before, pos_after, "disarmed drone should not move");
    }

    #[test]
    fn rtl_returns_home() {
        let sim = SimulatorAdapter::new(1, [100.0, 100.0, 50.0], SimulatorConfig::default());
        sim.execute_action(&Action::Arm).unwrap();
        sim.execute_action(&Action::RTL).unwrap();
        sim.step_n(3000);

        let dist = sim.distance_to([0.0, 0.0, 0.0]);
        assert!(dist < 5.0, "RTL distance = {dist:.1}");
    }
}
