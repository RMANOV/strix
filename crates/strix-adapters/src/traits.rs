//! Core platform adapter trait and shared types.
//!
//! Every drone platform (MAVLink autopilot, ROS2 node, simulator) implements
//! [`PlatformAdapter`] so the upper layers can issue commands and read telemetry
//! without caring about the transport protocol.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can arise in platform adapter operations.
#[derive(Debug, Error)]
pub enum AdapterError {
    /// The platform is not connected or has timed out.
    #[error("connection lost: {0}")]
    ConnectionLost(String),

    /// The command was rejected by the autopilot.
    #[error("command rejected: {0}")]
    CommandRejected(String),

    /// Telemetry data is stale or missing.
    #[error("telemetry unavailable: {0}")]
    TelemetryUnavailable(String),

    /// The requested action is not supported on this platform.
    #[error("unsupported action: {0}")]
    UnsupportedAction(String),

    /// Generic I/O or transport error.
    #[error("transport error: {0}")]
    TransportError(String),

    /// Internal adapter error.
    #[error("internal error: {0}")]
    Internal(String),
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

/// Health status of a platform adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Everything nominal.
    Healthy,
    /// Degraded but still operational (e.g. weak link, sensor offline).
    Degraded(String),
    /// Critical — the platform should be recalled or landed.
    Critical(String),
}

// ---------------------------------------------------------------------------
// GPS
// ---------------------------------------------------------------------------

/// GPS fix quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpsFix {
    /// No satellite fix.
    NoFix,
    /// 2-dimensional fix (lat/lon only).
    Fix2D,
    /// Full 3-D fix.
    Fix3D,
    /// Differential GPS.
    DGps,
    /// RTK float solution.
    RtkFloat,
    /// RTK fixed solution (centimeter accuracy).
    RtkFixed,
}

// ---------------------------------------------------------------------------
// Flight mode
// ---------------------------------------------------------------------------

/// Normalised flight mode across platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FlightMode {
    /// Pilot has full control.
    Manual,
    /// Attitude stabilisation only.
    Stabilize,
    /// Guided mode — accepts individual waypoints.
    Guided,
    /// Autonomous mission execution.
    Auto,
    /// Return to launch.
    RTL,
    /// Autonomous landing.
    Land,
    /// Position hold / loiter.
    Loiter,
}

// ---------------------------------------------------------------------------
// Sensor type
// ---------------------------------------------------------------------------

/// On-board sensor payload categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensorType {
    /// Electro-optical / visible-light camera.
    Camera,
    /// Thermal / infrared imager.
    Thermal,
    /// Light Detection and Ranging.
    Lidar,
    /// Radio Detection and Ranging.
    Radar,
    /// Electronic Warfare suite.
    EW,
    /// Underwater acoustic sensor.
    Sonar,
}

// ---------------------------------------------------------------------------
// Waypoint
// ---------------------------------------------------------------------------

/// A geographic waypoint the drone should fly through.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Waypoint {
    /// Latitude (degrees) or local-X (meters).
    pub lat: f64,
    /// Longitude (degrees) or local-Y (meters).
    pub lon: f64,
    /// Altitude above ground / MSL (meters).
    pub alt: f64,
    /// Desired speed at this waypoint (m/s).
    pub speed: f64,
    /// Desired heading at arrival (radians), if any.
    pub heading: Option<f64>,
}

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

/// Snapshot of the drone's current state as reported by the autopilot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Telemetry {
    /// Position: `[x, y, z]` in local NED or `[lat, lon, alt]`.
    pub position: [f64; 3],
    /// Velocity: `[vx, vy, vz]` in m/s.
    pub velocity: [f64; 3],
    /// Attitude: `[roll, pitch, yaw]` in radians.
    pub attitude: [f64; 3],
    /// Battery state of charge `[0.0, 1.0]`.
    pub battery: f64,
    /// Current GPS fix quality.
    pub gps_fix: GpsFix,
    /// Whether the motors are armed.
    pub armed: bool,
    /// Current flight mode.
    pub mode: FlightMode,
    /// Timestamp (seconds since epoch or monotonic clock).
    pub timestamp: f64,
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// A discrete command the swarm orchestrator can issue to a drone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    /// Arm the motors.
    Arm,
    /// Disarm the motors.
    Disarm,
    /// Take off to the specified altitude (meters AGL).
    Takeoff(f64),
    /// Land at the current position.
    Land,
    /// Return to launch.
    RTL,
    /// Fly to a specific waypoint.
    GoTo(Waypoint),
    /// Set cruise speed (m/s).
    SetSpeed(f64),
    /// Switch flight mode.
    SetMode(FlightMode),
    /// Release payload.
    PayloadDrop,
    /// Trigger camera capture.
    CameraCapture,
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

/// Static capabilities descriptor for a platform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capabilities {
    /// Maximum airspeed (m/s).
    pub max_speed: f64,
    /// Maximum operating altitude (meters AGL).
    pub max_altitude: f64,
    /// Flight endurance (seconds at cruise speed).
    pub endurance: f64,
    /// On-board sensors.
    pub sensors: Vec<SensorType>,
    /// Maximum payload mass (kg).
    pub payload_kg: f64,
    /// Reliable communications range (meters).
    pub comms_range: f64,
}

// ---------------------------------------------------------------------------
// Platform Adapter Trait
// ---------------------------------------------------------------------------

/// Unified interface for controlling a drone platform.
///
/// Implementations translate between the STRIX command/telemetry model and
/// whatever protocol the actual hardware speaks (MAVLink, ROS2, DDS, …).
pub trait PlatformAdapter: Send + Sync {
    /// Unique identifier for this drone instance.
    fn id(&self) -> u32;

    /// Human-readable platform name (e.g. "PX4 Quadrotor", "ArduRover").
    fn platform_name(&self) -> &str;

    /// Send a single waypoint to the autopilot.
    fn send_waypoint(&self, wp: &Waypoint) -> Result<(), AdapterError>;

    /// Read the latest telemetry snapshot.
    fn get_telemetry(&self) -> Result<Telemetry, AdapterError>;

    /// Execute a discrete action command.
    fn execute_action(&self, action: &Action) -> Result<(), AdapterError>;

    /// Static capabilities of this platform.
    fn capabilities(&self) -> &Capabilities;

    /// Whether the data link to the platform is alive.
    fn is_connected(&self) -> bool;

    /// Run a health check and return the platform's status.
    fn health_check(&self) -> Result<HealthStatus, AdapterError>;
}
