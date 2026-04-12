//! Core platform adapter trait and shared types.
//!
//! Every drone platform (MAVLink autopilot, ROS2 node, simulator) implements
//! [`PlatformAdapter`] so the upper layers can issue commands and read telemetry
//! without caring about the transport protocol.
//!
//! ## Migration path
//!
//! New code should prefer the richer sub-traits ([`TelemetrySource`],
//! [`CommandSink`], [`PlatformInfo`]) and [`RichTelemetry`] for explicit frame
//! tracking and command lifecycle.  The original [`PlatformAdapter`] and
//! [`Telemetry`] are preserved for backward compatibility.

use serde::{Deserialize, Serialize};
use strix_core::frames::{Frame, NedPosition};
use strix_core::units::Timestamp;
use thiserror::Error;

use crate::command::{CommandAcceptance, CommandId, CommandOutcome};

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
// Waypoint — frame-safe variants
// ---------------------------------------------------------------------------

/// A geographic waypoint in WGS-84 (for MAVLink / global missions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoWaypoint {
    /// Latitude in degrees (positive = north).
    pub lat_deg: f64,
    /// Longitude in degrees (positive = east).
    pub lon_deg: f64,
    /// Altitude above MSL (metres).
    pub alt_m: f64,
    /// Desired speed at this waypoint (m/s).
    pub speed: f64,
    /// Desired heading at arrival (radians), if any.
    pub heading: Option<f64>,
}

/// A local waypoint in NED frame (metres, relative to mission origin).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalWaypoint {
    /// North (metres from origin).
    pub north: f64,
    /// East (metres from origin).
    pub east: f64,
    /// Down (metres, positive = below origin; negate for altitude AGL).
    pub down: f64,
    /// Desired speed at this waypoint (m/s).
    pub speed: f64,
    /// Desired heading at arrival (radians), if any.
    pub heading: Option<f64>,
}

/// Frame-dispatched waypoint target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WaypointTarget {
    /// Global WGS-84 waypoint.
    Geo(GeoWaypoint),
    /// Local NED waypoint.
    Local(LocalWaypoint),
}

/// Legacy waypoint — kept for backward compatibility during migration.
/// New code should use [`WaypointTarget`] instead.
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

impl From<GeoWaypoint> for Waypoint {
    fn from(g: GeoWaypoint) -> Self {
        Self {
            lat: g.lat_deg,
            lon: g.lon_deg,
            alt: g.alt_m,
            speed: g.speed,
            heading: g.heading,
        }
    }
}

impl From<LocalWaypoint> for Waypoint {
    fn from(l: LocalWaypoint) -> Self {
        Self {
            lat: l.north,
            lon: l.east,
            alt: -l.down,
            speed: l.speed,
            heading: l.heading,
        }
    }
}

impl From<WaypointTarget> for Waypoint {
    fn from(wt: WaypointTarget) -> Self {
        match wt {
            WaypointTarget::Geo(g) => g.into(),
            WaypointTarget::Local(l) => l.into(),
        }
    }
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
// RichTelemetry — frame-safe, provenance-tagged telemetry
// ---------------------------------------------------------------------------

/// Enhanced telemetry with provenance and explicit frame information.
///
/// New code should prefer this over [`Telemetry`] because it:
/// - carries an explicit [`Frame`] tag to prevent NED/ENU confusion,
/// - records both *observed_at* (sensor sample time) and *received_at*
///   (adapter ingestion time) so staleness can be detected,
/// - optionally carries position uncertainty for sensor fusion consumers.
///
/// Use [`From<RichTelemetry> for Telemetry`] to downcast when interacting
/// with legacy code that still expects the flat [`Telemetry`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RichTelemetry {
    /// Position in the coordinate frame named by [`Self::frame`].
    pub position: NedPosition,
    /// Velocity `[vx, vy, vz]` in m/s (NED frame).
    pub velocity: [f64; 3],
    /// Attitude `[roll, pitch, yaw]` in radians.
    pub attitude: [f64; 3],
    /// Battery state of charge `[0, 1]`.
    pub battery: f64,
    /// GPS fix quality.
    pub gps_fix: GpsFix,
    /// Whether motors are armed.
    pub armed: bool,
    /// Current flight mode.
    pub mode: FlightMode,
    /// When the sensor sampled this data (platform/sensor clock).
    pub observed_at: Timestamp,
    /// When the adapter received this data (adapter clock).
    pub received_at: Timestamp,
    /// Coordinate frame of [`Self::position`].
    pub frame: Frame,
    /// Optional position uncertainty — diagonal of position covariance (NED), in m².
    pub position_covariance: Option<[f64; 3]>,
}

impl From<RichTelemetry> for Telemetry {
    fn from(r: RichTelemetry) -> Self {
        let p = r.position.0;
        Self {
            position: [p.x, p.y, p.z],
            velocity: r.velocity,
            attitude: r.attitude,
            battery: r.battery,
            gps_fix: r.gps_fix,
            armed: r.armed,
            mode: r.mode,
            timestamp: r.observed_at.as_secs(),
        }
    }
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

// ---------------------------------------------------------------------------
// Sub-traits for incremental adoption
// ---------------------------------------------------------------------------

/// Telemetry source with provenance and frame tracking.
///
/// Implementors provide richer telemetry than the legacy [`PlatformAdapter`]
/// by returning [`RichTelemetry`] which carries frame tags and dual timestamps.
pub trait TelemetrySource: Send + Sync {
    /// Unique drone identifier.
    fn id(&self) -> u32;

    /// Read the latest rich telemetry with explicit frame and timing info.
    fn get_rich_telemetry(&self) -> Result<RichTelemetry, AdapterError>;

    /// Whether the data link to the platform is alive.
    fn is_connected(&self) -> bool;
}

/// Command sink with lifecycle tracking.
///
/// Unlike [`PlatformAdapter::send_waypoint`] / [`PlatformAdapter::execute_action`]
/// which fire-and-forget, this trait returns a [`CommandId`] that can be used to
/// poll completion via [`CommandSink::command_status`].
pub trait CommandSink: Send + Sync {
    /// Submit a waypoint and receive a command ID (or rejection).
    fn submit_waypoint(&self, wp: &WaypointTarget) -> Result<CommandAcceptance, AdapterError>;

    /// Submit a discrete action and receive a command ID (or rejection).
    fn submit_action(&self, action: &Action) -> Result<CommandAcceptance, AdapterError>;

    /// Check the current outcome of a previously submitted command.
    ///
    /// Returns [`CommandOutcome::Unknown`] if the ID is unrecognised or has
    /// expired from the adapter's tracking window.
    fn command_status(&self, id: CommandId) -> CommandOutcome;
}

/// Platform identity and health information.
///
/// Separate from [`TelemetrySource`] and [`CommandSink`] so it can be
/// implemented by lightweight proxy objects that don't hold live connections.
pub trait PlatformInfo: Send + Sync {
    /// Human-readable platform name (e.g. "PX4 Quadrotor", "STRIX Simulator").
    fn platform_name(&self) -> &str;

    /// Static capability descriptor for this platform.
    fn capabilities(&self) -> &Capabilities;

    /// Run a health check and return the platform's status.
    fn health_check(&self) -> Result<HealthStatus, AdapterError>;
}
