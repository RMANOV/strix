//! MAVLink platform adapter (PX4 / ArduPilot).
//!
//! ## Modes
//!
//! | Feature flag | Behaviour |
//! |---|---|
//! | *(none)* | **Stub** — returns deterministic dummy data; zero external deps; safe for simulation |
//! | `mavlink-hw` | **Hardware** — real MAVLink over UDP / TCP / Serial via the `mavlink` crate |
//!
//! The stub mode is intentional and permanent; it lets the rest of the STRIX
//! stack compile and run in simulation without requiring flight hardware or a
//! SITL instance.  Hardware mode adds the `hw` sub-module that owns the actual
//! connection, heartbeat sending, and message parsing helpers.

use crate::traits::*;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Transport protocol for the MAVLink connection.
#[derive(Debug, Clone)]
pub enum MavlinkTransport {
    /// UDP socket (typical for SITL and telemetry radios).
    ///
    /// Use `"udpin:0.0.0.0:14550"` to listen or `"udpout:127.0.0.1:14550"` to
    /// send — the string is passed verbatim to `mavlink::connect()`.
    Udp(String),
    /// TCP stream (typical for companion computers).
    ///
    /// Use `"tcpin:0.0.0.0:5760"` or `"tcpout:127.0.0.1:5760"`.
    Tcp(String),
    /// Serial port (direct autopilot connection).
    Serial {
        /// Device path, e.g. `"/dev/ttyUSB0"`.
        port: String,
        /// Baud rate, e.g. `57600`.
        baud: u32,
    },
}

impl MavlinkTransport {
    /// Render the transport as a `mavlink::connect()`-compatible address string.
    ///
    /// # Format reference
    /// - UDP listen: `"udpin:0.0.0.0:14550"`
    /// - UDP send  : `"udpout:127.0.0.1:14550"`
    /// - TCP client: `"tcpout:127.0.0.1:5760"`
    /// - Serial    : `"serial:/dev/ttyUSB0:57600"`
    pub fn to_address(&self) -> String {
        match self {
            MavlinkTransport::Udp(addr) => addr.clone(),
            MavlinkTransport::Tcp(addr) => addr.clone(),
            MavlinkTransport::Serial { port, baud } => {
                format!("serial:{}:{}", port, baud)
            }
        }
    }
}

/// Configuration for a MAVLink adapter instance.
#[derive(Debug, Clone)]
pub struct MavlinkConfig {
    /// Transport layer to use.
    pub transport: MavlinkTransport,
    /// MAVLink system ID for this GCS adapter (1–255).
    pub system_id: u8,
    /// MAVLink component ID (typically `1` for GCS).
    pub component_id: u8,
    /// Target system ID on the autopilot side.
    pub target_system: u8,
    /// Target component ID on the autopilot side.
    pub target_component: u8,
    /// Heartbeat transmit interval (milliseconds).
    pub heartbeat_interval_ms: u64,
    /// Desired telemetry stream rate (Hz).
    pub telemetry_rate_hz: f64,
}

impl Default for MavlinkConfig {
    fn default() -> Self {
        Self {
            // Listen on the standard SITL UDP port.
            transport: MavlinkTransport::Udp("udpin:0.0.0.0:14550".into()),
            system_id: 255,
            component_id: 1,
            target_system: 1,
            target_component: 1,
            heartbeat_interval_ms: 1000,
            telemetry_rate_hz: 10.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

/// MAVLink-based platform adapter for PX4 and ArduPilot autopilots.
///
/// In **stub mode** (no `mavlink-hw` feature) all methods return deterministic
/// dummy values so the full STRIX swarm stack can be exercised in simulation
/// without any flight hardware.
///
/// In **hardware mode** (`mavlink-hw` feature) call [`MavlinkAdapter::connect`]
/// to open the transport.  Telemetry is updated by the background receive task
/// started during `connect`.
pub struct MavlinkAdapter {
    /// Unique drone ID in the STRIX fleet.
    drone_id: u32,
    /// Connection configuration.
    config: MavlinkConfig,
    /// Platform capability descriptor.
    caps: Capabilities,
    /// Connection state.
    ///
    /// In stub mode this is toggled by [`connect`] / [`disconnect`].
    /// In hardware mode it reflects whether the underlying socket is open.
    connected: bool,
}

impl MavlinkAdapter {
    /// Create a new (disconnected) MAVLink adapter.
    pub fn new(drone_id: u32, config: MavlinkConfig) -> Self {
        Self {
            drone_id,
            config,
            caps: Capabilities {
                max_speed: 20.0,
                max_altitude: 500.0,
                endurance: 1800.0,
                sensors: vec![SensorType::Camera],
                payload_kg: 1.0,
                comms_range: 5000.0,
            },
            connected: false,
        }
    }

    /// Attempt to open the transport and start the heartbeat loop.
    ///
    /// **Stub** — immediately marks the adapter as connected without opening
    /// any socket.  Use the `mavlink-hw` feature for real hardware support.
    pub fn connect(&mut self) -> Result<(), AdapterError> {
        // TODO(mavlink-hw): open UDP/TCP/Serial, spawn heartbeat + receive tasks.
        self.connected = true;
        Ok(())
    }

    /// Disconnect from the autopilot and stop background tasks.
    pub fn disconnect(&mut self) {
        self.connected = false;
    }

    /// Return the active MAVLink configuration.
    pub fn config(&self) -> &MavlinkConfig {
        &self.config
    }
}

impl PlatformAdapter for MavlinkAdapter {
    fn id(&self) -> u32 {
        self.drone_id
    }

    fn platform_name(&self) -> &str {
        "MAVLink (PX4/ArduPilot)"
    }

    fn send_waypoint(&self, _wp: &Waypoint) -> Result<(), AdapterError> {
        if !self.connected {
            return Err(AdapterError::ConnectionLost("not connected".into()));
        }
        // TODO: encode SET_POSITION_TARGET_GLOBAL_INT or MISSION_ITEM
        Ok(())
    }

    fn get_telemetry(&self) -> Result<Telemetry, AdapterError> {
        if !self.connected {
            return Err(AdapterError::ConnectionLost("not connected".into()));
        }
        // STUB: return zero-state telemetry
        Ok(Telemetry {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            attitude: [0.0, 0.0, 0.0],
            battery: 1.0,
            gps_fix: GpsFix::Fix3D,
            armed: false,
            mode: FlightMode::Stabilize,
            timestamp: 0.0,
        })
    }

    fn execute_action(&self, action: &Action) -> Result<(), AdapterError> {
        if !self.connected {
            return Err(AdapterError::ConnectionLost("not connected".into()));
        }
        // TODO: translate action → MAVLink COMMAND_LONG / COMMAND_INT
        match action {
            Action::Arm => { /* MAV_CMD_COMPONENT_ARM_DISARM param1=1 */ }
            Action::Disarm => { /* MAV_CMD_COMPONENT_ARM_DISARM param1=0 */ }
            Action::Takeoff(_alt) => { /* MAV_CMD_NAV_TAKEOFF */ }
            Action::Land => { /* MAV_CMD_NAV_LAND */ }
            Action::RTL => { /* MAV_CMD_NAV_RETURN_TO_LAUNCH */ }
            Action::GoTo(wp) => {
                self.send_waypoint(wp)?;
            }
            Action::SetSpeed(_spd) => { /* MAV_CMD_DO_CHANGE_SPEED */ }
            Action::SetMode(_mode) => { /* SET_MODE message */ }
            Action::PayloadDrop => { /* MAV_CMD_DO_SET_SERVO */ }
            Action::CameraCapture => { /* MAV_CMD_IMAGE_START_CAPTURE */ }
        }
        Ok(())
    }

    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn health_check(&self) -> Result<HealthStatus, AdapterError> {
        if !self.connected {
            return Err(AdapterError::ConnectionLost("not connected".into()));
        }
        // STUB: always healthy when connected
        Ok(HealthStatus::Healthy)
    }
}

// ---------------------------------------------------------------------------
// Hardware connection module (mavlink-hw feature only)
// ---------------------------------------------------------------------------

/// Real MAVLink connection scaffolding.
///
/// This module is compiled **only** when the `mavlink-hw` Cargo feature is
/// enabled.  It provides:
///
/// - [`MavlinkConnection`] — wrapper around the blocking `mavlink::Connection`
///   that owns the socket, tracks heartbeat timestamps, and caches the last
///   parsed telemetry.
/// - Static helpers for parsing raw MAVLink message structs into STRIX types.
///
/// # Usage pattern
///
/// ```rust,ignore
/// // Feature gate: only available with --features mavlink-hw
/// use strix_adapters::mavlink::MavlinkConnection;
///
/// let conn = MavlinkConnection::connect("udpin:0.0.0.0:14550")?;
/// conn.send_heartbeat()?;
///
/// loop {
///     let (_header, msg) = conn.recv()?;
///     // match msg { MavMessage::GLOBAL_POSITION_INT(data) => { … } … }
/// }
/// ```
#[cfg(feature = "mavlink-hw")]
pub mod hw {
    use super::*;

    use mavlink::ardupilotmega::{
        MavAutopilot, MavMessage, MavModeFlag, MavState, MavType, ATTITUDE_DATA,
        GLOBAL_POSITION_INT_DATA, HEARTBEAT_DATA,
    };
    use mavlink::{Connection, MavConnection, MavHeader};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    // -----------------------------------------------------------------------
    // MavlinkConnection
    // -----------------------------------------------------------------------

    /// Wrapper around a blocking `mavlink::Connection` for ArduPilot/PX4.
    ///
    /// Cloning the `Arc` fields lets a background receive task share access to
    /// `last_telemetry` and `last_heartbeat` while the foreground issues
    /// commands on the same connection.
    pub struct MavlinkConnection {
        /// Underlying blocking MAVLink connection.
        ///
        /// `Connection<MavMessage>` is the concrete type returned by
        /// `mavlink::connect::<MavMessage>(address)`.  It implements
        /// `MavConnection<MavMessage>` which provides `send()` and `recv()`.
        connection: Arc<Connection<MavMessage>>,
        /// Last parsed STRIX telemetry snapshot (updated by the receive loop).
        pub last_telemetry: Arc<Mutex<Option<Telemetry>>>,
        /// Monotonic timestamp of the last HEARTBEAT received.
        pub last_heartbeat: Arc<Mutex<Option<Instant>>>,
        /// System-ID of the vehicle we are talking to.
        pub vehicle_system_id: u8,
    }

    impl MavlinkConnection {
        /// Open a blocking MAVLink connection to `address`.
        ///
        /// # Address formats
        ///
        /// | Pattern | Example |
        /// |---|---|
        /// | UDP listen | `"udpin:0.0.0.0:14550"` |
        /// | UDP send   | `"udpout:127.0.0.1:14550"` |
        /// | TCP client | `"tcpout:127.0.0.1:5760"` |
        /// | Serial     | `"serial:/dev/ttyUSB0:57600"` |
        ///
        /// # Errors
        ///
        /// Returns [`AdapterError::ConnectionLost`] if the OS-level socket /
        /// port cannot be opened.
        pub fn connect(address: &str) -> Result<Self, AdapterError> {
            let connection = mavlink::connect::<MavMessage>(address).map_err(|e| {
                AdapterError::ConnectionLost(format!(
                    "MAVLink connect to '{}' failed: {}",
                    address, e
                ))
            })?;

            Ok(Self {
                connection: Arc::new(connection),
                last_telemetry: Arc::new(Mutex::new(None)),
                last_heartbeat: Arc::new(Mutex::new(None)),
                vehicle_system_id: 1,
            })
        }

        /// Transmit a GCS heartbeat so the autopilot knows we are alive.
        ///
        /// Sends a `HEARTBEAT` with system-ID 255 (GCS) and component-ID 1.
        /// Call this on a 1 Hz timer from the background heartbeat task.
        ///
        /// # Errors
        ///
        /// Returns [`AdapterError::ConnectionLost`] on write failure.
        pub fn send_heartbeat(&self) -> Result<(), AdapterError> {
            let msg = MavMessage::HEARTBEAT(HEARTBEAT_DATA {
                custom_mode: 0,
                // Field is named `mavtype` in rust-mavlink 0.17 (not `r#type`).
                mavtype: MavType::MAV_TYPE_GCS,
                autopilot: MavAutopilot::MAV_AUTOPILOT_INVALID,
                base_mode: MavModeFlag::empty(),
                system_status: MavState::MAV_STATE_ACTIVE,
                mavlink_version: 3,
            });

            let header = MavHeader {
                system_id: 255,
                component_id: 1,
                sequence: 0,
            };

            self.connection
                .send(&header, &msg)
                .map(|_| ())
                .map_err(|e| AdapterError::ConnectionLost(format!("Heartbeat send failed: {}", e)))
        }

        /// Block until one MAVLink message is received.
        ///
        /// Returns the raw `(MavHeader, MavMessage)` pair so the caller can
        /// match on any message type.
        ///
        /// # Errors
        ///
        /// Returns [`AdapterError::ConnectionLost`] on read failure or
        /// deserialisation error.
        pub fn recv(&self) -> Result<(MavHeader, MavMessage), AdapterError> {
            self.connection
                .recv()
                .map_err(|e| AdapterError::ConnectionLost(format!("MAVLink recv failed: {}", e)))
        }

        // -------------------------------------------------------------------
        // Message parsers (static helpers)
        // -------------------------------------------------------------------

        /// Convert a `GLOBAL_POSITION_INT` message into a `[lat, lon, alt]`
        /// triple using STRIX units (degrees, degrees, metres MSL).
        ///
        /// MAVLink encodes lat/lon as integer degrees × 10⁷ and altitude as
        /// millimetres MSL.
        ///
        /// # Example
        ///
        /// ```rust,ignore
        /// if let MavMessage::GLOBAL_POSITION_INT(data) = msg {
        ///     let pos = MavlinkConnection::parse_position(&data);
        ///     // pos == [lat_deg, lon_deg, alt_m]
        /// }
        /// ```
        pub fn parse_position(msg: &GLOBAL_POSITION_INT_DATA) -> [f64; 3] {
            [
                msg.lat as f64 / 1e7,    // integer degrees×1e7 → degrees
                msg.lon as f64 / 1e7,    // integer degrees×1e7 → degrees
                msg.alt as f64 / 1000.0, // mm MSL → m MSL
            ]
        }

        /// Convert a `GLOBAL_POSITION_INT` message into a `[vx, vy, vz]`
        /// velocity triple (m/s).
        ///
        /// MAVLink encodes ground-speed components as cm/s (`i16`).
        pub fn parse_velocity(msg: &GLOBAL_POSITION_INT_DATA) -> [f64; 3] {
            [
                msg.vx as f64 / 100.0, // cm/s → m/s  (positive = North)
                msg.vy as f64 / 100.0, // cm/s → m/s  (positive = East)
                msg.vz as f64 / 100.0, // cm/s → m/s  (positive = Down)
            ]
        }

        /// Extract `[roll, pitch, yaw]` in radians from an `ATTITUDE` message.
        ///
        /// MAVLink already uses radians (`f32`), so this is a lossless
        /// upcast.
        pub fn parse_attitude(msg: &ATTITUDE_DATA) -> [f64; 3] {
            [msg.roll as f64, msg.pitch as f64, msg.yaw as f64]
        }

        /// Derive a normalised battery level `[0.0, 1.0]` from a
        /// `SYS_STATUS` `battery_remaining` field.
        ///
        /// The field is `i8` and holds a percentage (0–100), or `-1` when the
        /// autopilot does not report it.  We clamp to `[0.0, 1.0]` and fall
        /// back to `1.0` for the `-1` sentinel.
        pub fn parse_battery_fraction(battery_remaining: i8) -> f64 {
            if battery_remaining < 0 {
                1.0 // autopilot did not send battery info — assume full
            } else {
                (battery_remaining as f64 / 100.0).clamp(0.0, 1.0)
            }
        }

        // -------------------------------------------------------------------
        // Arc accessors (for background task cloning)
        // -------------------------------------------------------------------

        /// Clone the Arc over the underlying connection so a background thread
        /// can call `recv()` independently.
        pub fn connection_arc(&self) -> Arc<Connection<MavMessage>> {
            Arc::clone(&self.connection)
        }

        /// Clone the Arc over the cached telemetry for sharing with a receive
        /// task.
        pub fn telemetry_arc(&self) -> Arc<Mutex<Option<Telemetry>>> {
            Arc::clone(&self.last_telemetry)
        }

        /// Clone the Arc over the heartbeat timestamp for the receive task.
        pub fn heartbeat_arc(&self) -> Arc<Mutex<Option<Instant>>> {
            Arc::clone(&self.last_heartbeat)
        }
    }
}

#[cfg(feature = "mavlink-hw")]
pub use hw::MavlinkConnection;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Stub tests (always run) -------------------------------------------

    #[test]
    fn connect_disconnect() {
        let mut adapter = MavlinkAdapter::new(1, MavlinkConfig::default());
        assert!(!adapter.is_connected());
        adapter.connect().unwrap();
        assert!(adapter.is_connected());
        adapter.disconnect();
        assert!(!adapter.is_connected());
    }

    #[test]
    fn telemetry_requires_connection() {
        let adapter = MavlinkAdapter::new(1, MavlinkConfig::default());
        assert!(adapter.get_telemetry().is_err());
    }

    #[test]
    fn stub_telemetry() {
        let mut adapter = MavlinkAdapter::new(42, MavlinkConfig::default());
        adapter.connect().unwrap();
        let telem = adapter.get_telemetry().unwrap();
        assert_eq!(telem.battery, 1.0);
        assert_eq!(adapter.id(), 42);
        assert_eq!(adapter.platform_name(), "MAVLink (PX4/ArduPilot)");
    }

    #[test]
    fn transport_address_formatting() {
        let udp = MavlinkTransport::Udp("udpin:0.0.0.0:14550".into());
        assert_eq!(udp.to_address(), "udpin:0.0.0.0:14550");

        let tcp = MavlinkTransport::Tcp("tcpout:127.0.0.1:5760".into());
        assert_eq!(tcp.to_address(), "tcpout:127.0.0.1:5760");

        let serial = MavlinkTransport::Serial {
            port: "/dev/ttyUSB0".into(),
            baud: 57600,
        };
        assert_eq!(serial.to_address(), "serial:/dev/ttyUSB0:57600");
    }

    // --- Hardware tests (mavlink-hw feature only) --------------------------

    #[cfg(feature = "mavlink-hw")]
    mod hw_tests {
        use super::super::hw::MavlinkConnection;
        use mavlink::ardupilotmega::{ATTITUDE_DATA, GLOBAL_POSITION_INT_DATA};

        /// Verify lat/lon/alt unit conversions for parse_position().
        ///
        /// MAVLink encodes:
        ///   lat/lon in integer degrees × 1e7
        ///   alt     in millimetres MSL
        #[test]
        fn test_parse_position() {
            let data = GLOBAL_POSITION_INT_DATA {
                time_boot_ms: 0,
                lat: 473_976_320, // 47.397632° N (near PX4 SITL home)
                lon: 85_455_939,  // 8.545594° E
                alt: 488_000,     // 488 m MSL
                relative_alt: 0,
                vx: 0,
                vy: 0,
                vz: 0,
                hdg: 0,
            };

            let pos = MavlinkConnection::parse_position(&data);

            // Tolerate f64 rounding at the 7th decimal place.
            assert!(
                (pos[0] - 47.397_632_0).abs() < 1e-6,
                "lat mismatch: {}",
                pos[0]
            );
            assert!(
                (pos[1] - 8.545_593_9).abs() < 1e-6,
                "lon mismatch: {}",
                pos[1]
            );
            assert!((pos[2] - 488.0).abs() < 1e-3, "alt mismatch: {}", pos[2]);
        }

        /// Verify velocity conversion from cm/s to m/s for parse_velocity().
        #[test]
        fn test_parse_velocity() {
            let data = GLOBAL_POSITION_INT_DATA {
                time_boot_ms: 0,
                lat: 0,
                lon: 0,
                alt: 0,
                relative_alt: 0,
                vx: 500,  //  5.00 m/s North
                vy: -200, // -2.00 m/s East (i.e. West)
                vz: 100,  //  1.00 m/s Down
                hdg: 0,
            };

            let vel = MavlinkConnection::parse_velocity(&data);
            assert!((vel[0] - 5.0).abs() < 1e-9, "vx mismatch");
            assert!((vel[1] - -2.0).abs() < 1e-9, "vy mismatch");
            assert!((vel[2] - 1.0).abs() < 1e-9, "vz mismatch");
        }

        /// Verify roll/pitch/yaw passthrough (f32 → f64 upcast).
        #[test]
        fn test_parse_attitude() {
            use std::f64::consts::PI;

            let data = ATTITUDE_DATA {
                time_boot_ms: 0,
                roll: 0.1_f32,
                pitch: 0.2_f32,
                yaw: (PI as f32) / 4.0,
                rollspeed: 0.0,
                pitchspeed: 0.0,
                yawspeed: 0.0,
            };

            let att = MavlinkConnection::parse_attitude(&data);

            assert!((att[0] - 0.1_f64).abs() < 1e-6, "roll mismatch");
            assert!((att[1] - 0.2_f64).abs() < 1e-6, "pitch mismatch");
            assert!((att[2] - (PI / 4.0)).abs() < 1e-5, "yaw mismatch");
        }

        /// Verify battery fraction normalisation.
        #[test]
        fn test_parse_battery_fraction() {
            assert_eq!(MavlinkConnection::parse_battery_fraction(-1), 1.0);
            assert_eq!(MavlinkConnection::parse_battery_fraction(0), 0.0);
            assert!((MavlinkConnection::parse_battery_fraction(75) - 0.75).abs() < 1e-9);
            assert_eq!(MavlinkConnection::parse_battery_fraction(100), 1.0);
            // clamp: value > 100 is invalid but should not panic
            assert_eq!(MavlinkConnection::parse_battery_fraction(127), 1.0);
        }
    }
}
