//! MAVLink platform adapter stub (PX4 / ArduPilot).
//!
//! This module defines the structure for communicating with MAVLink-based
//! autopilots over UDP or TCP. The actual MAVLink message encoding/decoding
//! is not yet implemented — all methods return stub data.

use crate::traits::*;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Transport protocol for the MAVLink connection.
#[derive(Debug, Clone)]
pub enum MavlinkTransport {
    /// UDP socket (typical for SITL and telemetry radios).
    Udp(String),
    /// TCP stream (typical for companion computers).
    Tcp(String),
    /// Serial port (direct autopilot connection).
    Serial {
        /// Device path, e.g. "/dev/ttyUSB0".
        port: String,
        /// Baud rate, e.g. 57600.
        baud: u32,
    },
}

/// Configuration for a MAVLink adapter instance.
#[derive(Debug, Clone)]
pub struct MavlinkConfig {
    /// Transport layer to use.
    pub transport: MavlinkTransport,
    /// MAVLink system ID for this adapter (1-255).
    pub system_id: u8,
    /// MAVLink component ID (usually 1 for GCS).
    pub component_id: u8,
    /// Target system ID on the autopilot side.
    pub target_system: u8,
    /// Target component ID on the autopilot side.
    pub target_component: u8,
    /// Heartbeat interval in milliseconds.
    pub heartbeat_interval_ms: u64,
    /// Telemetry request rate in Hz.
    pub telemetry_rate_hz: f64,
}

impl Default for MavlinkConfig {
    fn default() -> Self {
        Self {
            transport: MavlinkTransport::Udp("127.0.0.1:14550".into()),
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
/// **STUB** — all methods return dummy values. Real MAVLink encoding
/// (via the `mavlink` crate) will be integrated in a future milestone.
pub struct MavlinkAdapter {
    /// Unique drone ID in the STRIX fleet.
    drone_id: u32,
    /// Connection configuration.
    config: MavlinkConfig,
    /// Platform capability descriptor.
    caps: Capabilities,
    /// Simulated connection state.
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
    /// **STUB** — immediately marks the adapter as connected.
    pub fn connect(&mut self) -> Result<(), AdapterError> {
        // TODO: open UDP/TCP/Serial, spawn heartbeat task
        self.connected = true;
        Ok(())
    }

    /// Disconnect from the autopilot.
    pub fn disconnect(&mut self) {
        self.connected = false;
    }

    /// Return the MAVLink configuration.
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
            Action::Arm => { /* MAV_CMD_COMPONENT_ARM_DISARM, param1=1 */ }
            Action::Disarm => { /* MAV_CMD_COMPONENT_ARM_DISARM, param1=0 */ }
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
}
