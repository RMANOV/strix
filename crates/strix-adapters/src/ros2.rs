//! ROS2 platform adapter stub for ground vehicles (UGV) and surface vessels (USV).
//!
//! Communicates via ROS2 topics and services. The actual `rclrs` integration
//! is not yet wired — all methods return stub data.

use crate::traits::*;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// The type of robot this ROS2 adapter controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RobotType {
    /// Unmanned Ground Vehicle.
    Ugv,
    /// Unmanned Surface Vessel.
    Usv,
    /// Unmanned Aerial Vehicle (ROS2 flight stack).
    Uav,
}

/// ROS2 topic namespace and QoS configuration.
#[derive(Debug, Clone)]
pub struct Ros2Config {
    /// ROS2 node name for this adapter.
    pub node_name: String,
    /// Namespace prefix for all topics (e.g. "/drone_03").
    pub namespace: String,
    /// Topic for publishing velocity commands.
    pub cmd_vel_topic: String,
    /// Topic for receiving odometry.
    pub odom_topic: String,
    /// Topic for receiving battery state.
    pub battery_topic: String,
    /// Robot type determines which capabilities are reported.
    pub robot_type: RobotType,
    /// ROS2 domain ID.
    pub domain_id: u32,
}

impl Default for Ros2Config {
    fn default() -> Self {
        Self {
            node_name: "strix_adapter".into(),
            namespace: "/robot_0".into(),
            cmd_vel_topic: "cmd_vel".into(),
            odom_topic: "odom".into(),
            battery_topic: "battery_state".into(),
            robot_type: RobotType::Ugv,
            domain_id: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

/// ROS2-based platform adapter for ground and surface robots.
///
/// **STUB** — no actual ROS2 middleware is running. Methods return dummy data.
pub struct Ros2Adapter {
    /// Unique drone/robot ID in the STRIX fleet.
    drone_id: u32,
    /// ROS2 configuration.
    config: Ros2Config,
    /// Platform capabilities (derived from robot type).
    caps: Capabilities,
    /// Simulated connection state.
    connected: bool,
}

impl Ros2Adapter {
    /// Create a new ROS2 adapter with the given configuration.
    pub fn new(drone_id: u32, config: Ros2Config) -> Self {
        let caps = match config.robot_type {
            RobotType::Ugv => Capabilities {
                max_speed: 5.0,
                max_altitude: 0.0,
                endurance: 7200.0,
                sensors: vec![SensorType::Camera, SensorType::Lidar],
                payload_kg: 10.0,
                comms_range: 2000.0,
            },
            RobotType::Usv => Capabilities {
                max_speed: 8.0,
                max_altitude: 0.0,
                endurance: 14400.0,
                sensors: vec![SensorType::Camera, SensorType::Sonar, SensorType::Radar],
                payload_kg: 50.0,
                comms_range: 10000.0,
            },
            RobotType::Uav => Capabilities {
                max_speed: 15.0,
                max_altitude: 300.0,
                endurance: 2400.0,
                sensors: vec![SensorType::Camera, SensorType::Thermal],
                payload_kg: 2.0,
                comms_range: 5000.0,
            },
        };

        Self {
            drone_id,
            config,
            caps,
            connected: false,
        }
    }

    /// Initialise the ROS2 node and start topic subscriptions.
    ///
    /// **STUB** — immediately marks the adapter as connected.
    pub fn connect(&mut self) -> Result<(), AdapterError> {
        // TODO: rclrs::init(), create node, subscribe to odom/battery
        self.connected = true;
        Ok(())
    }

    /// Shut down the ROS2 node.
    pub fn disconnect(&mut self) {
        self.connected = false;
    }

    /// Return the ROS2 configuration.
    pub fn config(&self) -> &Ros2Config {
        &self.config
    }
}

impl PlatformAdapter for Ros2Adapter {
    fn id(&self) -> u32 {
        self.drone_id
    }

    fn platform_name(&self) -> &str {
        match self.config.robot_type {
            RobotType::Ugv => "ROS2 UGV",
            RobotType::Usv => "ROS2 USV",
            RobotType::Uav => "ROS2 UAV",
        }
    }

    fn send_waypoint(&self, _wp: &Waypoint) -> Result<(), AdapterError> {
        if !self.connected {
            return Err(AdapterError::ConnectionLost("ROS2 node not running".into()));
        }
        // TODO: publish geometry_msgs/PoseStamped to /move_base_simple/goal
        Ok(())
    }

    fn get_telemetry(&self) -> Result<Telemetry, AdapterError> {
        if !self.connected {
            return Err(AdapterError::ConnectionLost("ROS2 node not running".into()));
        }
        // STUB: return zero-state telemetry
        Ok(Telemetry {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            attitude: [0.0, 0.0, 0.0],
            battery: 1.0,
            gps_fix: GpsFix::Fix3D,
            armed: false,
            mode: FlightMode::Manual,
            timestamp: 0.0,
        })
    }

    fn execute_action(&self, action: &Action) -> Result<(), AdapterError> {
        if !self.connected {
            return Err(AdapterError::ConnectionLost("ROS2 node not running".into()));
        }
        match action {
            Action::Arm => { /* publish to /arm service */ }
            Action::Disarm => { /* publish to /disarm service */ }
            Action::Takeoff(_) if self.config.robot_type == RobotType::Ugv => {
                return Err(AdapterError::UnsupportedAction(
                    "UGV cannot take off".into(),
                ));
            }
            Action::Takeoff(_) => { /* publish takeoff command */ }
            Action::Land if self.config.robot_type == RobotType::Ugv => {
                return Err(AdapterError::UnsupportedAction("UGV cannot land".into()));
            }
            Action::Land => { /* publish land command */ }
            Action::RTL => { /* navigate to home waypoint */ }
            Action::GoTo(wp) => {
                self.send_waypoint(wp)?;
            }
            Action::SetSpeed(_) => { /* update max_vel parameter */ }
            Action::SetMode(_) => { /* ROS2 mode switch service */ }
            Action::PayloadDrop => { /* trigger servo via GPIO topic */ }
            Action::CameraCapture => { /* call image capture service */ }
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
            return Err(AdapterError::ConnectionLost("ROS2 node not running".into()));
        }
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
    fn ugv_cannot_takeoff() {
        let mut adapter = Ros2Adapter::new(1, Ros2Config::default());
        adapter.connect().unwrap();
        let result = adapter.execute_action(&Action::Takeoff(10.0));
        assert!(result.is_err());
    }

    #[test]
    fn usv_capabilities() {
        let config = Ros2Config {
            robot_type: RobotType::Usv,
            ..Default::default()
        };
        let adapter = Ros2Adapter::new(2, config);
        assert!(adapter.capabilities().sensors.contains(&SensorType::Sonar));
        assert_eq!(adapter.platform_name(), "ROS2 USV");
    }
}
