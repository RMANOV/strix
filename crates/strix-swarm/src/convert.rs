//! Type conversions between STRIX crates.
//!
//! Bridges the domain-specific types from each crate into a common representation
//! that the orchestrator can work with.

use nalgebra::Vector3;

use strix_adapters::traits::Telemetry;
use strix_auction::{Assignment, Capabilities, DroneState as AuctionDroneState, Position};
use strix_core::{DroneState as CoreDroneState, Regime};
use strix_mesh::gossip::DroneState as GossipDroneState;
use strix_mesh::{NodeId, Position3D};

fn finite_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value
    } else if fallback.is_finite() {
        fallback
    } else {
        0.0
    }
}

fn sanitize_vector(values: [f64; 3], fallback: [f64; 3]) -> [f64; 3] {
    [
        finite_or(values[0], fallback[0]),
        finite_or(values[1], fallback[1]),
        finite_or(values[2], fallback[2]),
    ]
}

fn sanitize_fraction(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

pub(crate) fn sanitize_dt(dt: f64) -> f64 {
    if dt.is_finite() && dt >= 0.0 {
        dt
    } else {
        0.0
    }
}

pub(crate) fn sanitize_telemetry(
    telem: &Telemetry,
    fallback_position: [f64; 3],
    fallback_timestamp: f64,
) -> Telemetry {
    Telemetry {
        position: sanitize_vector(telem.position, fallback_position),
        velocity: sanitize_vector(telem.velocity, [0.0; 3]),
        attitude: sanitize_vector(telem.attitude, [0.0; 3]),
        battery: sanitize_fraction(telem.battery),
        gps_fix: telem.gps_fix,
        armed: telem.armed,
        mode: telem.mode,
        timestamp: finite_or(telem.timestamp, fallback_timestamp),
    }
}

/// Build an auction DroneState from telemetry + orchestrator state.
pub fn telemetry_to_auction_drone(
    id: u32,
    telem: &Telemetry,
    regime: Regime,
    capabilities: &Capabilities,
) -> AuctionDroneState {
    let telem = sanitize_telemetry(telem, [0.0; 3], 0.0);
    AuctionDroneState {
        id,
        position: Position::new(telem.position[0], telem.position[1], telem.position[2]),
        velocity: telem.velocity,
        regime,
        capabilities: capabilities.clone(),
        energy: telem.battery,
        alive: true,
    }
}

/// Build a gossip DroneState from telemetry.
pub fn telemetry_to_gossip_drone(
    id: u32,
    telem: &Telemetry,
    regime: Regime,
    version: u64,
) -> GossipDroneState {
    let telem = sanitize_telemetry(telem, [0.0; 3], 0.0);
    GossipDroneState {
        node_id: NodeId(id),
        position: Position3D(telem.position),
        battery: telem.battery,
        regime: format!("{:?}", regime),
        version,
        timestamp: telem.timestamp,
    }
}

/// Build a core DroneState from telemetry.
pub fn telemetry_to_core_drone(id: u32, telem: &Telemetry, regime: Regime) -> CoreDroneState {
    let telem = sanitize_telemetry(telem, [0.0; 3], 0.0);
    CoreDroneState {
        position: Vector3::new(telem.position[0], telem.position[1], telem.position[2]),
        velocity: Vector3::new(telem.velocity[0], telem.velocity[1], telem.velocity[2]),
        regime,
        weight: 1.0,
        drone_id: id,
        capabilities: 0,
    }
}

/// Convert auction assignments to task-id → drone-id map.
pub fn assignments_to_map(assignments: &[Assignment]) -> std::collections::HashMap<u32, u32> {
    assignments
        .iter()
        .map(|a| (a.task_id, a.drone_id))
        .collect()
}

/// Compute the threat bearing unit vector from our centroid to a threat position.
pub fn threat_bearing(centroid: &Vector3<f64>, threat_pos: &Vector3<f64>) -> Vector3<f64> {
    let diff = threat_pos - centroid;
    let norm = diff.norm();
    if norm > 1e-6 {
        diff / norm
    } else {
        Vector3::zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strix_adapters::traits::{FlightMode, GpsFix};
    use strix_auction::Capabilities;

    fn invalid_telemetry() -> Telemetry {
        Telemetry {
            position: [f64::NAN, 1.0, f64::INFINITY],
            velocity: [f64::NEG_INFINITY, 2.0, f64::NAN],
            attitude: [f64::NAN, f64::INFINITY, 3.0],
            battery: f64::NAN,
            gps_fix: GpsFix::Fix3D,
            armed: true,
            mode: FlightMode::Guided,
            timestamp: f64::NAN,
        }
    }

    #[test]
    fn sanitize_telemetry_replaces_non_finite_values() {
        let telem = sanitize_telemetry(&invalid_telemetry(), [10.0, 20.0, 30.0], 42.0);
        assert_eq!(telem.position, [10.0, 1.0, 30.0]);
        assert_eq!(telem.velocity, [0.0, 2.0, 0.0]);
        assert_eq!(telem.attitude, [0.0, 0.0, 3.0]);
        assert_eq!(telem.battery, 0.0);
        assert_eq!(telem.timestamp, 42.0);
    }

    #[test]
    fn telemetry_conversions_use_sanitized_values() {
        let telem = invalid_telemetry();
        let caps = Capabilities {
            has_sensor: true,
            has_weapon: false,
            has_ew: false,
            has_relay: false,
        };

        let auction = telemetry_to_auction_drone(7, &telem, Regime::Patrol, &caps);
        assert_eq!(auction.position, Position::new(0.0, 1.0, 0.0));
        assert_eq!(auction.velocity, [0.0, 2.0, 0.0]);
        assert_eq!(auction.energy, 0.0);

        let gossip = telemetry_to_gossip_drone(7, &telem, Regime::Patrol, 3);
        assert_eq!(gossip.position, Position3D([0.0, 1.0, 0.0]));
        assert_eq!(gossip.battery, 0.0);
        assert_eq!(gossip.timestamp, 0.0);

        let core = telemetry_to_core_drone(7, &telem, Regime::Patrol);
        assert_eq!(core.position, Vector3::new(0.0, 1.0, 0.0));
        assert_eq!(core.velocity, Vector3::new(0.0, 2.0, 0.0));
    }

    #[test]
    fn sanitize_dt_rejects_invalid_values() {
        assert_eq!(sanitize_dt(0.5), 0.5);
        assert_eq!(sanitize_dt(-1.0), 0.0);
        assert_eq!(sanitize_dt(f64::NAN), 0.0);
        assert_eq!(sanitize_dt(f64::INFINITY), 0.0);
    }
}
