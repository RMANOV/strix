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

/// Build an auction DroneState from telemetry + orchestrator state.
pub fn telemetry_to_auction_drone(
    id: u32,
    telem: &Telemetry,
    regime: Regime,
    capabilities: &Capabilities,
) -> AuctionDroneState {
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
