//! # strix-mesh - Decentralized Mesh Coordination
//!
//! Handles drone swarm coordination without central authority:
//! - **Fractal hierarchy**: self-similar command structure at every scale
//! - **Stigmergy**: bio-inspired digital pheromone fields
//! - **Gossip protocol**: O(log N) convergence state synchronization
//! - **Consensus**: lightweight leader election and rank management
//! - **Comms**: radio abstraction layer with bandwidth-aware prioritization

pub mod belief;
pub mod bool_gates;
pub mod byzantine;
pub mod comms;
pub mod consensus;
pub mod contagion;
pub mod degradation;
pub mod evidence_graph;
pub mod fact;
pub mod fractal;
pub mod gbp;
pub mod gossip;
pub mod hypergraph;
pub mod partition;
pub mod quarantine;
pub mod spatial_belief;
pub mod stigmergy;
pub mod trust;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Mesh-specific types with From/Into conversions to strix-core
// ---------------------------------------------------------------------------

/// Unique drone identifier within the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(pub u32);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// 3-D position in world coordinates (metres).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position3D(pub [f64; 3]);

impl Position3D {
    /// Euclidean distance to another point.
    pub fn distance(&self, other: &Self) -> f64 {
        let dx = self.0[0] - other.0[0];
        let dy = self.0[1] - other.0[1];
        let dz = self.0[2] - other.0[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Zero origin.
    pub fn origin() -> Self {
        Self([0.0, 0.0, 0.0])
    }
}

// ---------------------------------------------------------------------------
// Cross-crate conversions: strix-core <-> strix-mesh
// ---------------------------------------------------------------------------

impl From<nalgebra::Vector3<f64>> for Position3D {
    fn from(v: nalgebra::Vector3<f64>) -> Self {
        Position3D([v.x, v.y, v.z])
    }
}

impl From<&Position3D> for nalgebra::Vector3<f64> {
    fn from(p: &Position3D) -> Self {
        nalgebra::Vector3::new(p.0[0], p.0[1], p.0[2])
    }
}

impl From<Position3D> for nalgebra::Vector3<f64> {
    fn from(p: Position3D) -> Self {
        nalgebra::Vector3::new(p.0[0], p.0[1], p.0[2])
    }
}

// ---------------------------------------------------------------------------
// Mesh message - the lingua franca of mesh communication
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinationDirectiveKind {
    StrikeCommit,
    Retreat,
    Reconfigure,
    Rally,
}

/// All message types that flow through the mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeshMessage {
    /// Periodic liveness proof. Carries sender id + timestamp.
    Heartbeat { sender: NodeId, timestamp: f64 },
    /// Compact situational awareness update from one drone.
    StateUpdate {
        sender: NodeId,
        position: Position3D,
        battery: f64,
        regime: String,
        timestamp: f64,
    },
    /// A task has been assigned (via auction or command).
    TaskAssignment {
        task_id: u64,
        assignee: NodeId,
        assigner: NodeId,
        description: String,
        timestamp: f64,
    },
    /// Threat detected - highest priority, never discarded.
    ThreatAlert {
        reporter: NodeId,
        position: Position3D,
        threat_level: f64,
        description: String,
        timestamp: f64,
    },
    /// Explicit non-pairwise coordination signals.
    CoordinationDirective {
        sender: NodeId,
        directive: CoordinationDirectiveKind,
        focus: Option<Position3D>,
        intensity: f64,
        timestamp: f64,
    },
    /// Fast affective signal used for panic/fear damping.
    AffectSignal {
        sender: NodeId,
        label: String,
        intensity: f64,
        timestamp: f64,
    },
    /// Digital pheromone broadcast (~20 bytes payload).
    PheromoneDeposit {
        depositor: NodeId,
        position: Position3D,
        ptype: stigmergy::PheromoneType,
        intensity: f64,
        timestamp: f64,
    },
}

impl MeshMessage {
    /// Returns the canonical priority (lower = more important).
    /// Used by the bandwidth manager for prioritized queuing.
    pub fn priority(&self) -> u8 {
        match self {
            MeshMessage::ThreatAlert { .. } => 0,
            MeshMessage::CoordinationDirective { .. } | MeshMessage::TaskAssignment { .. } => 1,
            MeshMessage::StateUpdate { .. } | MeshMessage::AffectSignal { .. } => 2,
            MeshMessage::PheromoneDeposit { .. } => 3,
            MeshMessage::Heartbeat { .. } => 4,
        }
    }

    /// Sender of this message.
    pub fn sender(&self) -> NodeId {
        match self {
            MeshMessage::Heartbeat { sender, .. } => *sender,
            MeshMessage::StateUpdate { sender, .. } => *sender,
            MeshMessage::TaskAssignment { assigner, .. } => *assigner,
            MeshMessage::ThreatAlert { reporter, .. } => *reporter,
            MeshMessage::CoordinationDirective { sender, .. } => *sender,
            MeshMessage::AffectSignal { sender, .. } => *sender,
            MeshMessage::PheromoneDeposit { depositor, .. } => *depositor,
        }
    }

    /// Timestamp of this message.
    pub fn timestamp(&self) -> f64 {
        match self {
            MeshMessage::Heartbeat { timestamp, .. }
            | MeshMessage::StateUpdate { timestamp, .. }
            | MeshMessage::TaskAssignment { timestamp, .. }
            | MeshMessage::ThreatAlert { timestamp, .. }
            | MeshMessage::CoordinationDirective { timestamp, .. }
            | MeshMessage::AffectSignal { timestamp, .. }
            | MeshMessage::PheromoneDeposit { timestamp, .. } => *timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh configuration
// ---------------------------------------------------------------------------

/// Tunable parameters for the mesh layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshConfig {
    /// Heartbeat interval in seconds.
    pub heartbeat_interval_s: f64,
    /// Number of missed heartbeats before declaring a node dead.
    pub heartbeat_miss_limit: u32,
    /// Pheromone exponential decay rate (per second).
    pub pheromone_decay_rate: f64,
    /// Spatial resolution of the pheromone grid (metres per cell).
    pub pheromone_grid_resolution: f64,
    /// Number of peers each gossip round targets.
    pub gossip_fanout: usize,
    /// Gossip period in seconds.
    pub gossip_interval_s: f64,
    /// Maximum message payload size in bytes.
    pub max_message_bytes: usize,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_s: 1.0,
            heartbeat_miss_limit: 3,
            pheromone_decay_rate: 0.05,
            pheromone_grid_resolution: 10.0,
            gossip_fanout: 3,
            gossip_interval_s: 2.0,
            max_message_bytes: 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_distance() {
        let a = Position3D([0.0, 0.0, 0.0]);
        let b = Position3D([3.0, 4.0, 0.0]);
        assert!((a.distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn position_distance_3d() {
        let a = Position3D([1.0, 2.0, 3.0]);
        let b = Position3D([4.0, 6.0, 3.0]);
        assert!((a.distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn message_priority_ordering() {
        let threat = MeshMessage::ThreatAlert {
            reporter: NodeId(0),
            position: Position3D::origin(),
            threat_level: 1.0,
            description: String::new(),
            timestamp: 0.0,
        };
        let hb = MeshMessage::Heartbeat {
            sender: NodeId(0),
            timestamp: 0.0,
        };
        assert!(threat.priority() < hb.priority());
    }

    #[test]
    fn coordination_directive_priority_matches_task_assignment() {
        let directive = MeshMessage::CoordinationDirective {
            sender: NodeId(0),
            directive: CoordinationDirectiveKind::StrikeCommit,
            focus: None,
            intensity: 0.8,
            timestamp: 0.0,
        };
        let task = MeshMessage::TaskAssignment {
            task_id: 1,
            assignee: NodeId(1),
            assigner: NodeId(0),
            description: String::new(),
            timestamp: 0.0,
        };
        assert_eq!(directive.priority(), task.priority());
    }

    #[test]
    fn mesh_config_default() {
        let cfg = MeshConfig::default();
        assert_eq!(cfg.heartbeat_miss_limit, 3);
        assert_eq!(cfg.gossip_fanout, 3);
    }

    #[test]
    fn node_id_display() {
        assert_eq!(format!("{}", NodeId(42)), "Node(42)");
    }

    #[test]
    fn message_roundtrip_serde() {
        let msg = MeshMessage::Heartbeat {
            sender: NodeId(7),
            timestamp: 123.456,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: MeshMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.sender(), NodeId(7));
    }
}
