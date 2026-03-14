//! # strix-auction
//!
//! Combinatorial task auction engine for drone swarm task allocation.
//!
//! "The Battlefield is a Market" — drones bid on tasks like traders bid on assets.
//! The scoring function derives from quantitative trading portfolio optimization,
//! while risk management borrows drawdown protection and anti-fragile adaptation
//! from Taleb's framework.
//!
//! ## Architecture
//!
//! - [`bidder`]: Individual drone bidding engine (sealed-bid evaluation)
//! - [`auctioneer`]: Combinatorial auction with modified Hungarian assignment
//! - [`portfolio`]: Fleet-as-portfolio optimization (diversification, correlation, risk budgets)
//! - [`risk`]: Attrition monitoring and drawdown protection
//! - [`antifragile`]: Anti-fragile loss recovery — the swarm gets *stronger* after losses

pub mod antifragile;
pub mod auctioneer;
pub mod bidder;
pub mod portfolio;
pub mod risk;

// ────────────────────────────────────────────────────────────────────────────────
// Stub types — will be replaced by `use strix_core::*` once the crate is built.
// ────────────────────────────────────────────────────────────────────────────────

use serde::{Deserialize, Serialize};

/// Operational regime for a drone or the entire swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Regime {
    /// Area surveillance / presence.
    Patrol,
    /// Active engagement with a designated target.
    Engage,
    /// Threat-avoidance manoeuvring.
    Evade,
}

/// Capability flags a drone may carry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    /// Carries EO/IR sensor.
    pub has_sensor: bool,
    /// Carries a weapon payload.
    pub has_weapon: bool,
    /// Electronic-warfare suite.
    pub has_ew: bool,
    /// Communications relay capability.
    pub has_relay: bool,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            has_sensor: true,
            has_weapon: false,
            has_ew: false,
            has_relay: false,
        }
    }
}

/// Minimal 3-D position.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Position {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Euclidean distance to another position.
    pub fn distance_to(&self, other: &Position) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2))
            .sqrt()
    }
}

/// Lightweight drone state used by the auction system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneState {
    pub id: u32,
    pub position: Position,
    pub velocity: [f64; 3],
    pub regime: Regime,
    pub capabilities: Capabilities,
    /// Remaining energy as a fraction [0, 1].
    pub energy: f64,
    /// Whether the drone is alive.
    pub alive: bool,
}

/// A task to be auctioned among drones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: u32,
    pub location: Position,
    /// What capabilities are required to execute this task.
    pub required_capabilities: Capabilities,
    /// Priority weight [0, 1] — higher means more critical.
    pub priority: f64,
    /// Urgency multiplier — time-sensitive tasks get a bonus.
    pub urgency: f64,
    /// Optional bundle ID — tasks with the same bundle must go to the same drone/group.
    pub bundle_id: Option<u32>,
    /// If `Some(sub_swarm_id)`, only drones in that sub-swarm may see this task ("dark pool").
    pub dark_pool: Option<u32>,
}

/// Threat information used for risk calculations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatState {
    pub id: u32,
    pub position: Position,
    /// Estimated lethality radius.
    pub lethal_radius: f64,
    /// Threat type classification.
    pub threat_type: ThreatType,
}

/// Classification of threat types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreatType {
    /// Surface-to-air missile system.
    Sam,
    /// Small arms / anti-aircraft artillery.
    SmallArms,
    /// Electronic warfare / jamming.
    ElectronicWarfare,
    /// Unknown threat.
    Unknown,
}

/// Assignment of a drone to a task, produced by the auction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    pub drone_id: u32,
    pub task_id: u32,
    pub bid_score: f64,
}

// Re-export key public types from submodules.
pub use antifragile::{LossAnalyzer, LossClassification, LossRecord};
pub use auctioneer::{AuctionResult, Auctioneer};
pub use bidder::{Bid, BidComponents, Bidder};
pub use portfolio::{CoverageMatrix, PortfolioOptimizer};
pub use risk::{AttritionMonitor, MaxDrawdown, RiskLevel, ValueAtRisk};
