//! Machine-readable reason codes for decision audit trails.
//!
//! Replaces free-form string reasons with structured, queryable codes.

use serde::{Deserialize, Serialize};

/// Machine-readable reason for a decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasonCode {
    /// New threat detected.
    ThreatDetected { threat_id: u64, confidence: f64 },
    /// Regime changed.
    RegimeShift { from: String, to: String },
    /// Drone lost.
    DroneAttrition { drone_id: u32, cause: String },
    /// Formation quality dropped below threshold.
    FormationDegradation { quality: f64, threshold: f64 },
    /// Communications quality degraded.
    CommunicationsDegraded { link_quality: f64 },
    /// Safety constraint activated (CBF, altitude, no-fly).
    SafetyConstraintActive {
        constraint_type: String,
        barrier_value: f64,
    },
    /// Human operator override.
    HumanOverride { operator_id: String },
    /// Auction reassignment.
    AuctionReassignment {
        task_id: u64,
        old_drone: Option<u32>,
        new_drone: u32,
        bid_score: f64,
    },
    /// Kill zone triggered risk avoidance.
    KillZoneAvoidance {
        zone_position: [f64; 3],
        zone_radius: f64,
    },
    /// Gossip convergence issue.
    ConvergenceIssue { convergence: f64, threshold: f64 },
    /// Energy conservation — low battery.
    EnergyConservation { battery_level: f64, threshold: f64 },
    /// Scheduled re-evaluation (periodic).
    PeriodicReEvaluation { interval_ticks: u32 },
}

impl ReasonCode {
    /// Short human-readable label for this reason.
    pub fn label(&self) -> &'static str {
        match self {
            Self::ThreatDetected { .. } => "threat_detected",
            Self::RegimeShift { .. } => "regime_shift",
            Self::DroneAttrition { .. } => "drone_attrition",
            Self::FormationDegradation { .. } => "formation_degradation",
            Self::CommunicationsDegraded { .. } => "comms_degraded",
            Self::SafetyConstraintActive { .. } => "safety_constraint",
            Self::HumanOverride { .. } => "human_override",
            Self::AuctionReassignment { .. } => "auction_reassignment",
            Self::KillZoneAvoidance { .. } => "kill_zone",
            Self::ConvergenceIssue { .. } => "convergence_issue",
            Self::EnergyConservation { .. } => "energy_conservation",
            Self::PeriodicReEvaluation { .. } => "periodic_reeval",
        }
    }

    /// Whether this is a safety-critical reason.
    pub fn is_safety_critical(&self) -> bool {
        matches!(
            self,
            Self::ThreatDetected { .. }
                | Self::SafetyConstraintActive { .. }
                | Self::KillZoneAvoidance { .. }
                | Self::DroneAttrition { .. }
        )
    }

    /// Whether this requires human attention.
    pub fn requires_human_attention(&self) -> bool {
        matches!(
            self,
            Self::HumanOverride { .. }
                | Self::DroneAttrition { .. }
                | Self::CommunicationsDegraded { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn all_variants() -> Vec<ReasonCode> {
        vec![
            ReasonCode::ThreatDetected {
                threat_id: 1,
                confidence: 0.9,
            },
            ReasonCode::RegimeShift {
                from: "Patrol".into(),
                to: "Engage".into(),
            },
            ReasonCode::DroneAttrition {
                drone_id: 42,
                cause: "downed".into(),
            },
            ReasonCode::FormationDegradation {
                quality: 0.3,
                threshold: 0.5,
            },
            ReasonCode::CommunicationsDegraded { link_quality: 0.2 },
            ReasonCode::SafetyConstraintActive {
                constraint_type: "CBF".into(),
                barrier_value: 0.01,
            },
            ReasonCode::HumanOverride {
                operator_id: "op-1".into(),
            },
            ReasonCode::AuctionReassignment {
                task_id: 100,
                old_drone: Some(3),
                new_drone: 7,
                bid_score: 0.88,
            },
            ReasonCode::KillZoneAvoidance {
                zone_position: [1.0, 2.0, 3.0],
                zone_radius: 50.0,
            },
            ReasonCode::ConvergenceIssue {
                convergence: 0.4,
                threshold: 0.7,
            },
            ReasonCode::EnergyConservation {
                battery_level: 0.15,
                threshold: 0.2,
            },
            ReasonCode::PeriodicReEvaluation { interval_ticks: 10 },
        ]
    }

    #[test]
    fn reason_labels() {
        for code in all_variants() {
            let label = code.label();
            assert!(!label.is_empty(), "label for {:?} must not be empty", code);
        }
    }

    #[test]
    fn safety_critical_reasons() {
        assert!(ReasonCode::ThreatDetected {
            threat_id: 1,
            confidence: 0.9
        }
        .is_safety_critical());
        assert!(ReasonCode::SafetyConstraintActive {
            constraint_type: "CBF".into(),
            barrier_value: 0.0
        }
        .is_safety_critical());
        assert!(ReasonCode::KillZoneAvoidance {
            zone_position: [0.0; 3],
            zone_radius: 10.0
        }
        .is_safety_critical());
        assert!(ReasonCode::DroneAttrition {
            drone_id: 1,
            cause: "crash".into()
        }
        .is_safety_critical());
    }

    #[test]
    fn non_safety_reasons() {
        assert!(!ReasonCode::AuctionReassignment {
            task_id: 1,
            old_drone: None,
            new_drone: 2,
            bid_score: 0.5
        }
        .is_safety_critical());
        assert!(!ReasonCode::PeriodicReEvaluation { interval_ticks: 5 }.is_safety_critical());
        assert!(!ReasonCode::RegimeShift {
            from: "a".into(),
            to: "b".into()
        }
        .is_safety_critical());
    }

    #[test]
    fn human_attention_reasons() {
        assert!(ReasonCode::HumanOverride {
            operator_id: "op-1".into()
        }
        .requires_human_attention());
        assert!(ReasonCode::DroneAttrition {
            drone_id: 5,
            cause: "jammed".into()
        }
        .requires_human_attention());
        assert!(ReasonCode::CommunicationsDegraded { link_quality: 0.1 }.requires_human_attention());

        // Not requiring human attention
        assert!(!ReasonCode::PeriodicReEvaluation { interval_ticks: 1 }.requires_human_attention());
        assert!(!ReasonCode::AuctionReassignment {
            task_id: 2,
            old_drone: None,
            new_drone: 3,
            bid_score: 0.7
        }
        .requires_human_attention());
    }

    #[test]
    fn serde_roundtrip() {
        for code in all_variants() {
            let json = serde_json::to_string(&code).expect("serialize");
            let decoded: ReasonCode = serde_json::from_str(&json).expect("deserialize");
            // Re-serialize and compare JSON strings for structural equality
            let json2 = serde_json::to_string(&decoded).expect("re-serialize");
            assert_eq!(json, json2, "roundtrip mismatch for label={}", code.label());
        }
    }
}
