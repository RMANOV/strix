//! Bid scoring policy trait and default implementation.
//!
//! Separates "how to score a bid" from "how to assign drones to tasks",
//! enabling pluggable strategies and deterministic testing.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Decomposed bid score showing contribution of each factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BidBreakdown {
    /// Distance-based score (closer = higher).
    pub proximity: f64,
    /// Capability match score.
    pub capability: f64,
    /// Energy reserve score.
    pub energy: f64,
    /// Risk penalty (higher risk = lower score).
    pub risk: f64,
    /// Task urgency contribution.
    pub urgency: f64,
    /// Penalty for switching from current assignment (stability).
    pub churn: f64,
    /// Bonus for good communication link to task area.
    pub comms_quality: f64,
    /// Final composite score.
    pub total: f64,
}

impl BidBreakdown {
    /// Create a zero breakdown.
    pub fn zero() -> Self {
        Self {
            proximity: 0.0,
            capability: 0.0,
            energy: 0.0,
            risk: 0.0,
            urgency: 0.0,
            churn: 0.0,
            comms_quality: 0.0,
            total: 0.0,
        }
    }

    /// Recompute total from components.
    pub fn recompute_total(&mut self) {
        self.total = self.proximity + self.capability + self.energy - self.risk + self.urgency
            - self.churn
            + self.comms_quality;
    }
}

/// Deterministic comparison for bid scores.
/// Primary: higher total wins. Tiebreak: lower drone_id wins.
pub fn deterministic_bid_cmp(
    a_score: f64,
    a_drone_id: u32,
    b_score: f64,
    b_drone_id: u32,
) -> Ordering {
    // Higher score wins
    match b_score.partial_cmp(&a_score) {
        Some(Ordering::Equal) | None => {}
        Some(ord) => return ord,
    }
    // Tiebreak: lower drone_id wins (reverse comparison)
    a_drone_id.cmp(&b_drone_id)
}

/// Configuration for churn penalty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChurnConfig {
    /// Bonus score for keeping current assignment (default: 0.15).
    pub stability_bonus: f64,
    /// Minimum score improvement needed to justify reassignment.
    pub reassignment_threshold: f64,
}

impl Default for ChurnConfig {
    fn default() -> Self {
        Self {
            stability_bonus: 0.15,
            reassignment_threshold: 0.10,
        }
    }
}

/// Trait for bid scoring policies.
///
/// Implementations define how drones evaluate tasks. The auctioneer
/// calls `evaluate()` for each drone-task pair and uses the breakdown
/// for assignment and auditing.
pub trait BidPolicy: Send + Sync {
    /// Evaluate a drone's bid for a task.
    ///
    /// Returns a breakdown showing each scoring factor.
    /// The `current_task_id` is Some if the drone is already assigned
    /// to a task — used for churn penalty calculation.
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &self,
        drone_id: u32,
        drone_position: [f64; 3],
        drone_energy: f64,
        drone_capabilities: u32, // bitmask
        task_id: u64,
        task_position: [f64; 3],
        task_priority: f64,
        task_required_capabilities: u32, // bitmask
        current_task_id: Option<u64>,
        fear: f64,
    ) -> BidBreakdown;
}

/// Default bid policy — extracts the scoring logic pattern from the existing bidder.
#[derive(Default)]
pub struct DefaultBidPolicy {
    pub churn_config: ChurnConfig,
}

impl BidPolicy for DefaultBidPolicy {
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &self,
        _drone_id: u32,
        drone_position: [f64; 3],
        drone_energy: f64,
        drone_capabilities: u32,
        task_id: u64,
        task_position: [f64; 3],
        task_priority: f64,
        task_required_capabilities: u32,
        current_task_id: Option<u64>,
        fear: f64,
    ) -> BidBreakdown {
        // Distance
        let dx = drone_position[0] - task_position[0];
        let dy = drone_position[1] - task_position[1];
        let dz = drone_position[2] - task_position[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1.0);
        let proximity = 1.0 / dist; // normalized: closer = higher

        // Capability match: full score if no requirements, or drone meets all of them.
        let capability = if task_required_capabilities == 0
            || (drone_capabilities & task_required_capabilities) == task_required_capabilities
        {
            1.0
        } else {
            0.0 // can't do this task
        };

        // Energy
        let energy = drone_energy.clamp(0.0, 1.0);

        // Risk (fear increases risk sensitivity)
        let risk_base = 0.1; // base risk
        let risk = risk_base * (1.0 + fear);

        // Urgency
        let urgency = task_priority.clamp(0.0, 1.0) * 0.5;

        // Churn penalty
        let churn = if let Some(current) = current_task_id {
            if current == task_id {
                -self.churn_config.stability_bonus // bonus (negative penalty)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Comms quality (placeholder: full quality for now)
        let comms_quality = 0.0;

        let total = proximity + capability + energy - risk + urgency - churn + comms_quality;

        BidBreakdown {
            proximity,
            capability,
            energy,
            risk,
            urgency,
            churn,
            comms_quality,
            total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_basic_scoring() {
        let policy = DefaultBidPolicy::default();
        let bd = policy.evaluate(
            1,
            [0.0, 0.0, 0.0],
            0.8,
            0b11,
            100,
            [100.0, 0.0, 0.0],
            0.5,
            0b01,
            None,
            0.0,
        );
        assert!(bd.total > 0.0);
        assert!(bd.proximity > 0.0);
        assert_eq!(bd.capability, 1.0);
    }

    #[test]
    fn capability_mismatch_zeros_score() {
        let policy = DefaultBidPolicy::default();
        let bd = policy.evaluate(
            1,
            [0.0, 0.0, 0.0],
            0.8,
            0b01, // has sensor only
            100,
            [10.0, 0.0, 0.0],
            0.5,
            0b10, // needs weapon
            None,
            0.0,
        );
        assert_eq!(bd.capability, 0.0);
    }

    #[test]
    fn churn_penalty_keeps_current_assignment() {
        let policy = DefaultBidPolicy::default();
        // Same task (current assignment)
        let bd_stay = policy.evaluate(
            1,
            [0.0, 0.0, 0.0],
            0.8,
            0b11,
            100,
            [50.0, 0.0, 0.0],
            0.5,
            0b01,
            Some(100),
            0.0,
        );
        // Different task
        let bd_switch = policy.evaluate(
            1,
            [0.0, 0.0, 0.0],
            0.8,
            0b11,
            200,
            [50.0, 0.0, 0.0],
            0.5,
            0b01,
            Some(100),
            0.0,
        );
        // Staying should score higher due to stability bonus
        assert!(
            bd_stay.total > bd_switch.total,
            "stay={} switch={}",
            bd_stay.total,
            bd_switch.total
        );
    }

    #[test]
    fn fear_increases_risk() {
        let policy = DefaultBidPolicy::default();
        let bd_calm = policy.evaluate(
            1,
            [0.0, 0.0, 0.0],
            0.8,
            0b11,
            100,
            [50.0, 0.0, 0.0],
            0.5,
            0b01,
            None,
            0.0,
        );
        let bd_scared = policy.evaluate(
            1,
            [0.0, 0.0, 0.0],
            0.8,
            0b11,
            100,
            [50.0, 0.0, 0.0],
            0.5,
            0b01,
            None,
            0.9,
        );
        assert!(bd_scared.risk > bd_calm.risk);
        assert!(bd_calm.total > bd_scared.total);
    }

    #[test]
    fn closer_drone_scores_higher() {
        let policy = DefaultBidPolicy::default();
        let bd_close = policy.evaluate(
            1,
            [0.0, 0.0, 0.0],
            0.8,
            0b11,
            100,
            [10.0, 0.0, 0.0],
            0.5,
            0b01,
            None,
            0.0,
        );
        let bd_far = policy.evaluate(
            1,
            [0.0, 0.0, 0.0],
            0.8,
            0b11,
            100,
            [1000.0, 0.0, 0.0],
            0.5,
            0b01,
            None,
            0.0,
        );
        assert!(bd_close.proximity > bd_far.proximity);
    }

    #[test]
    fn deterministic_tiebreak() {
        // Same score, lower drone_id wins
        assert_eq!(
            deterministic_bid_cmp(5.0, 1, 5.0, 2),
            Ordering::Less, // drone 1 wins (comes first)
        );
        assert_eq!(deterministic_bid_cmp(5.0, 2, 5.0, 1), Ordering::Greater,);
        // Higher score wins regardless of ID
        assert_eq!(
            deterministic_bid_cmp(6.0, 99, 5.0, 1),
            Ordering::Less, // drone 99 wins (higher score)
        );
    }

    #[test]
    fn bid_breakdown_recompute() {
        let mut bd = BidBreakdown::zero();
        bd.proximity = 1.0;
        bd.capability = 0.5;
        bd.energy = 0.3;
        bd.risk = 0.1;
        bd.urgency = 0.2;
        bd.churn = 0.05;
        bd.comms_quality = 0.1;
        bd.recompute_total();
        let expected = 1.0 + 0.5 + 0.3 - 0.1 + 0.2 - 0.05 + 0.1;
        assert!((bd.total - expected).abs() < 1e-10);
    }

    #[test]
    fn determinism_over_many_runs() {
        let policy = DefaultBidPolicy::default();
        let first = policy.evaluate(
            42,
            [10.0, 20.0, -5.0],
            0.65,
            0b111,
            999,
            [100.0, 200.0, -10.0],
            0.8,
            0b101,
            Some(888),
            0.4,
        );
        for _ in 0..1000 {
            let bd = policy.evaluate(
                42,
                [10.0, 20.0, -5.0],
                0.65,
                0b111,
                999,
                [100.0, 200.0, -10.0],
                0.8,
                0b101,
                Some(888),
                0.4,
            );
            assert_eq!(bd.total, first.total, "non-deterministic scoring detected");
        }
    }
}
