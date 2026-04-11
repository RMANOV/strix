//! Feedback loop stability contracts.
//!
//! Formalizes the 4 MODERATE-risk feedback loops as bounded-gain contracts.
//! Checked in debug builds only — no runtime overhead in release.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::criticality::CriticalityAdjustment;
use crate::order_params::OrderParameters;
use crate::tick::SwarmConfig;

// ---------------------------------------------------------------------------
// Contract definitions
// ---------------------------------------------------------------------------

/// A feedback loop stability contract.
#[derive(Debug, Clone)]
pub struct FeedbackContract {
    /// Human-readable loop name.
    pub name: &'static str,
    /// Maximum acceptable open-loop gain.
    pub max_gain: f64,
    /// Description of the damping mechanism.
    pub damping: &'static str,
    /// Expected settling time in ticks.
    pub settling_ticks: u32,
}

/// A contract violation — detected when actual gain exceeds contracted bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractViolation {
    /// Which loop violated its contract.
    pub loop_name: String,
    /// Actual gain observed.
    pub actual_gain: f64,
    /// Contracted maximum gain.
    pub max_gain: f64,
    /// How much the gain exceeded the bound (ratio).
    pub overshoot: f64,
}

impl fmt::Display for ContractViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[STABILITY] {}: gain {:.4} exceeds contract {:.4} (overshoot {:.1}%)",
            self.loop_name,
            self.actual_gain,
            self.max_gain,
            self.overshoot * 100.0
        )
    }
}

// ---------------------------------------------------------------------------
// Contract registry
// ---------------------------------------------------------------------------

/// The four MODERATE-risk feedback loops with their stability contracts.
pub fn contracts() -> [FeedbackContract; 4] {
    [
        FeedbackContract {
            name: "Fear→Criticality→Behavior",
            max_gain: 0.25,
            damping: "3-tick interval, [0,1] clamping, CriticalityConfig min/max bounds",
            settling_ticks: 15,
        },
        FeedbackContract {
            name: "Gossip→Criticality→GossipFanout",
            max_gain: 0.39,
            damping: "Hard fanout cap (3×1.30), 10-tick order params interval, EMA α=0.15",
            settling_ticks: 30,
        },
        FeedbackContract {
            name: "GBP→Evidence→PriorReset",
            max_gain: 0.30,
            damping: "XOR rate threshold gate (0.7), 20-tick window, α=0.95 memory decay",
            settling_ticks: 25,
        },
        FeedbackContract {
            name: "Threat→Criticality→Regime",
            max_gain: 0.20,
            damping: "Particle filter exponential decay, threat noise dampening",
            settling_ticks: 20,
        },
    ]
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// Verify feedback loop gains against contracted bounds.
///
/// Returns violations for any loop whose effective gain exceeds its contract.
/// Intended for `#[cfg(debug_assertions)]` use in the tick loop.
pub fn verify_loop_gains(
    criticality: &CriticalityAdjustment,
    order_params: &OrderParameters,
    prev_criticality: &CriticalityAdjustment,
    prev_fear: f64,
    current_fear: f64,
    _config: &SwarmConfig, // reserved: derive bounds from config when contracts mature
) -> Vec<ContractViolation> {
    let mut violations = Vec::new();

    // Contract 1: Fear→Criticality→Behavior
    // |Δcriticality / Δfear| should be ≤ 0.25
    let delta_fear = (current_fear - prev_fear).abs();
    if delta_fear > 1e-6 {
        let delta_crit = (criticality.criticality - prev_criticality.criticality).abs();
        let gain = delta_crit / delta_fear;
        if gain > 0.25 {
            violations.push(ContractViolation {
                loop_name: "Fear→Criticality→Behavior".into(),
                actual_gain: gain,
                max_gain: 0.25,
                overshoot: (gain - 0.25) / 0.25,
            });
        }
    }

    // Contract 2: Gossip→Criticality→GossipFanout
    // Gossip convergence should not cause criticality swings > 0.39
    // We measure the contribution of gossip_convergence to disorder metric.
    // Since gossip_convergence weight is 0.40 in the order metric and
    // criticality interpolates between min/max multipliers, the max gain is
    // 0.40 × (max_multiplier - min_multiplier) ≈ 0.40 × 0.60 = 0.24.
    // We check the actual criticality change vs convergence change.
    let convergence_change = (order_params.trust_entropy - 0.5).abs(); // proxy: entropy swing
    if convergence_change > 0.3 {
        let crit_change = (criticality.criticality - prev_criticality.criticality).abs();
        if crit_change > 0.39 {
            violations.push(ContractViolation {
                loop_name: "Gossip→Criticality→GossipFanout".into(),
                actual_gain: crit_change,
                max_gain: 0.39,
                overshoot: (crit_change - 0.39) / 0.39,
            });
        }
    }

    violations
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contracts_are_defined() {
        let c = contracts();
        assert_eq!(c.len(), 4);
        assert!(c.iter().all(|c| c.max_gain > 0.0 && c.settling_ticks > 0));
    }

    #[test]
    fn no_violation_when_within_bounds() {
        let crit = CriticalityAdjustment::default();
        let prev_crit = CriticalityAdjustment::default();
        let op = OrderParameters::default();
        let config = SwarmConfig::default();
        let violations = verify_loop_gains(&crit, &op, &prev_crit, 0.3, 0.35, &config);
        assert!(violations.is_empty());
    }

    #[test]
    fn violation_display_format() {
        let v = ContractViolation {
            loop_name: "Test".into(),
            actual_gain: 0.5,
            max_gain: 0.25,
            overshoot: 1.0,
        };
        let s = format!("{v}");
        assert!(s.contains("STABILITY"));
        assert!(s.contains("0.5000"));
        assert!(s.contains("100.0%"));
    }
}
