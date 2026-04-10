//! Boolean gate types for epistemic evidence processing.
//!
//! Provides the signal vocabulary used by the evidence graph to
//! detect conflicts (XOR), corroboration (XNOR), information vacuums
//! (NOR), forbidden states (NAND), and broken implications in the
//! swarm's inference pipeline.
//!
//! These are **not** computational gates — they are epistemic mode
//! selectors that tell the Bayesian layer how to handle evidence:
//! - XOR → widen uncertainty, investigate
//! - XNOR → boost confidence (scaled by source independence)
//! - NOR → preserve prior, flag information gap
//! - NAND → hard block, escalate

use serde::{Deserialize, Serialize};

use crate::trust::TrustDimension;
use crate::NodeId;

// ---------------------------------------------------------------------------
// Signal source — identifies where evidence comes from
// ---------------------------------------------------------------------------

/// Source of an evidence signal in the STRIX subsystem graph.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SignalSource {
    /// From Byzantine validation of a peer's update.
    Byzantine(NodeId),
    /// From trust tracker observation.
    Trust(NodeId),
    /// From quarantine state change.
    Quarantine(NodeId),
    /// From gossip merge result.
    Gossip(NodeId),
    /// From GBP fusion uncertainty.
    Gbp(NodeId),
    /// From pheromone field (zone identifier).
    Pheromone(u64),
    /// From macro order parameters (swarm-level).
    OrderParams,
}

// ---------------------------------------------------------------------------
// Gate signals — emitted by subsystems
// ---------------------------------------------------------------------------

/// Signal emitted when a subsystem detects an epistemic condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateSignal {
    /// XOR: two sources disagree on the same hypothesis.
    Conflict {
        source_a: SignalSource,
        source_b: SignalSource,
        /// Severity of the conflict ∈ [0, 1].
        severity: f64,
        timestamp: f64,
    },
    /// XNOR: two or more sources agree.
    Corroboration {
        sources: Vec<SignalSource>,
        /// Independence between sources ∈ [0, 1]. Low independence
        /// (correlated sources) discounts the consensus value.
        independence: f64,
        /// How much to strengthen belief.
        confidence_boost: f64,
        timestamp: f64,
    },
    /// NOR: no signal received from an expected source.
    Vacuum {
        expected_source: SignalSource,
        /// Timestamp when the source was last active.
        last_seen: f64,
        /// How long the vacuum has persisted (seconds).
        duration: f64,
    },
    /// NAND: a forbidden combination of conditions is active.
    Violation {
        condition_a: String,
        condition_b: String,
        severity: f64,
    },
    /// Implication A → B is broken (A is true but B is false).
    ImplicationBreak {
        antecedent: String,
        consequent: String,
        context: String,
    },
}

// ---------------------------------------------------------------------------
// Feedback actions — produced by evidence graph processing
// ---------------------------------------------------------------------------

/// Action the evidence graph instructs the orchestrator to perform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackAction {
    /// Update a specific trust dimension for a peer.
    UpdateTrust {
        peer: NodeId,
        dimension: TrustDimension,
        /// Positive = trust more, negative = trust less.
        /// Clamped to [-0.1, +0.05] per tick.
        delta: f64,
    },
    /// Modulate a threshold (quarantine strike weight, etc.).
    AdjustThreshold { system: String, multiplier: f64 },
    /// Reset prior toward uninformative (XOR-triggered forgetting).
    ResetPrior {
        /// The peer whose GBP belief should be reset.
        peer: NodeId,
        /// 0 = no reset, 1 = full reset to uninformative.
        alpha: f64,
    },
    /// Mark an information vacuum for a region or peer.
    MarkVacuum { source: SignalSource, severity: f64 },
    /// Escalate — the evidence graph cannot resolve the situation locally.
    Escalate { reason: String, severity: f64 },
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Quorum evaluation: passes if the weighted sum of positive votes
/// exceeds `threshold`.
///
/// Each vote is `(voter_id, vote_bool, weight)`.
pub fn quorum_passes(votes: &[(NodeId, bool, f64)], threshold: f64) -> bool {
    if votes.is_empty() {
        return false;
    }
    let weighted_yes: f64 = votes
        .iter()
        .filter(|(_, v, _)| *v)
        .map(|(_, _, w)| *w)
        .sum();
    let total_weight: f64 = votes.iter().map(|(_, _, w)| *w).sum();
    if total_weight < 1e-12 {
        return false;
    }
    weighted_yes / total_weight >= threshold
}

/// XOR disagreement rate between two binary signal histories.
///
/// Returns the fraction of positions where the two histories differ.
/// Empty histories return 0.0.
pub fn xor_rate(history_a: &[bool], history_b: &[bool]) -> f64 {
    let len = history_a.len().min(history_b.len());
    if len == 0 {
        return 0.0;
    }
    let conflicts: usize = history_a
        .iter()
        .zip(history_b.iter())
        .filter(|(a, b)| a != b)
        .count();
    conflicts as f64 / len as f64
}

/// XNOR consensus score with independence correction.
///
/// `agreements` = number of agreeing pairs out of `total_pairs`.
/// `independence` ∈ [0, 1] corrects for correlated sources:
/// effective_consensus = raw_consensus × independence.
pub fn xnor_consensus(agreements: f64, total_pairs: f64, independence: f64) -> f64 {
    if total_pairs < 1e-12 {
        return 0.0;
    }
    let raw = (agreements / total_pairs).clamp(0.0, 1.0);
    raw * independence.clamp(0.0, 1.0)
}

/// NAND check: returns `true` if both conditions are active
/// (a forbidden combination).
pub fn nand_violated(a: bool, b: bool) -> bool {
    a && b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_construction_roundtrip() {
        let sig = GateSignal::Conflict {
            source_a: SignalSource::Byzantine(NodeId(1)),
            source_b: SignalSource::Gossip(NodeId(2)),
            severity: 0.8,
            timestamp: 10.0,
        };
        let json = serde_json::to_string(&sig).unwrap();
        let back: GateSignal = serde_json::from_str(&json).unwrap();
        match back {
            GateSignal::Conflict { severity, .. } => {
                assert!((severity - 0.8).abs() < 1e-6);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn quorum_passes_weighted() {
        let votes = vec![
            (NodeId(1), true, 0.8),
            (NodeId(2), true, 0.6),
            (NodeId(3), false, 0.9),
        ];
        // weighted_yes = 1.4, total = 2.3, ratio ≈ 0.609
        assert!(quorum_passes(&votes, 0.5));
        assert!(!quorum_passes(&votes, 0.7));
    }

    #[test]
    fn quorum_empty_fails() {
        assert!(!quorum_passes(&[], 0.5));
    }

    #[test]
    fn xor_rate_calculation() {
        let a = vec![true, true, false, false, true];
        let b = vec![true, false, false, true, true];
        // differences at positions 1 and 3 → rate = 2/5 = 0.4
        assert!((xor_rate(&a, &b) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn xor_rate_empty_is_zero() {
        assert!((xor_rate(&[], &[])).abs() < 1e-6);
    }

    #[test]
    fn xnor_consensus_with_independence() {
        // 8 agreements out of 10 pairs, full independence
        assert!((xnor_consensus(8.0, 10.0, 1.0) - 0.8).abs() < 1e-6);
        // same but independence = 0.5 → effective = 0.4
        assert!((xnor_consensus(8.0, 10.0, 0.5) - 0.4).abs() < 1e-6);
        // zero independence → 0.0
        assert!((xnor_consensus(8.0, 10.0, 0.0)).abs() < 1e-6);
    }

    #[test]
    fn nand_violation_detection() {
        assert!(nand_violated(true, true));
        assert!(!nand_violated(true, false));
        assert!(!nand_violated(false, true));
        assert!(!nand_violated(false, false));
    }

    #[test]
    fn feedback_action_trust_clamp_semantics() {
        // Verify the delta semantics are clear in types
        let action = FeedbackAction::UpdateTrust {
            peer: NodeId(5),
            dimension: TrustDimension::Kinematic,
            delta: -0.05,
        };
        match action {
            FeedbackAction::UpdateTrust { delta, .. } => {
                assert!(delta >= -0.1 && delta <= 0.05);
            }
            _ => panic!("wrong variant"),
        }
    }
}
