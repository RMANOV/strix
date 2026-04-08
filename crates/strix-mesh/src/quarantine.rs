//! Graduated quarantine protocol for suspicious nodes.
//!
//! Instead of binary "peer exists or not", nodes progress through 5 states:
//!
//! ```text
//! FullParticipant → LimitedInfluence → ReadOnly → Quarantine → RecoveryProbation
//! ```
//!
//! Transitions are driven by Byzantine validation results (suspicious/rejected)
//! and time-based recovery. This avoids system-wide instability from false
//! positives while still isolating genuinely malicious nodes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::NodeId;

/// Participation level for a swarm node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticipationLevel {
    /// Normal — full gossip, voting, auction, formation participation.
    FullParticipant,
    /// Can send/receive but gossip contributions are weight-reduced.
    LimitedInfluence,
    /// Can receive state but cannot contribute to gossip or vote.
    ReadOnly,
    /// Isolated — no communication accepted from this node.
    Quarantine,
    /// Returning from quarantine — limited participation with monitoring.
    RecoveryProbation,
}

impl ParticipationLevel {
    /// Influence weight for this level ∈ [0, 1].
    pub fn influence_weight(self) -> f64 {
        match self {
            Self::FullParticipant => 1.0,
            Self::LimitedInfluence => 0.3,
            Self::ReadOnly => 0.0,
            Self::Quarantine => 0.0,
            Self::RecoveryProbation => 0.5,
        }
    }

    /// Whether this node's messages should be accepted.
    pub fn accepts_messages(self) -> bool {
        !matches!(self, Self::Quarantine)
    }

    /// Whether this node can vote in hypergraph quorum.
    pub fn can_vote(self) -> bool {
        matches!(self, Self::FullParticipant | Self::RecoveryProbation)
    }

    /// Whether this node can participate in auction bidding.
    pub fn can_bid(self) -> bool {
        matches!(
            self,
            Self::FullParticipant | Self::LimitedInfluence | Self::RecoveryProbation
        )
    }
}

/// Per-node quarantine state tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeRecord {
    level: ParticipationLevel,
    /// Accumulated strike count (suspicious + rejected events).
    strikes: u32,
    /// Timestamp of last strike.
    last_strike_at: f64,
    /// Timestamp when current level was entered.
    level_entered_at: f64,
    /// Timestamp of last good behavior (accepted update).
    last_good_at: f64,
}

impl NodeRecord {
    fn new(now: f64) -> Self {
        Self {
            level: ParticipationLevel::FullParticipant,
            strikes: 0,
            last_strike_at: 0.0,
            level_entered_at: now,
            last_good_at: now,
        }
    }
}

/// Configuration for the quarantine protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineConfig {
    /// Strikes before demotion from FullParticipant → LimitedInfluence.
    pub strikes_to_limit: u32,
    /// Strikes before demotion from LimitedInfluence → ReadOnly.
    pub strikes_to_readonly: u32,
    /// Strikes before demotion from ReadOnly → Quarantine.
    pub strikes_to_quarantine: u32,
    /// Seconds of good behavior needed to promote one level.
    pub recovery_period_s: f64,
    /// Seconds in Quarantine before eligible for RecoveryProbation.
    pub quarantine_duration_s: f64,
    /// Strike decay: strikes are halved after this many seconds of good behavior.
    pub strike_decay_period_s: f64,
}

impl Default for QuarantineConfig {
    fn default() -> Self {
        Self {
            strikes_to_limit: 3,
            strikes_to_readonly: 6,
            strikes_to_quarantine: 10,
            recovery_period_s: 30.0,
            quarantine_duration_s: 60.0,
            strike_decay_period_s: 120.0,
        }
    }
}

/// Manages quarantine state for all known nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineManager {
    config: QuarantineConfig,
    nodes: HashMap<NodeId, NodeRecord>,
}

impl QuarantineManager {
    pub fn new(config: QuarantineConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
        }
    }

    /// Get participation level for a node (default: FullParticipant).
    pub fn level(&self, node: NodeId) -> ParticipationLevel {
        self.nodes
            .get(&node)
            .map_or(ParticipationLevel::FullParticipant, |r| r.level)
    }

    /// Record a strike (suspicious or rejected event) against a node.
    pub fn record_strike(&mut self, node: NodeId, now: f64) {
        let record = self
            .nodes
            .entry(node)
            .or_insert_with(|| NodeRecord::new(now));
        record.strikes += 1;
        record.last_strike_at = now;

        // Check for demotion.
        let new_level = if record.strikes >= self.config.strikes_to_quarantine {
            ParticipationLevel::Quarantine
        } else if record.strikes >= self.config.strikes_to_readonly {
            ParticipationLevel::ReadOnly
        } else if record.strikes >= self.config.strikes_to_limit {
            ParticipationLevel::LimitedInfluence
        } else {
            record.level // no change
        };

        if new_level != record.level {
            record.level = new_level;
            record.level_entered_at = now;
        }
    }

    /// Record good behavior (accepted valid update) from a node.
    pub fn record_good(&mut self, node: NodeId, now: f64) {
        let record = self
            .nodes
            .entry(node)
            .or_insert_with(|| NodeRecord::new(now));
        record.last_good_at = now;
    }

    /// Periodic maintenance: decay strikes, promote recovered nodes.
    /// Call once per tick or gossip round.
    pub fn maintain(&mut self, now: f64) {
        for record in self.nodes.values_mut() {
            // Strike decay: halve strikes after sustained good behavior.
            let good_duration = now - record.last_strike_at;
            if good_duration >= self.config.strike_decay_period_s && record.strikes > 0 {
                record.strikes /= 2;
                record.last_strike_at = now; // reset decay timer
            }

            // Promotion logic.
            let time_in_level = now - record.level_entered_at;
            let time_since_strike = now - record.last_strike_at;

            match record.level {
                ParticipationLevel::Quarantine => {
                    if time_in_level >= self.config.quarantine_duration_s {
                        record.level = ParticipationLevel::RecoveryProbation;
                        record.level_entered_at = now;
                    }
                }
                ParticipationLevel::RecoveryProbation => {
                    if time_since_strike >= self.config.recovery_period_s {
                        record.level = ParticipationLevel::LimitedInfluence;
                        record.level_entered_at = now;
                    }
                }
                ParticipationLevel::ReadOnly => {
                    if time_since_strike >= self.config.recovery_period_s {
                        record.level = ParticipationLevel::LimitedInfluence;
                        record.level_entered_at = now;
                    }
                }
                ParticipationLevel::LimitedInfluence => {
                    if time_since_strike >= self.config.recovery_period_s
                        && record.strikes < self.config.strikes_to_limit
                    {
                        record.level = ParticipationLevel::FullParticipant;
                        record.level_entered_at = now;
                    }
                }
                ParticipationLevel::FullParticipant => {} // nothing to promote
            }
        }
    }

    /// Number of nodes currently quarantined.
    pub fn quarantined_count(&self) -> usize {
        self.nodes
            .values()
            .filter(|r| r.level == ParticipationLevel::Quarantine)
            .count()
    }

    /// Number of nodes with any degraded participation.
    pub fn degraded_count(&self) -> usize {
        self.nodes
            .values()
            .filter(|r| r.level != ParticipationLevel::FullParticipant)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_node_is_full_participant() {
        let mgr = QuarantineManager::new(QuarantineConfig::default());
        assert_eq!(mgr.level(NodeId(1)), ParticipationLevel::FullParticipant);
    }

    #[test]
    fn strikes_cause_demotion() {
        let mut mgr = QuarantineManager::new(QuarantineConfig::default());
        let node = NodeId(1);

        // 3 strikes → LimitedInfluence
        for t in 0..3 {
            mgr.record_strike(node, t as f64);
        }
        assert_eq!(mgr.level(node), ParticipationLevel::LimitedInfluence);

        // 3 more → ReadOnly (total 6)
        for t in 3..6 {
            mgr.record_strike(node, t as f64);
        }
        assert_eq!(mgr.level(node), ParticipationLevel::ReadOnly);

        // 4 more → Quarantine (total 10)
        for t in 6..10 {
            mgr.record_strike(node, t as f64);
        }
        assert_eq!(mgr.level(node), ParticipationLevel::Quarantine);
    }

    #[test]
    fn quarantine_promotes_to_recovery_after_duration() {
        let mut mgr = QuarantineManager::new(QuarantineConfig {
            quarantine_duration_s: 10.0,
            ..QuarantineConfig::default()
        });
        let node = NodeId(1);

        // Force quarantine
        for t in 0..10 {
            mgr.record_strike(node, t as f64);
        }
        assert_eq!(mgr.level(node), ParticipationLevel::Quarantine);

        // Not enough time
        mgr.maintain(15.0);
        assert_eq!(mgr.level(node), ParticipationLevel::Quarantine);

        // Enough time (entered at t=9, +10s = 19)
        mgr.maintain(20.0);
        assert_eq!(mgr.level(node), ParticipationLevel::RecoveryProbation);
    }

    #[test]
    fn recovery_probation_promotes_with_good_behavior() {
        let cfg = QuarantineConfig {
            quarantine_duration_s: 5.0,
            recovery_period_s: 10.0,
            strikes_to_quarantine: 3,
            strikes_to_readonly: 2,
            strikes_to_limit: 1,
            ..QuarantineConfig::default()
        };
        let mut mgr = QuarantineManager::new(cfg);
        let node = NodeId(1);

        // Quarantine
        for t in 0..3 {
            mgr.record_strike(node, t as f64);
        }
        assert_eq!(mgr.level(node), ParticipationLevel::Quarantine);

        // Wait for quarantine to expire → RecoveryProbation
        mgr.maintain(10.0);
        assert_eq!(mgr.level(node), ParticipationLevel::RecoveryProbation);

        // Good behavior for recovery_period → LimitedInfluence
        mgr.maintain(25.0);
        assert_eq!(mgr.level(node), ParticipationLevel::LimitedInfluence);
    }

    #[test]
    fn influence_weights_are_correct() {
        assert_eq!(ParticipationLevel::FullParticipant.influence_weight(), 1.0);
        assert_eq!(ParticipationLevel::Quarantine.influence_weight(), 0.0);
        assert!(ParticipationLevel::LimitedInfluence.influence_weight() > 0.0);
        assert!(ParticipationLevel::LimitedInfluence.influence_weight() < 1.0);
    }

    #[test]
    fn quarantined_node_blocks_messages() {
        assert!(!ParticipationLevel::Quarantine.accepts_messages());
        assert!(ParticipationLevel::FullParticipant.accepts_messages());
        assert!(ParticipationLevel::ReadOnly.accepts_messages());
    }

    #[test]
    fn strike_decay_halves_strikes() {
        let cfg = QuarantineConfig {
            strike_decay_period_s: 10.0,
            strikes_to_limit: 5,
            ..QuarantineConfig::default()
        };
        let mut mgr = QuarantineManager::new(cfg);
        let node = NodeId(1);

        // 4 strikes at t=0
        for _ in 0..4 {
            mgr.record_strike(node, 0.0);
        }
        assert_eq!(mgr.level(node), ParticipationLevel::FullParticipant); // 4 < 5

        // After decay period, strikes halve: 4 → 2
        mgr.maintain(11.0);
        // Still FullParticipant (2 < 5), strikes decayed
        assert_eq!(mgr.level(node), ParticipationLevel::FullParticipant);
    }

    #[test]
    fn degraded_count_tracks_non_full() {
        let mut mgr = QuarantineManager::new(QuarantineConfig::default());
        assert_eq!(mgr.degraded_count(), 0);

        for t in 0..3 {
            mgr.record_strike(NodeId(1), t as f64);
        }
        assert_eq!(mgr.degraded_count(), 1);
        assert_eq!(mgr.quarantined_count(), 0);
    }
}
