//! 4-dimensional trust decomposition for swarm nodes.
//!
//! Instead of a single scalar trust ∈ [0,1], each peer is tracked across
//! four independent dimensions, each updated from a different signal:
//!
//! - **integrity**: data consistency, absence of corruption
//! - **timeliness**: freshness and punctuality of updates
//! - **kinematic**: position/velocity accuracy vs. physical plausibility
//! - **consensus**: agreement with other peers on shared state

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::NodeId;

/// Four trust dimensions for a single peer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrustVector {
    /// Data integrity — no corruption, valid ranges, consistent history.
    pub integrity: f64,
    /// Timeliness — updates arrive on time, not stale.
    pub timeliness: f64,
    /// Kinematic plausibility — position/velocity within physical bounds.
    pub kinematic: f64,
    /// Consensus — agreement with majority of independent peers.
    pub consensus: f64,
}

impl Default for TrustVector {
    fn default() -> Self {
        Self {
            integrity: 0.5,
            timeliness: 0.5,
            kinematic: 0.5,
            consensus: 0.5,
        }
    }
}

impl TrustVector {
    /// Aggregate scalar trust — weighted mean of dimensions.
    pub fn aggregate(&self) -> f64 {
        let w = [0.30, 0.20, 0.25, 0.25]; // integrity, timeliness, kinematic, consensus
        let vals = [
            self.integrity,
            self.timeliness,
            self.kinematic,
            self.consensus,
        ];
        let sum: f64 = w.iter().zip(vals.iter()).map(|(wi, vi)| wi * vi).sum();
        sum.clamp(0.0, 1.0)
    }

    /// Minimum dimension — most pessimistic trust.
    pub fn min_dim(&self) -> f64 {
        self.integrity
            .min(self.timeliness)
            .min(self.kinematic)
            .min(self.consensus)
    }
}

/// Configuration for trust update dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustConfig {
    /// Learning rate for EMA trust updates ∈ (0, 1).
    pub learning_rate: f64,
    /// Decay halflife in seconds — trust drifts to 0.5 over time.
    pub decay_halflife_s: f64,
    /// Floor: trust never drops below this.
    pub floor: f64,
    /// Ceiling: trust never exceeds this.
    pub ceiling: f64,
}

impl Default for TrustConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.12,
            decay_halflife_s: 30.0,
            floor: 0.05,
            ceiling: 0.99,
        }
    }
}

/// Per-node trust tracker with 4-dim decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustTracker {
    config: TrustConfig,
    /// Per-peer trust vectors.
    peers: HashMap<NodeId, TrustVector>,
    /// Timestamp of last update per peer per dimension.
    last_update: HashMap<NodeId, [f64; 4]>,
}

impl TrustTracker {
    pub fn new(config: TrustConfig) -> Self {
        Self {
            config,
            peers: HashMap::new(),
            last_update: HashMap::new(),
        }
    }

    /// Get trust vector for a peer (default 0.5 on all dims if unknown).
    pub fn trust_for(&self, peer: NodeId) -> TrustVector {
        self.peers.get(&peer).copied().unwrap_or_default()
    }

    /// Get aggregate scalar trust for a peer.
    pub fn aggregate_trust(&self, peer: NodeId) -> f64 {
        self.trust_for(peer).aggregate()
    }

    /// Update a single trust dimension for a peer.
    ///
    /// `observation` ∈ [0, 1]: how well the peer performed on this dimension.
    pub fn observe(
        &mut self,
        peer: NodeId,
        dimension: TrustDimension,
        observation: f64,
        now_s: f64,
    ) {
        let obs = clamp01(observation);
        let tv = self.peers.entry(peer).or_default();
        let timestamps = self.last_update.entry(peer).or_insert([now_s; 4]);
        let dim_idx = dimension as usize;

        // Decay toward 0.5 based on elapsed time.
        let age = (now_s - timestamps[dim_idx]).max(0.0);
        let current = dimension.get(tv);
        let decayed = if self.config.decay_halflife_s > 1e-6 && age > 0.0 {
            let memory = f64::exp(-f64::ln(2.0) * age / self.config.decay_halflife_s);
            0.5 + (current - 0.5) * memory
        } else {
            current
        };

        // EMA update.
        let lr = self.config.learning_rate.clamp(0.0, 1.0);
        let updated =
            ((1.0 - lr) * decayed + lr * obs).clamp(self.config.floor, self.config.ceiling);

        dimension.set(tv, updated);
        timestamps[dim_idx] = now_s;
    }

    /// Observe all four dimensions at once.
    pub fn observe_all(
        &mut self,
        peer: NodeId,
        integrity: f64,
        timeliness: f64,
        kinematic: f64,
        consensus: f64,
        now_s: f64,
    ) {
        self.observe(peer, TrustDimension::Integrity, integrity, now_s);
        self.observe(peer, TrustDimension::Timeliness, timeliness, now_s);
        self.observe(peer, TrustDimension::Kinematic, kinematic, now_s);
        self.observe(peer, TrustDimension::Consensus, consensus, now_s);
    }

    /// Number of tracked peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Mean aggregate trust across all known peers.
    pub fn mean_aggregate(&self) -> f64 {
        if self.peers.is_empty() {
            return 0.5;
        }
        let sum: f64 = self.peers.values().map(|tv| tv.aggregate()).sum();
        sum / self.peers.len() as f64
    }
}

/// Which trust dimension to update.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustDimension {
    Integrity = 0,
    Timeliness = 1,
    Kinematic = 2,
    Consensus = 3,
}

impl TrustDimension {
    fn get(self, tv: &TrustVector) -> f64 {
        match self {
            Self::Integrity => tv.integrity,
            Self::Timeliness => tv.timeliness,
            Self::Kinematic => tv.kinematic,
            Self::Consensus => tv.consensus,
        }
    }

    fn set(self, tv: &mut TrustVector, val: f64) {
        match self {
            Self::Integrity => tv.integrity = val,
            Self::Timeliness => tv.timeliness = val,
            Self::Kinematic => tv.kinematic = val,
            Self::Consensus => tv.consensus = val,
        }
    }
}

fn clamp01(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_trust_is_neutral() {
        let tv = TrustVector::default();
        assert!((tv.aggregate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn observe_increases_trust() {
        let mut tracker = TrustTracker::new(TrustConfig::default());
        let peer = NodeId(1);

        // Observe high integrity repeatedly
        for t in 0..10 {
            tracker.observe(peer, TrustDimension::Integrity, 0.95, t as f64);
        }

        let tv = tracker.trust_for(peer);
        assert!(
            tv.integrity > 0.7,
            "repeated high obs should raise trust, got {}",
            tv.integrity
        );
        // Other dimensions stay at default
        assert!((tv.timeliness - 0.5).abs() < 0.15);
    }

    #[test]
    fn observe_decreases_trust() {
        let mut tracker = TrustTracker::new(TrustConfig::default());
        let peer = NodeId(2);

        for t in 0..10 {
            tracker.observe(peer, TrustDimension::Consensus, 0.1, t as f64);
        }

        let tv = tracker.trust_for(peer);
        assert!(
            tv.consensus < 0.3,
            "repeated low obs should lower trust, got {}",
            tv.consensus
        );
    }

    #[test]
    fn decay_toward_neutral() {
        let mut tracker = TrustTracker::new(TrustConfig {
            decay_halflife_s: 1.0,
            ..TrustConfig::default()
        });
        let peer = NodeId(3);

        // Set high trust
        tracker.observe(peer, TrustDimension::Kinematic, 0.99, 0.0);
        let before = tracker.trust_for(peer).kinematic;

        // Observe after long delay — should decay toward 0.5
        tracker.observe(peer, TrustDimension::Kinematic, 0.99, 100.0);
        // The decay should have pulled it back before the new obs
        // With halflife=1s and dt=100s, memory ≈ 0, so trust resets near 0.5 then EMA toward 0.99
        let after = tracker.trust_for(peer).kinematic;
        // After decay + fresh obs, should be near (1-lr)*0.5 + lr*0.99
        assert!(after > 0.5, "fresh obs should pull up from decayed neutral");
        assert!(
            after < before + 0.1,
            "shouldn't exceed pre-decay level significantly"
        );
    }

    #[test]
    fn aggregate_weighted() {
        let tv = TrustVector {
            integrity: 1.0,
            timeliness: 1.0,
            kinematic: 1.0,
            consensus: 1.0,
        };
        assert!((tv.aggregate() - 1.0).abs() < 1e-6);

        let tv2 = TrustVector {
            integrity: 0.0,
            timeliness: 0.0,
            kinematic: 0.0,
            consensus: 0.0,
        };
        assert!((tv2.aggregate() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn min_dim_returns_weakest() {
        let tv = TrustVector {
            integrity: 0.9,
            timeliness: 0.2,
            kinematic: 0.8,
            consensus: 0.7,
        };
        assert!((tv.min_dim() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn observe_all_updates_every_dimension() {
        let mut tracker = TrustTracker::new(TrustConfig::default());
        let peer = NodeId(4);
        tracker.observe_all(peer, 0.9, 0.8, 0.7, 0.6, 0.0);

        let tv = tracker.trust_for(peer);
        // After one observation from default 0.5: (1-0.12)*0.5 + 0.12*obs
        assert!(tv.integrity > 0.5);
        assert!(tv.timeliness > 0.5);
        assert!(tv.kinematic > 0.5);
        assert!(tv.consensus > 0.5);
    }

    // -- Edge-case / boundary tests --

    #[test]
    fn peer_count_tracks_observed_peers() {
        let mut tracker = TrustTracker::new(TrustConfig::default());
        assert_eq!(tracker.peer_count(), 0);
        tracker.observe(NodeId(1), TrustDimension::Integrity, 0.8, 0.0);
        assert_eq!(tracker.peer_count(), 1);
        tracker.observe(NodeId(2), TrustDimension::Kinematic, 0.7, 0.0);
        assert_eq!(tracker.peer_count(), 2);
        // Same peer again → no increase
        tracker.observe(NodeId(1), TrustDimension::Timeliness, 0.9, 0.0);
        assert_eq!(tracker.peer_count(), 2);
    }

    #[test]
    fn mean_aggregate_with_multiple_peers() {
        let mut tracker = TrustTracker::new(TrustConfig::default());
        // Observe two peers with high trust
        tracker.observe_all(NodeId(1), 0.9, 0.9, 0.9, 0.9, 0.0);
        tracker.observe_all(NodeId(2), 0.9, 0.9, 0.9, 0.9, 0.0);
        let mean = tracker.mean_aggregate();
        assert!(mean > 0.5, "high observations → mean aggregate > 0.5");
        assert!(mean <= 1.0);
    }

    #[test]
    fn aggregate_always_in_0_1() {
        // Extreme values
        let tv_zero = TrustVector {
            integrity: 0.0,
            timeliness: 0.0,
            kinematic: 0.0,
            consensus: 0.0,
        };
        let tv_one = TrustVector {
            integrity: 1.0,
            timeliness: 1.0,
            kinematic: 1.0,
            consensus: 1.0,
        };
        assert!(tv_zero.aggregate() >= 0.0);
        assert!(tv_one.aggregate() <= 1.0);
    }

    #[test]
    fn dimensions_independent() {
        let mut tracker = TrustTracker::new(TrustConfig::default());
        let peer = NodeId(10);
        // Observe only integrity
        tracker.observe(peer, TrustDimension::Integrity, 0.99, 0.0);
        let tv = tracker.trust_for(peer);
        assert!(tv.integrity > 0.5, "observed dimension should change");
        // Other dims stay at default 0.5
        assert!(
            (tv.timeliness - 0.5).abs() < 1e-6,
            "unobserved dimension should stay at default"
        );
        assert!(
            (tv.kinematic - 0.5).abs() < 1e-6,
            "unobserved dimension should stay at default"
        );
        assert!(
            (tv.consensus - 0.5).abs() < 1e-6,
            "unobserved dimension should stay at default"
        );
    }
}
