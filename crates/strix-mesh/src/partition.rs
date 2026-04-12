//! Network partition detection and recovery.

use crate::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Connectivity status of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectivityStatus {
    /// Normal connectivity.
    Connected,
    /// Degraded — some peers unreachable.
    Degraded { reachable_fraction: u8 }, // 0-100 percent
    /// Isolated — no peers reachable.
    Isolated,
}

/// Partition detector using gossip round connectivity.
pub struct PartitionDetector {
    /// Known peers and when we last heard from them.
    last_heard: HashMap<NodeId, f64>,
    /// Expected total peer count.
    expected_peers: usize,
    /// Max age before a peer is considered unreachable.
    max_age_s: f64,
}

impl PartitionDetector {
    /// Create a new detector.
    pub fn new(expected_peers: usize, max_age_s: f64) -> Self {
        Self {
            last_heard: HashMap::new(),
            expected_peers: expected_peers.max(1),
            max_age_s,
        }
    }

    /// Record that we heard from a peer.
    pub fn record_contact(&mut self, peer: NodeId, now: f64) {
        self.last_heard.insert(peer, now);
    }

    /// Get the set of reachable peers at time `now`.
    pub fn reachable_peers(&self, now: f64) -> HashSet<NodeId> {
        self.last_heard
            .iter()
            .filter(|(_, &last)| now - last <= self.max_age_s)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get the set of unreachable peers.
    pub fn unreachable_peers(&self, now: f64) -> HashSet<NodeId> {
        self.last_heard
            .iter()
            .filter(|(_, &last)| now - last > self.max_age_s)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Detect current connectivity status.
    pub fn detect(&self, now: f64) -> ConnectivityStatus {
        let reachable = self
            .last_heard
            .values()
            .filter(|&&last| now - last <= self.max_age_s)
            .count();
        if reachable == 0 {
            ConnectivityStatus::Isolated
        } else if reachable < self.expected_peers {
            let frac = ((reachable as f64 / self.expected_peers as f64) * 100.0) as u8;
            ConnectivityStatus::Degraded {
                reachable_fraction: frac,
            }
        } else {
            ConnectivityStatus::Connected
        }
    }

    /// Number of rounds since last hearing from any peer.
    /// Returns 0 if we heard from someone this round.
    pub fn silence_rounds(&self, now: f64, round_interval: f64) -> u32 {
        if round_interval <= 0.0 {
            return 0;
        }
        let most_recent = self.last_heard.values().cloned().fold(0.0_f64, f64::max);
        if most_recent <= 0.0 {
            return u32::MAX;
        }
        ((now - most_recent) / round_interval) as u32
    }

    /// Update expected peer count (e.g. after drone loss).
    pub fn set_expected_peers(&mut self, count: usize) {
        self.expected_peers = count.max(1);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a detector pre-seeded with `n` peers all heard at time `now`.
    fn make_detector(n: usize, now: f64, max_age: f64) -> PartitionDetector {
        let mut d = PartitionDetector::new(n, max_age);
        for i in 0..n {
            d.record_contact(NodeId(i as u32), now);
        }
        d
    }

    #[test]
    fn detect_connected() {
        let d = make_detector(4, 100.0, 10.0);
        assert_eq!(d.detect(105.0), ConnectivityStatus::Connected);
    }

    #[test]
    fn detect_degraded() {
        let mut d = PartitionDetector::new(4, 10.0);
        // Peers 0 and 1 heard recently; peers 2 and 3 timed out.
        d.record_contact(NodeId(0), 100.0);
        d.record_contact(NodeId(1), 100.0);
        d.record_contact(NodeId(2), 80.0); // 20 s ago → stale
        d.record_contact(NodeId(3), 80.0);

        let status = d.detect(105.0);
        // reachable = 2, expected = 4 → frac = 50
        assert_eq!(
            status,
            ConnectivityStatus::Degraded {
                reachable_fraction: 50
            }
        );
    }

    #[test]
    fn detect_isolated() {
        let mut d = PartitionDetector::new(3, 5.0);
        // All peers stale.
        d.record_contact(NodeId(0), 0.0);
        d.record_contact(NodeId(1), 0.0);
        d.record_contact(NodeId(2), 0.0);

        assert_eq!(d.detect(100.0), ConnectivityStatus::Isolated);
    }

    #[test]
    fn detect_isolated_no_contacts() {
        // Never heard from anyone.
        let d = PartitionDetector::new(3, 5.0);
        assert_eq!(d.detect(10.0), ConnectivityStatus::Isolated);
    }

    #[test]
    fn reachable_and_unreachable() {
        let mut d = PartitionDetector::new(4, 10.0);
        d.record_contact(NodeId(0), 100.0);
        d.record_contact(NodeId(1), 100.0);
        d.record_contact(NodeId(2), 80.0); // stale
        d.record_contact(NodeId(3), 80.0); // stale

        let now = 105.0;
        let reachable = d.reachable_peers(now);
        let unreachable = d.unreachable_peers(now);

        assert_eq!(reachable, [NodeId(0), NodeId(1)].into_iter().collect());
        assert_eq!(unreachable, [NodeId(2), NodeId(3)].into_iter().collect());
    }

    #[test]
    fn silence_rounds_counting() {
        let mut d = PartitionDetector::new(2, 10.0);
        d.record_contact(NodeId(0), 100.0);
        d.record_contact(NodeId(1), 90.0);

        // Most recent = 100.0; now = 130.0; interval = 10.0 → 3 rounds.
        assert_eq!(d.silence_rounds(130.0, 10.0), 3);
        // now == most_recent → 0 rounds.
        assert_eq!(d.silence_rounds(100.0, 10.0), 0);
    }

    #[test]
    fn silence_rounds_no_contacts() {
        let d = PartitionDetector::new(2, 10.0);
        // No contacts ever recorded → should return u32::MAX.
        assert_eq!(d.silence_rounds(50.0, 5.0), u32::MAX);
    }

    #[test]
    fn silence_rounds_zero_interval() {
        let d = make_detector(2, 10.0, 5.0);
        // Zero interval guard: should return 0.
        assert_eq!(d.silence_rounds(100.0, 0.0), 0);
    }

    #[test]
    fn record_updates_timestamp() {
        let mut d = PartitionDetector::new(1, 10.0);
        d.record_contact(NodeId(0), 10.0);
        // Before update, peer is stale at t=25.
        assert_eq!(d.detect(25.0), ConnectivityStatus::Isolated);

        // Update contact at t=24 — now fresh.
        d.record_contact(NodeId(0), 24.0);
        assert_eq!(d.detect(25.0), ConnectivityStatus::Connected);
    }

    #[test]
    fn set_expected_peers_updates_status() {
        let mut d = make_detector(4, 100.0, 10.0);
        // All 4 fresh → Connected.
        assert_eq!(d.detect(105.0), ConnectivityStatus::Connected);

        // Now expect 8 → only 4/8 reachable → Degraded.
        d.set_expected_peers(8);
        assert!(matches!(
            d.detect(105.0),
            ConnectivityStatus::Degraded { .. }
        ));

        // Reset to 4 → Connected again.
        d.set_expected_peers(4);
        assert_eq!(d.detect(105.0), ConnectivityStatus::Connected);
    }

    #[test]
    fn set_expected_peers_min_one() {
        let mut d = PartitionDetector::new(1, 5.0);
        d.set_expected_peers(0); // must clamp to 1
        d.record_contact(NodeId(0), 10.0);
        // 1/1 reachable → Connected.
        assert_eq!(d.detect(10.0), ConnectivityStatus::Connected);
    }

    #[test]
    fn serde_roundtrip_connectivity_status() {
        let cases = [
            ConnectivityStatus::Connected,
            ConnectivityStatus::Degraded {
                reachable_fraction: 50,
            },
            ConnectivityStatus::Isolated,
        ];
        for &status in &cases {
            let json = serde_json::to_string(&status).unwrap();
            let back: ConnectivityStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, back);
        }
    }
}
