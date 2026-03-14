//! # Leader Election and Rank Management
//!
//! Lightweight consensus mechanisms for fractal hierarchy operations:
//!
//! - **Leader election**: bully-algorithm variant — highest-ranked drone wins.
//!   Completes in O(group_size) messages.
//! - **Rank management**: initial rank from capability + experience, adjusted
//!   on mission performance, inherited on leader loss.
//! - **Heartbeat monitoring**: 3 missed heartbeats → node declared dead →
//!   triggers re-election and task re-auction.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::NodeId;

// ---------------------------------------------------------------------------
// Rank
// ---------------------------------------------------------------------------

/// Capability profile used to compute rank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneCapability {
    /// Sensor quality score (0.0–1.0).
    pub sensor_quality: f64,
    /// Remaining battery (0.0–1.0).
    pub battery: f64,
    /// How central the drone is to its group (0.0–1.0, higher = more central).
    pub position_centrality: f64,
    /// Mission experience (cumulative hours or missions completed).
    pub experience: f64,
}

impl DroneCapability {
    /// Compute a scalar rank from capability dimensions.
    ///
    /// Weights: sensor 0.3, battery 0.2, centrality 0.2, experience 0.3.
    pub fn rank_score(&self) -> f64 {
        0.3 * self.sensor_quality
            + 0.2 * self.battery
            + 0.2 * self.position_centrality
            + 0.3 * self.experience.min(1.0) // cap at 1.0 for normalization
    }
}

/// Tracks ranks for all drones in the hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankManager {
    /// Current rank per drone.
    pub ranks: HashMap<NodeId, f64>,
    /// Performance adjustment accumulator.
    performance_delta: HashMap<NodeId, f64>,
}

impl RankManager {
    /// Create a new rank manager.
    pub fn new() -> Self {
        Self {
            ranks: HashMap::new(),
            performance_delta: HashMap::new(),
        }
    }

    /// Set initial rank from capability profile.
    pub fn set_initial_rank(&mut self, node: NodeId, capability: &DroneCapability) {
        self.ranks.insert(node, capability.rank_score());
    }

    /// Get current rank for a drone (returns 0.0 if unknown).
    pub fn rank(&self, node: NodeId) -> f64 {
        self.ranks.get(&node).copied().unwrap_or(0.0)
    }

    /// Adjust rank based on mission performance.
    ///
    /// `delta` is typically in [-0.1, +0.1]. Rank is clamped to [0.0, 2.0].
    pub fn adjust_rank(&mut self, node: NodeId, delta: f64) {
        let current = self.rank(node);
        let new_rank = (current + delta).clamp(0.0, 2.0);
        self.ranks.insert(node, new_rank);
        *self.performance_delta.entry(node).or_insert(0.0) += delta;
    }

    /// Remove a drone (e.g. declared dead).
    pub fn remove(&mut self, node: NodeId) {
        self.ranks.remove(&node);
        self.performance_delta.remove(&node);
    }

    /// Get cumulative performance adjustment for a drone.
    pub fn performance_score(&self, node: NodeId) -> f64 {
        self.performance_delta.get(&node).copied().unwrap_or(0.0)
    }
}

impl Default for RankManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Leader election (bully variant)
// ---------------------------------------------------------------------------

/// Result of a leader election.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectionResult {
    /// The elected leader.
    pub leader: NodeId,
    /// Rank of the elected leader.
    pub leader_rank: f64,
    /// Number of messages exchanged during election.
    pub messages_exchanged: u32,
    /// All participants (including leader).
    pub participants: Vec<NodeId>,
}

/// Elect a leader for a group of drones.
///
/// Uses a bully-algorithm variant: the drone with the highest rank wins.
/// In case of a tie, the higher `NodeId` wins (deterministic tiebreak).
///
/// Criteria (embedded in rank):
/// - Sensor capability, battery, position centrality, experience
///
/// Completes in O(group_size) messages:
/// - Each candidate broadcasts its rank
/// - Only higher-ranked nodes respond with "I'm here"
/// - If nobody responds → you're the leader
pub fn elect_leader(candidates: &[NodeId], rank_manager: &RankManager) -> Option<ElectionResult> {
    if candidates.is_empty() {
        return None;
    }

    // The bully algorithm: highest rank wins. Tiebreak by NodeId.
    let leader = *candidates.iter().max_by(|&&a, &&b| {
        let ra = rank_manager.rank(a);
        let rb = rank_manager.rank(b);
        ra.partial_cmp(&rb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    })?;

    // In the bully algorithm, each node announces itself to all
    // higher-ranked nodes. Worst case: O(n) messages total.
    // Here we simulate the message count.
    let messages = candidates.len() as u32;

    Some(ElectionResult {
        leader,
        leader_rank: rank_manager.rank(leader),
        messages_exchanged: messages,
        participants: candidates.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Heartbeat monitoring
// ---------------------------------------------------------------------------

/// Tracks heartbeats and detects failed nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatMonitor {
    /// Last heartbeat time per node.
    last_heartbeat: HashMap<NodeId, f64>,
    /// Number of consecutive misses per node.
    miss_count: HashMap<NodeId, u32>,
    /// How many misses before declaring death.
    miss_limit: u32,
    /// Expected heartbeat interval (seconds).
    interval: f64,
    /// Nodes currently declared dead.
    dead_nodes: Vec<NodeId>,
}

/// Event emitted by the heartbeat monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HeartbeatEvent {
    /// A node has been declared dead.
    NodeDead {
        node: NodeId,
        /// True if this node was the leader of its group.
        was_leader: bool,
    },
    /// A node that was considered dead has come back.
    NodeResurrected { node: NodeId },
}

impl HeartbeatMonitor {
    /// Create a new monitor.
    ///
    /// - `interval`: expected heartbeat period in seconds.
    /// - `miss_limit`: how many consecutive misses → death (typically 3).
    pub fn new(interval: f64, miss_limit: u32) -> Self {
        Self {
            last_heartbeat: HashMap::new(),
            miss_count: HashMap::new(),
            miss_limit,
            interval,
            dead_nodes: Vec::new(),
        }
    }

    /// Record a heartbeat from a node.
    pub fn record_heartbeat(&mut self, node: NodeId, timestamp: f64) -> Option<HeartbeatEvent> {
        self.last_heartbeat.insert(node, timestamp);
        self.miss_count.insert(node, 0);

        // Resurrection?
        if self.dead_nodes.contains(&node) {
            self.dead_nodes.retain(|&n| n != node);
            return Some(HeartbeatEvent::NodeResurrected { node });
        }
        None
    }

    /// Register a node to be monitored.
    pub fn register(&mut self, node: NodeId, current_time: f64) {
        self.last_heartbeat.entry(node).or_insert(current_time);
        self.miss_count.entry(node).or_insert(0);
    }

    /// Check for dead nodes at the current time.
    ///
    /// `leader_check`: function that returns `true` if a node is a leader.
    ///
    /// Returns list of death/resurrection events.
    pub fn check<F>(&mut self, now: f64, leader_check: F) -> Vec<HeartbeatEvent>
    where
        F: Fn(NodeId) -> bool,
    {
        let mut events = Vec::new();
        let deadline = now - self.interval * self.miss_limit as f64;

        let nodes: Vec<NodeId> = self.last_heartbeat.keys().copied().collect();
        for node in nodes {
            if self.dead_nodes.contains(&node) {
                continue;
            }
            let last = self.last_heartbeat.get(&node).copied().unwrap_or(0.0);
            if last < deadline {
                let misses = self.miss_count.entry(node).or_insert(0);
                *misses = ((now - last) / self.interval).floor() as u32;
                if *misses >= self.miss_limit {
                    self.dead_nodes.push(node);
                    events.push(HeartbeatEvent::NodeDead {
                        node,
                        was_leader: leader_check(node),
                    });
                }
            }
        }

        events
    }

    /// List all currently dead nodes.
    pub fn dead_nodes(&self) -> &[NodeId] {
        &self.dead_nodes
    }

    /// Remove a node from tracking entirely.
    pub fn unregister(&mut self, node: NodeId) {
        self.last_heartbeat.remove(&node);
        self.miss_count.remove(&node);
        self.dead_nodes.retain(|&n| n != node);
    }

    /// Whether a specific node is considered dead.
    pub fn is_dead(&self, node: NodeId) -> bool {
        self.dead_nodes.contains(&node)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rank_manager(ranks: &[(u32, f64)]) -> RankManager {
        let mut rm = RankManager::new();
        for &(id, rank) in ranks {
            rm.ranks.insert(NodeId(id), rank);
        }
        rm
    }

    #[test]
    fn elect_highest_rank() {
        let rm = make_rank_manager(&[(0, 0.5), (1, 0.9), (2, 0.3)]);
        let candidates: Vec<NodeId> = vec![NodeId(0), NodeId(1), NodeId(2)];
        let result = elect_leader(&candidates, &rm).unwrap();
        assert_eq!(result.leader, NodeId(1));
    }

    #[test]
    fn elect_tiebreak_by_id() {
        let rm = make_rank_manager(&[(0, 0.5), (1, 0.5)]);
        let candidates: Vec<NodeId> = vec![NodeId(0), NodeId(1)];
        let result = elect_leader(&candidates, &rm).unwrap();
        // Same rank → higher NodeId wins.
        assert_eq!(result.leader, NodeId(1));
    }

    #[test]
    fn elect_empty_returns_none() {
        let rm = RankManager::new();
        assert!(elect_leader(&[], &rm).is_none());
    }

    #[test]
    fn elect_single_candidate() {
        let rm = make_rank_manager(&[(42, 0.7)]);
        let result = elect_leader(&[NodeId(42)], &rm).unwrap();
        assert_eq!(result.leader, NodeId(42));
        assert_eq!(result.messages_exchanged, 1);
    }

    #[test]
    fn rank_manager_adjust() {
        let mut rm = RankManager::new();
        let cap = DroneCapability {
            sensor_quality: 0.8,
            battery: 0.9,
            position_centrality: 0.5,
            experience: 0.6,
        };
        rm.set_initial_rank(NodeId(0), &cap);
        let initial = rm.rank(NodeId(0));

        rm.adjust_rank(NodeId(0), 0.1);
        assert!((rm.rank(NodeId(0)) - (initial + 0.1)).abs() < 1e-10);

        // Clamp test.
        rm.adjust_rank(NodeId(0), 10.0);
        assert!((rm.rank(NodeId(0)) - 2.0).abs() < 1e-10);

        rm.adjust_rank(NodeId(0), -10.0);
        assert!((rm.rank(NodeId(0)) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn heartbeat_detect_dead_node() {
        let mut monitor = HeartbeatMonitor::new(1.0, 3);
        monitor.register(NodeId(0), 0.0);
        monitor.register(NodeId(1), 0.0);

        // Node 0 sends heartbeats, node 1 doesn't.
        monitor.record_heartbeat(NodeId(0), 1.0);
        monitor.record_heartbeat(NodeId(0), 2.0);
        monitor.record_heartbeat(NodeId(0), 3.0);

        // At t=3.5, check: node 1 last heartbeat at t=0, deadline = 3.5 - 3*1 = 0.5.
        // Node 1 last = 0.0 < 0.5 → dead.
        let events = monitor.check(3.5, |_| false);
        assert_eq!(events.len(), 1);
        match &events[0] {
            HeartbeatEvent::NodeDead { node, was_leader } => {
                assert_eq!(*node, NodeId(1));
                assert!(!was_leader);
            }
            _ => panic!("expected NodeDead"),
        }
        assert!(monitor.is_dead(NodeId(1)));
    }

    #[test]
    fn heartbeat_resurrection() {
        let mut monitor = HeartbeatMonitor::new(1.0, 3);
        monitor.register(NodeId(0), 0.0);

        // Declare dead.
        let events = monitor.check(10.0, |_| false);
        assert_eq!(events.len(), 1);
        assert!(monitor.is_dead(NodeId(0)));

        // Node comes back.
        let event = monitor.record_heartbeat(NodeId(0), 11.0);
        assert!(matches!(
            event,
            Some(HeartbeatEvent::NodeResurrected { .. })
        ));
        assert!(!monitor.is_dead(NodeId(0)));
    }

    #[test]
    fn heartbeat_leader_detection() {
        let mut monitor = HeartbeatMonitor::new(1.0, 3);
        monitor.register(NodeId(5), 0.0);

        let events = monitor.check(10.0, |n| n == NodeId(5));
        assert_eq!(events.len(), 1);
        match &events[0] {
            HeartbeatEvent::NodeDead { was_leader, .. } => assert!(was_leader),
            _ => panic!("expected NodeDead"),
        }
    }

    #[test]
    fn capability_rank_score() {
        let cap = DroneCapability {
            sensor_quality: 1.0,
            battery: 1.0,
            position_centrality: 1.0,
            experience: 1.0,
        };
        // 0.3 + 0.2 + 0.2 + 0.3 = 1.0
        assert!((cap.rank_score() - 1.0).abs() < 1e-10);

        let zero_cap = DroneCapability {
            sensor_quality: 0.0,
            battery: 0.0,
            position_centrality: 0.0,
            experience: 0.0,
        };
        assert!((zero_cap.rank_score() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn rank_manager_remove() {
        let mut rm = make_rank_manager(&[(0, 0.5)]);
        assert!((rm.rank(NodeId(0)) - 0.5).abs() < 1e-10);
        rm.remove(NodeId(0));
        assert!((rm.rank(NodeId(0)) - 0.0).abs() < 1e-10); // default
    }
}
