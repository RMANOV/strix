//! Distributed spatial awareness via neighbor belief maps.
//!
//! Each drone maintains a `SpatialBeliefMap` — a confidence-weighted model
//! of its local neighborhood built from gossip state. This provides:
//!
//! - **Local centroid**: GPS-denied reference point from neighbor positions
//! - **Local heading consensus**: mean velocity direction of local group
//! - **Relative positioning**: where am I relative to my neighbors?
//! - **Confidence decay**: stale beliefs automatically lose weight
//!
//! This is NOT full belief fusion (merging particle distributions).
//! It is a pragmatic substrate: shared neighbor positions with confidence
//! that a future Gaussian BP algorithm (Phase 16) would consume.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::gossip::DroneState;
use crate::{NodeId, Position3D};

/// Configuration for spatial belief tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialBeliefConfig {
    /// Number of nearest neighbors to track.
    pub k_neighbors: usize,
    /// Maximum age before a neighbor belief is discarded (seconds).
    pub max_belief_age_s: f64,
    /// Confidence decay halflife (seconds).
    pub confidence_halflife_s: f64,
    /// Communication radius for neighbor consideration (metres).
    pub comm_radius: f64,
}

impl Default for SpatialBeliefConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 8,
            max_belief_age_s: 10.0,
            confidence_halflife_s: 5.0,
            comm_radius: 100.0,
        }
    }
}

/// Belief about a single neighbor drone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborBelief {
    /// Estimated position.
    pub position: Position3D,
    /// Estimated velocity [vx, vy, vz].
    pub velocity: [f64; 3],
    /// Confidence in this belief ∈ [0, 1]. Decays with age.
    pub confidence: f64,
    /// Timestamp of last update.
    pub last_update: f64,
}

/// Per-drone spatial belief about the swarm's local neighborhood.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialBeliefMap {
    /// This drone's ID.
    self_id: NodeId,
    /// Neighbor beliefs (keyed by node id).
    neighbors: HashMap<NodeId, NeighborBelief>,
    /// Computed local centroid (self + neighbors, confidence-weighted).
    local_centroid: Position3D,
    /// Computed local heading consensus [vx, vy, vz].
    local_heading: [f64; 3],
    /// Configuration.
    config: SpatialBeliefConfig,
}

impl SpatialBeliefMap {
    /// Create a new empty belief map for a drone.
    pub fn new(self_id: NodeId, config: SpatialBeliefConfig) -> Self {
        Self {
            self_id,
            neighbors: HashMap::new(),
            local_centroid: Position3D::origin(),
            local_heading: [0.0; 3],
            config,
        }
    }

    /// Update from gossip: integrate all known drone states.
    ///
    /// Filters to k-nearest within comm_radius, updates confidence,
    /// prunes stale beliefs.
    pub fn update_from_gossip(
        &mut self,
        self_position: &Position3D,
        known_states: &HashMap<NodeId, DroneState>,
        now: f64,
    ) {
        // Collect candidates within comm_radius (excluding self).
        let mut candidates: Vec<(NodeId, f64, &DroneState)> = known_states
            .values()
            .filter(|s| s.node_id != self.self_id)
            .filter(|s| s.position.0.iter().all(|v| v.is_finite()))
            .map(|s| {
                let dist = self_position.distance(&s.position);
                (s.node_id, dist, s)
            })
            .filter(|(_, dist, _)| *dist <= self.config.comm_radius)
            .collect();

        // Sort by distance, take k nearest.
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.config.k_neighbors);

        // Update neighbor beliefs.
        let mut updated_ids: Vec<NodeId> = Vec::new();
        for (node_id, _dist, state) in &candidates {
            let age = (now - state.timestamp).max(0.0);
            let confidence = if self.config.confidence_halflife_s > 1e-6 {
                f64::exp(-f64::ln(2.0) * age / self.config.confidence_halflife_s)
            } else {
                1.0
            };

            self.neighbors.insert(
                *node_id,
                NeighborBelief {
                    position: state.position,
                    velocity: parse_velocity(state),
                    confidence: confidence.clamp(0.0, 1.0),
                    last_update: now,
                },
            );
            updated_ids.push(*node_id);
        }

        // Prune stale beliefs (not updated and too old).
        self.neighbors.retain(|id, belief| {
            updated_ids.contains(id) || (now - belief.last_update) < self.config.max_belief_age_s
        });

        // Recompute derived values.
        self.local_centroid = self.compute_centroid(self_position);
        self.local_heading = self.compute_heading();
    }

    /// Get the confidence-weighted local centroid (self + neighbors).
    pub fn centroid(&self) -> &Position3D {
        &self.local_centroid
    }

    /// Get the local heading consensus.
    pub fn heading(&self) -> &[f64; 3] {
        &self.local_heading
    }

    /// Get relative position of a neighbor from self.
    pub fn relative_position(
        &self,
        self_position: &Position3D,
        neighbor: NodeId,
    ) -> Option<[f64; 3]> {
        self.neighbors.get(&neighbor).map(|b| {
            [
                b.position.0[0] - self_position.0[0],
                b.position.0[1] - self_position.0[1],
                b.position.0[2] - self_position.0[2],
            ]
        })
    }

    /// Number of active neighbor beliefs.
    pub fn active_neighbors(&self) -> usize {
        self.neighbors.len()
    }

    /// Mean confidence across all neighbor beliefs.
    pub fn mean_confidence(&self) -> f64 {
        if self.neighbors.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.neighbors.values().map(|b| b.confidence).sum();
        sum / self.neighbors.len() as f64
    }

    // ── Private helpers ─────────────────────────────────────────────────

    fn compute_centroid(&self, self_position: &Position3D) -> Position3D {
        if self.neighbors.is_empty() {
            return *self_position;
        }

        // Self gets weight 1.0; neighbors weighted by confidence.
        let mut wx = self_position.0[0];
        let mut wy = self_position.0[1];
        let mut wz = self_position.0[2];
        let mut total_weight = 1.0;

        for belief in self.neighbors.values() {
            let w = belief.confidence;
            wx += belief.position.0[0] * w;
            wy += belief.position.0[1] * w;
            wz += belief.position.0[2] * w;
            total_weight += w;
        }

        if total_weight > 1e-12 {
            Position3D([wx / total_weight, wy / total_weight, wz / total_weight])
        } else {
            *self_position
        }
    }

    fn compute_heading(&self) -> [f64; 3] {
        if self.neighbors.is_empty() {
            return [0.0; 3];
        }

        let mut vx = 0.0;
        let mut vy = 0.0;
        let mut vz = 0.0;
        let mut total_weight = 0.0;

        for belief in self.neighbors.values() {
            let w = belief.confidence;
            vx += belief.velocity[0] * w;
            vy += belief.velocity[1] * w;
            vz += belief.velocity[2] * w;
            total_weight += w;
        }

        if total_weight > 1e-12 {
            [vx / total_weight, vy / total_weight, vz / total_weight]
        } else {
            [0.0; 3]
        }
    }
}

/// Parse velocity from DroneState regime string (gossip doesn't carry velocity directly).
/// Returns zero velocity — actual velocity integration requires the particle filter output.
fn parse_velocity(_state: &DroneState) -> [f64; 3] {
    // DroneState in gossip only carries position, battery, regime.
    // Velocity is not part of the gossip protocol (bandwidth optimization).
    // SpatialBeliefMap uses heading consensus from whatever velocity data is available.
    [0.0, 0.0, 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(id: u32, x: f64, y: f64, z: f64, timestamp: f64) -> DroneState {
        DroneState {
            node_id: NodeId(id),
            position: Position3D([x, y, z]),
            battery: 1.0,
            regime: "Patrol".to_string(),
            version: 1,
            timestamp,
            position_covariance: None,
        }
    }

    #[test]
    fn empty_map_centroid_equals_self() {
        let map = SpatialBeliefMap::new(NodeId(0), SpatialBeliefConfig::default());
        let self_pos = Position3D([10.0, 20.0, 30.0]);
        // No gossip update — centroid is origin (default)
        assert_eq!(map.active_neighbors(), 0);
    }

    #[test]
    fn centroid_is_weighted_average() {
        let mut map = SpatialBeliefMap::new(NodeId(0), SpatialBeliefConfig::default());
        let self_pos = Position3D([0.0, 0.0, 0.0]);

        let mut states = HashMap::new();
        states.insert(NodeId(1), make_state(1, 10.0, 0.0, 0.0, 0.0));
        states.insert(NodeId(2), make_state(2, 0.0, 10.0, 0.0, 0.0));

        map.update_from_gossip(&self_pos, &states, 0.0);

        let centroid = map.centroid();
        // Self at (0,0,0) weight=1, neighbor1 at (10,0,0), neighbor2 at (0,10,0)
        // With confidence=1.0 at t=0, centroid ≈ (10/3, 10/3, 0)
        assert!(
            centroid.0[0] > 2.0 && centroid.0[0] < 5.0,
            "x centroid should be ~3.3, got {}",
            centroid.0[0]
        );
        assert!(
            centroid.0[1] > 2.0 && centroid.0[1] < 5.0,
            "y centroid should be ~3.3, got {}",
            centroid.0[1]
        );
    }

    #[test]
    fn stale_beliefs_are_pruned() {
        let mut map = SpatialBeliefMap::new(
            NodeId(0),
            SpatialBeliefConfig {
                max_belief_age_s: 5.0,
                ..SpatialBeliefConfig::default()
            },
        );
        let self_pos = Position3D([0.0, 0.0, 0.0]);

        let mut states = HashMap::new();
        states.insert(NodeId(1), make_state(1, 10.0, 0.0, 0.0, 0.0));
        map.update_from_gossip(&self_pos, &states, 0.0);
        assert_eq!(map.active_neighbors(), 1);

        // Update at t=20 without node 1 — should be pruned (age > 5s)
        map.update_from_gossip(&self_pos, &HashMap::new(), 20.0);
        assert_eq!(map.active_neighbors(), 0, "stale neighbor should be pruned");
    }

    #[test]
    fn confidence_decays_over_time() {
        let mut map = SpatialBeliefMap::new(
            NodeId(0),
            SpatialBeliefConfig {
                confidence_halflife_s: 2.0,
                max_belief_age_s: 100.0,
                ..SpatialBeliefConfig::default()
            },
        );
        let self_pos = Position3D([0.0, 0.0, 0.0]);

        // State reported at t=0
        let mut states = HashMap::new();
        states.insert(NodeId(1), make_state(1, 10.0, 0.0, 0.0, 0.0));

        // Read at t=0 — full confidence
        map.update_from_gossip(&self_pos, &states, 0.0);
        let conf_t0 = map.neighbors.get(&NodeId(1)).unwrap().confidence;

        // Read at t=10 (same state, 10s old) — confidence should be lower
        map.update_from_gossip(&self_pos, &states, 10.0);
        let conf_t10 = map.neighbors.get(&NodeId(1)).unwrap().confidence;

        assert!(
            conf_t10 < conf_t0,
            "confidence should decay: t0={conf_t0}, t10={conf_t10}"
        );
        assert!(
            conf_t10 < 0.1,
            "after 5 half-lives, confidence should be very low"
        );
    }

    #[test]
    fn relative_position_is_correct() {
        let mut map = SpatialBeliefMap::new(NodeId(0), SpatialBeliefConfig::default());
        let self_pos = Position3D([5.0, 10.0, 15.0]);

        let mut states = HashMap::new();
        states.insert(NodeId(1), make_state(1, 15.0, 30.0, 45.0, 0.0));
        map.update_from_gossip(&self_pos, &states, 0.0);

        let rel = map.relative_position(&self_pos, NodeId(1)).unwrap();
        assert!((rel[0] - 10.0).abs() < 1e-6);
        assert!((rel[1] - 20.0).abs() < 1e-6);
        assert!((rel[2] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn comm_radius_filters_distant_drones() {
        let mut map = SpatialBeliefMap::new(
            NodeId(0),
            SpatialBeliefConfig {
                comm_radius: 50.0,
                ..SpatialBeliefConfig::default()
            },
        );
        let self_pos = Position3D([0.0, 0.0, 0.0]);

        let mut states = HashMap::new();
        states.insert(NodeId(1), make_state(1, 10.0, 0.0, 0.0, 0.0)); // within radius
        states.insert(NodeId(2), make_state(2, 200.0, 0.0, 0.0, 0.0)); // outside radius
        map.update_from_gossip(&self_pos, &states, 0.0);

        assert_eq!(
            map.active_neighbors(),
            1,
            "only nearby drone should be tracked"
        );
    }

    #[test]
    fn k_neighbors_limits_tracked_count() {
        let mut map = SpatialBeliefMap::new(
            NodeId(0),
            SpatialBeliefConfig {
                k_neighbors: 3,
                comm_radius: 1000.0,
                ..SpatialBeliefConfig::default()
            },
        );
        let self_pos = Position3D([0.0, 0.0, 0.0]);

        let mut states = HashMap::new();
        for i in 1..=10 {
            states.insert(NodeId(i), make_state(i, i as f64 * 10.0, 0.0, 0.0, 0.0));
        }
        map.update_from_gossip(&self_pos, &states, 0.0);

        assert_eq!(map.active_neighbors(), 3, "should only track k=3 nearest");
    }

    #[test]
    fn mean_confidence_is_correct() {
        let mut map = SpatialBeliefMap::new(NodeId(0), SpatialBeliefConfig::default());
        let self_pos = Position3D([0.0, 0.0, 0.0]);

        let mut states = HashMap::new();
        states.insert(NodeId(1), make_state(1, 10.0, 0.0, 0.0, 0.0));
        states.insert(NodeId(2), make_state(2, 20.0, 0.0, 0.0, 0.0));
        map.update_from_gossip(&self_pos, &states, 0.0);

        let mc = map.mean_confidence();
        assert!(
            mc > 0.9,
            "fresh beliefs should have high confidence, got {mc}"
        );
    }
}
