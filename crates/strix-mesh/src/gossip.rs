//! # Gossip Protocol for State Synchronisation
//!
//! O(log N) convergence distributed state sync via epidemic gossip:
//!
//! 1. Each node periodically selects `fanout` random peers.
//! 2. Sends its state digest (compact hash).
//! 3. If peer has different state → exchange updates.
//!
//! Conflict resolution:
//! - **General data**: newer timestamp wins.
//! - **Threat data**: union — never discard threat information.
//!
//! Anti-entropy: periodic full state exchange with one random peer.

use rand::prelude::IteratorRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{NodeId, Position3D};

fn position_is_finite(position: &Position3D) -> bool {
    position.0.iter().all(|value| value.is_finite())
}

fn finite_timestamp(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn finite_fraction(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn sanitize_drone_state(state: &DroneState) -> Option<DroneState> {
    position_is_finite(&state.position).then(|| DroneState {
        node_id: state.node_id,
        position: state.position,
        battery: finite_fraction(state.battery),
        regime: state.regime.clone(),
        version: state.version,
        timestamp: finite_timestamp(state.timestamp),
    })
}

fn sanitize_threat(record: &ThreatRecord) -> Option<ThreatRecord> {
    position_is_finite(&record.position).then(|| ThreatRecord {
        threat_id: record.threat_id,
        reporter: record.reporter,
        position: record.position,
        threat_level: finite_fraction(record.threat_level),
        timestamp: finite_timestamp(record.timestamp),
        resolved: record.resolved,
    })
}

fn sort_drone_states(drone_states: &mut [DroneState]) {
    drone_states.sort_by(|a, b| {
        a.node_id
            .cmp(&b.node_id)
            .then_with(|| a.version.cmp(&b.version))
            .then_with(|| a.timestamp.total_cmp(&b.timestamp))
    });
}

fn sort_threats(threats: &mut [ThreatRecord]) {
    threats.sort_by(|a, b| {
        a.threat_id
            .cmp(&b.threat_id)
            .then_with(|| a.timestamp.total_cmp(&b.timestamp))
    });
}

// ---------------------------------------------------------------------------
// State snapshot — the thing we gossip about
// ---------------------------------------------------------------------------

/// Per-drone state as exchanged through gossip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneState {
    /// Which drone this state describes.
    pub node_id: NodeId,
    /// Last known position.
    pub position: Position3D,
    /// Battery level (0.0–1.0).
    pub battery: f64,
    /// Current operational regime (e.g. "search", "track").
    pub regime: String,
    /// Monotonically increasing version (lamport-ish).
    pub version: u64,
    /// Wall-clock timestamp of last update.
    pub timestamp: f64,
}

/// Convert a strix-core DroneState into the gossip DroneState.
///
/// Maps core fields to gossip's network-oriented representation.
/// Fields not present in core (version, timestamp) get sensible defaults.
impl From<&strix_core::DroneState> for DroneState {
    fn from(core: &strix_core::DroneState) -> Self {
        DroneState {
            node_id: NodeId(core.drone_id),
            position: Position3D([core.position.x, core.position.y, core.position.z]),
            battery: 1.0, // not tracked in core, default full
            regime: format!("{:?}", core.regime),
            version: 0,
            timestamp: 0.0,
        }
    }
}

/// A threat record — never discarded, only updated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatRecord {
    /// Unique threat identifier.
    pub threat_id: u64,
    /// Who first reported it.
    pub reporter: NodeId,
    /// Position of the threat.
    pub position: Position3D,
    /// Severity (0.0–1.0).
    pub threat_level: f64,
    /// When it was reported.
    pub timestamp: f64,
    /// Whether it has been neutralised / cleared.
    pub resolved: bool,
}

// ---------------------------------------------------------------------------
// Gossip message
// ---------------------------------------------------------------------------

/// Compact message exchanged during a gossip round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GossipMessage {
    /// Digest: compact summary of our state — the peer can decide whether
    /// a full sync is needed.
    Digest {
        sender: NodeId,
        /// version per known drone.
        versions: BTreeMap<NodeId, u64>,
        /// number of threat records we hold.
        threat_count: usize,
    },
    /// Full state update (sent when digests differ).
    StateExchange {
        sender: NodeId,
        /// Drone states that the receiver is missing or has stale.
        drone_states: Vec<DroneState>,
        /// Threat records (union semantics).
        threats: Vec<ThreatRecord>,
    },
}

// ---------------------------------------------------------------------------
// Gossip engine
// ---------------------------------------------------------------------------

/// Manages state dissemination for one node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipEngine {
    /// This node's identity.
    pub self_id: NodeId,
    /// Known drone states (keyed by node id).
    pub known_states: HashMap<NodeId, DroneState>,
    /// Known threats (keyed by threat id, union semantics).
    pub known_threats: HashMap<u64, ThreatRecord>,
    /// Known peer ids for fan-out selection.
    pub peers: HashSet<NodeId>,
    /// Number of peers to contact per round.
    pub fanout: usize,
}

impl GossipEngine {
    /// Create a new gossip engine.
    pub fn new(self_id: NodeId, fanout: usize) -> Self {
        Self {
            self_id,
            known_states: HashMap::new(),
            known_threats: HashMap::new(),
            peers: HashSet::new(),
            fanout,
        }
    }

    /// Dynamically adjust the gossip fanout (e.g., for EW degradation).
    pub fn set_fanout(&mut self, fanout: usize) {
        self.fanout = fanout.max(1);
    }

    /// Register a peer.
    pub fn add_peer(&mut self, peer: NodeId) {
        if peer != self.self_id {
            self.peers.insert(peer);
        }
    }

    /// Remove a peer (e.g. declared dead).
    pub fn remove_peer(&mut self, peer: NodeId) {
        self.peers.remove(&peer);
    }

    /// Update our own state.
    pub fn update_self_state(
        &mut self,
        position: Position3D,
        battery: f64,
        regime: String,
        timestamp: f64,
    ) {
        if !position_is_finite(&position) {
            return;
        }

        let version = self
            .known_states
            .get(&self.self_id)
            .map_or(1, |s| s.version + 1);
        self.known_states.insert(
            self.self_id,
            DroneState {
                node_id: self.self_id,
                position,
                battery: finite_fraction(battery),
                regime,
                version,
                timestamp: finite_timestamp(timestamp),
            },
        );
    }

    /// Report a threat (local event).
    pub fn report_threat(
        &mut self,
        threat_id: u64,
        position: Position3D,
        threat_level: f64,
        timestamp: f64,
    ) {
        if !position_is_finite(&position) {
            return;
        }

        self.known_threats.insert(
            threat_id,
            ThreatRecord {
                threat_id,
                reporter: self.self_id,
                position,
                threat_level: finite_fraction(threat_level),
                timestamp: finite_timestamp(timestamp),
                resolved: false,
            },
        );
    }

    /// Prune stale threats older than `max_age` relative to `now`.
    /// Also removes resolved threats. Call periodically to prevent unbounded growth.
    pub fn prune_threats(&mut self, now: f64, max_age: f64) {
        if !now.is_finite() || !max_age.is_finite() || max_age < 0.0 {
            return;
        }

        self.known_threats
            .retain(|_, t| !t.resolved && (now - finite_timestamp(t.timestamp)) < max_age);
    }

    // -----------------------------------------------------------------------
    // Gossip round
    // -----------------------------------------------------------------------

    /// Select `fanout` random peers for this gossip round.
    pub fn select_peers<R: Rng>(&self, rng: &mut R) -> Vec<NodeId> {
        let mut peers: Vec<NodeId> = self.peers.iter().copied().collect();
        peers.sort_unstable();
        let mut selected = peers.into_iter().choose_multiple(rng, self.fanout);
        selected.sort_unstable();
        selected
    }

    /// Build a digest message for sending to a peer.
    pub fn build_digest(&self) -> GossipMessage {
        let versions: BTreeMap<NodeId, u64> = self
            .known_states
            .values()
            .filter_map(sanitize_drone_state)
            .map(|state| (state.node_id, state.version))
            .collect();
        GossipMessage::Digest {
            sender: self.self_id,
            versions,
            threat_count: self
                .known_threats
                .values()
                .filter_map(sanitize_threat)
                .count(),
        }
    }

    /// Given a received digest, determine which states to send back and
    /// produce a `StateExchange` message.
    pub fn respond_to_digest(&self, digest: &GossipMessage) -> Option<GossipMessage> {
        let (peer_versions, peer_threat_count) = match digest {
            GossipMessage::Digest {
                versions,
                threat_count,
                ..
            } => (versions, *threat_count),
            _ => return None,
        };

        // Find drone states we have that are newer or missing at peer.
        let drone_states: Vec<DroneState> = self
            .known_states
            .values()
            .filter_map(sanitize_drone_state)
            .filter(|state| {
                peer_versions
                    .get(&state.node_id)
                    .is_none_or(|&pv| state.version > pv)
            })
            .collect();

        // If peer has fewer threats, send all (union semantics — safe to resend).
        let local_threat_count = self
            .known_threats
            .values()
            .filter_map(sanitize_threat)
            .count();
        let threats: Vec<ThreatRecord> = if peer_threat_count < local_threat_count {
            self.known_threats
                .values()
                .filter_map(sanitize_threat)
                .collect()
        } else {
            Vec::new()
        };

        let mut drone_states = drone_states;
        let mut threats = threats;
        sort_drone_states(&mut drone_states);
        sort_threats(&mut threats);

        if drone_states.is_empty() && threats.is_empty() {
            return None; // nothing to send
        }

        Some(GossipMessage::StateExchange {
            sender: self.self_id,
            drone_states,
            threats,
        })
    }

    /// Merge incoming state into our local knowledge.
    ///
    /// - Drone states: newer version wins.
    /// - Threats: union (insert if missing, update if newer timestamp).
    pub fn merge_state(&mut self, msg: &GossipMessage) {
        let (drone_states, threats) = match msg {
            GossipMessage::StateExchange {
                drone_states,
                threats,
                ..
            } => (drone_states, threats),
            _ => return,
        };

        // Merge drone states — newer version wins.
        for incoming in drone_states {
            let Some(incoming) = sanitize_drone_state(incoming) else {
                continue;
            };
            let dominated = self
                .known_states
                .get(&incoming.node_id)
                .is_none_or(|existing| {
                    sanitize_drone_state(existing).is_none() || incoming.version > existing.version
                });
            if dominated {
                self.known_states.insert(incoming.node_id, incoming);
            }
        }

        // Merge threats — union, newer timestamp wins per threat_id.
        for incoming in threats {
            let Some(incoming) = sanitize_threat(incoming) else {
                continue;
            };
            let should_insert =
                self.known_threats
                    .get(&incoming.threat_id)
                    .is_none_or(|existing| {
                        sanitize_threat(existing)
                            .is_none_or(|existing| incoming.timestamp > existing.timestamp)
                    });
            if should_insert {
                self.known_threats.insert(incoming.threat_id, incoming);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Convergence
    // -----------------------------------------------------------------------

    /// Estimate what fraction of peers have converged with us.
    ///
    /// This is a local estimate: we count how many peers we've received
    /// state from (indicating they're actively gossiping). A more accurate
    /// measure requires exchanging digests and comparing.
    pub fn convergence_estimate(&self) -> f64 {
        if self.peers.is_empty() {
            return 1.0;
        }
        let heard_from = self
            .peers
            .iter()
            .filter(|p| self.known_states.contains_key(p))
            .count();
        heard_from as f64 / self.peers.len() as f64
    }

    // -----------------------------------------------------------------------
    // Anti-entropy
    // -----------------------------------------------------------------------

    /// Build a full state exchange for anti-entropy with a single random peer.
    /// This sends *everything* — used periodically to guarantee convergence
    /// even if some gossip rounds were lost.
    pub fn build_anti_entropy(&self) -> GossipMessage {
        let mut drone_states: Vec<DroneState> = self
            .known_states
            .values()
            .filter_map(sanitize_drone_state)
            .collect();
        let mut threats: Vec<ThreatRecord> = self
            .known_threats
            .values()
            .filter_map(sanitize_threat)
            .collect();
        sort_drone_states(&mut drone_states);
        sort_threats(&mut threats);

        GossipMessage::StateExchange {
            sender: self.self_id,
            drone_states,
            threats,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_engine(id: u32, peers: &[u32]) -> GossipEngine {
        let mut engine = GossipEngine::new(NodeId(id), 2);
        for &p in peers {
            engine.add_peer(NodeId(p));
        }
        engine
    }

    #[test]
    fn basic_gossip_round() {
        let mut a = make_engine(0, &[1, 2]);
        let mut b = make_engine(1, &[0, 2]);

        a.update_self_state(Position3D([1.0, 2.0, 3.0]), 0.9, "search".into(), 1.0);
        b.update_self_state(Position3D([4.0, 5.0, 6.0]), 0.8, "track".into(), 1.0);

        // A sends digest to B.
        let digest = a.build_digest();
        // B responds with what A is missing.
        let response = b.respond_to_digest(&digest);
        assert!(response.is_some());
        // A merges.
        a.merge_state(&response.unwrap());
        assert!(a.known_states.contains_key(&NodeId(1)));
    }

    #[test]
    fn newer_version_wins() {
        let mut engine = make_engine(0, &[1]);

        let old = DroneState {
            node_id: NodeId(1),
            position: Position3D([0.0, 0.0, 0.0]),
            battery: 0.5,
            regime: "old".into(),
            version: 1,
            timestamp: 1.0,
        };
        let new = DroneState {
            node_id: NodeId(1),
            position: Position3D([1.0, 1.0, 1.0]),
            battery: 0.9,
            regime: "new".into(),
            version: 5,
            timestamp: 5.0,
        };

        // Insert old first.
        engine.merge_state(&GossipMessage::StateExchange {
            sender: NodeId(1),
            drone_states: vec![old],
            threats: vec![],
        });
        assert_eq!(engine.known_states[&NodeId(1)].version, 1);

        // Now merge newer.
        engine.merge_state(&GossipMessage::StateExchange {
            sender: NodeId(1),
            drone_states: vec![new],
            threats: vec![],
        });
        assert_eq!(engine.known_states[&NodeId(1)].version, 5);
        assert_eq!(engine.known_states[&NodeId(1)].regime, "new");
    }

    #[test]
    fn threat_union_semantics() {
        let mut a = make_engine(0, &[1]);
        let mut b = make_engine(1, &[0]);

        a.report_threat(100, Position3D([10.0, 10.0, 0.0]), 0.9, 1.0);
        b.report_threat(200, Position3D([20.0, 20.0, 0.0]), 0.5, 2.0);

        // Exchange.
        let a_full = a.build_anti_entropy();
        let b_full = b.build_anti_entropy();
        b.merge_state(&a_full);
        a.merge_state(&b_full);

        // Both should know both threats.
        assert!(a.known_threats.contains_key(&100));
        assert!(a.known_threats.contains_key(&200));
        assert!(b.known_threats.contains_key(&100));
        assert!(b.known_threats.contains_key(&200));
    }

    #[test]
    fn update_self_state_sanitizes_numeric_fields() {
        let mut engine = make_engine(0, &[]);
        engine.update_self_state(Position3D([1.0, 2.0, 3.0]), 2.0, "search".into(), f64::NAN);

        let state = engine.known_states.get(&NodeId(0)).unwrap();
        assert!((state.battery - 1.0).abs() < 1e-10);
        assert!((state.timestamp - 0.0).abs() < 1e-10);
    }

    #[test]
    fn report_threat_ignores_non_finite_position() {
        let mut engine = make_engine(0, &[]);
        engine.report_threat(7, Position3D([f64::NAN, 0.0, 0.0]), 0.8, 1.0);
        assert!(!engine.known_threats.contains_key(&7));
    }

    #[test]
    fn merge_state_replaces_non_finite_threat_timestamp() {
        let mut engine = make_engine(0, &[]);
        engine.known_threats.insert(
            5,
            ThreatRecord {
                threat_id: 5,
                reporter: NodeId(1),
                position: Position3D([0.0, 0.0, 0.0]),
                threat_level: 0.5,
                timestamp: f64::NAN,
                resolved: false,
            },
        );

        engine.merge_state(&GossipMessage::StateExchange {
            sender: NodeId(2),
            drone_states: vec![],
            threats: vec![ThreatRecord {
                threat_id: 5,
                reporter: NodeId(2),
                position: Position3D([1.0, 1.0, 0.0]),
                threat_level: 0.9,
                timestamp: 1.0,
                resolved: false,
            }],
        });

        let threat = engine.known_threats.get(&5).unwrap();
        assert!((threat.timestamp - 1.0).abs() < 1e-10);
        assert_eq!(threat.reporter, NodeId(2));
    }

    #[test]
    fn merge_state_replaces_invalid_existing_drone_state() {
        let mut engine = make_engine(0, &[]);
        engine.known_states.insert(
            NodeId(3),
            DroneState {
                node_id: NodeId(3),
                position: Position3D([f64::NAN, 0.0, 0.0]),
                battery: 0.5,
                regime: "bad".into(),
                version: 5,
                timestamp: 0.0,
            },
        );

        engine.merge_state(&GossipMessage::StateExchange {
            sender: NodeId(3),
            drone_states: vec![DroneState {
                node_id: NodeId(3),
                position: Position3D([1.0, 2.0, 3.0]),
                battery: 0.9,
                regime: "good".into(),
                version: 5,
                timestamp: 1.0,
            }],
            threats: vec![],
        });

        let state = engine.known_states.get(&NodeId(3)).unwrap();
        assert_eq!(state.position, Position3D([1.0, 2.0, 3.0]));
        assert_eq!(state.regime, "good");
    }

    #[test]
    fn convergence_estimate() {
        let mut engine = make_engine(0, &[1, 2, 3]);
        assert!((engine.convergence_estimate() - 0.0).abs() < 1e-10);

        // Simulate hearing from peer 1.
        engine.known_states.insert(
            NodeId(1),
            DroneState {
                node_id: NodeId(1),
                position: Position3D::origin(),
                battery: 1.0,
                regime: "idle".into(),
                version: 1,
                timestamp: 0.0,
            },
        );
        // 1 of 3 peers heard.
        assert!((engine.convergence_estimate() - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn select_peers_respects_fanout() {
        let engine = make_engine(0, &[1, 2, 3, 4, 5]);
        let mut rng = rand::thread_rng();
        let selected = engine.select_peers(&mut rng);
        assert!(selected.len() <= 2); // fanout = 2
    }

    #[test]
    fn select_peers_is_deterministic_for_same_seed() {
        let a = make_engine(0, &[5, 3, 1, 4, 2]);
        let b = make_engine(0, &[1, 2, 3, 4, 5]);
        let mut rng_a = StdRng::seed_from_u64(7);
        let mut rng_b = StdRng::seed_from_u64(7);

        assert_eq!(a.select_peers(&mut rng_a), b.select_peers(&mut rng_b));
    }

    #[test]
    fn digest_and_state_exchange_are_sorted() {
        let mut engine = make_engine(0, &[1, 2, 3]);
        engine.known_states.insert(
            NodeId(3),
            DroneState {
                node_id: NodeId(3),
                position: Position3D([3.0, 0.0, 0.0]),
                battery: 1.0,
                regime: "c".into(),
                version: 3,
                timestamp: 3.0,
            },
        );
        engine.known_states.insert(
            NodeId(1),
            DroneState {
                node_id: NodeId(1),
                position: Position3D([1.0, 0.0, 0.0]),
                battery: 1.0,
                regime: "a".into(),
                version: 1,
                timestamp: 1.0,
            },
        );
        engine.known_states.insert(
            NodeId(2),
            DroneState {
                node_id: NodeId(2),
                position: Position3D([2.0, 0.0, 0.0]),
                battery: 1.0,
                regime: "b".into(),
                version: 2,
                timestamp: 2.0,
            },
        );
        engine.known_threats.insert(
            7,
            ThreatRecord {
                threat_id: 7,
                reporter: NodeId(0),
                position: Position3D([0.0, 7.0, 0.0]),
                threat_level: 0.7,
                timestamp: 7.0,
                resolved: false,
            },
        );
        engine.known_threats.insert(
            5,
            ThreatRecord {
                threat_id: 5,
                reporter: NodeId(0),
                position: Position3D([0.0, 5.0, 0.0]),
                threat_level: 0.5,
                timestamp: 5.0,
                resolved: false,
            },
        );

        let digest = engine.build_digest();
        let version_ids = match digest {
            GossipMessage::Digest { versions, .. } => versions
                .into_iter()
                .map(|(node_id, _)| node_id)
                .collect::<Vec<_>>(),
            _ => unreachable!(),
        };
        assert_eq!(version_ids, vec![NodeId(1), NodeId(2), NodeId(3)]);

        let response = engine
            .respond_to_digest(&GossipMessage::Digest {
                sender: NodeId(9),
                versions: BTreeMap::new(),
                threat_count: 0,
            })
            .unwrap();

        match response {
            GossipMessage::StateExchange {
                drone_states,
                threats,
                ..
            } => {
                let drone_ids: Vec<NodeId> =
                    drone_states.iter().map(|state| state.node_id).collect();
                let threat_ids: Vec<u64> = threats.iter().map(|threat| threat.threat_id).collect();
                assert_eq!(drone_ids, vec![NodeId(1), NodeId(2), NodeId(3)]);
                assert_eq!(threat_ids, vec![5, 7]);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn add_remove_peer() {
        let mut engine = make_engine(0, &[]);
        assert!(engine.peers.is_empty());
        engine.add_peer(NodeId(1));
        assert_eq!(engine.peers.len(), 1);
        engine.add_peer(NodeId(1)); // duplicate
        assert_eq!(engine.peers.len(), 1);
        engine.add_peer(NodeId(0)); // self — ignored
        assert_eq!(engine.peers.len(), 1);
        engine.remove_peer(NodeId(1));
        assert!(engine.peers.is_empty());
    }

    #[test]
    fn multi_hop_convergence() {
        // A -- B -- C  (A and C don't know each other directly)
        let mut a = make_engine(0, &[1]);
        let mut b = make_engine(1, &[0, 2]);
        let mut c = make_engine(2, &[1]);

        a.update_self_state(Position3D([0.0, 0.0, 0.0]), 1.0, "alpha".into(), 1.0);
        c.update_self_state(Position3D([100.0, 0.0, 0.0]), 0.8, "gamma".into(), 1.0);

        // Round 1: A→B, C→B
        let ae = a.build_anti_entropy();
        let ce = c.build_anti_entropy();
        b.merge_state(&ae);
        b.merge_state(&ce);

        // B now knows A and C.
        assert!(b.known_states.contains_key(&NodeId(0)));
        assert!(b.known_states.contains_key(&NodeId(2)));

        // Round 2: B→A, B→C
        let be = b.build_anti_entropy();
        a.merge_state(&be);
        c.merge_state(&be);

        // Now A knows C and vice versa (through B).
        assert!(a.known_states.contains_key(&NodeId(2)));
        assert!(c.known_states.contains_key(&NodeId(0)));
    }
}
