//! Belief compositor — merges facts into coherent world state.
//!
//! Multiple FactEnvelopes about the same entity are resolved by
//! recency, confidence, and causal ordering.

use std::collections::{HashMap, HashSet};

use crate::NodeId;

use super::fact::{DroneObservation, FactId, MeshFact, ThreatFactPayload};

/// Composed belief about a single drone.
#[derive(Debug, Clone)]
pub struct DroneBelief {
    /// Latest observation.
    pub observation: DroneObservation,
    /// Effective confidence (age-decayed).
    pub confidence: f64,
    /// Source fact ID.
    pub source: FactId,
    /// Number of corroborating observations.
    pub corroboration_count: usize,
}

/// Composed belief about a single threat.
#[derive(Debug, Clone)]
pub struct ThreatBelief {
    /// Latest observation.
    pub observation: ThreatFactPayload,
    /// Effective confidence.
    pub confidence: f64,
    /// Source fact ID.
    pub source: FactId,
    /// Whether retracted.
    pub retracted: bool,
}

/// The belief compositor maintains a world view from facts.
pub struct BeliefCompositor {
    /// All received facts, keyed by FactId.
    facts: HashMap<FactId, MeshFact>,
    /// Retracted fact IDs.
    retracted: HashSet<FactId>,
    /// Next sequence number for facts we originate.
    next_seq: u64,
    /// Our node ID.
    self_id: NodeId,
}

impl BeliefCompositor {
    /// Create a new compositor for the given node.
    pub fn new(self_id: NodeId) -> Self {
        Self {
            facts: HashMap::new(),
            retracted: HashSet::new(),
            next_seq: 1,
            self_id,
        }
    }

    /// Ingest a fact. Returns true if the fact was new.
    pub fn ingest(&mut self, fact: MeshFact) -> bool {
        let id = fact.id();

        // Handle retractions: register the retracted ID regardless.
        if let MeshFact::Retraction(ref r) = fact {
            self.retracted.insert(r.payload.retracted_id);
        }

        // Don't accept if already retracted.
        if self.retracted.contains(&id) {
            return false;
        }

        // Insert if new or higher sequence from the same origin.
        if let Some(existing) = self.facts.get(&id) {
            if existing.stamp().sequence >= fact.stamp().sequence {
                return false; // already have this or newer
            }
        }

        self.facts.insert(id, fact);
        true
    }

    /// Generate a fact ID for a new fact we're creating.
    pub fn next_fact_id(&mut self) -> FactId {
        let id = FactId {
            origin: self.self_id,
            seq: self.next_seq,
        };
        self.next_seq += 1;
        id
    }

    /// Evict expired facts.
    pub fn evict_expired(&mut self, now: f64) {
        self.facts.retain(|_, f| !f.is_expired(now));
    }

    /// Compose drone beliefs from all non-retracted drone observations.
    pub fn drone_beliefs(&self, now: f64) -> HashMap<NodeId, DroneBelief> {
        let mut beliefs: HashMap<NodeId, DroneBelief> = HashMap::with_capacity(self.facts.len());

        for fact in self.facts.values() {
            if let MeshFact::Drone(ref envelope) = fact {
                if self.retracted.contains(&fact.id()) {
                    continue;
                }
                let drone_id = envelope.payload.drone_id;
                let eff_conf = envelope.effective_confidence(now);

                match beliefs.get_mut(&drone_id) {
                    Some(existing) => {
                        // Keep the one with higher effective confidence.
                        existing.corroboration_count += 1;
                        if eff_conf > existing.confidence {
                            existing.observation = envelope.payload.clone();
                            existing.confidence = eff_conf;
                            existing.source = envelope.id;
                        }
                    }
                    None => {
                        beliefs.insert(
                            drone_id,
                            DroneBelief {
                                observation: envelope.payload.clone(),
                                confidence: eff_conf,
                                source: envelope.id,
                                corroboration_count: 1,
                            },
                        );
                    }
                }
            }
        }

        beliefs
    }

    /// Compose threat beliefs.
    pub fn threat_beliefs(&self, now: f64) -> HashMap<u64, ThreatBelief> {
        let mut beliefs: HashMap<u64, ThreatBelief> = HashMap::with_capacity(self.facts.len());

        for fact in self.facts.values() {
            let is_retracted = self.retracted.contains(&fact.id());
            if let MeshFact::Threat(ref envelope) = fact {
                let threat_id = envelope.payload.threat_id;
                let eff_conf = envelope.effective_confidence(now);

                match beliefs.get_mut(&threat_id) {
                    Some(existing) => {
                        if eff_conf > existing.confidence && !is_retracted {
                            existing.observation = envelope.payload.clone();
                            existing.confidence = eff_conf;
                            existing.source = envelope.id;
                        }
                        if is_retracted {
                            existing.retracted = true;
                        }
                    }
                    None => {
                        beliefs.insert(
                            threat_id,
                            ThreatBelief {
                                observation: envelope.payload.clone(),
                                confidence: eff_conf,
                                source: envelope.id,
                                retracted: is_retracted,
                            },
                        );
                    }
                }
            }
        }

        beliefs
    }

    /// Total number of facts stored.
    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    /// Number of retracted fact IDs tracked.
    pub fn retraction_count(&self) -> usize {
        self.retracted.len()
    }

    /// Iterator over all stored facts (for replication/sync).
    pub fn all_facts(&self) -> impl Iterator<Item = &MeshFact> {
        self.facts.values()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fact::{
        CausalStamp, DroneObservation, FactEnvelope, FactKind, FactRetraction, ThreatFactPayload,
    };
    use crate::Position3D;

    fn make_stamp(originator: NodeId, seq: u64, observed_at: f64) -> CausalStamp {
        CausalStamp {
            originator,
            sequence: seq,
            observed_at,
            received_at: observed_at,
            hop_count: 0,
        }
    }

    fn drone_fact(
        origin: NodeId,
        seq: u64,
        drone_id: NodeId,
        observed_at: f64,
        ttl_s: f64,
        confidence: f64,
        pos: [f64; 3],
    ) -> MeshFact {
        MeshFact::Drone(FactEnvelope {
            id: FactId { origin, seq },
            kind: FactKind::Observation,
            payload: DroneObservation {
                drone_id,
                position: Position3D(pos),
                battery: 0.8,
                regime: "search".to_string(),
            },
            stamp: make_stamp(origin, seq, observed_at),
            ttl_s,
            confidence,
        })
    }

    fn threat_fact(
        origin: NodeId,
        seq: u64,
        threat_id: u64,
        observed_at: f64,
        ttl_s: f64,
        confidence: f64,
    ) -> MeshFact {
        MeshFact::Threat(FactEnvelope {
            id: FactId { origin, seq },
            kind: FactKind::Observation,
            payload: ThreatFactPayload {
                threat_id,
                position: Position3D([10.0, 20.0, 0.0]),
                threat_level: 0.7,
                resolved: false,
            },
            stamp: make_stamp(origin, seq, observed_at),
            ttl_s,
            confidence,
        })
    }

    fn retraction_fact(
        origin: NodeId,
        seq: u64,
        observed_at: f64,
        ttl_s: f64,
        retracted_id: FactId,
    ) -> MeshFact {
        MeshFact::Retraction(FactEnvelope {
            id: FactId { origin, seq },
            kind: FactKind::Retraction,
            payload: FactRetraction {
                retracted_id,
                reason: "retracted".to_string(),
            },
            stamp: make_stamp(origin, seq, observed_at),
            ttl_s,
            confidence: 1.0,
        })
    }

    #[test]
    fn ingest_new_fact() {
        let mut comp = BeliefCompositor::new(NodeId(1));
        let fact = drone_fact(NodeId(2), 1, NodeId(3), 0.0, 30.0, 0.9, [1.0, 2.0, 3.0]);

        // First ingest → new
        assert!(comp.ingest(fact.clone()), "first ingest should return true");
        assert_eq!(comp.fact_count(), 1);

        // Same fact again → duplicate (same id and sequence)
        assert!(!comp.ingest(fact), "duplicate ingest should return false");
        assert_eq!(comp.fact_count(), 1);
    }

    #[test]
    fn retraction_prevents_belief() {
        let mut comp = BeliefCompositor::new(NodeId(1));
        let origin = NodeId(2);
        let drone_id = NodeId(5);

        // Add a drone fact
        let fact = drone_fact(origin, 1, drone_id, 0.0, 30.0, 0.9, [1.0, 2.0, 3.0]);
        let fact_id = fact.id();
        comp.ingest(fact);

        // Beliefs present before retraction
        let beliefs_before = comp.drone_beliefs(0.0);
        assert!(
            beliefs_before.contains_key(&drone_id),
            "drone should be in beliefs before retraction"
        );

        // Retract the fact
        let ret = retraction_fact(origin, 2, 0.0, 120.0, fact_id);
        comp.ingest(ret);

        // Fact is retracted → should not appear in beliefs
        let beliefs_after = comp.drone_beliefs(0.0);
        assert!(
            !beliefs_after.contains_key(&drone_id),
            "drone should NOT be in beliefs after retraction"
        );
    }

    #[test]
    fn evict_expired_facts() {
        let mut comp = BeliefCompositor::new(NodeId(1));

        // TTL=5s, observed at t=0 → expires after t=5
        let fact1 = drone_fact(NodeId(2), 1, NodeId(10), 0.0, 5.0, 0.9, [0.0, 0.0, 0.0]);
        // TTL=100s, observed at t=0 → still alive at t=10
        let fact2 = drone_fact(NodeId(3), 1, NodeId(11), 0.0, 100.0, 0.9, [0.0, 0.0, 0.0]);

        comp.ingest(fact1);
        comp.ingest(fact2);
        assert_eq!(comp.fact_count(), 2);

        // Evict at t=10 — fact1 (TTL=5) should be gone, fact2 (TTL=100) remains
        comp.evict_expired(10.0);
        assert_eq!(comp.fact_count(), 1, "only long-lived fact should remain");

        let beliefs = comp.drone_beliefs(10.0);
        assert!(
            !beliefs.contains_key(&NodeId(10)),
            "evicted drone should not appear in beliefs"
        );
        assert!(
            beliefs.contains_key(&NodeId(11)),
            "surviving drone should appear in beliefs"
        );
    }

    #[test]
    fn drone_beliefs_picks_highest_confidence() {
        let mut comp = BeliefCompositor::new(NodeId(1));
        let drone_id = NodeId(42);

        // Two observations of the same drone from different nodes at t=0
        // High confidence from node 2
        let high_conf = drone_fact(NodeId(2), 1, drone_id, 0.0, 30.0, 0.9, [1.0, 0.0, 0.0]);
        // Low confidence from node 3
        let low_conf = drone_fact(NodeId(3), 1, drone_id, 0.0, 30.0, 0.3, [2.0, 0.0, 0.0]);

        comp.ingest(high_conf);
        comp.ingest(low_conf);

        let beliefs = comp.drone_beliefs(0.0);
        let belief = beliefs.get(&drone_id).expect("drone belief should exist");

        // The high-confidence source should win
        assert!(
            belief.confidence >= 0.9 - 1e-10,
            "expected high confidence ~0.9, got {}",
            belief.confidence
        );
        assert_eq!(
            belief.source,
            FactId {
                origin: NodeId(2),
                seq: 1
            },
            "source should be the high-confidence fact"
        );
        // Both observations count toward corroboration
        assert_eq!(
            belief.corroboration_count, 2,
            "both observations should corroborate"
        );
    }

    #[test]
    fn threat_beliefs_marks_retracted() {
        let mut comp = BeliefCompositor::new(NodeId(1));
        let origin = NodeId(2);
        let threat_id = 77u64;

        // Add threat fact
        let fact = threat_fact(origin, 1, threat_id, 0.0, 60.0, 0.8);
        let fact_id = fact.id();
        comp.ingest(fact);

        // Beliefs without retraction
        let before = comp.threat_beliefs(0.0);
        assert!(
            !before[&threat_id].retracted,
            "threat should not be retracted initially"
        );

        // Add retraction
        let ret = retraction_fact(origin, 2, 0.0, 120.0, fact_id);
        comp.ingest(ret);

        // The threat fact itself remains in the store (retracted set is checked at read time)
        let after = comp.threat_beliefs(0.0);
        // The threat fact is still in the store but the retracted set marks it
        // In current implementation: retracted facts are excluded from drone_beliefs
        // but for threats the retracted flag is set. Since the threat fact's ID is
        // in the retracted set, it should be marked retracted.
        assert!(
            after[&threat_id].retracted,
            "threat should be marked retracted after retraction"
        );
    }

    #[test]
    fn corroboration_counting() {
        let mut comp = BeliefCompositor::new(NodeId(1));
        let drone_id = NodeId(99);

        // Three independent observers of the same drone
        comp.ingest(drone_fact(
            NodeId(2),
            1,
            drone_id,
            0.0,
            30.0,
            0.5,
            [1.0, 0.0, 0.0],
        ));
        comp.ingest(drone_fact(
            NodeId(3),
            1,
            drone_id,
            0.0,
            30.0,
            0.6,
            [1.1, 0.0, 0.0],
        ));
        comp.ingest(drone_fact(
            NodeId(4),
            1,
            drone_id,
            0.0,
            30.0,
            0.4,
            [0.9, 0.0, 0.0],
        ));

        let beliefs = comp.drone_beliefs(0.0);
        let belief = beliefs.get(&drone_id).expect("drone belief should exist");

        assert_eq!(
            belief.corroboration_count, 3,
            "three observations should yield corroboration_count=3"
        );
        // Highest confidence (0.6 from node 3) wins
        assert!(
            (belief.confidence - 0.6).abs() < 1e-10,
            "highest confidence should be selected, got {}",
            belief.confidence
        );
        assert_eq!(
            belief.source,
            FactId {
                origin: NodeId(3),
                seq: 1
            }
        );
    }

    #[test]
    fn next_fact_id_increments() {
        let mut comp = BeliefCompositor::new(NodeId(7));
        let id1 = comp.next_fact_id();
        let id2 = comp.next_fact_id();
        let id3 = comp.next_fact_id();

        assert_eq!(id1.origin, NodeId(7));
        assert_eq!(id1.seq, 1);
        assert_eq!(id2.seq, 2);
        assert_eq!(id3.seq, 3);
    }

    #[test]
    fn partition_heal_simulation() {
        // Two compositors with disjoint facts (network partition).
        // After healing, merge all facts into one and verify complete world view.
        let mut comp_a = BeliefCompositor::new(NodeId(1));
        let mut comp_b = BeliefCompositor::new(NodeId(2));

        let drone_x = NodeId(10);
        let drone_y = NodeId(11);
        let threat_z = 42u64;

        // comp_a has observations of drone_x and threat_z
        let fact_drone_x = drone_fact(NodeId(1), 1, drone_x, 0.0, 60.0, 0.9, [5.0, 0.0, 0.0]);
        let fact_threat_z = threat_fact(NodeId(1), 2, threat_z, 0.0, 60.0, 0.85);
        comp_a.ingest(fact_drone_x.clone());
        comp_a.ingest(fact_threat_z.clone());

        // comp_b has observation of drone_y only
        let fact_drone_y = drone_fact(NodeId(2), 1, drone_y, 0.0, 60.0, 0.8, [10.0, 0.0, 0.0]);
        comp_b.ingest(fact_drone_y.clone());

        // Partition heals: share all facts bidirectionally
        // comp_b receives comp_a's facts
        comp_b.ingest(fact_drone_x.clone());
        comp_b.ingest(fact_threat_z.clone());
        // comp_a receives comp_b's facts
        comp_a.ingest(fact_drone_y.clone());

        // After healing both compositors should have the complete world view
        let beliefs_a = comp_a.drone_beliefs(0.0);
        let threats_a = comp_a.threat_beliefs(0.0);
        assert!(
            beliefs_a.contains_key(&drone_x),
            "comp_a should know about drone_x"
        );
        assert!(
            beliefs_a.contains_key(&drone_y),
            "comp_a should know about drone_y after healing"
        );
        assert!(
            threats_a.contains_key(&threat_z),
            "comp_a should know about threat_z"
        );

        let beliefs_b = comp_b.drone_beliefs(0.0);
        let threats_b = comp_b.threat_beliefs(0.0);
        assert!(
            beliefs_b.contains_key(&drone_x),
            "comp_b should know about drone_x after healing"
        );
        assert!(
            beliefs_b.contains_key(&drone_y),
            "comp_b should know about drone_y"
        );
        assert!(
            threats_b.contains_key(&threat_z),
            "comp_b should know about threat_z after healing"
        );

        // Both should agree on drone positions (same facts)
        let pos_a = beliefs_a[&drone_x].observation.position;
        let pos_b = beliefs_b[&drone_x].observation.position;
        assert_eq!(
            pos_a.0, pos_b.0,
            "both compositors should agree on drone_x position"
        );
    }

    #[test]
    fn retraction_ingested_before_fact_blocks_later_ingest() {
        let mut comp = BeliefCompositor::new(NodeId(1));
        let origin = NodeId(2);
        let drone_id = NodeId(50);
        let target_id = FactId { origin, seq: 1 };

        // Retraction arrives first (out-of-order / Byzantine scenario)
        let ret = retraction_fact(origin, 2, 1.0, 120.0, target_id);
        comp.ingest(ret);

        // Original fact arrives later — should be rejected
        let fact = drone_fact(origin, 1, drone_id, 0.0, 30.0, 0.9, [1.0, 2.0, 3.0]);
        let accepted = comp.ingest(fact);

        // The fact is rejected because its ID was pre-retracted
        assert!(
            !accepted,
            "fact arriving after its retraction should be rejected"
        );
        let beliefs = comp.drone_beliefs(1.0);
        assert!(
            !beliefs.contains_key(&drone_id),
            "retracted drone should not appear in beliefs"
        );
    }
}
