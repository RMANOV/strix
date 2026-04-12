//! Fact-based gossip model for STRIX mesh.
//!
//! Instead of sharing full state snapshots, nodes exchange typed,
//! confidence-stamped facts with causal ordering and TTL.

use serde::{Deserialize, Serialize};

use crate::{NodeId, Position3D};

/// Unique fact identifier: (originator, sequence number).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FactId {
    /// Node that created this fact.
    pub origin: NodeId,
    /// Monotonic sequence number from origin.
    pub seq: u64,
}

/// Causal stamp for ordering and provenance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CausalStamp {
    /// Who originated this fact.
    pub originator: NodeId,
    /// Monotonic sequence from originator.
    pub sequence: u64,
    /// When the fact was observed (seconds).
    pub observed_at: f64,
    /// When we received it (seconds).
    pub received_at: f64,
    /// How many hops this fact has traveled.
    pub hop_count: u8,
}

/// What kind of fact this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FactKind {
    /// Direct sensor observation.
    Observation,
    /// Derived from observations (e.g. intent classification).
    Inference,
    /// Cancels a previous fact.
    Retraction,
}

/// A typed, stamped fact envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactEnvelope<T> {
    /// Unique identifier.
    pub id: FactId,
    /// What kind of fact.
    pub kind: FactKind,
    /// The actual data.
    pub payload: T,
    /// Causal ordering and provenance.
    pub stamp: CausalStamp,
    /// Time-to-live in seconds from observed_at.
    pub ttl_s: f64,
    /// How confident the originator is in this fact [0, 1].
    pub confidence: f64,
}

impl<T> FactEnvelope<T> {
    /// Whether this fact has expired given the current time.
    pub fn is_expired(&self, now: f64) -> bool {
        now - self.stamp.observed_at > self.ttl_s
    }

    /// Age of this fact in seconds.
    pub fn age(&self, now: f64) -> f64 {
        (now - self.stamp.observed_at).max(0.0)
    }

    /// Effective confidence, decayed by age.
    /// Returns confidence * exp(-age / ttl), so older facts have less weight.
    pub fn effective_confidence(&self, now: f64) -> f64 {
        let age = self.age(now);
        if self.ttl_s <= 0.0 {
            return 0.0;
        }
        self.confidence * (-age / self.ttl_s).exp()
    }
}

// --- Fact payload types ---

/// Observation of a drone's state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneObservation {
    /// Which drone was observed.
    pub drone_id: NodeId,
    /// Observed position.
    pub position: Position3D,
    /// Observed battery level [0, 1].
    pub battery: f64,
    /// Observed regime (as string for compatibility).
    pub regime: String,
}

/// Observation of a threat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatFactPayload {
    /// Threat identifier.
    pub threat_id: u64,
    /// Observed position.
    pub position: Position3D,
    /// Threat level [0, 1].
    pub threat_level: f64,
    /// Whether the threat has been resolved.
    pub resolved: bool,
}

/// Retraction of a previously reported fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactRetraction {
    /// ID of the fact being retracted.
    pub retracted_id: FactId,
    /// Reason for retraction.
    pub reason: String,
}

/// Union of all mesh fact payloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeshFact {
    /// Observation of a drone.
    Drone(FactEnvelope<DroneObservation>),
    /// Observation of a threat.
    Threat(FactEnvelope<ThreatFactPayload>),
    /// Retraction of a previous fact.
    Retraction(FactEnvelope<FactRetraction>),
}

impl MeshFact {
    /// Get the fact ID regardless of variant.
    pub fn id(&self) -> FactId {
        match self {
            MeshFact::Drone(f) => f.id,
            MeshFact::Threat(f) => f.id,
            MeshFact::Retraction(f) => f.id,
        }
    }

    /// Get the originator.
    pub fn originator(&self) -> NodeId {
        match self {
            MeshFact::Drone(f) => f.stamp.originator,
            MeshFact::Threat(f) => f.stamp.originator,
            MeshFact::Retraction(f) => f.stamp.originator,
        }
    }

    /// Whether this fact is expired.
    pub fn is_expired(&self, now: f64) -> bool {
        match self {
            MeshFact::Drone(f) => f.is_expired(now),
            MeshFact::Threat(f) => f.is_expired(now),
            MeshFact::Retraction(f) => f.is_expired(now),
        }
    }

    /// Get the causal stamp.
    pub fn stamp(&self) -> &CausalStamp {
        match self {
            MeshFact::Drone(f) => &f.stamp,
            MeshFact::Threat(f) => &f.stamp,
            MeshFact::Retraction(f) => &f.stamp,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stamp(originator: NodeId, seq: u64, observed_at: f64) -> CausalStamp {
        CausalStamp {
            originator,
            sequence: seq,
            observed_at,
            received_at: observed_at,
            hop_count: 0,
        }
    }

    fn make_drone_fact(
        origin: NodeId,
        seq: u64,
        observed_at: f64,
        ttl_s: f64,
        confidence: f64,
        drone_id: NodeId,
    ) -> MeshFact {
        MeshFact::Drone(FactEnvelope {
            id: FactId { origin, seq },
            kind: FactKind::Observation,
            payload: DroneObservation {
                drone_id,
                position: Position3D([1.0, 2.0, 3.0]),
                battery: 0.8,
                regime: "search".to_string(),
            },
            stamp: make_stamp(origin, seq, observed_at),
            ttl_s,
            confidence,
        })
    }

    fn make_threat_fact(
        origin: NodeId,
        seq: u64,
        observed_at: f64,
        ttl_s: f64,
        confidence: f64,
        threat_id: u64,
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

    fn make_retraction_fact(
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
                reason: "no longer valid".to_string(),
            },
            stamp: make_stamp(origin, seq, observed_at),
            ttl_s,
            confidence: 1.0,
        })
    }

    #[test]
    fn fact_expiry() {
        let origin = NodeId(1);
        let fact = make_drone_fact(origin, 1, 0.0, 10.0, 1.0, NodeId(2));
        // Not expired at t=9
        assert!(
            !fact.is_expired(9.0),
            "should not be expired at t=9 with TTL=10"
        );
        // Expired at t=11
        assert!(
            fact.is_expired(11.0),
            "should be expired at t=11 with TTL=10"
        );
        // Exactly at TTL boundary: 10.0 - 0.0 = 10.0, not > 10.0
        assert!(
            !fact.is_expired(10.0),
            "exactly at TTL should not be expired (> not >=)"
        );
    }

    #[test]
    fn effective_confidence_decay() {
        let origin = NodeId(1);
        // confidence=1.0, ttl_s=10.0, observed_at=0.0
        let envelope = FactEnvelope {
            id: FactId { origin, seq: 1 },
            kind: FactKind::Observation,
            payload: DroneObservation {
                drone_id: NodeId(2),
                position: Position3D([0.0, 0.0, 0.0]),
                battery: 1.0,
                regime: "idle".to_string(),
            },
            stamp: make_stamp(origin, 1, 0.0),
            ttl_s: 10.0,
            confidence: 1.0,
        };

        // At t=0, confidence = 1.0 * exp(0) = 1.0
        let conf_t0 = envelope.effective_confidence(0.0);
        assert!(
            (conf_t0 - 1.0).abs() < 1e-10,
            "confidence at t=0 should be 1.0, got {conf_t0}"
        );

        // At t=10 (one TTL), confidence = 1.0 * exp(-1) ≈ 0.3679
        let conf_t10 = envelope.effective_confidence(10.0);
        let expected = (-1.0_f64).exp();
        assert!(
            (conf_t10 - expected).abs() < 1e-10,
            "confidence at t=10 should be exp(-1)={expected:.6}, got {conf_t10:.6}"
        );

        // At t=5 (half TTL), confidence = 1.0 * exp(-0.5)
        let conf_t5 = envelope.effective_confidence(5.0);
        let expected_half = (-0.5_f64).exp();
        assert!(
            (conf_t5 - expected_half).abs() < 1e-10,
            "confidence at t=5 should be exp(-0.5)={expected_half:.6}, got {conf_t5:.6}"
        );

        // Confidence is monotonically decreasing
        assert!(
            conf_t0 > conf_t5 && conf_t5 > conf_t10,
            "confidence should decay over time"
        );

        // Zero or negative TTL returns 0.0
        let envelope_zero_ttl = FactEnvelope {
            id: FactId { origin, seq: 2 },
            kind: FactKind::Observation,
            payload: DroneObservation {
                drone_id: NodeId(3),
                position: Position3D([0.0, 0.0, 0.0]),
                battery: 1.0,
                regime: "idle".to_string(),
            },
            stamp: make_stamp(origin, 2, 0.0),
            ttl_s: 0.0,
            confidence: 1.0,
        };
        assert_eq!(
            envelope_zero_ttl.effective_confidence(1.0),
            0.0,
            "zero TTL should return 0.0 confidence"
        );
    }

    #[test]
    fn mesh_fact_id_and_originator() {
        let node_a = NodeId(10);
        let node_b = NodeId(20);

        let drone_fact = make_drone_fact(node_a, 5, 0.0, 30.0, 0.9, NodeId(99));
        assert_eq!(
            drone_fact.id(),
            FactId {
                origin: node_a,
                seq: 5
            }
        );
        assert_eq!(drone_fact.originator(), node_a);

        let threat_fact = make_threat_fact(node_b, 7, 0.0, 60.0, 0.8, 42);
        assert_eq!(
            threat_fact.id(),
            FactId {
                origin: node_b,
                seq: 7
            }
        );
        assert_eq!(threat_fact.originator(), node_b);

        let retracted_id = FactId {
            origin: node_a,
            seq: 3,
        };
        let retraction = make_retraction_fact(node_b, 8, 0.0, 120.0, retracted_id);
        assert_eq!(
            retraction.id(),
            FactId {
                origin: node_b,
                seq: 8
            }
        );
        assert_eq!(retraction.originator(), node_b);
    }

    #[test]
    fn serde_roundtrip() {
        // DroneObservation envelope
        let drone_fact = make_drone_fact(NodeId(1), 1, 5.0, 30.0, 0.9, NodeId(2));
        let json = serde_json::to_string(&drone_fact).expect("serialize drone fact");
        let back: MeshFact = serde_json::from_str(&json).expect("deserialize drone fact");
        assert_eq!(back.id(), drone_fact.id());
        assert_eq!(back.originator(), drone_fact.originator());

        // ThreatFactPayload envelope
        let threat_fact = make_threat_fact(NodeId(3), 2, 10.0, 60.0, 0.75, 99);
        let json = serde_json::to_string(&threat_fact).expect("serialize threat fact");
        let back: MeshFact = serde_json::from_str(&json).expect("deserialize threat fact");
        assert_eq!(back.id(), threat_fact.id());

        // FactRetraction envelope
        let retracted_id = FactId {
            origin: NodeId(1),
            seq: 1,
        };
        let retraction = make_retraction_fact(NodeId(5), 3, 15.0, 120.0, retracted_id);
        let json = serde_json::to_string(&retraction).expect("serialize retraction");
        let back: MeshFact = serde_json::from_str(&json).expect("deserialize retraction");
        assert_eq!(back.id(), retraction.id());
        if let MeshFact::Retraction(env) = back {
            assert_eq!(env.payload.retracted_id, retracted_id);
            assert_eq!(env.payload.reason, "no longer valid");
        } else {
            panic!("expected Retraction variant");
        }
    }

    #[test]
    fn fact_age_clamped_to_zero_for_future() {
        let envelope = FactEnvelope {
            id: FactId {
                origin: NodeId(1),
                seq: 1,
            },
            kind: FactKind::Observation,
            payload: DroneObservation {
                drone_id: NodeId(2),
                position: Position3D([0.0, 0.0, 0.0]),
                battery: 1.0,
                regime: "idle".to_string(),
            },
            stamp: make_stamp(NodeId(1), 1, 100.0), // observed in the future
            ttl_s: 30.0,
            confidence: 0.9,
        };
        // now=50 is before observed_at=100, age clamped to 0
        assert_eq!(envelope.age(50.0), 0.0);
        // confidence at now=50 should equal full confidence (age=0 → exp(0)=1)
        assert!((envelope.effective_confidence(50.0) - 0.9).abs() < 1e-10);
    }

    #[test]
    fn causal_stamp_hop_count() {
        let stamp = CausalStamp {
            originator: NodeId(1),
            sequence: 42,
            observed_at: 1.0,
            received_at: 1.5,
            hop_count: 3,
        };
        let envelope = FactEnvelope {
            id: FactId {
                origin: NodeId(1),
                seq: 42,
            },
            kind: FactKind::Inference,
            payload: DroneObservation {
                drone_id: NodeId(2),
                position: Position3D([0.0, 0.0, 0.0]),
                battery: 0.5,
                regime: "track".to_string(),
            },
            stamp,
            ttl_s: 20.0,
            confidence: 0.7,
        };
        assert_eq!(envelope.stamp.hop_count, 3);
        assert_eq!(envelope.kind, FactKind::Inference);
    }
}
