//! Causal correlation and trace linking.
//!
//! Links decisions -> executions -> outcomes for full audit trails.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

use super::reason_codes::ReasonCode;

static NEXT_TRACE_ID: AtomicU64 = AtomicU64::new(1);

/// Unique trace identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TraceId(pub u64);

impl TraceId {
    /// Generate a new unique trace ID.
    pub fn next() -> Self {
        Self(NEXT_TRACE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Relationship between causally linked traces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalRelationship {
    /// This trace was caused by the parent.
    CausedBy,
    /// This trace supersedes/replaces the parent.
    Supersedes,
    /// This trace conflicts with the parent.
    ConflictsWith,
}

/// A link to a parent trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLink {
    /// Parent trace ID.
    pub parent: TraceId,
    /// Nature of the relationship.
    pub relationship: CausalRelationship,
}

/// A structured decision record with causal links.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredDecision {
    /// Unique ID for this decision.
    pub id: TraceId,
    /// When the decision was made (seconds).
    pub timestamp: f64,
    /// Which drone this decision affects.
    pub drone_id: Option<u32>,
    /// Machine-readable reason.
    pub reason: ReasonCode,
    /// Causal links to parent decisions.
    pub caused_by: Vec<CausalLink>,
    /// Human-readable summary.
    pub summary: String,
}

/// Record of what was actually sent to the platform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    /// Links to the decision that triggered this.
    pub decision_id: TraceId,
    /// Unique ID for this execution record.
    pub id: TraceId,
    /// What commands were issued.
    pub commands_issued: Vec<String>,
    /// When execution started.
    pub started_at: f64,
    /// When execution completed (if known).
    pub completed_at: Option<f64>,
}

/// What actually happened after execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeRecord {
    /// Links to the execution.
    pub execution_id: TraceId,
    /// Links to the original decision.
    pub decision_id: TraceId,
    /// Unique ID for this outcome.
    pub id: TraceId,
    /// Whether the outcome matched expectations.
    pub outcome: OutcomeStatus,
    /// Deviation from plan (0.0 = perfect, 1.0 = completely different).
    pub deviation: f64,
    /// When the outcome was observed.
    pub observed_at: f64,
}

/// Status of an outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutcomeStatus {
    /// Everything went as planned.
    Success,
    /// Partially successful.
    PartialSuccess,
    /// Failed.
    Failure,
    /// Superseded by a newer decision before completion.
    Superseded,
    /// Outcome unknown (e.g. drone lost contact).
    Unknown,
}

/// Escalation level for human-swarm interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EscalationLevel {
    /// FYI — no action needed.
    Informational,
    /// Heads up — situation developing.
    Advisory,
    /// Decision needed — swarm is asking for guidance.
    ActionRequired,
    /// Emergency — immediate attention.
    Critical,
}

impl EscalationLevel {
    /// Determine escalation level from a reason code.
    pub fn from_reason(reason: &ReasonCode) -> Self {
        match reason {
            ReasonCode::PeriodicReEvaluation { .. } => Self::Informational,
            ReasonCode::AuctionReassignment { .. } | ReasonCode::ConvergenceIssue { .. } => {
                Self::Advisory
            }
            ReasonCode::FormationDegradation { .. }
            | ReasonCode::CommunicationsDegraded { .. }
            | ReasonCode::EnergyConservation { .. }
            | ReasonCode::RegimeShift { .. } => Self::Advisory,
            ReasonCode::ThreatDetected { confidence, .. } => {
                if *confidence > 0.8 {
                    Self::Critical
                } else {
                    Self::ActionRequired
                }
            }
            ReasonCode::DroneAttrition { .. }
            | ReasonCode::SafetyConstraintActive { .. }
            | ReasonCode::KillZoneAvoidance { .. } => Self::Critical,
            ReasonCode::HumanOverride { .. } => Self::ActionRequired,
        }
    }
}

/// An escalation event for the human operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Escalation {
    /// Escalation level.
    pub level: EscalationLevel,
    /// The decision that triggered this escalation.
    pub decision_id: TraceId,
    /// The reason.
    pub reason: ReasonCode,
    /// Human-readable summary.
    pub summary: String,
    /// When this was generated.
    pub timestamp: f64,
}

/// Attention filter — suppresses low-priority escalations during high-tempo ops.
pub struct AttentionFilter {
    /// Minimum escalation level to surface.
    min_level: EscalationLevel,
    /// Cooldown between escalations of the same type (seconds).
    cooldown_s: f64,
    /// Last escalation time per label.
    last_escalation: std::collections::HashMap<&'static str, f64>,
}

impl AttentionFilter {
    /// Current minimum escalation level.
    pub fn min_level(&self) -> EscalationLevel {
        self.min_level
    }

    /// Set minimum escalation level.
    pub fn set_min_level(&mut self, level: EscalationLevel) {
        self.min_level = level;
    }

    /// Current cooldown in seconds.
    pub fn cooldown_s(&self) -> f64 {
        self.cooldown_s
    }

    /// Set cooldown, clearing cached escalation times.
    pub fn set_cooldown(&mut self, cooldown_s: f64) {
        self.cooldown_s = cooldown_s;
        self.last_escalation.clear();
    }
}

impl AttentionFilter {
    /// Create with default settings.
    pub fn new() -> Self {
        Self {
            min_level: EscalationLevel::Advisory,
            cooldown_s: 5.0,
            last_escalation: std::collections::HashMap::new(),
        }
    }

    /// Create a permissive filter (shows everything).
    pub fn permissive() -> Self {
        Self {
            min_level: EscalationLevel::Informational,
            cooldown_s: 0.0,
            last_escalation: std::collections::HashMap::new(),
        }
    }

    /// Whether this escalation should be surfaced.
    pub fn should_surface(&mut self, escalation: &Escalation) -> bool {
        // Always surface critical
        if escalation.level == EscalationLevel::Critical {
            return true;
        }

        // Check minimum level
        if escalation.level < self.min_level {
            return false;
        }

        // Check cooldown
        let label = escalation.reason.label();
        if let Some(&last) = self.last_escalation.get(label) {
            if escalation.timestamp - last < self.cooldown_s {
                return false;
            }
        }

        self.last_escalation.insert(label, escalation.timestamp);
        true
    }
}

impl Default for AttentionFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_decision(reason: ReasonCode) -> StructuredDecision {
        StructuredDecision {
            id: TraceId::next(),
            timestamp: 1.0,
            drone_id: Some(1),
            reason,
            caused_by: vec![],
            summary: "test decision".into(),
        }
    }

    fn make_execution(decision_id: TraceId) -> ExecutionRecord {
        ExecutionRecord {
            decision_id,
            id: TraceId::next(),
            commands_issued: vec!["move_north".into()],
            started_at: 1.5,
            completed_at: Some(2.0),
        }
    }

    fn make_outcome(decision_id: TraceId, execution_id: TraceId) -> OutcomeRecord {
        OutcomeRecord {
            execution_id,
            decision_id,
            id: TraceId::next(),
            outcome: OutcomeStatus::Success,
            deviation: 0.05,
            observed_at: 2.5,
        }
    }

    #[test]
    fn trace_id_uniqueness() {
        let ids: Vec<TraceId> = (0..100).map(|_| TraceId::next()).collect();
        let unique: std::collections::HashSet<u64> = ids.iter().map(|t| t.0).collect();
        assert_eq!(unique.len(), 100, "all 100 IDs must be distinct");
    }

    #[test]
    fn structured_decision_creation() {
        let reason = ReasonCode::ThreatDetected {
            threat_id: 77,
            confidence: 0.95,
        };
        let d = make_decision(reason);
        assert_eq!(d.drone_id, Some(1));
        assert_eq!(d.summary, "test decision");
        assert!(d.caused_by.is_empty());
        assert_eq!(d.reason.label(), "threat_detected");
    }

    #[test]
    fn execution_links_to_decision() {
        let d = make_decision(ReasonCode::PeriodicReEvaluation { interval_ticks: 5 });
        let exec = make_execution(d.id);
        assert_eq!(exec.decision_id, d.id);
        assert_eq!(exec.commands_issued, vec!["move_north"]);
        assert_eq!(exec.completed_at, Some(2.0));
    }

    #[test]
    fn outcome_links_to_both() {
        let d = make_decision(ReasonCode::PeriodicReEvaluation { interval_ticks: 5 });
        let exec = make_execution(d.id);
        let out = make_outcome(d.id, exec.id);
        assert_eq!(out.decision_id, d.id);
        assert_eq!(out.execution_id, exec.id);
        assert_eq!(out.outcome, OutcomeStatus::Success);
    }

    #[test]
    fn escalation_levels_ordered() {
        assert!(EscalationLevel::Informational < EscalationLevel::Advisory);
        assert!(EscalationLevel::Advisory < EscalationLevel::ActionRequired);
        assert!(EscalationLevel::ActionRequired < EscalationLevel::Critical);
    }

    #[test]
    fn escalation_from_reason() {
        // High-confidence threat -> Critical
        assert_eq!(
            EscalationLevel::from_reason(&ReasonCode::ThreatDetected {
                threat_id: 1,
                confidence: 0.9
            }),
            EscalationLevel::Critical
        );
        // Exactly at boundary (0.8 is NOT > 0.8)
        assert_eq!(
            EscalationLevel::from_reason(&ReasonCode::ThreatDetected {
                threat_id: 2,
                confidence: 0.8
            }),
            EscalationLevel::ActionRequired
        );
        // Low-confidence threat -> ActionRequired
        assert_eq!(
            EscalationLevel::from_reason(&ReasonCode::ThreatDetected {
                threat_id: 3,
                confidence: 0.5
            }),
            EscalationLevel::ActionRequired
        );
        // Periodic -> Informational
        assert_eq!(
            EscalationLevel::from_reason(&ReasonCode::PeriodicReEvaluation { interval_ticks: 10 }),
            EscalationLevel::Informational
        );
        // DroneAttrition -> Critical
        assert_eq!(
            EscalationLevel::from_reason(&ReasonCode::DroneAttrition {
                drone_id: 3,
                cause: "lost".into()
            }),
            EscalationLevel::Critical
        );
        // KillZone -> Critical
        assert_eq!(
            EscalationLevel::from_reason(&ReasonCode::KillZoneAvoidance {
                zone_position: [0.0; 3],
                zone_radius: 10.0
            }),
            EscalationLevel::Critical
        );
        // HumanOverride -> ActionRequired
        assert_eq!(
            EscalationLevel::from_reason(&ReasonCode::HumanOverride {
                operator_id: "op".into()
            }),
            EscalationLevel::ActionRequired
        );
        // AuctionReassignment -> Advisory
        assert_eq!(
            EscalationLevel::from_reason(&ReasonCode::AuctionReassignment {
                task_id: 1,
                old_drone: None,
                new_drone: 2,
                bid_score: 0.5
            }),
            EscalationLevel::Advisory
        );
    }

    fn make_escalation(
        reason: ReasonCode,
        level: EscalationLevel,
        timestamp: f64,
        id: TraceId,
    ) -> Escalation {
        Escalation {
            level,
            decision_id: id,
            reason,
            summary: "test".into(),
            timestamp,
        }
    }

    #[test]
    fn attention_filter_suppresses_low() {
        let mut filter = AttentionFilter::new(); // min_level = Advisory
        let id = TraceId::next();
        let esc = make_escalation(
            ReasonCode::PeriodicReEvaluation { interval_ticks: 1 },
            EscalationLevel::Informational,
            10.0,
            id,
        );
        assert!(
            !filter.should_surface(&esc),
            "Informational must be suppressed by Advisory-minimum filter"
        );
    }

    #[test]
    fn attention_filter_cooldown() {
        let mut filter = AttentionFilter::new(); // cooldown_s = 5.0
        let id = TraceId::next();
        let reason = ReasonCode::AuctionReassignment {
            task_id: 1,
            old_drone: None,
            new_drone: 2,
            bid_score: 0.5,
        };

        // First Advisory escalation — surfaces
        let esc1 = make_escalation(reason.clone(), EscalationLevel::Advisory, 10.0, id);
        assert!(filter.should_surface(&esc1), "first Advisory must surface");

        // Same type, 2 seconds later — still within 5s cooldown, suppressed
        let esc2 = make_escalation(reason.clone(), EscalationLevel::Advisory, 12.0, id);
        assert!(
            !filter.should_surface(&esc2),
            "second Advisory within cooldown must be suppressed"
        );

        // Same type, 6 seconds after first — cooldown expired, surfaces again
        let esc3 = make_escalation(reason, EscalationLevel::Advisory, 16.0, id);
        assert!(
            filter.should_surface(&esc3),
            "Advisory after cooldown must surface"
        );
    }

    #[test]
    fn attention_filter_always_surfaces_critical() {
        let mut filter = AttentionFilter::new(); // min_level = Advisory
        let id = TraceId::next();

        // Emit two Critical escalations of the same type in rapid succession —
        // both must surface regardless of cooldown or min_level.
        let reason = ReasonCode::DroneAttrition {
            drone_id: 1,
            cause: "destroyed".into(),
        };
        let esc1 = make_escalation(reason.clone(), EscalationLevel::Critical, 1.0, id);
        let esc2 = make_escalation(reason, EscalationLevel::Critical, 1.5, id);

        assert!(filter.should_surface(&esc1), "first Critical must surface");
        assert!(
            filter.should_surface(&esc2),
            "second Critical must also surface (bypasses cooldown)"
        );
    }

    #[test]
    fn causal_chain() {
        // Build a decision -> execution -> outcome chain and verify IDs thread through.
        let decision = make_decision(ReasonCode::RegimeShift {
            from: "Patrol".into(),
            to: "Engage".into(),
        });
        let d_id = decision.id;

        let exec = make_execution(d_id);
        let e_id = exec.id;

        let outcome = make_outcome(d_id, e_id);

        assert_eq!(exec.decision_id, d_id, "exec must reference its decision");
        assert_eq!(outcome.decision_id, d_id, "outcome must reference decision");
        assert_eq!(
            outcome.execution_id, e_id,
            "outcome must reference execution"
        );

        // All three IDs are distinct
        let ids = [d_id.0, e_id.0, outcome.id.0];
        let unique: std::collections::HashSet<u64> = ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            3,
            "decision, exec, outcome IDs must all differ"
        );
    }

    #[test]
    fn serde_roundtrip() {
        // StructuredDecision
        let d = StructuredDecision {
            id: TraceId::next(),
            timestamp: 42.0,
            drone_id: Some(7),
            reason: ReasonCode::FormationDegradation {
                quality: 0.4,
                threshold: 0.6,
            },
            caused_by: vec![CausalLink {
                parent: TraceId::next(),
                relationship: CausalRelationship::CausedBy,
            }],
            summary: "formation fell below threshold".into(),
        };
        let json = serde_json::to_string(&d).expect("serialize StructuredDecision");
        let d2: StructuredDecision =
            serde_json::from_str(&json).expect("deserialize StructuredDecision");
        assert_eq!(d.id, d2.id);
        assert_eq!(d.timestamp, d2.timestamp);
        assert_eq!(d.drone_id, d2.drone_id);
        assert_eq!(d.caused_by.len(), d2.caused_by.len());

        // ExecutionRecord
        let exec = ExecutionRecord {
            decision_id: d.id,
            id: TraceId::next(),
            commands_issued: vec!["hover".into(), "regroup".into()],
            started_at: 43.0,
            completed_at: None,
        };
        let json = serde_json::to_string(&exec).expect("serialize ExecutionRecord");
        let exec2: ExecutionRecord =
            serde_json::from_str(&json).expect("deserialize ExecutionRecord");
        assert_eq!(exec.id, exec2.id);
        assert_eq!(exec.decision_id, exec2.decision_id);
        assert_eq!(exec.completed_at, exec2.completed_at);

        // OutcomeRecord
        let out = OutcomeRecord {
            execution_id: exec.id,
            decision_id: d.id,
            id: TraceId::next(),
            outcome: OutcomeStatus::PartialSuccess,
            deviation: 0.3,
            observed_at: 45.0,
        };
        let json = serde_json::to_string(&out).expect("serialize OutcomeRecord");
        let out2: OutcomeRecord = serde_json::from_str(&json).expect("deserialize OutcomeRecord");
        assert_eq!(out.id, out2.id);
        assert_eq!(out.outcome, out2.outcome);
        assert_eq!(out.deviation, out2.deviation);
    }
}
