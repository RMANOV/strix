//! Decision trace recording — the audit log of STRIX's mind.
//!
//! Every time the orchestrator makes a meaningful decision (task assignment,
//! regime change, formation switch, threat response), it emits a
//! [`DecisionTrace`] that captures the full reasoning chain.
//!
//! [`TraceRecorder`] stores traces in an append-only log. Traces can be
//! queried by time, type, drone, or arbitrary predicates.

use serde::{Deserialize, Serialize};
use thiserror::Error;

fn sanitize_timestamp(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn sanitize_confidence(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn sanitize_unit_interval(value: Option<f64>) -> Option<f64> {
    value.and_then(|value| value.is_finite().then_some(value.clamp(0.0, 1.0)))
}

fn sanitize_tension(value: Option<f64>) -> Option<f64> {
    value.and_then(|value| value.is_finite().then_some(value.clamp(-1.0, 1.0)))
}

fn sanitize_inputs(inputs: &mut TraceInputs) {
    inputs.fear_level = sanitize_unit_interval(inputs.fear_level);
    inputs.courage_level = sanitize_unit_interval(inputs.courage_level);
    inputs.tension = sanitize_tension(inputs.tension);
    inputs.calibration_quality = sanitize_unit_interval(inputs.calibration_quality);
}

fn sanitize_trace(trace: &mut DecisionTrace) {
    trace.timestamp = sanitize_timestamp(trace.timestamp);
    sanitize_inputs(&mut trace.inputs);
    trace.confidence = sanitize_confidence(trace.confidence);
    trace
        .alternatives_considered
        .retain(|alternative| alternative.score.is_finite());
}

pub(crate) fn sanitized_trace(mut trace: DecisionTrace) -> DecisionTrace {
    sanitize_trace(&mut trace);
    trace
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from the trace recording system.
#[derive(Debug, Error)]
pub enum TraceError {
    /// Trace ID not found.
    #[error("trace not found: {0}")]
    NotFound(u64),

    /// Serialisation error.
    #[error("serialisation error: {0}")]
    SerializationError(String),

    /// Storage is full or I/O failed.
    #[error("storage error: {0}")]
    StorageError(String),
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// What kind of decision was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecisionType {
    /// A task was assigned to one or more drones.
    TaskAssignment,
    /// The swarm regime changed (Patrol / Engage / Evade).
    RegimeChange,
    /// The swarm formation was altered.
    FormationChange,
    /// Response to a detected or predicted threat.
    ThreatResponse,
    /// Tasks were re-auctioned after a drone loss.
    ReAuction,
    /// A new leader was elected in a sub-swarm.
    LeaderElection,
    /// CBF/GCBF+ safety constraint applied to velocities.
    SafetyClamp,
    /// Criticality scheduler modulated adaptive parameters.
    CriticalityAdjustment,
    /// Epistemic evidence graph detected unresolvable conflict.
    EpistemicEscalation,
    /// XOR conflict between subsystems.
    EpistemicConflict,
    /// NOR information vacuum detected.
    EpistemicVacuum,
}

/// A single step in the reasoning chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Ordinal step number (1-based).
    pub step: u32,
    /// Human-readable description of what was evaluated.
    pub description: String,
    /// Structured data associated with this step (metrics, scores, etc.).
    pub data: serde_json::Value,
}

/// An alternative that was considered but not chosen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alternative {
    /// What this alternative would have done.
    pub description: String,
    /// Composite score (higher = better).
    pub score: f64,
    /// Why this alternative was rejected.
    pub rejection_reason: String,
}

/// Structured inputs that drove the decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceInputs {
    /// IDs of the drones involved.
    pub drone_ids: Vec<u32>,
    /// Current regime at decision time.
    pub regime: String,
    /// Key metrics snapshot (e.g. battery levels, distances, threat scores).
    pub metrics: serde_json::Value,
    /// Additional context (sensor readings, comms state, etc.).
    pub context: serde_json::Value,
    // ── Phi-sim intelligence fields (None when phi-sim feature is disabled) ──
    /// Fear level at decision time [0, 1].
    pub fear_level: Option<f64>,
    /// Courage level at decision time [0, 1].
    pub courage_level: Option<f64>,
    /// Opponent process tension (C-F)/(1+F*C).
    pub tension: Option<f64>,
    /// Phi-sim model calibration quality [0, 1].
    pub calibration_quality: Option<f64>,
}

/// The output of the decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceOutput {
    /// Concise description of the action taken.
    pub action: String,
    /// Detailed structured data about the output.
    pub details: serde_json::Value,
}

/// A complete record of a single decision, from inputs through reasoning
/// to the final output — with alternatives that were considered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTrace {
    /// Unique monotonic trace ID.
    pub id: u64,
    /// Timestamp (seconds since mission start or epoch).
    pub timestamp: f64,
    /// Category of decision.
    pub decision_type: DecisionType,
    /// Structured inputs.
    pub inputs: TraceInputs,
    /// Step-by-step reasoning chain.
    pub reasoning: Vec<ReasoningStep>,
    /// Final output.
    pub output: TraceOutput,
    /// Confidence in the decision [0.0, 1.0].
    pub confidence: f64,
    /// Alternatives that were evaluated but not selected.
    pub alternatives_considered: Vec<Alternative>,
}

// ---------------------------------------------------------------------------
// Query filter
// ---------------------------------------------------------------------------

/// Filter predicate for querying traces.
#[derive(Debug, Clone, Default)]
pub struct TraceQuery {
    /// Only traces after this timestamp (inclusive).
    pub after: Option<f64>,
    /// Only traces before this timestamp (inclusive).
    pub before: Option<f64>,
    /// Only traces of this type.
    pub decision_type: Option<DecisionType>,
    /// Only traces involving this drone.
    pub drone_id: Option<u32>,
    /// Only traces with confidence above this threshold.
    pub min_confidence: Option<f64>,
    /// Maximum number of results.
    pub limit: Option<usize>,
}

// ---------------------------------------------------------------------------
// Trace recorder
// ---------------------------------------------------------------------------

/// Append-only decision trace log.
///
/// Stores all [`DecisionTrace`]s in memory with fast query support.
/// For persistent storage, traces can be serialised to JSON.
pub struct TraceRecorder {
    /// The trace log, ordered by insertion (and therefore by ID).
    traces: Vec<DecisionTrace>,
    /// Next ID to assign.
    next_id: u64,
}

impl TraceRecorder {
    /// Create an empty trace recorder.
    pub fn new() -> Self {
        Self {
            traces: Vec::new(),
            next_id: 1,
        }
    }

    /// Record a new decision, assigning it a unique ID.
    ///
    /// Returns the assigned trace ID.
    pub fn record(&mut self, mut trace: DecisionTrace) -> u64 {
        sanitize_trace(&mut trace);
        let id = self.next_id;
        self.next_id += 1;
        trace.id = id;
        if self.traces.len() >= 10_000 {
            self.traces.remove(0);
        }
        self.traces.push(trace);
        id
    }

    /// Get a trace by its ID.
    pub fn get(&self, id: u64) -> Result<&DecisionTrace, TraceError> {
        self.traces
            .iter()
            .find(|t| t.id == id)
            .ok_or(TraceError::NotFound(id))
    }

    /// Query traces matching a filter.
    pub fn query(&self, filter: &TraceQuery) -> Vec<&DecisionTrace> {
        let after = filter.after.filter(|value| value.is_finite());
        let before = filter.before.filter(|value| value.is_finite());
        let min_confidence = filter
            .min_confidence
            .filter(|value| value.is_finite())
            .map(|value| value.clamp(0.0, 1.0));
        let mut results: Vec<&DecisionTrace> = self
            .traces
            .iter()
            .filter(|t| {
                if let Some(after) = after {
                    if t.timestamp < after {
                        return false;
                    }
                }
                if let Some(before) = before {
                    if t.timestamp > before {
                        return false;
                    }
                }
                if let Some(dt) = filter.decision_type {
                    if t.decision_type != dt {
                        return false;
                    }
                }
                if let Some(drone_id) = filter.drone_id {
                    if !t.inputs.drone_ids.contains(&drone_id) {
                        return false;
                    }
                }
                if let Some(min_conf) = min_confidence {
                    if t.confidence < min_conf {
                        return false;
                    }
                }
                true
            })
            .collect();

        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }

        results
    }

    /// Total number of recorded traces.
    pub fn len(&self) -> usize {
        self.traces.len()
    }

    /// Whether the recorder is empty.
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    /// Iterate over all traces.
    pub fn iter(&self) -> impl Iterator<Item = &DecisionTrace> {
        self.traces.iter()
    }

    /// Export all traces as a JSON string.
    pub fn export_json(&self) -> Result<String, TraceError> {
        let traces: Vec<DecisionTrace> = self.traces.iter().cloned().map(sanitized_trace).collect();
        serde_json::to_string_pretty(&traces)
            .map_err(|e| TraceError::SerializationError(e.to_string()))
    }

    /// Import traces from a JSON string, appending to the existing log.
    ///
    /// IDs are re-assigned to maintain monotonicity.
    pub fn import_json(&mut self, json: &str) -> Result<usize, TraceError> {
        let imported: Vec<DecisionTrace> = serde_json::from_str(json)
            .map_err(|e| TraceError::SerializationError(e.to_string()))?;
        let count = imported.len();
        for trace in imported {
            self.record(trace);
        }
        Ok(count)
    }
}

impl Default for TraceRecorder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

impl DecisionTrace {
    /// Start building a new trace with required fields.
    pub fn new(timestamp: f64, decision_type: DecisionType) -> Self {
        Self {
            id: 0, // assigned by TraceRecorder
            timestamp: sanitize_timestamp(timestamp),
            decision_type,
            inputs: TraceInputs {
                drone_ids: Vec::new(),
                regime: String::new(),
                metrics: serde_json::Value::Null,
                context: serde_json::Value::Null,
                fear_level: None,
                courage_level: None,
                tension: None,
                calibration_quality: None,
            },
            reasoning: Vec::new(),
            output: TraceOutput {
                action: String::new(),
                details: serde_json::Value::Null,
            },
            confidence: 0.0,
            alternatives_considered: Vec::new(),
        }
    }

    /// Set the inputs.
    pub fn with_inputs(mut self, inputs: TraceInputs) -> Self {
        self.inputs = inputs;
        sanitize_inputs(&mut self.inputs);
        self
    }

    /// Add a reasoning step.
    pub fn with_step(mut self, step: u32, description: &str, data: serde_json::Value) -> Self {
        self.reasoning.push(ReasoningStep {
            step,
            description: description.to_string(),
            data,
        });
        self
    }

    /// Set the output.
    pub fn with_output(mut self, action: &str, details: serde_json::Value) -> Self {
        self.output = TraceOutput {
            action: action.to_string(),
            details,
        };
        self
    }

    /// Set the confidence.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = sanitize_confidence(confidence);
        self
    }

    /// Add an alternative that was considered.
    pub fn with_alternative(mut self, description: &str, score: f64, reason: &str) -> Self {
        if score.is_finite() {
            self.alternatives_considered.push(Alternative {
                description: description.to_string(),
                score,
                rejection_reason: reason.to_string(),
            });
        }
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_trace(ts: f64, dt: DecisionType) -> DecisionTrace {
        DecisionTrace::new(ts, dt)
            .with_inputs(TraceInputs {
                drone_ids: vec![1, 2, 3],
                regime: "Patrol".into(),
                metrics: serde_json::json!({"threat_prob": 0.73}),
                context: serde_json::Value::Null,
                fear_level: None,
                courage_level: None,
                tension: None,
                calibration_quality: None,
            })
            .with_step(1, "Evaluated threat probability", serde_json::json!(0.73))
            .with_step(
                2,
                "Checked risk budget",
                serde_json::json!({"used": 0.15, "limit": 0.20}),
            )
            .with_output(
                "Assign 3 drones to northern ridge",
                serde_json::json!({"drones": [1,2,3], "target": "northern_ridge"}),
            )
            .with_confidence(0.85)
            .with_alternative("Send only 2 drones", 0.72, "Insufficient coverage")
    }

    #[test]
    fn record_and_retrieve() {
        let mut recorder = TraceRecorder::new();
        let id = recorder.record(sample_trace(10.0, DecisionType::TaskAssignment));
        assert_eq!(id, 1);
        assert_eq!(recorder.len(), 1);

        let trace = recorder.get(id).unwrap();
        assert_eq!(trace.confidence, 0.85);
        assert_eq!(trace.reasoning.len(), 2);
    }

    #[test]
    fn query_by_type() {
        let mut recorder = TraceRecorder::new();
        recorder.record(sample_trace(1.0, DecisionType::TaskAssignment));
        recorder.record(sample_trace(2.0, DecisionType::RegimeChange));
        recorder.record(sample_trace(3.0, DecisionType::TaskAssignment));

        let query = TraceQuery {
            decision_type: Some(DecisionType::TaskAssignment),
            ..Default::default()
        };
        let results = recorder.query(&query);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn query_by_time_range() {
        let mut recorder = TraceRecorder::new();
        for i in 0..10 {
            recorder.record(sample_trace(i as f64, DecisionType::TaskAssignment));
        }

        let query = TraceQuery {
            after: Some(3.0),
            before: Some(7.0),
            ..Default::default()
        };
        let results = recorder.query(&query);
        assert_eq!(results.len(), 5); // 3,4,5,6,7
    }

    #[test]
    fn query_by_drone_id() {
        let mut recorder = TraceRecorder::new();
        let mut t1 = sample_trace(1.0, DecisionType::TaskAssignment);
        t1.inputs.drone_ids = vec![1, 2];
        recorder.record(t1);

        let mut t2 = sample_trace(2.0, DecisionType::TaskAssignment);
        t2.inputs.drone_ids = vec![3, 4];
        recorder.record(t2);

        let query = TraceQuery {
            drone_id: Some(3),
            ..Default::default()
        };
        let results = recorder.query(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].inputs.drone_ids, vec![3, 4]);
    }

    #[test]
    fn export_import_json() {
        let mut recorder = TraceRecorder::new();
        recorder.record(sample_trace(1.0, DecisionType::RegimeChange));
        recorder.record(sample_trace(2.0, DecisionType::ThreatResponse));

        let json = recorder.export_json().unwrap();
        assert!(json.contains("RegimeChange"));

        let mut new_recorder = TraceRecorder::new();
        let count = new_recorder.import_json(&json).unwrap();
        assert_eq!(count, 2);
        assert_eq!(new_recorder.len(), 2);
    }

    #[test]
    fn not_found() {
        let recorder = TraceRecorder::new();
        assert!(recorder.get(999).is_err());
    }

    #[test]
    fn query_limit() {
        let mut recorder = TraceRecorder::new();
        for i in 0..20 {
            recorder.record(sample_trace(i as f64, DecisionType::TaskAssignment));
        }

        let query = TraceQuery {
            limit: Some(5),
            ..Default::default()
        };
        assert_eq!(recorder.query(&query).len(), 5);
    }

    #[test]
    fn test_trace_inputs_phi_sim_fields_serialize() {
        let inputs = TraceInputs {
            drone_ids: vec![1],
            regime: "Patrol".into(),
            metrics: serde_json::Value::Null,
            context: serde_json::Value::Null,
            fear_level: Some(0.42),
            courage_level: Some(0.65),
            tension: Some(-0.15),
            calibration_quality: Some(0.88),
        };
        let json = serde_json::to_string(&inputs).unwrap();
        assert!(json.contains("0.42"), "fear_level should serialize");
        assert!(json.contains("0.65"), "courage_level should serialize");
        assert!(json.contains("-0.15"), "tension should serialize");
        assert!(
            json.contains("0.88"),
            "calibration_quality should serialize"
        );

        // Round-trip
        let decoded: TraceInputs = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.fear_level, Some(0.42));
        assert_eq!(decoded.courage_level, Some(0.65));
    }

    #[test]
    fn test_trace_inputs_phi_sim_fields_none_serialize() {
        let inputs = TraceInputs {
            drone_ids: vec![],
            regime: "Patrol".into(),
            metrics: serde_json::Value::Null,
            context: serde_json::Value::Null,
            fear_level: None,
            courage_level: None,
            tension: None,
            calibration_quality: None,
        };
        let json = serde_json::to_string(&inputs).unwrap();
        // None fields should serialize as null
        let decoded: TraceInputs = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.fear_level, None);
        assert_eq!(decoded.tension, None);
    }

    #[test]
    fn record_sanitizes_non_finite_trace_fields() {
        let mut recorder = TraceRecorder::new();
        let mut trace = sample_trace(1.0, DecisionType::TaskAssignment);
        trace.timestamp = f64::NAN;
        trace.confidence = f64::INFINITY;
        trace.inputs.fear_level = Some(f64::NAN);
        trace.inputs.courage_level = Some(2.0);
        trace.inputs.tension = Some(f64::NEG_INFINITY);
        trace.inputs.calibration_quality = Some(-1.0);
        trace.alternatives_considered.push(Alternative {
            description: "poison".into(),
            score: f64::NAN,
            rejection_reason: "invalid".into(),
        });

        let id = recorder.record(trace);
        let stored = recorder.get(id).unwrap();

        assert_eq!(stored.timestamp, 0.0);
        assert_eq!(stored.confidence, 0.0);
        assert_eq!(stored.inputs.fear_level, None);
        assert_eq!(stored.inputs.courage_level, Some(1.0));
        assert_eq!(stored.inputs.tension, None);
        assert_eq!(stored.inputs.calibration_quality, Some(0.0));
        assert!(stored
            .alternatives_considered
            .iter()
            .all(|alt| alt.score.is_finite()));
    }

    #[test]
    fn query_ignores_non_finite_filter_values() {
        let mut recorder = TraceRecorder::new();
        recorder.record(sample_trace(1.0, DecisionType::TaskAssignment));
        recorder.record(sample_trace(2.0, DecisionType::TaskAssignment));

        let query = TraceQuery {
            after: Some(f64::NAN),
            before: Some(f64::INFINITY),
            min_confidence: Some(f64::NAN),
            ..Default::default()
        };

        assert_eq!(recorder.query(&query).len(), 2);
    }
}
