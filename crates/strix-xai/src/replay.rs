//! After-action review — mission replay, what-if analysis, timeline export.
//!
//! [`MissionReplay`] aggregates all decision traces from a mission into a
//! structured timeline with statistics, key moments, and the ability to
//! re-run decisions with modified parameters.

use serde::{Deserialize, Serialize};

use crate::trace::{DecisionTrace, DecisionType, TraceRecorder};

// ---------------------------------------------------------------------------
// Timeline event
// ---------------------------------------------------------------------------

/// A single event on the mission timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    /// Timestamp (seconds since mission start).
    pub timestamp: f64,
    /// Short label for the event.
    pub label: String,
    /// Which decision trace this event came from.
    pub trace_id: u64,
    /// Event category for visual grouping.
    pub category: EventCategory,
}

/// Category of a timeline event (used for colour coding in UIs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventCategory {
    /// Task-level events (assignment, completion).
    Task,
    /// Regime transitions.
    Regime,
    /// Formation changes.
    Formation,
    /// Threat detections and responses.
    Threat,
    /// Drone loss / recovery events.
    Attrition,
    /// Leadership changes.
    Leadership,
}

impl From<DecisionType> for EventCategory {
    fn from(dt: DecisionType) -> Self {
        match dt {
            DecisionType::TaskAssignment => EventCategory::Task,
            DecisionType::RegimeChange => EventCategory::Regime,
            DecisionType::FormationChange => EventCategory::Formation,
            DecisionType::ThreatResponse => EventCategory::Threat,
            DecisionType::ReAuction => EventCategory::Attrition,
            DecisionType::LeaderElection => EventCategory::Leadership,
        }
    }
}

// ---------------------------------------------------------------------------
// Key moment
// ---------------------------------------------------------------------------

/// A notable moment during the mission, identified by heuristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMoment {
    /// Timestamp of the event.
    pub timestamp: f64,
    /// Human-readable description.
    pub description: String,
    /// Significance score [0.0, 1.0] — higher means more important.
    pub significance: f64,
    /// Related trace IDs.
    pub trace_ids: Vec<u64>,
}

// ---------------------------------------------------------------------------
// Mission statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics for a completed (or in-progress) mission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionStatistics {
    /// Total number of decisions made.
    pub total_decisions: u64,
    /// Average decision latency (derived from inter-trace timestamps).
    pub avg_decision_time_ms: f64,
    /// Fraction of time spent in each regime: `[PATROL, ENGAGE, EVADE]`.
    pub regime_time_pct: [f64; 3],
    /// Fraction of drones lost during the mission.
    pub attrition_rate: f64,
    /// Whether the mission was judged successful.
    pub mission_success: bool,
    /// Key moments identified by automated analysis.
    pub key_moments: Vec<KeyMoment>,
}

// ---------------------------------------------------------------------------
// Mission replay
// ---------------------------------------------------------------------------

/// Full mission replay with traces, timeline, and statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionReplay {
    /// Mission identifier.
    pub mission_id: String,
    /// All decision traces, ordered chronologically.
    pub traces: Vec<DecisionTrace>,
    /// Timeline events derived from traces.
    pub timeline: Vec<TimelineEvent>,
    /// Aggregate statistics.
    pub statistics: MissionStatistics,
}

// ---------------------------------------------------------------------------
// Plan comparison
// ---------------------------------------------------------------------------

/// Side-by-side comparison between two decision plans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanComparison {
    /// Label for the first plan (e.g. "Actual").
    pub plan_a_label: String,
    /// Label for the second plan (e.g. "Alternative").
    pub plan_b_label: String,
    /// Total decisions in plan A.
    pub plan_a_decisions: usize,
    /// Total decisions in plan B.
    pub plan_b_decisions: usize,
    /// Decisions that differ between plans.
    pub differences: Vec<PlanDiff>,
}

/// A single difference between two plans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanDiff {
    /// Approximate timestamp where the divergence occurs.
    pub timestamp: f64,
    /// What plan A did.
    pub plan_a_action: String,
    /// What plan B did.
    pub plan_b_action: String,
    /// Impact assessment.
    pub impact: String,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Build a [`MissionReplay`] from a [`TraceRecorder`].
pub fn build_replay(mission_id: &str, recorder: &TraceRecorder) -> MissionReplay {
    let traces: Vec<DecisionTrace> = recorder.iter().cloned().collect();

    // Build timeline from traces
    let timeline: Vec<TimelineEvent> = traces
        .iter()
        .map(|t| TimelineEvent {
            timestamp: t.timestamp,
            label: t.output.action.clone(),
            trace_id: t.id,
            category: EventCategory::from(t.decision_type),
        })
        .collect();

    // Compute statistics
    let total_decisions = traces.len() as u64;

    // Average inter-decision time
    let avg_decision_time_ms = if traces.len() >= 2 {
        let total_span = traces.last().unwrap().timestamp - traces.first().unwrap().timestamp;
        if total_decisions > 1 {
            (total_span / (total_decisions - 1) as f64) * 1000.0
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Regime time percentages — estimate from regime labels in traces
    let mut regime_counts = [0u64; 3];
    for t in &traces {
        match t.inputs.regime.to_uppercase().as_str() {
            "PATROL" => regime_counts[0] += 1,
            "ENGAGE" => regime_counts[1] += 1,
            "EVADE" => regime_counts[2] += 1,
            _ => {}
        }
    }
    let regime_total = regime_counts.iter().sum::<u64>().max(1) as f64;
    let regime_time_pct = [
        regime_counts[0] as f64 / regime_total,
        regime_counts[1] as f64 / regime_total,
        regime_counts[2] as f64 / regime_total,
    ];

    // Attrition rate — count ReAuction events as proxy
    let loss_events = traces
        .iter()
        .filter(|t| t.decision_type == DecisionType::ReAuction)
        .count();
    let attrition_rate = if total_decisions > 0 {
        loss_events as f64 / total_decisions as f64
    } else {
        0.0
    };

    // Key moments — regime changes and high-confidence threat responses
    let key_moments: Vec<KeyMoment> = traces
        .iter()
        .filter(|t| {
            matches!(
                t.decision_type,
                DecisionType::RegimeChange | DecisionType::ReAuction
            ) || (t.decision_type == DecisionType::ThreatResponse && t.confidence > 0.8)
        })
        .map(|t| KeyMoment {
            timestamp: t.timestamp,
            description: t.output.action.clone(),
            significance: t.confidence,
            trace_ids: vec![t.id],
        })
        .collect();

    let statistics = MissionStatistics {
        total_decisions,
        avg_decision_time_ms,
        regime_time_pct,
        attrition_rate,
        mission_success: attrition_rate < 0.5, // simple heuristic
        key_moments,
    };

    MissionReplay {
        mission_id: mission_id.to_string(),
        traces,
        timeline,
        statistics,
    }
}

// ---------------------------------------------------------------------------
// What-if types
// ---------------------------------------------------------------------------

/// Parameters that can be modified in a what-if scenario.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WhatIfParams {
    /// Modified confidence threshold (original decision must exceed this to proceed).
    pub confidence_threshold: Option<f64>,
    /// Modified fear level F ∈ [0,1] — affects risk weighting.
    pub fear_override: Option<f64>,
    /// Modified number of available drones.
    pub drone_count_override: Option<usize>,
    /// Modified threat distance (meters).
    pub threat_distance_override: Option<f64>,
}

/// Result of a what-if analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhatIfResult {
    /// The original action that was taken.
    pub original_action: String,
    /// The original confidence.
    pub original_confidence: f64,
    /// Whether the decision would still be executed under modified parameters.
    pub would_proceed: bool,
    /// What would happen instead if the decision is rejected.
    pub fallback: String,
    /// Impact assessment — how the modified params change the decision landscape.
    pub impact: WhatIfImpact,
}

/// Quantified impact of parameter changes on the decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhatIfImpact {
    /// Confidence delta: positive means more confident under new params.
    pub confidence_delta: f64,
    /// Risk assessment change description.
    pub risk_change: String,
    /// Overall recommendation.
    pub recommendation: WhatIfRecommendation,
}

/// Overall recommendation produced by a what-if analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WhatIfRecommendation {
    /// Original decision holds under modified parameters.
    Confirm,
    /// Original decision is marginal — review recommended.
    Review,
    /// Original decision would be overridden.
    Override,
}

// ---------------------------------------------------------------------------
// What-if engine
// ---------------------------------------------------------------------------

/// Parametric what-if analysis: re-evaluate a trace with modified parameters.
///
/// Computes an effective confidence by applying fear, drone-count, and
/// threat-distance modifiers on top of the original confidence, then
/// compares against the (optionally overridden) threshold to determine
/// whether the original decision would still proceed.
pub fn what_if(trace: &DecisionTrace, params: &WhatIfParams) -> WhatIfResult {
    let threshold = params.confidence_threshold.unwrap_or(0.0);
    let would_proceed_original = trace.confidence >= threshold;

    // Compute effective confidence based on parameter modifications.
    let mut effective_confidence = trace.confidence;

    // Fear override affects confidence: higher fear reduces confidence in aggressive actions.
    if let Some(fear) = params.fear_override {
        let fear = fear.clamp(0.0, 1.0);
        let is_aggressive = matches!(
            trace.decision_type,
            DecisionType::TaskAssignment | DecisionType::ThreatResponse
        );
        if is_aggressive {
            effective_confidence *= 1.0 - fear * 0.3; // up to 30% confidence reduction
        }
    }

    // Drone count affects confidence: fewer drones reduces confidence.
    if let Some(count) = params.drone_count_override {
        let original_count = trace.inputs.drone_ids.len().max(1);
        let ratio = count as f64 / original_count as f64;
        effective_confidence *= ratio.clamp(0.5, 1.5);
    }

    // Threat distance affects confidence: closer threats reduce confidence.
    if let Some(dist) = params.threat_distance_override {
        if dist < 500.0 {
            effective_confidence *= dist / 500.0;
        }
    }

    let confidence_delta = effective_confidence - trace.confidence;

    let best_alternative = trace
        .alternatives_considered
        .iter()
        .max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned();

    let recommendation = if effective_confidence >= threshold && would_proceed_original {
        WhatIfRecommendation::Confirm
    } else if effective_confidence >= threshold * 0.8 {
        WhatIfRecommendation::Review
    } else {
        WhatIfRecommendation::Override
    };

    let risk_change = if confidence_delta > 0.05 {
        "Risk decreased — parameters favor this decision".to_string()
    } else if confidence_delta < -0.05 {
        "Risk increased — decision becomes marginal".to_string()
    } else {
        "Risk unchanged — parameters have minimal effect".to_string()
    };

    WhatIfResult {
        original_action: trace.output.action.clone(),
        original_confidence: trace.confidence,
        would_proceed: effective_confidence >= threshold,
        fallback: best_alternative
            .map(|a| a.description)
            .unwrap_or_else(|| "No alternative available".to_string()),
        impact: WhatIfImpact {
            confidence_delta,
            risk_change,
            recommendation,
        },
    }
}

/// Simple what-if: just a confidence threshold check.
///
/// Convenience wrapper around [`what_if`] for the common case of only
/// varying the confidence threshold.
pub fn what_if_simple(trace: &DecisionTrace, confidence_threshold: f64) -> WhatIfResult {
    what_if(
        trace,
        &WhatIfParams {
            confidence_threshold: Some(confidence_threshold),
            ..Default::default()
        },
    )
}

/// Compare two sets of traces side-by-side.
pub fn compare_plans(
    label_a: &str,
    traces_a: &[DecisionTrace],
    label_b: &str,
    traces_b: &[DecisionTrace],
) -> PlanComparison {
    // Find differences by matching on timestamp proximity
    let mut differences = Vec::new();
    let tolerance = 1.0; // seconds

    for ta in traces_a {
        // Find the closest trace in B by timestamp
        let closest_b = traces_b.iter().min_by(|b1, b2| {
            let d1 = (b1.timestamp - ta.timestamp).abs();
            let d2 = (b2.timestamp - ta.timestamp).abs();
            d1.partial_cmp(&d2).unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(tb) = closest_b {
            if (ta.timestamp - tb.timestamp).abs() < tolerance
                && ta.output.action != tb.output.action
            {
                differences.push(PlanDiff {
                    timestamp: ta.timestamp,
                    plan_a_action: ta.output.action.clone(),
                    plan_b_action: tb.output.action.clone(),
                    impact: format!(
                        "Confidence delta: {:.0}%",
                        (ta.confidence - tb.confidence).abs() * 100.0
                    ),
                });
            }
        }
    }

    PlanComparison {
        plan_a_label: label_a.to_string(),
        plan_b_label: label_b.to_string(),
        plan_a_decisions: traces_a.len(),
        plan_b_decisions: traces_b.len(),
        differences,
    }
}

/// Export the mission timeline as a JSON string for external visualisation.
pub fn export_timeline(replay: &MissionReplay) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&replay.timeline)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{DecisionTrace, DecisionType, TraceInputs, TraceRecorder};

    fn make_trace(ts: f64, dt: DecisionType, regime: &str, action: &str) -> DecisionTrace {
        DecisionTrace::new(ts, dt)
            .with_inputs(TraceInputs {
                drone_ids: vec![1, 2],
                regime: regime.into(),
                metrics: serde_json::Value::Null,
                context: serde_json::Value::Null,
                fear_level: None,
                courage_level: None,
                tension: None,
                calibration_quality: None,
            })
            .with_output(action, serde_json::Value::Null)
            .with_confidence(0.8)
    }

    #[test]
    fn build_replay_basic() {
        let mut recorder = TraceRecorder::new();
        recorder.record(make_trace(
            0.0,
            DecisionType::TaskAssignment,
            "PATROL",
            "Assign recon",
        ));
        recorder.record(make_trace(
            5.0,
            DecisionType::RegimeChange,
            "PATROL",
            "Switch to ENGAGE",
        ));
        recorder.record(make_trace(
            10.0,
            DecisionType::ThreatResponse,
            "ENGAGE",
            "Counter threat",
        ));

        let replay = build_replay("test-mission", &recorder);
        assert_eq!(replay.mission_id, "test-mission");
        assert_eq!(replay.traces.len(), 3);
        assert_eq!(replay.timeline.len(), 3);
        assert_eq!(replay.statistics.total_decisions, 3);

        // Regime distribution: 2 PATROL, 1 ENGAGE
        assert!(replay.statistics.regime_time_pct[0] > 0.5);
    }

    #[test]
    fn key_moments_detected() {
        let mut recorder = TraceRecorder::new();
        recorder.record(make_trace(
            0.0,
            DecisionType::TaskAssignment,
            "PATROL",
            "routine",
        ));
        recorder.record(make_trace(
            5.0,
            DecisionType::RegimeChange,
            "PATROL",
            "regime shift",
        ));
        recorder.record(make_trace(
            10.0,
            DecisionType::ReAuction,
            "ENGAGE",
            "recover",
        ));

        let replay = build_replay("m1", &recorder);
        assert!(replay.statistics.key_moments.len() >= 2);
    }

    #[test]
    fn what_if_analysis() {
        let trace = DecisionTrace::new(1.0, DecisionType::TaskAssignment)
            .with_output("Send 3 drones", serde_json::Value::Null)
            .with_confidence(0.75)
            .with_alternative("Send 2 drones", 0.65, "Less coverage");

        // With threshold below confidence — should proceed
        let result = what_if_simple(&trace, 0.5);
        assert!(result.would_proceed);

        // With threshold above confidence — should not proceed
        let result = what_if_simple(&trace, 0.9);
        assert!(!result.would_proceed);
        assert_eq!(result.fallback, "Send 2 drones");
    }

    #[test]
    fn fear_override_reduces_confidence_for_aggressive_actions() {
        // ThreatResponse is an aggressive action type
        let trace = DecisionTrace::new(1.0, DecisionType::ThreatResponse)
            .with_inputs(crate::trace::TraceInputs {
                drone_ids: vec![1, 2, 3],
                regime: "ENGAGE".into(),
                metrics: serde_json::Value::Null,
                context: serde_json::Value::Null,
                fear_level: None,
                courage_level: None,
                tension: None,
                calibration_quality: None,
            })
            .with_output("Intercept target", serde_json::Value::Null)
            .with_confidence(0.9);

        // Max fear (1.0) should reduce confidence by 30%: 0.9 * 0.7 = 0.63
        let result = what_if(
            &trace,
            &WhatIfParams {
                fear_override: Some(1.0),
                ..Default::default()
            },
        );
        let expected = 0.9 * 0.7;
        assert!(
            (result.original_confidence - 0.9).abs() < 1e-9,
            "original_confidence should be unchanged"
        );
        assert!(
            (result.impact.confidence_delta - (expected - 0.9)).abs() < 1e-6,
            "delta should be {:.4}, got {:.4}",
            expected - 0.9,
            result.impact.confidence_delta
        );
        // With no threshold, effective_confidence >= 0.0, so would_proceed
        assert!(result.would_proceed);
        assert_eq!(result.impact.recommendation, WhatIfRecommendation::Confirm);

        // Non-aggressive action (RegimeChange) should be unaffected by fear
        let trace_regime = DecisionTrace::new(1.0, DecisionType::RegimeChange)
            .with_output("Switch to EVADE", serde_json::Value::Null)
            .with_confidence(0.9);

        let result_regime = what_if(
            &trace_regime,
            &WhatIfParams {
                fear_override: Some(1.0),
                ..Default::default()
            },
        );
        assert!(
            result_regime.impact.confidence_delta.abs() < 1e-9,
            "non-aggressive action should not be affected by fear override"
        );
    }

    #[test]
    fn drone_count_override_scales_confidence() {
        // Trace with 4 drones, confidence 0.8
        let trace = DecisionTrace::new(1.0, DecisionType::TaskAssignment)
            .with_inputs(crate::trace::TraceInputs {
                drone_ids: vec![1, 2, 3, 4],
                regime: "PATROL".into(),
                metrics: serde_json::Value::Null,
                context: serde_json::Value::Null,
                fear_level: None,
                courage_level: None,
                tension: None,
                calibration_quality: None,
            })
            .with_output("Sweep sector", serde_json::Value::Null)
            .with_confidence(0.8);

        // Halving the drone count: ratio = 2/4 = 0.5, effective = 0.8 * 0.5 = 0.4
        let result_half = what_if(
            &trace,
            &WhatIfParams {
                drone_count_override: Some(2),
                ..Default::default()
            },
        );
        let expected_half = 0.8 * 0.5;
        assert!(
            (result_half.impact.confidence_delta - (expected_half - 0.8)).abs() < 1e-9,
            "halving drones: delta should be {:.4}, got {:.4}",
            expected_half - 0.8,
            result_half.impact.confidence_delta
        );

        // Doubling the drone count: ratio = 8/4 = 2.0, clamped to 1.5, effective = 0.8 * 1.5 = 1.2
        let result_double = what_if(
            &trace,
            &WhatIfParams {
                drone_count_override: Some(8),
                ..Default::default()
            },
        );
        let expected_double = 0.8 * 1.5;
        assert!(
            (result_double.impact.confidence_delta - (expected_double - 0.8)).abs() < 1e-9,
            "doubling drones (clamped): delta should be {:.4}, got {:.4}",
            expected_double - 0.8,
            result_double.impact.confidence_delta
        );
    }

    #[test]
    fn threat_distance_override_reduces_confidence_when_close() {
        let trace = DecisionTrace::new(1.0, DecisionType::ThreatResponse)
            .with_inputs(crate::trace::TraceInputs {
                drone_ids: vec![1],
                regime: "ENGAGE".into(),
                metrics: serde_json::Value::Null,
                context: serde_json::Value::Null,
                fear_level: None,
                courage_level: None,
                tension: None,
                calibration_quality: None,
            })
            .with_output("Engage threat", serde_json::Value::Null)
            .with_confidence(1.0);

        // Distance 250 m: ratio = 250/500 = 0.5, effective = 1.0 * 0.5 = 0.5
        let result_close = what_if(
            &trace,
            &WhatIfParams {
                threat_distance_override: Some(250.0),
                ..Default::default()
            },
        );
        assert!(
            (result_close.impact.confidence_delta - (-0.5)).abs() < 1e-9,
            "250m distance should halve confidence"
        );

        // Distance 600 m (>= 500): no adjustment, delta = 0
        let result_far = what_if(
            &trace,
            &WhatIfParams {
                threat_distance_override: Some(600.0),
                ..Default::default()
            },
        );
        assert!(
            result_far.impact.confidence_delta.abs() < 1e-9,
            "distance >= 500m should not affect confidence"
        );
    }

    #[test]
    fn recommendation_logic() {
        // confidence=0.9, threshold=0.8 → effective=0.9 >= 0.8 and original proceeds → Confirm
        let trace_confirm = DecisionTrace::new(1.0, DecisionType::RegimeChange)
            .with_output("Switch regime", serde_json::Value::Null)
            .with_confidence(0.9);
        let result = what_if(
            &trace_confirm,
            &WhatIfParams {
                confidence_threshold: Some(0.8),
                ..Default::default()
            },
        );
        assert_eq!(result.impact.recommendation, WhatIfRecommendation::Confirm);
        assert!(result.would_proceed);

        // confidence=0.75, threshold=0.9, no modifiers → effective=0.75, threshold*0.8=0.72
        // 0.75 >= 0.72 → Review (effective >= threshold*0.8 but not >= threshold)
        let trace_review = DecisionTrace::new(1.0, DecisionType::RegimeChange)
            .with_output("Marginal switch", serde_json::Value::Null)
            .with_confidence(0.75);
        let result = what_if(
            &trace_review,
            &WhatIfParams {
                confidence_threshold: Some(0.9),
                ..Default::default()
            },
        );
        assert_eq!(result.impact.recommendation, WhatIfRecommendation::Review);
        assert!(!result.would_proceed);

        // confidence=0.5, threshold=0.9, no modifiers → effective=0.5, threshold*0.8=0.72
        // 0.5 < 0.72 → Override
        let trace_override = DecisionTrace::new(1.0, DecisionType::RegimeChange)
            .with_output("Weak decision", serde_json::Value::Null)
            .with_confidence(0.5);
        let result = what_if(
            &trace_override,
            &WhatIfParams {
                confidence_threshold: Some(0.9),
                ..Default::default()
            },
        );
        assert_eq!(result.impact.recommendation, WhatIfRecommendation::Override);
        assert!(!result.would_proceed);
    }

    #[test]
    fn compare_plans_finds_differences() {
        let plan_a = vec![
            make_trace(1.0, DecisionType::TaskAssignment, "PATROL", "Go north"),
            make_trace(5.0, DecisionType::TaskAssignment, "PATROL", "Go south"),
        ];
        let plan_b = vec![
            make_trace(1.0, DecisionType::TaskAssignment, "PATROL", "Go north"),
            make_trace(5.0, DecisionType::TaskAssignment, "PATROL", "Go east"),
        ];

        let comparison = compare_plans("Actual", &plan_a, "Alternative", &plan_b);
        assert_eq!(comparison.differences.len(), 1);
        assert_eq!(comparison.differences[0].plan_a_action, "Go south");
        assert_eq!(comparison.differences[0].plan_b_action, "Go east");
    }

    #[test]
    fn export_timeline_json() {
        let mut recorder = TraceRecorder::new();
        recorder.record(make_trace(
            0.0,
            DecisionType::TaskAssignment,
            "PATROL",
            "Start",
        ));
        let replay = build_replay("export-test", &recorder);
        let json = export_timeline(&replay).unwrap();
        assert!(json.contains("Start"));
        assert!(json.contains("timestamp"));
    }

    #[test]
    fn attrition_rate() {
        let mut recorder = TraceRecorder::new();
        for i in 0..10 {
            let dt = if i % 3 == 0 {
                DecisionType::ReAuction
            } else {
                DecisionType::TaskAssignment
            };
            recorder.record(make_trace(i as f64, dt, "ENGAGE", "action"));
        }
        let replay = build_replay("attrition", &recorder);
        // 4 out of 10 decisions are ReAuction (0, 3, 6, 9)
        assert!((replay.statistics.attrition_rate - 0.4).abs() < 0.01);
        assert!(replay.statistics.mission_success); // 0.4 < 0.5
    }

    #[test]
    fn empty_replay() {
        let recorder = TraceRecorder::new();
        let replay = build_replay("empty", &recorder);
        assert_eq!(replay.statistics.total_decisions, 0);
        assert_eq!(replay.statistics.avg_decision_time_ms, 0.0);
        assert!(replay.statistics.mission_success);
    }
}
