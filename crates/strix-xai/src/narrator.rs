//! Trace-to-text narration — making machine decisions human-readable.
//!
//! The narrator takes a [`DecisionTrace`](crate::trace::DecisionTrace) and
//! produces a natural-language explanation at one of three detail levels.
//!
//! # Example output (STANDARD)
//!
//! > Assigned 3 drones to northern ridge because: particle filter estimates
//! > 73% enemy probability there, regime=ENGAGE (CUSUM detected spike),
//! > risk budget 15% (within 20% limit).

use crate::trace::{DecisionTrace, DecisionType};

// ---------------------------------------------------------------------------
// Detail level
// ---------------------------------------------------------------------------

/// How much detail the narration should include.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetailLevel {
    /// One-liner summary — suitable for dashboards and logs.
    Brief,
    /// Multi-sentence explanation — good for operators.
    Standard,
    /// Full reasoning chain with all alternatives — for after-action review.
    Detailed,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Narrate any decision trace at the requested detail level.
pub fn narrate_decision(trace: &DecisionTrace, level: DetailLevel) -> String {
    match trace.decision_type {
        DecisionType::TaskAssignment => narrate_task_assignment(trace, level),
        DecisionType::RegimeChange => narrate_regime_change(trace, level),
        DecisionType::FormationChange => narrate_formation_change(trace, level),
        DecisionType::ThreatResponse => narrate_threat_response(trace, level),
        DecisionType::ReAuction => narrate_loss_response(trace, level),
        DecisionType::LeaderElection => narrate_leader_election(trace, level),
    }
}

/// Narrate a task assignment decision.
pub fn narrate_task_assignment(trace: &DecisionTrace, level: DetailLevel) -> String {
    let drone_count = trace.inputs.drone_ids.len();
    let action = &trace.output.action;

    match level {
        DetailLevel::Brief => {
            format!(
                "[t={:.1}s] {action} (confidence {:.0}%)",
                trace.timestamp,
                trace.confidence * 100.0,
            )
        }
        DetailLevel::Standard => {
            let mut msg = format!(
                "[t={:.1}s] {action} — {drone_count} drone(s) involved",
                trace.timestamp,
            );
            // Append key metrics if available
            if let Some(metrics) = trace.inputs.metrics.as_object() {
                let metric_strs: Vec<String> =
                    metrics.iter().map(|(k, v)| format!("{k}={v}")).collect();
                if !metric_strs.is_empty() {
                    msg.push_str(&format!(" because: {}", metric_strs.join(", ")));
                }
            }
            if !trace.inputs.regime.is_empty() {
                msg.push_str(&format!(", regime={}", trace.inputs.regime));
            }
            msg.push_str(&format!(". Confidence: {:.0}%.", trace.confidence * 100.0));
            msg
        }
        DetailLevel::Detailed => {
            let mut msg = format!(
                "=== Task Assignment (trace #{}) at t={:.1}s ===\n",
                trace.id, trace.timestamp,
            );
            msg.push_str(&format!("Action: {action}\n"));
            msg.push_str(&format!("Drones: {:?}\n", trace.inputs.drone_ids));
            msg.push_str(&format!("Regime: {}\n", trace.inputs.regime));
            msg.push_str(&format!("Confidence: {:.1}%\n\n", trace.confidence * 100.0));

            // Reasoning chain
            msg.push_str("Reasoning:\n");
            for step in &trace.reasoning {
                msg.push_str(&format!(
                    "  {}. {} — {}\n",
                    step.step, step.description, step.data
                ));
            }

            // Alternatives
            if !trace.alternatives_considered.is_empty() {
                msg.push_str("\nAlternatives considered:\n");
                for alt in &trace.alternatives_considered {
                    msg.push_str(&format!(
                        "  - {} (score={:.2}) — rejected: {}\n",
                        alt.description, alt.score, alt.rejection_reason
                    ));
                }
            }
            msg
        }
    }
}

/// Narrate a regime change decision.
pub fn narrate_regime_change(trace: &DecisionTrace, level: DetailLevel) -> String {
    let action = &trace.output.action;
    let from_regime = &trace.inputs.regime;

    // Try to extract the new regime from output details
    let to_regime = trace
        .output
        .details
        .get("new_regime")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    match level {
        DetailLevel::Brief => {
            format!(
                "[t={:.1}s] Regime: {from_regime} -> {to_regime}",
                trace.timestamp,
            )
        }
        DetailLevel::Standard => {
            let mut msg = format!(
                "[t={:.1}s] Regime changed from {from_regime} to {to_regime}.",
                trace.timestamp,
            );
            // Include trigger reason from first reasoning step
            if let Some(step) = trace.reasoning.first() {
                msg.push_str(&format!(" Trigger: {}.", step.description));
            }
            msg.push_str(&format!(" Confidence: {:.0}%.", trace.confidence * 100.0));
            msg
        }
        DetailLevel::Detailed => {
            let mut msg = format!(
                "=== Regime Change (trace #{}) at t={:.1}s ===\n",
                trace.id, trace.timestamp,
            );
            msg.push_str(&format!("Transition: {from_regime} -> {to_regime}\n"));
            msg.push_str(&format!("Action: {action}\n\n"));

            msg.push_str("Reasoning:\n");
            for step in &trace.reasoning {
                msg.push_str(&format!(
                    "  {}. {} — {}\n",
                    step.step, step.description, step.data
                ));
            }
            msg
        }
    }
}

/// Narrate a re-auction after drone loss.
pub fn narrate_loss_response(trace: &DecisionTrace, level: DetailLevel) -> String {
    let action = &trace.output.action;

    // Try to extract the lost drone ID
    let lost_id = trace
        .inputs
        .context
        .get("lost_drone_id")
        .and_then(|v| v.as_u64())
        .map(|id| format!("drone #{id}"))
        .unwrap_or_else(|| "a drone".to_string());

    match level {
        DetailLevel::Brief => {
            format!(
                "[t={:.1}s] Re-auction after loss of {lost_id}",
                trace.timestamp,
            )
        }
        DetailLevel::Standard => {
            let remaining = trace.inputs.drone_ids.len();
            let mut msg = format!(
                "[t={:.1}s] {lost_id} lost — re-auctioning tasks among {remaining} remaining drones.",
                trace.timestamp,
            );
            msg.push_str(&format!(" {action}."));
            msg.push_str(&format!(" Confidence: {:.0}%.", trace.confidence * 100.0));
            msg
        }
        DetailLevel::Detailed => {
            let mut msg = format!(
                "=== Re-Auction after Loss (trace #{}) at t={:.1}s ===\n",
                trace.id, trace.timestamp,
            );
            msg.push_str(&format!("Lost: {lost_id}\n"));
            msg.push_str(&format!("Remaining drones: {:?}\n", trace.inputs.drone_ids));
            msg.push_str(&format!("Action: {action}\n\n"));

            msg.push_str("Reasoning:\n");
            for step in &trace.reasoning {
                msg.push_str(&format!(
                    "  {}. {} — {}\n",
                    step.step, step.description, step.data
                ));
            }

            if !trace.alternatives_considered.is_empty() {
                msg.push_str("\nAlternatives:\n");
                for alt in &trace.alternatives_considered {
                    msg.push_str(&format!(
                        "  - {} (score={:.2}) — rejected: {}\n",
                        alt.description, alt.score, alt.rejection_reason
                    ));
                }
            }
            msg
        }
    }
}

/// Narrate a threat prediction or response.
pub fn narrate_threat_response(trace: &DecisionTrace, level: DetailLevel) -> String {
    let action = &trace.output.action;

    let threat_prob = trace
        .inputs
        .metrics
        .get("threat_probability")
        .and_then(|v| v.as_f64())
        .map(|p| format!("{:.0}%", p * 100.0))
        .unwrap_or_else(|| "unknown".to_string());

    match level {
        DetailLevel::Brief => {
            format!(
                "[t={:.1}s] Threat response (prob={threat_prob}): {action}",
                trace.timestamp,
            )
        }
        DetailLevel::Standard => {
            let mut msg = format!(
                "[t={:.1}s] Threat detected (probability {threat_prob}). {}.",
                trace.timestamp, action,
            );
            msg.push_str(&format!(" Confidence: {:.0}%.", trace.confidence * 100.0));
            msg
        }
        DetailLevel::Detailed => {
            let mut msg = format!(
                "=== Threat Response (trace #{}) at t={:.1}s ===\n",
                trace.id, trace.timestamp,
            );
            msg.push_str(&format!("Threat probability: {threat_prob}\n"));
            msg.push_str(&format!("Action: {action}\n\n"));

            msg.push_str("Reasoning:\n");
            for step in &trace.reasoning {
                msg.push_str(&format!(
                    "  {}. {} — {}\n",
                    step.step, step.description, step.data
                ));
            }
            msg
        }
    }
}

/// Narrate a formation change.
fn narrate_formation_change(trace: &DecisionTrace, level: DetailLevel) -> String {
    let action = &trace.output.action;

    match level {
        DetailLevel::Brief => {
            format!("[t={:.1}s] Formation change: {action}", trace.timestamp,)
        }
        DetailLevel::Standard => {
            let mut msg = format!("[t={:.1}s] Formation changed: {action}.", trace.timestamp,);
            if let Some(step) = trace.reasoning.first() {
                msg.push_str(&format!(" Reason: {}.", step.description));
            }
            msg
        }
        DetailLevel::Detailed => {
            let mut msg = format!(
                "=== Formation Change (trace #{}) at t={:.1}s ===\n",
                trace.id, trace.timestamp,
            );
            msg.push_str(&format!("Action: {action}\n\n"));
            msg.push_str("Reasoning:\n");
            for step in &trace.reasoning {
                msg.push_str(&format!(
                    "  {}. {} — {}\n",
                    step.step, step.description, step.data
                ));
            }
            msg
        }
    }
}

/// Narrate a leader election.
fn narrate_leader_election(trace: &DecisionTrace, level: DetailLevel) -> String {
    let action = &trace.output.action;

    let new_leader = trace
        .output
        .details
        .get("new_leader")
        .and_then(|v| v.as_u64())
        .map(|id| format!("drone #{id}"))
        .unwrap_or_else(|| "unknown".to_string());

    match level {
        DetailLevel::Brief => {
            format!("[t={:.1}s] New leader: {new_leader}", trace.timestamp,)
        }
        DetailLevel::Standard => {
            format!(
                "[t={:.1}s] Leader elected: {new_leader}. {action}. Confidence: {:.0}%.",
                trace.timestamp,
                trace.confidence * 100.0
            )
        }
        DetailLevel::Detailed => {
            let mut msg = format!(
                "=== Leader Election (trace #{}) at t={:.1}s ===\n",
                trace.id, trace.timestamp,
            );
            msg.push_str(&format!("New leader: {new_leader}\n"));
            msg.push_str(&format!("Action: {action}\n\n"));
            msg.push_str("Reasoning:\n");
            for step in &trace.reasoning {
                msg.push_str(&format!(
                    "  {}. {} — {}\n",
                    step.step, step.description, step.data
                ));
            }
            msg
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::TraceInputs;

    fn sample_assignment_trace() -> DecisionTrace {
        DecisionTrace::new(42.5, DecisionType::TaskAssignment)
            .with_inputs(TraceInputs {
                drone_ids: vec![1, 2, 3],
                regime: "ENGAGE".into(),
                metrics: serde_json::json!({
                    "threat_prob": 0.73,
                    "risk_budget": 0.15,
                }),
                context: serde_json::Value::Null,
                fear_level: None,
                courage_level: None,
                tension: None,
                calibration_quality: None,
            })
            .with_step(
                1,
                "Particle filter estimates 73% enemy probability at northern ridge",
                serde_json::json!(0.73),
            )
            .with_step(
                2,
                "CUSUM detected spike — regime=ENGAGE",
                serde_json::json!({"cusum_stat": 4.2, "threshold": 3.0}),
            )
            .with_step(
                3,
                "Risk budget check: 15% used of 20% limit",
                serde_json::json!({"used": 0.15, "limit": 0.20}),
            )
            .with_output(
                "Assign 3 drones to northern ridge",
                serde_json::json!({"drones": [1,2,3], "target": "northern_ridge"}),
            )
            .with_confidence(0.85)
            .with_alternative("Send 2 drones", 0.72, "Insufficient coverage probability")
            .with_alternative(
                "Hold position",
                0.45,
                "Threat probability exceeds threshold",
            )
    }

    fn sample_regime_trace() -> DecisionTrace {
        DecisionTrace::new(38.0, DecisionType::RegimeChange)
            .with_inputs(TraceInputs {
                drone_ids: vec![1, 2, 3, 4, 5],
                regime: "PATROL".into(),
                metrics: serde_json::json!({"cusum_stat": 4.2}),
                context: serde_json::Value::Null,
                fear_level: None,
                courage_level: None,
                tension: None,
                calibration_quality: None,
            })
            .with_step(
                1,
                "CUSUM statistic exceeded threshold",
                serde_json::json!({"cusum": 4.2, "threshold": 3.0}),
            )
            .with_output(
                "Switch to ENGAGE regime",
                serde_json::json!({"new_regime": "ENGAGE"}),
            )
            .with_confidence(0.92)
    }

    #[test]
    fn brief_task_assignment() {
        let trace = sample_assignment_trace();
        let text = narrate_decision(&trace, DetailLevel::Brief);
        assert!(text.contains("42.5"));
        assert!(text.contains("85%"));
        assert!(text.contains("northern ridge"));
    }

    #[test]
    fn standard_task_assignment() {
        let trace = sample_assignment_trace();
        let text = narrate_decision(&trace, DetailLevel::Standard);
        assert!(text.contains("3 drone(s)"));
        assert!(text.contains("ENGAGE"));
        assert!(text.contains("Confidence"));
    }

    #[test]
    fn detailed_task_assignment() {
        let trace = sample_assignment_trace();
        let text = narrate_decision(&trace, DetailLevel::Detailed);
        assert!(text.contains("Reasoning:"));
        assert!(text.contains("Particle filter"));
        assert!(text.contains("CUSUM"));
        assert!(text.contains("Alternatives considered:"));
        assert!(text.contains("Send 2 drones"));
        assert!(text.contains("Insufficient coverage"));
    }

    #[test]
    fn regime_change_narration() {
        let trace = sample_regime_trace();

        let brief = narrate_decision(&trace, DetailLevel::Brief);
        assert!(brief.contains("PATROL"));
        assert!(brief.contains("ENGAGE"));

        let standard = narrate_decision(&trace, DetailLevel::Standard);
        assert!(standard.contains("Regime changed"));
        assert!(standard.contains("CUSUM"));
    }

    #[test]
    fn loss_response_narration() {
        let trace = DecisionTrace::new(55.0, DecisionType::ReAuction)
            .with_inputs(TraceInputs {
                drone_ids: vec![2, 3, 4],
                regime: "ENGAGE".into(),
                metrics: serde_json::json!({}),
                context: serde_json::json!({"lost_drone_id": 1}),
                fear_level: None,
                courage_level: None,
                tension: None,
                calibration_quality: None,
            })
            .with_step(
                1,
                "Drone #1 lost — heartbeat timeout",
                serde_json::json!({"timeout_ms": 5000}),
            )
            .with_output(
                "Redistributed tasks to remaining drones",
                serde_json::Value::Null,
            )
            .with_confidence(0.78);

        let brief = narrate_decision(&trace, DetailLevel::Brief);
        assert!(brief.contains("drone #1"));

        let standard = narrate_decision(&trace, DetailLevel::Standard);
        assert!(standard.contains("3 remaining"));
    }

    #[test]
    fn threat_response_narration() {
        let trace = DecisionTrace::new(60.0, DecisionType::ThreatResponse)
            .with_inputs(TraceInputs {
                drone_ids: vec![1, 2],
                regime: "ENGAGE".into(),
                metrics: serde_json::json!({"threat_probability": 0.91}),
                context: serde_json::Value::Null,
                fear_level: None,
                courage_level: None,
                tension: None,
                calibration_quality: None,
            })
            .with_output("Evade north-east", serde_json::Value::Null)
            .with_confidence(0.95);

        let text = narrate_decision(&trace, DetailLevel::Standard);
        assert!(text.contains("91%"));
        assert!(text.contains("Evade"));
    }
}
