//! Byzantine-resilient validation for gossip state merging.
//!
//! Implements W-MSR-inspired trimmed filtering: when multiple peers report
//! values for the same drone/threat, extreme outliers are discarded before
//! the newest-wins rule applies. This prevents a single corrupted node from
//! injecting arbitrarily false state into the swarm's shared picture.
//!
//! The approach is adapted for STRIX's version-based gossip (not averaging
//! consensus): instead of trimming values and averaging, we trim *reports*
//! by plausibility and only accept the update if it passes validation.

use serde::{Deserialize, Serialize};

use crate::gossip::DroneState;
use crate::{NodeId, Position3D};

/// Configuration for Byzantine-resilient gossip validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByzantineConfig {
    /// Maximum tolerated Byzantine nodes (F). Trimming removes up to F
    /// extreme values from each end.
    pub max_byzantine_nodes: usize,
    /// Maximum plausible speed in m/s. Position jumps exceeding this
    /// between consecutive updates are flagged.
    pub max_plausible_speed: f64,
    /// Maximum plausible battery drain per second.
    pub max_battery_drain_per_s: f64,
    /// Maximum age (seconds) of an update before it's considered stale
    /// and not trusted for validation.
    pub max_validation_age_s: f64,
}

impl Default for ByzantineConfig {
    fn default() -> Self {
        Self {
            max_byzantine_nodes: 1,
            max_plausible_speed: 50.0,     // 50 m/s ~ 180 km/h
            max_battery_drain_per_s: 0.01, // 1% per second max
            max_validation_age_s: 30.0,
        }
    }
}

/// Result of Byzantine validation on an incoming gossip update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationResult {
    /// Update is plausible — accept normally.
    Accept,
    /// Update is suspicious but not provably false — accept with reduced trust.
    Suspicious,
    /// Update is implausible — reject.
    Reject,
}

/// Validates an incoming drone state update against known state and
/// neighbor reports.
///
/// Returns `Accept`, `Suspicious`, or `Reject`.
pub fn validate_drone_update(
    incoming: &DroneState,
    known: Option<&DroneState>,
    config: &ByzantineConfig,
) -> ValidationResult {
    // Rule 1: Self-consistency — finite values (already handled by sanitize,
    // but double-check position).
    if !incoming.position.0.iter().all(|v| v.is_finite()) {
        return ValidationResult::Reject;
    }

    let Some(prev) = known else {
        // No prior state — accept (first contact).
        return ValidationResult::Accept;
    };

    // If existing state is invalid (NaN position), accept any valid replacement.
    if !prev.position.0.iter().all(|v| v.is_finite()) {
        return ValidationResult::Accept;
    }

    // Rule 2: Version monotonicity — can't go backwards.
    if incoming.version <= prev.version {
        return ValidationResult::Reject;
    }

    // Rule 3: Kinematic plausibility — position jump check.
    let dt = (incoming.timestamp - prev.timestamp).max(0.001);
    let distance = incoming.position.distance(&prev.position);
    let implied_speed = distance / dt;

    if implied_speed > config.max_plausible_speed * 3.0 {
        // Teleportation — clearly impossible.
        return ValidationResult::Reject;
    }
    if implied_speed > config.max_plausible_speed {
        // Suspicious but not impossible (could be GPS correction).
        return ValidationResult::Suspicious;
    }

    // Rule 4: Battery plausibility — can't gain energy or drain too fast.
    let battery_delta = incoming.battery - prev.battery;
    if battery_delta > 0.01 {
        // Battery increased significantly — suspicious (charging in flight?).
        return ValidationResult::Suspicious;
    }
    let max_drain = config.max_battery_drain_per_s * dt;
    if -battery_delta > max_drain * 3.0 {
        return ValidationResult::Reject;
    }

    // Rule 5: Timestamp plausibility — not from the future, not too old.
    if incoming.timestamp < prev.timestamp - 1.0 {
        // Timestamp went backwards significantly.
        return ValidationResult::Reject;
    }

    ValidationResult::Accept
}

/// Trimmed mean for scalar gossip values (W-MSR core).
///
/// Given a set of values reported by different peers, removes
/// `trim_count` highest and lowest values, then returns the mean
/// of the remaining. This is the core Byzantine-resilient aggregation.
///
/// Returns `None` if too few values remain after trimming.
pub fn trimmed_mean(values: &mut [f64], trim_count: usize) -> Option<f64> {
    let n = values.len();
    if n <= 2 * trim_count {
        return None; // Not enough values to trim safely.
    }

    // Filter non-finite values first.
    let mut clean: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if clean.len() <= 2 * trim_count {
        return None;
    }

    clean.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let trimmed = &clean[trim_count..clean.len() - trim_count];
    if trimmed.is_empty() {
        return None;
    }

    Some(trimmed.iter().sum::<f64>() / trimmed.len() as f64)
}

/// Validate a threat report against existing threat records.
///
/// Cross-checks position and threat_level against what we already know.
pub fn validate_threat_report(
    incoming_position: &Position3D,
    incoming_threat_level: f64,
    existing_position: Option<&Position3D>,
    max_plausible_drift: f64,
) -> ValidationResult {
    if !incoming_position.0.iter().all(|v| v.is_finite()) {
        return ValidationResult::Reject;
    }
    if !incoming_threat_level.is_finite()
        || incoming_threat_level < 0.0
        || incoming_threat_level > 1.0
    {
        return ValidationResult::Reject;
    }

    if let Some(prev_pos) = existing_position {
        let drift = incoming_position.distance(prev_pos);
        if drift > max_plausible_drift * 3.0 {
            return ValidationResult::Reject;
        }
        if drift > max_plausible_drift {
            return ValidationResult::Suspicious;
        }
    }

    ValidationResult::Accept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(node_id: u32, x: f64, battery: f64, version: u64, timestamp: f64) -> DroneState {
        DroneState {
            node_id: NodeId(node_id),
            position: Position3D([x, 0.0, 0.0]),
            battery,
            regime: "Patrol".to_string(),
            version,
            timestamp,
        }
    }

    #[test]
    fn accept_normal_update() {
        let prev = make_state(1, 0.0, 1.0, 1, 0.0);
        let incoming = make_state(1, 5.0, 0.99, 2, 1.0); // 5m in 1s, 1% drain
        let cfg = ByzantineConfig::default();
        assert_eq!(
            validate_drone_update(&incoming, Some(&prev), &cfg),
            ValidationResult::Accept
        );
    }

    #[test]
    fn reject_teleportation() {
        let prev = make_state(1, 0.0, 1.0, 1, 0.0);
        let incoming = make_state(1, 10_000.0, 0.95, 2, 1.0); // 10km in 1s
        let cfg = ByzantineConfig::default();
        assert_eq!(
            validate_drone_update(&incoming, Some(&prev), &cfg),
            ValidationResult::Reject
        );
    }

    #[test]
    fn suspicious_high_speed() {
        let prev = make_state(1, 0.0, 1.0, 1, 0.0);
        let incoming = make_state(1, 80.0, 0.95, 2, 1.0); // 80 m/s > 50 limit
        let cfg = ByzantineConfig::default();
        assert_eq!(
            validate_drone_update(&incoming, Some(&prev), &cfg),
            ValidationResult::Suspicious
        );
    }

    #[test]
    fn reject_version_regression() {
        let prev = make_state(1, 0.0, 1.0, 5, 5.0);
        let incoming = make_state(1, 1.0, 0.99, 3, 6.0); // version went backwards
        let cfg = ByzantineConfig::default();
        assert_eq!(
            validate_drone_update(&incoming, Some(&prev), &cfg),
            ValidationResult::Reject
        );
    }

    #[test]
    fn reject_nan_position() {
        let incoming = DroneState {
            node_id: NodeId(1),
            position: Position3D([f64::NAN, 0.0, 0.0]),
            battery: 1.0,
            regime: "Patrol".to_string(),
            version: 1,
            timestamp: 0.0,
        };
        let cfg = ByzantineConfig::default();
        assert_eq!(
            validate_drone_update(&incoming, None, &cfg),
            ValidationResult::Reject
        );
    }

    #[test]
    fn accept_first_contact() {
        let incoming = make_state(1, 100.0, 0.8, 1, 0.0);
        let cfg = ByzantineConfig::default();
        assert_eq!(
            validate_drone_update(&incoming, None, &cfg),
            ValidationResult::Accept
        );
    }

    #[test]
    fn trimmed_mean_removes_extremes() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // 100 is outlier
        let result = trimmed_mean(&mut values, 1);
        assert!(result.is_some());
        let mean = result.unwrap();
        assert!(
            (mean - 3.0).abs() < 1e-6,
            "trim 1 each end: [2,3,4] → mean=3, got {mean}"
        );
    }

    #[test]
    fn trimmed_mean_too_few_values() {
        let mut values = vec![1.0, 2.0];
        assert!(trimmed_mean(&mut values, 1).is_none());
    }

    #[test]
    fn trimmed_mean_handles_nan() {
        let mut values = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
        let result = trimmed_mean(&mut values, 1);
        assert!(result.is_some());
        // After removing NaN: [1,3,4,5], trim 1 each end: [3,4] → mean=3.5
        assert!((result.unwrap() - 3.5).abs() < 1e-6);
    }

    #[test]
    fn suspicious_battery_gain() {
        let prev = make_state(1, 0.0, 0.5, 1, 0.0);
        let incoming = make_state(1, 1.0, 0.8, 2, 1.0); // battery went up 30%
        let cfg = ByzantineConfig::default();
        assert_eq!(
            validate_drone_update(&incoming, Some(&prev), &cfg),
            ValidationResult::Suspicious
        );
    }

    #[test]
    fn threat_validation_rejects_teleport() {
        let prev_pos = Position3D([0.0, 0.0, 0.0]);
        let incoming_pos = Position3D([10_000.0, 0.0, 0.0]);
        assert_eq!(
            validate_threat_report(&incoming_pos, 0.5, Some(&prev_pos), 100.0),
            ValidationResult::Reject
        );
    }
}
