//! System health monitor for runtime invariant checking.
//!
//! Detects oscillation, convergence stalls, trust collapse, quarantine
//! cascades, regime thrashing, and other system-level pathologies.
//! Advisory only — reports status but does not override decisions.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use strix_core::state::Regime;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for runtime health monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    /// Window size (ticks) for oscillation and trend detection.
    pub window_size: usize,
    /// Criticality/fear sign-change threshold for oscillation detection.
    pub oscillation_sign_changes: usize,
    /// Gossip convergence below this for `stall_ticks` → Degraded.
    pub convergence_stall_threshold: f64,
    /// Ticks of low convergence before declaring stall.
    pub convergence_stall_ticks: usize,
    /// Mean trust below this → Critical.
    pub trust_collapse_threshold: f64,
    /// Fraction of peers quarantined above this → Critical.
    pub quarantine_cascade_threshold: f64,
    /// Regime changes above this count in `window_size` ticks → Degraded.
    pub regime_thrash_threshold: usize,
    /// GBP uncertainty growing monotonically for this many ticks → Degraded.
    pub gbp_monotonic_growth_ticks: usize,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            window_size: 10,
            oscillation_sign_changes: 3,
            convergence_stall_threshold: 0.3,
            convergence_stall_ticks: 20,
            trust_collapse_threshold: 0.3,
            quarantine_cascade_threshold: 0.4,
            regime_thrash_threshold: 5,
            gbp_monotonic_growth_ticks: 15,
        }
    }
}

// ---------------------------------------------------------------------------
// Health status
// ---------------------------------------------------------------------------

/// System health assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    /// All invariants hold.
    Nominal,
    /// One or more soft invariants violated — system is operating sub-optimally.
    Degraded { reasons: Vec<HealthIssue> },
    /// Hard invariant violated — system may be in an unsafe state.
    Critical { reasons: Vec<HealthIssue> },
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self::Nominal
    }
}

impl HealthStatus {
    /// Returns true if the system is in a critical state.
    pub fn is_critical(&self) -> bool {
        matches!(self, Self::Critical { .. })
    }

    /// Returns true if the system is degraded or critical.
    pub fn is_degraded(&self) -> bool {
        !matches!(self, Self::Nominal)
    }
}

/// A single health issue with context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIssue {
    /// What was detected.
    pub kind: HealthIssueKind,
    /// Severity ∈ [0, 1] — 1.0 = most severe.
    pub severity: f64,
    /// Human-readable context.
    pub detail: String,
}

/// Categories of health issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthIssueKind {
    /// Criticality or fear oscillating rapidly.
    Oscillation,
    /// Gossip convergence stuck below threshold.
    ConvergenceStall,
    /// Mean peer trust dropped below threshold.
    TrustCollapse,
    /// Too many peers quarantined simultaneously.
    QuarantineCascade,
    /// A drone's regime is changing too frequently.
    RegimeThrashing,
    /// GBP uncertainty growing without bound.
    GbpDivergence,
}

// ---------------------------------------------------------------------------
// Monitor
// ---------------------------------------------------------------------------

/// Runtime health monitor. Call `check()` every tick with the current
/// `SwarmDecision` to get a health assessment.
#[derive(Debug, Clone)]
pub struct HealthMonitor {
    config: HealthConfig,
    // Rolling windows for trend detection.
    criticality_history: VecDeque<f64>,
    fear_history: VecDeque<f64>,
    convergence_history: VecDeque<f64>,
    gbp_uncertainty_history: VecDeque<f64>,
    // Per-drone regime change counters (drone_id → recent regime changes).
    regime_history: HashMap<u32, VecDeque<Regime>>,
    // Counters
    low_convergence_ticks: usize,
}

impl HealthMonitor {
    /// Create a new health monitor.
    pub fn new(config: HealthConfig) -> Self {
        Self {
            config,
            criticality_history: VecDeque::new(),
            fear_history: VecDeque::new(),
            convergence_history: VecDeque::new(),
            gbp_uncertainty_history: VecDeque::new(),
            regime_history: HashMap::new(),
            low_convergence_ticks: 0,
        }
    }

    /// Check system health against current tick data.
    ///
    /// Returns the aggregate health status and individual issue count.
    #[allow(clippy::too_many_arguments)]
    pub fn check(
        &mut self,
        fear_level: f64,
        criticality: f64,
        gossip_convergence: f64,
        trust_mean: f64,
        quarantine_fraction: f64,
        regimes: &HashMap<u32, Regime>,
        gbp_uncertainty: Option<f64>,
    ) -> HealthStatus {
        let w = self.config.window_size;

        // Record history.
        push_bounded(&mut self.criticality_history, criticality, w);
        push_bounded(&mut self.fear_history, fear_level, w);
        push_bounded(&mut self.convergence_history, gossip_convergence, w);
        if let Some(u) = gbp_uncertainty {
            push_bounded(
                &mut self.gbp_uncertainty_history,
                u,
                self.config.gbp_monotonic_growth_ticks + 1,
            );
        }

        // Record regime changes per drone; prune entries for dead drones.
        self.regime_history.retain(|id, _| regimes.contains_key(id));
        for (&drone_id, &regime) in regimes {
            let history = self.regime_history.entry(drone_id).or_default();
            push_bounded(history, regime, w);
        }

        // Track low convergence streak.
        if gossip_convergence < self.config.convergence_stall_threshold {
            self.low_convergence_ticks += 1;
        } else {
            self.low_convergence_ticks = 0;
        }

        // Collect issues.
        let mut degraded = Vec::new();
        let mut critical = Vec::new();

        // 1. Oscillation detection.
        if let Some(issue) = self.check_oscillation(&self.criticality_history, "criticality") {
            degraded.push(issue);
        }
        if let Some(issue) = self.check_oscillation(&self.fear_history, "fear") {
            degraded.push(issue);
        }

        // 2. Convergence stall.
        if self.low_convergence_ticks >= self.config.convergence_stall_ticks {
            degraded.push(HealthIssue {
                kind: HealthIssueKind::ConvergenceStall,
                severity: (self.low_convergence_ticks as f64
                    / (self.config.convergence_stall_ticks as f64 * 2.0))
                    .min(1.0),
                detail: format!(
                    "gossip convergence < {:.2} for {} ticks",
                    self.config.convergence_stall_threshold, self.low_convergence_ticks
                ),
            });
        }

        // 3. Trust collapse.
        if trust_mean < self.config.trust_collapse_threshold && trust_mean > 0.0 {
            critical.push(HealthIssue {
                kind: HealthIssueKind::TrustCollapse,
                severity: 1.0 - (trust_mean / self.config.trust_collapse_threshold),
                detail: format!(
                    "mean trust {trust_mean:.3} < threshold {:.3}",
                    self.config.trust_collapse_threshold
                ),
            });
        }

        // 4. Quarantine cascade.
        if quarantine_fraction > self.config.quarantine_cascade_threshold {
            critical.push(HealthIssue {
                kind: HealthIssueKind::QuarantineCascade,
                severity: ((quarantine_fraction - self.config.quarantine_cascade_threshold)
                    / (1.0 - self.config.quarantine_cascade_threshold))
                    .min(1.0),
                detail: format!(
                    "{:.0}% of peers quarantined (threshold: {:.0}%)",
                    quarantine_fraction * 100.0,
                    self.config.quarantine_cascade_threshold * 100.0
                ),
            });
        }

        // 5. Regime thrashing.
        for (&drone_id, history) in &self.regime_history {
            let changes = count_changes(history);
            if changes >= self.config.regime_thrash_threshold {
                degraded.push(HealthIssue {
                    kind: HealthIssueKind::RegimeThrashing,
                    severity: (changes as f64 / (self.config.regime_thrash_threshold as f64 * 2.0))
                        .min(1.0),
                    detail: format!("drone {drone_id}: {changes} regime changes in {w} ticks"),
                });
            }
        }

        // 6. GBP divergence.
        if self.gbp_uncertainty_history.len() >= self.config.gbp_monotonic_growth_ticks
            && is_monotonically_increasing(&self.gbp_uncertainty_history)
        {
            degraded.push(HealthIssue {
                kind: HealthIssueKind::GbpDivergence,
                severity: 0.7,
                detail: format!(
                    "GBP uncertainty grew monotonically for {} ticks",
                    self.gbp_uncertainty_history.len()
                ),
            });
        }

        // Build aggregate status.
        if !critical.is_empty() {
            HealthStatus::Critical { reasons: critical }
        } else if !degraded.is_empty() {
            HealthStatus::Degraded { reasons: degraded }
        } else {
            HealthStatus::Nominal
        }
    }

    fn check_oscillation(&self, history: &VecDeque<f64>, name: &str) -> Option<HealthIssue> {
        if history.len() < 3 {
            return None;
        }
        let changes = sign_changes(history);
        if changes >= self.config.oscillation_sign_changes {
            Some(HealthIssue {
                kind: HealthIssueKind::Oscillation,
                severity: (changes as f64 / (self.config.oscillation_sign_changes as f64 * 2.0))
                    .min(1.0),
                detail: format!("{name}: {changes} sign changes in {} ticks", history.len()),
            })
        } else {
            None
        }
    }

    /// Number of oscillation events detected so far.
    pub fn oscillation_count(&self) -> u32 {
        let mut count = 0;
        if sign_changes(&self.criticality_history) >= self.config.oscillation_sign_changes {
            count += 1;
        }
        if sign_changes(&self.fear_history) >= self.config.oscillation_sign_changes {
            count += 1;
        }
        count
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn push_bounded<T>(deque: &mut VecDeque<T>, value: T, max_len: usize) {
    deque.push_back(value);
    while deque.len() > max_len {
        deque.pop_front();
    }
}

/// Count sign changes in a sequence of deltas (positive/negative transitions).
/// Zero-allocation single-pass implementation.
fn sign_changes(history: &VecDeque<f64>) -> usize {
    if history.len() < 3 {
        return 0;
    }
    let mut count = 0;
    let mut prev_delta = 0.0_f64;
    let mut iter = history.iter();
    let mut a = iter.next().unwrap();
    for b in iter {
        let delta = b - a;
        if prev_delta.abs() > 1e-6 && delta.abs() > 1e-6 && (prev_delta > 0.0) != (delta > 0.0) {
            count += 1;
        }
        prev_delta = delta;
        a = b;
    }
    count
}

/// Count regime changes in a history.
fn count_changes<T: PartialEq>(history: &VecDeque<T>) -> usize {
    history
        .iter()
        .zip(history.iter().skip(1))
        .filter(|(a, b)| a != b)
        .count()
}

/// Check if values are monotonically increasing (all deltas > 0).
fn is_monotonically_increasing(history: &VecDeque<f64>) -> bool {
    if history.len() < 2 {
        return false;
    }
    history
        .iter()
        .zip(history.iter().skip(1))
        .all(|(a, b)| b > a)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_when_all_healthy() {
        let mut monitor = HealthMonitor::new(HealthConfig::default());
        let regimes = HashMap::from([(0, Regime::Patrol), (1, Regime::Patrol)]);
        let status = monitor.check(0.3, 0.5, 0.8, 0.7, 0.0, &regimes, Some(1.0));
        assert!(matches!(status, HealthStatus::Nominal));
    }

    #[test]
    fn detects_trust_collapse() {
        let mut monitor = HealthMonitor::new(HealthConfig::default());
        let regimes = HashMap::new();
        let status = monitor.check(0.3, 0.5, 0.8, 0.15, 0.0, &regimes, None);
        assert!(status.is_critical());
    }

    #[test]
    fn detects_quarantine_cascade() {
        let mut monitor = HealthMonitor::new(HealthConfig::default());
        let regimes = HashMap::new();
        let status = monitor.check(0.3, 0.5, 0.8, 0.7, 0.6, &regimes, None);
        assert!(status.is_critical());
    }

    #[test]
    fn detects_convergence_stall() {
        let mut monitor = HealthMonitor::new(HealthConfig {
            convergence_stall_ticks: 5,
            ..HealthConfig::default()
        });
        let regimes = HashMap::new();
        for _ in 0..6 {
            monitor.check(0.3, 0.5, 0.1, 0.7, 0.0, &regimes, None);
        }
        let status = monitor.check(0.3, 0.5, 0.1, 0.7, 0.0, &regimes, None);
        assert!(status.is_degraded());
    }

    #[test]
    fn detects_oscillation() {
        let mut monitor = HealthMonitor::new(HealthConfig {
            window_size: 10,
            oscillation_sign_changes: 2,
            ..HealthConfig::default()
        });
        let regimes = HashMap::new();
        // Feed oscillating criticality: 0.3, 0.7, 0.3, 0.7, 0.3, 0.7
        for i in 0..8 {
            let crit = if i % 2 == 0 { 0.3 } else { 0.7 };
            monitor.check(0.3, crit, 0.8, 0.7, 0.0, &regimes, None);
        }
        let status = monitor.check(0.3, 0.3, 0.8, 0.7, 0.0, &regimes, None);
        assert!(status.is_degraded());
    }

    #[test]
    fn detects_regime_thrashing() {
        let mut monitor = HealthMonitor::new(HealthConfig {
            window_size: 10,
            regime_thrash_threshold: 3,
            ..HealthConfig::default()
        });
        // Drone 0 flips between Patrol and Engage every tick.
        for i in 0..8 {
            let regime = if i % 2 == 0 {
                Regime::Patrol
            } else {
                Regime::Engage
            };
            let regimes = HashMap::from([(0, regime)]);
            monitor.check(0.3, 0.5, 0.8, 0.7, 0.0, &regimes, None);
        }
        let regimes = HashMap::from([(0, Regime::Patrol)]);
        let status = monitor.check(0.3, 0.5, 0.8, 0.7, 0.0, &regimes, None);
        assert!(status.is_degraded());
    }

    #[test]
    fn detects_gbp_divergence() {
        let mut monitor = HealthMonitor::new(HealthConfig {
            gbp_monotonic_growth_ticks: 5,
            ..HealthConfig::default()
        });
        let regimes = HashMap::new();
        for i in 0..7 {
            monitor.check(
                0.3,
                0.5,
                0.8,
                0.7,
                0.0,
                &regimes,
                Some(1.0 + i as f64 * 0.1),
            );
        }
        let status = monitor.check(0.3, 0.5, 0.8, 0.7, 0.0, &regimes, Some(2.0));
        assert!(status.is_degraded());
    }

    #[test]
    fn sign_changes_counts_correctly() {
        // 0.3, 0.7, 0.3, 0.7 → deltas: +0.4, -0.4, +0.4 → 2 sign changes
        let mut history = VecDeque::new();
        for &v in &[0.3, 0.7, 0.3, 0.7] {
            history.push_back(v);
        }
        assert_eq!(sign_changes(&history), 2);
    }

    #[test]
    fn monotonic_detection() {
        let mut history = VecDeque::from(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(is_monotonically_increasing(&history));
        history.push_back(3.5); // breaks monotonicity
        assert!(!is_monotonically_increasing(&history));
    }
}
