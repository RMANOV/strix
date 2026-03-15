//! Attrition Risk Management — drawdown protection from trading.
//!
//! Tracks drone losses over time and enforces escalating risk responses:
//!
//! | Attrition | Level     | Response                       |
//! |-----------|-----------|--------------------------------|
//! | < 10 %    | Normal    | Continue operations            |
//! | 10 – 20 % | Cautious  | Increase standoff, tighten bids |
//! | 20 – 30 % | Defensive | Consolidate, avoid high-risk   |
//! | 30 – 50 % | Retreat   | Disengage, regroup             |
//! | > 50 %    | Survival  | Full evasion, preserve assets   |
//!
//! Concepts borrowed directly from trading:
//! - **MaxDrawdown**: peak-to-trough loss — triggers strategic retreat.
//! - **ValueAtRisk**: probabilistic loss estimate for a mission plan.
//! - **Risk budget**: per-drone expendability based on remaining fleet size.

use serde::{Deserialize, Serialize};

use crate::Regime;

// ────────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────────

/// Escalating risk levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    /// < 10 % attrition.
    Normal,
    /// 10 – 20 %.
    Cautious,
    /// 20 – 30 %.
    Defensive,
    /// 30 – 50 %.
    Retreat,
    /// > 50 %.
    Survival,
}

impl RiskLevel {
    /// Map an attrition rate (0.0 – 1.0) to a risk level.
    pub fn from_attrition(rate: f64) -> Self {
        match rate {
            r if r < 0.10 => RiskLevel::Normal,
            r if r < 0.20 => RiskLevel::Cautious,
            r if r < 0.30 => RiskLevel::Defensive,
            r if r < 0.50 => RiskLevel::Retreat,
            _ => RiskLevel::Survival,
        }
    }

    /// Fear-adjusted attrition check: F shifts effective attrition up.
    ///
    /// At F=1.0, retreat triggers at ~15% attrition (not 30%) and
    /// survival at ~35% (not 50%). A fearful swarm preserves assets earlier.
    pub fn from_attrition_with_fear(rate: f64, fear: f64) -> Self {
        let f = fear.clamp(0.0, 1.0);
        Self::from_attrition(rate + f * 0.15)
    }

    /// Suggested regime for the fleet at this risk level.
    pub fn suggested_regime(&self) -> Regime {
        match self {
            RiskLevel::Normal => Regime::Patrol,
            RiskLevel::Cautious => Regime::Patrol,
            RiskLevel::Defensive => Regime::Patrol,
            RiskLevel::Retreat => Regime::Evade,
            RiskLevel::Survival => Regime::Evade,
        }
    }
}

/// Maximum drawdown tracker — analogous to trading drawdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxDrawdown {
    /// Historical peak fleet size.
    pub peak_fleet_size: u32,
    /// Current fleet size.
    pub current_fleet_size: u32,
    /// Maximum acceptable drawdown before strategic retreat [0, 1].
    pub threshold: f64,
}

impl MaxDrawdown {
    /// Create a new drawdown tracker.
    pub fn new(initial_fleet_size: u32, threshold: f64) -> Self {
        Self {
            peak_fleet_size: initial_fleet_size,
            current_fleet_size: initial_fleet_size,
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Update with a new fleet size reading.
    pub fn update(&mut self, current_size: u32) {
        if current_size > self.peak_fleet_size {
            self.peak_fleet_size = current_size;
        }
        self.current_fleet_size = current_size;
    }

    /// Current drawdown as a fraction of peak.
    pub fn drawdown(&self) -> f64 {
        if self.peak_fleet_size == 0 {
            return 0.0;
        }
        1.0 - (self.current_fleet_size as f64 / self.peak_fleet_size as f64)
    }

    /// Has the drawdown exceeded the threshold?
    pub fn is_breached(&self) -> bool {
        self.drawdown() >= self.threshold
    }
}

/// Probabilistic loss estimation for a mission plan.
///
/// Uses a simplified parametric model: given the number of drones committed,
/// threat density, and mission duration, estimate the probability distribution
/// of losses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueAtRisk {
    /// Confidence level (e.g. 0.95 for 95th percentile).
    pub confidence: f64,
    /// Estimated loss rate per unit time per threat (simple Poisson-like parameter).
    pub lambda: f64,
}

impl ValueAtRisk {
    /// Create a new VaR estimator.
    ///
    /// - `confidence`: quantile level, e.g. 0.95.
    /// - `lambda`: loss rate per drone per unit time per unit threat density.
    pub fn new(confidence: f64, lambda: f64) -> Self {
        Self {
            confidence: confidence.clamp(0.0, 1.0),
            lambda: lambda.max(0.0),
        }
    }

    /// Estimate worst-case losses (at the configured confidence level).
    ///
    /// # Arguments
    /// - `drones_committed`: number of drones in the mission.
    /// - `threat_density`: normalised threat count / area.
    /// - `mission_duration`: mission time in arbitrary units.
    ///
    /// Returns the estimated number of drones that could be lost.
    pub fn estimate(
        &self,
        drones_committed: u32,
        threat_density: f64,
        mission_duration: f64,
    ) -> f64 {
        // Expected losses: Poisson mean = lambda * n * d * t
        let mean = self.lambda * drones_committed as f64 * threat_density * mission_duration;

        // For Poisson distribution, the quantile at high confidence is approximately:
        //   Q(p) ≈ mean + z * sqrt(mean)
        // where z is the standard normal quantile.
        let z = standard_normal_quantile(self.confidence);
        let var = mean + z * mean.sqrt();

        var.max(0.0).min(drones_committed as f64)
    }
}

/// Attrition monitor — the main risk management interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttritionMonitor {
    /// Initial fleet size at mission start.
    pub initial_fleet_size: u32,
    /// Number of drones currently alive.
    pub alive_count: u32,
    /// Cumulative losses.
    pub total_losses: u32,
    /// Drawdown tracker.
    pub max_drawdown: MaxDrawdown,
    /// Per-drone risk budget: fraction of fleet that can be risked on any single task.
    pub per_drone_risk_budget: f64,
    /// Historical attrition readings (for trend analysis).
    pub history: Vec<f64>,
}

impl AttritionMonitor {
    /// Create a new monitor for a fleet.
    pub fn new(fleet_size: u32, drawdown_threshold: f64) -> Self {
        Self {
            initial_fleet_size: fleet_size,
            alive_count: fleet_size,
            total_losses: 0,
            max_drawdown: MaxDrawdown::new(fleet_size, drawdown_threshold),
            per_drone_risk_budget: 1.0 / fleet_size.max(1) as f64,
            history: vec![0.0],
        }
    }

    /// Record the loss of `count` drones.
    pub fn record_losses(&mut self, count: u32) {
        self.total_losses += count;
        self.alive_count = self.alive_count.saturating_sub(count);
        self.max_drawdown.update(self.alive_count);
        self.history.push(self.current_attrition_rate());
    }

    /// Record that drones have been added (reinforcements / returned from repair).
    pub fn record_reinforcements(&mut self, count: u32) {
        self.alive_count += count;
        self.max_drawdown.update(self.alive_count);
    }

    /// Current attrition rate: total losses / initial fleet size.
    pub fn current_attrition_rate(&self) -> f64 {
        if self.initial_fleet_size == 0 {
            return 0.0;
        }
        self.total_losses as f64 / self.initial_fleet_size as f64
    }

    /// Current risk level based on attrition rate.
    pub fn risk_level(&self) -> RiskLevel {
        RiskLevel::from_attrition(self.current_attrition_rate())
    }

    /// Drawdown check: has the peak-to-trough loss exceeded the threshold?
    ///
    /// If `true`, the entire fleet should transition to EVADE regime.
    pub fn drawdown_check(&self) -> bool {
        self.max_drawdown.is_breached()
    }

    /// Per-drone risk budget: what fraction of the fleet can be risked on one task.
    ///
    /// Decreases as the fleet shrinks (remaining assets become more precious).
    pub fn risk_budget(&self) -> f64 {
        if self.alive_count == 0 {
            return 0.0;
        }
        // As fleet shrinks, each drone's share of the budget grows but the
        // absolute budget (drones we can afford to lose) shrinks.
        // Budget = 1 / alive_count — i.e. losing one drone is (1/N) of remaining fleet.
        1.0 / self.alive_count as f64
    }

    /// Attrition trend: positive means losses are accelerating.
    pub fn attrition_trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let n = self.history.len();
        self.history[n - 1] - self.history[n - 2]
    }
}

// ────────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────────

/// Approximate inverse of the standard normal CDF (Abramowitz & Stegun rational approx.).
///
/// Valid for `p` in (0, 1). Accuracy ~4.5e-4.
fn standard_normal_quantile(p: f64) -> f64 {
    // Clamp to avoid infinities.
    let p = p.clamp(1e-10, 1.0 - 1e-10);

    // Rational approximation (Abramowitz & Stegun 26.2.23).
    let sign = if p < 0.5 { -1.0 } else { 1.0 };
    let p_adj = if p < 0.5 { p } else { 1.0 - p };

    let t = (-2.0 * p_adj.ln()).sqrt();
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    sign * z
}

// ────────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_level_thresholds() {
        assert_eq!(RiskLevel::from_attrition(0.0), RiskLevel::Normal);
        assert_eq!(RiskLevel::from_attrition(0.05), RiskLevel::Normal);
        assert_eq!(RiskLevel::from_attrition(0.10), RiskLevel::Cautious);
        assert_eq!(RiskLevel::from_attrition(0.15), RiskLevel::Cautious);
        assert_eq!(RiskLevel::from_attrition(0.20), RiskLevel::Defensive);
        assert_eq!(RiskLevel::from_attrition(0.30), RiskLevel::Retreat);
        assert_eq!(RiskLevel::from_attrition(0.50), RiskLevel::Survival);
        assert_eq!(RiskLevel::from_attrition(0.80), RiskLevel::Survival);
    }

    #[test]
    fn test_suggested_regime() {
        assert_eq!(RiskLevel::Normal.suggested_regime(), Regime::Patrol);
        assert_eq!(RiskLevel::Retreat.suggested_regime(), Regime::Evade);
        assert_eq!(RiskLevel::Survival.suggested_regime(), Regime::Evade);
    }

    #[test]
    fn test_max_drawdown() {
        let mut dd = MaxDrawdown::new(20, 0.25);
        assert!((dd.drawdown()).abs() < 1e-12);
        assert!(!dd.is_breached());

        dd.update(15); // 25% drawdown
        assert!((dd.drawdown() - 0.25).abs() < 1e-12);
        assert!(dd.is_breached());

        dd.update(18); // recovery — but peak was 20 so drawdown is 10%.
        assert!((dd.drawdown() - 0.10).abs() < 1e-12);
        assert!(!dd.is_breached());
    }

    #[test]
    fn test_max_drawdown_peak_update() {
        let mut dd = MaxDrawdown::new(10, 0.5);
        dd.update(15); // new peak
        assert_eq!(dd.peak_fleet_size, 15);
        dd.update(10);
        assert!((dd.drawdown() - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_attrition_monitor_basic() {
        let mut mon = AttritionMonitor::new(20, 0.3);
        assert_eq!(mon.risk_level(), RiskLevel::Normal);
        assert!(!mon.drawdown_check());

        mon.record_losses(2); // 10%
        assert_eq!(mon.risk_level(), RiskLevel::Cautious);
        assert_eq!(mon.alive_count, 18);

        mon.record_losses(2); // 20%
        assert_eq!(mon.risk_level(), RiskLevel::Defensive);

        mon.record_losses(2); // 30%
        assert_eq!(mon.risk_level(), RiskLevel::Retreat);
        assert!(mon.drawdown_check()); // 30% >= threshold 30%
    }

    #[test]
    fn test_attrition_rate() {
        let mut mon = AttritionMonitor::new(10, 0.5);
        mon.record_losses(3);
        assert!((mon.current_attrition_rate() - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_risk_budget_shrinks() {
        let mut mon = AttritionMonitor::new(10, 0.5);
        let budget_full = mon.risk_budget();
        mon.record_losses(5);
        let budget_half = mon.risk_budget();
        assert!(
            budget_half > budget_full,
            "per-drone budget should increase as fleet shrinks (each drone is more precious)"
        );
    }

    #[test]
    fn test_reinforcements() {
        let mut mon = AttritionMonitor::new(10, 0.5);
        mon.record_losses(5);
        assert_eq!(mon.alive_count, 5);
        mon.record_reinforcements(3);
        assert_eq!(mon.alive_count, 8);
    }

    #[test]
    fn test_attrition_trend() {
        let mut mon = AttritionMonitor::new(20, 0.5);
        assert!((mon.attrition_trend()).abs() < 1e-12);
        mon.record_losses(2); // 10%
        assert!(mon.attrition_trend() > 0.0);
        mon.record_losses(2); // 20%
        assert!(mon.attrition_trend() > 0.0);
    }

    #[test]
    fn test_value_at_risk() {
        let var = ValueAtRisk::new(0.95, 0.01);
        let loss = var.estimate(10, 0.5, 2.0);
        // Expected mean = 0.01 * 10 * 0.5 * 2.0 = 0.1
        // Should be some small positive number above 0.1.
        assert!(loss > 0.0, "VaR should be positive, got {loss}");
        assert!(
            loss <= 10.0,
            "VaR can't exceed committed drones, got {loss}"
        );
    }

    #[test]
    fn test_value_at_risk_zero_threat() {
        let var = ValueAtRisk::new(0.95, 0.01);
        let loss = var.estimate(10, 0.0, 2.0);
        assert!(
            (loss).abs() < 1e-6,
            "no threat should mean ~zero VaR, got {loss}"
        );
    }

    #[test]
    fn test_standard_normal_quantile() {
        // z(0.5) should be ~0.
        let z50 = standard_normal_quantile(0.5);
        assert!(z50.abs() < 0.01, "z(0.5) = {z50}");

        // z(0.975) should be ~1.96.
        let z975 = standard_normal_quantile(0.975);
        assert!((z975 - 1.96).abs() < 0.01, "z(0.975) = {z975}");

        // z(0.025) should be ~-1.96.
        let z025 = standard_normal_quantile(0.025);
        assert!((z025 + 1.96).abs() < 0.01, "z(0.025) = {z025}");
    }

    #[test]
    fn test_fear_adjusted_risk_level() {
        // At F=0, 15% attrition = Cautious
        assert_eq!(
            RiskLevel::from_attrition_with_fear(0.15, 0.0),
            RiskLevel::Cautious
        );

        // At F=1, 15% attrition + 15% fear shift = 30% effective → Retreat
        assert_eq!(
            RiskLevel::from_attrition_with_fear(0.15, 1.0),
            RiskLevel::Retreat
        );

        // At F=0.5, 20% attrition + 7.5% = 27.5% → Defensive
        assert_eq!(
            RiskLevel::from_attrition_with_fear(0.20, 0.5),
            RiskLevel::Defensive
        );

        // At F=1, 35% + 15% = 50% → Survival
        assert_eq!(
            RiskLevel::from_attrition_with_fear(0.35, 1.0),
            RiskLevel::Survival
        );

        // Fear should be clamped - F=2.0 should behave like F=1.0
        assert_eq!(
            RiskLevel::from_attrition_with_fear(0.15, 2.0),
            RiskLevel::from_attrition_with_fear(0.15, 1.0)
        );
    }
}
