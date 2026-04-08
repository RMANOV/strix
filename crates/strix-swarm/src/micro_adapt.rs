//! Runtime micro-adaptation: adjusts SwarmConfig intervals based on
//! order parameters and criticality, within bounded ranges.
//!
//! NOT a replacement for the offline optimizer — just reactive guardrails
//! that nudge gossip, formation, and observation frequencies based on
//! macroscopic swarm health metrics.
//!
//! Safety invariant: NEVER touches particle filter, CBF, ROE, or auction.
//! Only gossip fanout and multi-timescale intervals are adjustable.

use serde::{Deserialize, Serialize};

use crate::order_params::OrderParameters;
use crate::CriticalityAdjustment;

/// Configuration for runtime micro-adaptation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroAdaptConfig {
    /// Enable/disable runtime adaptation. Default: false.
    pub enabled: bool,
    /// Minimum ticks between adaptations. Default: 10.
    pub adapt_interval: u32,
    /// EMA smoothing factor for order parameters. Default: 0.15.
    pub smoothing_alpha: f64,
    /// Fragmentation threshold above which formation_interval is tightened.
    pub fragmentation_high: f64,
    /// Alignment threshold below which gossip_fanout is increased.
    pub alignment_low: f64,
    /// Trust entropy threshold above which gossip frequency increases.
    pub entropy_high: f64,
    /// Maximum multiplier on any config parameter.
    pub max_multiplier: f64,
    /// Minimum multiplier.
    pub min_multiplier: f64,
}

impl Default for MicroAdaptConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            adapt_interval: 10,
            smoothing_alpha: 0.15,
            fragmentation_high: 0.7,
            alignment_low: 0.3,
            entropy_high: 0.8,
            max_multiplier: 2.0,
            min_multiplier: 0.5,
        }
    }
}

/// Nudges to apply to SwarmConfig intervals. All multipliers default to 1.0 (no change).
#[derive(Debug, Clone, Copy)]
pub struct ConfigNudge {
    /// Multiplier on gossip_fanout (>1 = more gossip).
    pub gossip_fanout_multiplier: f64,
    /// Divisor on formation_interval (>1 = more frequent formation updates).
    pub formation_interval_divisor: f64,
    /// Divisor on gossip_interval (>1 = more frequent gossip).
    pub gossip_interval_divisor: f64,
}

impl ConfigNudge {
    /// Neutral nudge — no change.
    pub fn neutral() -> Self {
        Self {
            gossip_fanout_multiplier: 1.0,
            formation_interval_divisor: 1.0,
            gossip_interval_divisor: 1.0,
        }
    }

    /// Clamp all multipliers to [min, max].
    pub fn clamp(&mut self, min: f64, max: f64) {
        self.gossip_fanout_multiplier = self.gossip_fanout_multiplier.clamp(min, max);
        self.formation_interval_divisor = self.formation_interval_divisor.clamp(min, max);
        self.gossip_interval_divisor = self.gossip_interval_divisor.clamp(min, max);
    }
}

/// Runtime micro-adapter. Call `adapt()` each tick with the previous tick's
/// order parameters and criticality. Returns nudges to apply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroAdapter {
    config: MicroAdaptConfig,
    smoothed: OrderParameters,
    ticks_since_adapt: u32,
}

impl MicroAdapter {
    pub fn new(config: MicroAdaptConfig) -> Self {
        Self {
            config,
            smoothed: OrderParameters::default(),
            ticks_since_adapt: 0,
        }
    }

    /// Produce config nudges based on EMA-smoothed order parameters.
    ///
    /// Uses `last_order_params` and `last_criticality` from the previous tick
    /// (1-tick latency, acceptable for macro-level controller).
    pub fn adapt(&mut self, op: &OrderParameters, _crit: &CriticalityAdjustment) -> ConfigNudge {
        if !self.config.enabled {
            return ConfigNudge::neutral();
        }

        // EMA smooth order parameters.
        let a = self.config.smoothing_alpha.clamp(0.0, 1.0);
        self.smoothed.alignment_order =
            (1.0 - a) * self.smoothed.alignment_order + a * op.alignment_order;
        self.smoothed.fragmentation_index =
            (1.0 - a) * self.smoothed.fragmentation_index + a * op.fragmentation_index;
        self.smoothed.trust_entropy =
            (1.0 - a) * self.smoothed.trust_entropy + a * op.trust_entropy;
        self.smoothed.coverage_dispersion =
            (1.0 - a) * self.smoothed.coverage_dispersion + a * op.coverage_dispersion;
        self.smoothed.mission_progress =
            (1.0 - a) * self.smoothed.mission_progress + a * op.mission_progress;

        self.ticks_since_adapt += 1;
        if self.ticks_since_adapt < self.config.adapt_interval {
            return ConfigNudge::neutral();
        }
        self.ticks_since_adapt = 0;

        let mut nudge = ConfigNudge::neutral();

        // Rule 1: High fragmentation → tighten formation (run more often).
        if self.smoothed.fragmentation_index > self.config.fragmentation_high {
            let excess = (self.smoothed.fragmentation_index - self.config.fragmentation_high)
                / (1.0 - self.config.fragmentation_high).max(0.01);
            nudge.formation_interval_divisor = 1.0 + excess.clamp(0.0, 1.0);
        }

        // Rule 2: Low alignment → increase gossip fanout.
        if self.smoothed.alignment_order < self.config.alignment_low {
            let deficit = (self.config.alignment_low - self.smoothed.alignment_order)
                / self.config.alignment_low.max(0.01);
            nudge.gossip_fanout_multiplier = 1.0 + deficit.clamp(0.0, 1.0) * 0.5;
        }

        // Rule 3: High trust entropy → increase gossip frequency.
        if self.smoothed.trust_entropy > self.config.entropy_high {
            nudge.gossip_interval_divisor = 1.5;
        }

        nudge.clamp(self.config.min_multiplier, self.config.max_multiplier);
        nudge
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_returns_neutral() {
        let mut adapter = MicroAdapter::new(MicroAdaptConfig::default());
        let op = OrderParameters::default();
        let crit = CriticalityAdjustment::default();
        let nudge = adapter.adapt(&op, &crit);
        assert!((nudge.gossip_fanout_multiplier - 1.0).abs() < 1e-6);
        assert!((nudge.formation_interval_divisor - 1.0).abs() < 1e-6);
    }

    #[test]
    fn high_fragmentation_tightens_formation() {
        let mut adapter = MicroAdapter::new(MicroAdaptConfig {
            enabled: true,
            adapt_interval: 1,
            fragmentation_high: 0.5,
            ..MicroAdaptConfig::default()
        });

        let op = OrderParameters {
            fragmentation_index: 0.9,
            ..OrderParameters::default()
        };
        let crit = CriticalityAdjustment::default();

        // Warm up EMA
        for _ in 0..20 {
            adapter.adapt(&op, &crit);
        }
        let nudge = adapter.adapt(&op, &crit);
        assert!(
            nudge.formation_interval_divisor > 1.0,
            "high fragmentation should tighten formation, got {}",
            nudge.formation_interval_divisor
        );
    }

    #[test]
    fn low_alignment_increases_fanout() {
        let mut adapter = MicroAdapter::new(MicroAdaptConfig {
            enabled: true,
            adapt_interval: 1,
            alignment_low: 0.5,
            ..MicroAdaptConfig::default()
        });

        let op = OrderParameters {
            alignment_order: 0.1,
            ..OrderParameters::default()
        };
        let crit = CriticalityAdjustment::default();

        for _ in 0..20 {
            adapter.adapt(&op, &crit);
        }
        let nudge = adapter.adapt(&op, &crit);
        assert!(
            nudge.gossip_fanout_multiplier > 1.0,
            "low alignment should increase fanout, got {}",
            nudge.gossip_fanout_multiplier
        );
    }

    #[test]
    fn nudge_clamps_within_bounds() {
        let mut adapter = MicroAdapter::new(MicroAdaptConfig {
            enabled: true,
            adapt_interval: 1,
            max_multiplier: 1.5,
            min_multiplier: 0.8,
            fragmentation_high: 0.1,
            alignment_low: 0.99,
            ..MicroAdaptConfig::default()
        });

        let op = OrderParameters {
            fragmentation_index: 1.0,
            alignment_order: 0.0,
            trust_entropy: 1.0,
            ..OrderParameters::default()
        };
        let crit = CriticalityAdjustment::default();

        for _ in 0..30 {
            adapter.adapt(&op, &crit);
        }
        let nudge = adapter.adapt(&op, &crit);
        assert!(nudge.gossip_fanout_multiplier <= 1.5 + 1e-6);
        assert!(nudge.formation_interval_divisor <= 1.5 + 1e-6);
        assert!(nudge.gossip_interval_divisor >= 0.8 - 1e-6);
    }

    #[test]
    fn ema_smoothing_prevents_jitter() {
        let mut adapter = MicroAdapter::new(MicroAdaptConfig {
            enabled: true,
            adapt_interval: 1,
            smoothing_alpha: 0.1,
            fragmentation_high: 0.5,
            ..MicroAdaptConfig::default()
        });
        let crit = CriticalityAdjustment::default();

        // Alternate between extreme values
        let mut nudges = Vec::new();
        for i in 0..40 {
            let frag = if i % 2 == 0 { 1.0 } else { 0.0 };
            let op = OrderParameters {
                fragmentation_index: frag,
                ..OrderParameters::default()
            };
            nudges.push(adapter.adapt(&op, &crit));
        }

        // Smoothed output should converge to ~0.5 fragmentation (neutral)
        let last = nudges.last().unwrap();
        // With alpha=0.1, alternating 0/1 → smoothed ≈ 0.5 (neutral territory)
        assert!(
            last.formation_interval_divisor < 1.3,
            "EMA should smooth jitter, got {}",
            last.formation_interval_divisor
        );
    }
}
