//! Fear Adapter — bridges STRIX swarm telemetry to PhiSim's fear meta-parameter.
//!
//! PhiSim computes F ∈ [0,1] from a `FearState` that maps financial concepts
//! to drone warfare:
//!
//! | PhiSim concept       | STRIX signal                        |
//! |----------------------|-------------------------------------|
//! | `drawdown`           | Attrition rate (1 - alive/initial)  |
//! | `vol_ratio`          | Threat intensity (1 + intent score) |
//! | `anomaly_count`      | CUSUM breaks this tick              |
//! | `consecutive_losses` | Consecutive ticks with drone loss   |
//!
//! The adapter is **advisory**: STRIX keeps its own PF, regimes, and auction.
//! F only modulates parameters — it never overrides decisions.

#[cfg(feature = "phi-sim")]
use phi_sim::config::Config;
#[cfg(feature = "phi-sim")]
use phi_sim::types::FearState;
#[cfg(feature = "phi-sim")]
use phi_sim::PhiSim;

/// Adapts STRIX telemetry into PhiSim's fear framework.
#[cfg(feature = "phi-sim")]
pub struct FearAdapter {
    phi_sim: PhiSim,
    /// Consecutive ticks where at least one drone was lost.
    consecutive_loss_ticks: u32,
    /// Loss count at previous tick (for streak detection).
    prev_loss_count: usize,
}

#[cfg(feature = "phi-sim")]
impl FearAdapter {
    /// Create a new adapter with PhiSim's drone swarm preset.
    pub fn new() -> Self {
        Self {
            phi_sim: PhiSim::new(Config::drone_swarm()),
            consecutive_loss_ticks: 0,
            prev_loss_count: 0,
        }
    }

    /// Compute fear from swarm-wide telemetry.
    ///
    /// Call at the start of each tick. Returns F ∈ [0,1].
    ///
    /// # Arguments
    /// * `alive` — number of drones currently alive
    /// * `total_losses` — cumulative drone losses since mission start
    /// * `max_intent_score` — peak threat intent from previous tick [-1, 1]
    /// * `cusum_breaks` — number of drones with CUSUM anomaly this tick
    pub fn update(
        &mut self,
        alive: u32,
        total_losses: usize,
        max_intent_score: f64,
        cusum_breaks: u32,
    ) -> f64 {
        // Track consecutive loss ticks.
        if total_losses > self.prev_loss_count {
            self.consecutive_loss_ticks += 1;
        } else {
            self.consecutive_loss_ticks = 0;
        }
        self.prev_loss_count = total_losses;

        // Map STRIX telemetry → PhiSim FearState.
        let initial = (alive as usize + total_losses).max(1);
        let attrition = 1.0 - (alive as f64 / initial as f64);
        // Only hostile approach (positive intent) inflates vol_ratio.
        // Retreating threats (negative) should not increase fear through this channel.
        let threat_intensity = 1.0 + max_intent_score.max(0.0);

        let state = FearState::new(
            attrition,                   // drawdown
            threat_intensity,            // vol_ratio
            cusum_breaks,                // anomaly_count
            self.consecutive_loss_ticks, // consecutive_losses
        );

        self.phi_sim.update_fear(&state);
        self.phi_sim.fear_level()
    }

    /// Record mission tick outcome for RL experience replay.
    ///
    /// `outcome` should be a scalar quality metric — e.g. task completion
    /// rate, survival rate, or a composite mission score in [0, 1].
    pub fn record_outcome(&mut self, outcome: f64) {
        self.phi_sim.record_outcome(outcome);
    }

    /// Train the RL meta-learner if the replay buffer is full enough.
    pub fn train(&mut self) {
        self.phi_sim.train();
    }

    /// Current fear level.
    pub fn fear_level(&self) -> f64 {
        self.phi_sim.fear_level()
    }

    /// How many experiences the meta-learner has recorded.
    pub fn experience_count(&self) -> usize {
        self.phi_sim.experience_count()
    }
}

#[cfg(feature = "phi-sim")]
impl Default for FearAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Modulation helpers ──────────────────────────────────────────────────
// These do NOT depend on phi-sim — they work with raw fear values.

/// Modulate regime detection thresholds by fear.
///
/// Higher F → larger evade distance (150→500m), lower closing rate
/// threshold (2.0→0.5 m/s). The swarm becomes "jumpier" — it evades
/// sooner and from further away.
pub fn modulate_detection_config(
    base: &strix_core::regime::DetectionConfig,
    f: f64,
) -> strix_core::regime::DetectionConfig {
    let f = f.clamp(0.0, 1.0);
    strix_core::regime::DetectionConfig {
        engage_distance: base.engage_distance * (1.0 + f * 0.5), // 500→750m
        evade_distance: base.evade_distance * (1.0 + f * 2.3),   // 150→500m
        closing_rate_threshold: base.closing_rate_threshold * (1.0 - f * 0.75), // 2.0→0.5 m/s
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "phi-sim"))]
mod tests {
    use super::*;
    use strix_core::regime::DetectionConfig;

    // ── modulate_detection_config ────────────────────────────────────────────

    #[test]
    fn test_modulate_detection_config_f0_unchanged() {
        let base = DetectionConfig::default();
        let result = modulate_detection_config(&base, 0.0);
        assert!(
            (result.engage_distance - base.engage_distance).abs() < 1e-12,
            "engage_distance should be unchanged at F=0"
        );
        assert!(
            (result.evade_distance - base.evade_distance).abs() < 1e-12,
            "evade_distance should be unchanged at F=0"
        );
        assert!(
            (result.closing_rate_threshold - base.closing_rate_threshold).abs() < 1e-12,
            "closing_rate_threshold should be unchanged at F=0"
        );
    }

    #[test]
    fn test_modulate_detection_config_f1_formulas() {
        let base = DetectionConfig::default();
        let result = modulate_detection_config(&base, 1.0);

        let expected_evade = base.evade_distance * 3.3;
        assert!(
            (result.evade_distance - expected_evade).abs() < 1e-9,
            "evade_distance at F=1: got {}, expected {}",
            result.evade_distance,
            expected_evade
        );

        let expected_closing = base.closing_rate_threshold * 0.25;
        assert!(
            (result.closing_rate_threshold - expected_closing).abs() < 1e-9,
            "closing_rate_threshold at F=1: got {}, expected {}",
            result.closing_rate_threshold,
            expected_closing
        );
    }

    #[test]
    fn test_modulate_detection_config_clamps_above_one() {
        let base = DetectionConfig::default();
        let result_clamped = modulate_detection_config(&base, 1.5);
        let result_f1 = modulate_detection_config(&base, 1.0);
        assert!(
            (result_clamped.evade_distance - result_f1.evade_distance).abs() < 1e-12,
            "F > 1.0 should be clamped to 1.0"
        );
        assert!(
            (result_clamped.closing_rate_threshold - result_f1.closing_rate_threshold).abs()
                < 1e-12,
            "F > 1.0 should be clamped to 1.0"
        );
    }

    // ── FearAdapter ──────────────────────────────────────────────────────────

    #[test]
    fn test_fear_adapter_new_starts_at_baseline() {
        // PhiSim initialises at 0.3 (slightly pessimistic factory setting).
        // The drone_swarm preset has fear_floor = 0.1, so fear never drops to 0.
        let adapter = FearAdapter::new();
        let level = adapter.fear_level();
        assert!(
            level >= 0.0 && level <= 1.0,
            "initial fear must be in [0,1], got {level}"
        );
        // Factory default is 0.3; must be strictly below 1.0 (not maximum fear).
        assert!(
            level < 1.0,
            "new FearAdapter should not start at maximum fear, got {level}"
        );
    }

    #[test]
    fn test_fear_adapter_update_zero_losses_low_fear() {
        let mut adapter = FearAdapter::new();
        // 10 alive, 0 losses, no threats, no CUSUM breaks.
        // After one calm tick the heuristic resolves to a low value (drone_swarm fear_floor = 0.1).
        let fear = adapter.update(10, 0, 0.0, 0);
        assert!(
            fear < 0.5,
            "zero losses/threats should produce below-median fear, got {fear}"
        );
    }

    #[test]
    fn test_fear_adapter_update_increasing_losses_increases_fear() {
        let mut adapter = FearAdapter::new();
        // Start with no losses
        let fear_0 = adapter.update(10, 0, 0.0, 0);

        // Simulate escalating losses over multiple ticks
        let fear_1 = adapter.update(8, 2, 0.3, 1);
        let fear_2 = adapter.update(5, 5, 0.6, 2);
        let fear_3 = adapter.update(3, 7, 0.9, 3);

        assert!(
            fear_3 > fear_0,
            "fear should increase with cumulative losses: {} vs {}",
            fear_3,
            fear_0
        );
        // At least one step must show an increase
        assert!(
            fear_1 > fear_0 || fear_2 > fear_1 || fear_3 > fear_2,
            "fear should rise monotonically with increasing losses"
        );
    }
}
