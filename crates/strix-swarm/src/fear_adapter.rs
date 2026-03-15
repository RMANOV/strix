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

use phi_sim::config::Config;
use phi_sim::types::FearState;
use phi_sim::PhiSim;

/// Adapts STRIX telemetry into PhiSim's fear framework.
pub struct FearAdapter {
    phi_sim: PhiSim,
    /// Consecutive ticks where at least one drone was lost.
    consecutive_loss_ticks: u32,
    /// Loss count at previous tick (for streak detection).
    prev_loss_count: usize,
}

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

impl Default for FearAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Modulation helpers ──────────────────────────────────────────────────

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
