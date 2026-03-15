//! Fear Adapter — bridges STRIX swarm telemetry to PhiSim's fear meta-parameter.
//!
//! PhiSim computes F in [0,1] from a `FearState` that maps financial concepts
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
//!
//! ## SwarmFearAdapter (v2)
//!
//! Wraps `SwarmPhiSim` instead of a single `PhiSim`, giving each drone its
//! own fear/courage/tension state with collective signal aggregation. Late-
//! arriving drones get their own standalone `PhiSim` instances; removed drones
//! are soft-deleted (skipped in updates) to avoid index invalidation.

#[cfg(feature = "phi-sim")]
use std::collections::{HashMap, HashSet};

#[cfg(feature = "phi-sim")]
use phi_sim::config::Config;
#[cfg(feature = "phi-sim")]
use phi_sim::control::decision::Decision;
#[cfg(feature = "phi-sim")]
use phi_sim::fear::calibration::CalibrationMetrics;
#[cfg(feature = "phi-sim")]
use phi_sim::swarm::aggregation::SwarmSignals;
#[cfg(feature = "phi-sim")]
use phi_sim::swarm::coordinator::{SwarmConfig as PhiSwarmConfig, SwarmPhiSim};
#[cfg(feature = "phi-sim")]
use phi_sim::types::{Action, FearAxes, FearState, Observation};
#[cfg(feature = "phi-sim")]
use phi_sim::PhiSim;

// ── DroneFearInputs ────────────────────────────────────────────────────────

/// Per-drone telemetry inputs for fear computation.
///
/// Collected from the swarm tick loop and fed into `SwarmFearAdapter::update_fear`.
#[cfg(feature = "phi-sim")]
#[derive(Debug, Clone)]
pub struct DroneFearInputs {
    /// Nearest threat distance (meters).
    pub threat_distance: f64,
    /// Rate of closure to nearest threat (m/s, negative = approaching).
    pub closing_rate: f64,
    /// Whether this drone's CUSUM detector triggered this tick.
    pub cusum_triggered: bool,
    /// Current operating regime for this drone.
    pub regime: strix_core::Regime,
    /// Drone's current speed (m/s).
    pub speed: f64,
    /// Fleet formation coherence metric [0, 1].
    pub fleet_coherence: f64,
}

// ── SwarmFearAdapter ───────────────────────────────────────────────────────

/// Per-drone fear adapter wrapping `SwarmPhiSim` for collective intelligence.
///
/// Each drone maps to a PhiSim child inside the swarm coordinator. The adapter
/// translates STRIX telemetry into `FearState` values and invokes the PhiSim
/// pipeline. On auction ticks it runs the full `step()` (inference + prediction
/// + decision); on regular ticks it only updates fear heuristics.
#[cfg(feature = "phi-sim")]
pub struct SwarmFearAdapter {
    phi_swarm: SwarmPhiSim,
    consecutive_loss_ticks: u32,
    prev_loss_count: usize,
    /// drone_id -> child index mapping (into SwarmPhiSim.children)
    drone_index: HashMap<u32, usize>,
    /// Cached per-drone FearAxes from last update
    cached_axes: HashMap<u32, FearAxes>,
    /// Last decisions from most recent step() (contains scenario data)
    cached_decisions: HashMap<u32, Decision>,
    /// Last collective signals
    collective: Option<SwarmSignals>,
    /// Drones that have been removed (soft-delete — skip in updates)
    removed_drones: HashSet<u32>,
    /// Late-added drones that don't have a SwarmPhiSim child index
    late_children: HashMap<u32, PhiSim>,
}

#[cfg(feature = "phi-sim")]
impl SwarmFearAdapter {
    /// Create a new swarm fear adapter for the given drone IDs.
    ///
    /// Each drone gets its own PhiSim child inside a flat-topology swarm
    /// coordinator. The `drone_swarm()` config preset is used for all children.
    pub fn new(drone_ids: &[u32]) -> Self {
        let swarm_config = PhiSwarmConfig::default_flat(drone_ids.len());
        let phi_swarm = SwarmPhiSim::new(Config::drone_swarm(), swarm_config);

        let drone_index: HashMap<u32, usize> = drone_ids
            .iter()
            .enumerate()
            .map(|(idx, &id)| (id, idx))
            .collect();

        Self {
            phi_swarm,
            consecutive_loss_ticks: 0,
            prev_loss_count: 0,
            drone_index,
            cached_axes: HashMap::new(),
            cached_decisions: HashMap::new(),
            collective: None,
            removed_drones: HashSet::new(),
            late_children: HashMap::new(),
        }
    }

    // ── Core update ────────────────────────────────────────────────────────

    /// Update fear for all drones from per-drone telemetry.
    ///
    /// # Tick types
    ///
    /// - **Non-auction tick**: calls each child's `update_fear()` with a
    ///   per-drone `FearState`, updating fear/courage/tension heuristics only.
    /// - **Auction tick**: runs the full PhiSim pipeline via
    ///   `phi_swarm.step()` with per-drone observations and fear states,
    ///   caching the resulting `Decision` per drone.
    ///
    /// In both cases, per-drone `FearAxes` are cached for downstream use.
    pub fn update_fear(
        &mut self,
        per_drone: &[(u32, DroneFearInputs)],
        fleet_attrition: f64,
        fleet_cusum_breaks: u32,
        is_auction_tick: bool,
    ) {
        // Track consecutive loss ticks from fleet-wide attrition.
        let current_losses = (fleet_attrition * 100.0).round() as usize;
        if current_losses > self.prev_loss_count {
            self.consecutive_loss_ticks += 1;
        } else {
            self.consecutive_loss_ticks = 0;
        }
        self.prev_loss_count = current_losses;

        if is_auction_tick {
            self.step_full(per_drone, fleet_attrition, fleet_cusum_breaks);
        } else {
            self.step_fear_only(per_drone, fleet_attrition, fleet_cusum_breaks);
        }

        // Cache FearAxes for all active drones.
        for &(drone_id, _) in per_drone {
            if self.removed_drones.contains(&drone_id) {
                continue;
            }
            if let Some(&idx) = self.drone_index.get(&drone_id) {
                let axes = self.phi_swarm.child(idx).fear_axes();
                self.cached_axes.insert(drone_id, axes);
            } else if let Some(phi) = self.late_children.get(&drone_id) {
                self.cached_axes.insert(drone_id, phi.fear_axes());
            }
        }

        // Cache collective signals.
        self.collective = self.phi_swarm.collective_signals().cloned();
    }

    /// Fear-only update: call update_fear on each child without running
    /// the full inference/prediction/decision pipeline.
    fn step_fear_only(
        &mut self,
        per_drone: &[(u32, DroneFearInputs)],
        fleet_attrition: f64,
        fleet_cusum_breaks: u32,
    ) {
        for &(drone_id, ref inputs) in per_drone {
            if self.removed_drones.contains(&drone_id) {
                continue;
            }
            let state = self.build_fear_state(inputs, fleet_attrition, fleet_cusum_breaks);
            if let Some(&idx) = self.drone_index.get(&drone_id) {
                self.phi_swarm.child_mut(idx).update_fear(&state);
            } else if let Some(phi) = self.late_children.get_mut(&drone_id) {
                phi.update_fear(&state);
            }
        }
    }

    /// Full pipeline: build per-drone observations + fear states, run
    /// `phi_swarm.step()`, and cache the resulting decisions.
    fn step_full(
        &mut self,
        per_drone: &[(u32, DroneFearInputs)],
        fleet_attrition: f64,
        fleet_cusum_breaks: u32,
    ) {
        let swarm_size = self.phi_swarm.swarm_size();

        // Build ordered observation and fear_state vectors for the swarm children.
        // Drones not present in `per_drone` get default values.
        let mut observations = vec![Observation::new(vec![0.0; 4], 0.0); swarm_size];
        let mut fear_states = vec![FearState::default(); swarm_size];
        let actions = vec![Action::hold()];

        // Map per_drone inputs into the swarm-indexed arrays.
        let mut drone_order: Vec<Option<u32>> = vec![None; swarm_size];
        for &(drone_id, ref inputs) in per_drone {
            if self.removed_drones.contains(&drone_id) {
                continue;
            }
            if let Some(&idx) = self.drone_index.get(&drone_id) {
                if idx < swarm_size {
                    let state = self.build_fear_state(inputs, fleet_attrition, fleet_cusum_breaks);
                    observations[idx] = Observation::new(
                        vec![
                            inputs.threat_distance,
                            inputs.closing_rate,
                            inputs.speed,
                            inputs.fleet_coherence,
                        ],
                        0.0,
                    );
                    fear_states[idx] = state;
                    drone_order[idx] = Some(drone_id);
                }
            }
        }

        // Run the full swarm step.
        let decisions = self.phi_swarm.step(&observations, &actions, &fear_states);

        // Cache decisions keyed by drone_id.
        self.cached_decisions.clear();
        for (idx, decision) in decisions.into_iter().enumerate() {
            if let Some(drone_id) = drone_order[idx] {
                self.cached_decisions.insert(drone_id, decision);
            }
        }

        // Also update late children (they don't participate in swarm step).
        // Pre-build states to avoid overlapping borrows on self.
        let late_updates: Vec<_> = per_drone
            .iter()
            .filter(|&&(drone_id, _)| {
                !self.removed_drones.contains(&drone_id)
                    && self.late_children.contains_key(&drone_id)
            })
            .map(|&(drone_id, ref inputs)| {
                let state = build_fear_state(
                    inputs,
                    fleet_attrition,
                    fleet_cusum_breaks,
                    self.consecutive_loss_ticks,
                );
                let obs = Observation::new(
                    vec![
                        inputs.threat_distance,
                        inputs.closing_rate,
                        inputs.speed,
                        inputs.fleet_coherence,
                    ],
                    0.0,
                );
                (drone_id, obs, state)
            })
            .collect();

        for (drone_id, obs, state) in late_updates {
            if let Some(phi) = self.late_children.get_mut(&drone_id) {
                let decision = phi.step(&obs, &actions, &state);
                self.cached_decisions.insert(drone_id, decision);
            }
        }
    }

    /// Build a PhiSim FearState from STRIX drone telemetry (delegates to free fn).
    fn build_fear_state(
        &self,
        inputs: &DroneFearInputs,
        fleet_attrition: f64,
        fleet_cusum_breaks: u32,
    ) -> FearState {
        build_fear_state(
            inputs,
            fleet_attrition,
            fleet_cusum_breaks,
            self.consecutive_loss_ticks,
        )
    }

    // ── Per-drone accessors ────────────────────────────────────────────────

    /// Current fear level for a specific drone.
    pub fn fear_level(&self, drone_id: u32) -> f64 {
        if let Some(&idx) = self.drone_index.get(&drone_id) {
            self.phi_swarm.child(idx).fear_level()
        } else if let Some(phi) = self.late_children.get(&drone_id) {
            phi.fear_level()
        } else {
            0.3 // factory default
        }
    }

    /// Current courage level for a specific drone.
    pub fn courage_level(&self, drone_id: u32) -> f64 {
        if let Some(&idx) = self.drone_index.get(&drone_id) {
            self.phi_swarm.child(idx).courage_level()
        } else if let Some(phi) = self.late_children.get(&drone_id) {
            phi.courage_level()
        } else {
            0.3
        }
    }

    /// Current tension for a specific drone.
    pub fn tension(&self, drone_id: u32) -> f64 {
        if let Some(&idx) = self.drone_index.get(&drone_id) {
            self.phi_swarm.child(idx).tension()
        } else if let Some(phi) = self.late_children.get(&drone_id) {
            phi.tension()
        } else {
            0.0
        }
    }

    /// Current FearAxes for a specific drone (from cache).
    pub fn fear_axes(&self, drone_id: u32) -> FearAxes {
        self.cached_axes
            .get(&drone_id)
            .copied()
            .unwrap_or_else(|| FearAxes::from_fear(0.3))
    }

    // ── Collective accessors ───────────────────────────────────────────────

    /// Collective fear across the swarm.
    pub fn collective_fear(&self) -> f64 {
        self.phi_swarm.collective_fear()
    }

    /// Collective courage across the swarm.
    pub fn collective_courage(&self) -> f64 {
        self.phi_swarm.collective_courage()
    }

    /// Collective tension across the swarm.
    pub fn collective_tension(&self) -> f64 {
        self.phi_swarm.collective_tension()
    }

    /// Average calibration metrics across all children (including late ones).
    pub fn calibration(&self) -> CalibrationMetrics {
        let swarm_size = self.phi_swarm.swarm_size();
        let late_count = self.late_children.len();
        let total = swarm_size + late_count;

        if total == 0 {
            return CalibrationMetrics::neutral();
        }

        let mut width_sum = 0.0;
        let mut depth_sum = 0.0;
        let mut realism_sum = 0.0;
        let mut speed_sum = 0.0;
        let mut compression_sum = 0.0;

        for i in 0..swarm_size {
            let cal = self.phi_swarm.child(i).running_calibration();
            width_sum += cal.width_score;
            depth_sum += cal.depth_score;
            realism_sum += cal.realism_score;
            speed_sum += cal.speed_score;
            compression_sum += cal.compression_score;
        }

        for phi in self.late_children.values() {
            let cal = phi.running_calibration();
            width_sum += cal.width_score;
            depth_sum += cal.depth_score;
            realism_sum += cal.realism_score;
            speed_sum += cal.speed_score;
            compression_sum += cal.compression_score;
        }

        let n = total as f64;
        CalibrationMetrics {
            width_score: width_sum / n,
            depth_score: depth_sum / n,
            realism_score: realism_sum / n,
            speed_score: speed_sum / n,
            compression_score: compression_sum / n,
        }
    }

    /// Last cached decision for a specific drone, if available.
    pub fn decision_for_drone(&self, drone_id: u32) -> Option<&Decision> {
        self.cached_decisions.get(&drone_id)
    }

    // ── Training & outcomes ────────────────────────────────────────────────

    /// Record a mission tick outcome for RL experience replay (all children).
    pub fn record_outcome(&mut self, outcome: f64) {
        self.phi_swarm.record_outcome_all(outcome);
        for phi in self.late_children.values_mut() {
            phi.record_outcome(outcome);
        }
    }

    /// Train the RL meta-learner on all children's replay buffers.
    pub fn train(&mut self) {
        self.phi_swarm.train_all();
        for phi in self.late_children.values_mut() {
            phi.train();
        }
    }

    /// Counterfactual regret corrections from the first child.
    pub fn counterfactual_corrections(&self) -> (f64, f64) {
        if self.phi_swarm.swarm_size() > 0 {
            self.phi_swarm.child(0).counterfactual_regret_corrections()
        } else if let Some(phi) = self.late_children.values().next() {
            phi.counterfactual_regret_corrections()
        } else {
            (0.0, 0.0)
        }
    }

    /// How many experiences the meta-learner has recorded (from first child).
    pub fn experience_count(&self) -> usize {
        if self.phi_swarm.swarm_size() > 0 {
            self.phi_swarm.child(0).experience_count()
        } else if let Some(phi) = self.late_children.values().next() {
            phi.experience_count()
        } else {
            0
        }
    }

    // ── Dynamic membership ─────────────────────────────────────────────────

    /// Register a new drone that joined after initial construction.
    ///
    /// The drone gets its own standalone `PhiSim` instance (not part of the
    /// `SwarmPhiSim` coordinator, since it cannot dynamically add children).
    /// Late-added drones still participate in `update_fear` and get their
    /// own `FearAxes` and `Decision` caches.
    pub fn register_drone(&mut self, id: u32) {
        if self.drone_index.contains_key(&id) || self.late_children.contains_key(&id) {
            return; // already registered
        }
        self.removed_drones.remove(&id); // un-remove if previously removed
        self.late_children
            .insert(id, PhiSim::new(Config::drone_swarm()));
    }

    /// Remove a drone from active updates (soft-delete).
    ///
    /// The drone is not actually removed from `SwarmPhiSim` to avoid index
    /// invalidation. Instead it is added to the removed set and skipped
    /// during updates.
    pub fn remove_drone(&mut self, id: u32) {
        self.removed_drones.insert(id);
        self.cached_axes.remove(&id);
        self.cached_decisions.remove(&id);
        self.late_children.remove(&id);
    }
}

// ── Free helper — avoids borrow conflicts inside SwarmFearAdapter methods ──

/// Build a PhiSim FearState from STRIX drone telemetry.
#[cfg(feature = "phi-sim")]
fn build_fear_state(
    inputs: &DroneFearInputs,
    fleet_attrition: f64,
    fleet_cusum_breaks: u32,
    consecutive_loss_ticks: u32,
) -> FearState {
    // Threat intensity: combine closing rate and regime.
    let threat_intensity = if inputs.closing_rate < 0.0 {
        1.0 + (-inputs.closing_rate / 10.0).min(1.0)
    } else {
        1.0
    };

    let cusum_count = if inputs.cusum_triggered {
        fleet_cusum_breaks.max(1)
    } else {
        0
    };

    FearState::new(
        fleet_attrition,        // drawdown
        threat_intensity,       // vol_ratio
        cusum_count,            // anomaly_count
        consecutive_loss_ticks, // consecutive_losses
    )
}

// ── Modulation helpers ──────────────────────────────────────────────────
// These do NOT depend on phi-sim — they work with raw fear values.

/// Modulate regime detection thresholds by fear.
///
/// Higher F -> larger evade distance (150->500m), lower closing rate
/// threshold (2.0->0.5 m/s). The swarm becomes "jumpier" — it evades
/// sooner and from further away.
pub fn modulate_detection_config(
    base: &strix_core::regime::DetectionConfig,
    f: f64,
) -> strix_core::regime::DetectionConfig {
    let f = f.clamp(0.0, 1.0);
    strix_core::regime::DetectionConfig {
        engage_distance: base.engage_distance * (1.0 + f * 0.5), // 500->750m
        evade_distance: base.evade_distance * (1.0 + f * 2.3),   // 150->500m
        closing_rate_threshold: base.closing_rate_threshold * (1.0 - f * 0.75), // 2.0->0.5 m/s
    }
}

// ── FearAxes-aware modulation helpers ────────────────────────────────────
// These work with raw FearAxes values — NOT feature-gated.

/// Modulate formation geometry by fear axes.
///
/// Higher fear -> wider spacing, tighter deadband, reduced correction speed.
/// The swarm spreads out (harder to hit with area weapons) but holds
/// formation more strictly (reduced chaos tolerance).
///
/// | FearAxes field | Effect                                               |
/// |----------------|------------------------------------------------------|
/// | `bias`         | Spacing multiplier: 1.0 + bias * 0.5 (up to +50%)   |
/// | `speed`        | Correction speed multiplier: 1.0 / speed             |
/// | `threshold`    | Deadband shrinks: threshold * base (tighter holding) |
pub fn modulate_formation_config(
    base: &strix_core::formation::FormationConfig,
    axes: &phi_sim::types::FearAxes,
) -> strix_core::formation::FormationConfig {
    strix_core::formation::FormationConfig {
        // Wider spacing under fear: up to +50% at max fear (bias=1.0).
        spacing: base.spacing * (1.0 + axes.bias * 0.5),
        // Vee angle unchanged — geometric identity, not risk-dependent.
        vee_angle_deg: base.vee_angle_deg,
        // Slower corrections under fear: speed axis [1.0, 3.0] -> correction
        // speed reduced by 1/speed factor.
        max_correction_speed: base.max_correction_speed / axes.speed,
        // Tighter deadband under fear: threshold axis [0.3, 1.0] scales the
        // deadband down, making the formation hold more strictly.
        deadband: base.deadband * axes.threshold,
    }
}

/// Modulate EW severity multiplier by fear axes.
///
/// Higher fear amplifies EW severity response: a fearful swarm reacts more
/// aggressively to electronic warfare detections.
///
/// Returns `base_multiplier * (1.0 + bias * 0.5)`, scaled by the speed axis
/// to react faster under high fear.
pub fn modulate_ew_severity(base_multiplier: f64, axes: &phi_sim::types::FearAxes) -> f64 {
    // bias [0,1] amplifies the response by up to 50%.
    // speed [1.0, 3.0] further amplifies: sqrt(speed) to avoid over-scaling.
    base_multiplier * (1.0 + axes.bias * 0.5) * axes.speed.sqrt()
}

/// Modulate pheromone intensity and decay by fear axes.
///
/// Higher fear -> stronger pheromone deposits (avoid dangerous areas more
/// aggressively) with slower decay (longer memory of threats).
///
/// Returns `(modulated_intensity, modulated_decay)`.
pub fn modulate_pheromone(
    intensity: f64,
    decay: f64,
    axes: &phi_sim::types::FearAxes,
) -> (f64, f64) {
    // Intensity increases with fear bias: up to 2x at max fear.
    let modulated_intensity = intensity * (1.0 + axes.bias);

    // Decay slows under fear: multiply by threshold [0.3, 1.0].
    // At max fear (threshold=0.3), decay is 30% of base -> pheromones persist
    // 3.3x longer. This encodes "don't forget danger zones quickly".
    let modulated_decay = decay * axes.threshold;

    (modulated_intensity, modulated_decay)
}

/// Modulate gossip fanout by fear level.
///
/// Higher fear -> more gossip peers (increased information sharing under
/// stress). The swarm communicates more aggressively when threatened.
///
/// Returns `base_fanout + floor(f * base_fanout)`, capped at a sensible
/// upper bound to avoid flooding the mesh.
pub fn modulate_gossip_fanout(base_fanout: usize, f: f64) -> usize {
    let f = f.clamp(0.0, 1.0);
    let extra = (f * base_fanout as f64).floor() as usize;
    let max_fanout = base_fanout * 3; // hard cap at 3x base
    (base_fanout + extra).min(max_fanout)
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

    // ── SwarmFearAdapter ──────────────────────────────────────────────────────

    #[test]
    fn test_swarm_fear_adapter_new_starts_at_baseline() {
        let adapter = SwarmFearAdapter::new(&[1, 2, 3]);
        for &id in &[1, 2, 3] {
            let level = adapter.fear_level(id);
            assert!(
                level >= 0.0 && level <= 1.0,
                "initial fear must be in [0,1], got {level} for drone {id}"
            );
            assert!(
                level < 1.0,
                "new SwarmFearAdapter should not start at maximum fear, got {level}"
            );
        }
    }

    #[test]
    fn test_swarm_fear_adapter_per_drone_fear() {
        let adapter = SwarmFearAdapter::new(&[10, 20]);
        // Each drone should have independent (but initially equal) fear levels.
        let f10 = adapter.fear_level(10);
        let f20 = adapter.fear_level(20);
        assert!(
            (f10 - f20).abs() < 1e-12,
            "initial fear should be equal across drones"
        );
    }

    #[test]
    fn test_swarm_fear_adapter_update_increases_fear() {
        let mut adapter = SwarmFearAdapter::new(&[1, 2]);

        let calm_inputs = DroneFearInputs {
            threat_distance: 1000.0,
            closing_rate: 0.0,
            cusum_triggered: false,
            regime: strix_core::Regime::Patrol,
            speed: 10.0,
            fleet_coherence: 1.0,
        };

        // Start calm.
        adapter.update_fear(
            &[(1, calm_inputs.clone()), (2, calm_inputs.clone())],
            0.0,
            0,
            false,
        );
        let fear_calm = adapter.fear_level(1);

        // Escalate threat.
        let hot_inputs = DroneFearInputs {
            threat_distance: 50.0,
            closing_rate: -8.0,
            cusum_triggered: true,
            regime: strix_core::Regime::Evade,
            speed: 25.0,
            fleet_coherence: 0.3,
        };

        // Multiple escalating ticks.
        for _ in 0..5 {
            adapter.update_fear(
                &[(1, hot_inputs.clone()), (2, hot_inputs.clone())],
                0.4,
                3,
                false,
            );
        }
        let fear_hot = adapter.fear_level(1);

        assert!(
            fear_hot > fear_calm,
            "fear should increase with escalating threats: calm={fear_calm}, hot={fear_hot}"
        );
    }

    #[test]
    fn test_swarm_fear_adapter_collective_accessors() {
        let adapter = SwarmFearAdapter::new(&[1, 2, 3]);
        // Before any step, collective values should be 0.0 (no aggregation yet).
        let cf = adapter.collective_fear();
        let cc = adapter.collective_courage();
        let ct = adapter.collective_tension();
        // These are valid f64 values.
        assert!(cf.is_finite(), "collective_fear must be finite");
        assert!(cc.is_finite(), "collective_courage must be finite");
        assert!(ct.is_finite(), "collective_tension must be finite");
    }

    #[test]
    fn test_swarm_fear_adapter_register_drone() {
        let mut adapter = SwarmFearAdapter::new(&[1, 2]);
        adapter.register_drone(3);

        // New drone should have baseline fear.
        let f3 = adapter.fear_level(3);
        assert!(
            f3 >= 0.0 && f3 <= 1.0,
            "registered drone should have valid fear: {f3}"
        );
        assert!(
            f3 < 1.0,
            "registered drone should not start at max fear: {f3}"
        );
    }

    #[test]
    fn test_swarm_fear_adapter_remove_drone() {
        let mut adapter = SwarmFearAdapter::new(&[1, 2, 3]);
        adapter.remove_drone(2);

        // Removed drone should return default fear (not panic).
        let f2 = adapter.fear_level(2);
        // It still returns a value from the underlying child (soft-delete
        // only skips updates, the child still exists in the swarm).
        assert!(f2.is_finite(), "removed drone fear should be finite");
    }

    #[test]
    fn test_swarm_fear_adapter_unknown_drone_returns_default() {
        let adapter = SwarmFearAdapter::new(&[1, 2]);
        // Drone 99 was never registered.
        let f = adapter.fear_level(99);
        assert!(
            (f - 0.3).abs() < 1e-12,
            "unknown drone should return factory default 0.3, got {f}"
        );
    }

    #[test]
    fn test_swarm_fear_adapter_calibration() {
        let adapter = SwarmFearAdapter::new(&[1, 2, 3]);
        let cal = adapter.calibration();
        // Neutral calibration: all scores at 0.5.
        assert!(
            (cal.overall() - 0.5).abs() < 1e-6,
            "initial calibration should be neutral (0.5), got {}",
            cal.overall()
        );
    }

    #[test]
    fn test_swarm_fear_adapter_experience_count() {
        let adapter = SwarmFearAdapter::new(&[1, 2]);
        assert_eq!(
            adapter.experience_count(),
            0,
            "initial experience count should be 0"
        );
    }

    #[test]
    fn test_swarm_fear_adapter_counterfactual_corrections() {
        let adapter = SwarmFearAdapter::new(&[1, 2]);
        let (f_corr, c_corr) = adapter.counterfactual_corrections();
        assert!(f_corr.is_finite(), "fear correction must be finite");
        assert!(c_corr.is_finite(), "courage correction must be finite");
    }

    #[test]
    fn test_swarm_fear_adapter_decision_for_drone() {
        let adapter = SwarmFearAdapter::new(&[1, 2]);
        // No auction tick yet — no decisions cached.
        assert!(
            adapter.decision_for_drone(1).is_none(),
            "no decision before auction tick"
        );
    }

    #[test]
    fn test_swarm_fear_adapter_auction_tick_caches_decisions() {
        let mut adapter = SwarmFearAdapter::new(&[1, 2]);
        let inputs = DroneFearInputs {
            threat_distance: 500.0,
            closing_rate: -2.0,
            cusum_triggered: false,
            regime: strix_core::Regime::Patrol,
            speed: 15.0,
            fleet_coherence: 0.8,
        };

        adapter.update_fear(
            &[(1, inputs.clone()), (2, inputs)],
            0.1,
            0,
            true, // auction tick
        );

        // After auction tick, decisions should be cached.
        assert!(
            adapter.decision_for_drone(1).is_some(),
            "decision should be cached after auction tick for drone 1"
        );
        assert!(
            adapter.decision_for_drone(2).is_some(),
            "decision should be cached after auction tick for drone 2"
        );
    }

    #[test]
    fn test_swarm_fear_adapter_register_then_update() {
        let mut adapter = SwarmFearAdapter::new(&[1]);
        adapter.register_drone(5);

        let inputs = DroneFearInputs {
            threat_distance: 200.0,
            closing_rate: -3.0,
            cusum_triggered: true,
            regime: strix_core::Regime::Engage,
            speed: 20.0,
            fleet_coherence: 0.6,
        };

        // Update should not panic for late-added drone.
        adapter.update_fear(&[(1, inputs.clone()), (5, inputs)], 0.2, 1, false);

        let f5 = adapter.fear_level(5);
        assert!(f5.is_finite(), "late drone fear should be finite: {f5}");
    }

    // ── modulate_formation_config ─────────────────────────────────────────

    #[test]
    fn test_modulate_formation_config_zero_fear() {
        let base = strix_core::formation::FormationConfig::default();
        let axes = FearAxes::from_fear(0.0);
        let result = modulate_formation_config(&base, &axes);

        assert!(
            (result.spacing - base.spacing).abs() < 1e-12,
            "spacing should be unchanged at F=0"
        );
        assert!(
            (result.vee_angle_deg - base.vee_angle_deg).abs() < 1e-12,
            "vee_angle_deg should be unchanged"
        );
        assert!(
            (result.max_correction_speed - base.max_correction_speed).abs() < 1e-9,
            "correction speed should be unchanged at F=0 (speed=1.0)"
        );
        assert!(
            (result.deadband - base.deadband).abs() < 1e-9,
            "deadband should be unchanged at F=0 (threshold=1.0)"
        );
    }

    #[test]
    fn test_modulate_formation_config_max_fear() {
        let base = strix_core::formation::FormationConfig::default();
        let axes = FearAxes::from_fear(1.0);
        let result = modulate_formation_config(&base, &axes);

        // bias=1.0 -> spacing * 1.5
        let expected_spacing = base.spacing * 1.5;
        assert!(
            (result.spacing - expected_spacing).abs() < 1e-9,
            "spacing at F=1: got {}, expected {}",
            result.spacing,
            expected_spacing
        );

        // speed=3.0 -> correction / 3.0
        let expected_correction = base.max_correction_speed / 3.0;
        assert!(
            (result.max_correction_speed - expected_correction).abs() < 1e-9,
            "correction speed at F=1: got {}, expected {}",
            result.max_correction_speed,
            expected_correction
        );

        // threshold=0.3 -> deadband * 0.3
        let expected_deadband = base.deadband * 0.3;
        assert!(
            (result.deadband - expected_deadband).abs() < 1e-9,
            "deadband at F=1: got {}, expected {}",
            result.deadband,
            expected_deadband
        );
    }

    #[test]
    fn test_modulate_formation_config_preserves_vee_angle() {
        let base = strix_core::formation::FormationConfig {
            vee_angle_deg: 42.0,
            ..Default::default()
        };
        for f in [0.0, 0.3, 0.5, 0.8, 1.0] {
            let axes = FearAxes::from_fear(f);
            let result = modulate_formation_config(&base, &axes);
            assert!(
                (result.vee_angle_deg - 42.0).abs() < 1e-12,
                "vee_angle_deg must never change, got {} at F={}",
                result.vee_angle_deg,
                f
            );
        }
    }

    // ── modulate_ew_severity ──────────────────────────────────────────────

    #[test]
    fn test_modulate_ew_severity_zero_fear() {
        let axes = FearAxes::from_fear(0.0);
        let result = modulate_ew_severity(2.0, &axes);
        // bias=0, speed=1.0 -> 2.0 * 1.0 * 1.0 = 2.0
        assert!(
            (result - 2.0).abs() < 1e-9,
            "EW severity at F=0 should be unchanged: got {result}"
        );
    }

    #[test]
    fn test_modulate_ew_severity_max_fear() {
        let axes = FearAxes::from_fear(1.0);
        let result = modulate_ew_severity(2.0, &axes);
        // bias=1.0 -> 2.0 * 1.5 = 3.0; speed=3.0 -> 3.0 * sqrt(3) ~ 5.196
        let expected = 2.0 * 1.5 * 3.0_f64.sqrt();
        assert!(
            (result - expected).abs() < 1e-6,
            "EW severity at F=1: got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_modulate_ew_severity_monotonic() {
        let base = 1.0;
        let mut prev = modulate_ew_severity(base, &FearAxes::from_fear(0.0));
        for step in 1..=10 {
            let f = step as f64 / 10.0;
            let current = modulate_ew_severity(base, &FearAxes::from_fear(f));
            assert!(
                current >= prev - 1e-12,
                "EW severity should be monotonically increasing: f={f}, prev={prev}, cur={current}"
            );
            prev = current;
        }
    }

    // ── modulate_pheromone ────────────────────────────────────────────────

    #[test]
    fn test_modulate_pheromone_zero_fear() {
        let axes = FearAxes::from_fear(0.0);
        let (intensity, decay) = modulate_pheromone(1.0, 0.05, &axes);
        // bias=0 -> intensity * 1.0 = 1.0; threshold=1.0 -> decay * 1.0 = 0.05
        assert!(
            (intensity - 1.0).abs() < 1e-12,
            "pheromone intensity at F=0: got {intensity}"
        );
        assert!(
            (decay - 0.05).abs() < 1e-12,
            "pheromone decay at F=0: got {decay}"
        );
    }

    #[test]
    fn test_modulate_pheromone_max_fear() {
        let axes = FearAxes::from_fear(1.0);
        let (intensity, decay) = modulate_pheromone(1.0, 0.05, &axes);
        // bias=1.0 -> intensity * 2.0 = 2.0
        assert!(
            (intensity - 2.0).abs() < 1e-9,
            "pheromone intensity at F=1: got {intensity}"
        );
        // threshold=0.3 -> decay * 0.3 = 0.015
        let expected_decay = 0.05 * 0.3;
        assert!(
            (decay - expected_decay).abs() < 1e-9,
            "pheromone decay at F=1: got {decay}, expected {expected_decay}"
        );
    }

    #[test]
    fn test_modulate_pheromone_intensity_increases_with_fear() {
        let (i_low, _) = modulate_pheromone(1.0, 0.05, &FearAxes::from_fear(0.0));
        let (i_high, _) = modulate_pheromone(1.0, 0.05, &FearAxes::from_fear(1.0));
        assert!(
            i_high > i_low,
            "pheromone intensity should increase with fear: low={i_low}, high={i_high}"
        );
    }

    #[test]
    fn test_modulate_pheromone_decay_decreases_with_fear() {
        let (_, d_low) = modulate_pheromone(1.0, 0.05, &FearAxes::from_fear(0.0));
        let (_, d_high) = modulate_pheromone(1.0, 0.05, &FearAxes::from_fear(1.0));
        assert!(
            d_high < d_low,
            "pheromone decay should decrease with fear (longer persistence): low={d_low}, high={d_high}"
        );
    }

    // ── modulate_gossip_fanout ────────────────────────────────────────────

    #[test]
    fn test_modulate_gossip_fanout_zero_fear() {
        let result = modulate_gossip_fanout(3, 0.0);
        assert_eq!(result, 3, "gossip fanout at F=0 should be unchanged");
    }

    #[test]
    fn test_modulate_gossip_fanout_max_fear() {
        let result = modulate_gossip_fanout(3, 1.0);
        // base + floor(1.0 * 3) = 3 + 3 = 6
        assert_eq!(result, 6, "gossip fanout at F=1: got {result}");
    }

    #[test]
    fn test_modulate_gossip_fanout_capped() {
        // With F=1.0, base=10: 10 + 10 = 20, but cap = 3*10 = 30 -> 20.
        let result = modulate_gossip_fanout(10, 1.0);
        assert!(
            result <= 30,
            "gossip fanout should be capped at 3x base: got {result}"
        );
        assert_eq!(result, 20, "gossip fanout at F=1, base=10: got {result}");
    }

    #[test]
    fn test_modulate_gossip_fanout_clamps_negative() {
        let result = modulate_gossip_fanout(3, -0.5);
        assert_eq!(
            result, 3,
            "negative fear should be clamped to 0: got {result}"
        );
    }

    #[test]
    fn test_modulate_gossip_fanout_monotonic() {
        let base = 4;
        let mut prev = modulate_gossip_fanout(base, 0.0);
        for step in 1..=10 {
            let f = step as f64 / 10.0;
            let current = modulate_gossip_fanout(base, f);
            assert!(
                current >= prev,
                "gossip fanout should be monotonically increasing: f={f}, prev={prev}, cur={current}"
            );
            prev = current;
        }
    }
}
