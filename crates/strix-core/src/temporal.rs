//! Multi-horizon temporal management.
//!
//! Adapted from the trading multi-timeframe analysis pattern:
//!
//! | Horizon | dt      | Particles | Purpose                       |
//! |---------|---------|-----------|-------------------------------|
//! | H1      | 0.1 s   | 100       | Tactical — obstacle avoidance |
//! | H2      | 5.0 s   | 500       | Operational — formation hold  |
//! | H3      | 60.0 s  | 2000      | Strategic — mission planning  |
//!
//! Information cascades downward: H3 sets constraints for H2, H2
//! provides waypoints for H1.

use nalgebra::Vector3;

use crate::anomaly::{self, CusumConfig};
use crate::particle_nav::ParticleNavFilter;
use crate::state::{Observation, Regime};

// ---------------------------------------------------------------------------
// Horizon Configuration
// ---------------------------------------------------------------------------

/// Configuration for a single temporal horizon.
#[derive(Debug, Clone)]
pub struct HorizonConfig {
    /// Label for this horizon.
    pub name: String,
    /// Time step in seconds.
    pub dt: f64,
    /// Number of particles.
    pub n_particles: usize,
    /// ESS resampling threshold (fraction of N).
    pub resample_threshold: f64,
    /// CUSUM config for anomaly monitoring at this time scale.
    pub cusum_config: CusumConfig,
}

/// Default tactical horizon (H1).
pub fn tactical_config() -> HorizonConfig {
    HorizonConfig {
        name: "H1_tactical".to_string(),
        dt: 0.1,
        n_particles: 100,
        resample_threshold: 0.5,
        cusum_config: CusumConfig {
            threshold_h: 0.3,
            min_samples: 5,
        },
    }
}

/// Default operational horizon (H2).
pub fn operational_config() -> HorizonConfig {
    HorizonConfig {
        name: "H2_operational".to_string(),
        dt: 5.0,
        n_particles: 500,
        resample_threshold: 0.5,
        cusum_config: CusumConfig {
            threshold_h: 0.5,
            min_samples: 10,
        },
    }
}

/// Default strategic horizon (H3).
pub fn strategic_config() -> HorizonConfig {
    HorizonConfig {
        name: "H3_strategic".to_string(),
        dt: 60.0,
        n_particles: 2000,
        resample_threshold: 0.5,
        cusum_config: CusumConfig {
            threshold_h: 0.8,
            min_samples: 20,
        },
    }
}

// ---------------------------------------------------------------------------
// Single-Horizon Filter
// ---------------------------------------------------------------------------

/// A particle filter wrapped with horizon-specific monitoring.
#[derive(Debug, Clone)]
pub struct HorizonFilter {
    /// Configuration.
    pub config: HorizonConfig,
    /// The underlying particle navigation filter.
    pub filter: ParticleNavFilter,
    /// Rolling history of estimated positions (for CUSUM / Hurst).
    pub position_history: Vec<f64>,
    /// Maximum history length.
    pub max_history: usize,
    /// Current best-estimate position.
    pub current_position: Vector3<f64>,
    /// Current best-estimate velocity.
    pub current_velocity: Vector3<f64>,
    /// Current regime probabilities.
    pub regime_probs: [f64; 3],
}

impl HorizonFilter {
    /// Create a new horizon filter at the given initial position.
    pub fn new(config: HorizonConfig, initial_pos: Vector3<f64>) -> Self {
        let filter = ParticleNavFilter::new(config.n_particles, initial_pos);
        Self {
            config,
            filter,
            position_history: Vec::with_capacity(200),
            max_history: 200,
            current_position: initial_pos,
            current_velocity: Vector3::zeros(),
            regime_probs: [1.0, 0.0, 0.0],
        }
    }

    /// Run one step of the horizon filter.
    pub fn step(
        &mut self,
        observations: &[Observation],
        threat_bearing: &Vector3<f64>,
        vel_gain: f64,
    ) -> (Vector3<f64>, Vector3<f64>, [f64; 3]) {
        let (pos, vel, probs) =
            self.filter
                .step(observations, threat_bearing, vel_gain, self.config.dt);

        self.current_position = pos;
        self.current_velocity = vel;
        self.regime_probs = probs;

        // Record position norm for CUSUM monitoring.
        self.position_history.push(pos.norm());
        if self.position_history.len() > self.max_history {
            self.position_history.remove(0);
        }

        (pos, vel, probs)
    }

    /// Check for anomalies at this horizon's time scale.
    pub fn check_anomaly(&self) -> (bool, i32, f64) {
        anomaly::cusum_test(
            &self.position_history,
            self.config.cusum_config.threshold_h,
            self.config.cusum_config.min_samples,
        )
    }

    /// Get the dominant regime.
    pub fn dominant_regime(&self) -> Regime {
        let idx = self
            .regime_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Regime::from_index(idx as u8).unwrap_or(Regime::Patrol)
    }
}

// ---------------------------------------------------------------------------
// Multi-Horizon Manager
// ---------------------------------------------------------------------------

/// Constraint passed from a higher horizon to a lower one.
#[derive(Debug, Clone)]
pub struct HorizonConstraint {
    /// Suggested waypoint position from the higher horizon.
    pub waypoint: Vector3<f64>,
    /// Suggested regime from the higher horizon.
    pub suggested_regime: Regime,
    /// Confidence in the constraint (0.0–1.0).
    pub confidence: f64,
}

/// Manages three particle filters at different temporal scales with
/// top-down information cascade.
#[derive(Debug, Clone)]
pub struct TemporalManager {
    /// Strategic horizon (H3) — mission-level.
    pub strategic: HorizonFilter,
    /// Operational horizon (H2) — formation-level.
    pub operational: HorizonFilter,
    /// Tactical horizon (H1) — immediate.
    pub tactical: HorizonFilter,
    /// Step counter for throttling higher horizons.
    step_count: u64,
    /// How many tactical steps per operational step.
    pub tactical_per_operational: u64,
    /// How many operational steps per strategic step.
    pub operational_per_strategic: u64,
}

impl TemporalManager {
    /// Create a new temporal manager with default configurations.
    pub fn new(initial_pos: Vector3<f64>) -> Self {
        Self {
            strategic: HorizonFilter::new(strategic_config(), initial_pos),
            operational: HorizonFilter::new(operational_config(), initial_pos),
            tactical: HorizonFilter::new(tactical_config(), initial_pos),
            step_count: 0,
            tactical_per_operational: 50,  // 5.0s / 0.1s
            operational_per_strategic: 12, // 60s / 5s
        }
    }

    /// Create with custom configurations.
    pub fn with_configs(
        tactical_cfg: HorizonConfig,
        operational_cfg: HorizonConfig,
        strategic_cfg: HorizonConfig,
        initial_pos: Vector3<f64>,
    ) -> Self {
        let tpo = (operational_cfg.dt / tactical_cfg.dt).round().max(1.0) as u64;
        let ops = (strategic_cfg.dt / operational_cfg.dt).round().max(1.0) as u64;

        Self {
            strategic: HorizonFilter::new(strategic_cfg, initial_pos),
            operational: HorizonFilter::new(operational_cfg, initial_pos),
            tactical: HorizonFilter::new(tactical_cfg, initial_pos),
            step_count: 0,
            tactical_per_operational: tpo,
            operational_per_strategic: ops,
        }
    }

    /// Run one tactical step, and conditionally update higher horizons.
    ///
    /// Returns `(position, velocity, regime_probs)` from the tactical
    /// filter, plus any constraints cascaded from higher horizons.
    pub fn step(
        &mut self,
        observations: &[Observation],
        threat_bearing: &Vector3<f64>,
        vel_gain: f64,
    ) -> (
        Vector3<f64>,
        Vector3<f64>,
        [f64; 3],
        Option<HorizonConstraint>,
    ) {
        self.step_count += 1;

        let mut constraint = None;

        // Strategic update (least frequent).
        if self
            .step_count
            .is_multiple_of(self.tactical_per_operational * self.operational_per_strategic)
        {
            self.strategic.step(observations, threat_bearing, vel_gain);
            // Strategic provides constraint to operational.
            constraint = Some(HorizonConstraint {
                waypoint: self.strategic.current_position,
                suggested_regime: self.strategic.dominant_regime(),
                confidence: self
                    .strategic
                    .regime_probs
                    .iter()
                    .cloned()
                    .fold(0.0_f64, f64::max),
            });
        }

        // Operational update.
        if self
            .step_count
            .is_multiple_of(self.tactical_per_operational)
        {
            // Apply strategic constraint as a virtual observation if available.
            self.operational
                .step(observations, threat_bearing, vel_gain);

            // Operational provides constraint to tactical.
            if constraint.is_none() {
                constraint = Some(HorizonConstraint {
                    waypoint: self.operational.current_position,
                    suggested_regime: self.operational.dominant_regime(),
                    confidence: self
                        .operational
                        .regime_probs
                        .iter()
                        .cloned()
                        .fold(0.0_f64, f64::max),
                });
            }
        }

        // Tactical update (every step).
        let (pos, vel, probs) = self.tactical.step(observations, threat_bearing, vel_gain);

        (pos, vel, probs, constraint)
    }

    /// Check for anomalies across all horizons.
    ///
    /// Returns a vec of `(horizon_name, is_break, direction, cusum_value)`.
    pub fn check_all_anomalies(&self) -> Vec<(String, bool, i32, f64)> {
        let mut results = Vec::new();
        for h in [&self.tactical, &self.operational, &self.strategic] {
            let (is_break, dir, val) = h.check_anomaly();
            if is_break {
                results.push((h.config.name.clone(), is_break, dir, val));
            }
        }
        results
    }

    /// Get current estimates from all three horizons.
    pub fn all_estimates(&self) -> [(Vector3<f64>, Vector3<f64>, [f64; 3]); 3] {
        [
            (
                self.tactical.current_position,
                self.tactical.current_velocity,
                self.tactical.regime_probs,
            ),
            (
                self.operational.current_position,
                self.operational.current_velocity,
                self.operational.regime_probs,
            ),
            (
                self.strategic.current_position,
                self.strategic.current_velocity,
                self.strategic.regime_probs,
            ),
        ]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizon_filter_step() {
        let cfg = tactical_config();
        let mut h = HorizonFilter::new(cfg, Vector3::new(0.0, 0.0, -50.0));
        let obs = vec![Observation::Barometer {
            altitude: 50.0,
            timestamp: 0.0,
        }];
        let tb = Vector3::zeros();
        let (pos, _vel, probs) = h.step(&obs, &tb, 1.0);
        assert!(pos.norm() < 200.0);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8);
    }

    #[test]
    fn temporal_manager_multi_step() {
        let mut tm = TemporalManager::new(Vector3::new(0.0, 0.0, -50.0));
        let obs = vec![Observation::Barometer {
            altitude: 50.0,
            timestamp: 0.0,
        }];
        let tb = Vector3::new(1.0, 0.0, 0.0);

        // Run enough steps to trigger operational and strategic updates.
        for _ in 0..700 {
            let (pos, _vel, probs, _constraint) = tm.step(&obs, &tb, 1.0);
            assert!(pos.norm() < 1e6);
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn temporal_manager_all_estimates() {
        let tm = TemporalManager::new(Vector3::new(10.0, 20.0, -30.0));
        let ests = tm.all_estimates();
        // All should be near initial position.
        for (pos, _vel, _probs) in &ests {
            assert!(pos.norm() < 200.0);
        }
    }

    #[test]
    fn dominant_regime_default_is_patrol() {
        let cfg = tactical_config();
        let h = HorizonFilter::new(cfg, Vector3::zeros());
        assert_eq!(h.dominant_regime(), Regime::Patrol);
    }

    #[test]
    fn check_anomaly_on_constant_series() {
        // Test the anomaly detector directly with a perfectly stable
        // series — this verifies the CUSUM integration without the
        // noise injected by the particle filter.
        let cfg = tactical_config();
        let mut h = HorizonFilter::new(cfg, Vector3::zeros());
        // Inject a constant position history manually.
        h.position_history = vec![10.0; 50];
        let (is_break, _, _) = h.check_anomaly();
        assert!(!is_break);
    }
}
