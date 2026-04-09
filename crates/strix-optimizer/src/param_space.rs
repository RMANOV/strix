//! Parameter space definition and mapping to/from SwarmConfig.
//!
//! Defines the 54-dimensional continuous/discrete parameter space that the
//! SMCO optimizer searches.  The central bridge is `to_swarm_config()` which
//! converts a flat `ParamVec` back to a fully-populated `SwarmConfig`.
//!
//! ## Parameter index layout (54 total)
//!
//! | Range   | Sub-system               |
//! |---------|--------------------------|
//! | 0–1     | Swarm topology           |
//! | 2–3     | Auction                  |
//! | 4–5     | CUSUM                    |
//! | 6–8     | Regime detection         |
//! | 9–12    | Hysteresis               |
//! | 13–22   | Intent pipeline          |
//! | 23–26   | Formation                |
//! | 27–32   | CBF                      |
//! | 33–44   | Process noise (patrol 6, engage 6) |
//! | 45–50   | Process noise evade      |
//! | 51      | Fear                     |
//! | 52–53   | Pheromone field          |

use rand::Rng;
use strix_core::{
    anomaly::CusumConfig,
    cbf::CbfConfig,
    formation::FormationConfig,
    hysteresis::HysteresisConfig,
    intent::IntentConfig,
    particle_nav::{ProcessNoiseConfig, RegimeNoise},
    regime::DetectionConfig,
};
use strix_swarm::tick::SwarmConfig;

// ── Types ─────────────────────────────────────────────────────────────────

/// A flat vector of parameter values. Length == `ParamSpace::len()`.
pub type ParamVec = Vec<f64>;

/// Domain of a single parameter.
#[derive(Debug, Clone)]
pub enum ParamKind {
    /// Continuous parameter with inclusive bounds.
    Continuous { min: f64, max: f64 },
    /// Discrete integer parameter; stored as f64, rounded on use.
    Discrete { min: i64, max: i64 },
}

/// Definition of a single optimizable parameter.
#[derive(Debug, Clone)]
pub struct ParamDef {
    /// Human-readable name (maps to SwarmConfig field path).
    pub name: &'static str,
    /// Domain.
    pub kind: ParamKind,
    /// Default value (from `SwarmConfig::default()`).
    pub default: f64,
}

impl ParamDef {
    fn continuous(name: &'static str, min: f64, max: f64, default: f64) -> Self {
        Self {
            name,
            kind: ParamKind::Continuous { min, max },
            default,
        }
    }

    fn discrete(name: &'static str, min: i64, max: i64, default: i64) -> Self {
        Self {
            name,
            kind: ParamKind::Discrete { min, max },
            default: default as f64,
        }
    }

    /// Clamp `v` to this parameter's domain.
    pub fn clamp(&self, v: f64) -> f64 {
        match &self.kind {
            ParamKind::Continuous { min, max } => v.clamp(*min, *max),
            ParamKind::Discrete { min, max } => (v.round() as i64).clamp(*min, *max) as f64,
        }
    }

    /// Sample a random value from this parameter's domain.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        match &self.kind {
            ParamKind::Continuous { min, max } => min + rng.random::<f64>() * (max - min),
            ParamKind::Discrete { min, max } => rng.random_range(*min..=*max) as f64,
        }
    }
}

// ── ParamSpace ────────────────────────────────────────────────────────────

/// Full description of the searchable parameter space.
#[derive(Debug, Clone)]
pub struct ParamSpace {
    pub params: Vec<ParamDef>,
}

impl ParamSpace {
    /// Number of dimensions.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Default point (matches `SwarmConfig::default()`).
    pub fn defaults(&self) -> ParamVec {
        self.params.iter().map(|p| p.default).collect()
    }

    /// Clamp every dimension of `v` to its declared bounds.
    pub fn clamp(&self, v: &ParamVec) -> ParamVec {
        assert_eq!(v.len(), self.len(), "ParamVec length mismatch");
        v.iter()
            .zip(&self.params)
            .map(|(&val, def)| def.clamp(val))
            .collect()
    }

    /// Sample a uniformly-random point from the parameter space.
    pub fn sample_random<R: Rng>(&self, rng: &mut R) -> ParamVec {
        self.params.iter().map(|p| p.sample(rng)).collect()
    }

    /// Perturb `v` by Gaussian noise scaled by `sigma` (fraction of range).
    /// Result is clamped.
    pub fn perturb<R: Rng>(&self, v: &ParamVec, sigma: f64, rng: &mut R) -> ParamVec {
        assert_eq!(v.len(), self.len(), "ParamVec length mismatch");
        let mut out = v.clone();
        for (i, def) in self.params.iter().enumerate() {
            let range = match &def.kind {
                ParamKind::Continuous { min, max } => max - min,
                ParamKind::Discrete { min, max } => (max - min) as f64,
            };
            let noise: f64 = rng.sample(rand_distr::StandardNormal);
            out[i] = def.clamp(out[i] + noise * sigma * range);
        }
        out
    }

    /// Convert a `ParamVec` to a fully-populated `SwarmConfig`.
    ///
    /// This is the critical bridge: the optimizer produces a flat vector;
    /// this function injects it into the SwarmConfig type hierarchy.
    pub fn to_swarm_config(&self, v: &ParamVec) -> SwarmConfig {
        assert_eq!(v.len(), self.len(), "ParamVec length mismatch");

        let p = |i: usize| v[i];
        let pu = |i: usize| v[i].round() as usize;

        // ── 0–1: Swarm topology ────────────────────────────────────────
        let n_particles = pu(0).max(10);
        let n_threat_particles = pu(1).max(10);

        // ── 2–3: Auction ───────────────────────────────────────────────
        let auction_interval = (v[2].round() as u32).max(1);
        let gossip_fanout = pu(3).max(1);

        // ── 4–5: CUSUM ────────────────────────────────────────────────
        let cusum_config = CusumConfig {
            threshold_h: p(4),
            min_samples: pu(5).max(3),
        };

        // ── 6–8: Regime detection ─────────────────────────────────────
        let detection_config = DetectionConfig {
            engage_distance: p(6),
            evade_distance: p(7),
            closing_rate_threshold: p(8),
        };

        // ── 9–12: Hysteresis ──────────────────────────────────────────
        let hysteresis_config = HysteresisConfig {
            min_dwell_secs: p(9),
            cooldown_secs: p(10),
            max_transitions_per_window: v[11].round() as u32,
            rate_window_secs: p(12),
        };

        // ── 13–22: Intent pipeline ────────────────────────────────────
        // Weights must sum to 1; we renormalise the three primary weights.
        let wh = p(15).max(0.0);
        let wc = p(16).max(0.0);
        let wv = p(17).max(0.0);
        let wsum = wh + wc + wv;
        let (wh, wc, wv) = if wsum < 1e-9 {
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        } else {
            (wh / wsum, wc / wsum, wv / wsum)
        };
        let intent_config = IntentConfig {
            hurst_purposeful: p(13),
            hurst_retreating: p(14),
            w_hurst: wh,
            w_closing: wc,
            w_volatility: wv,
            closing_accel_threshold: p(18),
            vol_compression_threshold: p(19),
            w_coherence: p(20),
            attack_threshold: p(21),
            retreat_threshold: p(22),
        };

        // ── 23–26: Formation ──────────────────────────────────────────
        let formation_config = FormationConfig {
            spacing: p(23),
            vee_angle_deg: p(24),
            max_correction_speed: p(25),
            deadband: p(26),
        };

        // ── 27–32: CBF ────────────────────────────────────────────────
        let cbf_config = Some(CbfConfig {
            min_separation: p(27),
            altitude_floor_ned: p(28),
            altitude_ceiling_ned: p(29),
            alpha: p(30),
            max_correction: p(31),
        });
        // p(32) = cbf_enabled flag (>0.5 = enabled)
        let cbf_config = if p(32) > 0.5 { cbf_config } else { None };

        // Process noise params (indices 33–50) extracted separately via noise_config()

        // ── 51: Fear ──────────────────────────────────────────────────
        let fear = p(51);

        // ── 52–53: Pheromone ──────────────────────────────────────────
        let pheromone_resolution = p(52);
        let pheromone_decay_rate = p(53);

        let default = SwarmConfig::default();
        SwarmConfig {
            n_particles,
            n_threat_particles,
            auction_interval,
            cusum_config,
            detection_config,
            pheromone_resolution,
            pheromone_decay_rate,
            gossip_fanout,
            default_capabilities: default.default_capabilities,
            hysteresis_config,
            intent_config,
            fear,
            formation_type: default.formation_type,
            formation_config,
            roe_engine: default.roe_engine,
            cbf_config,
            no_fly_zones: Vec::new(),
            criticality_config: default.criticality_config,
            ew_stale_age: default.ew_stale_age,
            gossip_interval: default.gossip_interval,
            formation_interval: default.formation_interval,
            criticality_interval: default.criticality_interval,
            order_params_interval: default.order_params_interval,
            adaptive_gossip: default.adaptive_gossip,
            gbp_config: default.gbp_config,
        }
    }

    /// Extract the `ProcessNoiseConfig` from a `ParamVec` (indices 33–50).
    pub fn noise_config(&self, v: &ParamVec) -> ProcessNoiseConfig {
        assert_eq!(v.len(), self.len(), "ParamVec length mismatch");
        ProcessNoiseConfig {
            patrol: regime_noise(v, 33),
            engage: regime_noise(v, 39),
            evade: regime_noise(v, 45),
        }
    }
}

fn regime_noise(v: &ParamVec, start: usize) -> RegimeNoise {
    RegimeNoise {
        pos_noise: [v[start], v[start + 1], v[start + 2]],
        vel_noise: [v[start + 3], v[start + 4], v[start + 5]],
    }
}

// ── strix_full() ──────────────────────────────────────────────────────────

/// Build the full 54-parameter space for STRIX SwarmConfig optimisation.
///
/// Defaults match `SwarmConfig::default()` exactly.
pub fn strix_full() -> ParamSpace {
    let params = vec![
        // ── 0–1: Swarm topology ───────────────────────────────────────
        ParamDef::discrete("n_particles", 50, 2000, 200),
        ParamDef::discrete("n_threat_particles", 20, 500, 100),
        // ── 2–3: Auction / mesh ───────────────────────────────────────
        ParamDef::discrete("auction_interval", 1, 20, 5),
        ParamDef::discrete("gossip_fanout", 1, 10, 3),
        // ── 4–5: CUSUM ────────────────────────────────────────────────
        ParamDef::continuous("cusum_threshold_h", 0.1, 3.0, 0.5),
        ParamDef::discrete("cusum_min_samples", 5, 50, 10),
        // ── 6–8: Regime detection ─────────────────────────────────────
        ParamDef::continuous("engage_distance", 50.0, 2000.0, 500.0),
        ParamDef::continuous("evade_distance", 10.0, 500.0, 150.0),
        ParamDef::continuous("closing_rate_threshold", 0.1, 20.0, 2.0),
        // ── 9–12: Hysteresis ──────────────────────────────────────────
        ParamDef::continuous("min_dwell_secs", 0.1, 10.0, 2.0),
        ParamDef::continuous("cooldown_secs", 0.1, 5.0, 1.0),
        ParamDef::discrete("max_transitions_per_window", 1, 10, 3),
        ParamDef::continuous("rate_window_secs", 2.0, 60.0, 10.0),
        // ── 13–22: Intent pipeline ────────────────────────────────────
        ParamDef::continuous("hurst_purposeful", 0.50, 0.80, 0.55),
        ParamDef::continuous("hurst_retreating", 0.30, 0.50, 0.45),
        // Weights (renormalised in to_swarm_config so raw values are free)
        ParamDef::continuous("w_hurst", 0.05, 0.90, 0.40),
        ParamDef::continuous("w_closing", 0.05, 0.90, 0.35),
        ParamDef::continuous("w_volatility", 0.05, 0.90, 0.25),
        ParamDef::continuous("closing_accel_threshold", 0.05, 5.0, 0.5),
        ParamDef::continuous("vol_compression_threshold", 0.1, 1.0, 0.5),
        ParamDef::continuous("w_coherence", 0.0, 0.5, 0.0),
        ParamDef::continuous("attack_threshold", 0.1, 0.8, 0.3),
        ParamDef::continuous("retreat_threshold", -0.8, -0.1, -0.3),
        // ── 23–26: Formation ──────────────────────────────────────────
        ParamDef::continuous("formation_spacing", 5.0, 100.0, 15.0),
        ParamDef::continuous("vee_angle_deg", 10.0, 60.0, 30.0),
        ParamDef::continuous("max_correction_speed", 0.5, 20.0, 5.0),
        ParamDef::continuous("formation_deadband", 0.1, 10.0, 1.0),
        // ── 27–32: CBF ────────────────────────────────────────────────
        ParamDef::continuous("cbf_min_separation", 1.0, 30.0, 5.0),
        ParamDef::continuous("cbf_altitude_floor_ned", -1000.0, -50.0, -500.0),
        ParamDef::continuous("cbf_altitude_ceiling_ned", -50.0, -1.0, -5.0),
        ParamDef::continuous("cbf_alpha", 0.1, 5.0, 1.0),
        ParamDef::continuous("cbf_max_correction", 1.0, 50.0, 10.0),
        ParamDef::continuous("cbf_enabled", 0.0, 1.0, 1.0), // >0.5 = on
        // ── 33–38: Patrol noise ───────────────────────────────────────
        ParamDef::continuous("patrol_pos_noise_x", 0.01, 1.0, 0.05),
        ParamDef::continuous("patrol_pos_noise_y", 0.01, 1.0, 0.05),
        ParamDef::continuous("patrol_pos_noise_z", 0.01, 0.5, 0.02),
        ParamDef::continuous("patrol_vel_noise_x", 0.01, 2.0, 0.10),
        ParamDef::continuous("patrol_vel_noise_y", 0.01, 2.0, 0.10),
        ParamDef::continuous("patrol_vel_noise_z", 0.01, 1.0, 0.05),
        // ── 39–44: Engage noise ───────────────────────────────────────
        ParamDef::continuous("engage_pos_noise_x", 0.05, 2.0, 0.15),
        ParamDef::continuous("engage_pos_noise_y", 0.05, 2.0, 0.15),
        ParamDef::continuous("engage_pos_noise_z", 0.02, 1.0, 0.08),
        ParamDef::continuous("engage_vel_noise_x", 0.05, 3.0, 0.30),
        ParamDef::continuous("engage_vel_noise_y", 0.05, 3.0, 0.30),
        ParamDef::continuous("engage_vel_noise_z", 0.02, 1.5, 0.15),
        // ── 45–50: Evade noise ────────────────────────────────────────
        ParamDef::continuous("evade_pos_noise_x", 0.1, 5.0, 0.40),
        ParamDef::continuous("evade_pos_noise_y", 0.1, 5.0, 0.40),
        ParamDef::continuous("evade_pos_noise_z", 0.05, 2.0, 0.20),
        ParamDef::continuous("evade_vel_noise_x", 0.2, 8.0, 0.80),
        ParamDef::continuous("evade_vel_noise_y", 0.2, 8.0, 0.80),
        ParamDef::continuous("evade_vel_noise_z", 0.1, 4.0, 0.40),
        // ── 51: Fear ──────────────────────────────────────────────────
        ParamDef::continuous("fear", 0.0, 1.0, 0.0),
        // ── 52–53: Pheromone field ────────────────────────────────────
        ParamDef::continuous("pheromone_resolution", 1.0, 50.0, 10.0),
        ParamDef::continuous("pheromone_decay_rate", 0.001, 0.5, 0.05),
    ];

    ParamSpace { params }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_count_is_54() {
        let space = strix_full();
        assert_eq!(space.len(), 54, "Expected 54 params, got {}", space.len());
    }

    #[test]
    fn defaults_roundtrip_to_valid_swarm_config() {
        let space = strix_full();
        let defaults = space.defaults();
        assert_eq!(defaults.len(), 54);

        let cfg = space.to_swarm_config(&defaults);

        // Spot-check defaults match SwarmConfig::default()
        assert_eq!(cfg.n_particles, 200);
        assert_eq!(cfg.n_threat_particles, 100);
        assert_eq!(cfg.auction_interval, 5);
        assert_eq!(cfg.gossip_fanout, 3);
        assert!((cfg.cusum_config.threshold_h - 0.5).abs() < 1e-9);
        assert_eq!(cfg.cusum_config.min_samples, 10);
        assert!((cfg.detection_config.engage_distance - 500.0).abs() < 1e-9);
        assert!((cfg.detection_config.evade_distance - 150.0).abs() < 1e-9);
        assert!((cfg.fear - 0.0).abs() < 1e-9);
        assert!(cfg.cbf_config.is_some()); // cbf_enabled default = 1.0 > 0.5
    }

    #[test]
    fn clamp_respects_bounds() {
        let space = strix_full();
        // Build a wildly out-of-bounds vector.
        let wild: ParamVec = space.params.iter().map(|_| 1e12).collect();
        let clamped = space.clamp(&wild);

        for (i, (val, def)) in clamped.iter().zip(&space.params).enumerate() {
            match &def.kind {
                ParamKind::Continuous { min: _, max } => {
                    assert!(
                        *val <= *max + 1e-9,
                        "param[{}] {} = {} exceeds max {}",
                        i,
                        def.name,
                        val,
                        max
                    );
                }
                ParamKind::Discrete { min: _, max } => {
                    assert!(
                        (*val as i64) <= *max,
                        "param[{}] {} = {} exceeds max {}",
                        i,
                        def.name,
                        val,
                        max
                    );
                }
            }
        }

        // Also clamp from below.
        let low: ParamVec = space.params.iter().map(|_| -1e12).collect();
        let clamped_low = space.clamp(&low);
        for (i, (val, def)) in clamped_low.iter().zip(&space.params).enumerate() {
            match &def.kind {
                ParamKind::Continuous { min, max: _ } => {
                    assert!(
                        *val >= *min - 1e-9,
                        "param[{}] {} = {} below min {}",
                        i,
                        def.name,
                        val,
                        min
                    );
                }
                ParamKind::Discrete { min, max: _ } => {
                    assert!(
                        (*val as i64) >= *min,
                        "param[{}] {} = {} below min {}",
                        i,
                        def.name,
                        val,
                        min
                    );
                }
            }
        }
    }

    #[test]
    fn sample_random_within_bounds() {
        let space = strix_full();
        let mut rng = rand::rng();
        let sample = space.sample_random(&mut rng);
        let clamped = space.clamp(&sample);
        // If all within bounds, clamping should not change values.
        for (i, (a, b)) in sample.iter().zip(&clamped).enumerate() {
            assert!(
                (a - b).abs() < 1e-9,
                "param[{}] {} sample {} out of bounds (clamped to {})",
                i,
                space.params[i].name,
                a,
                b
            );
        }
    }

    #[test]
    fn intent_weights_normalised_in_config() {
        let space = strix_full();
        let mut v = space.defaults();
        // Set weights to arbitrary unnormalised values.
        v[15] = 2.0; // w_hurst
        v[16] = 3.0; // w_closing
        v[17] = 5.0; // w_volatility
        let cfg = space.to_swarm_config(&v);
        let sum = cfg.intent_config.w_hurst
            + cfg.intent_config.w_closing
            + cfg.intent_config.w_volatility;
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Intent weights sum = {}, expected 1.0",
            sum
        );
    }

    #[test]
    fn cbf_disabled_when_flag_below_half() {
        let space = strix_full();
        let mut v = space.defaults();
        v[32] = 0.3; // cbf_enabled < 0.5
        let cfg = space.to_swarm_config(&v);
        assert!(cfg.cbf_config.is_none());
    }

    #[test]
    fn noise_config_matches_params() {
        let space = strix_full();
        let defaults = space.defaults();
        let nc = space.noise_config(&defaults);
        // Patrol defaults: pos [0.05, 0.05, 0.02], vel [0.10, 0.10, 0.05]
        assert!((nc.patrol.pos_noise[0] - 0.05).abs() < 1e-9);
        assert!((nc.evade.vel_noise[2] - 0.40).abs() < 1e-9);
    }

    #[test]
    fn perturb_stays_in_bounds() {
        let space = strix_full();
        let mut rng = rand::rng();
        let base = space.defaults();
        for _ in 0..20 {
            let p = space.perturb(&base, 0.3, &mut rng);
            let c = space.clamp(&p);
            for (i, (a, b)) in p.iter().zip(&c).enumerate() {
                assert!(
                    (a - b).abs() < 1e-9,
                    "param[{}] {} perturbed value {} out of bounds",
                    i,
                    space.params[i].name,
                    a
                );
            }
        }
    }
}
