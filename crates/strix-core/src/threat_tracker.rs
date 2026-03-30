//! Enemy tracking via a **dual** particle filter.
//!
//! The KEY INNOVATION of STRIX: two particle filters run simultaneously.
//!
//! - **Friendly PF** (see [`particle_nav`]) tracks our drone positions.
//! - **Enemy PF** (this module) tracks hypotheses about enemy intent and
//!   position using its own regime model:
//!   - `DEFEND`: mean-reverting around a position (stationary enemy)
//!   - `COUNTER_ATTACK`: velocity tracks toward our fleet centroid
//!   - `RETREAT`: high-noise moving away
//!
//! The module also provides predictive projection and Hurst-based enemy
//! movement classification.

use std::collections::HashMap;

use nalgebra::Vector3;
use ndarray::Array2;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::particle_common::{
    effective_sample_size, gaussian_likelihood, normalize_weights, systematic_resample_6d,
};
use crate::state::{ThreatRegime, ThreatState};

fn vector_is_finite(vector: &Vector3<f64>) -> bool {
    vector.iter().all(|value| value.is_finite())
}

// ---------------------------------------------------------------------------
// Threat noise profiles
// ---------------------------------------------------------------------------

/// Per-regime noise parameters for the threat particle filter.
#[derive(Debug, Clone)]
pub struct ThreatNoiseConfig {
    /// DEFEND — low noise, mean-reverting.
    pub defend_pos_noise: f64,
    pub defend_vel_noise: f64,
    /// COUNTER_ATTACK — medium noise, directed.
    pub counter_pos_noise: f64,
    pub counter_vel_noise: f64,
    /// RETREAT — high noise, away.
    pub retreat_pos_noise: f64,
    pub retreat_vel_noise: f64,
}

impl Default for ThreatNoiseConfig {
    fn default() -> Self {
        Self {
            defend_pos_noise: 0.05,
            defend_vel_noise: 0.1,
            counter_pos_noise: 0.2,
            counter_vel_noise: 0.4,
            retreat_pos_noise: 0.5,
            retreat_vel_noise: 1.0,
        }
    }
}

impl ThreatNoiseConfig {
    /// Amplify threat tracking uncertainty under fear.
    /// Higher fear → wider uncertainty bounds.
    pub fn scaled_by_fear(&self, f: f64) -> Self {
        let f = if f.is_nan() || f.is_infinite() {
            0.0
        } else {
            f.clamp(0.0, 1.0)
        };
        Self {
            defend_pos_noise: self.defend_pos_noise * (1.0 + f * 0.3),
            defend_vel_noise: self.defend_vel_noise * (1.0 + f * 0.3),
            counter_pos_noise: self.counter_pos_noise * (1.0 + f * 0.5),
            counter_vel_noise: self.counter_vel_noise * (1.0 + f * 0.5),
            retreat_pos_noise: self.retreat_pos_noise * (1.0 + f * 0.2),
            retreat_vel_noise: self.retreat_vel_noise * (1.0 + f * 0.2),
        }
    }
}

// ---------------------------------------------------------------------------
// Threat sensor observations
// ---------------------------------------------------------------------------

/// Observation of a threat from one of our sensors.
#[derive(Debug, Clone)]
pub enum ThreatObservation {
    /// Radar return — position estimate with noise.
    Radar {
        position: Vector3<f64>,
        sigma: f64,
        timestamp: f64,
    },
    /// Visual detection — bearing + estimated range.
    Visual {
        bearing: Vector3<f64>,
        estimated_range: f64,
        sigma: f64,
        timestamp: f64,
    },
    /// Radio intercept — bearing only (relative to observing drone's position).
    RadioIntercept {
        bearing: Vector3<f64>,
        /// Position of the observing drone (bearing is relative to this point).
        observer_position: Vector3<f64>,
        sigma: f64,
        timestamp: f64,
    },
}

// ---------------------------------------------------------------------------
// Threat Particle Filter
// ---------------------------------------------------------------------------

/// Particle filter for tracking a single enemy entity.
#[derive(Debug, Clone)]
pub struct ThreatTracker {
    /// Nx6 particle array: `[x, y, z, vx, vy, vz]`.
    pub particles: Array2<f64>,
    /// Regime hypothesis per particle.
    pub regimes: Vec<u8>,
    /// Importance weights.
    pub weights: Vec<f64>,
    /// Noise configuration.
    pub noise_cfg: ThreatNoiseConfig,
    /// Threat track ID.
    pub threat_id: u32,
    /// 3x3 Markov transition matrix for threat regimes.
    pub transition_matrix: [[f64; 3]; 3],
    /// Pre-allocated scratch buffer — reused across `predict_threat` calls to avoid
    /// heap allocation on every tick.
    scratch_buf: Vec<[f64; 6]>,
    /// EMA-smoothed bearing per observer drone ID (keyed by u64 hash of observer pos).
    smoothed_bearings: HashMap<u64, Vector3<f64>>,
}

impl ThreatTracker {
    /// Create a new tracker with `n` particles centred on `initial_pos`.
    pub fn new(threat_id: u32, n: usize, initial_pos: Vector3<f64>) -> Self {
        let mut particles = Array2::<f64>::zeros((n, 6));
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).expect("Normal(0,1) has valid parameters");
        for i in 0..n {
            particles[[i, 0]] = initial_pos.x + normal.sample(&mut rng) * 2.0;
            particles[[i, 1]] = initial_pos.y + normal.sample(&mut rng) * 2.0;
            particles[[i, 2]] = initial_pos.z + normal.sample(&mut rng) * 1.0;
        }

        Self {
            particles,
            regimes: vec![ThreatRegime::Defend as u8; n],
            weights: vec![1.0 / n as f64; n],
            noise_cfg: ThreatNoiseConfig::default(),
            threat_id,
            transition_matrix: crate::state::default_threat_transition_matrix(),
            scratch_buf: vec![[0.0; 6]; n],
            smoothed_bearings: HashMap::new(),
        }
    }

    /// Propagate threat particles forward by `dt` seconds.
    ///
    /// `our_centroid` is the position of our fleet centroid, used by
    /// COUNTER_ATTACK particles to steer toward us.
    pub fn predict_threat(&mut self, our_centroid: &Vector3<f64>, dt: f64) {
        let n = self.particles.nrows();
        let dt = if dt.is_finite() && dt >= 0.0 { dt } else { 0.0 };
        let dt_sqrt = dt.max(1e-8).sqrt();
        let centroid = if vector_is_finite(our_centroid) {
            *our_centroid
        } else {
            Vector3::zeros()
        };
        let noise_cfg = self.noise_cfg.clone();

        // Reuse scratch buffer — resize only if particle count changed.
        let mut buf = std::mem::take(&mut self.scratch_buf);
        buf.resize(n, [0.0; 6]);
        for (i, p) in buf.iter_mut().enumerate() {
            *p = [
                self.particles[[i, 0]],
                self.particles[[i, 1]],
                self.particles[[i, 2]],
                self.particles[[i, 3]],
                self.particles[[i, 4]],
                self.particles[[i, 5]],
            ];
        }

        let regimes = self.regimes.clone();
        let normal = Normal::new(0.0, 1.0).expect("Normal(0,1) has valid parameters");

        let update_particle = |(i, p): (usize, &mut [f64; 6])| {
            let mut rng = rand::thread_rng();
            let regime = regimes[i];

            let rp: [f64; 3] = [
                normal.sample(&mut rng),
                normal.sample(&mut rng),
                normal.sample(&mut rng),
            ];
            let rv: [f64; 3] = [
                normal.sample(&mut rng),
                normal.sample(&mut rng),
                normal.sample(&mut rng),
            ];

            let vx = p[3];
            let vy = p[4];
            let vz = p[5];

            let (pos_noise, vel_noise) = match regime {
                0 => (noise_cfg.defend_pos_noise, noise_cfg.defend_vel_noise),
                1 => (noise_cfg.counter_pos_noise, noise_cfg.counter_vel_noise),
                _ => (noise_cfg.retreat_pos_noise, noise_cfg.retreat_vel_noise),
            };

            let (new_vx, new_vy, new_vz) = match regime {
                0 => {
                    // DEFEND — mean-reverting velocity (hold position)
                    (
                        0.5 * vx + rv[0] * vel_noise * dt_sqrt,
                        0.5 * vy + rv[1] * vel_noise * dt_sqrt,
                        0.5 * vz + rv[2] * vel_noise * dt_sqrt,
                    )
                }
                1 => {
                    // COUNTER_ATTACK — velocity tracks toward our centroid
                    let pos_vec = Vector3::new(p[0], p[1], p[2]);
                    let to_us = centroid - pos_vec;
                    let dist = to_us.norm().max(1e-6);
                    let dir = to_us / dist;
                    let speed = 5.0; // assumed enemy closing speed
                    (
                        vx + 0.3 * (speed * dir.x - vx) * dt + rv[0] * vel_noise * dt_sqrt,
                        vy + 0.3 * (speed * dir.y - vy) * dt + rv[1] * vel_noise * dt_sqrt,
                        vz + 0.3 * (speed * dir.z - vz) * dt + rv[2] * vel_noise * dt_sqrt,
                    )
                }
                _ => {
                    // RETREAT — high-noise, moving away from our centroid
                    let pos_vec = Vector3::new(p[0], p[1], p[2]);
                    let away = pos_vec - centroid;
                    let dist = away.norm().max(1e-6);
                    let dir = away / dist;
                    let speed = 3.0;
                    (
                        vx + 0.2 * (speed * dir.x - vx) * dt + rv[0] * vel_noise * dt_sqrt,
                        vy + 0.2 * (speed * dir.y - vy) * dt + rv[1] * vel_noise * dt_sqrt,
                        vz + 0.2 * (speed * dir.z - vz) * dt + rv[2] * vel_noise * dt_sqrt,
                    )
                }
            };

            p[0] += new_vx * dt + rp[0] * pos_noise * dt_sqrt;
            p[1] += new_vy * dt + rp[1] * pos_noise * dt_sqrt;
            p[2] += new_vz * dt + rp[2] * pos_noise * dt_sqrt;
            p[3] = new_vx;
            p[4] = new_vy;
            p[5] = new_vz;
        };

        // Only parallelize above break-even; below 500 particles rayon overhead exceeds gain.
        if n > 500 {
            buf.par_iter_mut().enumerate().for_each(update_particle);
        } else {
            buf.iter_mut().enumerate().for_each(update_particle);
        }

        for (i, row) in buf.iter().enumerate().take(n) {
            for (j, val) in row.iter().enumerate() {
                self.particles[[i, j]] = *val;
            }
        }
        for i in 0..n {
            for j in 0..6 {
                if !self.particles[[i, j]].is_finite() {
                    self.particles[[i, j]] = 0.0;
                }
            }
        }
        // Restore the scratch buffer.
        self.scratch_buf = buf;
    }

    /// Update threat particle weights from sensor observations.
    pub fn update_threat(&mut self, observations: &[ThreatObservation]) {
        let n = self.particles.nrows();

        for obs in observations {
            match obs {
                ThreatObservation::Radar {
                    position,
                    sigma,
                    timestamp: _,
                } => {
                    if !vector_is_finite(position) || !sigma.is_finite() || *sigma < 1e-6 {
                        continue;
                    }
                    for i in 0..n {
                        let dx = self.particles[[i, 0]] - position.x;
                        let dy = self.particles[[i, 1]] - position.y;
                        let dz = self.particles[[i, 2]] - position.z;
                        let diff_sq = dx * dx + dy * dy + dz * dz;
                        self.weights[i] *= gaussian_likelihood(diff_sq, *sigma);
                    }
                }
                ThreatObservation::Visual {
                    bearing,
                    estimated_range,
                    sigma,
                    timestamp: _,
                } => {
                    if !vector_is_finite(bearing)
                        || !estimated_range.is_finite()
                        || *estimated_range < 0.0
                        || !sigma.is_finite()
                        || *sigma < 1e-6
                    {
                        continue;
                    }
                    let expected_pos = bearing * *estimated_range;
                    for i in 0..n {
                        let dx = self.particles[[i, 0]] - expected_pos.x;
                        let dy = self.particles[[i, 1]] - expected_pos.y;
                        let dz = self.particles[[i, 2]] - expected_pos.z;
                        let diff_sq = dx * dx + dy * dy + dz * dz;
                        self.weights[i] *= gaussian_likelihood(diff_sq, *sigma);
                    }
                }
                ThreatObservation::RadioIntercept {
                    bearing,
                    observer_position,
                    sigma,
                    timestamp: _,
                } => {
                    if !vector_is_finite(bearing)
                        || !vector_is_finite(observer_position)
                        || !sigma.is_finite()
                        || *sigma < 1e-6
                    {
                        continue;
                    }
                    // EMA-smooth the bearing using observer position bits as key.
                    let obs_key = observer_position.x.to_bits()
                        ^ observer_position.y.to_bits().wrapping_shl(21)
                        ^ observer_position.z.to_bits().wrapping_shl(42);
                    const BEARING_ALPHA: f64 = 0.3;
                    let smoothed = if let Some(prev) = self.smoothed_bearings.get(&obs_key) {
                        *bearing * BEARING_ALPHA + *prev * (1.0 - BEARING_ALPHA)
                    } else {
                        *bearing
                    };
                    // Renormalise after blending.
                    let smoothed_norm = smoothed.norm();
                    let smoothed_unit = if smoothed_norm > 1e-6 {
                        smoothed / smoothed_norm
                    } else {
                        *bearing
                    };
                    self.smoothed_bearings.insert(obs_key, smoothed_unit);

                    for i in 0..n {
                        let particle_pos = Vector3::new(
                            self.particles[[i, 0]],
                            self.particles[[i, 1]],
                            self.particles[[i, 2]],
                        );
                        // Bearing is relative to the observing drone's position.
                        let relative = particle_pos - observer_position;
                        let norm = relative.norm();
                        if norm > 1e-6 {
                            let unit = relative / norm;
                            let diff_sq = (unit - smoothed_unit).norm_squared();
                            self.weights[i] *= gaussian_likelihood(diff_sq, *sigma);
                        } else {
                            // Particle is at or very near the observer — assign minimum weight.
                            self.weights[i] *= 0.01;
                        }
                    }
                }
            }
        }

        // Detect filter collapse: if all raw weights are effectively zero,
        // the +1e-300 rescue produces uniform weights and ESS=N, hiding the
        // collapse from the resampler. Log a warning so upstream can react.
        let max_raw = self.weights.iter().cloned().fold(0.0_f64, f64::max);
        if max_raw < 1e-100 {
            let ess = effective_sample_size(&self.weights);
            tracing::warn!(
                max_weight = max_raw,
                ess = ess,
                threat_id = self.threat_id,
                "threat_tracker: weight collapse detected, ESS={}",
                ess
            );
        }

        // Normalise with underflow protection.
        normalize_weights(&mut self.weights);
    }

    /// Weighted mean threat position + velocity + regime probabilities.
    pub fn estimate_threat(&self) -> (Vector3<f64>, Vector3<f64>, [f64; 3]) {
        let n = self.particles.nrows();
        let mut pos = Vector3::zeros();
        let mut vel = Vector3::zeros();
        let mut regime_probs = [0.0_f64; 3];

        for i in 0..n {
            let w = self.weights[i];
            pos.x += self.particles[[i, 0]] * w;
            pos.y += self.particles[[i, 1]] * w;
            pos.z += self.particles[[i, 2]] * w;
            vel.x += self.particles[[i, 3]] * w;
            vel.y += self.particles[[i, 4]] * w;
            vel.z += self.particles[[i, 5]] * w;

            let r = (self.regimes[i] as usize).min(2);
            regime_probs[r] += w;
        }

        (pos, vel, regime_probs)
    }

    /// Project the threat position `dt_future` seconds into the future.
    ///
    /// Uses current weighted mean velocity for linear extrapolation.
    pub fn predict_future_threat(&self, dt_future: f64) -> Vector3<f64> {
        let (pos, vel, _) = self.estimate_threat();
        pos + vel * dt_future
    }

    /// Resample threat particles when ESS is too low.
    pub fn resample_if_needed(&mut self, ess_threshold_frac: f64) {
        let n = self.weights.len();
        let ess = effective_sample_size(&self.weights);
        if ess < ess_threshold_frac * n as f64 {
            self.resample();
        }
    }

    /// Force a systematic resample.
    ///
    /// Delegates to [`particle_common::systematic_resample_6d`] for the
    /// core O(N) systematic resampling algorithm.
    pub fn resample(&mut self) {
        systematic_resample_6d(&mut self.particles, &mut self.weights, &mut self.regimes);
    }

    /// Convert the current estimate to a [`ThreatState`].
    pub fn to_threat_state(&self) -> ThreatState {
        let (pos, vel, probs) = self.estimate_threat();
        let regime_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let regime = ThreatRegime::from_index(regime_idx as u8).unwrap_or(ThreatRegime::Defend);

        ThreatState {
            position: pos,
            velocity: vel,
            regime,
            weight: 1.0,
            threat_id: self.threat_id,
        }
    }

    /// Transition threat regimes using the Markov transition matrix.
    pub fn transition_regimes(&mut self) {
        let n = self.regimes.len();
        let mut rng = rand::thread_rng();
        let random_uniform: Vec<f64> = (0..n).map(|_| rng.gen()).collect();
        crate::regime::transition_regimes(
            &mut self.regimes,
            &self.transition_matrix,
            &random_uniform,
        );
    }

    /// Full predict–update–resample cycle.
    pub fn step(
        &mut self,
        our_centroid: &Vector3<f64>,
        observations: &[ThreatObservation],
        dt: f64,
    ) -> (Vector3<f64>, Vector3<f64>, [f64; 3]) {
        self.transition_regimes();
        self.predict_threat(our_centroid, dt);
        self.update_threat(observations);
        let est = self.estimate_threat();
        self.resample_if_needed(0.5);
        est
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_predict_moves_particles() {
        let mut tracker = ThreatTracker::new(1, 50, Vector3::new(100.0, 100.0, -30.0));
        let centroid = Vector3::new(0.0, 0.0, -50.0);
        tracker.predict_threat(&centroid, 0.1);

        // Smoke test: predict_threat must not panic. With random noise, at least
        // some particles will have moved from the initial x = 100.0.
        let any_moved = (0..50).any(|i| tracker.particles[[i, 0]] != 100.0);
        assert!(
            any_moved,
            "predict_threat should move at least one particle"
        );
    }

    #[test]
    fn tracker_predict_sanitizes_invalid_centroid_and_dt() {
        let mut tracker = ThreatTracker::new(1, 32, Vector3::new(10.0, 20.0, -5.0));
        tracker.predict_threat(&Vector3::new(f64::NAN, f64::INFINITY, 0.0), f64::NAN);
        assert!(tracker.particles.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn tracker_estimate_within_range() {
        let tracker = ThreatTracker::new(1, 200, Vector3::new(50.0, 50.0, -20.0));
        let (pos, _vel, probs) = tracker.estimate_threat();
        // Mean should be near initial position.
        assert!((pos.x - 50.0).abs() < 20.0);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8);
    }

    #[test]
    fn tracker_future_projection() {
        let tracker = ThreatTracker::new(1, 100, Vector3::new(0.0, 0.0, 0.0));
        let future = tracker.predict_future_threat(10.0);
        // With zero initial velocity + noise, should be near origin.
        assert!(future.norm() < 50.0);
    }

    #[test]
    fn momentum_score_advancing() {
        // Distance decreasing = enemy advancing → negative momentum.
        let history: Vec<f64> = (0..20).map(|i| 100.0 - i as f64 * 2.0).collect();
        let m = crate::uncertainty::momentum_score(&history, 20);
        assert!(m < 0.0);
    }

    #[test]
    fn momentum_score_retreating() {
        // Distance increasing = enemy retreating → positive momentum.
        let history: Vec<f64> = (0..20).map(|i| 10.0 + i as f64 * 2.0).collect();
        let m = crate::uncertainty::momentum_score(&history, 20);
        assert!(m > 0.0);
    }

    // ── Fear-scaled noise ──────────────────────────────────────────────

    #[test]
    fn test_threat_noise_scaled_by_fear_zero() {
        let cfg = ThreatNoiseConfig::default();
        let scaled = cfg.scaled_by_fear(0.0);
        assert!((scaled.defend_pos_noise - cfg.defend_pos_noise).abs() < 1e-12);
        assert!((scaled.counter_pos_noise - cfg.counter_pos_noise).abs() < 1e-12);
        assert!((scaled.retreat_pos_noise - cfg.retreat_pos_noise).abs() < 1e-12);
    }

    #[test]
    fn test_threat_noise_scaled_by_fear_one() {
        let cfg = ThreatNoiseConfig::default();
        let scaled = cfg.scaled_by_fear(1.0);
        assert!((scaled.defend_pos_noise - cfg.defend_pos_noise * 1.3).abs() < 1e-12);
        assert!((scaled.counter_pos_noise - cfg.counter_pos_noise * 1.5).abs() < 1e-12);
        assert!((scaled.retreat_pos_noise - cfg.retreat_pos_noise * 1.2).abs() < 1e-12);
    }

    #[test]
    fn test_threat_noise_scaled_by_fear_clamps() {
        let cfg = ThreatNoiseConfig::default();
        let s1 = cfg.scaled_by_fear(-0.5);
        let s2 = cfg.scaled_by_fear(0.0);
        assert!((s1.defend_pos_noise - s2.defend_pos_noise).abs() < 1e-12);
    }

    #[test]
    fn tracker_full_step() {
        let mut tracker = ThreatTracker::new(1, 100, Vector3::new(50.0, 50.0, -20.0));
        let centroid = Vector3::new(0.0, 0.0, -50.0);
        let obs = vec![ThreatObservation::Radar {
            position: Vector3::new(48.0, 52.0, -22.0),
            sigma: 5.0,
            timestamp: 0.0,
        }];
        let (pos, _vel, probs) = tracker.step(&centroid, &obs, 0.1);
        assert!(pos.norm() < 200.0);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8);
    }

    #[test]
    fn tracker_update_ignores_non_finite_observations() {
        let mut tracker = ThreatTracker::new(1, 32, Vector3::new(50.0, 50.0, -20.0));
        let obs = vec![
            ThreatObservation::Radar {
                position: Vector3::new(f64::NAN, 0.0, 0.0),
                sigma: 5.0,
                timestamp: 0.0,
            },
            ThreatObservation::Visual {
                bearing: Vector3::new(1.0, 0.0, 0.0),
                estimated_range: f64::INFINITY,
                sigma: 3.0,
                timestamp: 0.0,
            },
            ThreatObservation::RadioIntercept {
                bearing: Vector3::new(0.0, f64::NAN, 0.0),
                observer_position: Vector3::new(0.0, 0.0, 0.0),
                sigma: 0.5,
                timestamp: 0.0,
            },
        ];

        tracker.update_threat(&obs);

        let sum: f64 = tracker.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(tracker
            .weights
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0));
    }
}
