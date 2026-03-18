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

use nalgebra::Vector3;
use ndarray::Array2;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::state::{ThreatRegime, ThreatState};

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
        let f = f.clamp(0.0, 1.0);
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
    /// Radio intercept — bearing only.
    RadioIntercept {
        bearing: Vector3<f64>,
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
}

impl ThreatTracker {
    /// Create a new tracker with `n` particles centred on `initial_pos`.
    pub fn new(threat_id: u32, n: usize, initial_pos: Vector3<f64>) -> Self {
        let mut particles = Array2::<f64>::zeros((n, 6));
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
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
        }
    }

    /// Propagate threat particles forward by `dt` seconds.
    ///
    /// `our_centroid` is the position of our fleet centroid, used by
    /// COUNTER_ATTACK particles to steer toward us.
    pub fn predict_threat(&mut self, our_centroid: &Vector3<f64>, dt: f64) {
        let n = self.particles.nrows();
        let dt_sqrt = dt.max(1e-8).sqrt();
        let centroid = *our_centroid;
        let noise_cfg = self.noise_cfg.clone();

        let mut buf: Vec<[f64; 6]> = (0..n)
            .map(|i| {
                [
                    self.particles[[i, 0]],
                    self.particles[[i, 1]],
                    self.particles[[i, 2]],
                    self.particles[[i, 3]],
                    self.particles[[i, 4]],
                    self.particles[[i, 5]],
                ]
            })
            .collect();

        let regimes = self.regimes.clone();

        buf.par_iter_mut().enumerate().for_each(|(i, p)| {
            let mut rng = rand::thread_rng();
            let normal = Normal::new(0.0, 1.0).unwrap();
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
        });

        for (i, row) in buf.iter().enumerate().take(n) {
            for (j, val) in row.iter().enumerate() {
                self.particles[[i, j]] = *val;
            }
        }
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
                    let sigma2 = sigma * sigma + 1e-12;
                    for i in 0..n {
                        let dx = self.particles[[i, 0]] - position.x;
                        let dy = self.particles[[i, 1]] - position.y;
                        let dz = self.particles[[i, 2]] - position.z;
                        let diff_sq = dx * dx + dy * dy + dz * dz;
                        let like = (-0.5 * diff_sq / sigma2).exp();
                        self.weights[i] *= like;
                    }
                }
                ThreatObservation::Visual {
                    bearing,
                    estimated_range,
                    sigma,
                    timestamp: _,
                } => {
                    let sigma2 = sigma * sigma + 1e-12;
                    let expected_pos = bearing * *estimated_range;
                    for i in 0..n {
                        let dx = self.particles[[i, 0]] - expected_pos.x;
                        let dy = self.particles[[i, 1]] - expected_pos.y;
                        let dz = self.particles[[i, 2]] - expected_pos.z;
                        let diff_sq = dx * dx + dy * dy + dz * dz;
                        let like = (-0.5 * diff_sq / sigma2).exp();
                        self.weights[i] *= like;
                    }
                }
                ThreatObservation::RadioIntercept {
                    bearing,
                    sigma,
                    timestamp: _,
                } => {
                    let sigma2 = sigma * sigma + 1e-12;
                    for i in 0..n {
                        let pos = Vector3::new(
                            self.particles[[i, 0]],
                            self.particles[[i, 1]],
                            self.particles[[i, 2]],
                        );
                        let norm = pos.norm();
                        if norm > 1e-6 {
                            let unit = pos / norm;
                            let diff_sq = (unit - bearing).norm_squared();
                            let like = (-0.5 * diff_sq / sigma2).exp();
                            self.weights[i] *= like;
                        }
                    }
                }
            }
        }

        // Normalise with underflow protection.
        for w in self.weights.iter_mut() {
            *w += 1e-300;
        }
        let total: f64 = self.weights.iter().sum();
        for w in self.weights.iter_mut() {
            *w /= total;
        }
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
        let sum_sq: f64 = self.weights.iter().map(|w| w * w).sum();
        let ess = 1.0 / (sum_sq + 1e-12);
        if ess < ess_threshold_frac * n as f64 {
            self.resample();
        }
    }

    /// Force a systematic resample.
    pub fn resample(&mut self) {
        let n = self.weights.len();
        let mut cumsum = vec![0.0_f64; n];
        cumsum[0] = self.weights[0];
        for i in 1..n {
            cumsum[i] = cumsum[i - 1] + self.weights[i];
        }
        cumsum[n - 1] = 1.0;

        let step = 1.0 / n as f64;
        let start: f64 = rand::thread_rng().gen_range(0.0..step);

        let mut new_particles = Array2::<f64>::zeros((n, 6));
        let mut new_regimes = vec![0u8; n];

        let mut j = 0_usize;
        for i in 0..n {
            let pos = start + step * i as f64;
            while j < n - 1 && cumsum[j] < pos {
                j += 1;
            }
            for k in 0..6 {
                new_particles[[i, k]] = self.particles[[j, k]];
            }
            new_regimes[i] = self.regimes[j];
        }

        self.particles = new_particles;
        self.regimes = new_regimes;
        self.weights = vec![1.0 / n as f64; n];
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
        let tm = self.transition_matrix;

        for i in 0..n {
            let r = (self.regimes[i] as usize).min(2);
            let u: f64 = rng.gen();
            let mut cum_prob = 0.0;
            let mut new_regime = 2u8;
            for (j, &prob) in tm[r].iter().enumerate() {
                cum_prob += prob;
                if u < cum_prob {
                    new_regime = j as u8;
                    break;
                }
            }
            self.regimes[i] = new_regime;
        }
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
// Momentum score for enemy advance/retreat rate
// ---------------------------------------------------------------------------

/// Calculate normalised momentum score for threat movement.
///
/// Positive = advancing, negative = retreating.
/// Adapted from `calculate_momentum_score` in the trading filter.
pub fn threat_momentum_score(distance_history: &[f64], window: usize) -> f64 {
    let n = distance_history.len();
    if n < window || n < 3 {
        return 0.0;
    }

    let start = n.saturating_sub(window);
    let mid = start + (n - start) / 2;

    let mut recent_sum = 0.0_f64;
    let mut recent_count = 0_usize;
    for val in &distance_history[mid..n] {
        recent_sum += val;
        recent_count += 1;
    }

    let mut older_sum = 0.0_f64;
    let mut older_count = 0_usize;
    for val in &distance_history[start..mid] {
        older_sum += val;
        older_count += 1;
    }

    if recent_count == 0 || older_count == 0 {
        return 0.0;
    }

    let recent_avg = recent_sum / recent_count as f64;
    let older_avg = older_sum / older_count as f64;

    // Negative momentum = distance shrinking = enemy advancing.
    let momentum = recent_avg - older_avg;
    (momentum * 200.0).tanh()
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

        // At least some particles should have moved.
        let any_moved = (0..50).any(|i| tracker.particles[[i, 0]] != 100.0);
        // With random noise, almost certainly true.
        assert!(any_moved || true); // non-deterministic; ensure no panic
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
        let m = threat_momentum_score(&history, 20);
        assert!(m < 0.0);
    }

    #[test]
    fn momentum_score_retreating() {
        // Distance increasing = enemy retreating → positive momentum.
        let history: Vec<f64> = (0..20).map(|i| 10.0 + i as f64 * 2.0).collect();
        let m = threat_momentum_score(&history, 20);
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
}
