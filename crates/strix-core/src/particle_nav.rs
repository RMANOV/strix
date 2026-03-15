//! GPS-denied navigation via 6D particle filter.
//!
//! Extends the original 2D `[log_price, velocity]` particle filter to
//! `[x, y, z, vx, vy, vz]` with regime-specific dynamics and multi-sensor
//! fusion.
//!
//! Regimes:
//! - **PATROL**: mean-reverting velocity (hold position / pattern), low noise
//! - **ENGAGE**: velocity tracks `threat_bearing` with beta=0.3, medium noise
//! - **EVADE**: high-noise random walk, rapid direction changes

use nalgebra::Vector3;
use ndarray::Array2;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::state::{DroneState, Observation, Regime, SensorConfig};

// ---------------------------------------------------------------------------
// Process noise profiles per regime (position, velocity std-devs in m, m/s)
// ---------------------------------------------------------------------------

/// Regime-dependent noise parameters: `(position_noise, velocity_noise)`.
#[derive(Debug, Clone)]
pub struct RegimeNoise {
    /// Position noise std-dev per axis (meters).
    pub pos_noise: [f64; 3],
    /// Velocity noise std-dev per axis (m/s).
    pub vel_noise: [f64; 3],
}

impl Default for RegimeNoise {
    fn default() -> Self {
        Self {
            pos_noise: [0.1, 0.1, 0.05],
            vel_noise: [0.2, 0.2, 0.1],
        }
    }
}

/// Full noise configuration for all three regimes.
#[derive(Debug, Clone)]
pub struct ProcessNoiseConfig {
    /// PATROL noise — low.
    pub patrol: RegimeNoise,
    /// ENGAGE noise — medium.
    pub engage: RegimeNoise,
    /// EVADE noise — high.
    pub evade: RegimeNoise,
}

impl Default for ProcessNoiseConfig {
    fn default() -> Self {
        Self {
            patrol: RegimeNoise {
                pos_noise: [0.05, 0.05, 0.02],
                vel_noise: [0.1, 0.1, 0.05],
            },
            engage: RegimeNoise {
                pos_noise: [0.15, 0.15, 0.08],
                vel_noise: [0.3, 0.3, 0.15],
            },
            evade: RegimeNoise {
                pos_noise: [0.4, 0.4, 0.2],
                vel_noise: [0.8, 0.8, 0.4],
            },
        }
    }
}

// ---------------------------------------------------------------------------
// 6D Particle Prediction
// ---------------------------------------------------------------------------

/// Propagate N particles through 6D dynamics with regime-specific noise.
///
/// This is the direct analogue of `predict_particles` from the original
/// 2D trading filter, extended to six dimensions.
///
/// # Arguments
/// * `particles` — Nx6 `ndarray` of `[x, y, z, vx, vy, vz]`.
/// * `regimes` — length-N slice of regime indices.
/// * `threat_bearing` — 3D unit-vector pointing toward the threat
///   (replaces the scalar `imbalance` from the trading filter).
/// * `vel_gain` — how strongly ENGAGE particles track threat bearing.
/// * `dt` — time step in seconds.
/// * `noise_cfg` — process noise per regime.
///
/// The function mutates `particles` in-place and uses `rayon` for
/// parallel iteration.
pub fn predict_particles_6d(
    particles: &mut Array2<f64>,
    regimes: &[u8],
    threat_bearing: &Vector3<f64>,
    vel_gain: f64,
    dt: f64,
    noise_cfg: &ProcessNoiseConfig,
) {
    let n = particles.nrows();
    assert_eq!(n, regimes.len(), "particle / regime length mismatch");
    assert_eq!(particles.ncols(), 6, "particles must have 6 columns");

    let dt_sqrt = dt.max(1e-8).sqrt();

    // Collect rows into owned buffer for safe parallel mutation.
    let mut buf: Vec<[f64; 6]> = (0..n)
        .map(|i| {
            [
                particles[[i, 0]],
                particles[[i, 1]],
                particles[[i, 2]],
                particles[[i, 3]],
                particles[[i, 4]],
                particles[[i, 5]],
            ]
        })
        .collect();

    let tb = *threat_bearing;

    buf.par_iter_mut().enumerate().for_each(|(i, p)| {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let regime = regimes[i];
        let noise = match regime {
            0 => &noise_cfg.patrol,
            1 => &noise_cfg.engage,
            _ => &noise_cfg.evade,
        };

        // Random draws — one per axis for pos and vel.
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

        // Current velocity
        let vx = p[3];
        let vy = p[4];
        let vz = p[5];

        let (new_vx, new_vy, new_vz) = match regime {
            0 => {
                // PATROL — mean-reverting velocity (alpha=0.5), hold pattern
                (
                    0.5 * vx + rv[0] * noise.vel_noise[0] * dt_sqrt,
                    0.5 * vy + rv[1] * noise.vel_noise[1] * dt_sqrt,
                    0.5 * vz + rv[2] * noise.vel_noise[2] * dt_sqrt,
                )
            }
            1 => {
                // ENGAGE — velocity tracks threat_bearing, beta=0.3
                let target_vx = vel_gain * tb.x;
                let target_vy = vel_gain * tb.y;
                let target_vz = vel_gain * tb.z;
                (
                    vx + 0.3 * (target_vx - vx) * dt + rv[0] * noise.vel_noise[0] * dt_sqrt,
                    vy + 0.3 * (target_vy - vy) * dt + rv[1] * noise.vel_noise[1] * dt_sqrt,
                    vz + 0.3 * (target_vz - vz) * dt + rv[2] * noise.vel_noise[2] * dt_sqrt,
                )
            }
            _ => {
                // EVADE — high-noise random walk
                (
                    vx + rv[0] * noise.vel_noise[0] * dt_sqrt,
                    vy + rv[1] * noise.vel_noise[1] * dt_sqrt,
                    vz + rv[2] * noise.vel_noise[2] * dt_sqrt,
                )
            }
        };

        // Position update: x += v * dt + noise
        p[0] += new_vx * dt + rp[0] * noise.pos_noise[0] * dt_sqrt;
        p[1] += new_vy * dt + rp[1] * noise.pos_noise[1] * dt_sqrt;
        p[2] += new_vz * dt + rp[2] * noise.pos_noise[2] * dt_sqrt;
        p[3] = new_vx;
        p[4] = new_vy;
        p[5] = new_vz;
    });

    // Write back.
    for i in 0..n {
        for j in 0..6 {
            particles[[i, j]] = buf[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// 6D Measurement Update
// ---------------------------------------------------------------------------

/// Update particle weights from a batch of sensor observations.
///
/// For each observation the function computes a Gaussian likelihood and
/// multiplies the corresponding particle weight. Weights are normalised
/// with underflow protection (`+1e-300`) at the end — the same pattern
/// used in the original trading filter.
pub fn update_weights_6d(
    particles: &Array2<f64>,
    weights: &mut [f64],
    observations: &[Observation],
    sensor_cfg: &SensorConfig,
) {
    let n = particles.nrows();
    assert_eq!(n, weights.len());

    for obs in observations {
        match obs {
            Observation::Imu {
                acceleration,
                timestamp: _,
                gyro: _,
            } => {
                // IMU acceleration → velocity likelihood.
                // Particle velocity change should be consistent with
                // observed acceleration.
                let sigma2 = sensor_cfg.imu_accel_noise * sensor_cfg.imu_accel_noise + 1e-12;
                for i in 0..n {
                    let vx = particles[[i, 3]];
                    let vy = particles[[i, 4]];
                    let vz = particles[[i, 5]];
                    let diff_sq = (vx - acceleration.x).powi(2)
                        + (vy - acceleration.y).powi(2)
                        + (vz - acceleration.z).powi(2);
                    let like = (-0.5 * diff_sq / sigma2).exp();
                    weights[i] *= like;
                }
            }
            Observation::Barometer {
                altitude,
                timestamp: _,
            } => {
                let sigma2 = sensor_cfg.baro_noise * sensor_cfg.baro_noise + 1e-12;
                for i in 0..n {
                    let diff = particles[[i, 2]] - altitude;
                    let like = (-0.5 * diff * diff / sigma2).exp();
                    weights[i] *= like;
                }
            }
            Observation::Magnetometer {
                heading,
                timestamp: _,
            } => {
                let sigma2 = sensor_cfg.mag_noise * sensor_cfg.mag_noise + 1e-12;
                for i in 0..n {
                    // Heading likelihood: compare velocity direction vs
                    // magnetometer heading vector.
                    let vel = Vector3::new(particles[[i, 3]], particles[[i, 4]], particles[[i, 5]]);
                    let speed = vel.norm();
                    if speed > 1e-6 {
                        let unit_vel = vel / speed;
                        let diff_sq = (unit_vel - heading).norm_squared();
                        let like = (-0.5 * diff_sq / sigma2).exp();
                        weights[i] *= like;
                    }
                    // If nearly stationary, heading is uninformative — skip.
                }
            }
            Observation::Rangefinder {
                distance,
                direction: _,
                timestamp: _,
            } => {
                // Altitude cross-check — rangefinder measures AGL.
                let sigma2 = sensor_cfg.rangefinder_noise * sensor_cfg.rangefinder_noise + 1e-12;
                for i in 0..n {
                    // In NED, z is down, so altitude above ground ≈ -z.
                    let alt = -particles[[i, 2]];
                    let diff = alt - distance;
                    let like = (-0.5 * diff * diff / sigma2).exp();
                    weights[i] *= like;
                }
            }
            Observation::VisualOdometry {
                delta_position,
                confidence,
                timestamp: _,
            } => {
                // Position delta likelihood — scale noise by inverse
                // confidence so low-confidence VO is down-weighted.
                let base_sigma2 = sensor_cfg.vo_noise * sensor_cfg.vo_noise + 1e-12;
                let sigma2 = base_sigma2 / (confidence.max(0.01));
                for i in 0..n {
                    let diff_sq = (particles[[i, 0]] - delta_position.x).powi(2)
                        + (particles[[i, 1]] - delta_position.y).powi(2)
                        + (particles[[i, 2]] - delta_position.z).powi(2);
                    let like = (-0.5 * diff_sq / sigma2).exp();
                    weights[i] *= like;
                }
            }
            Observation::RadioBearing {
                bearing,
                signal_strength: _,
                emitter_id: _,
                timestamp: _,
            } => {
                let sigma2 =
                    sensor_cfg.radio_bearing_noise * sensor_cfg.radio_bearing_noise + 1e-12;
                for i in 0..n {
                    let pos = Vector3::new(particles[[i, 0]], particles[[i, 1]], particles[[i, 2]]);
                    let norm = pos.norm();
                    if norm > 1e-6 {
                        let unit_pos = pos / norm;
                        let diff_sq = (unit_pos - bearing).norm_squared();
                        let like = (-0.5 * diff_sq / sigma2).exp();
                        weights[i] *= like;
                    }
                }
            }
        }
    }

    // Normalise with underflow protection (same as original).
    for w in weights.iter_mut() {
        *w += 1e-300;
    }
    let total: f64 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= total;
    }
}

// ---------------------------------------------------------------------------
// Systematic Resampling (6D)
// ---------------------------------------------------------------------------

/// O(N) systematic resampling for 6D particles.
///
/// Returns `(new_particles, new_regimes)` with uniform weights `1/N`.
/// Same two-pointer algorithm as the original `systematic_resample`.
pub fn systematic_resample_6d(
    particles: &Array2<f64>,
    regimes: &[u8],
    weights: &[f64],
) -> (Array2<f64>, Vec<u8>, Vec<f64>) {
    let n = weights.len();
    assert!(n > 0, "cannot resample zero particles");
    assert_eq!(particles.nrows(), n);
    assert_eq!(regimes.len(), n);

    // Cumulative sum of weights.
    let mut cumsum = vec![0.0_f64; n];
    cumsum[0] = weights[0];
    for i in 1..n {
        cumsum[i] = cumsum[i - 1] + weights[i];
    }
    cumsum[n - 1] = 1.0; // pin last element

    let step = 1.0 / n as f64;
    let start_offset: f64 = rand::thread_rng().gen_range(0.0..step);

    let mut new_particles = Array2::<f64>::zeros((n, 6));
    let mut new_regimes = vec![0u8; n];
    let uniform_weight = 1.0 / n as f64;

    let mut j = 0_usize;
    for i in 0..n {
        let pos = start_offset + step * i as f64;
        while j < n - 1 && cumsum[j] < pos {
            j += 1;
        }
        for k in 0..6 {
            new_particles[[i, k]] = particles[[j, k]];
        }
        new_regimes[i] = regimes[j];
    }

    let uniform_weights = vec![uniform_weight; n];
    (new_particles, new_regimes, uniform_weights)
}

// ---------------------------------------------------------------------------
// Estimation
// ---------------------------------------------------------------------------

/// Weighted mean position/velocity + regime probabilities.
///
/// Returns `(mean_position, mean_velocity, regime_probs)` where
/// `regime_probs` has length `Regime::COUNT`.
pub fn estimate_6d(
    particles: &Array2<f64>,
    weights: &[f64],
    regimes: &[u8],
) -> (Vector3<f64>, Vector3<f64>, [f64; 3]) {
    let n = particles.nrows();
    assert_eq!(n, weights.len());
    assert_eq!(n, regimes.len());

    let mut mean_pos = Vector3::zeros();
    let mut mean_vel = Vector3::zeros();
    let mut regime_probs = [0.0_f64; 3];

    for i in 0..n {
        let w = weights[i];
        mean_pos.x += particles[[i, 0]] * w;
        mean_pos.y += particles[[i, 1]] * w;
        mean_pos.z += particles[[i, 2]] * w;
        mean_vel.x += particles[[i, 3]] * w;
        mean_vel.y += particles[[i, 4]] * w;
        mean_vel.z += particles[[i, 5]] * w;

        let r = (regimes[i] as usize).min(2);
        regime_probs[r] += w;
    }

    (mean_pos, mean_vel, regime_probs)
}

/// Effective Sample Size: `ESS = 1 / sum(w^2)`.
pub fn effective_sample_size(weights: &[f64]) -> f64 {
    let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
    1.0 / (sum_sq + 1e-12)
}

/// Compute weighted position variance (scalar) across all 3 axes.
pub fn position_variance(particles: &Array2<f64>, weights: &[f64], mean_pos: &Vector3<f64>) -> f64 {
    let n = particles.nrows();
    let mut var = 0.0_f64;
    for i in 0..n {
        let dx = particles[[i, 0]] - mean_pos.x;
        let dy = particles[[i, 1]] - mean_pos.y;
        let dz = particles[[i, 2]] - mean_pos.z;
        var += weights[i] * (dx * dx + dy * dy + dz * dz);
    }
    var
}

// ---------------------------------------------------------------------------
// High-level navigation filter
// ---------------------------------------------------------------------------

/// A complete 6D particle navigation filter wrapping all the primitives above.
#[derive(Debug, Clone)]
pub struct ParticleNavFilter {
    /// Nx6 particle array.
    pub particles: Array2<f64>,
    /// Regime per particle.
    pub regimes: Vec<u8>,
    /// Importance weights.
    pub weights: Vec<f64>,
    /// Process noise configuration.
    pub noise_cfg: ProcessNoiseConfig,
    /// Sensor noise configuration.
    pub sensor_cfg: SensorConfig,
    /// ESS threshold below which resampling is triggered (fraction of N).
    pub resample_threshold: f64,
}

impl ParticleNavFilter {
    /// Create a new filter centred on `initial_pos` with `n` particles.
    pub fn new(n: usize, initial_pos: Vector3<f64>) -> Self {
        let mut particles = Array2::<f64>::zeros((n, 6));
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in 0..n {
            particles[[i, 0]] = initial_pos.x + normal.sample(&mut rng) * 0.5;
            particles[[i, 1]] = initial_pos.y + normal.sample(&mut rng) * 0.5;
            particles[[i, 2]] = initial_pos.z + normal.sample(&mut rng) * 0.2;
        }

        Self {
            particles,
            regimes: vec![Regime::Patrol as u8; n],
            weights: vec![1.0 / n as f64; n],
            noise_cfg: ProcessNoiseConfig::default(),
            sensor_cfg: SensorConfig::default(),
            resample_threshold: 0.5,
        }
    }

    /// Run one predict–update–resample cycle.
    pub fn step(
        &mut self,
        observations: &[Observation],
        threat_bearing: &Vector3<f64>,
        vel_gain: f64,
        dt: f64,
    ) -> (Vector3<f64>, Vector3<f64>, [f64; 3]) {
        // Predict
        predict_particles_6d(
            &mut self.particles,
            &self.regimes,
            threat_bearing,
            vel_gain,
            dt,
            &self.noise_cfg,
        );

        // Update
        update_weights_6d(
            &self.particles,
            &mut self.weights,
            observations,
            &self.sensor_cfg,
        );

        // Estimate
        let (pos, vel, probs) = estimate_6d(&self.particles, &self.weights, &self.regimes);

        // Resample if ESS too low
        let n = self.weights.len() as f64;
        let ess = effective_sample_size(&self.weights);
        if ess < self.resample_threshold * n {
            let (new_p, new_r, new_w) =
                systematic_resample_6d(&self.particles, &self.regimes, &self.weights);
            self.particles = new_p;
            self.regimes = new_r;
            self.weights = new_w;
        }

        (pos, vel, probs)
    }

    /// Current best-estimate state as a [`DroneState`].
    pub fn to_drone_state(&self, drone_id: u32) -> DroneState {
        let (pos, vel, probs) = estimate_6d(&self.particles, &self.weights, &self.regimes);
        let regime_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let regime = Regime::from_index(regime_idx as u8).unwrap_or(Regime::Patrol);

        DroneState {
            position: pos,
            velocity: vel,
            regime,
            weight: 1.0,
            drone_id,
            capabilities: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_preserves_shape() {
        let n = 50;
        let mut particles = Array2::<f64>::zeros((n, 6));
        let regimes = vec![0u8; n];
        let tb = Vector3::new(1.0, 0.0, 0.0);
        let noise = ProcessNoiseConfig::default();

        predict_particles_6d(&mut particles, &regimes, &tb, 1.0, 0.1, &noise);

        assert_eq!(particles.nrows(), n);
        assert_eq!(particles.ncols(), 6);
        // Particles should have moved from zero
        let any_moved = (0..n).any(|i| particles[[i, 0]].abs() > 1e-15);
        assert!(any_moved);
    }

    #[test]
    fn update_normalises_weights() {
        let n = 20;
        let particles = Array2::<f64>::zeros((n, 6));
        let mut weights = vec![1.0 / n as f64; n];
        let obs = vec![Observation::Barometer {
            altitude: 0.0,
            timestamp: 0.0,
        }];
        let cfg = SensorConfig::default();

        update_weights_6d(&particles, &mut weights, &obs, &cfg);

        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn systematic_resample_uniform() {
        let n = 100;
        let particles = Array2::<f64>::zeros((n, 6));
        let regimes = vec![0u8; n];
        let weights = vec![1.0 / n as f64; n];

        let (new_p, new_r, new_w) = systematic_resample_6d(&particles, &regimes, &weights);

        assert_eq!(new_p.nrows(), n);
        assert_eq!(new_r.len(), n);
        let sum: f64 = new_w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn estimate_weighted_mean() {
        let n = 2;
        let mut particles = Array2::<f64>::zeros((n, 6));
        particles[[0, 0]] = 0.0;
        particles[[1, 0]] = 10.0;
        let weights = vec![0.5, 0.5];
        let regimes = vec![0u8; n];

        let (pos, _vel, _probs) = estimate_6d(&particles, &weights, &regimes);
        assert!((pos.x - 5.0).abs() < 1e-12);
    }

    #[test]
    fn ess_uniform() {
        let n = 100;
        let weights = vec![1.0 / n as f64; n];
        let ess = effective_sample_size(&weights);
        assert!((ess - n as f64).abs() < 1.0);
    }

    #[test]
    fn nav_filter_step() {
        let mut filter = ParticleNavFilter::new(100, Vector3::new(0.0, 0.0, -50.0));
        let obs = vec![Observation::Barometer {
            altitude: 50.0,
            timestamp: 0.0,
        }];
        let tb = Vector3::zeros();
        let (pos, _vel, probs) = filter.step(&obs, &tb, 1.0, 0.1);
        // Position should be near initial
        assert!(pos.norm() < 100.0);
        // Regime probs should sum to ~1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8);
    }
}
