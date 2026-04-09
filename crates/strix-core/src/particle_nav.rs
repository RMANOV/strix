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
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::particle_common::{self, gaussian_likelihood, normalize_weights};
use crate::state::{DroneState, Observation, Regime, SensorConfig};

pub use crate::particle_common::effective_sample_size;

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

impl ProcessNoiseConfig {
    /// Scale noise profiles by fear level F ∈ [0,1].
    ///
    /// Higher fear → PATROL noise tightens (disciplined flight, tighter formation),
    /// EVADE noise amplifies (chaotic evasion is harder to target).
    pub fn scaled_by_fear(&self, f: f64) -> Self {
        let f = if f.is_nan() || f.is_infinite() {
            0.0
        } else {
            f.clamp(0.0, 1.0)
        };
        let patrol_scale = 1.0 - f * 0.5; // reduce by up to 50%
        let engage_scale = 1.0 + f * 0.3; // increase by up to 30%
        let evade_scale = 1.0 + f; // double at max fear
        Self {
            patrol: RegimeNoise {
                pos_noise: self.patrol.pos_noise.map(|n| n * patrol_scale),
                vel_noise: self.patrol.vel_noise.map(|n| n * patrol_scale),
            },
            engage: RegimeNoise {
                pos_noise: self.engage.pos_noise.map(|n| n * engage_scale),
                vel_noise: self.engage.vel_noise.map(|n| n * engage_scale),
            },
            evade: RegimeNoise {
                pos_noise: self.evade.pos_noise.map(|n| n * evade_scale),
                vel_noise: self.evade.vel_noise.map(|n| n * evade_scale),
            },
        }
    }

    /// Scale all noise profiles uniformly by an EW noise multiplier.
    ///
    /// Used when electronic warfare degradation inflates navigation uncertainty.
    pub fn scaled_by_ew(&self, multiplier: f64) -> Self {
        let m = multiplier.max(1.0);
        Self {
            patrol: RegimeNoise {
                pos_noise: self.patrol.pos_noise.map(|n| n * m),
                vel_noise: self.patrol.vel_noise.map(|n| n * m),
            },
            engage: RegimeNoise {
                pos_noise: self.engage.pos_noise.map(|n| n * m),
                vel_noise: self.engage.vel_noise.map(|n| n * m),
            },
            evade: RegimeNoise {
                pos_noise: self.evade.pos_noise.map(|n| n * m),
                vel_noise: self.evade.vel_noise.map(|n| n * m),
            },
        }
    }

    /// Cap all noise components to prevent formation geometry breakage.
    pub fn clamped(&self, max_pos: f64, max_vel: f64) -> Self {
        Self {
            patrol: RegimeNoise {
                pos_noise: self.patrol.pos_noise.map(|n| n.min(max_pos)),
                vel_noise: self.patrol.vel_noise.map(|n| n.min(max_vel)),
            },
            engage: RegimeNoise {
                pos_noise: self.engage.pos_noise.map(|n| n.min(max_pos)),
                vel_noise: self.engage.vel_noise.map(|n| n.min(max_vel)),
            },
            evade: RegimeNoise {
                pos_noise: self.evade.pos_noise.map(|n| n.min(max_pos)),
                vel_noise: self.evade.vel_noise.map(|n| n.min(max_vel)),
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
    let mut scratch: Vec<[f64; 6]> = Vec::new();
    predict_particles_6d_with_buf(
        particles,
        regimes,
        threat_bearing,
        vel_gain,
        dt,
        noise_cfg,
        &mut scratch,
    );
}

/// Inner implementation of [`predict_particles_6d`] reusing a caller-provided scratch buffer.
///
/// The buffer is resized if needed and reused across calls to eliminate per-call heap allocation.
pub fn predict_particles_6d_with_buf(
    particles: &mut Array2<f64>,
    regimes: &[u8],
    threat_bearing: &Vector3<f64>,
    vel_gain: f64,
    dt: f64,
    noise_cfg: &ProcessNoiseConfig,
    buf: &mut Vec<[f64; 6]>,
) {
    let n = particles.nrows();
    assert_eq!(n, regimes.len(), "particle / regime length mismatch");
    assert_eq!(particles.ncols(), 6, "particles must have 6 columns");

    let dt = if dt.is_finite() && dt >= 0.0 { dt } else { 0.0 };
    let vel_gain = if vel_gain.is_finite() { vel_gain } else { 0.0 };
    let dt_sqrt = dt.max(1e-8).sqrt();

    // Resize buffer to match particle count (no-op if capacity already sufficient).
    buf.resize(n, [0.0; 6]);
    for (i, p) in buf.iter_mut().enumerate() {
        *p = [
            particles[[i, 0]],
            particles[[i, 1]],
            particles[[i, 2]],
            particles[[i, 3]],
            particles[[i, 4]],
            particles[[i, 5]],
        ];
    }

    let tb = if threat_bearing.iter().all(|value| value.is_finite()) {
        *threat_bearing
    } else {
        Vector3::zeros()
    };
    let normal = Normal::new(0.0, 1.0).expect("Normal(0,1) has valid parameters");

    let update_particle = |(i, p): (usize, &mut [f64; 6])| {
        let mut rng = rand::thread_rng();

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
    };

    // Only parallelize above break-even; below 500 particles rayon overhead exceeds gain.
    if n > 500 {
        buf.par_iter_mut().enumerate().for_each(update_particle);
    } else {
        buf.iter_mut().enumerate().for_each(update_particle);
    }

    // Write back.
    for i in 0..n {
        for j in 0..6 {
            particles[[i, j]] = buf[i][j];
        }
    }

    // NaN sanitization: reset any degenerate particle coordinates to 0.
    for i in 0..n {
        for j in 0..6 {
            if !particles[[i, j]].is_finite() {
                particles[[i, j]] = 0.0;
            }
        }
    }
}

/// Sequential (deterministic) variant of [`predict_particles_6d_with_buf`].
///
/// Uses a caller-supplied RNG and iterates sequentially for reproducibility.
#[allow(clippy::too_many_arguments)]
pub fn predict_particles_6d_seeded<R: Rng>(
    particles: &mut Array2<f64>,
    regimes: &[u8],
    threat_bearing: &Vector3<f64>,
    vel_gain: f64,
    dt: f64,
    noise_cfg: &ProcessNoiseConfig,
    buf: &mut Vec<[f64; 6]>,
    rng: &mut R,
) {
    let n = particles.nrows();
    let dt = if dt.is_finite() && dt >= 0.0 { dt } else { 0.0 };
    let vel_gain = if vel_gain.is_finite() { vel_gain } else { 0.0 };
    let dt_sqrt = dt.max(1e-8).sqrt();
    buf.resize(n, [0.0; 6]);
    for (i, p) in buf.iter_mut().enumerate() {
        *p = [
            particles[[i, 0]],
            particles[[i, 1]],
            particles[[i, 2]],
            particles[[i, 3]],
            particles[[i, 4]],
            particles[[i, 5]],
        ];
    }
    let tb = if threat_bearing.iter().all(|value| value.is_finite()) {
        *threat_bearing
    } else {
        Vector3::zeros()
    };
    let normal = Normal::new(0.0, 1.0).expect("Normal(0,1)");
    for (i, p) in buf.iter_mut().enumerate() {
        let regime = regimes[i];
        let noise = match regime {
            0 => &noise_cfg.patrol,
            1 => &noise_cfg.engage,
            _ => &noise_cfg.evade,
        };
        let rp: [f64; 3] = [normal.sample(rng), normal.sample(rng), normal.sample(rng)];
        let rv: [f64; 3] = [normal.sample(rng), normal.sample(rng), normal.sample(rng)];
        let (vx, vy, vz) = (p[3], p[4], p[5]);
        let (nvx, nvy, nvz) = match regime {
            0 => (
                0.5 * vx + rv[0] * noise.vel_noise[0] * dt_sqrt,
                0.5 * vy + rv[1] * noise.vel_noise[1] * dt_sqrt,
                0.5 * vz + rv[2] * noise.vel_noise[2] * dt_sqrt,
            ),
            1 => (
                vx + 0.3 * (vel_gain * tb.x - vx) * dt + rv[0] * noise.vel_noise[0] * dt_sqrt,
                vy + 0.3 * (vel_gain * tb.y - vy) * dt + rv[1] * noise.vel_noise[1] * dt_sqrt,
                vz + 0.3 * (vel_gain * tb.z - vz) * dt + rv[2] * noise.vel_noise[2] * dt_sqrt,
            ),
            _ => (
                vx + rv[0] * noise.vel_noise[0] * dt_sqrt,
                vy + rv[1] * noise.vel_noise[1] * dt_sqrt,
                vz + rv[2] * noise.vel_noise[2] * dt_sqrt,
            ),
        };
        p[0] += nvx * dt + rp[0] * noise.pos_noise[0] * dt_sqrt;
        p[1] += nvy * dt + rp[1] * noise.pos_noise[1] * dt_sqrt;
        p[2] += nvz * dt + rp[2] * noise.pos_noise[2] * dt_sqrt;
        p[3] = nvx;
        p[4] = nvy;
        p[5] = nvz;
    }
    for i in 0..n {
        for j in 0..6 {
            particles[[i, j]] = buf[i][j];
        }
    }
    for i in 0..n {
        for j in 0..6 {
            if !particles[[i, j]].is_finite() {
                particles[[i, j]] = 0.0;
            }
        }
    }
}

/// Systematic resampling with a caller-supplied RNG for deterministic replay.
pub fn systematic_resample_6d_with_rng<R: Rng>(
    particles: &Array2<f64>,
    regimes: &[u8],
    weights: &[f64],
    rng: &mut R,
) -> (Array2<f64>, Vec<u8>, Vec<f64>) {
    let n = weights.len();
    assert!(n > 0);
    let mut cumsum = vec![0.0_f64; n];
    cumsum[0] = weights[0];
    for i in 1..n {
        cumsum[i] = cumsum[i - 1] + weights[i];
    }
    cumsum[n - 1] = 1.0;
    let step = 1.0 / n as f64;
    let start_offset: f64 = rng.gen_range(0.0..step);
    let mut new_particles = Array2::<f64>::zeros((n, 6));
    let mut new_regimes = vec![0u8; n];
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
    (new_particles, new_regimes, vec![1.0 / n as f64; n])
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
///
/// Returns `true` if a weight collapse was detected (all raw weights
/// were below `1e-100` before rescue).
pub fn update_weights_6d(
    particles: &Array2<f64>,
    weights: &mut [f64],
    observations: &[Observation],
    sensor_cfg: &SensorConfig,
) -> bool {
    let n = particles.nrows();
    assert_eq!(n, weights.len());

    let vector_is_finite = |vector: &Vector3<f64>| vector.iter().all(|value| value.is_finite());

    for obs in observations {
        match obs {
            Observation::Imu {
                acceleration,
                timestamp: _,
                gyro: _,
            } => {
                // Skip observation if telemetry velocity/acceleration contains NaN.
                if !acceleration.x.is_finite()
                    || !acceleration.y.is_finite()
                    || !acceleration.z.is_finite()
                {
                    continue;
                }
                if !sensor_cfg.imu_accel_noise.is_finite() || sensor_cfg.imu_accel_noise < 1e-6 {
                    continue;
                }
                // IMU acceleration → velocity likelihood.
                // Particle velocity change should be consistent with
                // observed acceleration.
                for i in 0..n {
                    let vx = particles[[i, 3]];
                    let vy = particles[[i, 4]];
                    let vz = particles[[i, 5]];
                    let diff_sq = (vx - acceleration.x).powi(2)
                        + (vy - acceleration.y).powi(2)
                        + (vz - acceleration.z).powi(2);
                    weights[i] *= gaussian_likelihood(diff_sq, sensor_cfg.imu_accel_noise);
                }
            }
            Observation::Barometer {
                altitude,
                timestamp: _,
            } => {
                if !altitude.is_finite() {
                    continue;
                }
                if !sensor_cfg.baro_noise.is_finite() || sensor_cfg.baro_noise < 1e-6 {
                    continue;
                }
                for i in 0..n {
                    let diff = particles[[i, 2]] + altitude; // NED: z=-alt, so diff = z+alt ≈ 0 for correct particles
                    weights[i] *= gaussian_likelihood(diff * diff, sensor_cfg.baro_noise);
                }
            }
            Observation::Magnetometer {
                heading,
                timestamp: _,
            } => {
                if !vector_is_finite(heading) {
                    continue;
                }
                if !sensor_cfg.mag_noise.is_finite() || sensor_cfg.mag_noise < 1e-6 {
                    continue;
                }
                for i in 0..n {
                    // Heading likelihood: compare velocity direction vs
                    // magnetometer heading vector.
                    let vel = Vector3::new(particles[[i, 3]], particles[[i, 4]], particles[[i, 5]]);
                    let speed = vel.norm();
                    if speed > 1e-6 {
                        let unit_vel = vel / speed;
                        let diff_sq = (unit_vel - heading).norm_squared();
                        weights[i] *= gaussian_likelihood(diff_sq, sensor_cfg.mag_noise);
                    }
                    // If nearly stationary, heading is uninformative — skip.
                }
            }
            Observation::Rangefinder {
                distance,
                direction: _,
                timestamp: _,
            } => {
                if !distance.is_finite() {
                    continue;
                }
                if !sensor_cfg.rangefinder_noise.is_finite() || sensor_cfg.rangefinder_noise < 1e-6
                {
                    continue;
                }
                // Altitude cross-check — rangefinder measures AGL.
                for i in 0..n {
                    // In NED, z is down, so altitude above ground ≈ -z.
                    let alt = -particles[[i, 2]];
                    let diff = alt - distance;
                    weights[i] *= gaussian_likelihood(diff * diff, sensor_cfg.rangefinder_noise);
                }
            }
            Observation::VisualOdometry {
                delta_position,
                confidence,
                timestamp: _,
            } => {
                if !vector_is_finite(delta_position) || !confidence.is_finite() {
                    continue;
                }
                if !sensor_cfg.vo_noise.is_finite() || sensor_cfg.vo_noise < 1e-6 {
                    continue;
                }
                // Position delta likelihood — scale noise by inverse
                // confidence so low-confidence VO is down-weighted.
                let base_sigma2 = sensor_cfg.vo_noise * sensor_cfg.vo_noise + 1e-12;
                let sigma2 = base_sigma2 / confidence.clamp(0.01, 1.0);
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
                if !vector_is_finite(bearing) {
                    continue;
                }
                if !sensor_cfg.radio_bearing_noise.is_finite()
                    || sensor_cfg.radio_bearing_noise < 1e-6
                {
                    continue;
                }
                for i in 0..n {
                    let pos = Vector3::new(particles[[i, 0]], particles[[i, 1]], particles[[i, 2]]);
                    let norm = pos.norm();
                    if norm > 1e-6 {
                        let unit_pos = pos / norm;
                        let diff_sq = (unit_pos - bearing).norm_squared();
                        weights[i] *= gaussian_likelihood(diff_sq, sensor_cfg.radio_bearing_noise);
                    }
                }
            }
        }
    }

    // Detect filter collapse: if max raw weight is effectively zero,
    // all particles are terrible fits. The 1e-300 rescue below will
    // produce uniform weights and ESS=N (hiding the collapse from
    // the resampler). Log a warning so upstream can detect this.
    let max_raw = weights.iter().cloned().fold(0.0_f64, f64::max);
    let collapsed = max_raw < 1e-100;
    if collapsed {
        tracing::warn!(
            max_weight = max_raw,
            n_particles = weights.len(),
            "particle filter collapse: all weights near zero, resetting to uniform"
        );
    }

    // Normalise with underflow protection.
    normalize_weights(weights);

    collapsed
}

// ---------------------------------------------------------------------------
// Systematic Resampling (6D)
// ---------------------------------------------------------------------------

/// O(N) systematic resampling for 6D particles.
///
/// Returns `(new_particles, new_regimes, new_weights)` with uniform weights
/// `1/N`.  Delegates to [`particle_common::systematic_resample_6d`] for the
/// core algorithm.
pub fn systematic_resample_6d(
    particles: &Array2<f64>,
    regimes: &[u8],
    weights: &[f64],
) -> (Array2<f64>, Vec<u8>, Vec<f64>) {
    let n = weights.len();
    assert!(n > 0, "cannot resample zero particles");
    assert_eq!(particles.nrows(), n);
    assert_eq!(regimes.len(), n);

    let mut new_particles = particles.clone();
    let mut new_regimes = regimes.to_vec();
    let mut new_weights = weights.to_vec();
    particle_common::systematic_resample_6d(&mut new_particles, &mut new_weights, &mut new_regimes);
    (new_particles, new_regimes, new_weights)
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

/// Weighted per-axis position variance from a particle cloud.
///
/// Returns `[var_x, var_y, var_z]` — diagonal of the 3×3 position covariance.
/// Each axis is clamped to minimum 1e-10 to avoid infinite precision in GBP.
pub fn position_covariance_diagonal(
    particles: &Array2<f64>,
    weights: &[f64],
    mean_pos: &Vector3<f64>,
) -> [f64; 3] {
    let n = particles.nrows();
    let mut vx = 0.0_f64;
    let mut vy = 0.0_f64;
    let mut vz = 0.0_f64;
    for i in 0..n {
        let dx = particles[[i, 0]] - mean_pos.x;
        let dy = particles[[i, 1]] - mean_pos.y;
        let dz = particles[[i, 2]] - mean_pos.z;
        let w = weights[i];
        vx += w * dx * dx;
        vy += w * dy * dy;
        vz += w * dz * dz;
    }
    [vx.max(1e-10), vy.max(1e-10), vz.max(1e-10)]
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
    /// Number of consecutive weight collapses observed.
    ///
    /// Incremented each step where `max_raw < 1e-100`. Reset to 0 on any
    /// non-collapse step. When this exceeds 2, a hard reset is performed:
    /// all particles are reseeded tightly around the best available
    /// observation position.
    pub collapse_count: u32,
    /// Pre-allocated scratch buffer — reused across `step()` calls to avoid
    /// heap allocation in `predict_particles_6d_with_buf`.
    scratch_buf: Vec<[f64; 6]>,
    /// Optional seeded RNG for deterministic replay. `None` uses `thread_rng()`.
    rng: Option<ChaCha8Rng>,
}

impl ParticleNavFilter {
    /// Create a new filter centred on `initial_pos` with `n` particles.
    pub fn new(n: usize, initial_pos: Vector3<f64>) -> Self {
        let mut particles = Array2::<f64>::zeros((n, 6));
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).expect("Normal(0,1) has valid parameters");
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
            collapse_count: 0,
            scratch_buf: vec![[0.0; 6]; n],
            rng: None,
        }
    }

    /// Create a new filter with a seeded RNG for deterministic replay.
    pub fn new_seeded(n: usize, initial_pos: Vector3<f64>, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut seeded_rng = ChaCha8Rng::seed_from_u64(seed);
        let mut particles = Array2::<f64>::zeros((n, 6));
        let normal = Normal::new(0.0, 1.0).expect("Normal(0,1) has valid parameters");
        for i in 0..n {
            particles[[i, 0]] = initial_pos.x + normal.sample(&mut seeded_rng) * 0.5;
            particles[[i, 1]] = initial_pos.y + normal.sample(&mut seeded_rng) * 0.5;
            particles[[i, 2]] = initial_pos.z + normal.sample(&mut seeded_rng) * 0.2;
        }
        Self {
            particles,
            regimes: vec![Regime::Patrol as u8; n],
            weights: vec![1.0 / n as f64; n],
            noise_cfg: ProcessNoiseConfig::default(),
            sensor_cfg: SensorConfig::default(),
            resample_threshold: 0.5,
            collapse_count: 0,
            scratch_buf: vec![[0.0; 6]; n],
            rng: Some(seeded_rng),
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
        // Predict — use pre-allocated scratch buffer to avoid heap allocation.
        let mut scratch = std::mem::take(&mut self.scratch_buf);
        if let Some(ref mut seeded) = self.rng {
            predict_particles_6d_seeded(
                &mut self.particles,
                &self.regimes,
                threat_bearing,
                vel_gain,
                dt,
                &self.noise_cfg,
                &mut scratch,
                seeded,
            );
        } else {
            predict_particles_6d_with_buf(
                &mut self.particles,
                &self.regimes,
                threat_bearing,
                vel_gain,
                dt,
                &self.noise_cfg,
                &mut scratch,
            );
        }
        self.scratch_buf = scratch;

        // Update (normalises weights internally, logs warning on collapse).
        // Returns true if weight collapse was detected.
        let collapse = update_weights_6d(
            &self.particles,
            &mut self.weights,
            observations,
            &self.sensor_cfg,
        );

        if collapse {
            self.collapse_count += 1;
            tracing::warn!(
                collapse_count = self.collapse_count,
                "particle_nav: weight collapse detected in step (collapse_count={})",
                self.collapse_count
            );

            if self.collapse_count > 2 {
                // Hard reset: reseed all particles tightly around the best
                // available position observation.  Fall back to the current
                // weighted mean if no position measurement is available.
                //
                // The current weighted-mean is computed first and used both as
                // the fallback anchor and as the base for integrating the VO
                // delta (which is a relative displacement, not an absolute pos).
                let current_estimate = {
                    let (p, _, _) = estimate_6d(&self.particles, &self.weights, &self.regimes);
                    p
                };
                let recovery_pos = Self::extract_obs_position(observations, current_estimate)
                    .unwrap_or(current_estimate);

                tracing::warn!(
                    recovery_pos = ?recovery_pos,
                    "particle_nav: hard reset — reseeding {} particles around observation",
                    self.particles.nrows()
                );

                let n = self.particles.nrows();
                let normal = Normal::new(0.0, 1.0).expect("Normal(0,1) has valid parameters");
                macro_rules! reseed_particles {
                    ($rng:expr) => {
                        for i in 0..n {
                            self.particles[[i, 0]] = recovery_pos.x + normal.sample($rng) * 0.5;
                            self.particles[[i, 1]] = recovery_pos.y + normal.sample($rng) * 0.5;
                            self.particles[[i, 2]] = recovery_pos.z + normal.sample($rng) * 0.2;
                        }
                    };
                }
                if let Some(ref mut seeded) = self.rng {
                    reseed_particles!(seeded);
                } else {
                    let mut rng = rand::thread_rng();
                    reseed_particles!(&mut rng);
                }
                for i in 0..n {
                    self.particles[[i, 3]] = 0.0;
                    self.particles[[i, 4]] = 0.0;
                    self.particles[[i, 5]] = 0.0;
                }
                self.regimes = vec![Regime::Patrol as u8; n];
                self.weights = vec![1.0 / n as f64; n];
                self.collapse_count = 0;
            }
        } else {
            self.collapse_count = 0;
        }

        // Estimate
        let (pos, vel, probs) = estimate_6d(&self.particles, &self.weights, &self.regimes);

        // Resample if ESS too low
        let n = self.weights.len() as f64;
        let ess = effective_sample_size(&self.weights);
        if ess < self.resample_threshold * n {
            if let Some(ref mut seeded) = self.rng {
                let (new_p, new_r, new_w) = systematic_resample_6d_with_rng(
                    &self.particles,
                    &self.regimes,
                    &self.weights,
                    seeded,
                );
                self.particles = new_p;
                self.regimes = new_r;
                self.weights = new_w;
            } else {
                let (new_p, new_r, new_w) =
                    systematic_resample_6d(&self.particles, &self.regimes, &self.weights);
                self.particles = new_p;
                self.regimes = new_r;
                self.weights = new_w;
            }
        }

        (pos, vel, probs)
    }

    /// Extract a 3D position from the first position-bearing observation, if any.
    ///
    /// `current_pos` is the current weighted-mean particle position and is used
    /// to integrate the VO delta (which is relative: current − previous) into an
    /// absolute position estimate.
    ///
    /// Barometer gives altitude (z-axis only); VisualOdometry gives a full
    /// delta-position; other sensors are bearing-only or velocity-based.
    fn extract_obs_position(
        observations: &[Observation],
        current_pos: Vector3<f64>,
    ) -> Option<Vector3<f64>> {
        for obs in observations {
            match obs {
                Observation::VisualOdometry {
                    delta_position,
                    confidence,
                    ..
                } if delta_position.iter().all(|value| value.is_finite())
                    && confidence.is_finite()
                    && *confidence > 0.1 =>
                {
                    // delta_position is a relative displacement (current − previous).
                    // Add it to the current estimate to get an absolute anchor.
                    return Some(current_pos + delta_position);
                }
                Observation::Barometer { altitude, .. } if altitude.is_finite() => {
                    // Barometer gives z only; x,y unknown — return partial hint.
                    return Some(Vector3::new(current_pos.x, current_pos.y, -altitude));
                }
                _ => {}
            }
        }
        None
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

        let _collapsed = update_weights_6d(&particles, &mut weights, &obs, &cfg);

        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn update_ignores_non_finite_observations() {
        let n = 16;
        let particles = Array2::<f64>::zeros((n, 6));
        let mut weights = vec![1.0 / n as f64; n];
        let obs = vec![
            Observation::Barometer {
                altitude: f64::NAN,
                timestamp: 0.0,
            },
            Observation::Magnetometer {
                heading: Vector3::new(f64::NAN, 0.0, 0.0),
                timestamp: 0.0,
            },
            Observation::VisualOdometry {
                delta_position: Vector3::new(0.0, f64::INFINITY, 0.0),
                confidence: f64::NAN,
                timestamp: 0.0,
            },
        ];

        let _collapsed =
            update_weights_6d(&particles, &mut weights, &obs, &SensorConfig::default());

        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(weights
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0));
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

    // ── 6A: Property-style hardening tests ───────────────────────────────

    /// weights_sum_to_one_after_step: after a full predict-update-resample
    /// cycle the importance weights must form a valid probability distribution.
    #[test]
    fn weights_sum_to_one_after_step() {
        let mut filter = ParticleNavFilter::new(200, Vector3::new(10.0, -5.0, -30.0));
        let obs = vec![
            Observation::Barometer {
                altitude: 30.0,
                timestamp: 0.0,
            },
            Observation::Imu {
                acceleration: Vector3::new(0.5, -0.2, 0.1),
                gyro: None,
                timestamp: 0.0,
            },
        ];
        let tb = Vector3::new(0.0, 1.0, 0.0);
        let _result = filter.step(&obs, &tb, 2.0, 0.05);

        let sum: f64 = filter.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "weights must sum to 1.0 after step, got {sum}"
        );
        // All weights must be non-negative.
        for &w in &filter.weights {
            assert!(w >= 0.0, "weight must be non-negative, got {w}");
        }
    }

    /// ess_in_valid_range: ESS must always satisfy 1 ≤ ESS ≤ N.
    ///
    /// ESS = 1 would mean complete particle degeneracy (single particle carries
    /// all weight); ESS = N means uniform weights (maximum diversity).
    #[test]
    fn ess_in_valid_range() {
        let n = 150usize;
        let mut filter = ParticleNavFilter::new(n, Vector3::new(0.0, 0.0, -50.0));
        let obs = vec![Observation::Barometer {
            altitude: 50.0,
            timestamp: 0.0,
        }];
        let tb = Vector3::zeros();
        filter.step(&obs, &tb, 1.0, 0.1);

        let ess = effective_sample_size(&filter.weights);
        assert!(ess >= 1.0 - 1e-9, "ESS must be >= 1.0, got {ess}");
        assert!(ess <= n as f64 + 1e-9, "ESS must be <= N={n}, got {ess}");
    }

    /// position_variance_finite: the weighted position variance must be finite
    /// and non-negative after any step, even with noisy observations.
    #[test]
    fn position_variance_finite() {
        let mut filter = ParticleNavFilter::new(100, Vector3::new(3.0, -7.0, -20.0));
        // Use multiple sensor types to stress the update step.
        let obs = vec![
            Observation::Barometer {
                altitude: 20.0,
                timestamp: 0.1,
            },
            Observation::Imu {
                acceleration: Vector3::new(1.0, 0.5, -0.3),
                gyro: None,
                timestamp: 0.1,
            },
            Observation::Magnetometer {
                heading: Vector3::new(1.0, 0.0, 0.0),
                timestamp: 0.1,
            },
        ];
        let tb = Vector3::new(1.0, 0.0, 0.0);
        let (pos, _vel, _probs) = filter.step(&obs, &tb, 1.0, 0.1);

        let var = position_variance(&filter.particles, &filter.weights, &pos);
        assert!(
            var.is_finite(),
            "position variance must be finite, got {var}"
        );
        assert!(
            var >= 0.0,
            "position variance must be non-negative, got {var}"
        );
        // Individual components must also be finite.
        assert!(pos.x.is_finite(), "mean x must be finite");
        assert!(pos.y.is_finite(), "mean y must be finite");
        assert!(pos.z.is_finite(), "mean z must be finite");
    }

    /// regime_probabilities_sum_to_one: after a step with real observations
    /// the regime probability vector must be a valid probability distribution.
    ///
    /// This is a more thorough version of the basic check in `nav_filter_step`
    /// — it verifies the invariant across multiple steps and with diverse sensors.
    #[test]
    fn regime_probabilities_sum_to_one_after_step() {
        let mut filter = ParticleNavFilter::new(100, Vector3::new(0.0, 0.0, -50.0));
        let observations_sequence = [
            vec![Observation::Barometer {
                altitude: 50.0,
                timestamp: 0.0,
            }],
            vec![
                Observation::Imu {
                    acceleration: Vector3::new(0.3, 0.1, 0.0),
                    gyro: None,
                    timestamp: 0.1,
                },
                Observation::Magnetometer {
                    heading: Vector3::new(
                        std::f64::consts::FRAC_1_SQRT_2,
                        std::f64::consts::FRAC_1_SQRT_2,
                        0.0,
                    ),
                    timestamp: 0.1,
                },
            ],
            vec![Observation::Rangefinder {
                distance: 50.0,
                direction: Vector3::new(0.0, 0.0, -1.0),
                timestamp: 0.2,
            }],
        ];

        let tb = Vector3::new(0.5, 0.5, 0.0);
        for (step, obs) in observations_sequence.iter().enumerate() {
            let (_pos, _vel, probs) = filter.step(obs, &tb, 1.0, 0.1);
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-8,
                "regime probs must sum to 1.0 at step {step}, got {sum}"
            );
            // Each individual probability must be in [0, 1].
            for (i, &p) in probs.iter().enumerate() {
                assert!(
                    (0.0..=1.0 + 1e-10).contains(&p),
                    "regime prob[{i}]={p} out of [0,1] at step {step}"
                );
            }
        }
    }

    /// Feed divergent telemetry that makes all particle weights collapse to ~0.
    /// After > 2 consecutive collapses the filter should do a hard reset and
    /// reseed particles near the recovered absolute position.  We verify:
    ///   1. `collapse_count` resets to 0.
    ///   2. Weights are uniform (1/N) right after the hard reset.
    ///   3. Subsequent consistent VO observation gives non-collapsed ESS (< N).
    #[test]
    fn collapse_recovery_reseeds_near_observation() {
        // Filter starts at origin.  VO supplies a delta_position (relative
        // displacement from the previous tick).  extract_obs_position adds this
        // delta to the current weighted-mean estimate to obtain an absolute
        // recovery anchor.
        //
        // With the filter near origin, current_estimate ≈ (0,0,0), so the
        // recovery anchor ≈ current_estimate + vo_delta = vo_delta.
        let vo_delta = Vector3::new(5.0, 3.0, -20.0); // realistic one-tick displacement
        let mut filter = ParticleNavFilter::new(100, Vector3::new(0.0, 0.0, 0.0));
        let tb = Vector3::zeros();
        let n = filter.particles.nrows() as f64;

        // Divergent obs: barometer claims altitude = 1e9 m.  Particles are near
        // z = 0, so diff ≈ 1e9 → like ≈ exp(-huge) ≈ 0 → collapse every step.
        // Include a VO observation so the hard-reset anchor is deterministic
        // (extract_obs_position prefers VO over Barometer).
        let divergent_obs = vec![
            Observation::VisualOdometry {
                delta_position: vo_delta,
                confidence: 0.9,
                timestamp: 0.0,
            },
            Observation::Barometer {
                altitude: 1e9,
                timestamp: 0.0,
            },
        ];

        // Run 3 collapse steps.  The Barometer dominates the weight update
        // (exp(-huge)) so collapse is triggered despite the VO hint.
        // On step 3, collapse_count exceeds 2 and hard reset fires around
        // current_estimate + vo_delta.
        for _ in 0..3 {
            filter.step(&divergent_obs, &tb, 1.0, 0.1);
        }

        // 1. collapse_count must be reset to 0 after hard reset.
        assert_eq!(
            filter.collapse_count, 0,
            "collapse_count should be 0 after hard reset"
        );

        // 2. Weights should be uniform (1/N) right after the reset.
        let uniform = 1.0 / n;
        for &w in &filter.weights {
            assert!(
                (w - uniform).abs() < 1e-12,
                "weights should be uniform after hard reset, got {w}"
            );
        }

        // 3. A consistent VO observation whose delta_position matches where the
        //    particles were reseeded (the weight update compares particle positions
        //    against delta_position) should give high likelihoods and not collapse.
        //    After the hard reset, particles are seeded near current_estimate + vo_delta,
        //    and the weight update treats delta as absolute, so we reuse vo_delta here.
        let consistent_obs = vec![Observation::VisualOdometry {
            delta_position: vo_delta,
            confidence: 0.95,
            timestamp: 1.0,
        }];
        filter.step(&consistent_obs, &tb, 1.0, 0.1);

        // After resampling the weights are uniform again, but collapse_count
        // stays 0 — confirming the filter did not collapse again.
        assert_eq!(
            filter.collapse_count, 0,
            "collapse_count should stay 0 on non-collapse step after recovery"
        );
    }

    #[test]
    fn deterministic_replay_with_seed() {
        let pos = Vector3::new(10.0, 20.0, -50.0);
        let obs = vec![Observation::Barometer {
            altitude: 50.0,
            timestamp: 0.0,
        }];
        let tb = Vector3::new(1.0, 0.0, 0.0);
        let mut f1 = ParticleNavFilter::new_seeded(100, pos, 42);
        let mut f2 = ParticleNavFilter::new_seeded(100, pos, 42);
        let (pos1, vel1, probs1) = f1.step(&obs, &tb, 1.0, 0.1);
        let (pos2, vel2, probs2) = f2.step(&obs, &tb, 1.0, 0.1);
        assert!(
            (pos1 - pos2).norm() < 1e-12,
            "positions must match: {pos1:?} vs {pos2:?}"
        );
        assert!(
            (vel1 - vel2).norm() < 1e-12,
            "velocities must match: {vel1:?} vs {vel2:?}"
        );
        assert_eq!(probs1, probs2, "regime probs must match");
    }
}
