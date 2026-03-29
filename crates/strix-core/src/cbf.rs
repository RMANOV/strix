//! Control Barrier Functions — provable safety for aggressive maneuvers.
//!
//! CBFs provide formal safety guarantees by minimally modifying velocity
//! commands to maintain constraint satisfaction. Unlike soft repulsion
//! (pheromones), CBFs provide provable bounds:
//!
//! - **Inter-drone separation**: h(x) = ||p_i - p_j||^2 - d_min^2 >= 0
//! - **Altitude floor/ceiling**: h(x) = z - z_min >= 0, z_max - z >= 0
//! - **No-fly zone avoidance**: h(x) = ||p - center||^2 - r^2 >= 0
//!
//! The CBF filter adjusts the desired velocity with minimum correction:
//!   safe_vel = desired_vel + correction
//! where correction is the minimum needed to keep dh/dt + alpha*h >= 0
//! (the CBF constraint ensuring the barrier stays positive).
//!
//! # Coordinate Convention — NED (North-East-Down)
//!
//! All positions follow the NED frame: X=North, Y=East, **Z=Down**.
//! Negative Z values represent altitude above ground level (AGL).
//! - `altitude_floor_ned = -500.0` means the drone may not exceed 500 m AGL.
//! - `altitude_ceiling_ned = -5.0`  means the drone must stay above 5 m AGL.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// CBF tuning parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CbfConfig {
    /// Minimum inter-drone separation distance (meters).
    pub min_separation: f64,
    /// Altitude floor in NED (negative = above ground). Default: -500.0 (max 500m AGL).
    pub altitude_floor_ned: f64,
    /// Altitude ceiling in NED (negative = above ground). Default: -5.0 (min 5m AGL).
    pub altitude_ceiling_ned: f64,
    /// CBF decay rate alpha — controls aggressiveness of correction.
    /// Higher alpha = earlier, gentler corrections. Lower = later, sharper.
    pub alpha: f64,
    /// Maximum correction magnitude (m/s). Prevents CBF from dominating.
    pub max_correction: f64,
}

impl Default for CbfConfig {
    fn default() -> Self {
        Self {
            min_separation: 5.0,
            altitude_floor_ned: -500.0, // max 500m altitude (NED: down is positive)
            altitude_ceiling_ned: -5.0, // min 5m altitude
            alpha: 1.0,
            max_correction: 10.0,
        }
    }
}

impl CbfConfig {
    /// Modulate safety parameters by fear level F ∈ [0,1].
    ///
    /// Higher fear → wider separation (5m→15m), stronger corrections (10→25 m/s),
    /// earlier intervention (alpha 1→2).
    pub fn with_fear(&self, f: f64) -> Self {
        let f = if f.is_nan() || f.is_infinite() {
            0.0
        } else {
            f.clamp(0.0, 1.0)
        };
        Self {
            min_separation: self.min_separation + f * 10.0, // 5→15m
            max_correction: self.max_correction + f * 15.0, // 10→25 m/s
            alpha: self.alpha + f * 1.0,                    // 1→2
            ..*self
        }
    }
}

/// Spherical no-fly zone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoFlyZone {
    /// Center position in NED.
    pub center: Vector3<f64>,
    /// Exclusion radius (meters).
    pub radius: f64,
}

/// Neighbor state used by the CBF to account for relative motion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborState {
    /// Neighbor position in NED.
    pub position: Vector3<f64>,
    /// Neighbor velocity in NED.
    pub velocity: Vector3<f64>,
}

impl NeighborState {
    /// Convenience constructor for legacy call sites that only know position.
    pub fn stationary(position: Vector3<f64>) -> Self {
        Self {
            position,
            velocity: Vector3::zeros(),
        }
    }
}

/// Result of a CBF filter operation.
#[derive(Debug, Clone)]
pub struct CbfResult {
    /// The safe velocity after CBF correction.
    pub safe_velocity: Vector3<f64>,
    /// Total correction applied (safe - desired).
    pub correction: Vector3<f64>,
    /// Whether any constraint was active (correction was needed).
    pub any_active: bool,
    /// Number of active constraints.
    pub active_count: u32,
}

/// Apply CBF safety filter to a desired velocity command.
///
/// Given the drone's position, desired velocity, neighbor positions,
/// and no-fly zones, compute the minimum velocity correction needed
/// to maintain all safety constraints.
///
/// # Arguments
/// * `my_pos` - Current drone position (NED)
/// * `desired_vel` - Desired velocity from waypoint controller
/// * `neighbors` - Positions of other drones
/// * `nfz` - No-fly zones to avoid
/// * `config` - CBF parameters
pub fn cbf_filter(
    my_pos: &Vector3<f64>,
    desired_vel: &Vector3<f64>,
    neighbors: &[Vector3<f64>],
    nfz: &[NoFlyZone],
    config: &CbfConfig,
) -> CbfResult {
    let neighbor_states: Vec<NeighborState> = neighbors
        .iter()
        .cloned()
        .map(NeighborState::stationary)
        .collect();
    cbf_filter_with_neighbor_states(my_pos, desired_vel, &neighbor_states, nfz, config)
}

/// Apply the CBF using full neighbor state (position + velocity).
pub fn cbf_filter_with_neighbor_states(
    my_pos: &Vector3<f64>,
    desired_vel: &Vector3<f64>,
    neighbors: &[NeighborState],
    nfz: &[NoFlyZone],
    config: &CbfConfig,
) -> CbfResult {
    let mut correction = Vector3::zeros();
    let mut active_count = 0u32;

    // ── Constraint 1: Inter-drone separation with relative-motion inflation ──
    for neighbor in neighbors {
        let effective_vel = desired_vel + correction;
        let diff = my_pos - neighbor.position;
        let dist_sq = diff.norm_squared();
        let dist = dist_sq.sqrt();
        let direction = outward_direction(&diff, &effective_vel);
        let relative_vel = effective_vel - neighbor.velocity;
        let closing_speed = (-relative_vel.dot(&direction)).max(0.0);
        // TTC-aware margin: scale by time-to-collision for faster response
        let ttc = if closing_speed > 0.1 {
            dist / closing_speed
        } else {
            f64::INFINITY
        };
        let margin_scale = if ttc < 3.0 {
            1.5
        } else if ttc < 6.0 {
            1.0
        } else {
            0.5
        };
        let effective_separation = config.min_separation + margin_scale * closing_speed;
        let h = dist_sq - effective_separation * effective_separation;

        if h < 0.0 {
            let radial_rel = relative_vel.dot(&direction);
            let cancel_approach = (-radial_rel).max(0.0);
            let penetration = (effective_separation - dist).max(0.5);
            let push = (config.alpha * penetration + cancel_approach).min(config.max_correction);
            correction += direction * push;
            let mag = correction.norm();
            if mag > config.max_correction {
                correction *= config.max_correction / mag;
            }
            active_count += 1;
            continue;
        }

        let dh_dt = 2.0 * diff.dot(&(effective_vel - neighbor.velocity));
        let constraint = dh_dt + config.alpha * h;
        if constraint < 0.0 {
            let needed = -constraint / (2.0 * dist.max(1e-6));
            let radial_rel = relative_vel.dot(&direction);
            let cancel_approach = (-radial_rel).max(0.0) * 0.5;
            correction += direction * (needed + cancel_approach).min(config.max_correction);
            let mag = correction.norm();
            if mag > config.max_correction {
                correction *= config.max_correction / mag;
            }
            active_count += 1;
        }
    }

    // ── Constraint 2: Altitude bounds with predictive vertical margins ──────
    {
        let effective_vz = desired_vel.z + correction.z;

        // Max altitude bound (NED floor): start correcting earlier when climbing fast.
        let climb_margin = 1.0 + (-effective_vz).max(0.0) * 0.5;
        let floor_limit = config.altitude_floor_ned + climb_margin;
        let h_floor = my_pos.z - floor_limit;
        if h_floor < 0.0 {
            let penetration = (-h_floor).max(0.5);
            correction.z +=
                (config.alpha * penetration + (-effective_vz).max(0.0)).min(config.max_correction);
            let mag = correction.norm();
            if mag > config.max_correction {
                correction *= config.max_correction / mag;
            }
            active_count += 1;
        } else {
            let dh_dt = desired_vel.z + correction.z;
            let constraint = dh_dt + config.alpha * h_floor;
            if constraint < 0.0 {
                correction.z += (-constraint).min(config.max_correction);
                let mag = correction.norm();
                if mag > config.max_correction {
                    correction *= config.max_correction / mag;
                }
                active_count += 1;
            }
        }

        // Min altitude bound (NED ceiling): start correcting earlier when descending fast.
        let effective_vz = desired_vel.z + correction.z;
        let descent_margin = 1.0 + effective_vz.max(0.0) * 0.5;
        let ceiling_limit = config.altitude_ceiling_ned - descent_margin;
        let h_ceil = ceiling_limit - my_pos.z;
        if h_ceil < 0.0 {
            let penetration = (-h_ceil).max(0.5);
            correction.z -=
                (config.alpha * penetration + effective_vz.max(0.0)).min(config.max_correction);
            let mag = correction.norm();
            if mag > config.max_correction {
                correction *= config.max_correction / mag;
            }
            active_count += 1;
        } else {
            let dh_dt = -(desired_vel.z + correction.z);
            let constraint = dh_dt + config.alpha * h_ceil;
            if constraint < 0.0 {
                correction.z -= (-constraint).min(config.max_correction);
                let mag = correction.norm();
                if mag > config.max_correction {
                    correction *= config.max_correction / mag;
                }
                active_count += 1;
            }
        }
    }

    // ── Constraint 3: No-fly zone avoidance with ingress-aware inflation ────
    for zone in nfz {
        let effective_vel = desired_vel + correction;
        let diff = my_pos - zone.center;
        let dist_sq = diff.norm_squared();
        let dist = dist_sq.sqrt();
        let direction = outward_direction(&diff, &effective_vel);
        let inward_speed = (-effective_vel.dot(&direction)).max(0.0);
        let effective_radius = zone.radius + 0.5 * inward_speed + config.min_separation * 0.2;
        let h = dist_sq - effective_radius * effective_radius;

        if h < 0.0 {
            let penetration = (effective_radius - dist).max(0.5);
            let push = (config.alpha * penetration + inward_speed).min(config.max_correction);
            correction += direction * push;
            let mag = correction.norm();
            if mag > config.max_correction {
                correction *= config.max_correction / mag;
            }
            active_count += 1;
            continue;
        }

        let dh_dt = 2.0 * diff.dot(&effective_vel);
        let constraint = dh_dt + config.alpha * h;
        if constraint < 0.0 {
            let needed = -constraint / (2.0 * dist.max(1e-6));
            correction += direction * (needed + inward_speed * 0.5).min(config.max_correction);
            let mag = correction.norm();
            if mag > config.max_correction {
                correction *= config.max_correction / mag;
            }
            active_count += 1;
        }
    }

    // ── Clamp total correction ──────────────────────────────────────
    let corr_mag = correction.norm();
    if corr_mag > config.max_correction {
        correction *= config.max_correction / corr_mag;
    }

    CbfResult {
        safe_velocity: desired_vel + correction,
        correction,
        any_active: active_count > 0,
        active_count,
    }
}

fn outward_direction(diff: &Vector3<f64>, fallback_velocity: &Vector3<f64>) -> Vector3<f64> {
    let diff_norm = diff.norm();
    if diff_norm > 1e-6 {
        return Vector3::new(diff.x / diff_norm, diff.y / diff_norm, diff.z / diff_norm);
    }

    let vel_norm = fallback_velocity.norm();
    if vel_norm > 1e-6 {
        return Vector3::new(
            -fallback_velocity.x / vel_norm,
            -fallback_velocity.y / vel_norm,
            -fallback_velocity.z / vel_norm,
        );
    }

    Vector3::new(1.0, 0.0, 0.0)
}

/// Quick check: is a position safe (all constraints satisfied)?
pub fn is_position_safe(
    pos: &Vector3<f64>,
    neighbors: &[Vector3<f64>],
    nfz: &[NoFlyZone],
    config: &CbfConfig,
) -> bool {
    let d_min_sq = config.min_separation * config.min_separation;

    // Inter-drone separation.
    for neighbor in neighbors {
        if (pos - neighbor).norm_squared() < d_min_sq {
            return false;
        }
    }

    // Altitude bounds.
    if pos.z < config.altitude_floor_ned || pos.z > config.altitude_ceiling_ned {
        return false;
    }

    // No-fly zones.
    for zone in nfz {
        if (pos - zone.center).norm_squared() < zone.radius * zone.radius {
            return false;
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Deadlock Detection & Escape
// ---------------------------------------------------------------------------

/// Result of a deadlock detection check.
#[derive(Debug, Clone)]
pub struct DeadlockResult {
    /// Whether a mutual deadlock was detected.
    pub is_deadlocked: bool,
    /// Number of drones involved in the deadlock cluster.
    pub involved_count: usize,
    /// Indices of involved drones.
    pub involved_indices: Vec<usize>,
}

/// Lightweight pairwise separation check (no altitude/NFZ overhead).
fn is_pair_blocking(
    pos_i: &Vector3<f64>,
    vel_i: &Vector3<f64>,
    pos_j: &Vector3<f64>,
    vel_j: &Vector3<f64>,
    config: &CbfConfig,
) -> bool {
    let diff = pos_i - pos_j;
    let dist = diff.norm();
    if dist < 1e-6 {
        return true;
    }
    let dir = diff / dist;
    let rel_vel = vel_i - vel_j;
    let closing = (-rel_vel.dot(&dir)).max(0.0);
    let ttc = if closing > 0.1 {
        dist / closing
    } else {
        f64::INFINITY
    };
    let scale = if ttc < 3.0 {
        1.5
    } else if ttc < 6.0 {
        1.0
    } else {
        0.5
    };
    let eff_sep = config.min_separation + scale * closing;
    dist < eff_sep
}

/// Detect mutual CBF blocking among a group of drones.
///
/// A deadlock exists when 3+ drones each have active CBF constraints against
/// at least 2 of the others, forming a tightly coupled cluster.
pub fn detect_deadlock(
    positions: &[Vector3<f64>],
    velocities: &[Vector3<f64>],
    _nfz: &[NoFlyZone],
    config: &CbfConfig,
) -> DeadlockResult {
    let n = positions.len();
    if n < 3 {
        return DeadlockResult {
            is_deadlocked: false,
            involved_count: 0,
            involved_indices: vec![],
        };
    }
    let mut degree = vec![0usize; n];
    for i in 0..n {
        for j in (i + 1)..n {
            if is_pair_blocking(
                &positions[i],
                &velocities[i],
                &positions[j],
                &velocities[j],
                config,
            ) {
                degree[i] += 1;
                degree[j] += 1;
            }
        }
    }
    // Deadlock cluster: all drones with degree >= 2
    let involved: Vec<usize> = (0..n).filter(|&i| degree[i] >= 2).collect();
    let is_deadlocked = involved.len() >= 3;
    DeadlockResult {
        is_deadlocked,
        involved_count: involved.len(),
        involved_indices: involved,
    }
}

/// Generate escape velocity corrections for deadlocked drones.
///
/// Strategy: each drone gets a perpendicular velocity (rotated from
/// centroid-to-drone) plus an alternating altitude shift.
pub fn generate_escape_maneuvers(
    positions: &[Vector3<f64>],
    config: &CbfConfig,
) -> Vec<Vector3<f64>> {
    let n = positions.len();
    if n == 0 {
        return vec![];
    }
    let centroid = positions.iter().fold(Vector3::zeros(), |a, p| a + p) / n as f64;
    let escape_speed = config.max_correction * 0.5;
    positions
        .iter()
        .enumerate()
        .map(|(i, pos)| {
            let outward = pos - centroid;
            let norm = outward.norm().max(1e-6);
            let dir = outward / norm;
            // Perpendicular in xy-plane
            let perp = Vector3::new(-dir.y, dir.x, 0.0);
            // Alternating altitude shift
            let alt = if i % 2 == 0 { 3.0 } else { -3.0 };
            (dir * escape_speed * 0.5 + perp * escape_speed * 0.5 + Vector3::new(0.0, 0.0, alt))
                .cap_magnitude(config.max_correction)
        })
        .collect()
}

/// Clamp vector magnitude (helper).
trait CapMagnitude {
    fn cap_magnitude(self, max: f64) -> Self;
}
impl CapMagnitude for Vector3<f64> {
    fn cap_magnitude(self, max: f64) -> Self {
        let n = self.norm();
        if n > max {
            self * (max / n)
        } else {
            self
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> CbfConfig {
        CbfConfig::default()
    }

    #[test]
    fn no_neighbors_no_correction() {
        let pos = Vector3::new(0.0, 0.0, -50.0); // 50m altitude
        let vel = Vector3::new(5.0, 0.0, 0.0);
        let result = cbf_filter(&pos, &vel, &[], &[], &default_config());
        assert!(!result.any_active);
        assert!(result.correction.norm() < 1e-10);
    }

    #[test]
    fn close_neighbor_gets_corrected() {
        let config = default_config(); // min_separation = 5
        let pos = Vector3::new(0.0, 0.0, -50.0);
        let neighbor = Vector3::new(3.0, 0.0, -50.0); // 3m away < 5m min
        let vel = Vector3::new(5.0, 0.0, 0.0); // heading toward neighbor

        let result = cbf_filter(&pos, &vel, &[neighbor], &[], &config);
        assert!(result.any_active);
        // Correction should push away from neighbor (negative x direction).
        assert!(
            result.correction.x < 0.0,
            "should push away: correction.x = {}",
            result.correction.x
        );
    }

    #[test]
    fn approaching_neighbor_corrected_before_collision() {
        let config = CbfConfig {
            min_separation: 5.0,
            alpha: 2.0, // more aggressive
            ..default_config()
        };
        let pos = Vector3::new(0.0, 0.0, -50.0);
        let neighbor = Vector3::new(8.0, 0.0, -50.0); // 8m away > 5m but close
        let vel = Vector3::new(10.0, 0.0, 0.0); // fast approach

        let result = cbf_filter(&pos, &vel, &[neighbor], &[], &config);
        // The barrier should activate because dh/dt + alpha*h could be negative.
        // With h = 64 - 25 = 39, dh/dt = 2*(-8)*10 = -160, constraint = -160 + 2*39 = -82 < 0
        assert!(result.any_active, "should activate before collision");
    }

    #[test]
    fn nfz_avoidance() {
        let config = default_config();
        let nfz = vec![NoFlyZone {
            center: Vector3::new(20.0, 0.0, -50.0),
            radius: 15.0,
        }];
        let pos = Vector3::new(10.0, 0.0, -50.0); // 10m from center, inside 15m radius
        let vel = Vector3::new(5.0, 0.0, 0.0); // heading into NFZ

        let result = cbf_filter(&pos, &vel, &[], &nfz, &config);
        assert!(result.any_active);
        // Should push away from NFZ center (negative x).
        assert!(
            result.correction.x < 0.0,
            "should push away from NFZ: {}",
            result.correction.x
        );
    }

    #[test]
    fn altitude_floor_enforced() {
        let config = default_config(); // floor = -500 (max altitude 500m)
        let pos = Vector3::new(0.0, 0.0, -490.0); // 490m altitude, near limit
        let vel = Vector3::new(0.0, 0.0, -5.0); // climbing (decreasing z)

        let result = cbf_filter(&pos, &vel, &[], &[], &config);
        // h = -490 - (-500) = 10, dh/dt = -5, constraint = -5 + 1*10 = 5 >= 0
        // Should be OK at alpha=1, but let's test at the limit
        let pos_over = Vector3::new(0.0, 0.0, -510.0); // above max altitude
        let result2 = cbf_filter(&pos_over, &vel, &[], &[], &config);
        assert!(result2.any_active, "should activate above max altitude");
        assert!(
            result2.correction.z > 0.0,
            "should push down: {}",
            result2.correction.z
        );
        let _ = result;
    }

    #[test]
    fn altitude_ceiling_enforced() {
        let config = default_config(); // ceiling = -5 (min 5m altitude)
        let pos = Vector3::new(0.0, 0.0, -3.0); // only 3m altitude, below minimum
        let vel = Vector3::new(5.0, 0.0, 1.0); // descending

        let result = cbf_filter(&pos, &vel, &[], &[], &config);
        assert!(result.any_active, "should activate below min altitude");
        assert!(
            result.correction.z < 0.0,
            "should push up (decrease z): {}",
            result.correction.z
        );
    }

    #[test]
    fn safe_position_check() {
        let config = default_config();
        let pos_safe = Vector3::new(0.0, 0.0, -50.0);
        let neighbor = Vector3::new(10.0, 0.0, -50.0); // 10m > 5m min
        assert!(is_position_safe(&pos_safe, &[neighbor], &[], &config));

        let pos_unsafe = Vector3::new(3.0, 0.0, -50.0); // position near close_neighbor
        let close_neighbor = Vector3::new(4.0, 0.0, -50.0); // 1m away
        assert!(!is_position_safe(
            &pos_unsafe,
            &[close_neighbor],
            &[],
            &config
        ));
    }

    #[test]
    fn nfz_safety_check() {
        let config = default_config();
        let nfz = vec![NoFlyZone {
            center: Vector3::new(100.0, 0.0, -50.0),
            radius: 30.0,
        }];
        let inside = Vector3::new(110.0, 0.0, -50.0); // 10m from center < 30m radius
        let outside = Vector3::new(200.0, 0.0, -50.0); // 100m from center > 30m radius

        assert!(!is_position_safe(&inside, &[], &nfz, &config));
        assert!(is_position_safe(&outside, &[], &nfz, &config));
    }

    #[test]
    fn correction_clamped_to_max() {
        let config = CbfConfig {
            max_correction: 5.0,
            ..default_config()
        };
        let pos = Vector3::new(0.0, 0.0, -50.0);
        // Place many close neighbors to force large correction.
        let neighbors = vec![
            Vector3::new(2.0, 0.0, -50.0),
            Vector3::new(0.0, 2.0, -50.0),
            Vector3::new(-2.0, 0.0, -50.0),
        ];
        let vel = Vector3::zeros();

        let result = cbf_filter(&pos, &vel, &neighbors, &[], &config);
        assert!(
            result.correction.norm() <= config.max_correction + 1e-6,
            "correction magnitude {} exceeds max {}",
            result.correction.norm(),
            config.max_correction
        );
    }

    #[test]
    fn moving_away_from_neighbor_no_correction() {
        let config = CbfConfig {
            min_separation: 5.0,
            alpha: 1.0,
            ..default_config()
        };
        let pos = Vector3::new(0.0, 0.0, -50.0);
        let neighbor = Vector3::new(8.0, 0.0, -50.0); // 8m, barrier positive
        let vel = Vector3::new(-5.0, 0.0, 0.0); // moving AWAY

        let result = cbf_filter(&pos, &vel, &[neighbor], &[], &config);
        // h = 64 - 25 = 39, dh/dt = 2*(-8)*(-5) = 80, constraint = 80 + 39 = 119 >> 0
        assert!(
            !result.any_active,
            "moving away should not trigger CBF, correction = {:?}",
            result.correction
        );
    }

    #[test]
    fn moving_neighbor_velocity_is_respected() {
        let config = default_config();
        let pos = Vector3::new(0.0, 0.0, -50.0);
        let vel = Vector3::zeros();
        let neighbors = vec![NeighborState {
            position: Vector3::new(8.0, 0.0, -50.0),
            velocity: Vector3::new(-8.0, 0.0, 0.0),
        }];

        let result = cbf_filter_with_neighbor_states(&pos, &vel, &neighbors, &[], &config);
        assert!(
            result.any_active,
            "closing neighbor velocity should trigger CBF"
        );
        assert!(
            result.correction.x < 0.0,
            "correction should push away: {:?}",
            result.correction
        );
    }

    #[test]
    fn fast_climb_triggers_altitude_margin_before_floor_breach() {
        let config = default_config();
        let pos = Vector3::new(0.0, 0.0, -494.0);
        let vel = Vector3::new(0.0, 0.0, -5.0);

        let result = cbf_filter(&pos, &vel, &[], &[], &config);
        assert!(
            result.any_active,
            "predictive altitude margin should activate early"
        );
        assert!(
            result.correction.z > 0.0,
            "correction should push down: {:?}",
            result.correction
        );
    }

    #[test]
    fn fast_ingress_to_nfz_triggers_predictive_margin() {
        let config = default_config();
        let nfz = vec![NoFlyZone {
            center: Vector3::new(26.0, 0.0, -50.0),
            radius: 15.0,
        }];
        let pos = Vector3::new(0.0, 0.0, -50.0);
        let vel = Vector3::new(8.0, 0.0, 0.0);

        let result = cbf_filter(&pos, &vel, &[], &nfz, &config);
        assert!(
            result.any_active,
            "predictive NFZ margin should activate before crossing"
        );
        assert!(
            result.correction.x < 0.0,
            "correction should push away from NFZ: {:?}",
            result.correction
        );
    }

    // ── C3: TTC-aware margins ──

    #[test]
    fn fast_closing_gets_larger_margin_than_slow() {
        let config = CbfConfig::default();
        let pos = Vector3::new(0.0, 0.0, -50.0);
        let neighbor_pos = Vector3::new(12.0, 0.0, -50.0);
        let n = NeighborState {
            position: neighbor_pos,
            velocity: Vector3::zeros(),
        };
        let r_slow = cbf_filter_with_neighbor_states(
            &pos,
            &Vector3::new(2.0, 0.0, 0.0),
            &[n.clone()],
            &[],
            &config,
        );
        let r_fast = cbf_filter_with_neighbor_states(
            &pos,
            &Vector3::new(15.0, 0.0, 0.0),
            &[n],
            &[],
            &config,
        );
        assert!(
            r_fast.correction.norm() > r_slow.correction.norm(),
            "fast={:.3} should exceed slow={:.3}",
            r_fast.correction.norm(),
            r_slow.correction.norm()
        );
    }

    // ── C1: Deadlock detection ──

    #[test]
    fn detect_deadlock_triangle() {
        let config = CbfConfig::default();
        let positions = vec![
            Vector3::new(0.0, 0.0, -50.0),
            Vector3::new(3.0, 0.0, -50.0),
            Vector3::new(1.5, 2.6, -50.0),
        ];
        let velocities = vec![Vector3::zeros(); 3];
        let result = detect_deadlock(&positions, &velocities, &[], &config);
        assert!(result.is_deadlocked, "tight triangle should deadlock");
        assert!(result.involved_count >= 3);
    }

    #[test]
    fn no_deadlock_when_spread() {
        let config = CbfConfig::default();
        let positions = vec![
            Vector3::new(0.0, 0.0, -50.0),
            Vector3::new(20.0, 0.0, -50.0),
            Vector3::new(40.0, 0.0, -50.0),
        ];
        let velocities = vec![Vector3::zeros(); 3];
        let result = detect_deadlock(&positions, &velocities, &[], &config);
        assert!(!result.is_deadlocked, "spread drones should not deadlock");
    }

    // ── C2: Escape maneuvers ──

    #[test]
    fn escape_maneuvers_separate_drones() {
        let config = CbfConfig::default();
        let positions = vec![
            Vector3::new(0.0, 0.0, -50.0),
            Vector3::new(3.0, 0.0, -50.0),
            Vector3::new(1.5, 2.6, -50.0),
        ];
        let escapes = generate_escape_maneuvers(&positions, &config);
        assert_eq!(escapes.len(), 3);
        // After applying escapes, centroid-relative distances should increase
        let centroid = positions.iter().fold(Vector3::zeros(), |a, p| a + p) / 3.0;
        for (i, esc) in escapes.iter().enumerate() {
            let old_dist = (positions[i] - centroid).norm();
            let new_dist = (positions[i] + esc - centroid).norm();
            assert!(
                new_dist > old_dist,
                "drone {i} should move away from centroid"
            );
        }
    }

    #[test]
    fn escape_uses_altitude_shift() {
        let config = CbfConfig::default();
        let positions = vec![Vector3::new(0.0, 0.0, -50.0), Vector3::new(3.0, 0.0, -50.0)];
        let escapes = generate_escape_maneuvers(&positions, &config);
        assert!(
            escapes.iter().any(|e| e.z.abs() > 0.1),
            "at least one escape should shift altitude"
        );
    }
}
