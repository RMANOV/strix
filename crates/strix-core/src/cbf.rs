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

/// Spherical no-fly zone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoFlyZone {
    /// Center position in NED.
    pub center: Vector3<f64>,
    /// Exclusion radius (meters).
    pub radius: f64,
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
    let mut correction = Vector3::zeros();
    let mut active_count = 0u32;

    // ── Constraint 1: Inter-drone separation ────────────────────────
    let d_min_sq = config.min_separation * config.min_separation;

    for neighbor in neighbors {
        let diff = my_pos - neighbor; // vector from neighbor to me
        let dist_sq = diff.norm_squared();

        // h(x) = ||p_i - p_j||^2 - d_min^2
        let h = dist_sq - d_min_sq;

        if h < 0.0 {
            // Already inside minimum separation — push away hard.
            let dist = dist_sq.sqrt().max(1e-6);
            correction += diff / dist * config.max_correction;
            active_count += 1;
            continue;
        }

        // dh/dt = 2 * (p_i - p_j) . (v_i - v_j)
        // For safety: dh/dt + alpha * h >= 0
        // Assume neighbor velocity ≈ 0 for worst case.
        let dh_dt = 2.0 * diff.dot(&(desired_vel + &correction));
        let constraint = dh_dt + config.alpha * h;

        if constraint < 0.0 {
            // Need to correct: push velocity away from neighbor.
            let dist = dist_sq.sqrt().max(1e-6);
            let direction = diff / dist;
            // Correction magnitude: just enough to satisfy constraint.
            // dh/dt_corrected = dh/dt + 2*||diff||*correction_magnitude
            let needed = -constraint / (2.0 * dist).max(1e-6);
            correction += direction * needed.min(config.max_correction);
            active_count += 1;
        }
    }

    // ── Constraint 2: Altitude floor (NED: z positive is down) ──────
    // h(x) = z - ceiling_ned >= 0 (ceiling_ned is the most negative = highest altitude)
    // In NED: altitude_ceiling_ned = -5.0 means 5m AGL, z must be >= -5.0 (must not go higher than -5.0)
    // Wait — NED convention: z positive = down. altitude_floor_ned = -500 means max altitude 500m.
    // The drone must stay BELOW the floor and ABOVE the ceiling in NED terms:
    //   z >= altitude_floor_ned (don't go too high — floor is the negative limit)
    //   z <= altitude_ceiling_ned (don't go too low — ceiling is the positive limit)
    //
    // Actually, let's think about this clearly:
    // NED: z=0 is ground level, z<0 is above ground, z>0 is below ground (underground).
    // For drones: z is always negative (they're in the air).
    // altitude_floor_ned = -500: max altitude (z must be >= -500, i.e. no higher than 500m)
    // altitude_ceiling_ned = -5: min altitude (z must be <= -5, i.e. no lower than 5m)
    {
        // Max altitude constraint: h = z - altitude_floor_ned >= 0
        let h_floor = my_pos.z - config.altitude_floor_ned;
        if h_floor < 0.0 {
            // Above max altitude — push down (increase z).
            correction.z += config.max_correction;
            active_count += 1;
        } else {
            let dh_dt = desired_vel.z + correction.z;
            let constraint = dh_dt + config.alpha * h_floor;
            if constraint < 0.0 {
                // Approaching max altitude — add downward correction.
                correction.z += (-constraint).min(config.max_correction);
                active_count += 1;
            }
        }

        // Min altitude constraint: h = altitude_ceiling_ned - z >= 0
        let h_ceil = config.altitude_ceiling_ned - my_pos.z;
        if h_ceil < 0.0 {
            // Below min altitude — push up (decrease z).
            correction.z -= config.max_correction;
            active_count += 1;
        } else {
            let dh_dt = -(desired_vel.z + correction.z);
            let constraint = dh_dt + config.alpha * h_ceil;
            if constraint < 0.0 {
                correction.z -= (-constraint).min(config.max_correction);
                active_count += 1;
            }
        }
    }

    // ── Constraint 3: No-fly zone avoidance ─────────────────────────
    for zone in nfz {
        let diff = my_pos - &zone.center;
        let dist_sq = diff.norm_squared();
        let r_sq = zone.radius * zone.radius;

        // h(x) = ||p - center||^2 - r^2
        let h = dist_sq - r_sq;

        if h < 0.0 {
            // Inside NFZ — push out maximally.
            let dist = dist_sq.sqrt().max(1e-6);
            correction += diff / dist * config.max_correction;
            active_count += 1;
            continue;
        }

        let dh_dt = 2.0 * diff.dot(&(desired_vel + &correction));
        let constraint = dh_dt + config.alpha * h;

        if constraint < 0.0 {
            let dist = dist_sq.sqrt().max(1e-6);
            let direction = diff / dist;
            let needed = -constraint / (2.0 * dist).max(1e-6);
            correction += direction * needed.min(config.max_correction);
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
        if (pos - &zone.center).norm_squared() < zone.radius * zone.radius {
            return false;
        }
    }

    true
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

        let pos_unsafe = Vector3::new(3.0, 0.0, -50.0); // 7m from neighbor — wait, that's >5m
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
}
