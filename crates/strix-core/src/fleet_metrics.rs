//! Fleet-level metrics — velocity coherence and formation health.
//!
//! Provides aggregate measures of fleet coordination quality.

use nalgebra::Vector3;

/// Velocity coherence: magnitude of the mean unit-velocity vector.
///
/// Returns a value in [0, 1]:
/// - 1.0 → all drones flying in the same direction (perfect coherence)
/// - 0.0 → velocities are randomly oriented (no coherence)
///
/// Stationary drones (speed < `min_speed`) are excluded from the calculation.
pub fn velocity_coherence(velocities: &[Vector3<f64>], min_speed: f64) -> f64 {
    let mut sum = Vector3::zeros();
    let mut count = 0usize;

    for v in velocities {
        let speed = v.norm();
        if speed > min_speed {
            sum += v / speed; // unit velocity
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    sum.norm() / count as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_velocities_high_coherence() {
        let vels = vec![
            Vector3::new(5.0, 0.0, 0.0),
            Vector3::new(10.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
        ];
        let c = velocity_coherence(&vels, 0.1);
        assert!(c > 0.99, "parallel → coherence {c} should be ~1.0");
    }

    #[test]
    fn opposite_velocities_low_coherence() {
        let vels = vec![Vector3::new(5.0, 0.0, 0.0), Vector3::new(-5.0, 0.0, 0.0)];
        let c = velocity_coherence(&vels, 0.1);
        assert!(c < 0.01, "opposite → coherence {c} should be ~0.0");
    }

    #[test]
    fn random_directions_low_coherence() {
        // 4 cardinal directions cancel out.
        let vels = vec![
            Vector3::new(5.0, 0.0, 0.0),
            Vector3::new(-5.0, 0.0, 0.0),
            Vector3::new(0.0, 5.0, 0.0),
            Vector3::new(0.0, -5.0, 0.0),
        ];
        let c = velocity_coherence(&vels, 0.1);
        assert!(c < 0.01, "cardinal cancel → coherence {c}");
    }

    #[test]
    fn stationary_drones_excluded() {
        let vels = vec![
            Vector3::new(5.0, 0.0, 0.0),
            Vector3::new(0.001, 0.0, 0.0), // nearly stationary
        ];
        let c = velocity_coherence(&vels, 0.1);
        assert!(c > 0.99, "stationary excluded → coherence {c}");
    }

    #[test]
    fn empty_returns_zero() {
        let c = velocity_coherence(&[], 0.1);
        assert!((c).abs() < 1e-12);
    }

    #[test]
    fn all_stationary_returns_zero() {
        let vels = vec![Vector3::new(0.01, 0.0, 0.0); 5];
        let c = velocity_coherence(&vels, 0.1);
        assert!((c).abs() < 1e-12);
    }
}
