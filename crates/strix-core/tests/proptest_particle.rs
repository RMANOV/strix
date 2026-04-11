//! Property-based tests for ParticleNavFilter invariants.
//!
//! These tests verify that the particle filter maintains its mathematical
//! contracts under arbitrary inputs — something unit tests cannot cover.

use nalgebra::Vector3;
use proptest::prelude::*;
use strix_core::particle_common::effective_sample_size;
use strix_core::particle_nav::ParticleNavFilter;
use strix_core::state::Observation;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

fn reasonable_f64() -> impl Strategy<Value = f64> {
    prop::num::f64::NORMAL.prop_map(|v| v.clamp(-1000.0, 1000.0))
}

fn vec3_strategy() -> impl Strategy<Value = Vector3<f64>> {
    (reasonable_f64(), reasonable_f64(), reasonable_f64())
        .prop_map(|(x, y, z)| Vector3::new(x, y, z))
}

fn imu_observation() -> impl Strategy<Value = Observation> {
    (vec3_strategy(), 0.0..100.0f64).prop_map(|(accel, ts)| Observation::Imu {
        acceleration: accel,
        gyro: None,
        timestamp: ts,
    })
}

fn visual_odometry_observation() -> impl Strategy<Value = Observation> {
    (vec3_strategy(), 0.0..1.0f64, 0.0..100.0f64).prop_map(|(delta, conf, ts)| {
        Observation::VisualOdometry {
            delta_position: delta * 0.01, // small deltas
            confidence: conf,
            timestamp: ts,
        }
    })
}

fn observation_strategy() -> impl Strategy<Value = Observation> {
    prop_oneof![imu_observation(), visual_odometry_observation(),]
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// After any single step, particle weights must sum to 1 within tolerance.
    #[test]
    fn weights_sum_to_one_after_step(
        obs in prop::collection::vec(observation_strategy(), 1..5),
        bearing in vec3_strategy(),
        dt in 0.01..0.5f64,
        seed in 0u64..10000,
    ) {
        let mut filter = ParticleNavFilter::new_seeded(100, Vector3::zeros(), seed);
        let _ = filter.step(&obs, &bearing, 0.3, dt);

        let sum: f64 = filter.weights.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-6,
            "weights sum = {sum}, expected 1.0"
        );
    }

    /// No NaN should appear in estimated position or velocity after any step.
    #[test]
    fn no_nan_in_estimates(
        obs in prop::collection::vec(observation_strategy(), 0..5),
        bearing in vec3_strategy(),
        dt in 0.01..0.5f64,
        seed in 0u64..10000,
    ) {
        let mut filter = ParticleNavFilter::new_seeded(100, Vector3::zeros(), seed);
        let (pos, vel, probs) = filter.step(&obs, &bearing, 0.3, dt);

        prop_assert!(pos.iter().all(|v| v.is_finite()), "NaN in position: {pos:?}");
        prop_assert!(vel.iter().all(|v| v.is_finite()), "NaN in velocity: {vel:?}");
        prop_assert!(probs.iter().all(|v| v.is_finite()), "NaN in regime probs: {probs:?}");
    }

    /// Effective sample size must be in [1, n_particles].
    #[test]
    fn ess_in_valid_range(
        obs in prop::collection::vec(observation_strategy(), 0..5),
        bearing in vec3_strategy(),
        dt in 0.01..0.5f64,
        seed in 0u64..10000,
    ) {
        let n = 100;
        let mut filter = ParticleNavFilter::new_seeded(n, Vector3::zeros(), seed);
        let _ = filter.step(&obs, &bearing, 0.3, dt);

        let ess = effective_sample_size(&filter.weights);
        prop_assert!(
            ess >= 0.0 && ess <= n as f64 + 1e-6,
            "ESS = {ess}, expected [0, {n}]"
        );
    }

    /// No NaN in weights after any step.
    #[test]
    fn no_nan_in_weights(
        obs in prop::collection::vec(observation_strategy(), 0..5),
        bearing in vec3_strategy(),
        dt in 0.01..0.5f64,
        seed in 0u64..10000,
    ) {
        let mut filter = ParticleNavFilter::new_seeded(100, Vector3::zeros(), seed);
        let _ = filter.step(&obs, &bearing, 0.3, dt);

        prop_assert!(
            filter.weights.iter().all(|w| w.is_finite() && *w >= 0.0),
            "Invalid weights: {:?}",
            filter.weights.iter().filter(|w| !w.is_finite()).collect::<Vec<_>>()
        );
    }

    /// Regime probabilities must sum to 1.
    #[test]
    fn regime_probs_sum_to_one(
        obs in prop::collection::vec(observation_strategy(), 0..5),
        bearing in vec3_strategy(),
        dt in 0.01..0.5f64,
        seed in 0u64..10000,
    ) {
        let mut filter = ParticleNavFilter::new_seeded(100, Vector3::zeros(), seed);
        let (_, _, probs) = filter.step(&obs, &bearing, 0.3, dt);

        let sum: f64 = probs.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "regime probs sum = {sum}, expected 1.0"
        );
    }

    /// Multiple consecutive steps should not cause divergence.
    #[test]
    fn multi_step_no_divergence(
        obs_count in 1..4usize,
        seed in 0u64..10000,
    ) {
        let mut filter = ParticleNavFilter::new_seeded(100, Vector3::zeros(), seed);
        let obs = vec![Observation::Imu {
            acceleration: Vector3::new(0.0, 0.0, -9.81),
            gyro: None,
            timestamp: 0.0,
        }];
        let bearing = Vector3::new(1.0, 0.0, 0.0);

        for i in 0..obs_count * 10 {
            let (pos, _, _) = filter.step(&obs, &bearing, 0.3, 0.1);
            prop_assert!(
                pos.norm() < 1e6,
                "position diverged at step {i}: {pos:?}"
            );
        }
    }
}
