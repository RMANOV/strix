//! Shared particle filter primitives used by both [`particle_nav`] and
//! [`threat_tracker`].
//!
//! These functions implement the core numerical routines — weight
//! normalisation, effective sample size, and systematic resampling — that
//! are identical across the friendly-navigation and enemy-tracking particle
//! filters.  Centralising them here ensures a single source of truth and
//! makes the underflow / numerical-stability choices reviewable in one place.

use ndarray::Array2;
use rand::Rng;

// ---------------------------------------------------------------------------
// Weight normalisation
// ---------------------------------------------------------------------------

/// Normalise importance weights in-place with underflow protection.
///
/// The `+1e-300` floor prevents any weight from being exactly zero before
/// normalisation, which would cause division by zero or degenerate resampling
/// when all raw weights have underflowed to 0.0.
///
/// This is the same pattern used throughout the original 2D trading filter.
///
/// # Arguments
/// * `weights` — mutable slice of raw (unnormalised) importance weights.
///   After this call the weights will sum to 1.0.
pub fn normalize_weights(weights: &mut [f64]) {
    for w in weights.iter_mut() {
        *w += 1e-300;
    }
    let total: f64 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= total;
    }
}

// ---------------------------------------------------------------------------
// Gaussian Likelihood
// ---------------------------------------------------------------------------

/// Gaussian likelihood kernel: `exp(-0.5 * diff_sq / (sigma² + ε))`.
///
/// `diff_sq` is the squared Mahalanobis distance (or squared scalar diff).
/// `sigma` is the measurement noise standard deviation.
///
/// The `+1e-12` epsilon prevents division by zero for zero-noise sensors.
#[inline]
pub fn gaussian_likelihood(diff_sq: f64, sigma: f64) -> f64 {
    let sigma2 = sigma * sigma + 1e-12;
    (-0.5 * diff_sq / sigma2).exp()
}

// ---------------------------------------------------------------------------
// Effective Sample Size
// ---------------------------------------------------------------------------

/// Effective Sample Size: `ESS = 1 / (∑ wᵢ² + ε)`.
///
/// A measure of particle diversity.  With uniform weights `ESS ≈ N`;
/// with all weight on one particle `ESS ≈ 1`.  The `+1e-12` epsilon
/// prevents division by zero when weights are all zero.
///
/// # Arguments
/// * `weights` — normalised importance weights (should sum to 1.0).
pub fn effective_sample_size(weights: &[f64]) -> f64 {
    let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
    1.0 / (sum_sq + 1e-12)
}

// ---------------------------------------------------------------------------
// Systematic Resampling
// ---------------------------------------------------------------------------

/// O(N) systematic resampling for 6D particles, mutating in-place.
///
/// Resamples `particles` (Nx6) and `regimes` (length N) according to the
/// current `weights`, then resets all weights to the uniform value `1/N`.
///
/// The algorithm uses a single random offset `u ~ Uniform[0, 1/N)` and then
/// advances a pointer through the cumulative-weight distribution at regular
/// intervals of `1/N`.  This ensures O(N) runtime and lower variance than
/// independent multinomial sampling.
///
/// # Arguments
/// * `particles` — Nx6 array of `[x, y, z, vx, vy, vz]` particles (mutated).
/// * `weights`   — importance weights (mutated; reset to `1/N` on return).
/// * `regimes`   — regime label per particle (mutated in parallel with particles).
///
/// # Panics
/// Panics if `weights` is empty or if `particles.nrows()` / `regimes.len()`
/// do not match `weights.len()`.
pub fn systematic_resample_6d(
    particles: &mut Array2<f64>,
    weights: &mut [f64],
    regimes: &mut [u8],
) {
    let n = weights.len();
    assert!(n > 0, "cannot resample zero particles");
    assert_eq!(
        particles.nrows(),
        n,
        "particles row count must match weight count"
    );
    assert_eq!(regimes.len(), n, "regimes length must match weight count");

    // Build cumulative-weight array; pin the last element to exactly 1.0 to
    // avoid floating-point overshoot.
    let mut cumsum = vec![0.0_f64; n];
    cumsum[0] = weights[0];
    for i in 1..n {
        cumsum[i] = cumsum[i - 1] + weights[i];
    }
    cumsum[n - 1] = 1.0;

    let step = 1.0 / n as f64;
    let start_offset: f64 = rand::thread_rng().gen_range(0.0..step);

    // Allocate output buffers.
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

    *particles = new_particles;
    regimes.copy_from_slice(&new_regimes);
    let uniform_weight = 1.0 / n as f64;
    for w in weights.iter_mut() {
        *w = uniform_weight;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_weights_sums_to_one() {
        let mut weights = vec![2.0, 3.0, 5.0];
        normalize_weights(&mut weights);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn normalize_weights_underflow_protection() {
        // All weights zero — should not divide by zero; result should be uniform.
        let mut weights = vec![0.0, 0.0, 0.0];
        normalize_weights(&mut weights);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // All should be equal.
        assert!((weights[0] - weights[1]).abs() < 1e-15);
        assert!((weights[1] - weights[2]).abs() < 1e-15);
    }

    #[test]
    fn ess_uniform() {
        let n = 100_usize;
        let weights = vec![1.0 / n as f64; n];
        let ess = effective_sample_size(&weights);
        // Should be very close to N.
        assert!((ess - n as f64).abs() < 1.0);
    }

    #[test]
    fn ess_degenerate() {
        // All weight on one particle → ESS ≈ 1.
        let mut weights = vec![0.0_f64; 100];
        weights[0] = 1.0;
        let ess = effective_sample_size(&weights);
        assert!(ess < 2.0);
    }

    #[test]
    fn systematic_resample_uniform_weights() {
        let n = 50;
        let mut particles = Array2::<f64>::zeros((n, 6));
        // Give particles distinct x values so we can check copy fidelity.
        for i in 0..n {
            particles[[i, 0]] = i as f64;
        }
        let mut weights = vec![1.0 / n as f64; n];
        let mut regimes = vec![0u8; n];

        systematic_resample_6d(&mut particles, &mut weights, &mut regimes);

        assert_eq!(particles.nrows(), n);
        assert_eq!(regimes.len(), n);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        // All weights uniform after resample.
        let expected = 1.0 / n as f64;
        for &w in &weights {
            assert!((w - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn systematic_resample_concentrated_weights() {
        // Put all weight on particle 0 → all resampled particles should
        // copy from index 0 (x == 0.0).
        let n = 20;
        let mut particles = Array2::<f64>::zeros((n, 6));
        for i in 0..n {
            particles[[i, 0]] = i as f64;
        }
        let mut weights = vec![0.0_f64; n];
        weights[0] = 1.0;
        let mut regimes = vec![1u8; n];

        systematic_resample_6d(&mut particles, &mut weights, &mut regimes);

        for i in 0..n {
            assert_eq!(
                particles[[i, 0]],
                0.0,
                "all resampled particles should come from index 0"
            );
        }
    }

    // ── Invariant tests: normalize_weights edge cases ──────────────────────

    #[test]
    fn normalize_weights_all_equal_gives_uniform() {
        let n = 10;
        let mut weights = vec![1.0_f64; n];
        normalize_weights(&mut weights);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "weights must sum to 1.0");
        let expected = 1.0 / n as f64;
        for &w in &weights {
            assert!(
                (w - expected).abs() < 1e-12,
                "equal weights should stay equal"
            );
        }
    }

    #[test]
    fn normalize_weights_one_dominant() {
        // Only particle 0 has weight; rest are zero.
        let n = 50;
        let mut weights = vec![0.0_f64; n];
        weights[0] = 1.0;
        normalize_weights(&mut weights);
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "weights must sum to 1.0 after normalization"
        );
        // Weight 0 should dominate (be much larger than the rest after +1e-300 floor).
        assert!(weights[0] > 0.5, "dominant weight must remain largest");
    }

    #[test]
    fn normalize_weights_no_nan_or_negative_after_all_zero_input() {
        let mut weights = vec![0.0_f64; 20];
        normalize_weights(&mut weights);
        for &w in &weights {
            assert!(!w.is_nan(), "no weight should be NaN");
            assert!(w >= 0.0, "no weight should be negative");
        }
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "weights must sum to 1.0");
    }

    // ── Invariant tests: effective_sample_size bounds ──────────────────────

    #[test]
    fn ess_in_valid_range_uniform() {
        let n = 100_usize;
        let weights = vec![1.0 / n as f64; n];
        let ess = effective_sample_size(&weights);
        assert!(ess >= 1.0, "ESS must be >= 1.0");
        assert!(ess <= n as f64 + 1.0, "ESS must be <= N (allowing epsilon)");
    }

    #[test]
    fn ess_in_valid_range_one_dominant() {
        let n = 100_usize;
        let mut weights = vec![0.0_f64; n];
        weights[0] = 1.0;
        let ess = effective_sample_size(&weights);
        // ESS ≈ 1 for degenerate distribution; epsilon pushes it slightly above 1.
        assert!(
            ess >= 1.0 - 1e-6,
            "ESS must be >= 1.0 for degenerate weights"
        );
        assert!(
            ess <= 2.0,
            "ESS should be near 1 for single-particle concentration"
        );
    }

    // ── Invariant tests: systematic_resample_6d particle count unchanged ──

    #[test]
    fn systematic_resample_preserves_particle_count() {
        for n in [1_usize, 5, 17, 100] {
            let mut particles = Array2::<f64>::zeros((n, 6));
            let mut weights = vec![1.0 / n as f64; n];
            let mut regimes = vec![0u8; n];
            systematic_resample_6d(&mut particles, &mut weights, &mut regimes);
            assert_eq!(
                particles.nrows(),
                n,
                "particle count must be unchanged after resample"
            );
            assert_eq!(
                regimes.len(),
                n,
                "regime count must be unchanged after resample"
            );
        }
    }
}
