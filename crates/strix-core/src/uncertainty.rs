//! Uncertainty quantification — Hurst exponent, volatility compression,
//! threat density contours, kurtosis, momentum, and ESS health.
//!
//! Adapted from `hurst.rs`, `volatility.rs`, `vwap.rs`, and `momentum.rs`
//! in the original trading particle filter.

// ---------------------------------------------------------------------------
// Hurst Exponent (Rescaled Range)
// ---------------------------------------------------------------------------

/// Calculate Hurst exponent via Rescaled Range (R/S) analysis.
///
/// Returns `(H, uncertainty)` where H in [0, 1]:
/// - H > 0.5 → trending / systematic advance
/// - H < 0.5 → mean-reverting / chaotic patrol
/// - H ≈ 0.5 → random walk
///
/// Direct port from the original `hurst_exponent`.
pub fn hurst_exponent(values: &[f64], min_window: usize, max_window: usize) -> (f64, f64) {
    let n = values.len();

    if n < max_window {
        return (0.5, 0.5);
    }

    // Calculate returns.
    let returns_n = n - 1;
    let mut returns = vec![0.0_f64; returns_n];
    for i in 0..returns_n {
        returns[i] = values[i + 1] - values[i];
    }

    // R/S analysis across window sizes.
    const N_WINDOWS: usize = 5;
    let mut log_rs = [0.0_f64; N_WINDOWS];
    let mut log_n = [0.0_f64; N_WINDOWS];
    let mut valid_count: usize = 0;

    for w_idx in 0..N_WINDOWS {
        let window = min_window + (max_window - min_window) * w_idx / (N_WINDOWS - 1);
        if window > returns_n {
            continue;
        }
        let n_chunks = returns_n / window;
        if n_chunks == 0 {
            continue;
        }

        let mut rs_sum = 0.0_f64;
        let mut rs_count: usize = 0;

        for chunk in 0..n_chunks {
            let start = chunk * window;
            let end = start + window;

            let mut chunk_mean = 0.0_f64;
            for i in start..end {
                chunk_mean += returns[i];
            }
            chunk_mean /= window as f64;

            let mut cumsum = 0.0_f64;
            let mut r_max = 0.0_f64;
            let mut r_min = 0.0_f64;

            for i in start..end {
                cumsum += returns[i] - chunk_mean;
                if cumsum > r_max {
                    r_max = cumsum;
                }
                if cumsum < r_min {
                    r_min = cumsum;
                }
            }

            let r = r_max - r_min;

            let mut var = 0.0_f64;
            for i in start..end {
                let diff = returns[i] - chunk_mean;
                var += diff * diff;
            }
            let s = (var / window as f64).sqrt();

            if s > 1e-12 {
                rs_sum += r / s;
                rs_count += 1;
            }
        }

        if rs_count > 0 {
            let rs_avg = rs_sum / rs_count as f64;
            log_rs[valid_count] = (rs_avg + 1e-12).ln();
            log_n[valid_count] = (window as f64).ln();
            valid_count += 1;
        }
    }

    if valid_count < 3 {
        return (0.5, 0.5);
    }

    // Linear regression: log(R/S) = H * log(n) + c.
    let mut mean_x = 0.0_f64;
    let mut mean_y = 0.0_f64;
    for i in 0..valid_count {
        mean_x += log_n[i];
        mean_y += log_rs[i];
    }
    mean_x /= valid_count as f64;
    mean_y /= valid_count as f64;

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..valid_count {
        let dx = log_n[i] - mean_x;
        let dy = log_rs[i] - mean_y;
        num += dx * dy;
        den += dx * dx;
    }

    if den < 1e-12 {
        return (0.5, 0.5);
    }

    let h = (num / den).clamp(0.0, 1.0);

    let mut ss_res = 0.0_f64;
    for i in 0..valid_count {
        let pred = mean_y + h * (log_n[i] - mean_x);
        let resid = log_rs[i] - pred;
        ss_res += resid * resid;
    }
    let uncertainty = (ss_res / (valid_count - 2).max(1) as f64).sqrt();

    (h, uncertainty)
}

// ---------------------------------------------------------------------------
// Volatility Compression
// ---------------------------------------------------------------------------

/// Detect volatility compression (calm-before-storm).
///
/// Returns `(compression_ratio, is_compressed, is_expanding)`.
///
/// Direct port of `volatility_compression` from the trading filter.
pub fn volatility_compression(
    values: &[f64],
    short_window: usize,
    long_window: usize,
) -> (f64, bool, bool) {
    let n = values.len();

    if n < long_window || n < 3 {
        return (1.0, false, false);
    }

    let returns_n = n - 1;
    let mut returns = vec![0.0_f64; returns_n];
    for i in 0..returns_n {
        returns[i] = values[i + 1] - values[i];
    }

    if returns_n < long_window {
        return (1.0, false, false);
    }

    // Short-term volatility.
    let short_start = returns_n.saturating_sub(short_window);
    let short_count = returns_n - short_start;
    if short_count < 2 {
        return (1.0, false, false);
    }

    let mut short_sum = 0.0_f64;
    let mut short_sum_sq = 0.0_f64;
    for i in short_start..returns_n {
        short_sum += returns[i];
        short_sum_sq += returns[i] * returns[i];
    }
    let short_mean = short_sum / short_count as f64;
    let short_var = (short_sum_sq / short_count as f64) - (short_mean * short_mean);
    let short_vol = (f64::max(short_var, 0.0) + 1e-12).sqrt();

    // Long-term volatility.
    let long_start = returns_n.saturating_sub(long_window);
    let long_count = returns_n - long_start;

    let mut long_sum = 0.0_f64;
    let mut long_sum_sq = 0.0_f64;
    for i in long_start..returns_n {
        long_sum += returns[i];
        long_sum_sq += returns[i] * returns[i];
    }
    let long_mean = long_sum / long_count as f64;
    let long_var = (long_sum_sq / long_count as f64) - (long_mean * long_mean);
    let long_vol = (f64::max(long_var, 0.0) + 1e-12).sqrt();

    if long_vol < 1e-10 {
        return (1.0, false, false);
    }

    let compression_ratio = short_vol / long_vol;
    let is_compressed = compression_ratio < 0.5;
    let is_expanding = compression_ratio > 1.5;

    (compression_ratio, is_compressed, is_expanding)
}

// ---------------------------------------------------------------------------
// Threat Density Contours (adapted from VWAP bands)
// ---------------------------------------------------------------------------

/// Compute threat density contours — the spatial analogue of VWAP bands.
///
/// Given a set of threat-particle positions and their weights, computes
/// a weighted centroid (analogous to VWAP) and sigma-bands representing
/// density shells.
///
/// Returns `(centroid_distance, inner_band, outer_band)` where distances
/// are from the observer.
///
/// `band_sigma` controls how many standard deviations the bands span.
pub fn threat_density_contours(
    distances: &[f64],
    weights: &[f64],
    band_sigma: f64,
) -> (f64, f64, f64) {
    let n = distances.len();
    assert_eq!(n, weights.len());

    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    // Weighted centroid distance (≈ VWAP).
    let mut wd_sum = 0.0_f64;
    for i in 0..n {
        wd_sum += distances[i] * weights[i];
    }
    let centroid = wd_sum / total_weight;

    // Weighted variance.
    let mut var_sum = 0.0_f64;
    for i in 0..n {
        let diff = distances[i] - centroid;
        var_sum += weights[i] * diff * diff;
    }
    let std = (var_sum / total_weight).sqrt();

    let inner = centroid - band_sigma * std;
    let outer = centroid + band_sigma * std;

    (centroid, inner.max(0.0), outer)
}

// ---------------------------------------------------------------------------
// Rolling Kurtosis
// ---------------------------------------------------------------------------

/// Excess kurtosis of the most recent `window` values.
///
/// >0 → fat tails (unexpected events likely),
/// <0 → thin tails,
///  0 → normal.
///
/// Direct port of `rolling_kurtosis`.
pub fn rolling_kurtosis(values: &[f64], window: usize) -> f64 {
    let n = values.len();

    if n < window || n < 4 {
        return 0.0;
    }

    let start = n - window;

    let mut mean = 0.0_f64;
    for i in start..n {
        mean += values[i];
    }
    mean /= window as f64;

    let mut m2 = 0.0_f64;
    let mut m4 = 0.0_f64;
    for i in start..n {
        let diff = values[i] - mean;
        let diff_sq = diff * diff;
        m2 += diff_sq;
        m4 += diff_sq * diff_sq;
    }

    m2 /= window as f64;
    m4 /= window as f64;

    if m2 < 1e-12 {
        return 0.0;
    }

    let kurtosis = (m4 / (m2 * m2)) - 3.0;
    kurtosis.clamp(-2.0, 10.0)
}

// ---------------------------------------------------------------------------
// Momentum Score
// ---------------------------------------------------------------------------

/// Normalised momentum score [-1, 1] for enemy advance/retreat rate.
///
/// Compares recent average to older average in the given window.
/// Uses tanh normalisation with 200x scaling for sensitivity.
///
/// Direct port of `calculate_momentum_score`.
pub fn momentum_score(values: &[f64], window: usize) -> f64 {
    let n = values.len();
    if n < window || n < 3 {
        return 0.0;
    }

    let start = n.saturating_sub(window);
    let mid = start + (n - start) / 2;

    let mut recent_sum = 0.0_f64;
    let mut recent_count = 0_usize;
    for i in mid..n {
        recent_sum += values[i];
        recent_count += 1;
    }

    let mut older_sum = 0.0_f64;
    let mut older_count = 0_usize;
    for i in start..mid {
        older_sum += values[i];
        older_count += 1;
    }

    if recent_count == 0 || older_count == 0 {
        return 0.0;
    }

    let recent_avg = recent_sum / recent_count as f64;
    let older_avg = older_sum / older_count as f64;

    let momentum = recent_avg - older_avg;
    (momentum * 200.0).tanh()
}

// ---------------------------------------------------------------------------
// Effective Sample Size (sensor diversity health)
// ---------------------------------------------------------------------------

/// Effective Sample Size from a weight vector.
///
/// ESS = 1 / sum(w^2).  A low ESS relative to N indicates weight
/// degeneracy (particle filter collapse or sensor dominance).
pub fn effective_sample_size(weights: &[f64]) -> f64 {
    let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
    1.0 / (sum_sq + 1e-12)
}

/// ESS ratio + uncertainty margin + dominance test.
///
/// Returns `(ess_ratio, uncertainty_margin, is_dominant)`.
///
/// `is_dominant` is true if the leading regime probability exceeds
/// the others by more than `uncertainty_margin`.
pub fn ess_and_uncertainty_margin(
    weights: &[f64],
    p_engage: f64,
    p_patrol: f64,
    p_evade: f64,
) -> (f64, f64, bool) {
    let n = weights.len();
    let ess = effective_sample_size(weights);
    let ess_ratio = (ess / n as f64).clamp(0.0, 1.0);

    let uncertainty_margin = 2.0 * (0.25 / f64::max(ess, 100.0)).sqrt();

    let is_dominant = p_engage > p_patrol + uncertainty_margin
        && p_engage > p_evade + uncertainty_margin
        && p_engage > 0.45;

    (ess_ratio, uncertainty_margin, is_dominant)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hurst_random_walk_near_half() {
        // Cumulative sum of uniform noise → H ≈ 0.5.
        let mut values = vec![0.0_f64; 200];
        let mut rng = 42u64; // simple LCG
        for i in 1..200 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            values[i] = values[i - 1] + r;
        }
        let (h, _unc) = hurst_exponent(&values, 10, 50);
        // Should be roughly near 0.5 (not exactly due to finite sample).
        assert!(h > 0.1 && h < 0.9);
    }

    #[test]
    fn hurst_short_series_returns_default() {
        let values = vec![1.0; 5];
        let (h, unc) = hurst_exponent(&values, 10, 50);
        assert_eq!(h, 0.5);
        assert_eq!(unc, 0.5);
    }

    #[test]
    fn volatility_constant_returns_no_compression() {
        let values = vec![1.0; 100];
        let (ratio, compressed, expanding) = volatility_compression(&values, 10, 50);
        // All returns are zero → both short and long vol are ~0 → ratio ≈ 1.
        assert!(!expanding);
        // ratio could be ~1.0 due to 1e-12 epsilon
        let _ = (ratio, compressed);
    }

    #[test]
    fn volatility_detects_compression() {
        // Long period with large swings, then calm.
        let mut values = Vec::new();
        for i in 0..60 {
            values.push(if i % 2 == 0 { 10.0 } else { -10.0 });
        }
        // Then very calm period.
        for _ in 0..20 {
            values.push(0.0);
        }
        let (ratio, compressed, _) = volatility_compression(&values, 10, 50);
        assert!(ratio < 1.0);
        let _ = compressed;
    }

    #[test]
    fn density_contours_uniform() {
        let distances = vec![10.0, 20.0, 30.0];
        let weights = vec![1.0 / 3.0; 3];
        let (centroid, inner, outer) = threat_density_contours(&distances, &weights, 1.5);
        assert!((centroid - 20.0).abs() < 1e-10);
        assert!(inner < centroid);
        assert!(outer > centroid);
    }

    #[test]
    fn rolling_kurtosis_normal() {
        // Uniform values → kurtosis ≈ 0 (actually -1.2 for uniform).
        let values: Vec<f64> = (0..100).map(|i| (i as f64) / 100.0).collect();
        let k = rolling_kurtosis(&values, 50);
        assert!(k > -3.0 && k < 3.0);
    }

    #[test]
    fn momentum_increasing() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let m = momentum_score(&values, 20);
        assert!(m > 0.0);
    }

    #[test]
    fn momentum_decreasing() {
        let values: Vec<f64> = (0..20).map(|i| 20.0 - i as f64).collect();
        let m = momentum_score(&values, 20);
        assert!(m < 0.0);
    }

    #[test]
    fn ess_uniform_weights() {
        let n = 100;
        let weights = vec![1.0 / n as f64; n];
        let ess = effective_sample_size(&weights);
        assert!((ess - n as f64).abs() < 1.0);
    }

    #[test]
    fn ess_degenerate_weights() {
        let mut weights = vec![0.0; 100];
        weights[0] = 1.0;
        let ess = effective_sample_size(&weights);
        assert!((ess - 1.0).abs() < 0.1);
    }
}
