//! CUSUM anomaly detection adapted for drone warfare.
//!
//! Uses Page's algorithm (the same as the original `cusum_test`) but with
//! domain-specific wrappers for jamming detection, threat shift detection,
//! and environmental change detection.

// ---------------------------------------------------------------------------
// Core CUSUM Algorithm
// ---------------------------------------------------------------------------

/// Page's CUSUM test for structural break detection.
///
/// Returns `(is_break, direction, cusum_value)` where:
/// - `direction`:  1 = positive break, -1 = negative break, 0 = none
/// - `cusum_value`: the accumulated CUSUM statistic
///
/// Direct port of the original `cusum_test`, without PyO3 wrappers.
pub fn cusum_test(values: &[f64], threshold_h: f64, min_samples: usize) -> (bool, i32, f64) {
    let n = values.len();

    if n < min_samples {
        return (false, 0, 0.0);
    }

    let half = n / 2;
    if half < 3 {
        return (false, 0, 0.0);
    }

    // Reference mean from first half.
    let mut mean = 0.0_f64;
    for v in values.iter().take(half) {
        mean += v;
    }
    mean /= half as f64;

    // Standard deviation.
    let mut var = 0.0_f64;
    for v in values.iter().take(half) {
        let diff = v - mean;
        var += diff * diff;
    }
    var /= (half - 1).max(1) as f64;
    let std = (var + 1e-12).sqrt();

    let h = threshold_h * std;
    let control_limit = 5.0 * std;

    // CUSUM accumulation on second half.
    let mut cusum_pos = 0.0_f64;
    let mut cusum_neg = 0.0_f64;

    for v in values.iter().skip(half) {
        cusum_pos = f64::max(0.0, cusum_pos + (v - mean - h));
        cusum_neg = f64::min(0.0, cusum_neg + (v - mean + h));

        if cusum_pos > control_limit {
            return (true, 1, cusum_pos);
        }
        if cusum_neg < -control_limit {
            return (true, -1, cusum_neg);
        }
    }

    if cusum_pos.abs() > cusum_neg.abs() {
        (false, 0, cusum_pos)
    } else {
        (false, 0, cusum_neg)
    }
}

// ---------------------------------------------------------------------------
// Domain-Specific CUSUM Wrappers
// ---------------------------------------------------------------------------

/// Configuration for the domain-specific CUSUM detectors.
#[derive(Debug, Clone)]
pub struct CusumConfig {
    /// CUSUM threshold multiplier (in units of std).
    pub threshold_h: f64,
    /// Minimum number of samples before CUSUM is active.
    pub min_samples: usize,
}

impl Default for CusumConfig {
    fn default() -> Self {
        Self {
            threshold_h: 0.5,
            min_samples: 10,
        }
    }
}

/// Detect GPS/radio jamming onset via CUSUM on signal-quality metrics.
///
/// `signal_metrics` is a time series of signal quality values (e.g. SNR,
/// carrier-to-noise).  A sudden drop indicates jamming.
///
/// Returns `(is_jamming, direction, cusum_value)`.
pub fn detect_jamming(signal_metrics: &[f64], config: &CusumConfig) -> (bool, i32, f64) {
    cusum_test(signal_metrics, config.threshold_h, config.min_samples)
}

/// Detect enemy regime change via CUSUM on threat-bearing angle.
///
/// `threat_bearings` is a time series of bearing angles (radians) from
/// our centroid to the threat.  A structural break means the enemy has
/// changed direction.
///
/// Returns `(is_shift, direction, cusum_value)`.
pub fn detect_threat_shift(threat_bearings: &[f64], config: &CusumConfig) -> (bool, i32, f64) {
    cusum_test(threat_bearings, config.threshold_h, config.min_samples)
}

/// Detect environmental changes (weather, urban canyon entry) via CUSUM
/// on sensor noise levels.
///
/// `noise_levels` is a time series of estimated sensor noise variance.
/// A break indicates the operating environment has changed significantly.
///
/// Returns `(is_env_shift, direction, cusum_value)`.
pub fn detect_environment_shift(noise_levels: &[f64], config: &CusumConfig) -> (bool, i32, f64) {
    cusum_test(noise_levels, config.threshold_h, config.min_samples)
}

// ---------------------------------------------------------------------------
// Multi-signal CUSUM monitor
// ---------------------------------------------------------------------------

/// Aggregated CUSUM result across multiple signal channels.
#[derive(Debug, Clone)]
pub struct CusumAlert {
    /// Which channel triggered.
    pub channel: String,
    /// Direction of the break.
    pub direction: i32,
    /// CUSUM statistic value.
    pub cusum_value: f64,
}

/// Run CUSUM on multiple named channels and return any that triggered.
///
/// This is a convenience function for systems that monitor several
/// signals simultaneously.
pub fn multi_channel_cusum(channels: &[(&str, &[f64])], config: &CusumConfig) -> Vec<CusumAlert> {
    let mut alerts = Vec::new();
    for &(name, values) in channels {
        let (triggered, direction, cusum_value) =
            cusum_test(values, config.threshold_h, config.min_samples);
        if triggered {
            alerts.push(CusumAlert {
                channel: name.to_string(),
                direction,
                cusum_value,
            });
        }
    }
    alerts
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cusum_no_break_on_constant() {
        let values = vec![1.0; 30];
        let (is_break, dir, _) = cusum_test(&values, 0.5, 10);
        assert!(!is_break);
        assert_eq!(dir, 0);
    }

    #[test]
    fn cusum_detects_positive_shift() {
        // First half ~ 0, second half ~ 100 → massive positive break.
        let mut values = vec![0.0; 20];
        values.extend(vec![100.0; 20]);
        let (is_break, dir, _) = cusum_test(&values, 0.5, 10);
        assert!(is_break);
        assert_eq!(dir, 1);
    }

    #[test]
    fn cusum_detects_negative_shift() {
        let mut values = vec![100.0; 20];
        values.extend(vec![0.0; 20]);
        let (is_break, dir, _) = cusum_test(&values, 0.5, 10);
        assert!(is_break);
        assert_eq!(dir, -1);
    }

    #[test]
    fn cusum_insufficient_samples() {
        let values = vec![1.0; 5];
        let (is_break, _, _) = cusum_test(&values, 0.5, 10);
        assert!(!is_break);
    }

    #[test]
    fn detect_jamming_wrapper() {
        let mut metrics = vec![30.0; 20]; // good SNR
        metrics.extend(vec![5.0; 20]); // jammed
        let (jammed, dir, _) = detect_jamming(&metrics, &CusumConfig::default());
        assert!(jammed);
        assert_eq!(dir, -1);
    }

    #[test]
    fn multi_channel_no_alerts_on_stable() {
        let stable = vec![1.0; 30];
        let channels: Vec<(&str, &[f64])> = vec![("gps_snr", &stable), ("baro_noise", &stable)];
        let alerts = multi_channel_cusum(&channels, &CusumConfig::default());
        assert!(alerts.is_empty());
    }

    #[test]
    fn multi_channel_catches_one() {
        let stable = vec![1.0; 30];
        let mut shifting = vec![1.0; 15];
        shifting.extend(vec![50.0; 15]);
        let channels: Vec<(&str, &[f64])> =
            vec![("gps_snr", &stable), ("threat_bearing", &shifting)];
        let alerts = multi_channel_cusum(&channels, &CusumConfig::default());
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].channel, "threat_bearing");
    }
}
