//! Battlespace regime detection — Markov regime transitions.
//!
//! Adapted from the trading `regime.rs` which uses RANGE / TREND / PANIC.
//! STRIX uses PATROL / ENGAGE / EVADE with a 3x3 Markov transition matrix
//! and automatic regime detection via CUSUM, Hurst and volatility signals.

use crate::state::Regime;

// ---------------------------------------------------------------------------
// Markov Regime Transition
// ---------------------------------------------------------------------------

/// Apply Markov regime transitions to a vector of regime indices.
///
/// For each particle, draw from the row of the transition matrix
/// corresponding to its current regime.
///
/// This is the direct analogue of `transition_regimes` from the
/// original trading filter, stripped of PyO3 wrappers.
pub fn transition_regimes(
    regimes: &mut [u8],
    transition_matrix: &[[f64; 3]; 3],
    random_uniform: &[f64],
) {
    let n = regimes.len();
    assert_eq!(n, random_uniform.len());

    for i in 0..n {
        let r = (regimes[i] as usize).min(2);
        let u = random_uniform[i];

        let mut cum_prob = 0.0;
        // Default to current regime (not EVADE) if floating-point cumsum
        // doesn't reach u due to rounding. Prevents bias toward EVADE.
        let mut new_regime = r as u8;

        for (j, &prob) in transition_matrix[r].iter().enumerate() {
            cum_prob += prob;
            if u < cum_prob {
                new_regime = j as u8;
                break;
            }
        }

        regimes[i] = new_regime;
    }
}

// ---------------------------------------------------------------------------
// Automatic Regime Detection
// ---------------------------------------------------------------------------

/// Signals used by the regime detector to classify the current tactical
/// situation.
#[derive(Debug, Clone)]
pub struct RegimeSignals {
    /// CUSUM test result for threat bearing shift (true = break detected).
    pub cusum_triggered: bool,
    /// CUSUM direction: 1 = positive, -1 = negative, 0 = none.
    pub cusum_direction: i32,
    /// Hurst exponent of recent threat movement.
    pub hurst: f64,
    /// Volatility compression ratio (short-term / long-term volatility).
    pub volatility_ratio: f64,
    /// Current threat distance to fleet centroid (meters).
    pub threat_distance: f64,
    /// Threat closing rate (m/s, negative = approaching).
    pub closing_rate: f64,
    /// Evade bias from antifragile kill zone data (0.0 = none, >0.5 = forced evade).
    pub evade_bias: f64,
}

/// Detect the most appropriate regime from tactical signals.
///
/// Decision logic:
/// 0. If evade_bias > 0.5 (antifragile kill zone) → EVADE
/// 1. If CUSUM triggered AND threat is closing → EVADE
/// 2. If threat within engagement range AND Hurst > 0.5 (systematic) → ENGAGE
///    unless time-to-contact < 5s (prefer EVADE)
/// 3. If volatility compressed (calm before storm) → stay in current, but flag
/// 4. Otherwise → PATROL
pub fn detect_regime(signals: &RegimeSignals, current: Regime, config: &DetectionConfig) -> Regime {
    // Priority 0: antifragile kill zone demands evasion.
    if signals.evade_bias > 0.5 {
        return Regime::Evade;
    }

    // Priority 1: imminent threat requiring evasion.
    if signals.cusum_triggered
        && signals.closing_rate < -config.closing_rate_threshold
        && signals.threat_distance < config.evade_distance
    {
        return Regime::Evade;
    }

    // Priority 2: engageable threat — systematic advance, within range.
    // But if time-to-contact < 5s, prefer EVADE over ENGAGE (tactical gap fix).
    if signals.threat_distance < config.engage_distance
        && signals.hurst > 0.5
        && signals.closing_rate < 0.0
    {
        let time_to_contact = signals.threat_distance / (-signals.closing_rate).max(1e-6);
        if time_to_contact < 5.0 {
            return Regime::Evade;
        }
        return Regime::Engage;
    }

    // Priority 3: CUSUM triggered but threat is far — shift to engage.
    if signals.cusum_triggered && signals.threat_distance < config.engage_distance * 1.5 {
        return Regime::Engage;
    }

    // Priority 4: volatility compression — pre-storm, hold current regime
    // but only if already engaged.
    if signals.volatility_ratio < 0.5 && current != Regime::Patrol {
        return current;
    }

    // Default: patrol.
    Regime::Patrol
}

/// Tuning parameters for automatic regime detection.
#[derive(Debug, Clone)]
pub struct DetectionConfig {
    /// Engagement distance threshold (meters).
    pub engage_distance: f64,
    /// Evasion distance threshold (meters).
    pub evade_distance: f64,
    /// Minimum closing rate (m/s) to trigger evasion.
    pub closing_rate_threshold: f64,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            engage_distance: 500.0,
            evade_distance: 150.0,
            closing_rate_threshold: 2.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Custom Transition Matrices for Tactical Situations
// ---------------------------------------------------------------------------

/// Generate a transition matrix biased toward engagement.
pub fn aggressive_transition_matrix() -> [[f64; 3]; 3] {
    [
        [0.70, 0.25, 0.05], // PATROL → high chance of ENGAGE
        [0.05, 0.90, 0.05], // ENGAGE → very sticky
        [0.10, 0.20, 0.70], // EVADE → moderate chance of re-engaging
    ]
}

/// Generate a transition matrix biased toward evasion/survival.
pub fn defensive_transition_matrix() -> [[f64; 3]; 3] {
    [
        [0.85, 0.05, 0.10], // PATROL → moderate chance of EVADE
        [0.15, 0.70, 0.15], // ENGAGE → likely to disengage
        [0.10, 0.05, 0.85], // EVADE → very sticky
    ]
}

/// Interpolate between two transition matrices with weight `alpha` in [0,1].
///
/// `alpha = 0` → matrix_a, `alpha = 1` → matrix_b.
pub fn blend_transition_matrices(
    a: &[[f64; 3]; 3],
    b: &[[f64; 3]; 3],
    alpha: f64,
) -> [[f64; 3]; 3] {
    let alpha = alpha.clamp(0.0, 1.0);
    let mut result = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i][j] = (1.0 - alpha) * a[i][j] + alpha * b[i][j];
        }
        // Renormalise row.
        let sum: f64 = result[i].iter().sum();
        if sum > 1e-12 {
            for val in &mut result[i] {
                *val /= sum;
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_preserves_length() {
        let mut regimes = vec![0u8, 1, 2, 0, 1];
        let tm = crate::state::default_transition_matrix();
        let ru = vec![0.1, 0.5, 0.9, 0.3, 0.8];
        transition_regimes(&mut regimes, &tm, &ru);
        assert_eq!(regimes.len(), 5);
        for &r in &regimes {
            assert!(r < 3);
        }
    }

    #[test]
    fn transition_low_uniform_stays_same() {
        // With u=0.0 and diagonal > 0.5, should stay in same regime.
        let mut regimes = vec![0u8; 10];
        let tm = crate::state::default_transition_matrix();
        let ru = vec![0.0; 10]; // very low → always picks first column
        transition_regimes(&mut regimes, &tm, &ru);
        // All should remain PATROL (0) since tm[0][0]=0.90 > 0.0.
        for &r in &regimes {
            assert_eq!(r, 0);
        }
    }

    #[test]
    fn detect_regime_evade_on_closing_threat() {
        let signals = RegimeSignals {
            cusum_triggered: true,
            cusum_direction: -1,
            hurst: 0.7,
            volatility_ratio: 1.0,
            threat_distance: 100.0,
            closing_rate: -5.0,
            evade_bias: 0.0,
        };
        let cfg = DetectionConfig::default();
        let result = detect_regime(&signals, Regime::Patrol, &cfg);
        assert_eq!(result, Regime::Evade);
    }

    #[test]
    fn detect_regime_engage_on_systematic_approach() {
        let signals = RegimeSignals {
            cusum_triggered: false,
            cusum_direction: 0,
            hurst: 0.7,
            volatility_ratio: 1.0,
            threat_distance: 300.0,
            closing_rate: -1.0,
            evade_bias: 0.0,
        };
        let cfg = DetectionConfig::default();
        let result = detect_regime(&signals, Regime::Patrol, &cfg);
        assert_eq!(result, Regime::Engage);
    }

    #[test]
    fn detect_regime_patrol_default() {
        let signals = RegimeSignals {
            cusum_triggered: false,
            cusum_direction: 0,
            hurst: 0.4,
            volatility_ratio: 1.0,
            threat_distance: 1000.0,
            closing_rate: 0.0,
            evade_bias: 0.0,
        };
        let cfg = DetectionConfig::default();
        let result = detect_regime(&signals, Regime::Patrol, &cfg);
        assert_eq!(result, Regime::Patrol);
    }

    #[test]
    fn blend_matrices_extremes() {
        let a = crate::state::default_transition_matrix();
        let b = aggressive_transition_matrix();

        let at_zero = blend_transition_matrices(&a, &b, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                assert!((at_zero[i][j] - a[i][j]).abs() < 1e-12);
            }
        }

        let at_one = blend_transition_matrices(&a, &b, 1.0);
        for i in 0..3 {
            for j in 0..3 {
                assert!((at_one[i][j] - b[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn blended_rows_sum_to_one() {
        let a = crate::state::default_transition_matrix();
        let b = defensive_transition_matrix();
        let blended = blend_transition_matrices(&a, &b, 0.3);
        for row in &blended {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12);
        }
    }
}
