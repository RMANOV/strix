//! Threat intent detection pipeline — from signals to assessed intent.
//!
//! 3-layer pipeline that fuses multiple indicators into a continuous
//! intent score, replacing discrete regime thresholds with graduated
//! tactical assessment:
//!
//! ```text
//! Layer 1: Hurst persistence     → purposeful trajectory? [H > 0.55]
//! Layer 2: Closing acceleration   → accelerating toward us?
//! Layer 3: Volatility compression → formation tightening?
//!                    ↓
//!          Confidence-weighted fusion
//!                    ↓
//!          IntentScore ∈ [-1, 1] + IntentClass
//! ```
//!
//! The key innovation: intent feeds back into the auction urgency
//! multiplier, creating a market-driven tactical response that doesn't
//! require hard regime transitions.

use serde::{Deserialize, Serialize};

/// Configuration for the intent detection pipeline.
#[derive(Debug, Clone)]
pub struct IntentConfig {
    /// Hurst threshold above which trajectory is considered purposeful.
    pub hurst_purposeful: f64,
    /// Hurst threshold below which trajectory is considered retreating.
    pub hurst_retreating: f64,
    /// Closing acceleration threshold (m/s^2) for attack signal.
    pub closing_accel_threshold: f64,
    /// Volatility compression ratio below which formation is tightening.
    pub vol_compression_threshold: f64,

    // Layer weights for fusion (must sum to 1.0).
    /// Weight for Hurst persistence layer.
    pub w_hurst: f64,
    /// Weight for closing acceleration layer.
    pub w_closing: f64,
    /// Weight for volatility compression layer.
    pub w_volatility: f64,

    /// Optional coherence multiplier weight (0 to disable).
    pub w_coherence: f64,

    /// Intent score threshold for attack classification.
    pub attack_threshold: f64,
    /// Intent score threshold for retreat classification (negative).
    pub retreat_threshold: f64,
}

impl Default for IntentConfig {
    fn default() -> Self {
        Self {
            hurst_purposeful: 0.55,
            hurst_retreating: 0.45,
            closing_accel_threshold: 0.5,
            vol_compression_threshold: 0.5,
            w_hurst: 0.4,
            w_closing: 0.35,
            w_volatility: 0.25,
            w_coherence: 0.0, // disabled by default, activate when fleet_metrics available
            attack_threshold: 0.3,
            retreat_threshold: -0.3,
        }
    }
}

/// Raw signals fed into the intent pipeline (one per drone-threat pair).
#[derive(Debug, Clone)]
pub struct IntentSignals {
    /// Hurst exponent of threat distance series.
    pub hurst: f64,
    /// Hurst estimation uncertainty.
    pub hurst_uncertainty: f64,
    /// Closing rate (m/s, negative = approaching).
    pub closing_rate: f64,
    /// Closing acceleration (m/s^2, negative = accelerating toward us).
    pub closing_acceleration: f64,
    /// Volatility compression ratio (short/long vol).
    pub volatility_ratio: f64,
    /// Threat distance (meters).
    pub threat_distance: f64,
    /// Optional: fleet velocity coherence [0, 1].
    pub fleet_coherence: Option<f64>,
}

/// Classified threat intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntentClass {
    /// Threat is actively attacking — closing with purpose.
    Attacking,
    /// Threat behavior is ambiguous or neutral.
    Neutral,
    /// Threat is withdrawing or disengaging.
    Retreating,
    /// Insufficient data for classification.
    Unknown,
}

/// Result of the intent detection pipeline.
#[derive(Debug, Clone)]
pub struct IntentAssessment {
    /// Continuous intent score in [-1, 1].
    /// Positive = attacking, negative = retreating.
    pub score: f64,
    /// Classified intent.
    pub class: IntentClass,
    /// Per-layer confidence scores [0, 1].
    pub layer_scores: LayerScores,
    /// Overall confidence in the assessment [0, 1].
    pub confidence: f64,
}

/// Individual layer outputs before fusion.
#[derive(Debug, Clone)]
pub struct LayerScores {
    /// Hurst persistence score [-1, 1].
    pub hurst: f64,
    /// Closing acceleration score [-1, 1].
    pub closing: f64,
    /// Volatility compression score [-1, 1].
    pub volatility: f64,
    /// Coherence modifier [0, 1] (1.0 if disabled).
    pub coherence: f64,
}

/// Run the 3-layer intent detection pipeline.
///
/// Returns an `IntentAssessment` with a continuous score and classification.
pub fn detect_intent(signals: &IntentSignals, config: &IntentConfig) -> IntentAssessment {
    // ── Layer 1: Hurst persistence ──────────────────────────────────
    // H > purposeful → systematic advance (positive contribution)
    // H < retreating → mean-reverting / retreating (negative)
    // H ≈ 0.5 → random walk (neutral)
    let hurst_score = if signals.hurst > config.hurst_purposeful {
        // Scale [purposeful, 1.0] → [0, 1]
        let range = 1.0 - config.hurst_purposeful;
        let raw = (signals.hurst - config.hurst_purposeful) / range.max(0.01);
        // Only positive if also closing
        if signals.closing_rate < 0.0 {
            raw.clamp(0.0, 1.0)
        } else {
            -raw.clamp(0.0, 1.0) * 0.5 // purposeful but retreating
        }
    } else if signals.hurst < config.hurst_retreating {
        // Mean-reverting → not purposefully advancing
        let range = config.hurst_retreating;
        let raw = (config.hurst_retreating - signals.hurst) / range.max(0.01);
        -raw.clamp(0.0, 1.0)
    } else {
        0.0 // ambiguous
    };

    // Confidence decreases with Hurst uncertainty.
    let hurst_confidence = (1.0 - signals.hurst_uncertainty.min(1.0)).max(0.1);

    // ── Layer 2: Closing acceleration ───────────────────────────────
    // Negative closing_acceleration = accelerating toward us = attacking
    let closing_score = if signals.closing_acceleration < -config.closing_accel_threshold {
        // Attacking: scale by magnitude
        let magnitude = (-signals.closing_acceleration - config.closing_accel_threshold).min(5.0);
        (magnitude / 5.0).clamp(0.0, 1.0)
    } else if signals.closing_acceleration > config.closing_accel_threshold {
        // Decelerating / pulling away
        let magnitude = (signals.closing_acceleration - config.closing_accel_threshold).min(5.0);
        -(magnitude / 5.0).clamp(0.0, 1.0)
    } else {
        // Near-zero acceleration
        0.0
    };

    // Also factor in raw closing rate for baseline signal.
    let closing_rate_contribution = if signals.closing_rate < -1.0 {
        0.3_f64.min((-signals.closing_rate - 1.0) / 10.0)
    } else if signals.closing_rate > 1.0 {
        -0.3_f64.min((signals.closing_rate - 1.0) / 10.0)
    } else {
        0.0
    };

    let combined_closing = (closing_score + closing_rate_contribution).clamp(-1.0, 1.0);

    // ── Layer 3: Volatility compression ─────────────────────────────
    // Low vol ratio = calm before storm = formation tightening = attack prep
    let vol_score = if signals.volatility_ratio < config.vol_compression_threshold {
        // Compressed — attacking signal
        let raw = (config.vol_compression_threshold - signals.volatility_ratio)
            / config.vol_compression_threshold.max(0.01);
        raw.clamp(0.0, 1.0)
    } else if signals.volatility_ratio > 1.5 {
        // Expanding volatility — disorganized, possibly retreating
        let raw = (signals.volatility_ratio - 1.5).min(1.0);
        -raw * 0.5
    } else {
        0.0
    };

    // ── Coherence modifier (optional Layer 4) ───────────────────────
    let coherence_mod = signals.fleet_coherence.unwrap_or(1.0);

    // ── Fusion ──────────────────────────────────────────────────────
    let w_total = config.w_hurst + config.w_closing + config.w_volatility;
    let norm = if w_total > 1e-9 { w_total } else { 1.0 };

    let raw_score = (config.w_hurst * hurst_score * hurst_confidence
        + config.w_closing * combined_closing
        + config.w_volatility * vol_score)
        / norm;

    // Apply coherence as a confidence multiplier (high coherence = more confident).
    let coherence_factor = if config.w_coherence > 0.0 {
        1.0 - config.w_coherence * (1.0 - coherence_mod)
    } else {
        1.0
    };

    let score = (raw_score * coherence_factor).clamp(-1.0, 1.0);

    // ── Classification ──────────────────────────────────────────────
    let class = if signals.hurst_uncertainty > 0.4 && signals.threat_distance > 1000.0 {
        IntentClass::Unknown
    } else if score > config.attack_threshold {
        IntentClass::Attacking
    } else if score < config.retreat_threshold {
        IntentClass::Retreating
    } else {
        IntentClass::Neutral
    };

    // Overall confidence: weighted average of layer confidences.
    let confidence = (hurst_confidence * 0.5
        + (1.0 - (combined_closing.abs() - 0.5).abs().min(0.5)) * 0.3
        + coherence_mod * 0.2)
        .clamp(0.0, 1.0);

    IntentAssessment {
        score,
        class,
        layer_scores: LayerScores {
            hurst: hurst_score,
            closing: combined_closing,
            volatility: vol_score,
            coherence: coherence_mod,
        },
        confidence,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_signals() -> IntentSignals {
        IntentSignals {
            hurst: 0.5,
            hurst_uncertainty: 0.2,
            closing_rate: 0.0,
            closing_acceleration: 0.0,
            volatility_ratio: 1.0,
            threat_distance: 300.0,
            fleet_coherence: None,
        }
    }

    #[test]
    fn attack_trajectory_detected() {
        let signals = IntentSignals {
            hurst: 0.75,                // purposeful
            hurst_uncertainty: 0.1,     // confident
            closing_rate: -8.0,         // fast approach
            closing_acceleration: -2.0, // accelerating toward
            volatility_ratio: 0.3,      // compressed
            threat_distance: 200.0,
            fleet_coherence: None,
        };
        let result = detect_intent(&signals, &IntentConfig::default());
        assert_eq!(result.class, IntentClass::Attacking);
        assert!(result.score > 0.3, "score = {}", result.score);
    }

    #[test]
    fn retreat_trajectory_detected() {
        let signals = IntentSignals {
            hurst: 0.35, // mean-reverting
            hurst_uncertainty: 0.15,
            closing_rate: 5.0,         // moving away
            closing_acceleration: 2.0, // accelerating away
            volatility_ratio: 2.0,     // expanding
            threat_distance: 500.0,
            fleet_coherence: None,
        };
        let result = detect_intent(&signals, &IntentConfig::default());
        assert_eq!(result.class, IntentClass::Retreating);
        assert!(result.score < -0.3, "score = {}", result.score);
    }

    #[test]
    fn neutral_trajectory() {
        let signals = default_signals();
        let result = detect_intent(&signals, &IntentConfig::default());
        assert_eq!(result.class, IntentClass::Neutral);
        assert!(result.score.abs() < 0.3, "score = {}", result.score);
    }

    #[test]
    fn unknown_when_uncertain_and_far() {
        let signals = IntentSignals {
            hurst_uncertainty: 0.5,  // high uncertainty
            threat_distance: 2000.0, // very far
            ..default_signals()
        };
        let result = detect_intent(&signals, &IntentConfig::default());
        assert_eq!(result.class, IntentClass::Unknown);
    }

    #[test]
    fn coherence_modulates_confidence() {
        let config = IntentConfig {
            w_coherence: 0.3,
            ..IntentConfig::default()
        };
        let attack_signals = IntentSignals {
            hurst: 0.75,
            hurst_uncertainty: 0.1,
            closing_rate: -8.0,
            closing_acceleration: -2.0,
            volatility_ratio: 0.3,
            threat_distance: 200.0,
            fleet_coherence: Some(0.9),
        };

        let high_coh = detect_intent(&attack_signals, &config);

        let low_coh_signals = IntentSignals {
            fleet_coherence: Some(0.1),
            ..attack_signals
        };
        let low_coh = detect_intent(&low_coh_signals, &config);

        assert!(
            high_coh.score.abs() > low_coh.score.abs(),
            "high coherence should amplify signal: {:.3} vs {:.3}",
            high_coh.score,
            low_coh.score
        );
    }

    #[test]
    fn purposeful_but_retreating() {
        // High Hurst (purposeful) but moving away — should not be Attacking.
        let signals = IntentSignals {
            hurst: 0.8,
            hurst_uncertainty: 0.1,
            closing_rate: 5.0, // moving away
            closing_acceleration: 1.0,
            volatility_ratio: 1.0,
            threat_distance: 300.0,
            fleet_coherence: None,
        };
        let result = detect_intent(&signals, &IntentConfig::default());
        assert_ne!(result.class, IntentClass::Attacking);
    }

    #[test]
    fn auction_urgency_boost() {
        // Verify that high intent score produces a multiplier > 1.
        let signals = IntentSignals {
            hurst: 0.75,
            hurst_uncertainty: 0.1,
            closing_rate: -8.0,
            closing_acceleration: -2.0,
            volatility_ratio: 0.3,
            threat_distance: 200.0,
            fleet_coherence: None,
        };
        let result = detect_intent(&signals, &IntentConfig::default());
        let urgency_multiplier = 1.0 + result.score.max(0.0);
        assert!(
            urgency_multiplier > 1.0,
            "attacking intent should boost urgency: {urgency_multiplier}"
        );
    }
}
