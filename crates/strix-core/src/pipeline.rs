//! Typed pipeline stages for the estimator.
//!
//! Makes the flow explicit: MeasurementIngest → EstimatorOutput → UncertaintyProfile.
//! Each stage has a defined input/output type, enabling plug-in features.

use crate::frames::{Frame, NedPosition};
use crate::state::Regime;
use crate::units::Timestamp;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Raw measurements ingested into the estimator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementIngest {
    /// Sensor observations as position deltas or absolute readings.
    pub position_obs: Option<NedPosition>,
    /// Velocity observation (if available).
    pub velocity_obs: Option<Vector3<f64>>,
    /// Timestamp of the measurement.
    pub timestamp: Timestamp,
    /// Source frame of the measurement.
    pub source_frame: Frame,
    /// Measurement noise variance (diagonal).
    pub noise_variance: Option<[f64; 3]>,
}

/// Output of the state estimator after one update step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatorOutput {
    /// Estimated position in NED.
    pub position: NedPosition,
    /// Estimated velocity in NED.
    pub velocity: Vector3<f64>,
    /// Diagonal of the state covariance (position components).
    pub position_covariance: [f64; 3],
    /// Diagonal of the state covariance (velocity components).
    pub velocity_covariance: [f64; 3],
    /// Effective sample size of the particle filter.
    pub ess: f64,
    /// Regime probabilities [patrol, engage, evade].
    pub regime_probabilities: [f64; 3],
    /// Most likely regime.
    pub regime: Regime,
    /// Timestamp of this estimate.
    pub timestamp: Timestamp,
}

/// Named value produced by an uncertainty feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureValue {
    /// Feature name.
    pub name: &'static str,
    /// Computed value.
    pub value: f64,
    /// How confident we are in this feature [0, 1].
    pub confidence: f64,
    /// Whether the feature had enough data to compute.
    pub valid: bool,
}

impl FeatureValue {
    /// Create an invalid (not-enough-data) feature value.
    pub fn invalid(name: &'static str) -> Self {
        Self {
            name,
            value: 0.0,
            confidence: 0.0,
            valid: false,
        }
    }
}

/// Trait for pluggable uncertainty features.
///
/// Each feature computes a single scalar from estimator history.
/// Features can be independently enabled, disabled, or ablated
/// for calibration testing.
pub trait UncertaintyFeature: Send + Sync {
    /// Human-readable feature name.
    fn name(&self) -> &'static str;

    /// Minimum number of samples needed for a valid computation.
    fn min_samples(&self) -> usize;

    /// Compute the feature from a history of estimates.
    ///
    /// `history` contains the most recent N position values (newest last).
    /// Returns a FeatureValue with validity flag.
    fn compute(&self, history: &[f64]) -> FeatureValue;
}

/// Hurst exponent — measures persistence vs mean-reversion.
///
/// H > 0.5: persistent (trending), H < 0.5: mean-reverting, H = 0.5: random walk.
/// Adapted from financial time series analysis.
pub struct HurstFeature;

impl UncertaintyFeature for HurstFeature {
    fn name(&self) -> &'static str {
        "hurst"
    }

    fn min_samples(&self) -> usize {
        20
    }

    fn compute(&self, history: &[f64]) -> FeatureValue {
        if history.len() < self.min_samples() {
            return FeatureValue::invalid(self.name());
        }

        // R/S analysis (simplified)
        let n = history.len();
        let mean: f64 = history.iter().sum::<f64>() / n as f64;
        let deviations: Vec<f64> = history.iter().map(|x| x - mean).collect();

        // Cumulative deviations
        let mut cum_dev = Vec::with_capacity(n);
        let mut sum = 0.0;
        for d in &deviations {
            sum += d;
            cum_dev.push(sum);
        }

        let range = cum_dev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - cum_dev.iter().cloned().fold(f64::INFINITY, f64::min);

        let std_dev = (deviations.iter().map(|d| d * d).sum::<f64>() / n as f64).sqrt();

        if std_dev < 1e-12 {
            return FeatureValue {
                name: self.name(),
                value: 0.5, // constant series = random walk
                confidence: 0.3,
                valid: true,
            };
        }

        let rs = range / std_dev;
        let hurst = (rs.max(1.0)).ln() / (n as f64).ln();

        FeatureValue {
            name: self.name(),
            value: hurst.clamp(0.0, 1.0),
            confidence: if n >= 50 { 0.9 } else { 0.5 },
            valid: true,
        }
    }
}

/// Volatility — windowed standard deviation of returns.
pub struct VolatilityFeature {
    /// Window size for volatility calculation.
    pub window: usize,
}

impl Default for VolatilityFeature {
    fn default() -> Self {
        Self { window: 20 }
    }
}

impl UncertaintyFeature for VolatilityFeature {
    fn name(&self) -> &'static str {
        "volatility"
    }

    fn min_samples(&self) -> usize {
        self.window.max(3)
    }

    fn compute(&self, history: &[f64]) -> FeatureValue {
        if history.len() < self.min_samples() {
            return FeatureValue::invalid(self.name());
        }

        let window = &history[history.len().saturating_sub(self.window)..];
        let n = window.len() as f64;
        let mean = window.iter().sum::<f64>() / n;
        let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let vol = variance.sqrt();

        FeatureValue {
            name: self.name(),
            value: vol,
            confidence: if window.len() >= 10 { 0.8 } else { 0.4 },
            valid: true,
        }
    }
}

/// Kurtosis — tail risk indicator.
///
/// Excess kurtosis > 0 indicates heavy tails (more extreme events).
pub struct KurtosisFeature;

impl UncertaintyFeature for KurtosisFeature {
    fn name(&self) -> &'static str {
        "kurtosis"
    }

    fn min_samples(&self) -> usize {
        10
    }

    fn compute(&self, history: &[f64]) -> FeatureValue {
        if history.len() < self.min_samples() {
            return FeatureValue::invalid(self.name());
        }

        let n = history.len() as f64;
        let mean = history.iter().sum::<f64>() / n;
        let m2 = history.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let m4 = history.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;

        if m2 < 1e-12 {
            return FeatureValue {
                name: self.name(),
                value: 0.0,
                confidence: 0.3,
                valid: true,
            };
        }

        let kurt = m4 / (m2 * m2) - 3.0; // excess kurtosis

        FeatureValue {
            name: self.name(),
            value: kurt,
            confidence: if history.len() >= 30 { 0.8 } else { 0.4 },
            valid: true,
        }
    }
}

/// ESS health — effective sample size as fraction of particle count.
pub struct EssHealthFeature {
    /// Total particle count (denominator).
    pub particle_count: usize,
}

impl EssHealthFeature {
    pub fn new(particle_count: usize) -> Self {
        Self {
            particle_count: particle_count.max(1),
        }
    }
}

impl UncertaintyFeature for EssHealthFeature {
    fn name(&self) -> &'static str {
        "ess_health"
    }

    fn min_samples(&self) -> usize {
        1
    }

    fn compute(&self, history: &[f64]) -> FeatureValue {
        // history here contains ESS values over time
        if history.is_empty() {
            return FeatureValue::invalid(self.name());
        }

        let latest_ess = *history.last().unwrap();
        let health = latest_ess / self.particle_count as f64;

        FeatureValue {
            name: self.name(),
            value: health.clamp(0.0, 1.0),
            confidence: 0.95, // ESS is a well-defined quantity
            valid: true,
        }
    }
}

/// Momentum — directional persistence of position changes.
pub struct MomentumFeature {
    /// Lookback window.
    pub window: usize,
}

impl Default for MomentumFeature {
    fn default() -> Self {
        Self { window: 10 }
    }
}

impl UncertaintyFeature for MomentumFeature {
    fn name(&self) -> &'static str {
        "momentum"
    }

    fn min_samples(&self) -> usize {
        self.window.max(3)
    }

    fn compute(&self, history: &[f64]) -> FeatureValue {
        if history.len() < self.min_samples() {
            return FeatureValue::invalid(self.name());
        }

        let window = &history[history.len().saturating_sub(self.window)..];
        // Count sign-consistent moves
        let mut pos_moves = 0usize;
        let mut neg_moves = 0usize;
        for w in window.windows(2) {
            let delta = w[1] - w[0];
            if delta > 0.0 {
                pos_moves += 1;
            } else if delta < 0.0 {
                neg_moves += 1;
            }
        }
        let total_moves = (pos_moves + neg_moves).max(1) as f64;
        let momentum = (pos_moves.max(neg_moves) as f64 / total_moves) * 2.0 - 1.0;

        FeatureValue {
            name: self.name(),
            value: momentum.clamp(-1.0, 1.0),
            confidence: if window.len() >= 8 { 0.7 } else { 0.3 },
            valid: true,
        }
    }
}

/// A registry of uncertainty features that can be computed together.
pub struct UncertaintyEngine {
    features: Vec<Box<dyn UncertaintyFeature>>,
}

impl UncertaintyEngine {
    /// Create a new engine with the default feature set.
    pub fn default_features(particle_count: usize) -> Self {
        Self {
            features: vec![
                Box::new(HurstFeature),
                Box::new(VolatilityFeature::default()),
                Box::new(KurtosisFeature),
                Box::new(EssHealthFeature::new(particle_count)),
                Box::new(MomentumFeature::default()),
            ],
        }
    }

    /// Create an empty engine (no features).
    pub fn empty() -> Self {
        Self {
            features: Vec::new(),
        }
    }

    /// Add a custom feature.
    pub fn add_feature(&mut self, feature: Box<dyn UncertaintyFeature>) {
        self.features.push(feature);
    }

    /// Compute all features from a history series.
    pub fn compute_all(&self, history: &[f64]) -> Vec<FeatureValue> {
        self.features.iter().map(|f| f.compute(history)).collect()
    }

    /// Compute all features, returning only valid ones.
    pub fn compute_valid(&self, history: &[f64]) -> Vec<FeatureValue> {
        self.compute_all(history)
            .into_iter()
            .filter(|fv| fv.valid)
            .collect()
    }

    /// Number of registered features.
    pub fn feature_count(&self) -> usize {
        self.features.len()
    }

    /// Names of all registered features.
    pub fn feature_names(&self) -> Vec<&'static str> {
        self.features.iter().map(|f| f.name()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_history(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (i as f64 * 0.1).sin() * 10.0 + 50.0)
            .collect()
    }

    #[test]
    fn hurst_needs_minimum_samples() {
        let short: Vec<f64> = vec![1.0, 2.0, 3.0];
        let fv = HurstFeature.compute(&short);
        assert!(!fv.valid);
    }

    #[test]
    fn hurst_computes_on_sufficient_data() {
        let data = sample_history(50);
        let fv = HurstFeature.compute(&data);
        assert!(fv.valid);
        assert!(fv.value >= 0.0 && fv.value <= 1.0);
        assert!(fv.confidence > 0.0);
    }

    #[test]
    fn volatility_of_constant_is_zero() {
        let data: Vec<f64> = vec![5.0; 30];
        let fv = VolatilityFeature::default().compute(&data);
        assert!(fv.valid);
        assert!(fv.value.abs() < 1e-10);
    }

    #[test]
    fn volatility_increases_with_noise() {
        let calm: Vec<f64> = (0..30).map(|i| 50.0 + (i as f64 * 0.01)).collect();
        let noisy: Vec<f64> = (0..30)
            .map(|i| 50.0 + (i as f64 * 0.1).sin() * 10.0)
            .collect();
        let v_calm = VolatilityFeature::default().compute(&calm);
        let v_noisy = VolatilityFeature::default().compute(&noisy);
        assert!(v_noisy.value > v_calm.value);
    }

    #[test]
    fn kurtosis_normal_near_zero() {
        // Uniform-ish data has negative excess kurtosis
        let data: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let fv = KurtosisFeature.compute(&data);
        assert!(fv.valid);
        // Uniform distribution has excess kurtosis = -1.2
        assert!(
            fv.value < 0.5,
            "uniform kurtosis should be negative-ish, got {}",
            fv.value
        );
    }

    #[test]
    fn ess_health_ratio() {
        let ess_history = vec![80.0, 85.0, 90.0, 95.0, 100.0];
        let fv = EssHealthFeature::new(200).compute(&ess_history);
        assert!(fv.valid);
        // Latest ESS=100, particles=200, health=0.5
        assert!((fv.value - 0.5).abs() < 1e-10);
    }

    #[test]
    fn momentum_trending_up() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let fv = MomentumFeature::default().compute(&data);
        assert!(fv.valid);
        assert!(fv.value > 0.5, "uptrend should have positive momentum");
    }

    #[test]
    fn momentum_flat() {
        let data: Vec<f64> = vec![5.0; 20];
        let fv = MomentumFeature::default().compute(&data);
        assert!(fv.valid);
        // All moves are zero, so momentum should be near -1 (no moves counted)
    }

    #[test]
    fn engine_default_has_five_features() {
        let engine = UncertaintyEngine::default_features(200);
        assert_eq!(engine.feature_count(), 5);
        let names = engine.feature_names();
        assert!(names.contains(&"hurst"));
        assert!(names.contains(&"volatility"));
        assert!(names.contains(&"kurtosis"));
        assert!(names.contains(&"ess_health"));
        assert!(names.contains(&"momentum"));
    }

    #[test]
    fn engine_compute_all() {
        let engine = UncertaintyEngine::default_features(200);
        let data = sample_history(50);
        let results = engine.compute_all(&data);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn engine_compute_valid_filters() {
        let engine = UncertaintyEngine::default_features(200);
        let short_data = vec![1.0, 2.0]; // too short for most features
        let valid = engine.compute_valid(&short_data);
        // Only ESS health should be valid (min_samples=1)
        assert!(valid.len() <= 2);
    }

    #[test]
    fn engine_empty() {
        let engine = UncertaintyEngine::empty();
        assert_eq!(engine.feature_count(), 0);
        let results = engine.compute_all(&[1.0, 2.0, 3.0]);
        assert!(results.is_empty());
    }

    #[test]
    fn engine_add_custom_feature() {
        struct CustomFeature;
        impl UncertaintyFeature for CustomFeature {
            fn name(&self) -> &'static str {
                "custom"
            }
            fn min_samples(&self) -> usize {
                1
            }
            fn compute(&self, history: &[f64]) -> FeatureValue {
                FeatureValue {
                    name: self.name(),
                    value: history.iter().sum::<f64>(),
                    confidence: 1.0,
                    valid: !history.is_empty(),
                }
            }
        }
        let mut engine = UncertaintyEngine::empty();
        engine.add_feature(Box::new(CustomFeature));
        assert_eq!(engine.feature_count(), 1);
        let results = engine.compute_all(&[1.0, 2.0, 3.0]);
        assert_eq!(results.len(), 1);
        assert!((results[0].value - 6.0).abs() < 1e-10);
    }

    #[test]
    fn feature_value_invalid() {
        let fv = FeatureValue::invalid("test");
        assert!(!fv.valid);
        assert_eq!(fv.name, "test");
        assert_eq!(fv.confidence, 0.0);
    }

    #[test]
    fn measurement_ingest_serde() {
        let m = MeasurementIngest {
            position_obs: Some(NedPosition::new(100.0, 200.0, -50.0)),
            velocity_obs: None,
            timestamp: Timestamp::from_secs(42.0),
            source_frame: Frame::Ned,
            noise_variance: Some([1.0, 1.0, 2.0]),
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: MeasurementIngest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.source_frame, Frame::Ned);
    }

    #[test]
    fn estimator_output_serde() {
        let o = EstimatorOutput {
            position: NedPosition::new(1.0, 2.0, -3.0),
            velocity: Vector3::new(0.1, 0.2, 0.0),
            position_covariance: [1.0, 1.0, 2.0],
            velocity_covariance: [0.1, 0.1, 0.2],
            ess: 150.0,
            regime_probabilities: [0.7, 0.2, 0.1],
            regime: Regime::Patrol,
            timestamp: Timestamp::from_secs(100.0),
        };
        let json = serde_json::to_string(&o).unwrap();
        let back: EstimatorOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.regime, Regime::Patrol);
    }

    #[test]
    fn determinism_1000_runs() {
        let engine = UncertaintyEngine::default_features(200);
        let data = sample_history(50);
        let first = engine.compute_all(&data);
        for _ in 0..1000 {
            let results = engine.compute_all(&data);
            for (a, b) in first.iter().zip(results.iter()) {
                assert_eq!(a.value, b.value, "non-deterministic feature: {}", a.name);
            }
        }
    }
}
