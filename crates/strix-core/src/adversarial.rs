//! Swarm-level adversarial intent prediction.
//!
//! Instead of modeling individual enemy intent (DEFENDING/ATTACKING/RETREATING),
//! this module models aggregate swarm-level intent patterns.

use serde::{Deserialize, Serialize};

/// Swarm-level intent classification for adversary forces.
///
/// Goes beyond individual entity intent to capture coordinated behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SwarmIntent {
    /// Holding positions, not advancing.
    Defensive,
    /// Testing defenses, gathering intelligence.
    Probing,
    /// Concentrating forces — likely preparing attack.
    Massing,
    /// Attempting to surround or outflank.
    Flanking,
    /// Pulling back, disengaging.
    Withdrawing,
}

impl SwarmIntent {
    /// Threat level associated with this intent [0, 1].
    pub fn threat_level(&self) -> f64 {
        match self {
            Self::Withdrawing => 0.1,
            Self::Defensive => 0.2,
            Self::Probing => 0.4,
            Self::Flanking => 0.7,
            Self::Massing => 0.9,
        }
    }

    /// Whether this intent suggests imminent action.
    pub fn is_imminent(&self) -> bool {
        matches!(self, Self::Massing | Self::Flanking)
    }

    /// Suggested regime for our swarm in response.
    pub fn suggested_response(&self) -> &'static str {
        match self {
            Self::Withdrawing => "patrol",
            Self::Defensive => "patrol",
            Self::Probing => "engage",
            Self::Flanking => "evade",
            Self::Massing => "evade",
        }
    }
}

/// Observation of adversary swarm behavior for intent classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversaryObservation {
    /// Average inter-enemy distance (meters). Decreasing = massing.
    pub avg_spacing: f64,
    /// Centroid velocity magnitude (m/s). Positive toward us = advancing.
    pub approach_speed: f64,
    /// Spread of enemy positions (standard deviation in meters).
    pub formation_spread: f64,
    /// Angular spread around our position (radians, 0..2π).
    /// Large spread = flanking.
    pub angular_coverage: f64,
    /// Number of adversary entities observed.
    pub entity_count: usize,
    /// Timestamp of observation.
    pub timestamp: f64,
}

/// Classifier for swarm-level adversarial intent.
pub struct SwarmIntentClassifier {
    /// History of observations for trend detection.
    history: Vec<AdversaryObservation>,
    /// Maximum history length.
    max_history: usize,
    /// Current classification.
    current_intent: SwarmIntent,
    /// Confidence in current classification [0, 1].
    confidence: f64,
}

impl SwarmIntentClassifier {
    /// Create a new classifier.
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history: max_history.max(5),
            current_intent: SwarmIntent::Defensive,
            confidence: 0.0,
        }
    }

    /// Ingest a new observation and update classification.
    pub fn observe(&mut self, obs: AdversaryObservation) {
        self.history.push(obs);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
        self.classify();
    }

    /// Current intent classification.
    pub fn intent(&self) -> SwarmIntent {
        self.current_intent
    }

    /// Confidence in current classification.
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Classification with threat level.
    pub fn threat_assessment(&self) -> (SwarmIntent, f64, f64) {
        (
            self.current_intent,
            self.current_intent.threat_level(),
            self.confidence,
        )
    }

    fn classify(&mut self) {
        if self.history.len() < 2 {
            self.current_intent = SwarmIntent::Defensive;
            self.confidence = 0.1;
            return;
        }

        let latest = self.history.last().unwrap();
        let prev = &self.history[self.history.len() - 2];

        // Massing: spacing decreasing, entities stable or increasing
        let spacing_trend = latest.avg_spacing - prev.avg_spacing;
        let is_compacting = spacing_trend < -5.0;

        // Approaching: positive approach speed
        let is_approaching = latest.approach_speed > 2.0;

        // Flanking: high angular coverage
        let is_flanking = latest.angular_coverage > std::f64::consts::PI;

        // Withdrawing: negative approach speed, spacing increasing
        let is_withdrawing = latest.approach_speed < -2.0 && spacing_trend > 0.0;

        // Score each intent
        let mut scores = [0.0f64; 5]; // Defensive, Probing, Massing, Flanking, Withdrawing

        if is_withdrawing {
            scores[4] += 2.0;
        }

        if is_compacting && is_approaching {
            scores[2] += 3.0; // Massing
        } else if is_compacting {
            scores[2] += 1.5;
        }

        if is_flanking {
            scores[3] += 2.5; // Flanking
        }

        if is_approaching && !is_compacting && !is_flanking {
            scores[1] += 2.0; // Probing
        }

        if !is_approaching && !is_compacting && !is_flanking && !is_withdrawing {
            scores[0] += 1.5; // Defensive
        }

        // Find winner
        let intents = [
            SwarmIntent::Defensive,
            SwarmIntent::Probing,
            SwarmIntent::Massing,
            SwarmIntent::Flanking,
            SwarmIntent::Withdrawing,
        ];

        let max_score = scores.iter().cloned().fold(0.0_f64, f64::max);
        let total_score: f64 = scores.iter().sum::<f64>().max(0.01);

        for (i, &score) in scores.iter().enumerate() {
            if (score - max_score).abs() < 1e-10 {
                self.current_intent = intents[i];
                self.confidence = (score / total_score).clamp(0.0, 1.0);
                break;
            }
        }
    }

    /// History length.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_obs(
        avg_spacing: f64,
        approach_speed: f64,
        angular_coverage: f64,
    ) -> AdversaryObservation {
        AdversaryObservation {
            avg_spacing,
            approach_speed,
            formation_spread: 50.0,
            angular_coverage,
            entity_count: 10,
            timestamp: 0.0,
        }
    }

    #[test]
    fn intent_threat_levels() {
        let intents = [
            SwarmIntent::Defensive,
            SwarmIntent::Probing,
            SwarmIntent::Massing,
            SwarmIntent::Flanking,
            SwarmIntent::Withdrawing,
        ];
        for intent in &intents {
            let tl = intent.threat_level();
            assert!(
                (0.0..=1.0).contains(&tl),
                "{intent:?} threat level {tl} out of [0,1]"
            );
        }
    }

    #[test]
    fn imminent_intents() {
        assert!(SwarmIntent::Massing.is_imminent());
        assert!(SwarmIntent::Flanking.is_imminent());

        assert!(!SwarmIntent::Defensive.is_imminent());
        assert!(!SwarmIntent::Probing.is_imminent());
        assert!(!SwarmIntent::Withdrawing.is_imminent());
    }

    #[test]
    fn classifier_starts_defensive() {
        let clf = SwarmIntentClassifier::new(10);
        assert_eq!(clf.intent(), SwarmIntent::Defensive);
        assert_eq!(clf.history_len(), 0);
    }

    #[test]
    fn massing_detection() {
        let mut clf = SwarmIntentClassifier::new(10);
        // First observation: large spacing, no approach
        clf.observe(make_obs(200.0, 0.0, 1.0));
        // Second observation: compacting + approaching
        clf.observe(make_obs(180.0, 5.0, 1.0)); // spacing -20, approaching at 5 m/s
        assert_eq!(clf.intent(), SwarmIntent::Massing);
        assert!(clf.confidence() > 0.0);
    }

    #[test]
    fn flanking_detection() {
        let mut clf = SwarmIntentClassifier::new(10);
        // Large angular coverage — encirclement
        clf.observe(make_obs(100.0, 0.0, 3.5)); // angular > π
        clf.observe(make_obs(100.0, 0.0, 3.8)); // still flanking
        assert_eq!(clf.intent(), SwarmIntent::Flanking);
    }

    #[test]
    fn withdrawing_detection() {
        let mut clf = SwarmIntentClassifier::new(10);
        // Pulling back: negative approach speed, spacing increasing
        clf.observe(make_obs(100.0, 0.0, 1.0));
        clf.observe(make_obs(120.0, -5.0, 1.0)); // spacing +20, retreating
        assert_eq!(clf.intent(), SwarmIntent::Withdrawing);
    }

    #[test]
    fn probing_detection() {
        let mut clf = SwarmIntentClassifier::new(10);
        // Approaching but not compacting, not flanking
        clf.observe(make_obs(200.0, 0.0, 1.0));
        clf.observe(make_obs(202.0, 3.0, 1.0)); // slight spacing increase, approaching
        assert_eq!(clf.intent(), SwarmIntent::Probing);
    }

    #[test]
    fn history_bounded() {
        let max = 5;
        let mut clf = SwarmIntentClassifier::new(max);
        for i in 0..20 {
            clf.observe(make_obs(100.0 + i as f64, 0.0, 1.0));
        }
        assert_eq!(clf.history_len(), max);
    }

    #[test]
    fn serde_roundtrip() {
        // SwarmIntent
        let intent = SwarmIntent::Massing;
        let json = serde_json::to_string(&intent).expect("serialize intent");
        let back: SwarmIntent = serde_json::from_str(&json).expect("deserialize intent");
        assert_eq!(intent, back);

        // AdversaryObservation
        let obs = make_obs(150.0, 3.5, 2.0);
        let json = serde_json::to_string(&obs).expect("serialize obs");
        let back: AdversaryObservation = serde_json::from_str(&json).expect("deserialize obs");
        assert!((obs.avg_spacing - back.avg_spacing).abs() < 1e-12);
        assert!((obs.approach_speed - back.approach_speed).abs() < 1e-12);
    }
}
