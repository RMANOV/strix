//! Side-effect-free defensive policy kernel.
//!
//! The kernel returns recommendations only. It is not called by the runtime
//! pipeline, owns no actuator, and performs no persistence or network access.

use std::{collections::HashMap, fmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NominalMode {
    Observe,
    Relay,
    Reposition,
    Recover,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefensiveAction {
    CollisionBreak,
    EmissionControl,
    Engage,
    Evade,
    Nominal(NominalMode),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionTier {
    Collision,
    Exposure,
    Threat,
    Throughput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReason {
    CollisionLimit,
    ExposureLimit,
    ThreatGatesPassed,
    SensorFallback,
    EngagementExposure,
    EngagementGateFailed,
    HysteresisHold,
    HighestUtility,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DefensiveDecision {
    pub action: DefensiveAction,
    pub tier: DecisionTier,
    pub reason: DecisionReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorFallback {
    EmissionControl,
    Evade,
}

impl SensorFallback {
    fn action(self) -> DefensiveAction {
        match self {
            Self::EmissionControl => DefensiveAction::EmissionControl,
            Self::Evade => DefensiveAction::Evade,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensorGate {
    pub sensor_id: String,
    pub minimum_predictability: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyGenome {
    pub collision_break_threshold: f64,
    pub emission_control_threshold: f64,
    pub threat_response_threshold: f64,
    pub engagement_exposure_limit: f64,
    pub minimum_energy_reserve: f64,
    pub throughput_weight: f64,
    pub safety_weight: f64,
    pub energy_weight: f64,
    pub switch_margin: f64,
    pub sensor_gates: Vec<SensorGate>,
    pub unpredictable_fallback: SensorFallback,
    pub allow_deceptive_actions: bool,
    pub allow_online_mutation: bool,
}

impl Default for PolicyGenome {
    fn default() -> Self {
        Self {
            collision_break_threshold: 0.8,
            emission_control_threshold: 0.8,
            threat_response_threshold: 0.65,
            engagement_exposure_limit: 0.4,
            minimum_energy_reserve: 0.3,
            throughput_weight: 0.45,
            safety_weight: 0.4,
            energy_weight: 0.15,
            switch_margin: 0.08,
            sensor_gates: vec![SensorGate {
                sensor_id: "primary".into(),
                minimum_predictability: 0.6,
            }],
            unpredictable_fallback: SensorFallback::EmissionControl,
            allow_deceptive_actions: false,
            allow_online_mutation: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyError {
    NonFiniteInput,
    OutOfRangeInput,
    EmptySensorId,
    DuplicateSensorGate,
    DuplicateSensorPrediction,
    MissingSensorGates,
    EmptyNominalModes,
    DuplicateNominalMode,
    UnsafeConfiguration,
    InvalidWeights,
}

impl fmt::Display for PolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "defensive policy error: {self:?}")
    }
}

impl std::error::Error for PolicyError {}

fn validate_unit(value: f64) -> Result<(), PolicyError> {
    if !value.is_finite() {
        return Err(PolicyError::NonFiniteInput);
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(PolicyError::OutOfRangeInput);
    }
    Ok(())
}

impl PolicyGenome {
    pub fn validate(&self) -> Result<(), PolicyError> {
        for value in [
            self.collision_break_threshold,
            self.emission_control_threshold,
            self.threat_response_threshold,
            self.engagement_exposure_limit,
            self.minimum_energy_reserve,
            self.switch_margin,
        ] {
            validate_unit(value)?;
        }
        let weights = [
            self.throughput_weight,
            self.safety_weight,
            self.energy_weight,
        ];
        if weights
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            || weights.iter().sum::<f64>() <= 0.0
        {
            return Err(PolicyError::InvalidWeights);
        }
        if self.sensor_gates.is_empty() {
            return Err(PolicyError::MissingSensorGates);
        }
        let mut sensor_ids = std::collections::BTreeSet::new();
        for gate in &self.sensor_gates {
            if gate.sensor_id.trim().is_empty() {
                return Err(PolicyError::EmptySensorId);
            }
            validate_unit(gate.minimum_predictability)?;
            if !sensor_ids.insert(gate.sensor_id.as_str()) {
                return Err(PolicyError::DuplicateSensorGate);
            }
        }
        if self.allow_deceptive_actions || self.allow_online_mutation {
            return Err(PolicyError::UnsafeConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensorPrediction {
    pub sensor_id: String,
    pub predictability: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeMetrics {
    pub mode: NominalMode,
    pub throughput: f64,
    pub safety: f64,
    pub energy_efficiency: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TacticalSnapshot {
    pub collision_risk: f64,
    pub exposure: f64,
    pub threat_confidence: f64,
    pub energy_reserve: f64,
    pub engagement_capacity: bool,
    pub roe_authorized: bool,
    pub sensor_predictions: Vec<SensorPrediction>,
    pub current_nominal_mode: NominalMode,
    pub nominal_modes: Vec<ModeMetrics>,
}

impl TacticalSnapshot {
    fn validate(&self) -> Result<(), PolicyError> {
        for value in [
            self.collision_risk,
            self.exposure,
            self.threat_confidence,
            self.energy_reserve,
        ] {
            validate_unit(value)?;
        }
        if self.nominal_modes.is_empty() {
            return Err(PolicyError::EmptyNominalModes);
        }
        let mut modes = std::collections::BTreeSet::new();
        for metrics in &self.nominal_modes {
            for value in [
                metrics.throughput,
                metrics.safety,
                metrics.energy_efficiency,
            ] {
                validate_unit(value)?;
            }
            if !modes.insert(metrics.mode) {
                return Err(PolicyError::DuplicateNominalMode);
            }
        }
        let mut sensor_ids = std::collections::BTreeSet::new();
        for prediction in &self.sensor_predictions {
            if prediction.sensor_id.trim().is_empty() {
                return Err(PolicyError::EmptySensorId);
            }
            if !sensor_ids.insert(prediction.sensor_id.as_str()) {
                return Err(PolicyError::DuplicateSensorPrediction);
            }
            if let Some(value) = prediction.predictability {
                validate_unit(value)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DefensivePolicy {
    genome: PolicyGenome,
}

impl DefensivePolicy {
    pub fn new(genome: PolicyGenome) -> Result<Self, PolicyError> {
        genome.validate()?;
        Ok(Self { genome })
    }

    pub fn genome(&self) -> &PolicyGenome {
        &self.genome
    }

    pub fn evaluate(&self, snapshot: &TacticalSnapshot) -> Result<DefensiveDecision, PolicyError> {
        snapshot.validate()?;

        if snapshot.collision_risk >= self.genome.collision_break_threshold {
            return Ok(DefensiveDecision {
                action: DefensiveAction::CollisionBreak,
                tier: DecisionTier::Collision,
                reason: DecisionReason::CollisionLimit,
            });
        }
        if snapshot.exposure >= self.genome.emission_control_threshold {
            return Ok(DefensiveDecision {
                action: DefensiveAction::EmissionControl,
                tier: DecisionTier::Exposure,
                reason: DecisionReason::ExposureLimit,
            });
        }
        if snapshot.threat_confidence >= self.genome.threat_response_threshold {
            return Ok(self.threat_decision(snapshot));
        }
        self.throughput_decision(snapshot)
    }

    fn predictability_gate(&self, snapshot: &TacticalSnapshot) -> bool {
        let predictions = snapshot
            .sensor_predictions
            .iter()
            .map(|prediction| (prediction.sensor_id.as_str(), prediction.predictability))
            .collect::<HashMap<_, _>>();
        self.genome.sensor_gates.iter().all(|gate| {
            predictions
                .get(gate.sensor_id.as_str())
                .copied()
                .flatten()
                .is_some_and(|value| value >= gate.minimum_predictability)
        })
    }

    fn threat_decision(&self, snapshot: &TacticalSnapshot) -> DefensiveDecision {
        if !self.predictability_gate(snapshot) {
            return DefensiveDecision {
                action: self.genome.unpredictable_fallback.action(),
                tier: DecisionTier::Threat,
                reason: DecisionReason::SensorFallback,
            };
        }
        if snapshot.exposure > self.genome.engagement_exposure_limit {
            return DefensiveDecision {
                action: DefensiveAction::EmissionControl,
                tier: DecisionTier::Threat,
                reason: DecisionReason::EngagementExposure,
            };
        }
        if snapshot.energy_reserve >= self.genome.minimum_energy_reserve
            && snapshot.engagement_capacity
            && snapshot.roe_authorized
        {
            return DefensiveDecision {
                action: DefensiveAction::Engage,
                tier: DecisionTier::Threat,
                reason: DecisionReason::ThreatGatesPassed,
            };
        }
        DefensiveDecision {
            action: DefensiveAction::Evade,
            tier: DecisionTier::Threat,
            reason: DecisionReason::EngagementGateFailed,
        }
    }

    fn throughput_decision(
        &self,
        snapshot: &TacticalSnapshot,
    ) -> Result<DefensiveDecision, PolicyError> {
        let rows = &snapshot.nominal_modes;
        let score = |row: &ModeMetrics| {
            self.genome.throughput_weight * row.throughput
                + self.genome.safety_weight * row.safety
                + self.genome.energy_weight * row.energy_efficiency
        };
        let best = rows
            .iter()
            .max_by(|left, right| {
                score(left)
                    .total_cmp(&score(right))
                    .then_with(|| right.mode.cmp(&left.mode))
            })
            .ok_or(PolicyError::EmptyNominalModes)?;
        if let Some(current) = rows
            .iter()
            .find(|row| row.mode == snapshot.current_nominal_mode)
        {
            if score(best) < score(current) + self.genome.switch_margin {
                return Ok(DefensiveDecision {
                    action: DefensiveAction::Nominal(current.mode),
                    tier: DecisionTier::Throughput,
                    reason: DecisionReason::HysteresisHold,
                });
            }
        }
        Ok(DefensiveDecision {
            action: DefensiveAction::Nominal(best.mode),
            tier: DecisionTier::Throughput,
            reason: DecisionReason::HighestUtility,
        })
    }
}
