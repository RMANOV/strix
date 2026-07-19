use strix_core::defensive_policy::{
    DecisionReason, DecisionTier, DefensiveAction, DefensivePolicy, ModeMetrics, NominalMode,
    PolicyError, PolicyGenome, SensorFallback, SensorGate, SensorPrediction, TacticalSnapshot,
};

fn mode(mode: NominalMode, throughput: f64, safety: f64, energy: f64) -> ModeMetrics {
    ModeMetrics {
        mode,
        throughput,
        safety,
        energy_efficiency: energy,
    }
}

fn snapshot() -> TacticalSnapshot {
    TacticalSnapshot {
        collision_risk: 0.1,
        exposure: 0.1,
        threat_confidence: 0.1,
        energy_reserve: 0.8,
        engagement_capacity: true,
        roe_authorized: true,
        sensor_predictions: vec![SensorPrediction {
            sensor_id: "primary".into(),
            predictability: Some(0.9),
        }],
        current_nominal_mode: NominalMode::Observe,
        nominal_modes: vec![
            mode(NominalMode::Observe, 0.4, 0.8, 0.8),
            mode(NominalMode::Relay, 0.9, 0.8, 0.8),
        ],
    }
}

fn policy() -> DefensivePolicy {
    DefensivePolicy::new(PolicyGenome::default()).expect("default genome must validate")
}

#[test]
fn collision_break_is_absolute_priority() {
    let mut input = snapshot();
    input.collision_risk = 0.9;
    input.exposure = 1.0;
    input.threat_confidence = 1.0;

    let decision = policy().evaluate(&input).unwrap();

    assert_eq!(decision.action, DefensiveAction::CollisionBreak);
    assert_eq!(decision.tier, DecisionTier::Collision);
}

#[test]
fn exposure_control_precedes_threat_response() {
    let mut input = snapshot();
    input.exposure = 0.9;
    input.threat_confidence = 1.0;

    let decision = policy().evaluate(&input).unwrap();

    assert_eq!(decision.action, DefensiveAction::EmissionControl);
    assert_eq!(decision.tier, DecisionTier::Exposure);
}

#[test]
fn engagement_requires_energy_capacity_roe_exposure_and_sensors() {
    let mut input = snapshot();
    input.threat_confidence = 0.9;
    input.exposure = 0.2;

    let decision = policy().evaluate(&input).unwrap();

    assert_eq!(decision.action, DefensiveAction::Engage);
    assert_eq!(decision.reason, DecisionReason::ThreatGatesPassed);
}

#[test]
fn failed_engagement_authority_or_capacity_evades() {
    for change in 0..3 {
        let mut input = snapshot();
        input.threat_confidence = 0.9;
        match change {
            0 => input.energy_reserve = 0.2,
            1 => input.engagement_capacity = false,
            _ => input.roe_authorized = false,
        }
        assert_eq!(
            policy().evaluate(&input).unwrap().action,
            DefensiveAction::Evade
        );
    }
}

#[test]
fn threat_exposure_above_engagement_limit_blocks_engagement() {
    let mut input = snapshot();
    input.threat_confidence = 0.9;
    input.exposure = 0.5;

    let decision = policy().evaluate(&input).unwrap();

    assert_eq!(decision.action, DefensiveAction::EmissionControl);
    assert_eq!(decision.tier, DecisionTier::Threat);
    assert_eq!(decision.reason, DecisionReason::EngagementExposure);
}

#[test]
fn each_required_sensor_uses_its_own_threshold() {
    let genome = PolicyGenome {
        sensor_gates: vec![
            SensorGate {
                sensor_id: "primary".into(),
                minimum_predictability: 0.6,
            },
            SensorGate {
                sensor_id: "secondary".into(),
                minimum_predictability: 0.8,
            },
        ],
        ..PolicyGenome::default()
    };
    let policy = DefensivePolicy::new(genome).unwrap();
    let mut input = snapshot();
    input.threat_confidence = 0.9;
    input.sensor_predictions.push(SensorPrediction {
        sensor_id: "secondary".into(),
        predictability: Some(0.79),
    });

    assert_eq!(
        policy.evaluate(&input).unwrap().action,
        DefensiveAction::EmissionControl
    );
    input.sensor_predictions[1].predictability = Some(0.8);
    assert_eq!(
        policy.evaluate(&input).unwrap().action,
        DefensiveAction::Engage
    );
}

#[test]
fn missing_or_unknown_sensor_uses_configured_fallback() {
    let mut genome = PolicyGenome {
        unpredictable_fallback: SensorFallback::Evade,
        ..PolicyGenome::default()
    };
    let mut input = snapshot();
    input.threat_confidence = 0.9;
    input.sensor_predictions.clear();
    assert_eq!(
        DefensivePolicy::new(genome.clone())
            .unwrap()
            .evaluate(&input)
            .unwrap()
            .action,
        DefensiveAction::Evade
    );

    genome.unpredictable_fallback = SensorFallback::EmissionControl;
    input.sensor_predictions.push(SensorPrediction {
        sensor_id: "primary".into(),
        predictability: None,
    });
    assert_eq!(
        DefensivePolicy::new(genome)
            .unwrap()
            .evaluate(&input)
            .unwrap()
            .action,
        DefensiveAction::EmissionControl
    );
}

#[test]
fn nominal_argmax_is_deterministic_and_weighted() {
    let mut input = snapshot();
    input.current_nominal_mode = NominalMode::Recover;
    input
        .nominal_modes
        .push(mode(NominalMode::Recover, 0.1, 0.1, 0.1));

    let decision = policy().evaluate(&input).unwrap();

    assert_eq!(
        decision.action,
        DefensiveAction::Nominal(NominalMode::Relay)
    );
    assert_eq!(decision.reason, DecisionReason::HighestUtility);
}

#[test]
fn nominal_hysteresis_holds_small_improvements() {
    let mut input = snapshot();
    input.nominal_modes = vec![
        mode(NominalMode::Observe, 0.7, 0.7, 0.7),
        mode(NominalMode::Relay, 0.75, 0.75, 0.75),
    ];

    let decision = policy().evaluate(&input).unwrap();

    assert_eq!(
        decision.action,
        DefensiveAction::Nominal(NominalMode::Observe)
    );
    assert_eq!(decision.reason, DecisionReason::HysteresisHold);
}

#[test]
fn equal_nominal_scores_choose_stable_enum_order() {
    let mut input = snapshot();
    input.current_nominal_mode = NominalMode::Recover;
    input.nominal_modes = vec![
        mode(NominalMode::Relay, 0.7, 0.7, 0.7),
        mode(NominalMode::Observe, 0.7, 0.7, 0.7),
        mode(NominalMode::Recover, 0.1, 0.1, 0.1),
    ];

    assert_eq!(
        policy().evaluate(&input).unwrap().action,
        DefensiveAction::Nominal(NominalMode::Observe)
    );
}

#[test]
fn non_finite_and_out_of_range_inputs_fail_closed() {
    let mut input = snapshot();
    input.collision_risk = f64::NAN;
    assert_eq!(policy().evaluate(&input), Err(PolicyError::NonFiniteInput));

    input.collision_risk = 1.1;
    assert_eq!(policy().evaluate(&input), Err(PolicyError::OutOfRangeInput));
}

#[test]
fn unsafe_adaptation_controls_are_rejected() {
    for genome in [
        PolicyGenome {
            allow_deceptive_actions: true,
            ..PolicyGenome::default()
        },
        PolicyGenome {
            allow_online_mutation: true,
            ..PolicyGenome::default()
        },
    ] {
        assert_eq!(
            DefensivePolicy::new(genome).unwrap_err(),
            PolicyError::UnsafeConfiguration
        );
    }
}

#[test]
fn duplicate_sensor_observations_fail_closed() {
    let mut input = snapshot();
    input.sensor_predictions.push(SensorPrediction {
        sensor_id: "primary".into(),
        predictability: Some(0.9),
    });

    assert_eq!(
        policy().evaluate(&input),
        Err(PolicyError::DuplicateSensorPrediction)
    );
}
