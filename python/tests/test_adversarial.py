"""Tests for the adversarial Python engine."""

import math

from strix.adversarial import AdversarialEngine, EnemyDoctrine, SensorReading
from strix.brain import Vec3


def test_doctrine_probabilities_normalize_after_update():
    engine = AdversarialEngine(n_enemy_particles=64)
    engine.set_friendly_centroid(Vec3(100.0, 0.0, 0.0))
    engine.init_track(1, Vec3(0.0, 0.0, 0.0))

    engine.update_from_sensor(
        SensorReading(
            threat_id=1,
            position=Vec3(0.0, 0.0, 0.0),
            velocity=Vec3(18.0, 0.0, 0.0),
            confidence=0.9,
        )
    )

    estimate = engine.predict_enemy(0.1)[1]
    total = sum(estimate.doctrine_probabilities.values())

    assert math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert estimate.dominant_doctrine in estimate.doctrine_probabilities


def test_rf_signature_promotes_jamming_and_deception():
    engine = AdversarialEngine(n_enemy_particles=64)
    engine.set_friendly_centroid(Vec3(0.0, 0.0, 0.0))
    engine.init_track(2, Vec3(250.0, 0.0, 0.0))

    engine.update_from_sensor(
        SensorReading(
            threat_id=2,
            position=Vec3(250.0, 0.0, 0.0),
            velocity=Vec3(0.0, 0.0, 0.0),
            signal_strength_dbm=-32.0,
            radar_cross_section=0.05,
            confidence=0.95,
            sensor_type="ew",
        )
    )

    estimate = engine.predict_enemy(0.1)[2]

    assert estimate.doctrine_probabilities[EnemyDoctrine.EW_JAMMING] > 0.20
    assert estimate.deception_score > 0.30


def test_flanking_hypothesis_beats_direct_assault_for_lateral_motion():
    engine = AdversarialEngine(n_enemy_particles=64)
    engine.set_friendly_centroid(Vec3(0.0, 0.0, 0.0))
    engine.init_track(3, Vec3(100.0, 0.0, 0.0))

    engine.update_from_sensor(
        SensorReading(
            threat_id=3,
            position=Vec3(100.0, 0.0, 0.0),
            velocity=Vec3(-3.0, 18.0, 0.0),
            confidence=0.9,
        )
    )

    estimate = engine.predict_enemy(0.1)[3]

    assert (
        estimate.doctrine_probabilities[EnemyDoctrine.FLANKING]
        > estimate.doctrine_probabilities[EnemyDoctrine.DIRECT_ASSAULT]
    )


def test_withdrawal_hypothesis_beats_direct_assault_when_opening_range():
    engine = AdversarialEngine(n_enemy_particles=64)
    engine.set_friendly_centroid(Vec3(0.0, 0.0, 0.0))
    engine.init_track(4, Vec3(50.0, 0.0, 0.0))

    engine.update_from_sensor(
        SensorReading(
            threat_id=4,
            position=Vec3(50.0, 0.0, 0.0),
            velocity=Vec3(12.0, 0.0, 0.0),
            confidence=0.9,
        )
    )

    estimate = engine.predict_enemy(0.1)[4]

    assert (
        estimate.doctrine_probabilities[EnemyDoctrine.FIGHTING_WITHDRAWAL]
        > estimate.doctrine_probabilities[EnemyDoctrine.DIRECT_ASSAULT]
    )


def test_deterministic_replay_with_seed():
    """Two engines with the same seed produce identical estimates."""
    for seed in [42, 12345, 0]:
        e1 = AdversarialEngine(n_enemy_particles=64, seed=seed)
        e2 = AdversarialEngine(n_enemy_particles=64, seed=seed)
        for eng in [e1, e2]:
            eng.set_friendly_centroid(Vec3(100.0, 0.0, 0.0))
            eng.init_track(1, Vec3(0.0, 0.0, 0.0))
            eng.update_from_sensor(SensorReading(
                threat_id=1, position=Vec3(0.0, 0.0, 0.0),
                velocity=Vec3(10.0, 0.0, 0.0), confidence=0.8,
            ))
        est1 = e1.predict_enemy(0.1)[1]
        est2 = e2.predict_enemy(0.1)[1]
        assert est1.position.x == est2.position.x
        assert est1.confidence == est2.confidence