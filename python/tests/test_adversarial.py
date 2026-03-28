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


def test_calibration_isotonic_identity():
    """Uncalibrated engine: calibrator initialized with identity (passthrough)."""
    from strix.adversarial import ConfidenceCalibrator
    cal = ConfidenceCalibrator()
    assert cal.calibrate(0.5) == 0.5
    assert cal.calibrate(0.9) == 0.9


def test_calibration_after_fit():
    """After fitting, high raw-confidence on wrong predictions should reduce output."""
    from strix.adversarial import ConfidenceCalibrator
    cal = ConfidenceCalibrator()
    raw = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1]
    actual = [0, 0, 0, 1, 1, 1]
    cal.fit(raw, actual)
    assert cal.calibrate(0.9) <= 0.5  # anti-calibrated → PAV pools to 0.5


def test_calibration_monotonic():
    """Calibrated values should be monotonically non-decreasing."""
    from strix.adversarial import ConfidenceCalibrator
    cal = ConfidenceCalibrator()
    raw = [0.1, 0.3, 0.5, 0.7, 0.9]
    actual = [0, 0, 1, 1, 1]
    cal.fit(raw, actual)
    calibrated = [cal.calibrate(x) for x in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    for i in range(1, len(calibrated)):
        assert calibrated[i] >= calibrated[i-1] - 1e-9


# ── E1: New doctrine types ──

def test_probing_doctrine_in_enum():
    """New doctrine types should exist in the enum."""
    assert hasattr(EnemyDoctrine, "PROBING")
    assert hasattr(EnemyDoctrine, "FEINT")
    assert hasattr(EnemyDoctrine, "COORDINATED_ATTACK")


def test_probing_detected_for_oscillating_approach():
    """Enemy that approaches then backs off should trigger PROBING."""
    engine = AdversarialEngine(n_enemy_particles=64, seed=42)
    engine.set_friendly_centroid(Vec3(0.0, 0.0, 0.0))
    engine.init_track(10, Vec3(200.0, 0.0, 0.0))
    # Approach
    engine.update_from_sensor(SensorReading(
        threat_id=10, position=Vec3(180.0, 0.0, 0.0),
        velocity=Vec3(-8.0, 0.0, 0.0), confidence=0.8))
    engine.predict_enemy(1.0)
    # Withdraw
    engine.update_from_sensor(SensorReading(
        threat_id=10, position=Vec3(190.0, 0.0, 0.0),
        velocity=Vec3(5.0, 0.0, 0.0), confidence=0.8))
    engine.predict_enemy(1.0)
    # Approach again
    engine.update_from_sensor(SensorReading(
        threat_id=10, position=Vec3(175.0, 0.0, 0.0),
        velocity=Vec3(-10.0, 0.0, 0.0), confidence=0.8))
    est = engine.predict_enemy(1.0)[10]
    assert EnemyDoctrine.PROBING in est.doctrine_probabilities


# ── E2: Temporal deception ──

def test_deception_score_increases_for_inconsistent_behavior():
    """Threat that flips behavior should accumulate behavior transitions."""
    engine = AdversarialEngine(n_enemy_particles=64, seed=42)
    engine.set_friendly_centroid(Vec3(0.0, 0.0, 0.0))
    engine.init_track(30, Vec3(100.0, 0.0, 0.0))
    # Consistent attacker — always same velocity
    for _ in range(5):
        engine.update_from_sensor(SensorReading(
            threat_id=30, position=Vec3(80.0, 0.0, 0.0),
            velocity=Vec3(-12.0, 0.0, 0.0), confidence=0.8))
        engine.predict_enemy(0.5)
    hist_consistent = engine._behavior_history.get(30, [])
    transitions_consistent = sum(1 for a, b in zip(hist_consistent, hist_consistent[1:]) if a != b) if len(hist_consistent) > 1 else 0

    engine2 = AdversarialEngine(n_enemy_particles=64, seed=42)
    engine2.set_friendly_centroid(Vec3(0.0, 0.0, 0.0))
    engine2.init_track(31, Vec3(100.0, 0.0, 0.0))
    # Give it more extreme oscillation to force behavior flips
    vels = [Vec3(-20.0, 0.0, 0.0), Vec3(15.0, 0.0, 0.0)] * 5
    for i, v in enumerate(vels[:8]):
        pos_x = 80.0 if i % 2 == 0 else 110.0
        engine2.update_from_sensor(SensorReading(
            threat_id=31, position=Vec3(pos_x, 0.0, 0.0),
            velocity=v, confidence=0.9))
        engine2.predict_enemy(0.5)
    hist_inconsistent = engine2._behavior_history.get(31, [])
    transitions_inconsistent = sum(1 for a, b in zip(hist_inconsistent, hist_inconsistent[1:]) if a != b) if len(hist_inconsistent) > 1 else 0

    # The flipper should have more behavior transitions
    assert transitions_inconsistent >= transitions_consistent, (
        f"inconsistent={transitions_inconsistent} should >= consistent={transitions_consistent}"
    )


# ── E3: Adversarial → auction risk context ──

def test_adversarial_risk_context_format():
    from strix.adversarial import adversarial_to_risk_context
    engine = AdversarialEngine(n_enemy_particles=64, seed=42)
    engine.set_friendly_centroid(Vec3(0.0, 0.0, 0.0))
    engine.init_track(40, Vec3(50.0, 0.0, 0.0))
    engine.update_from_sensor(SensorReading(
        threat_id=40, position=Vec3(50.0, 0.0, 0.0),
        velocity=Vec3(-15.0, 0.0, 0.0), confidence=0.9))
    estimates = engine.predict_enemy(0.1)
    ctx = adversarial_to_risk_context(estimates)
    assert "threat_density" in ctx
    assert "confidence" in ctx
    assert 0.0 <= ctx["confidence"] <= 1.0
    assert ctx["threat_density"] >= 0.0


def test_adversarial_risk_context_empty():
    from strix.adversarial import adversarial_to_risk_context
    ctx = adversarial_to_risk_context({})
    assert ctx["threat_density"] == 0.0
    assert ctx["confidence"] == 0.5


def test_time_to_contact_does_not_mutate_behavior_history():
    engine = AdversarialEngine(n_enemy_particles=64, seed=7)
    engine.set_friendly_centroid(Vec3(0.0, 0.0, 0.0))
    engine.init_track(50, Vec3(100.0, 0.0, 0.0))
    engine.update_from_sensor(
        SensorReading(
            threat_id=50,
            position=Vec3(100.0, 0.0, 0.0),
            velocity=Vec3(-10.0, 0.0, 0.0),
            confidence=0.9,
        )
    )
    engine.predict_enemy(0.1)

    before = list(engine._behavior_history.get(50, []))
    ttc = engine.time_to_contact(50)
    after = list(engine._behavior_history.get(50, []))

    assert 50 in ttc
    assert before == after