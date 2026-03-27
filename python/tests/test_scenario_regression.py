"""Scenario regression harness tests."""
import pathlib


def test_pass_envelope_validates():
    """A metric within envelope should pass; outside should fail."""
    from strix.temporal.scenario_harness import check_envelope
    envelope = {"drone_survival_rate": {"min": 0.5, "max": 1.0}}
    assert check_envelope({"drone_survival_rate": 0.7}, envelope) == []
    assert len(check_envelope({"drone_survival_rate": 0.3}, envelope)) == 1


def test_pass_envelope_multiple_metrics():
    """Multiple metrics: one fail is enough to report violation."""
    from strix.temporal.scenario_harness import check_envelope
    envelope = {
        "drone_survival_rate": {"min": 0.5, "max": 1.0},
        "time_to_first_strike_s": {"min": 0, "max": 300},
    }
    metrics = {"drone_survival_rate": 0.8, "time_to_first_strike_s": 500}
    violations = check_envelope(metrics, envelope)
    assert len(violations) == 1
    assert "time_to_first_strike_s" in violations[0]


def test_load_scenario_has_pass_envelope():
    """All scenario YAML files should have pass_envelope section."""
    from strix.temporal.scenario_harness import load_scenario
    scenario_dir = pathlib.Path("/home/rmanov/strix/sim/scenarios")
    for f in scenario_dir.glob("*.yaml"):
        data = load_scenario(str(f))
        assert "pass_envelope" in data, f"{f.name} missing pass_envelope"
