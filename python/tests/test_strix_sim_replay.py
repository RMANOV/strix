# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "strix_sim_replay.py"
    spec = importlib.util.spec_from_file_location("strix_sim_replay", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_scenario(path: Path) -> None:
    path.write_text(
        """
scenario_id: replay_case
seed: 77
name: Replay Case
description: Public replay test case
duration_seconds: 30
drones:
  count: 2
  initial_positions:
    - [0, 0, -50]
    - [20, 0, -50]
  max_speed_ms: 10
  endurance_s: 300
environment:
  gps_available: false
mission:
  type: recon
  area:
    center: [0, 0, 0]
    radius: 100
events:
  - time: 10
    type: gps_loss
metrics:
  - active_agents
pass_envelope:
  active_agents:
    min: 2
    max: 2
""",
        encoding="utf-8",
    )


def test_replay_is_deterministic_for_same_seed(tmp_path):
    module = _load_module()
    scenario = tmp_path / "scenario.yaml"
    _write_scenario(scenario)

    first = module.build_replay(scenario, tick_s=10)
    second = module.build_replay(scenario, tick_s=10)

    assert first["frames"] == second["frames"]
    assert first["scenario"]["seed"] == 77
    assert first["metrics"]["active_agents"] == 2
    assert first["envelope"]["status"] == "passed"


def test_replay_outputs_public_safe_paths(tmp_path):
    module = _load_module()
    scenario = tmp_path / "scenario.yaml"
    _write_scenario(scenario)

    replay = module.build_replay(scenario, tick_s=10)

    assert str(tmp_path) not in replay["scenario"]["path"]
    assert replay["scenario"]["path"] == "<external>/scenario.yaml"


def test_public_path_redacts_parent_traversal(tmp_path, monkeypatch):
    module = _load_module()
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "secret" / "scenario.yaml"
    outside.parent.mkdir()
    outside.write_text("scenario_id: external\n", encoding="utf-8")

    monkeypatch.setattr(module, "ROOT", root)

    reported = module.public_path(root / ".." / "secret" / "scenario.yaml")

    assert reported == "<external>/scenario.yaml"
    assert ".." not in reported
    assert "secret" not in reported


def test_replay_html_embeds_visualizer_data(tmp_path):
    module = _load_module()
    scenario = tmp_path / "scenario.yaml"
    _write_scenario(scenario)
    replay = module.build_replay(scenario, tick_s=10)

    html = module.render_html(replay)

    assert "STRIX Replay" in html
    assert "Software-only deterministic kinematic replay" in html
    assert "replay_case" in html
    assert str(tmp_path) not in html


def test_replay_html_escapes_scenario_id_in_title(tmp_path):
    module = _load_module()
    scenario = tmp_path / "scenario.yaml"
    _write_scenario(scenario)
    data = scenario.read_text(encoding="utf-8").replace("scenario_id: replay_case", "scenario_id: \"<bad>&id\"")
    scenario.write_text(data, encoding="utf-8")
    replay = module.build_replay(scenario, tick_s=10)

    html = module.render_html(replay)

    assert "<title>STRIX Software Replay - &lt;bad&gt;&amp;id</title>" in html
    assert "<title>STRIX Software Replay - <bad>&id</title>" not in html


def test_envelope_fails_when_required_metric_is_missing(tmp_path):
    module = _load_module()
    scenario = tmp_path / "scenario.yaml"
    _write_scenario(scenario)
    data = scenario.read_text(encoding="utf-8").replace(
        "  active_agents:\n    min: 2\n    max: 2\n",
        "  missing_metric:\n    min: 1\n    max: 2\n",
    )
    scenario.write_text(data, encoding="utf-8")

    replay = module.build_replay(scenario, tick_s=10)

    assert replay["envelope"]["status"] == "failed"
    assert replay["envelope"]["checks"][0]["status"] == "not_observed"


def test_write_replay_creates_json_and_html(tmp_path):
    module = _load_module()
    scenario = tmp_path / "scenario.yaml"
    _write_scenario(scenario)
    replay = module.build_replay(scenario, tick_s=10)
    output = tmp_path / "replay.json"
    html_output = tmp_path / "replay.html"

    module.write_replay(replay, output, html_output)

    assert json.loads(output.read_text(encoding="utf-8"))["kind"] == "software_replay"
    assert "STRIX Replay" in html_output.read_text(encoding="utf-8")


def test_replay_handles_zero_index_attrition(tmp_path):
    module = _load_module()
    scenario = tmp_path / "scenario.yaml"
    _write_scenario(scenario)
    data = scenario.read_text(encoding="utf-8")
    scenario.write_text(
        data
        + """
attrition_schedule:
  - time: 10
    drone_id: 0
    cause: public_test_event
""",
        encoding="utf-8",
    )

    replay = module.build_replay(scenario, tick_s=10)

    assert replay["metrics"]["active_agents"] == 1
    assert replay["metrics"]["offline_agents"] == 1
    assert replay["envelope"]["status"] == "failed"
    offline_events = [event for event in replay["frames"][1]["events"] if event["type"] == "agent_offline"]
    assert offline_events[0]["agent_index"] == 0


@pytest.mark.parametrize("scenario_name", ["gps_denied_recon", "mass_attrition", "multi_domain"])
def test_official_replay_does_not_claim_uninstrumented_targets(scenario_name):
    module = _load_module()
    scenario = module.ROOT / "sim" / "scenarios" / f"{scenario_name}.yaml"

    replay = module.build_replay(scenario, tick_s=10)
    required = module.read_scenario(scenario)["pass_envelope"]

    assert replay["envelope"]["status"] == "failed"
    assert all(check["status"] == "not_observed" for check in replay["envelope"]["checks"])
    for metric_name in required:
        assert metric_name not in replay["metrics"]
        assert replay["metric_evidence"][metric_name]["status"] == "not_observed"
        assert replay["metric_evidence"][metric_name]["reason"]


def test_scenario_id_cannot_create_measurements(tmp_path):
    module = _load_module()
    scenario = tmp_path / "scenario.yaml"
    _write_scenario(scenario)
    ordinary = module.build_replay(scenario, tick_s=10)
    for scenario_id in ("gps_denied_recon", "mass_attrition", "multi_domain"):
        data = scenario.read_text(encoding="utf-8")
        data = data.replace(data.splitlines()[1], f"scenario_id: {scenario_id}")
        scenario.write_text(data, encoding="utf-8")
        named = module.build_replay(scenario, tick_s=10)
        assert named["frames"] == ordinary["frames"]
        assert named["metrics"] == ordinary["metrics"]


def test_no_constraints_means_clearance_is_not_observed(tmp_path):
    module = _load_module()
    scenario = tmp_path / "scenario.yaml"
    _write_scenario(scenario)

    replay = module.build_replay(scenario, tick_s=10)

    assert replay["constraints"] == []
    assert "min_constraint_clearance_m" not in replay["metrics"]
    assert replay["metric_evidence"]["min_constraint_clearance_m"]["status"] == "not_observed"


@pytest.mark.parametrize("observed", [float("nan"), float("inf"), float("-inf"), True, "not-a-number"])
def test_invalid_value_cannot_pass_envelope(observed):
    module = _load_module()

    result = module.evaluate_envelope({"value": observed}, {"pass_envelope": {"value": {"min": 0}}})

    assert result["status"] == "failed"
    assert result["checks"][0]["status"] == "invalid_observation"


def test_official_gps_replay_cli_preserves_targets_and_returns_failure(tmp_path):
    module = _load_module()
    scenario = module.ROOT / "sim" / "scenarios" / "gps_denied_recon.yaml"
    original_scenario = scenario.read_bytes()
    output = tmp_path / "replay.json"
    html_output = tmp_path / "replay.html"

    result = subprocess.run(
        [sys.executable, str(module.ROOT / "scripts" / "strix_sim_replay.py"),
         "--scenario", str(scenario), "--output", str(output), "--html", str(html_output)],
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 1, result.stderr
    assert scenario.read_bytes() == original_scenario
    replay = json.loads(output.read_text(encoding="utf-8"))
    assert replay["envelope"]["status"] == "failed"
    assert replay["metrics"]["frame_count"] == 61
    assert replay["metrics"]["active_agents"] == 4
    assert "not_observed" in html_output.read_text(encoding="utf-8")


@pytest.mark.parametrize("bounds", [None, {}, {"min": float("nan")}, {"max": float("inf")}, {"min": True}, {"max": "not-a-number"}, {"min": 2, "max": 1}])
def test_invalid_requirement_cannot_be_skipped_or_passed(bounds):
    module = _load_module()

    result = module.evaluate_envelope({"value": 1}, {"pass_envelope": {"value": bounds}})

    assert result["status"] == "failed"
    assert result["checks"][0]["status"] == "invalid_envelope"


@pytest.mark.parametrize(
    ("metrics", "bounds", "expected_check"),
    [
        ({}, {"min": 10 ** 1000}, "not_observed"),
        ({"value": 1}, {"min": 10 ** 1000}, "failed"),
        ({"value": 1}, {"max": 10 ** 1000}, "passed"),
        ({"value": 10 ** 1000}, {"min": 10 ** 1000, "max": 10 ** 1000}, "passed"),
    ],
)
def test_large_integer_bounds_and_observations_keep_exact_comparisons(metrics, bounds, expected_check):
    module = _load_module()

    result = module.evaluate_envelope(metrics, {"pass_envelope": {"value": bounds}})

    assert result["checks"][0]["status"] == expected_check
    assert result["status"] == ("passed" if expected_check == "passed" else "failed")
