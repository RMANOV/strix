# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


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
  - area_coverage_pct
pass_envelope:
  area_coverage_pct:
    min: 0
    max: 100
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
        "  area_coverage_pct:\n    min: 0\n    max: 100\n",
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
    offline_events = [event for event in replay["frames"][1]["events"] if event["type"] == "agent_offline"]
    assert offline_events[0]["agent_index"] == 0
