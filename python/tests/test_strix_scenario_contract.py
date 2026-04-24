# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "strix_scenario_contract.py"
    spec = importlib.util.spec_from_file_location("strix_scenario_contract", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validate_scenario_accepts_minimal_contract(tmp_path):
    module = _load_module()
    scenario = tmp_path / "valid.yaml"
    scenario.write_text(
        """
scenario_id: valid_case
seed: 42
name: Valid Case
description: Public test case
duration_seconds: 10
drones:
  count: 1
environment: {}
mission: {}
metrics:
  - coverage
pass_envelope:
  coverage:
    min: 0
    max: 1
""",
        encoding="utf-8",
    )

    result = module.validate_scenario(scenario)

    assert result["status"] == "passed"
    assert result["errors"] == []


def test_validate_scenario_rejects_missing_seed_and_bad_bounds(tmp_path):
    module = _load_module()
    scenario = tmp_path / "invalid.yaml"
    scenario.write_text(
        """
scenario_id: Invalid Case
name: Invalid Case
description: Public test case
duration_seconds: 10
drones:
  count: 1
environment: {}
mission: {}
metrics:
  - coverage
pass_envelope:
  coverage:
    min: 2
    max: 1
""",
        encoding="utf-8",
    )

    result = module.validate_scenario(scenario)

    assert result["status"] == "failed"
    assert "missing required field: seed" in result["errors"]
    assert "scenario_id must be lowercase kebab/snake style" in result["errors"]
    assert "coverage: min must be <= max" in result["errors"]


def test_public_scenarios_satisfy_contract():
    module = _load_module()
    scenario_dir = Path(__file__).resolve().parents[2] / "sim" / "scenarios"

    report = module.validate_directory(scenario_dir)

    assert report["summary"]["failed"] == 0
    assert report["summary"]["total"] >= 1


def test_write_report_outputs_json(tmp_path):
    module = _load_module()
    output = tmp_path / "report.json"
    report = {"summary": {"failed": 0}, "results": []}

    module.write_report(report, output)

    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["failed"] == 0
