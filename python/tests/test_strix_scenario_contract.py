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
    assert str(tmp_path) not in result["path"]


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


def test_validate_scenario_parse_errors_are_public_safe(tmp_path):
    module = _load_module()
    scenario = tmp_path / "broken.yaml"
    scenario.write_text("scenario_id: [unterminated\n", encoding="utf-8")

    result = module.validate_scenario(scenario)

    assert result["status"] == "failed"
    assert str(tmp_path) not in result["path"]
    assert str(tmp_path) not in result["errors"][0]


def test_validate_scenario_read_errors_are_public_safe(tmp_path):
    module = _load_module()
    scenario = tmp_path / "missing.yaml"

    result = module.validate_scenario(scenario)

    assert result["status"] == "failed"
    assert str(tmp_path) not in result["path"]
    assert str(tmp_path) not in result["errors"][0]


def test_public_exception_message_redacts_windows_paths():
    module = _load_module()
    windows_path = "C:\\tmp\\missing.yaml"
    error = FileNotFoundError(2, "No such file or directory", windows_path)

    sanitized = module.public_exception_message(error, Path(windows_path))

    assert windows_path not in sanitized
    assert repr(windows_path)[1:-1] not in sanitized
    assert "<external>/missing.yaml" in sanitized


def test_public_path_prefers_repo_relative_windows_paths(monkeypatch):
    module = _load_module()
    root = module.PureWindowsPath("C:/repo")
    scenario = module.PureWindowsPath("C:/repo/sim/scenarios/a.yaml")

    monkeypatch.setattr(module, "ROOT", root)

    assert module.public_path(scenario) == "sim\\scenarios\\a.yaml"


def test_public_scenarios_satisfy_contract():
    module = _load_module()
    scenario_dir = Path(__file__).resolve().parents[2] / "sim" / "scenarios"

    report = module.validate_directory(scenario_dir)

    assert report["summary"]["failed"] == 0
    assert report["summary"]["total"] >= 1


def test_validate_directory_rejects_missing_directory(tmp_path):
    module = _load_module()

    report = module.validate_directory(tmp_path / "missing")

    assert report["summary"]["failed"] == 1
    assert str(tmp_path) not in report["scenario_dir"]
    assert "does not exist" in report["results"][0]["errors"][0]
    assert str(tmp_path) not in report["results"][0]["errors"][0]


def test_validate_directory_rejects_file_path(tmp_path):
    module = _load_module()
    not_a_dir = tmp_path / "scenario.yaml"
    not_a_dir.write_text("scenario_id: x\n", encoding="utf-8")

    report = module.validate_directory(not_a_dir)

    assert report["summary"]["failed"] == 1
    assert "not a directory" in report["results"][0]["errors"][0]
    assert str(tmp_path) not in report["results"][0]["errors"][0]


def test_validate_directory_rejects_empty_directory(tmp_path):
    module = _load_module()

    report = module.validate_directory(tmp_path)

    assert report["summary"]["failed"] == 1
    assert "no scenario files" in report["results"][0]["errors"][0]
    assert str(tmp_path) not in report["results"][0]["errors"][0]


def test_write_report_outputs_json(tmp_path):
    module = _load_module()
    output = tmp_path / "report.json"
    report = {"summary": {"failed": 0}, "results": []}

    module.write_report(report, output)

    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["failed"] == 0
