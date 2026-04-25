# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "strix_test_matrix.py"
    spec = importlib.util.spec_from_file_location("strix_test_matrix", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_matrix(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "matrix_version": 1,
                "suite": "unit",
                "commands": [
                    {
                        "id": "pass",
                        "name": "passing command",
                        "tags": ["smoke"],
                        "command": [sys.executable, "-c", "print('ok')"],
                        "expected_exit": 0,
                        "timeout_s": 10,
                    },
                    {
                        "id": "manual",
                        "name": "manual command",
                        "tags": ["manual"],
                        "command": [sys.executable, "-c", "raise SystemExit(7)"],
                        "expected_exit": 7,
                        "timeout_s": 10,
                        "manual": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_matrix_loads_and_selects_non_manual_entries(tmp_path):
    module = _load_module()
    matrix_path = tmp_path / "matrix.json"
    _write_matrix(matrix_path)

    matrix = module.load_matrix(matrix_path)
    selected = module.selected_entries(matrix["commands"], {"smoke"}, include_manual=False)

    assert [entry["id"] for entry in selected] == ["pass"]


def test_dry_run_report_does_not_execute_commands(tmp_path):
    module = _load_module()
    matrix_path = tmp_path / "matrix.json"
    _write_matrix(matrix_path)
    matrix = module.load_matrix(matrix_path)
    entries = module.selected_entries(matrix["commands"], set(), include_manual=True)

    report = module.build_report(matrix_path, matrix, entries, set(), dry_run=True)

    assert report["summary"]["dry_run"] == 2
    assert report["summary"]["failed"] == 0
    assert {result["status"] for result in report["results"]} == {"dry_run"}


def test_run_entry_captures_pass_and_failure():
    module = _load_module()

    passed = module.run_entry(
        {
            "id": "pass",
            "command": [sys.executable, "-c", "print('ok')"],
            "expected_exit": 0,
            "timeout_s": 10,
            "tags": ["unit"],
        },
        dry_run=False,
    )
    failed = module.run_entry(
        {
            "id": "fail",
            "command": [sys.executable, "-c", "raise SystemExit(3)"],
            "expected_exit": 0,
            "timeout_s": 10,
            "tags": ["unit"],
        },
        dry_run=False,
    )

    assert passed["status"] == "passed"
    assert passed["stdout_tail"] == "ok"
    assert failed["status"] == "failed"
    assert failed["exit_code"] == 3


def test_run_entry_captures_missing_executable():
    module = _load_module()

    result = module.run_entry(
        {
            "id": "missing",
            "command": ["definitely-not-a-strix-command"],
            "expected_exit": 0,
            "timeout_s": 10,
            "tags": ["unit"],
        },
        dry_run=False,
    )

    assert result["status"] == "failed"
    assert result["exit_code"] is None
    assert "definitely-not-a-strix-command" in result["stderr_tail"]


def test_empty_selection_is_a_failed_report(tmp_path):
    module = _load_module()
    matrix_path = tmp_path / "matrix.json"
    _write_matrix(matrix_path)
    matrix = module.load_matrix(matrix_path)
    entries = module.selected_entries(matrix["commands"], {"missing-tag"}, include_manual=False)

    report = module.build_report(matrix_path, matrix, entries, {"missing-tag"}, dry_run=False)

    assert report["summary"]["total"] == 1
    assert report["summary"]["failed"] == 1
    assert report["results"][0]["id"] == "__selection__"
    assert "missing-tag" in report["results"][0]["stderr_tail"]


def test_write_reports_creates_json_and_markdown(tmp_path):
    module = _load_module()
    report = {
        "suite": "unit",
        "repo": {"commit": "abc", "branch": "main", "working_tree_clean": True},
        "started_at": "2026-04-24T00:00:00+00:00",
        "completed_at": "2026-04-24T00:00:01+00:00",
        "selected_tags": ["smoke"],
        "results": [
            {
                "id": "pass",
                "status": "passed",
                "elapsed_s": 0.01,
                "tags": ["smoke"],
            }
        ],
    }
    output = tmp_path / "report.json"

    module.write_reports(report, output)

    assert json.loads(output.read_text(encoding="utf-8"))["suite"] == "unit"
    assert "| `pass` | `passed` |" in output.with_suffix(".md").read_text(encoding="utf-8")
