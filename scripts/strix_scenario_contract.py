#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate public STRIX scenario files against the canonical test contract."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in missing-dependency environments
    raise RuntimeError(
        "scripts/strix_scenario_contract.py requires PyYAML. "
        "Install it with `pip install pyyaml` or use the project dependencies."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_DIR = ROOT / "sim" / "scenarios"
SCENARIO_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def public_path(path: Path) -> str:
    """Return a report-safe path without leaking local checkout layout."""

    if path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT))
    if path.is_absolute():
        name = path.name or "."
        return f"<external>/{name}"
    return str(path)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scenario file must contain a YAML mapping")
    return data


def is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def validate_scenario(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        data = load_yaml(path)
    except Exception as exc:  # noqa: BLE001 - report parse errors as validation failures
        return {"path": str(path), "status": "failed", "errors": [str(exc)], "warnings": []}

    required = [
        "scenario_id",
        "seed",
        "name",
        "description",
        "duration_seconds",
        "environment",
        "mission",
        "metrics",
        "pass_envelope",
    ]
    for key in required:
        if key not in data:
            errors.append(f"missing required field: {key}")

    scenario_id = data.get("scenario_id")
    if scenario_id is not None and (
        not isinstance(scenario_id, str) or not SCENARIO_ID_RE.fullmatch(scenario_id)
    ):
        errors.append("scenario_id must be lowercase kebab/snake style")

    seed = data.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        errors.append("seed must be a non-negative integer")

    duration = data.get("duration_seconds")
    if duration is not None and (not is_number(duration) or duration <= 0):
        errors.append("duration_seconds must be a positive number")

    if "drones" not in data and "platforms" not in data:
        errors.append("scenario must define either drones or platforms")

    for mapping_key in ("environment", "mission"):
        value = data.get(mapping_key)
        if value is not None and not isinstance(value, dict):
            errors.append(f"{mapping_key} must be a mapping")

    metrics = data.get("metrics")
    metric_names: set[str] = set()
    if metrics is not None:
        if not isinstance(metrics, list) or not metrics:
            errors.append("metrics must be a non-empty list")
        else:
            for metric in metrics:
                if not isinstance(metric, str) or not metric:
                    errors.append("metrics entries must be non-empty strings")
                else:
                    metric_names.add(metric)

    envelope = data.get("pass_envelope")
    if envelope is not None:
        if not isinstance(envelope, dict) or not envelope:
            errors.append("pass_envelope must be a non-empty mapping")
        else:
            for metric_name, bounds in envelope.items():
                if not isinstance(metric_name, str) or not metric_name:
                    errors.append("pass_envelope metric names must be non-empty strings")
                    continue
                if metric_names and metric_name not in metric_names:
                    warnings.append(f"pass_envelope metric {metric_name!r} is not listed in metrics")
                if not isinstance(bounds, dict):
                    errors.append(f"{metric_name}: bounds must be a mapping")
                    continue
                has_min = "min" in bounds
                has_max = "max" in bounds
                if not has_min and not has_max:
                    errors.append(f"{metric_name}: bounds must include min and/or max")
                if has_min and not is_number(bounds["min"]):
                    errors.append(f"{metric_name}: min must be numeric")
                if has_max and not is_number(bounds["max"]):
                    errors.append(f"{metric_name}: max must be numeric")
                if has_min and has_max and is_number(bounds["min"]) and is_number(bounds["max"]):
                    if bounds["min"] > bounds["max"]:
                        errors.append(f"{metric_name}: min must be <= max")

    status = "failed" if errors else "passed"
    return {
        "path": public_path(path),
        "scenario_id": scenario_id,
        "status": status,
        "errors": errors,
        "warnings": warnings,
    }


def validate_directory(scenario_dir: Path) -> dict[str, Any]:
    scenario_dir_str = public_path(scenario_dir)

    if not scenario_dir.exists():
        return directory_failure_report(
            scenario_dir_str,
            f"scenario directory does not exist: {scenario_dir_str}",
        )
    if not scenario_dir.is_dir():
        return directory_failure_report(
            scenario_dir_str,
            f"scenario path is not a directory: {scenario_dir_str}",
        )

    files = sorted(scenario_dir.glob("*.yaml"))
    if not files:
        return directory_failure_report(
            scenario_dir_str,
            f"no scenario files found in directory: {scenario_dir_str}",
        )

    results = [validate_scenario(path) for path in files]
    failed = [result for result in results if result["status"] != "passed"]
    warnings = sum(len(result["warnings"]) for result in results)
    return {
        "report_version": 1,
        "scenario_dir": scenario_dir_str,
        "summary": {
            "total": len(results),
            "passed": len(results) - len(failed),
            "failed": len(failed),
            "warnings": warnings,
        },
        "results": results,
    }


def directory_failure_report(scenario_dir: str, error: str) -> dict[str, Any]:
    return {
        "report_version": 1,
        "scenario_dir": scenario_dir,
        "summary": {
            "total": 1,
            "passed": 0,
            "failed": 1,
            "warnings": 0,
        },
        "results": [
            {
                "path": scenario_dir,
                "scenario_id": None,
                "status": "failed",
                "errors": [error],
                "warnings": [],
            }
        ],
    }


def write_report(report: dict[str, Any], output: Path | None) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(encoded)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario_dir = args.scenario_dir if args.scenario_dir.is_absolute() else ROOT / args.scenario_dir
    report = validate_directory(scenario_dir)
    output = args.output if args.output is None or args.output.is_absolute() else ROOT / args.output
    write_report(report, output)
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
