"""Scenario regression harness — validate metrics against pass/fail envelopes."""
from __future__ import annotations
import yaml
import pathlib


def check_envelope(metrics: dict[str, float], envelope: dict[str, dict]) -> list[str]:
    """Check metrics against envelope bounds. Returns list of violation messages."""
    violations = []
    for metric_name, bounds in envelope.items():
        if metric_name not in metrics:
            violations.append(f"{metric_name}: missing from metrics")
            continue
        value = metrics[metric_name]
        if "min" in bounds and value < bounds["min"]:
            violations.append(f"{metric_name}: {value} < min {bounds['min']}")
        if "max" in bounds and value > bounds["max"]:
            violations.append(f"{metric_name}: {value} > max {bounds['max']}")
    return violations


def load_scenario(path: str) -> dict:
    """Load and return a scenario YAML file."""
    return yaml.safe_load(pathlib.Path(path).read_text())
