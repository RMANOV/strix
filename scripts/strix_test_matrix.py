#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run repeatable STRIX public test matrices and write evidence reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "Project_Docs" / "testing" / "public_test_matrix.json"
DEFAULT_OUTPUT = ROOT / "target" / "strix-test-reports" / "latest.json"
TAIL_LINES = 80


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def working_tree_clean() -> bool | None:
    status = git_value(["status", "--porcelain"])
    if status is None:
        return None
    return status == ""


def load_matrix(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("commands")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected top-level 'commands' list")

    seen_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: each command entry must be an object")
        command_id = entry.get("id")
        if not isinstance(command_id, str) or not command_id:
            raise ValueError(f"{path}: command entry missing non-empty 'id'")
        if command_id in seen_ids:
            raise ValueError(f"{path}: duplicate command id {command_id!r}")
        seen_ids.add(command_id)
        command = entry.get("command")
        if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
            raise ValueError(f"{path}: {command_id}: 'command' must be a list of strings")
        tags = entry.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"{path}: {command_id}: 'tags' must be a list of strings")
    return data


def selected_entries(
    entries: list[dict[str, Any]],
    selected_tags: set[str],
    include_manual: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for entry in entries:
        tags = set(entry.get("tags", []))
        if entry.get("manual", False) and not include_manual:
            continue
        if selected_tags and tags.isdisjoint(selected_tags):
            continue
        selected.append(entry)
    return selected


def tail(text: str, lines: int = TAIL_LINES) -> str:
    split = text.splitlines()
    return "\n".join(split[-lines:])


def run_entry(entry: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    command = entry["command"]
    expected_exit = int(entry.get("expected_exit", 0))
    timeout_s = float(entry.get("timeout_s", 120))
    started = time.monotonic()

    base_result: dict[str, Any] = {
        "id": entry["id"],
        "name": entry.get("name", entry["id"]),
        "tags": entry.get("tags", []),
        "command": command,
        "expected_exit": expected_exit,
        "timeout_s": timeout_s,
    }

    if dry_run:
        return {
            **base_result,
            "status": "dry_run",
            "exit_code": None,
            "elapsed_s": 0.0,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            **base_result,
            "status": "timeout",
            "exit_code": None,
            "elapsed_s": round(elapsed, 3),
            "stdout_tail": tail(stdout),
            "stderr_tail": tail(stderr),
        }

    elapsed = time.monotonic() - started
    status = "passed" if completed.returncode == expected_exit else "failed"
    return {
        **base_result,
        "status": status,
        "exit_code": completed.returncode,
        "elapsed_s": round(elapsed, 3),
        "stdout_tail": tail(completed.stdout),
        "stderr_tail": tail(completed.stderr),
    }


def build_report(
    matrix_path: Path,
    matrix: dict[str, Any],
    entries: list[dict[str, Any]],
    selected_tags: set[str],
    dry_run: bool,
) -> dict[str, Any]:
    started_at = utc_now()
    results = [run_entry(entry, dry_run=dry_run) for entry in entries]
    completed_at = utc_now()
    failed = [result for result in results if result["status"] not in {"passed", "dry_run"}]
    return {
        "report_version": 1,
        "matrix_path": str(matrix_path.relative_to(ROOT) if matrix_path.is_relative_to(ROOT) else matrix_path),
        "matrix_version": matrix.get("matrix_version"),
        "suite": matrix.get("suite"),
        "description": matrix.get("description"),
        "selected_tags": sorted(selected_tags),
        "dry_run": dry_run,
        "started_at": started_at,
        "completed_at": completed_at,
        "repo": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["branch", "--show-current"]),
            "remote_origin": git_value(["config", "--get", "remote.origin.url"]),
            "working_tree_clean": working_tree_clean(),
        },
        "summary": {
            "total": len(results),
            "passed": sum(1 for result in results if result["status"] == "passed"),
            "failed": len(failed),
            "dry_run": sum(1 for result in results if result["status"] == "dry_run"),
        },
        "results": results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# STRIX Test Matrix Report",
        "",
        f"- Suite: `{report.get('suite')}`",
        f"- Commit: `{report['repo'].get('commit')}`",
        f"- Branch: `{report['repo'].get('branch')}`",
        f"- Working tree clean: `{report['repo'].get('working_tree_clean')}`",
        f"- Started: `{report.get('started_at')}`",
        f"- Completed: `{report.get('completed_at')}`",
        f"- Selected tags: `{', '.join(report.get('selected_tags') or []) or 'all non-manual'}`",
        "",
        "| ID | Status | Elapsed | Tags |",
        "|---|---:|---:|---|",
    ]
    for result in report["results"]:
        lines.append(
            f"| `{result['id']}` | `{result['status']}` | `{result['elapsed_s']}` | "
            f"`{', '.join(result.get('tags', []))}` |"
        )
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")


def print_matrix(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        tags = ", ".join(entry.get("tags", []))
        manual = " manual" if entry.get("manual", False) else ""
        print(f"{entry['id']}{manual}: {entry.get('name', entry['id'])} [{tags}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--select", action="append", default=[], help="Run entries tagged with this value.")
    parser.add_argument("--include-manual", action="store_true", help="Include entries marked manual.")
    parser.add_argument("--dry-run", action="store_true", help="Build the report without executing commands.")
    parser.add_argument("--list", action="store_true", help="List selected matrix entries and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix_path = args.matrix if args.matrix.is_absolute() else ROOT / args.matrix
    output = args.output if args.output.is_absolute() else ROOT / args.output
    matrix = load_matrix(matrix_path)
    tags = set(args.select)
    entries = selected_entries(matrix["commands"], tags, include_manual=args.include_manual)

    if args.list:
        print_matrix(entries)
        return 0

    report = build_report(matrix_path, matrix, entries, tags, dry_run=args.dry_run)
    write_reports(report, output)
    print(output)

    if report["summary"]["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
