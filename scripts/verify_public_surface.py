#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check that the public repository does not reintroduce private-boundary leaks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BANNED_TEXT = [
    "repo-split",
    "PUBLIC_PRIVATE_MAP",
    "MOVE_LOG",
    "OPEN_QUESTIONS",
    "strix-private",
    "BUSL-1.1",
    "BSL-1.1",
    "Apache-2.0 AND",
    "Military R&D evaluators",
]

BANNED_PATH_PARTS = [
    "Project_Docs/bug_hunt/",
    "demo/video/",
]

BANNED_PATHS = {
    "LICENSE-BSL",
    "LICENSING.md",
    "demo/video_script.md",
    "sim/scenarios/contested_strike.yaml",
}

SKIP_TEXT_SCAN = {
    "scripts/verify_public_surface.py",
}


def git_files() -> list[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [item for item in output.decode().split("\0") if item]


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def main() -> int:
    failures: list[str] = []

    for rel in git_files():
        normalized = rel.replace("\\", "/")
        if normalized in BANNED_PATHS:
            failures.append(f"banned path is tracked: {normalized}")
        for banned_part in BANNED_PATH_PARTS:
            if banned_part in normalized:
                failures.append(f"banned path segment is tracked: {normalized}")

        if normalized in SKIP_TEXT_SCAN:
            continue

        text = read_text(ROOT / rel)
        if text is None:
            continue

        for banned in BANNED_TEXT:
            if banned in text:
                failures.append(f"{normalized}: contains banned marker {banned!r}")

    if failures:
        print("Public surface verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Public surface verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
