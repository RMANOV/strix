#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Install local STRIX git hooks for public-surface and upstream safety checks."""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOOK_NAMES = ("pre-commit", "pre-push")


def default_git_dir() -> Path:
    output = subprocess.check_output(["git", "rev-parse", "--git-dir"], cwd=ROOT, text=True).strip()
    git_dir = Path(output)
    if not git_dir.is_absolute():
        git_dir = ROOT / git_dir
    return git_dir


def hook_body(name: str) -> str:
    if name == "pre-commit":
        return """#!/bin/sh
set -eu

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

python_cmd="${PYTHON:-python3}"
if ! command -v "$python_cmd" >/dev/null 2>&1; then
    python_cmd=python
fi

"$python_cmd" scripts/verify_public_surface.py

if git diff --cached --name-only | grep -Eq '(^|/)strix-release-authority\\.json$'; then
    echo "Refusing to commit local STRIX release authority state." >&2
    exit 1
fi
"""
    if name == "pre-push":
        return """#!/bin/sh
set -eu

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

python_cmd="${PYTHON:-python3}"
if ! command -v "$python_cmd" >/dev/null 2>&1; then
    python_cmd=python
fi

"$python_cmd" scripts/verify_public_surface.py

while read local_ref local_oid remote_ref remote_oid; do
    case "$remote_ref" in
        refs/heads/main|refs/heads/master)
            if [ "${STRIX_ALLOW_MAIN_PUSH:-}" != "1" ]; then
                echo "Direct push to protected branch blocked by local STRIX hook." >&2
                echo "Push a PR branch, or set STRIX_ALLOW_MAIN_PUSH=1 for an intentional maintainer release." >&2
                exit 1
            fi
            ;;
    esac
done
"""
    raise ValueError(f"unknown hook: {name}")


def install_hooks(git_dir: Path, force: bool) -> list[Path]:
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name in HOOK_NAMES:
        hook_path = hooks_dir / name
        if hook_path.exists() and not force:
            raise SystemExit(f"{hook_path} already exists; pass --force to replace it")

        hook_path.write_text(hook_body(name), encoding="utf-8")
        current_mode = hook_path.stat().st_mode
        hook_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        written.append(hook_path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-dir", type=Path, help="Override the target .git directory")
    parser.add_argument("--force", action="store_true", help="Replace existing local hooks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    git_dir = args.git_dir if args.git_dir is not None else default_git_dir()
    written = install_hooks(git_dir, args.force)
    for hook_path in written:
        print(hook_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"git command failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
