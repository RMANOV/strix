# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "install_local_git_hooks.py"


def load_hooks_module():
    spec = importlib.util.spec_from_file_location("install_local_git_hooks", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hook_bodies_run_public_surface_guard() -> None:
    hooks = load_hooks_module()

    for name in hooks.HOOK_NAMES:
        body = hooks.hook_body(name)
        assert "scripts/verify_public_surface.py" in body
        assert body.startswith("#!/bin/sh\nset -eu\n")


def test_pre_push_blocks_direct_main_without_explicit_override() -> None:
    hooks = load_hooks_module()

    body = hooks.hook_body("pre-push")
    assert "refs/heads/main" in body
    assert "STRIX_ALLOW_MAIN_PUSH" in body


def test_install_hooks_writes_executable_files(tmp_path: Path) -> None:
    hooks = load_hooks_module()
    git_dir = tmp_path / ".git"

    written = hooks.install_hooks(git_dir, force=False)

    assert {path.name for path in written} == {"pre-commit", "pre-push"}
    for hook_path in written:
        mode = hook_path.stat().st_mode
        assert mode & stat.S_IXUSR
        assert "scripts/verify_public_surface.py" in hook_path.read_text(encoding="utf-8")


def test_install_hooks_requires_force_for_existing_hooks(tmp_path: Path) -> None:
    hooks = load_hooks_module()
    git_dir = tmp_path / ".git"
    hooks.install_hooks(git_dir, force=False)

    try:
        hooks.install_hooks(git_dir, force=False)
    except SystemExit as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("expected SystemExit for existing hooks")

    hooks.install_hooks(git_dir, force=True)
