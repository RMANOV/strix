#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Generate local STRIX release authority state and release manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTHORITY_PATH = ROOT / ".git" / "strix-release-authority.json"
OFFICIAL_SOURCE_REPO = "https://github.com/RMANOV/strix"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def machine_id_hash() -> str | None:
    candidates = [Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")]
    for candidate in candidates:
        if not candidate.exists():
            continue
        raw = candidate.read_text(encoding="utf-8").strip()
        if raw:
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return None


def authority_payload() -> dict[str, Any]:
    return {
        "authority_id": f"strix-local-{uuid.uuid4()}",
        "created_at": utc_now(),
        "hostname": platform.node(),
        "machine_id_sha256": machine_id_hash(),
        "note": "Local-only maintainer release authority state. Do not commit.",
    }


def init_local_authority(path: Path, overwrite: bool) -> int:
    if path.exists() and not overwrite:
        print(f"release authority already exists: {path}", file=sys.stderr)
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(authority_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0


def load_authority(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def working_tree_clean() -> bool:
    return run_git(["status", "--porcelain"]) == ""


def build_manifest(authority_path: Path, allow_missing_authority: bool) -> dict[str, Any]:
    authority = load_authority(authority_path)
    if authority is None and not allow_missing_authority:
        raise SystemExit(
            f"missing local release authority at {authority_path}; "
            "run `python scripts/strix_release_manifest.py init-local-authority` first"
        )

    manifest = {
        "manifest_version": 1,
        "project": "STRIX",
        "official_source_repo": OFFICIAL_SOURCE_REPO,
        "repo_remote_origin": run_git(["config", "--get", "remote.origin.url"]),
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "working_tree_clean": working_tree_clean(),
        "created_at": utc_now(),
        "authority": {
            "authority_id": None,
            "machine_id_sha256": None,
            "public_key_fingerprint": os.environ.get("STRIX_RELEASE_PUBLIC_KEY_FINGERPRINT"),
        },
    }

    if authority is not None:
        manifest["authority"]["authority_id"] = authority.get("authority_id")
        manifest["authority"]["machine_id_sha256"] = authority.get("machine_id_sha256")

    return manifest


def write_manifest(manifest: dict[str, Any], output: Path | None) -> None:
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(encoded)
        return
    output.write_text(encoded, encoding="utf-8")
    print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-local-authority")
    init_parser.add_argument("--authority-path", type=Path, default=DEFAULT_AUTHORITY_PATH)
    init_parser.add_argument("--overwrite", action="store_true")

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--authority-path", type=Path, default=DEFAULT_AUTHORITY_PATH)
    manifest_parser.add_argument("--allow-missing-authority", action="store_true")
    manifest_parser.add_argument("--output", type=Path)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "init-local-authority":
        return init_local_authority(args.authority_path, args.overwrite)
    if args.command == "manifest":
        manifest = build_manifest(args.authority_path, args.allow_missing_authority)
        write_manifest(manifest, args.output)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
