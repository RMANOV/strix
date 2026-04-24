# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


def _load_release_manifest_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "strix_release_manifest.py"
    spec = importlib.util.spec_from_file_location("strix_release_manifest", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_release_manifest_can_be_built_without_local_authority():
    module = _load_release_manifest_module()
    manifest = module.build_manifest(
        Path("/tmp/strix-release-authority-does-not-exist.json"),
        allow_missing_authority=True,
    )

    assert manifest["project"] == "STRIX"
    assert manifest["official_source_repo"] == "https://github.com/RMANOV/strix"
    assert manifest["commit"]
    assert manifest["authority"]["authority_id"] is None
