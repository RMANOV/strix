#!/usr/bin/env python3
"""Check Cargo's public workspace from a tracked-source copy with no sibling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

EXPECTED = {
    "strix-core", "strix-auction", "strix-mesh", "strix-adapters", "strix-xai",
    "strix-swarm", "strix-python", "strix-playground", "strix-optimizer",
}


def verify(source: Path) -> None:
    source = source.resolve(strict=True)
    paths = subprocess.check_output(
        ["git", "-C", str(source), "ls-files", "-z"], timeout=15
    ).split(b"\0")
    with tempfile.TemporaryDirectory(prefix="strix-public-cargo-") as temporary:
        root = Path(temporary) / "strix"
        root.mkdir()
        for raw in paths:
            if not raw:
                continue
            relative = Path(raw.decode())
            original = source / relative
            if original.is_symlink():
                raise AssertionError(f"tracked symlink is not a standalone input: {relative}")
            if original.is_file():
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, destination)
        assert not (root.parent / "phi-sim").exists(), "private sibling in cold fixture"
        for feature_args in ([], ["--features", "temporal"], ["--all-features"]):
            command = ["cargo", "metadata", "--locked", "--offline", "--no-deps",
                       "--format-version", "1", *feature_args]
            result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=45)
            assert result.returncode == 0, (
                f"standalone public Cargo metadata failed ({feature_args or 'default'}):\n"
                f"{result.stderr}"
            )
            metadata = json.loads(result.stdout)
            packages = metadata["packages"]
            assert {p["name"] for p in packages} == EXPECTED, "public workspace members changed"
            assert set(metadata["workspace_members"]) == {p["id"] for p in packages}
            for package in packages:
                assert Path(package["manifest_path"]).is_relative_to(root)
                for dependency in package["dependencies"]:
                    assert dependency["name"] != "phi-sim", "private Phi dependency leaked into public Cargo"
                    if dependency.get("path"):
                        assert Path(dependency["path"]).is_relative_to(root), (
                            f"dependency path escapes public checkout: {dependency['name']}"
                        )
            swarm = next(p for p in packages if p["name"] == "strix-swarm")
            assert set(swarm["features"]) == {"default", "temporal", "gcbf"}, "public swarm feature boundary changed"
            expected_targets = {
                ("strix_swarm", "lib"), ("tick", "bench"),
                *((name, "test") for name in ["fault_injection", "fear_modulation", "feedback_stability",
                    "island_module_integration", "phi_sim_integration", "regression", "twenty_drone_integration"]),
            }
            assert {(t["name"], t["kind"][0]) for t in swarm["targets"]} == expected_targets, "swarm target lost or changed"
            playground = next(p for p in packages if p["name"] == "strix-playground")
            assert {(t["name"], t["kind"][0]) for t in playground["targets"]} == {
                ("strix_playground", "lib"), ("presets_smoke", "test")
            }, "playground library or test omitted"
            optimizer = next(p for p in packages if p["name"] == "strix-optimizer")
            assert {(t["name"], t["kind"][0]) for t in optimizer["targets"]} == {
                ("strix_optimizer", "lib"), ("strix-optimize", "bin")
            }, "optimizer library or binary omitted"
            print(f"PASS standalone Cargo boundary: {feature_args or 'default'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    verify(parser.parse_args().source)


if __name__ == "__main__":
    main()
