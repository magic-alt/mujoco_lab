import hashlib
import json
from pathlib import Path

import pytest

from mujoco_lab.robots.g1 import G1_CONTROLLED_JOINTS, G1_SPEC
from mujoco_lab.robots.resolver import (
    AssetResolutionError,
    AssetRevisionError,
    resolve_robot_model,
)


def _cache_target(cache_root: Path) -> Path:
    return cache_root / "models" / G1_SPEC.name / G1_SPEC.revision


def _manifest_metadata() -> dict[str, object]:
    return {
        "schema_version": 1,
        "robot": G1_SPEC.name,
        "source_repository": G1_SPEC.source_repository,
        "source_url": G1_SPEC.source_url,
        "revision": G1_SPEC.revision,
        "model_subdir": G1_SPEC.model_subdir,
        "license_id": G1_SPEC.license_id,
    }


def test_g1_spec_is_pinned_and_machine_checkable() -> None:
    assert len(G1_SPEC.revision) == 40
    assert G1_SPEC.expected_joint_count == 29
    assert G1_SPEC.expected_actuator_count == 29
    assert len(G1_CONTROLLED_JOINTS) == 15
    assert {"left_foot", "right_foot"}.issubset(G1_SPEC.required_sites)
    assert G1_SPEC.source_repository == "google-deepmind/mujoco_menagerie"
    assert G1_SPEC.source_url == "https://github.com/google-deepmind/mujoco_menagerie"
    assert G1_SPEC.license_id == "BSD-3-Clause"


def test_offline_resolution_requires_existing_cache(tmp_path: Path) -> None:
    with pytest.raises(AssetResolutionError, match="rerun without --offline"):
        resolve_robot_model(G1_SPEC, cache_root=tmp_path, offline=True)


def test_revision_metadata_mismatch_is_rejected(tmp_path: Path) -> None:
    target = _cache_target(tmp_path)
    target.mkdir(parents=True)
    manifest = _manifest_metadata()
    manifest["revision"] = "0" * 40
    manifest["files"] = {"g1.xml": "unused"}
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AssetRevisionError, match="metadata mismatch"):
        resolve_robot_model(G1_SPEC, cache_root=tmp_path, offline=True)


def test_cached_asset_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    target = _cache_target(tmp_path)
    target.mkdir(parents=True)
    model_path = target / G1_SPEC.model_file
    model_path.write_text("<mujoco/>", encoding="utf-8")
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert actual_hash != "0" * 64

    manifest = _manifest_metadata()
    manifest["files"] = {G1_SPEC.model_file: "0" * 64}
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AssetResolutionError, match="hash mismatch"):
        resolve_robot_model(G1_SPEC, cache_root=tmp_path, offline=True)
