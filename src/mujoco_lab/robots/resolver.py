from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from mujoco_lab.robots.spec import RobotModelSpec


class AssetResolutionError(RuntimeError):
    """Raised when a pinned third-party robot model cannot be resolved safely."""


class AssetRevisionError(AssetResolutionError):
    """Raised when cache metadata does not match the requested pinned revision."""


@dataclass(frozen=True)
class ResolvedRobotModel:
    spec: RobotModelSpec
    root: Path
    model_path: Path
    scene_path: Path
    license_path: Path
    manifest_path: Path


def default_cache_root() -> Path:
    explicit = os.environ.get("MUJOCO_LAB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "mujoco_lab" / "cache"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "mujoco_lab"
    return Path.home() / ".cache" / "mujoco_lab"


def _safe_relative(path: str) -> str:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise AssetResolutionError(f"unsafe upstream asset path: {path!r}")
    return candidate.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_response(response: BinaryIO, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "mujoco-lab/robot-model-resolver"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:  # noqa: S310
            _copy_response(response, destination)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise AssetResolutionError(f"failed to download {url}: {exc}") from exc


def _manifest_payload(spec: RobotModelSpec, root: Path, files: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "robot": spec.name,
        "source_repository": spec.source_repository,
        "source_url": spec.source_url,
        "revision": spec.revision,
        "model_subdir": spec.model_subdir,
        "license_id": spec.license_id,
        "files": {path: _sha256(root / path) for path in sorted(files)},
    }


def _validate_manifest(spec: RobotModelSpec, target: Path, manifest_path: Path) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetResolutionError(
            f"invalid robot cache manifest at {manifest_path}; remove the cache and retry"
        ) from exc

    expected = {
        "robot": spec.name,
        "source_repository": spec.source_repository,
        "source_url": spec.source_url,
        "revision": spec.revision,
        "model_subdir": spec.model_subdir,
        "license_id": spec.license_id,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise AssetRevisionError(
                f"robot cache metadata mismatch for {spec.name}: expected {key}={value!r}, "
                f"found {manifest.get(key)!r}. Remove {target} or use --force."
            )

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise AssetResolutionError(f"robot cache manifest has no files: {manifest_path}")
    for relative_path, expected_hash in files.items():
        path = target / _safe_relative(str(relative_path))
        if not path.is_file():
            raise AssetResolutionError(f"cached robot asset is missing: {path}")
        if _sha256(path) != expected_hash:
            raise AssetResolutionError(
                f"cached robot asset hash mismatch: {path}; remove the cache and retry"
            )


def _discover_meshes(model_path: Path) -> list[str]:
    try:
        root = ET.parse(model_path).getroot()
    except ET.ParseError as exc:
        raise AssetResolutionError(f"invalid upstream MJCF: {model_path}") from exc

    compiler = root.find("compiler")
    meshdir = _safe_relative(compiler.get("meshdir", "assets") if compiler is not None else "assets")
    meshes: list[str] = []
    for mesh in root.findall("./asset/mesh"):
        file_name = mesh.get("file")
        if not file_name:
            continue
        meshes.append(_safe_relative(f"{meshdir}/{_safe_relative(file_name)}"))
    if not meshes:
        raise AssetResolutionError(f"no mesh assets discovered in {model_path}")
    return sorted(set(meshes))


def resolve_robot_model(
    spec: RobotModelSpec,
    *,
    cache_root: str | Path | None = None,
    offline: bool = False,
    force: bool = False,
) -> ResolvedRobotModel:
    """Resolve a pinned upstream MJCF model into a revision-scoped local cache."""

    cache = Path(cache_root).expanduser() if cache_root is not None else default_cache_root()
    target = cache / "models" / spec.name / spec.revision
    manifest_path = target / "manifest.json"

    if force and target.exists():
        shutil.rmtree(target)

    if target.exists():
        if not manifest_path.is_file():
            raise AssetRevisionError(
                f"robot cache exists without revision metadata: {target}; remove it or use --force"
            )
        _validate_manifest(spec, target, manifest_path)
        return ResolvedRobotModel(
            spec=spec,
            root=target,
            model_path=target / spec.model_file,
            scene_path=target / spec.scene_file,
            license_path=target / spec.license_file,
            manifest_path=manifest_path,
        )

    if offline:
        raise AssetResolutionError(
            f"{spec.name} revision {spec.revision} is not cached at {target}; "
            "rerun without --offline while network access is available"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = target.parent
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{spec.revision[:12]}-", dir=temp_parent))
    downloaded: list[str] = []
    try:
        primary_files = [spec.model_file, spec.scene_file, spec.license_file, spec.readme_file]
        for relative_path in primary_files:
            relative_path = _safe_relative(relative_path)
            _download(spec.raw_url(relative_path), temp_dir / relative_path)
            downloaded.append(relative_path)

        for relative_path in _discover_meshes(temp_dir / spec.model_file):
            _download(spec.raw_url(relative_path), temp_dir / relative_path)
            downloaded.append(relative_path)

        payload = _manifest_payload(spec, temp_dir, downloaded)
        (temp_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temp_dir.replace(target)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    _validate_manifest(spec, target, manifest_path)
    return ResolvedRobotModel(
        spec=spec,
        root=target,
        model_path=target / spec.model_file,
        scene_path=target / spec.scene_file,
        license_path=target / spec.license_file,
        manifest_path=manifest_path,
    )
