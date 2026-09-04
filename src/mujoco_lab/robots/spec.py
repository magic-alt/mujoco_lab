from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotModelSpec:
    """Pinned upstream robot-model contract.

    The repository stores metadata and validation rules only. Third-party MJCF and
    mesh assets are resolved into a local cache at runtime.
    """

    name: str
    source_repository: str
    revision: str
    model_subdir: str
    model_file: str
    scene_file: str
    license_id: str
    license_file: str = "LICENSE"
    readme_file: str = "README.md"
    expected_joint_count: int | None = None
    expected_actuator_count: int | None = None
    required_sites: tuple[str, ...] = ()
    required_bodies: tuple[str, ...] = ()
    required_controlled_joints: tuple[str, ...] = ()

    @property
    def source_url(self) -> str:
        return f"https://github.com/{self.source_repository}"

    def raw_url(self, relative_path: str) -> str:
        relative_path = relative_path.lstrip("/")
        return (
            f"https://raw.githubusercontent.com/{self.source_repository}/"
            f"{self.revision}/{self.model_subdir}/{relative_path}"
        )
