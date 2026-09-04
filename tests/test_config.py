from pathlib import Path

import pytest

from mujoco_lab.config import load_config


def test_load_humanoid_config() -> None:
    config = load_config(Path("configs/humanoid/humanoid_v5_ppo.yaml"))
    assert config.name == "humanoid-v5-ppo"
    assert config.task.family == "humanoid"
    assert config.task.env_id == "Humanoid-v5"
    assert config.algorithm.name == "ppo"
    assert config.runtime.n_envs == 4
    assert config.runtime.device == "auto"


def test_invalid_family_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: bad\ntask:\n  family: spaceship\n  env_id: X\nalgorithm: {}\nruntime: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="task.family"):
        load_config(path)


def test_invalid_runtime_device_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad-device.yaml"
    path.write_text(
        "name: bad-device\n"
        "task:\n  family: humanoid\n  env_id: Humanoid-v5\n"
        "algorithm: {}\n"
        "runtime:\n  device: quantum\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runtime.device"):
        load_config(path)
