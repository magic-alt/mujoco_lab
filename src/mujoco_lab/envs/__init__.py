from __future__ import annotations

import gymnasium as gym

from mujoco_lab.config import ExperimentConfig
from mujoco_lab.envs.bimanual import make_bimanual_env
from mujoco_lab.envs.humanoid import make_humanoid_env


def make_env(config: ExperimentConfig, render_mode: str | None = None) -> gym.Env:
    """Create one Gymnasium-compatible environment from a checked-in config."""
    if config.task.family == "humanoid":
        return make_humanoid_env(config, render_mode=render_mode)
    if config.task.family == "bimanual":
        return make_bimanual_env(config, render_mode=render_mode)
    raise ValueError(f"Unsupported task family: {config.task.family}")
