from __future__ import annotations

import gymnasium as gym

from mujoco_lab.config import ExperimentConfig


def make_humanoid_env(config: ExperimentConfig, render_mode: str | None = None) -> gym.Env:
    """Create the current Gymnasium humanoid baseline.

    A custom Menagerie G1/H1 environment will replace this as the primary locomotion
    tutorial in Phase 2. Keeping this function behind the common factory means the
    training code does not need to change when that happens.
    """
    return gym.make(
        config.task.env_id,
        render_mode=render_mode,
        **config.task.kwargs,
    )
