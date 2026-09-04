from __future__ import annotations

import gymnasium as gym

from mujoco_lab.config import ExperimentConfig


def make_bimanual_env(config: ExperimentConfig, render_mode: str | None = None) -> gym.Env:
    """Create a robosuite two-arm task and adapt it to the Gymnasium API."""
    try:
        import robosuite
        from robosuite.wrappers import GymWrapper
    except ImportError as exc:
        raise RuntimeError(
            "Bimanual tasks require the optional dependency: "
            "uv sync --extra train --extra bimanual"
        ) from exc

    kwargs = dict(config.task.kwargs)
    has_renderer = render_mode == "human"
    kwargs.setdefault("has_renderer", has_renderer)
    kwargs.setdefault("has_offscreen_renderer", False)
    kwargs.setdefault("use_camera_obs", False)

    env = robosuite.make(env_name=config.task.env_id, **kwargs)
    wrapped = GymWrapper(env, flatten_obs=True)
    if wrapped.metadata is None:
        wrapped.metadata = {}
    return wrapped
