from __future__ import annotations

from typing import Any

from mujoco_lab.config import ExperimentConfig
from mujoco_lab.envs import make_env
from mujoco_lab.runtime import resolve_vec_env_backend


def make_sb3_vec_env(
    config: ExperimentConfig,
    *,
    n_envs: int | None = None,
    backend: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build a monitored SB3 VecEnv from the common environment contract."""

    try:
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:
        raise RuntimeError("Vector environments require: uv sync --extra train") from exc

    resolved_n_envs = config.runtime.n_envs if n_envs is None else n_envs
    requested_backend = config.runtime.vec_env_backend if backend is None else backend
    resolved_backend, diagnostics = resolve_vec_env_backend(
        requested_backend,
        n_envs=resolved_n_envs,
    )
    vec_env_cls = DummyVecEnv if resolved_backend == "dummy" else SubprocVecEnv
    vec_env = make_vec_env(
        lambda: make_env(config),
        n_envs=resolved_n_envs,
        seed=config.algorithm.seed,
        vec_env_cls=vec_env_cls,
    )
    return vec_env, diagnostics
