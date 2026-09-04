from __future__ import annotations

from pathlib import Path

import numpy as np

from mujoco_lab.config import ExperimentConfig
from mujoco_lab.envs import make_env


def evaluate_ppo(
    config: ExperimentConfig,
    checkpoint: str | Path,
    episodes: int = 5,
    render: bool = True,
) -> list[float]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:
        raise RuntimeError("Evaluation requires: uv sync --extra train") from exc

    checkpoint = Path(checkpoint)
    render_mode = "human" if render else None
    vec_env = DummyVecEnv([lambda: make_env(config, render_mode=render_mode)])

    stats_path = Path(config.runtime.output_dir) / "vecnormalize.pkl"
    if stats_path.exists():
        vec_env = VecNormalize.load(stats_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(checkpoint, env=vec_env)
    returns: list[float] = []
    try:
        for _ in range(episodes):
            obs = vec_env.reset()
            done = np.array([False])
            episode_return = 0.0
            while not bool(done[0]):
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, done, _ = vec_env.step(action)
                episode_return += float(rewards[0])
            returns.append(episode_return)
    finally:
        vec_env.close()
    return returns
