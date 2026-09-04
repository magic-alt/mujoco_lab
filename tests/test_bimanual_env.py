from pathlib import Path

import numpy as np
import pytest

from mujoco_lab.config import load_config
from mujoco_lab.envs import make_env

pytest.importorskip("robosuite")


def test_two_arm_lift_env_smoke() -> None:
    config = load_config(Path("configs/bimanual/two_arm_lift_ppo.yaml"))
    env = make_env(config)
    try:
        obs, info = env.reset(seed=123)
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)
        assert np.isfinite(obs).all()

        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(obs).all()
        assert np.isfinite(float(reward))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()
