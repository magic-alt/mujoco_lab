from __future__ import annotations

import math
import platform
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any

import numpy as np
from gymnasium.spaces import Box

from mujoco_lab.config import ExperimentConfig
from mujoco_lab.training.vec_env import make_sb3_vec_env


@dataclass(frozen=True)
class VecEnvBenchmarkResult:
    backend: str
    n_envs: int
    startup_seconds: float
    vector_steps: int
    transitions: int
    elapsed_seconds: float
    transitions_per_second: float


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _zero_actions(action_space: Any, n_envs: int) -> np.ndarray:
    if not isinstance(action_space, Box):
        raise ValueError("VecEnv benchmark currently requires a continuous Box action space")
    return np.zeros((n_envs, *action_space.shape), dtype=action_space.dtype)


def benchmark_vec_envs(
    config: ExperimentConfig,
    *,
    env_counts: Iterable[int],
    backends: Iterable[str],
    transitions: int,
    warmup_steps: int,
) -> dict[str, Any]:
    """Measure startup and steady-state environment throughput without policy compute."""

    counts = tuple(dict.fromkeys(int(value) for value in env_counts))
    requested_backends = tuple(dict.fromkeys(value.lower() for value in backends))
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("env_counts must contain positive integers")
    if not requested_backends or any(
        value not in {"dummy", "subproc"} for value in requested_backends
    ):
        raise ValueError("backends must contain only: dummy, subproc")
    if transitions <= 0:
        raise ValueError("transitions must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for n_envs in counts:
        for backend in requested_backends:
            if backend == "subproc" and n_envs < 2:
                skipped.append(
                    {
                        "backend": backend,
                        "n_envs": n_envs,
                        "reason": "SubprocVecEnv requires at least two environments",
                    }
                )
                continue

            startup_start = time.perf_counter()
            vec_env, diagnostics = make_sb3_vec_env(
                config,
                n_envs=n_envs,
                backend=backend,
            )
            try:
                vec_env.reset()
                startup_seconds = time.perf_counter() - startup_start
                actions = _zero_actions(vec_env.action_space, n_envs)
                for _ in range(warmup_steps):
                    vec_env.step(actions)

                vector_steps = max(1, math.ceil(transitions / n_envs))
                measured_transitions = vector_steps * n_envs
                step_start = time.perf_counter()
                for _ in range(vector_steps):
                    vec_env.step(actions)
                elapsed_seconds = time.perf_counter() - step_start
            finally:
                vec_env.close()

            result = VecEnvBenchmarkResult(
                backend=diagnostics["resolved"],
                n_envs=n_envs,
                startup_seconds=startup_seconds,
                vector_steps=vector_steps,
                transitions=measured_transitions,
                elapsed_seconds=elapsed_seconds,
                transitions_per_second=measured_transitions / elapsed_seconds,
            )
            results.append(asdict(result))

    return {
        "benchmark": "vec-env",
        "experiment": config.name,
        "task": {
            "family": config.task.family,
            "env_id": config.task.env_id,
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in ("mujoco", "gymnasium", "stable-baselines3", "numpy")
        },
        "requested_transitions_per_case": transitions,
        "warmup_vector_steps": warmup_steps,
        "results": results,
        "skipped": skipped,
    }
