from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TaskConfig:
    family: str
    env_id: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str = "ppo"
    total_timesteps: int = 1_000_000
    seed: int = 42
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    net_arch: tuple[int, ...] = (256, 256)


@dataclass(frozen=True)
class RuntimeConfig:
    n_envs: int = 1
    normalize_obs: bool = True
    normalize_reward: bool = True
    checkpoint_freq: int = 100_000
    output_dir: str = "runs/default"
    device: str = "auto"
    vec_env_backend: str = "auto"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    task: TaskConfig
    algorithm: AlgorithmConfig
    runtime: RuntimeConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_mapping(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a mapping")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = _require_mapping(raw, "root")

    task_raw = _require_mapping(raw.get("task"), "task")
    algo_raw = _require_mapping(raw.get("algorithm", {}), "algorithm")
    runtime_raw = _require_mapping(raw.get("runtime", {}), "runtime")

    task = TaskConfig(
        family=str(task_raw.get("family", "")),
        env_id=str(task_raw.get("env_id", "")),
        kwargs=dict(task_raw.get("kwargs", {})),
    )
    algorithm = AlgorithmConfig(
        name=str(algo_raw.get("name", "ppo")),
        total_timesteps=int(algo_raw.get("total_timesteps", 1_000_000)),
        seed=int(algo_raw.get("seed", 42)),
        learning_rate=float(algo_raw.get("learning_rate", 3e-4)),
        n_steps=int(algo_raw.get("n_steps", 2048)),
        batch_size=int(algo_raw.get("batch_size", 64)),
        gamma=float(algo_raw.get("gamma", 0.99)),
        gae_lambda=float(algo_raw.get("gae_lambda", 0.95)),
        clip_range=float(algo_raw.get("clip_range", 0.2)),
        ent_coef=float(algo_raw.get("ent_coef", 0.0)),
        net_arch=tuple(int(x) for x in algo_raw.get("net_arch", [256, 256])),
    )
    runtime = RuntimeConfig(
        n_envs=int(runtime_raw.get("n_envs", 1)),
        normalize_obs=bool(runtime_raw.get("normalize_obs", True)),
        normalize_reward=bool(runtime_raw.get("normalize_reward", True)),
        checkpoint_freq=int(runtime_raw.get("checkpoint_freq", 100_000)),
        output_dir=str(runtime_raw.get("output_dir", f"runs/{raw.get('name', 'default')}")),
        device=str(runtime_raw.get("device", "auto")).lower(),
        vec_env_backend=str(runtime_raw.get("vec_env_backend", "auto")).lower(),
    )

    if not raw.get("name"):
        raise ValueError("'name' is required")
    if task.family not in {"humanoid", "bimanual"}:
        raise ValueError("task.family must be 'humanoid' or 'bimanual'")
    if not task.env_id:
        raise ValueError("task.env_id is required")
    if algorithm.name != "ppo":
        raise ValueError("bootstrap trainer currently supports algorithm.name='ppo' only")
    if algorithm.total_timesteps <= 0:
        raise ValueError("algorithm.total_timesteps must be positive")
    if runtime.n_envs <= 0:
        raise ValueError("runtime.n_envs must be positive")
    if algorithm.n_steps <= 0 or algorithm.batch_size <= 0:
        raise ValueError("algorithm.n_steps and batch_size must be positive")
    if runtime.device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("runtime.device must be one of: auto, cuda, mps, cpu")
    if runtime.vec_env_backend not in {"auto", "dummy", "subproc"}:
        raise ValueError("runtime.vec_env_backend must be one of: auto, dummy, subproc")
    if runtime.vec_env_backend == "subproc" and runtime.n_envs < 2:
        raise ValueError("runtime.vec_env_backend='subproc' requires runtime.n_envs >= 2")

    return ExperimentConfig(
        name=str(raw["name"]),
        task=task,
        algorithm=algorithm,
        runtime=runtime,
    )
