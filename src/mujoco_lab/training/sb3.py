from __future__ import annotations

import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from mujoco_lab.config import ExperimentConfig
from mujoco_lab.runtime import WORKLOAD_SB3_PPO_MLP_NATIVE, resolve_torch_device
from mujoco_lab.training.vec_env import make_sb3_vec_env


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _print_device_report(diagnostics: dict[str, Any]) -> None:
    print("mujoco_lab accelerator report")
    print(f"- requested: {diagnostics['requested']}")
    print(f"- workload: {diagnostics['workload']}")
    print(f"- resolved: {diagnostics['resolved']}")
    print(f"- selection: {diagnostics['selection_reason']}")
    print(f"- torch: {diagnostics['torch_version']}")
    print(f"- CUDA available: {diagnostics['cuda_available']}")
    print(f"- PyTorch CUDA runtime: {diagnostics['torch_cuda_version']}")
    for device in diagnostics["cuda_devices"]:
        gib = float(device["total_memory_bytes"]) / (1024**3)
        capability = ".".join(str(value) for value in device["compute_capability"])
        print(
            f"- cuda:{device['index']}: {device['name']} "
            f"({gib:.1f} GiB, compute capability {capability})"
        )


def _print_vec_env_report(diagnostics: dict[str, Any]) -> None:
    print("mujoco_lab vector-environment report")
    print(f"- requested: {diagnostics['requested']}")
    print(f"- resolved: {diagnostics['resolved']}")
    print(f"- environments: {diagnostics['n_envs']}")
    print(f"- selection: {diagnostics['selection_reason']}")


def train_ppo(config: ExperimentConfig) -> Path:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import VecNormalize
    except ImportError as exc:
        raise RuntimeError("Training requires: uv sync --extra train") from exc

    device, device_diagnostics = resolve_torch_device(
        config.runtime.device,
        workload=WORKLOAD_SB3_PPO_MLP_NATIVE,
    )
    _print_device_report(device_diagnostics)

    output_dir = Path(config.runtime.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / "tensorboard"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )

    vec_env, vec_env_diagnostics = make_sb3_vec_env(config)
    _print_vec_env_report(vec_env_diagnostics)
    if config.runtime.normalize_obs or config.runtime.normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=config.runtime.normalize_obs,
            norm_reward=config.runtime.normalize_reward,
        )

    save_freq = max(config.runtime.checkpoint_freq // config.runtime.n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoint_dir),
        name_prefix=config.name,
        save_vecnormalize=True,
    )

    algo = config.algorithm
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=algo.learning_rate,
        n_steps=algo.n_steps,
        batch_size=algo.batch_size,
        gamma=algo.gamma,
        gae_lambda=algo.gae_lambda,
        clip_range=algo.clip_range,
        ent_coef=algo.ent_coef,
        policy_kwargs={"net_arch": list(algo.net_arch)},
        seed=algo.seed,
        tensorboard_log=str(tensorboard_dir),
        device=device,
        verbose=1,
    )
    try:
        model.learn(total_timesteps=algo.total_timesteps, callback=checkpoint_callback)
        model_path = output_dir / "model"
        model.save(model_path)
        if isinstance(vec_env, VecNormalize):
            vec_env.save(output_dir / "vecnormalize.pkl")
        _write_metadata(
            output_dir,
            config,
            device_diagnostics,
            vec_env_diagnostics,
        )
    finally:
        vec_env.close()

    return Path(f"{model_path}.zip")


def _write_metadata(
    output_dir: Path,
    config: ExperimentConfig,
    device_diagnostics: dict[str, Any],
    vec_env_diagnostics: dict[str, Any],
) -> None:
    packages = ["mujoco", "gymnasium", "stable-baselines3", "torch", "numpy", "robosuite"]
    payload: dict[str, Any] = {
        "experiment": config.name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in packages},
        "accelerator": device_diagnostics,
        "vector_environment": vec_env_diagnostics,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
