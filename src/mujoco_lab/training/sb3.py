from __future__ import annotations

import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from mujoco_lab.config import ExperimentConfig
from mujoco_lab.envs import make_env


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def resolve_torch_device(requested: str = "auto") -> tuple[str, dict[str, Any]]:
    """Resolve the training accelerator with GPU-first semantics.

    auto means CUDA first, then Apple MPS, then CPU. An explicitly requested
    accelerator fails fast when it is unavailable instead of silently falling back.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Training requires PyTorch via: uv sync --extra train") from exc

    requested = requested.lower()
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())

    if requested == "auto":
        resolved = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    elif requested == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "runtime.device='cuda' was requested, but torch.cuda.is_available() is false. "
                "Check the NVIDIA driver and the PyTorch build installed in this uv environment."
            )
        resolved = "cuda"
    elif requested == "mps":
        if not mps_available:
            raise RuntimeError(
                "runtime.device='mps' was requested, but the PyTorch MPS backend is unavailable."
            )
        resolved = "mps"
    elif requested == "cpu":
        resolved = "cpu"
    else:
        raise ValueError("requested device must be one of: auto, cuda, mps, cpu")

    cuda_devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [int(properties.major), int(properties.minor)],
                }
            )

    diagnostics: dict[str, Any] = {
        "requested": requested,
        "resolved": resolved,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_available": bool(torch.backends.cudnn.is_available()),
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cuda_devices": cuda_devices,
        "mps_available": mps_available,
    }
    return resolved, diagnostics


def _print_device_report(diagnostics: dict[str, Any]) -> None:
    resolved = diagnostics["resolved"]
    print("mujoco_lab accelerator report")
    print(f"- requested: {diagnostics['requested']}")
    print(f"- resolved: {resolved}")
    print(f"- torch: {diagnostics['torch_version']}")
    print(f"- CUDA available: {diagnostics['cuda_available']}")
    print(f"- PyTorch CUDA runtime: {diagnostics['torch_cuda_version']}")
    if diagnostics["cuda_devices"]:
        for device in diagnostics["cuda_devices"]:
            gib = float(device["total_memory_bytes"]) / (1024**3)
            capability = ".".join(str(value) for value in device["compute_capability"])
            print(
                f"- cuda:{device['index']}: {device['name']} "
                f"({gib:.1f} GiB, compute capability {capability})"
            )
    if resolved == "cpu" and diagnostics["requested"] == "auto":
        print("- fallback: no CUDA or MPS accelerator is available to this PyTorch environment")


def train_ppo(config: ExperimentConfig) -> Path:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import VecNormalize
    except ImportError as exc:
        raise RuntimeError("Training requires: uv sync --extra train") from exc

    device, device_diagnostics = resolve_torch_device(config.runtime.device)
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

    vec_env = make_vec_env(
        lambda: make_env(config),
        n_envs=config.runtime.n_envs,
        seed=config.algorithm.seed,
    )
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
        _write_metadata(output_dir, config, device_diagnostics)
    finally:
        vec_env.close()

    return Path(f"{model_path}.zip")


def _write_metadata(
    output_dir: Path,
    config: ExperimentConfig,
    device_diagnostics: dict[str, Any],
) -> None:
    packages = ["mujoco", "gymnasium", "stable-baselines3", "torch", "numpy", "robosuite"]
    payload: dict[str, Any] = {
        "experiment": config.name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in packages},
        "accelerator": device_diagnostics,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
