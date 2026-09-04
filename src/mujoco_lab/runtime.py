from __future__ import annotations

from typing import Any

WORKLOAD_GENERIC = "generic"
WORKLOAD_SB3_PPO_MLP_NATIVE = "sb3-ppo-mlp-native-mujoco"


def select_auto_device(
    workload: str,
    *,
    cuda_available: bool,
    mps_available: bool,
) -> tuple[str, str]:
    """Choose an automatic torch device for a known workload."""

    if workload == WORKLOAD_SB3_PPO_MLP_NATIVE:
        return (
            "cpu",
            "SB3 PPO with MlpPolicy and native MuJoCo is CPU-preferred because the "
            "simulator remains on CPU and the policy network is small; set device explicitly "
            "to override this workload policy",
        )
    if workload != WORKLOAD_GENERIC:
        raise ValueError(f"unsupported workload profile: {workload}")
    if cuda_available:
        return "cuda", "CUDA is the hardware-preferred torch accelerator"
    if mps_available:
        return "mps", "Apple MPS is the hardware-preferred torch accelerator"
    return "cpu", "no CUDA or MPS accelerator is available to this PyTorch environment"


def resolve_torch_device(
    requested: str = "auto",
    *,
    workload: str = WORKLOAD_GENERIC,
) -> tuple[str, dict[str, Any]]:
    """Resolve a torch device while keeping hardware and workload policy separate."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Training requires PyTorch via: uv sync --extra train") from exc

    requested = requested.lower()
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())

    if requested == "auto":
        resolved, selection_reason = select_auto_device(
            workload,
            cuda_available=cuda_available,
            mps_available=mps_available,
        )
    elif requested == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "runtime.device='cuda' was requested, but torch.cuda.is_available() is false. "
                "Check the NVIDIA driver and the PyTorch build installed in this uv environment."
            )
        resolved = "cuda"
        selection_reason = "CUDA was explicitly requested"
    elif requested == "mps":
        if not mps_available:
            raise RuntimeError(
                "runtime.device='mps' was requested, but the PyTorch MPS backend is unavailable."
            )
        resolved = "mps"
        selection_reason = "Apple MPS was explicitly requested"
    elif requested == "cpu":
        resolved = "cpu"
        selection_reason = "CPU was explicitly requested"
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
        "workload": workload,
        "selection_reason": selection_reason,
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


def resolve_vec_env_backend(
    requested: str,
    *,
    n_envs: int,
) -> tuple[str, dict[str, Any]]:
    """Resolve the SB3 vector-environment implementation."""

    requested = requested.lower()
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if requested == "auto":
        resolved = "dummy"
        reason = (
            "auto keeps SB3's conservative DummyVecEnv baseline; benchmark this machine before "
            "selecting SubprocVecEnv because multiprocessing IPC can outweigh parallelism"
        )
    elif requested == "dummy":
        resolved = "dummy"
        reason = "DummyVecEnv was explicitly requested"
    elif requested == "subproc":
        if n_envs < 2:
            raise ValueError("SubprocVecEnv requires n_envs >= 2")
        resolved = "subproc"
        reason = "SubprocVecEnv was explicitly requested"
    else:
        raise ValueError("vec env backend must be one of: auto, dummy, subproc")

    return resolved, {
        "requested": requested,
        "resolved": resolved,
        "n_envs": n_envs,
        "selection_reason": reason,
    }
