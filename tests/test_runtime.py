import pytest

from mujoco_lab.runtime import (
    WORKLOAD_GENERIC,
    WORKLOAD_SB3_PPO_MLP_NATIVE,
    resolve_vec_env_backend,
    select_auto_device,
)


def test_native_sb3_mlp_auto_prefers_cpu_even_with_cuda() -> None:
    device, reason = select_auto_device(
        WORKLOAD_SB3_PPO_MLP_NATIVE,
        cuda_available=True,
        mps_available=False,
    )
    assert device == "cpu"
    assert "CPU-preferred" in reason


def test_generic_auto_prefers_cuda_when_available() -> None:
    device, reason = select_auto_device(
        WORKLOAD_GENERIC,
        cuda_available=True,
        mps_available=False,
    )
    assert device == "cuda"
    assert "hardware-preferred" in reason


def test_vec_env_auto_keeps_dummy_baseline() -> None:
    backend, diagnostics = resolve_vec_env_backend("auto", n_envs=8)
    assert backend == "dummy"
    assert diagnostics["requested"] == "auto"
    assert diagnostics["resolved"] == "dummy"


def test_subproc_requires_multiple_envs() -> None:
    with pytest.raises(ValueError, match="requires n_envs >= 2"):
        resolve_vec_env_backend("subproc", n_envs=1)


def test_explicit_subproc_is_preserved() -> None:
    backend, diagnostics = resolve_vec_env_backend("subproc", n_envs=4)
    assert backend == "subproc"
    assert diagnostics["n_envs"] == 4
