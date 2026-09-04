import pytest

pytest.importorskip("torch")

from mujoco_lab.training.sb3 import resolve_torch_device


def test_auto_device_prefers_available_accelerator() -> None:
    import torch

    resolved, diagnostics = resolve_torch_device("auto")

    if torch.cuda.is_available():
        assert resolved == "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        assert resolved == "mps"
    else:
        assert resolved == "cpu"

    assert diagnostics["requested"] == "auto"
    assert diagnostics["resolved"] == resolved


def test_cpu_device_can_be_forced() -> None:
    resolved, diagnostics = resolve_torch_device("cpu")
    assert resolved == "cpu"
    assert diagnostics["resolved"] == "cpu"


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="requested device"):
        resolve_torch_device("quantum")
