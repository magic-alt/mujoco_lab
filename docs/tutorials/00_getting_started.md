# 00 — Getting started

## 1. Install Python and uv

Use Python 3.11 or 3.12. Install `uv`, clone the repository, then create the core humanoid training environment:

```bash
uv sync --extra train --extra dev
uv run mujoco-lab doctor
```

For two-arm tasks:

```bash
uv sync --extra train --extra bimanual --extra dev
```

The `bimanual` extra intentionally constrains MuJoCo to the compatibility window required by robosuite (`>=3.3,<3.10` at the time this tutorial was validated). The core project itself only requires `mujoco>=3.3`, so humanoid/MJX work can move to newer MuJoCo releases independently.

If you actively develop both a newest-MuJoCo/MJX path and robosuite in parallel, prefer separate `uv` environments/worktrees rather than repeatedly upgrading and downgrading one environment. The checked-in extras make the dependency boundary explicit.

## 2. Validate MuJoCo without training

```bash
uv run mujoco-lab inspect-env configs/humanoid/humanoid_v5_ppo.yaml
```

This command performs exactly one reset and one random step. It exists to separate **simulation/environment failures** from **RL failures**.

## 3. Train the first humanoid baseline

```bash
uv run mujoco-lab train configs/humanoid/humanoid_v5_ppo.yaml
```

The default runtime device is `auto`, with explicit GPU-first semantics:

```text
CUDA -> Apple MPS -> CPU
```

Before SB3 starts, the trainer prints an accelerator report with the resolved device, PyTorch version, CUDA runtime, visible CUDA devices and the CPU fallback reason when no accelerator is visible. The same information is persisted in `runs/<experiment>/metadata.json`.

To diagnose the current environment before training:

```bash
uv run mujoco-lab doctor
```

On an NVIDIA machine, the key line is `CUDA available: True`. If it is `False`, verify what the exact `uv` environment sees:

```bash
uv run python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('torch_cuda=', torch.version.cuda); print('devices=', torch.cuda.device_count()); print('device0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

`device: auto` falls back to CPU only when CUDA/MPS is unavailable to PyTorch. To require NVIDIA CUDA and fail fast instead of falling back, set:

```yaml
runtime:
  device: cuda
```

This distinction is useful in automated experiments: `auto` is portable, while `cuda` catches a broken CUDA environment immediately.

Outputs land under `runs/humanoid-v5-ppo/`. TensorBoard can inspect learning curves:

```bash
uv run tensorboard --logdir runs/humanoid-v5-ppo/tensorboard
```

## 4. Evaluate

```bash
uv run mujoco-lab evaluate \
  configs/humanoid/humanoid_v5_ppo.yaml \
  runs/humanoid-v5-ppo/model.zip \
  --episodes 5
```

On a headless server add `--no-render`.

## 5. Bimanual environment smoke test

```bash
uv run mujoco-lab inspect-env configs/bimanual/two_arm_lift_ppo.yaml
```

The project adapter normalizes robosuite's flattened observations to the dtype declared by its Gymnasium observation space. This is deliberate: an environment should satisfy its own `observation_space.contains(obs)` contract before it is passed into SB3.

robosuite is most straightforward on Linux/macOS. On Windows, WSL2/Ubuntu is the recommended training environment if native graphics/input dependencies become a distraction.

## 6. Headless rendering

Training and dynamics CI should normally be headless. Do **not** force `MUJOCO_GL=egl` merely to run non-rendering reset/step tests: a machine without an EGL display can then fail during MuJoCo import even though no pixels are requested.

When a task actually needs rendered observations or videos, configure EGL/OSMesa (or the platform renderer) in a dedicated environment/job and test that rendering path explicitly.

## Debugging order

When something fails, diagnose in this order:

1. `doctor` — packages, versions and accelerator visible?
2. `inspect-env` — model loads, reset and one step work?
3. `observation_space.contains(obs)` / `action_space.contains(action)` — wrapper contract valid?
4. zero/random action rollout — stable contacts and sensible ranges?
5. short training budget — learning loop works?
6. full training — only after the first five are clean.

This order prevents hours of reward tuning when the real problem is an invalid model, version incompatibility, action scale, wrapper dtype, accelerator setup or renderer.
