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

## 3. Understand hardware availability vs workload scheduling

Run:

```bash
uv run mujoco-lab doctor
```

`doctor` answers two different questions:

1. what torch accelerators are visible to the current environment;
2. what the current native SB3 PPO/MLP workload would choose when `device: auto` is used.

On an NVIDIA workstation a healthy report can look like:

```text
CUDA available: True
cuda:0: NVIDIA ...
native SB3 PPO/MlpPolicy auto device: cpu
```

This is not a contradiction. Native MuJoCo still advances physics on CPU, while the bootstrap PPO policy is a small MLP. Moving only that MLP to CUDA can add host/device synchronization without removing the simulator bottleneck. Therefore the current workload policy is:

```text
native MuJoCo + SB3 PPO + MlpPolicy + device:auto -> CPU
```

Explicit requests remain authoritative:

```yaml
runtime:
  device: cuda
```

With `device: cuda`, training fails fast if CUDA is unavailable. This is useful for controlled CPU-vs-GPU comparisons, but GPU utilization itself is not a success metric.

If CUDA should be visible but is not, inspect the exact `uv` environment:

```bash
uv run python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('torch_cuda=', torch.version.cuda); print('devices=', torch.cuda.device_count()); print('device0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## 4. Benchmark native MuJoCo vectorization

Torch device selection and environment parallelism are separate decisions. The configuration supports:

```yaml
runtime:
  vec_env_backend: auto  # auto | dummy | subproc
```

`auto` intentionally keeps the SB3 `DummyVecEnv` reference. Do not assume `SubprocVecEnv` is faster: multiprocessing adds startup, serialization and IPC overhead.

Measure the current machine:

```bash
uv run mujoco-lab benchmark-vec-env \
  configs/humanoid/humanoid_v5_ppo.yaml \
  --env-counts 1,2,4,8 \
  --backends dummy,subproc \
  --transitions 10000
```

The default report is written under `runs/benchmarks/` and records for each case:

- backend;
- environment count;
- startup + first-reset time;
- measured transitions;
- elapsed steady-state time;
- transitions/s;
- platform and package versions.

The benchmark intentionally uses zero actions and excludes policy forward/backward computation. It measures the native simulator/VecEnv layer, not PPO learning speed.

`SubprocVecEnv` requires at least two environments. On Windows it also uses multiprocessing semantics, so run it through the packaged `mujoco-lab` entry point rather than copying isolated snippets into an unguarded script.

## 5. Train the first humanoid baseline

```bash
uv run mujoco-lab train configs/humanoid/humanoid_v5_ppo.yaml
```

Before SB3 starts, the trainer prints two reports:

```text
mujoco_lab accelerator report
...
mujoco_lab vector-environment report
...
```

The same resolved choices and reasons are persisted in `runs/<experiment>/metadata.json` together with package versions. The checked-in Humanoid configuration uses workload-aware `device: auto` and conservative `vec_env_backend: auto`.

Outputs land under `runs/humanoid-v5-ppo/`. TensorBoard can inspect learning curves:

```bash
uv run tensorboard --logdir runs/humanoid-v5-ppo/tensorboard
```

## 6. Evaluate

```bash
uv run mujoco-lab evaluate \
  configs/humanoid/humanoid_v5_ppo.yaml \
  runs/humanoid-v5-ppo/model.zip \
  --episodes 5
```

On a headless server add `--no-render`.

## 7. Validate the pinned G1 model

Before implementing or training custom G1 tasks:

```bash
uv run mujoco-lab inspect-robot g1
```

The command resolves the immutable Menagerie revision, verifies cached SHA-256 hashes, compiles the model and checks the joint/actuator/site/contact/stand-keyframe contract. Phase 2B uses this model to build a **native MuJoCo CPU standing reference**. The early MJX milestone follows only after standing semantics are explicit and tested.

## 8. Bimanual environment smoke test

```bash
uv run mujoco-lab inspect-env configs/bimanual/two_arm_lift_ppo.yaml
```

The project adapter normalizes robosuite's flattened observations to the dtype declared by its Gymnasium observation space. This is deliberate: an environment should satisfy its own `observation_space.contains(obs)` contract before it is passed into SB3.

robosuite is most straightforward on Linux/macOS. On Windows, WSL2/Ubuntu is the recommended training environment if native graphics/input dependencies become a distraction.

## 9. Headless rendering

Training and dynamics CI should normally be headless. Do **not** force `MUJOCO_GL=egl` merely to run non-rendering reset/step tests: a machine without an EGL display can then fail during MuJoCo import even though no pixels are requested.

When a task actually needs rendered observations or videos, configure EGL/OSMesa (or the platform renderer) in a dedicated environment/job and test that rendering path explicitly.

## Debugging order

When something fails, diagnose in this order:

1. `doctor` — packages, hardware accelerators and workload recommendation visible?
2. `inspect-env` / `inspect-robot` — model loads and one step works?
3. `observation_space.contains(obs)` / `action_space.contains(action)` — wrapper contract valid?
4. `benchmark-vec-env` — native execution topology measured rather than guessed?
5. zero/random action rollout — stable contacts and sensible ranges?
6. short training budget — learning loop works?
7. full multi-seed training — only after the first six are clean.

This order prevents hours of reward or CUDA tuning when the real problem is an invalid model, version incompatibility, action scale, wrapper dtype, process topology or renderer.
