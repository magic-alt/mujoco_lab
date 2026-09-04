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

1. `doctor` — packages and versions installed?
2. `inspect-env` — model loads, reset and one step work?
3. `observation_space.contains(obs)` / `action_space.contains(action)` — wrapper contract valid?
4. zero/random action rollout — stable contacts and sensible ranges?
5. short training budget — learning loop works?
6. full training — only after the first five are clean.

This order prevents hours of reward tuning when the real problem is an invalid model, version incompatibility, action scale, wrapper dtype or renderer.
