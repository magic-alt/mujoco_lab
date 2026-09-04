# 00 — Getting started

## 1. Install Python and uv

Use Python 3.11 or 3.12. Install `uv`, clone the repository, then create the core training environment:

```bash
uv sync --extra train --extra dev
uv run mujoco-lab doctor
```

For two-arm tasks:

```bash
uv sync --extra train --extra bimanual --extra dev
```

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

robosuite officially targets macOS and Linux most cleanly. Windows support can require platform-specific rendering fixes; for Windows development, WSL2/Ubuntu is the recommended training environment if native rendering becomes a distraction.

## 6. Headless rendering

Training should normally be headless. On Linux servers MuJoCo can use EGL/OSMesa depending on the workload. CI sets `MUJOCO_GL=egl` for tests, although the smoke tests do not request rendered pixels.

## Debugging order

When something fails, diagnose in this order:

1. `doctor` — packages installed?
2. `inspect-env` — model loads, reset and one step work?
3. zero/random action rollout — stable contacts and sensible ranges?
4. short training budget — learning loop works?
5. full training — only after the first four are clean.

This order prevents hours of reward tuning when the real problem is an invalid model, action scale or renderer.
