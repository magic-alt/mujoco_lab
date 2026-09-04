# Contributing

## Setup

```bash
uv sync --extra train --extra dev
```

Add `--extra bimanual` when working on robosuite tasks.

## Local quality gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run mujoco-lab doctor
uv run mujoco-lab inspect-env configs/humanoid/humanoid_v5_ppo.yaml
```

Before changing robot dynamics or reward logic, also run a short local rollout and record the exact command in the PR.

## Change design

Every non-trivial change should answer:

1. What task contract changes?
2. Which observation/action/reward/termination behavior changes?
3. How is the change tested?
4. What metric would falsify the claim that the change is better?
5. Which tutorial, config, ADR or development-log entry must change with it?

## Pull requests

Prefer small vertical slices. A PR should include tests and docs with the implementation rather than creating a documentation follow-up that can drift from the code.

Do not commit generated training runs, checkpoints, third-party robot meshes or datasets. Link to their source and record a version/revision instead.
