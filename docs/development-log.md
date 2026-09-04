# Development log

## 2026-09-04 — Bootstrap architecture

### Goal

Turn the empty repository into a reproducible MuJoCo learning lab for humanoid walking and bimanual manipulation.

### Research performed

Reviewed current MuJoCo Python/model/MJX documentation, MuJoCo Menagerie, Gymnasium Humanoid, Stable-Baselines3 continuous-control guidance, robosuite two-arm environments/GymWrapper, `dm_control`, ALOHA/ACT simulation structure and several Unitree G1 MuJoCo/RL implementations. See `docs/research/2026-09-04-landscape.md`.

### Decisions

- Gymnasium is the public environment contract.
- SB3 PPO is the first trainer, not a permanent framework lock-in.
- Menagerie is the preferred source for G1/H1 assets.
- robosuite is the initial bimanual task framework.
- experiments are config-driven and write resolved config + runtime metadata + normalization state.
- AI coding follows issue/spec → vertical slice → tests → docs → experiment evidence.

### Implemented

- packaging and CLI;
- config validation;
- Humanoid-v5 factory and PPO baseline;
- robosuite two-arm factory and PPO baseline config;
- evaluation path;
- CI and smoke tests;
- architecture/roadmap/ADR/tutorial/research docs.

### CI integration notes

The bootstrap PR was exercised on GitHub Actions rather than accepted from static review alone.

1. Run 1 installed the environment and passed Ruff lint, then failed Ruff's format check on one string in `envs/bimanual.py`. The branch was formatted and updated.
2. Run 2 passed installation, lint and formatting, then exposed an environment mistake: CI forced `MUJOCO_GL=egl` even though the dynamics smoke test does not render. The hosted runner had no usable EGL display, so MuJoCo failed during import before the test reached environment construction.
3. The CI workflow was corrected to keep non-rendering dynamics tests renderer-independent. Run 3 passed install, Ruff lint, Ruff format, pytest (including a real `Humanoid-v5` reset/step), and `mujoco-lab doctor` on Ubuntu with Python 3.11.

This failure history is intentionally recorded because renderer setup and simulator correctness are separate concerns. A later rendering-specific CI job should install and test its graphics backend explicitly rather than changing the global dynamics-test environment.

### Explicitly not claimed yet

No full training run has been executed by this bootstrap change. The checked-in hyperparameters are starting baselines, not benchmark results. Phase 1 requires empirical multi-seed validation; Phase 2 implements the first Unitree G1 custom locomotion task.
