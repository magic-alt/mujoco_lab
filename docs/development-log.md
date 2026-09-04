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
- CI and smoke tests for both learning tracks;
- architecture/roadmap/ADR/tutorial/research docs.

### CI integration notes

The bootstrap PR was exercised on GitHub Actions rather than accepted from static review alone. The failures below are retained because each one exposed a real integration boundary.

1. **Formatting gate** — the first run installed successfully and passed Ruff lint, then Ruff format rejected one string layout in `envs/bimanual.py`. The branch was formatted and the gate was kept strict.
2. **Headless rendering boundary** — the next run failed while importing MuJoCo because CI globally forced `MUJOCO_GL=egl` on a runner without a usable EGL display. The dynamics smoke tests never request pixels, so the correct fix was to remove the renderer override. Rendering tests will use a separate graphics-aware job later.
3. **Humanoid baseline verified** — after separating dynamics from rendering, installation, Ruff, pytest, `Humanoid-v5` reset/step and `mujoco-lab doctor` all passed on Ubuntu / Python 3.11.
4. **Exercise the optional bimanual stack** — CI was then expanded to install the `bimanual` extra and execute a real robosuite `TwoArmLift` reset/step instead of leaving the second learning track untested.
5. **robosuite wrapper dependency** — importing `robosuite.wrappers` also imports demonstration utilities that require `h5py`; the base robosuite install did not provide it in the resolved environment. `h5py` was therefore made an explicit dependency of this project's `bimanual` extra so the documented install command is sufficient.
6. **MuJoCo / robosuite compatibility** — with `h5py` installed, `TwoArmLift` failed inside robosuite when resolved against MuJoCo 3.12. Current robosuite development metadata constrains MuJoCo to `>=3.3,<3.10` because newer API changes break parts of its control stack. The compatibility constraint is deliberately scoped to the `bimanual` extra rather than globally pinning the humanoid/MJX path. CI then resolved MuJoCo 3.9.0 and successfully constructed/reset `TwoArmLift`.
7. **Gymnasium dtype contract** — robosuite's flattened `GymWrapper` declared a `float32` observation space while returning a `float64` array, causing `observation_space.contains(obs)` to fail. `mujoco_lab` now wraps it with a small observation adapter that casts to the declared dtype at the integration boundary.
8. **Both tracks verified** — the resulting CI run passed installation, lint, format, all tests, `Humanoid-v5` reset/step, `TwoArmLift` reset/step and `mujoco-lab doctor`.

The main lesson is architectural: simulator version constraints, rendering, third-party wrappers and RL training are separate concerns. The project keeps those boundaries explicit instead of hiding them with skipped tests or patches to third-party internals.

### Explicitly not claimed yet

No full training run has been executed by this bootstrap change. The checked-in hyperparameters are starting baselines, not benchmark results. Phase 1 requires empirical multi-seed validation; Phase 2 implements the first Unitree G1 custom locomotion task.
