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

## 2026-09-04 — Phase 2A Unitree G1 model contract

### Goal

Complete issue #3 before introducing a custom standing environment: pin a real Unitree G1 model, make its provenance reproducible, and turn every assumption needed by later locomotion tasks into a machine-checkable contract.

### Branch hygiene

The original G1 development branch was created before the GPU-first training fix was merged. It therefore diverged from `main` and temporarily contained duplicated accelerator work. Before opening the Phase 2A PR, the branch was synchronized to the current `main` and its effective diff was reduced to G1-only changes. The early torque-actuator conversion prototype was also removed from Phase 2A because explicit PD/torque semantics belong to issue #4.

### Implemented

- pinned MuJoCo Menagerie repository revision `e4049d0a3bfd58d2a3081614e6777d4007e3f86a`;
- machine-readable `RobotModelSpec` including source URL and `BSD-3-Clause` SPDX license metadata;
- revision-scoped local cache that downloads only G1 MJCF, README, LICENSE and referenced meshes;
- SHA-256 manifest validation on every cache reuse;
- explicit offline, force-rebuild, revision-mismatch, missing-file and hash-mismatch behavior;
- `mujoco-lab inspect-robot g1` with cache/output controls;
- model contract for 29 articulated joints, 29 actuators, initial 15 leg/waist controlled joints, IMU/foot sites, pelvis/foot bodies, contact geoms and stand keyframe limits;
- one-step native MuJoCo finite-state smoke validation;
- JSON inspection artifact with timestep, integrator and actuator force/control metadata;
- network-independent resolver tests plus a real pinned-Menagerie GitHub Actions integration check.

### CI integration notes

Phase 2A was deliberately validated through the same strict gate rather than accepted from static inspection.

1. **Lint contract** — the first clean-branch run exposed Typer default-call `B008` violations and two line-length violations. New CLI parameters were moved to `Annotated[...]` metadata and the long expressions were structurally shortened; no Ruff rule was disabled.
2. **Formatter contract** — the next run passed lint but Ruff formatter preferred a different layout for two G1 conditions. The code was rewritten with a local `expected_actuators` value so both the 100-character lint rule and formatter agree.
3. **CI PyTorch observation** — hosted Linux runners currently resolve Torch 2.14 with CUDA 13 runtime packages even though no GPU is present. A trial of `uv sync --torch-backend=cpu` failed because the installed `uv 0.12.9` did not accept that argument for `sync`, so the experiment was reverted rather than weakening or complicating the G1 PR. CI backend isolation remains a separate infrastructure improvement.
4. **Full Phase 2A validation** — GitHub Actions run #24 passed install, Ruff lint, Ruff format, all tests, dependency doctor and the live G1 model contract.

The validated environment was Python 3.11 with MuJoCo 3.9.0, Gymnasium 1.3.0, Stable-Baselines3 2.9.0, Torch 2.14.0 and robosuite 1.5.1. Pytest reported **12 passed**; the two remaining warnings are the already-known robosuite/Gymnasium float64-to-float32 Box precision warnings at the wrapper boundary.

The live G1 inspection resolved the pinned upstream files, compiled `scene.xml`, reset to the `stand` keyframe, executed forward dynamics and one MuJoCo step, then reported:

- `nq=36`, `nv=35`;
- 29 articulated joints and 29 actuators;
- 15 initial leg/waist controlled joints;
- four contact-enabled geoms on each foot;
- stand base height `0.79 m`;
- timestep `0.002 s`;
- integrator `mjINT_IMPLICITFAST`;
- actuator control ranges and joint actuator-force limits available through the MuJoCo 3.9 Python binding.

### Boundary for the next phase

Phase 2A does not claim that G1 can stand or walk under a learned policy. Issue #4 starts from this validated source model and adds the task-owned actuator layer, nominal standing state, residual joint targets, explicit PD gains/torque limits, observation contract, standing reward and termination semantics. Issue #5 adds velocity commands and walking rewards only after standing is independently stable and learnable.

## 2026-09-04 — Workload-aware runtime and VecEnv benchmark

### Trigger

A real Windows training run on an RTX 5060 confirmed that the CUDA-enabled PyTorch environment was healthy (`torch 2.14.0+cu130`, CUDA 13.0, compute capability 12.0), but SB3 warned that PPO with `MlpPolicy` can be slower on GPU. Native MuJoCo physics remained CPU-side and the configured four environments were still using SB3's default `DummyVecEnv`, so “CUDA is available” had been incorrectly treated as equivalent to “CUDA is the best default for this workload”.

### Research

Current SB3 documentation was re-checked for `make_vec_env` and vectorized-environment behavior. `make_vec_env` defaults to `DummyVecEnv`, and subprocess vectorization is an explicit alternative whose benefit depends on environment cost and IPC overhead. Current MuJoCo MJX documentation was also re-checked: accelerator throughput comes from batched device-resident physics and is best suited to large batches rather than single-scene simulation. See `docs/research/2026-09-04-runtime-scheduling.md`.

### Decisions

- hardware accelerator visibility and workload scheduling are separate contracts;
- `doctor` continues to report CUDA/MPS capability even when the current trainer recommendation is CPU;
- `device: auto` for native MuJoCo + SB3 PPO + `MlpPolicy` is CPU-preferred;
- explicit `device: cuda|mps|cpu` remains authoritative and fail-fast;
- `runtime.vec_env_backend` is independent of torch device and supports `auto|dummy|subproc`;
- VecEnv `auto` keeps `DummyVecEnv` until a machine benchmark justifies changing it;
- `benchmark-vec-env` measures startup and simulator transitions/s without policy compute;
- the G1 roadmap is now #4 native standing → #9 early MJX throughput → #5 command-conditioned walking.

### Implemented in issue #13

- workload profiles and selection reasons in `runtime.py`;
- shared SB3 VecEnv factory;
- configurable Dummy/Subproc training backend;
- standalone VecEnv benchmark artifact with platform/package metadata;
- training metadata records both accelerator and VecEnv decisions;
- CI smoke target for both vectorization implementations;
- ADR 0003, architecture/tutorial/roadmap updates;
- issue #4 explicitly designated as the native CPU semantic reference;
- issue #9 moved forward to a 512/1024/2048+ G1 standing throughput milestone;
- issue #5 renamed Phase 2D and made dependent on the early acceleration findings without coupling task semantics to MJX.

### Evidence still required

This section records architecture and implementation intent only. The branch must still pass GitHub Actions, including the real `SubprocVecEnv` benchmark smoke, before the runtime change is considered complete. No claim is made yet about which VecEnv backend is faster on the RTX 5060 workstation; that result must come from the new benchmark command on the target machine.
