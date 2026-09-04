# mujoco_lab

A documentation-first MuJoCo robotics learning lab for **humanoid locomotion** and **bimanual manipulation**.

The repository is intentionally built as a small research platform rather than a collection of one-off scripts: environments expose Gymnasium-compatible interfaces, algorithms live behind training adapters, robot assets are external/versioned, runtime scheduling is explicit, and every experiment is driven by a checked-in configuration.

## Learning tracks

| Track | First runnable baseline | Next production-grade target |
| --- | --- | --- |
| Humanoid locomotion | Gymnasium `Humanoid-v5` + PPO | Unitree G1/H1 command-conditioned walking with Menagerie assets, curriculum and domain randomization |
| Bimanual manipulation | robosuite `TwoArmLift` + PPO | Handover / peg-in-hole, demonstrations, ALOHA-style ACT and Diffusion Policy |
| Acceleration | native MuJoCo + measured Dummy/Subproc VecEnv | early G1 MJX throughput prototype at 512/1024/2048+ envs, then semantic-parity accelerator backend |
| Deployment | deterministic evaluation | policy export, sim-to-sim and sim-to-real interfaces |

## Why this stack

- **MuJoCo** is the physics source of truth.
- **Gymnasium** is the environment contract, which prevents training code from depending on one simulator wrapper.
- **Stable-Baselines3 PPO** is the first baseline because it is readable, mature and supports continuous actions and vectorized environments.
- **robosuite** supplies well-designed two-arm MuJoCo tasks and an official Gym-compatible wrapper.
- **MuJoCo Menagerie** supplies curated robot models such as Unitree G1 and H1 without vendoring large third-party assets into this repository.
- **MJX** is the accelerator path for large batched physics, not a reason to push every small native-MuJoCo MLP workload onto CUDA.

## Quick start

Python 3.11 or 3.12 is recommended. `uv` is the preferred environment manager.

```bash
git clone https://github.com/magic-alt/mujoco_lab.git
cd mujoco_lab

uv sync --extra train --extra dev
uv run mujoco-lab doctor
uv run mujoco-lab inspect-env configs/humanoid/humanoid_v5_ppo.yaml
uv run mujoco-lab train configs/humanoid/humanoid_v5_ppo.yaml
```

For the current native MuJoCo + SB3 PPO + `MlpPolicy` workload, `runtime.device: auto` is **workload-aware and CPU-preferred** even when CUDA is available. `doctor` still reports all visible accelerators. Set `device: cuda` only when you intentionally want to override the workload policy.

Vector-environment selection is independent of torch device selection. `runtime.vec_env_backend: auto` keeps the conservative `DummyVecEnv` reference. Benchmark the actual machine before switching to multiprocessing:

```bash
uv run mujoco-lab benchmark-vec-env \
  configs/humanoid/humanoid_v5_ppo.yaml \
  --env-counts 1,2,4,8 \
  --backends dummy,subproc \
  --transitions 10000
```

The report records startup cost and steady-state environment transitions/s. It measures simulator/VecEnv throughput only; it deliberately excludes policy forward/backward compute so CUDA effects do not contaminate the comparison.

Before implementing custom G1 locomotion tasks, resolve and validate the exact external robot model:

```bash
uv run mujoco-lab inspect-robot g1
```

The command pins MuJoCo Menagerie to an immutable revision, caches only the required G1 files, verifies their SHA-256 manifest, compiles the model, checks the standing keyframe/joint/actuator/site/contact contract and writes an `inspection.json` report. No third-party meshes are committed to this repository.

For the bimanual track:

```bash
uv sync --extra train --extra bimanual --extra dev
uv run mujoco-lab inspect-env configs/bimanual/two_arm_lift_ppo.yaml
uv run mujoco-lab train configs/bimanual/two_arm_lift_ppo.yaml
```

Evaluate a checkpoint:

```bash
uv run mujoco-lab evaluate \
  configs/humanoid/humanoid_v5_ppo.yaml \
  runs/humanoid-v5-ppo/model.zip \
  --episodes 5
```

Use `--no-render` for headless servers.

## Architecture

```mermaid
flowchart LR
    C[configs/] --> CLI[mujoco-lab CLI]
    CLI --> RUNTIME[Runtime policy + benchmarks]
    CLI --> R[Robot model resolver/contract]
    CLI --> F[Environment factory]
    R --> G1[Pinned Menagerie G1]
    F --> H[Gymnasium Humanoid]
    F --> B[robosuite bimanual]
    F --> G[Custom G1/H1 tasks]
    G1 --> G
    H --> API[Gymnasium Env API]
    B --> API
    G --> API
    API --> V[DummyVecEnv / SubprocVecEnv]
    V --> T[Training adapters]
    RUNTIME --> V
    RUNTIME --> T
    T --> PPO[SB3 PPO]
    T -. accelerator backend .-> MJX[MJX/JAX]
    PPO --> A[runs/: model + config + normalization + metadata]
    MJX --> A
```

The dependency rule is strict: **task/environment code must not import the RL algorithm implementation**. Hardware availability, workload scheduling and simulator parallelism are separate concerns. Robot-model provenance and validation also stay below the task layer, so a model-contract failure is not confused with an RL failure.

## Repository map

```text
mujoco_lab/
├── configs/                 # versioned experiment inputs
│   ├── humanoid/
│   └── bimanual/
├── docs/
│   ├── adr/                 # architecture decisions
│   ├── research/            # source-backed landscape notes
│   └── tutorials/           # progressive learning path
├── src/mujoco_lab/
│   ├── benchmarks/          # runtime/simulator throughput measurements
│   ├── envs/                # simulator/task adapters
│   ├── robots/              # external model specs, resolver and contracts
│   ├── training/            # VecEnv + algorithm adapters
│   ├── runtime.py           # workload/device/backend selection policy
│   ├── config.py            # config schema + validation
│   └── cli.py               # train/evaluate/inspect/benchmark/doctor
├── tests/                   # fast contract and smoke tests
├── AGENTS.md                # repository rules for AI coding agents
└── .github/workflows/ci.yml
```

## Tutorial path

1. [Environment setup](docs/tutorials/00_getting_started.md)
2. [MuJoCo fundamentals](docs/tutorials/01_mujoco_fundamentals.md)
3. [Humanoid walking](docs/tutorials/02_humanoid_walking.md)
4. [Bimanual training](docs/tutorials/03_bimanual_training.md)
5. [AI-coding workflow](docs/tutorials/04_ai_coding_workflow.md)
6. [Experiment protocol](docs/experiment_protocol.md)
7. [Requirements](docs/requirements.md)
8. [Architecture](docs/architecture.md)
9. [Roadmap](docs/roadmap.md)
10. [Research landscape](docs/research/2026-09-04-landscape.md)

## Current scope

Phase 2A is complete: the repository has a pinned, machine-checkable Unitree G1 model contract. The next humanoid vertical slices are deliberately ordered:

1. **#4** — native MuJoCo G1 standing environment + explicit PD actuator semantics;
2. **#9** — early MJX/JAX throughput prototype using the standing contract at 512/1024/2048+ environments;
3. **#5** — command-conditioned walking + decomposed rewards, informed by but not coupled to the accelerator prototype;
4. **#6** — robustness/domain randomization and export.

The Humanoid-v5 PPO settings remain starting baselines until issue #2 reports multi-seed empirical results. No full-million-step training success is implied by the existence of a runnable config.

## Research discipline

A training run is not considered evidence because a video looks plausible. Results should report multiple seeds, task success/return statistics, training budget, simulator/model version and evaluation conditions. Performance claims must additionally identify the runtime backend, environment count and hardware. Do not commit large model assets, datasets or checkpoints; record their provenance and version instead.

## License

Apache-2.0. Third-party robot assets and datasets retain their own licenses; verify them before redistribution.
