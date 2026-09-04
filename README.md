# mujoco_lab

A documentation-first MuJoCo robotics learning lab for **humanoid locomotion** and **bimanual manipulation**.

The repository is intentionally built as a small research platform rather than a collection of one-off scripts: environments expose Gymnasium-compatible interfaces, algorithms live behind training adapters, robot assets are external/versioned, and every experiment is driven by a checked-in configuration.

## Learning tracks

| Track | First runnable baseline | Next production-grade target |
| --- | --- | --- |
| Humanoid locomotion | Gymnasium `Humanoid-v5` + PPO | Unitree G1/H1 command-conditioned walking with Menagerie assets, curriculum and domain randomization |
| Bimanual manipulation | robosuite `TwoArmLift` + PPO | Handover / peg-in-hole, demonstrations, ALOHA-style ACT and Diffusion Policy |
| Acceleration | CPU/GPU SB3 | MJX/JAX batched simulation and high-throughput training |
| Deployment | deterministic evaluation | policy export, sim-to-sim and sim-to-real interfaces |

## Why this stack

- **MuJoCo** is the physics source of truth.
- **Gymnasium** is the environment contract, which prevents training code from depending on one simulator wrapper.
- **Stable-Baselines3 PPO** is the first baseline because it is readable, mature and supports continuous actions and vectorized environments.
- **robosuite** supplies well-designed two-arm MuJoCo tasks and an official Gym-compatible wrapper.
- **MuJoCo Menagerie** supplies curated robot models such as Unitree G1 and H1 without vendoring large third-party assets into this repository.
- **MJX** is a planned second backend for accelerator-parallel locomotion once the CPU baseline and task semantics are proven.

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
    API --> T[Training adapters]
    T --> PPO[SB3 PPO]
    T -. future .-> MJX[MJX/JAX or RSL-RL]
    PPO --> A[runs/: model + config + normalization + metadata]
    MJX --> A
```

The dependency rule is strict: **task/environment code must not import the RL algorithm implementation**. Robot-model provenance and validation are also kept below the task layer, so a model-contract failure is not confused with an RL failure.

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
│   ├── envs/                # simulator/task adapters
│   ├── robots/              # external model specs, resolver and contracts
│   ├── training/            # algorithm adapters
│   ├── config.py            # config schema + validation
│   └── cli.py               # train/evaluate/inspect/doctor
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

The repository has completed the Phase 0/1 research-platform bootstrap and now enters Phase 2 humanoid locomotion. The first Phase 2 slice establishes a machine-checkable Unitree G1 model contract from a pinned Menagerie revision before any custom PD controller or walking reward is introduced.

The next vertical slices remain deliberately ordered:

1. G1 model provenance/inspection contract;
2. G1 standing environment + explicit PD actuator semantics;
3. command-conditioned walking + decomposed rewards;
4. robustness/domain randomization and export.

The Humanoid-v5 PPO settings remain starting baselines until issue #2 reports multi-seed empirical results. No full-million-step training success is implied by the existence of a runnable config.

## Research discipline

A training run is not considered evidence because a video looks plausible. Results should report multiple seeds, task success/return statistics, training budget, simulator/model version and evaluation conditions. Do not commit large model assets, datasets or checkpoints; record their provenance and version instead.

## License

Apache-2.0. Third-party robot assets and datasets retain their own licenses; verify them before redistribution.
