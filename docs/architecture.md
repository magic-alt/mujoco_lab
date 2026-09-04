# Architecture

## Goals

The architecture must support two very different learning problems without turning the repository into two unrelated projects:

- high-contact, whole-body humanoid locomotion;
- multi-arm manipulation with state, vision and eventually demonstration data.

The common denominator is not the algorithm. It is the **experiment contract**: simulator/model + task + observation + action + reward + termination + runtime/backend + algorithm + evaluation.

## Layered design

```text
configs/ + CLI
      |
      v
ExperimentConfig
      |
      +----------------------+
      |                      |
      v                      v
runtime.py              benchmarks/
(device/workload,       (measured backend
 VecEnv policy)          throughput)
      |                      |
      +----------+-----------+
                 |
                 v
envs/ -------------------- external model sources
  |                             |-- Gymnasium built-ins
  |                             |-- MuJoCo Menagerie
  |                             `-- robosuite models/tasks
  v
Gymnasium Env contract
      |
      v
training/vec_env.py
      |
      +-------------------+
      |                   |
      v                   v
DummyVecEnv          SubprocVecEnv
      |                   |
      +---------+---------+
                |
                v
          training/sb3.py
                |
                v
model + normalizer + resolved config + metadata + metrics

Future accelerator path:
CPU-reference task semantics -> MJX/JAX batched physics -> accelerator trainer
```

### `config.py`

Owns the portable experiment schema. Configuration is validated before simulator creation. The schema stays small; task-specific scientific parameters remain in `task.kwargs` until they become a shared contract.

Runtime configuration deliberately separates two concepts:

- `runtime.device` — torch policy/trainer device selection;
- `runtime.vec_env_backend` — native MuJoCo environment vectorization (`dummy` or `subproc`).

These are not aliases for one another.

### `runtime.py`

Owns scheduling policy, not simulator or algorithm semantics. Hardware availability and workload recommendation are separate facts.

For the current native MuJoCo + SB3 PPO + `MlpPolicy` workload, `device: auto` resolves to CPU because MuJoCo physics remains on CPU and the policy network is small. An explicit `device: cuda` still overrides the automatic policy when CUDA is available.

For generic torch capability reporting, CUDA remains the preferred accelerator when available. `doctor` therefore may correctly print both:

```text
CUDA available: True
native SB3 PPO/MlpPolicy auto device: cpu
```

The reason for every automatic choice is persisted in run metadata.

### `envs/`

Owns simulator and task integration. It returns a Gymnasium-compatible environment and nothing algorithm-specific. Optional frameworks such as robosuite are imported lazily.

Custom G1/H1 code is split into model loading, observation extraction, action/control application, reward terms and termination functions. Reward components must be individually observable for debugging.

### `training/vec_env.py`

Owns SB3 vector-environment construction from the shared Gymnasium task contract.

`DummyVecEnv` and `SubprocVecEnv` are explicit backends. `auto` intentionally remains `DummyVecEnv` until a machine-specific benchmark says otherwise: subprocess startup, serialization and IPC can erase the benefit of parallel CPU stepping for lighter tasks.

### `benchmarks/`

Owns performance experiments that should not be confused with learning results. `benchmark-vec-env` measures:

- environment startup + first reset time;
- steady-state environment transitions/s;
- backend and environment count;
- platform/package metadata.

It uses zero actions and excludes policy forward/backward computation so it isolates simulator/vectorization throughput. PPO wall-clock benchmarks remain a separate experiment.

### `training/`

Owns optimization backends. The initial backend is SB3 PPO. A later MJX/JAX backend must not require changing native task semantics merely to satisfy a different execution engine.

### Artifact contract

Each training output directory must be self-describing:

```text
runs/<experiment>/
├── model.zip
├── vecnormalize.pkl        # when normalization is enabled
├── resolved_config.yaml
├── metadata.json           # accelerator + VecEnv decisions included
├── checkpoints/
└── tensorboard/
```

Performance benchmark artifacts live separately under `runs/benchmarks/` by default. Later phases will add evaluation JSON, videos and policy exports. Large artifacts remain outside Git.

## Native MuJoCo vs MJX execution model

The project uses two intentionally different performance regimes.

### Native reference regime

```text
MuJoCo physics: CPU
Gymnasium/task: CPU
Dummy/Subproc VecEnv: CPU process topology
SB3 PPO MLP: CPU by default, explicit CUDA allowed
```

This regime is optimized first for inspectability and task correctness. The Phase 2B G1 standing environment is built here.

### Accelerator-batched regime

```text
many G1 states/models
       |
       v
MJX/JAX batched physics
       |
       v
512 / 1024 / 2048+ environments
       |
       v
accelerator-native policy/update path
```

The early Phase 2C milestone measures this regime after standing semantics are stable. The CPU task remains the semantic reference and backend differences must be explicit.

## Humanoid control architecture

The production locomotion task will use a command-conditioned policy rather than “walk forward as fast as possible”. A target command such as `[vx, vy, yaw_rate]` enters the observation. The policy emits normalized joint targets or residual joint targets; a lower-level PD actuator layer converts targets to torque within model limits.

Recommended separation:

```text
velocity command
      |
      v
observation builder --> policy --> normalized action
                               |
                               v
                      action scaler / limits
                               |
                               v
                         joint target
                               |
                               v
                         PD actuator
                               |
                               v
                            MuJoCo
```

This mirrors how learned locomotion policies are commonly deployed while keeping motor/gearbox and servo dynamics explicit enough for later sim-to-real work.

## Bimanual control architecture

The bootstrap uses robosuite's control stack and `GymWrapper`. The tutorial progresses from proprioceptive/object state to image observations and demonstrations. Do not mix raw camera tensors into the MLP baseline simply because the simulator can render them; vision policies require a separate observation and model design.

## Time scales

Physics timestep, controller rate and policy rate are different concepts. Every custom robot task must publish all three. In contact-rich systems, numerical stability is a model property; MuJoCo documentation identifies timestep as one of the most important stability/performance parameters and generally recommends `implicitfast` for typical models. Treat changes to timestep/integrator as model changes, not harmless speed tweaks.

## Extension points

Planned stable interfaces:

- `make_env(config, render_mode)` — simulator/task factory;
- `make_sb3_vec_env(...)` — explicit native vectorization boundary;
- workload-aware runtime resolution — device/backend scheduling;
- config-driven trainer functions — algorithms;
- `RobotModelSpec` — external asset provenance and joints/sites;
- later: `RewardTerm` diagnostics — decomposed locomotion rewards;
- later: `PolicyArtifact` — checkpoint + normalization + exported policy metadata;
- later: `Evaluator` — repeatable multi-seed metrics.

## Non-goals

- Vendoring every robot model.
- Building a new general-purpose RL library.
- Maximizing GPU utilization as an objective independent of wall-clock learning performance.
- Hiding MuJoCo details behind so many abstractions that learners cannot inspect model/data/contact state.
- Claiming sim-to-real transfer from a visually convincing simulation.
