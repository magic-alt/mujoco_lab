# Architecture

## Goals

The architecture must support two very different learning problems without turning the repository into two unrelated projects:

- high-contact, whole-body humanoid locomotion;
- multi-arm manipulation with state, vision and eventually demonstration data.

The common denominator is not the algorithm. It is the **experiment contract**: simulator/model + task + observation + action + reward + termination + algorithm + evaluation.

## Layered design

```text
configs/ + CLI
      |
      v
ExperimentConfig
      |
      v
envs/ -------------------- external model sources
  |                             |-- Gymnasium built-ins
  |                             |-- MuJoCo Menagerie
  |                             `-- robosuite models/tasks
  v
Gymnasium Env contract
      |
      +-------------------+
      |                   |
      v                   v
training/sb3.py      future training/mjx.py
      |                   |
      v                   v
model + normalizer + resolved config + metadata + metrics
```

### `config.py`

Owns the portable experiment schema. Configuration is validated before simulator creation. The bootstrap schema is intentionally small; future fields should be added only when two or more tasks need them, otherwise task-specific values remain in `task.kwargs`.

### `envs/`

Owns simulator and task integration. It returns a Gymnasium-compatible environment and nothing algorithm-specific. Optional frameworks such as robosuite are imported lazily.

Future custom G1/H1 code will be split further into model loading, observation extraction, action/control application, reward terms and termination functions. Reward components must be individually observable for debugging.

### `training/`

Owns optimization backends. The initial backend is SB3 PPO. The next backend should not require changing task code; this is the architectural test for adding MJX/JAX or RSL-RL.

### Artifact contract

Each training output directory must be self-describing:

```text
runs/<experiment>/
├── model.zip
├── vecnormalize.pkl        # when normalization is enabled
├── resolved_config.yaml
├── metadata.json
├── checkpoints/
└── tensorboard/
```

Later phases will add evaluation JSON, videos and policy exports. Large artifacts remain outside Git.

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
- config-driven trainer functions — algorithms;
- later: `RobotModelSpec` — external asset provenance and joints/sites;
- later: `RewardTerm` diagnostics — decomposed locomotion rewards;
- later: `PolicyArtifact` — checkpoint + normalization + exported policy metadata;
- later: `Evaluator` — repeatable multi-seed metrics.

## Non-goals

- Vendoring every robot model.
- Building a new general-purpose RL library.
- Hiding MuJoCo details behind so many abstractions that learners cannot inspect model/data/contact state.
- Claiming sim-to-real transfer from a visually convincing simulation.
