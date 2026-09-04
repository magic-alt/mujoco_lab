# System requirements

This document defines the engineering and research requirements for `mujoco_lab`. The roadmap determines **when** capabilities are added; this document defines **what must remain true** as the repository evolves.

## 1. Product goals

`mujoco_lab` is a tutorial-quality robotics research laboratory, not a collection of isolated training scripts. It must teach and support two end-to-end workflows:

1. **Humanoid locomotion** — canonical MuJoCo continuous control → Unitree G1/H1 standing → command-conditioned walking → robustness → policy export / sim-to-real preparation.
2. **Bimanual manipulation** — robosuite state-based manipulation → reproducible success evaluation → demonstrations → behavior cloning / ACT → vision-conditioned policies.

The project must remain understandable enough for a learner to inspect MuJoCo state, contacts, actuators and reward terms instead of treating the simulator as a black box.

## 2. Environment contract

### ENV-001 — common API

Every trainable task exposed by this repository must implement Gymnasium-compatible reset/step, observation space and action space semantics.

Acceptance:

- `reset(seed=...)` returns `(observation, info)`;
- `step(action)` returns `(observation, reward, terminated, truncated, info)`;
- returned observations satisfy `observation_space.contains(observation)`;
- sampled valid actions can execute at least one simulation step without NaN/Inf under the smoke-test state.

### ENV-002 — simulator transparency

Task adapters may wrap MuJoCo/robosuite, but must preserve access to the information needed to debug model state, contacts and controls. Abstraction must not prevent model inspection.

### ENV-003 — algorithm independence

Environment/task modules must not import Stable-Baselines3, JAX trainers or another optimization implementation. Training backends consume the environment contract rather than owning task semantics.

## 3. Robot-model requirements

### ROB-001 — provenance

Every external robot model must record:

- upstream project/repository;
- immutable revision/tag when practical;
- model entry point;
- license;
- model-specific modifications made by this repository, if any.

Large third-party meshes/models must not be silently vendored.

### ROB-002 — machine-checkable interface

For custom G1/H1 tasks, the following must be explicit and tested:

- joint ordering and controlled DoFs;
- qpos/qvel addresses where relevant;
- actuator ordering/type;
- joint/control/force limits;
- bodies/sites/geoms used by reward or termination logic;
- nominal pose;
- physics timestep and integrator.

### ROB-003 — control time scales

Physics timestep, low-level control rate and policy rate/decimation must be separate configuration values or otherwise explicitly derivable and documented.

## 4. Humanoid locomotion requirements

### HUM-001 — command-conditioned policy

The production locomotion task must support a target command containing at least forward velocity and should progress to `[vx, vy, yaw_rate]` tracking. Global x/y position must not be required when translation invariance is intended.

### HUM-002 — explicit action semantics

The first G1 task will use normalized joint targets/residual targets with an explicit scaling and lower-level actuator/PD mapping. Every action dimension must map to a documented physical joint/control quantity and obey configured saturation.

### HUM-003 — decomposed observations

The observation contract must document ordering, units/frame and normalization. Expected groups include base motion/orientation representation, command, joint offsets, joint velocities and previous action; additions must be justified.

### HUM-004 — decomposed rewards

Locomotion reward components must be separately observable. At minimum the design must consider:

- command tracking;
- upright/orientation and base-height behavior;
- torque/control effort;
- joint-limit behavior;
- action-rate/smoothness;
- foot slip/contact quality.

No production reward should exist only as an opaque scalar expression.

### HUM-005 — curriculum before complexity

The default learning progression is standing → slow forward walking → wider velocity commands → lateral/yaw → disturbances/randomization → terrain. Domain randomization must not be used to conceal an unstable nominal task.

### HUM-006 — quantitative evaluation

A walking result must report more than return/video. The evaluator must evolve to include command error, fall/survival statistics, foot slip/contact behavior, action/torque statistics, joint-limit violations and robustness.

## 5. Bimanual manipulation requirements

### BIM-001 — baseline tasks

The state-based benchmark track must cover at least `TwoArmLift`, `TwoArmHandover` and `TwoArmPegInHole` (or documented upstream-equivalent task names if the framework changes).

### BIM-002 — controller semantics

Robot arrangement, controller type, action dimensions, action scaling, control frequency, episode horizon and observation selection must be recorded for each task configuration.

### BIM-003 — success-first evaluation

For tasks with a discrete goal, success rate is the primary outcome. Return may be reported as a diagnostic but must not substitute for task success.

Useful secondary metrics include time-to-success, grasp/drop failures, collisions/safety events and action/path smoothness.

### BIM-004 — state before vision

The first RL baseline uses state/proprioceptive/object observations. Camera observations must use a deliberate vision-policy architecture and may not be silently flattened into the state MLP baseline.

### BIM-005 — imitation-learning data contract

Before ACT/Diffusion/BC training, the project must define a versioned demonstration schema including observation/action timestamps or ordering, episode boundaries, task metadata and deterministic dataset splits. Dataset provenance is mandatory.

## 6. Training requirements

### TRN-001 — configuration-driven behavior

Experiment behavior that changes scientific meaning belongs in checked-in configuration, not hidden CLI literals. The resolved configuration must be copied into the run artifact directory.

### TRN-002 — reproducible artifacts

A reported run must preserve or record:

- git commit;
- resolved config;
- seed;
- relevant package/simulator versions;
- model/asset revision;
- training budget;
- normalization statistics;
- checkpoint identity;
- evaluation conditions.

### TRN-003 — multiple seeds

Claims about RL performance require multiple training seeds. One visually successful trajectory or one lucky seed is not sufficient evidence.

### TRN-004 — evaluation isolation

Evaluation uses a separate rollout path with deterministic policy inference by default and frozen observation normalization where applicable.

### TRN-005 — trainer replaceability

Adding MJX/JAX, RSL-RL, SAC, ACT or another trainer must not require copying/reimplementing the underlying task definition merely to satisfy a different optimizer API.

### TRN-006 — workload-aware runtime scheduling

Hardware accelerator availability, policy/trainer device selection and simulator vectorization are separate contracts.

Acceptance:

- `doctor` reports visible torch accelerators independently from the current workload recommendation;
- `runtime.device: auto` may choose CPU even when CUDA is available when the named workload is CPU-preferred;
- explicit `device: cuda|mps|cpu` overrides the automatic policy and fails fast if an explicitly requested accelerator is unavailable;
- native SB3 vectorization is configured independently through `runtime.vec_env_backend`;
- `DummyVecEnv` and `SubprocVecEnv` choices are benchmarkable on the target machine rather than inferred from environment count alone;
- run metadata records the resolved device/backend and selection reasons;
- accelerator throughput claims identify backend, batch/environment count and hardware.

The initial workload policy is that native MuJoCo + SB3 PPO + `MlpPolicy` uses CPU when `device: auto`. Large-batch MJX/JAX simulation is a distinct accelerator workload and must not be modeled as merely another torch-device setting.

## 7. Dependency and platform requirements

### DEP-001 — optional dependency isolation

Bimanual, training, vision/data and future accelerator stacks should remain optional where practical. Importing the core package must not require every heavy optional framework.

### DEP-002 — compatibility boundaries

When an upstream framework requires a simulator-version window, encode that constraint in the smallest relevant dependency group. Do not globally downgrade unrelated tracks.

The bootstrap example is robosuite: the `bimanual` extra constrains MuJoCo to its compatible range while the humanoid core can adopt newer MuJoCo releases.

### DEP-003 — headless separation

Non-rendering dynamics tests must not require a graphics display. Rendering/image tests should run in a graphics-aware environment/job with an explicit backend.

## 8. AI-coding requirements

### AI-001 — spec before implementation

A non-trivial AI-generated change starts from an issue/spec that defines behavior and acceptance gates. The agent must not choose undocumented reward/control semantics merely because they are convenient to implement.

### AI-002 — vertical slices

Prefer one inspectable vertical slice per PR: model → task contract → tests → short execution evidence → docs. Avoid large generated rewrites spanning unrelated phases.

### AI-003 — upstream verification

Before using external simulator/framework APIs, verify current official docs/source. Do not invent function names, defaults, robot limits or compatibility claims.

### AI-004 — evidence-based completion

An AI-generated PR is complete only when code, config, tests, tutorial/ADR and executed verification agree. “Code generated successfully” is not a robotics acceptance criterion.

### AI-005 — preserve failures

Meaningful failed experiments/integration findings should be recorded when they teach a dependency, simulator, reward or control boundary. Do not erase negative evidence by repeatedly changing parameters until one run looks plausible.

## 9. Quality requirements

### QLT-001 — CI

Every PR must pass formatting, linting and fast contract/smoke tests. Full RL training is not required in ordinary CI, but environment construction/reset/step should be tested for supported baseline tracks. Runtime/backend changes must execute the affected backend at least once in CI when the hosted platform supports it; configuration-only tests are insufficient for multiprocessing boundaries.

### QLT-002 — documentation co-change

When a change modifies observation/action semantics, reward terms, model provenance, dependency constraints, commands, runtime/backend policy or experiment protocol, the affected documentation must change in the same PR.

### QLT-003 — no large generated artifacts in Git

Checkpoints, TensorBoard runs, datasets and large robot assets remain out of Git unless a future explicit artifact policy says otherwise.

## 10. Future sim-to-real requirements

Hardware deployment is out of scope for the bootstrap but the architecture must not block it. Before a policy can be considered hardware-ready, the project must make machine-checkable:

- joint ordering/sign conventions/units;
- input/output normalization;
- policy and actuator update rates;
- position/velocity/torque/rate limits;
- estimator assumptions;
- latency/watchdog behavior;
- safe fallback behavior;
- policy/model/interface version compatibility;
- replay/shadow-mode validation.

A simulation video does not satisfy these requirements.

## 11. Definition of done by milestone

A milestone is done only when all applicable requirements above are met **and** its roadmap acceptance gate is demonstrated with executed evidence. Documentation may describe planned work, but planned work must never be labeled as an achieved benchmark.
