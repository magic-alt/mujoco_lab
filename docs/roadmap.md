# Roadmap

The roadmap is organized as testable vertical slices. A phase is complete only when its acceptance gates are met and documented.

## Phase 0 — repository and experiment foundation

**Status: implemented in bootstrap PR.**

Deliverables:

- packaged `src/` layout and CLI;
- YAML experiment schema;
- common Gymnasium environment factory;
- SB3 PPO training/evaluation artifact path;
- lint/test CI;
- AI-agent development rules;
- architecture, research notes and tutorial skeleton.

Gate: a clean environment can run `doctor`, `inspect-env`, lint and tests.

## Phase 1 — canonical MuJoCo learning loop

**Status: baseline included; empirical training validation remains in #2.**

Deliverables:

- Gymnasium `Humanoid-v5` PPO config;
- deterministic evaluation command;
- normalization statistics saved with policy;
- workload-aware device selection rather than unconditional GPU preference;
- configurable `DummyVecEnv` / `SubprocVecEnv` execution with machine-specific throughput benchmark;
- 3–5 seed training report with return curves and wall-clock/step throughput.

Gate: repeated runs produce a clearly improving locomotion policy and evaluation report; no result is accepted from one seed alone.

## Phase 2 — Unitree G1 locomotion and early acceleration

Phase 2 is intentionally split so task semantics are proven before walking complexity, while accelerator throughput is measured early enough to influence the expensive training phases.

### Phase 2A — pinned G1 model contract

**Status: completed in #3 / PR #12.**

Deliverables:

- versioned external Menagerie G1 model resolver;
- immutable upstream revision + license provenance;
- explicit joint/actuator/site/contact contract;
- stand keyframe and actuator-limit inspection;
- native one-step finite-state validation.

Gate: the pinned model resolves, compiles and passes the machine-checkable contract in CI.

### Phase 2B — native G1 standing reference (#4)

Deliverables:

- native MuJoCo CPU reference task;
- explicit controlled-joint map and actuator contract;
- normalized joint-target or residual-target action space;
- configurable PD layer and action decimation;
- projected gravity/base angular velocity/joint-offset/joint-velocity/previous-action observation contract;
- standing/upright/base-height reward diagnostics;
- fall/timeout termination;
- deterministic reset contract.

Gate: zero/random action rollouts are numerically stable and a short PPO run can improve standing survival without any walking reward. This environment becomes the semantic reference for accelerator work.

### Phase 2C — early MJX/JAX throughput prototype (#9)

Deliverables:

- map the Phase 2B standing contract onto the minimum compatible MJX subset without silently changing semantics;
- benchmark 512, 1024 and 2048+ batched G1 environments where device memory permits;
- report compile time, environment steps/s, device memory and wall-clock on specified hardware;
- first target profile: RTX 5060 8 GB, with results explicitly treated as hardware-specific;
- CPU/MJX reset/observation/action/reward/termination comparisons for matched states where practical;
- preserve the native CPU standing task as the correctness reference.

Gate: at least one feasible accelerator batch materially improves aggregate environment throughput over the native reference, or the bottleneck/blocker is quantitatively documented. Backend semantic differences remain explicit.

### Phase 2D — command-conditioned G1 walking (#5)

Deliverables:

- command observation `[vx, vy, yaw_rate]`;
- decomposed reward diagnostics: command tracking, upright/base orientation, base height, foot slip, air time/contact schedule if used, joint-limit/velocity/torque penalties, action-rate/smoothness;
- standing → slow forward → commanded velocity curriculum;
- fixed held-out command evaluation;
- multi-seed training and backend identification in every reported result.

Gate: policy follows several held-out velocity commands, remains upright, alternates contacts without pathological foot sliding, and passes a documented robustness sanity check.

## Phase 3 — robust humanoid locomotion (#6)

Deliverables:

- H1 as a second embodiment to prove task abstraction;
- terrain and push perturbations;
- mass/inertia/friction/motor-strength/sensor-noise/latency randomization;
- reward and randomization ablations;
- ONNX or equivalent policy export contract;
- sim-to-sim replay harness.

Gate: quantify robustness rather than showing only nominal walking video.

## Phase 4 — bimanual state-based RL (#7)

Deliverables:

- robosuite `TwoArmLift`, `TwoArmHandover`, `TwoArmPegInHole` configs;
- task-specific success evaluator;
- PPO baseline plus at least one off-policy continuous-control comparison where appropriate;
- controller/action-space tutorial (joint vs operational-space control);
- demonstrations of failure cases and reward shaping.

Gate: report success rate over fixed evaluation seeds, not just episodic return.

## Phase 5 — demonstrations and imitation learning (#8)

Deliverables:

- robosuite data collection/replay path;
- ALOHA-style transfer-cube and bimanual-insertion study;
- dataset schema and provenance checks;
- behavior cloning baseline;
- ACT baseline;
- optional Diffusion Policy baseline;
- image/proprioception observation pipeline.

Gate: deterministic dataset split and reproducible held-out success evaluation.

## Phase 6 — accelerator backend hardening

The original MJX work has been split: Phase 2C proves early throughput on the standing task, while this later phase turns the prototype into a production research backend after walking and manipulation requirements are better understood.

Deliverables:

- broader G1/H1 task coverage on the accelerator backend;
- domain-randomization batching and memory/performance tuning;
- stable backend/trainer configuration contract;
- stronger native/MJX semantic regression suite;
- multi-device scaling where justified;
- documented MJX-JAX/MJX-Warp tradeoffs if both are evaluated.

Gate: demonstrated end-to-end training throughput improvement with no silent task-definition drift and a maintained native reference path.

## Phase 7 — sim-to-real research path (#10)

Deliverables:

- actuator/gearbox/latency abstraction aligned with target joint modules;
- sensor and command interfaces;
- policy safety envelope and saturation;
- log replay and policy shadow-mode tools;
- sim-to-real checklist and hardware-in-the-loop plan.

Gate: deployment is blocked until policy outputs, units, limits, watchdog behavior and failure recovery are explicit.

## Cross-cutting work

Every phase also carries:

- architecture decision records when dependencies or contracts change;
- tutorial updates in the same PR as code changes;
- reproducibility metadata;
- quantitative evaluation scripts;
- performance claims that name hardware, backend and environment count;
- license/provenance checks for external assets and datasets.
