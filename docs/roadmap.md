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

**Status: baseline included; empirical training validation remains.**

Deliverables:

- Gymnasium `Humanoid-v5` PPO config;
- deterministic evaluation command;
- normalization statistics saved with policy;
- 3–5 seed training report with return curves and wall-clock/step throughput.

Gate: repeated runs produce a clearly improving locomotion policy and evaluation report; no result is accepted from one seed alone.

## Phase 2 — Unitree G1 walking task

Deliverables:

- versioned external Menagerie G1 model resolver;
- explicit controlled-joint map and actuator contract;
- command observation `[vx, vy, yaw_rate]`;
- normalized joint-target or residual-target action space;
- configurable PD layer and action decimation;
- decomposed reward diagnostics: command tracking, upright/base orientation, base height, foot slip, air time/contact schedule if used, joint-limit/velocity/torque penalties, action-rate/smoothness;
- fall/timeout termination;
- standing → slow forward → commanded velocity curriculum;
- multi-seed evaluation.

Gate: policy follows several held-out velocity commands, remains upright, alternates contacts without pathological foot sliding, and passes a documented robustness test.

## Phase 3 — robust humanoid locomotion

Deliverables:

- H1 as a second embodiment to prove task abstraction;
- terrain and push perturbations;
- mass/inertia/friction/motor-strength/sensor-noise/latency randomization;
- reward and randomization ablations;
- ONNX or equivalent policy export contract;
- sim-to-sim replay harness.

Gate: quantify robustness rather than showing only nominal walking video.

## Phase 4 — bimanual state-based RL

Deliverables:

- robosuite `TwoArmLift`, `TwoArmHandover`, `TwoArmPegInHole` configs;
- task-specific success evaluator;
- PPO baseline plus at least one off-policy continuous-control comparison where appropriate;
- controller/action-space tutorial (joint vs operational-space control);
- demonstrations of failure cases and reward shaping.

Gate: report success rate over fixed evaluation seeds, not just episodic return.

## Phase 5 — demonstrations and imitation learning

Deliverables:

- robosuite data collection/replay path;
- ALOHA-style transfer-cube and bimanual-insertion study;
- dataset schema and provenance checks;
- behavior cloning baseline;
- ACT baseline;
- optional Diffusion Policy baseline;
- image/proprioception observation pipeline.

Gate: deterministic dataset split and reproducible held-out success evaluation.

## Phase 6 — MJX acceleration

Deliverables:

- identify model/task subset compatible with MJX;
- batched environment backend;
- throughput benchmark against CPU MuJoCo baseline;
- algorithm backend that does not change task semantics;
- regression tests comparing reset/observation/reward semantics where feasible.

Gate: demonstrated throughput improvement with no silent task-definition drift.

## Phase 7 — sim-to-real research path

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
- license/provenance checks for external assets and datasets.
