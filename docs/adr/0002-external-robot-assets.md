# ADR 0002 — Keep curated robot assets external and versioned

- Status: Accepted
- Date: 2026-09-04

## Context

High-quality humanoid and manipulator models include meshes and model-specific licenses. Copying them into this tutorial repository creates provenance, update and repository-size problems. Pulling a mutable `main` branch at experiment time is also insufficient: an upstream model change could silently alter dynamics, joints, contacts or actuator semantics between training runs.

## Decision

Prefer maintained upstream assets such as MuJoCo Menagerie and robosuite models. Do not commit third-party meshes or whole external repositories. A task that depends on an external robot must record upstream URL, immutable revision, model-specific license and expected model path.

The first implementation is Unitree G1 from `google-deepmind/mujoco_menagerie`, pinned to:

```text
e4049d0a3bfd58d2a3081614e6777d4007e3f86a
```

`mujoco_lab` resolves only the required G1 MJCF, README, LICENSE and mesh files into a revision-scoped local cache. A SHA-256 manifest is written after download and validated on every reuse. Revision metadata mismatch, missing files and hash mismatch are hard failures rather than warnings.

The original third-party files remain byte-for-byte upstream artifacts. Task-specific derived models, such as the torque-actuated model planned for the standing environment, must be generated outside the repository from this validated source and carry source-revision metadata.

## Model contract before RL

Before any G1 environment or trainer is allowed to depend on the model, `mujoco-lab inspect-robot g1` must compile the pinned scene and validate machine-checkable assumptions including:

- 29 articulated joints and 29 actuators;
- required leg/waist controlled joints;
- required pelvis/foot bodies and IMU/foot sites;
- at least four collision geoms per foot;
- the upstream `stand` keyframe inside joint limits;
- finite state after reset, forward dynamics and one MuJoCo step;
- timestep, integrator and actuator force metadata recorded in the inspection report.

## Consequences

- the repository remains small;
- model updates are deliberate code-review events;
- experiments remain reproducible against an immutable source revision;
- licensing stays visible;
- corrupted or manually edited caches are detected;
- offline use works after one successful asset resolution;
- CI may use network access for a pinned upstream integration smoke test, while unit tests remain network-independent.
