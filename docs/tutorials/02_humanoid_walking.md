# 02 — Humanoid walking

## Stage A: learn the RL loop with `Humanoid-v5`

Gymnasium's MuJoCo `Humanoid-v5` has a continuous 17-dimensional action space and a dense locomotion reward. It is not our final robot model; it is the fastest way to learn the train/evaluate/debug loop on a canonical benchmark.

```bash
uv run mujoco-lab train configs/humanoid/humanoid_v5_ppo.yaml
```

Questions to answer before moving on:

- Does episodic return improve across multiple seeds?
- Does evaluation remain stable when observation normalization is frozen?
- What fraction of return comes from forward progress vs healthy/control terms?
- Is a visually upright policy actually progressing forward?

The full multi-seed answer belongs to issue #2. A single successful video is not treated as benchmark evidence.

## Stage B0: establish the Unitree G1 model contract

Before writing a standing reward or PD controller, validate the actual robot model that later tasks will depend on. `mujoco_lab` uses the Unitree G1 29-DoF model from MuJoCo Menagerie and pins the upstream repository to an immutable commit.

```bash
uv run mujoco-lab inspect-robot g1
```

The first run downloads only the required G1 MJCF, README, LICENSE and meshes into the local cache. Repeated runs validate the revision-scoped SHA-256 manifest instead of downloading again.

For a custom cache root:

```bash
uv run mujoco-lab inspect-robot g1 --cache-root ./tmp/mujoco_lab_cache
```

After one successful online resolution, verify that the same model works without network access:

```bash
uv run mujoco-lab inspect-robot g1 --offline
```

Use `--force` only when deliberately rebuilding the cache. A revision mismatch or hash mismatch is considered an error, because silently accepting changed robot assets would invalidate experiment reproducibility.

The inspection contract checks:

- 29 articulated joints and 29 joint-matched position actuators;
- the initial 15 leg/waist joints planned for the first locomotion curriculum;
- pelvis, ankle-roll bodies, IMU sites and left/right foot sites;
- at least four collision geoms per foot;
- the upstream `stand` keyframe against joint limits;
- finite state after reset, forward dynamics and one MuJoCo step;
- timestep, integrator, actuator control metadata and actuator force ranges.

The report is printed as JSON and persisted as `inspection.json` beside the cached revision. Record the revision from this report in every downstream experiment artifact.

This step deliberately contains **no RL and no custom PD controller**. If the robot model contract fails, fix the asset/model layer before touching rewards or training code.

## Stage B1: stand before walking

Issue #4 builds the first custom G1 environment on top of the validated model contract. The initial policy should control a simpler leg/waist subset before exposing all 29 DoFs. This is a curriculum decision, not a permanent architectural limitation.

The first action contract will use normalized residual joint targets:

```text
a in [-1, 1]
q_target = q_nominal + action_scale * a
tau = kp * (q_target - q) - kd * qdot
```

The task-specific torque-actuated MJCF should be generated from the validated pinned source at runtime rather than committed as a modified copy of Menagerie assets. Target and torque must be clipped against explicit limits, and the exact gains/action scale become versioned task configuration.

Standing comes first because it isolates model, reset, observation, action, PD and termination semantics from walking reward engineering.

## Stage B2: command-conditioned walking

Only after standing is numerically stable and learnable does issue #5 add target commands `[vx, vy, yaw_rate]` and locomotion rewards.

## Observation design

A command-conditioned locomotion observation will typically include a subset of:

- base angular velocity in body frame;
- projected gravity / base orientation representation;
- commanded `[vx, vy, yaw_rate]`;
- joint positions relative to nominal pose;
- joint velocities;
- previous action;
- optional foot/contact information.

Avoid feeding global x/y position if the desired policy should be translation-invariant.

## Reward design

Do not begin with one opaque scalar function. Implement named components such as:

```text
r = w_track * r_velocity_tracking
  + w_upright * r_upright
  + w_height * r_base_height
  + w_air * r_foot_air_time
  - w_slip * p_foot_slip
  - w_torque * p_torque
  - w_rate * p_action_rate
  - w_limits * p_joint_limits
```

Log every component. A policy can maximize a scalar reward through an unintended shortcut; decomposed metrics reveal it.

## Curriculum

Recommended progression:

1. validate the pinned robot model contract;
2. stand without falling;
3. track a small positive forward command;
4. expand forward-speed range;
5. add lateral and yaw commands;
6. add command changes during an episode;
7. add pushes and dynamics randomization;
8. add terrain.

## Evaluation

Walking quality is not “the video looks human”. Measure command error, survival/fall rate, foot slip, contact alternation, action smoothness, joint-limit violations and robustness. Use multiple training seeds and fixed evaluation conditions.
