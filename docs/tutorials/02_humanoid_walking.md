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

## Stage B: move to Unitree G1

The next environment uses the Unitree G1 model from MuJoCo Menagerie. Menagerie currently provides a 29-DoF G1 model and maintains model-specific licensing/readme information.

The first custom G1 task should lock upper-body joints if necessary and train a simpler leg/waist policy before exposing all DoFs. This is a curriculum decision, not a permanent architectural limitation.

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

## Action design

Start with normalized residual joint targets:

```text
a in [-1, 1]
q_target = q_nominal + action_scale * a
tau = kp * (q_target - q) - kd * qdot
```

Then clip target/torque against explicit robot limits. The exact gains and action scale are part of the task definition and must be versioned.

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

1. stand without falling;
2. track a small positive forward command;
3. expand forward-speed range;
4. add lateral and yaw commands;
5. add command changes during an episode;
6. add pushes and dynamics randomization;
7. add terrain.

## Evaluation

Walking quality is not “the video looks human”. Measure command error, survival/fall rate, foot slip, contact alternation, action smoothness, joint-limit violations and robustness. Use multiple training seeds and fixed evaluation conditions.
