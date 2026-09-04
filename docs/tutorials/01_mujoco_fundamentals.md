# 01 — MuJoCo fundamentals for robot learning

The goal of this chapter is to make the simulator observable before using RL to hide mistakes behind a neural network.

## Model vs data

MuJoCo separates the mostly-static compiled model (`MjModel`) from mutable simulation state (`MjData`). In robot tasks you will repeatedly inspect:

- `qpos`: generalized positions;
- `qvel`: generalized velocities;
- actuator controls / forces;
- body/site transforms;
- contacts and contact forces;
- simulation time.

A Gymnasium wrapper may flatten these into a task observation, but the underlying MuJoCo state remains the debugging ground truth.

## MJCF and assets

Prefer a maintained MJCF model when available. Menagerie is valuable because it curates models rather than treating URDF conversion as finished simulation work. Keep the robot-only model separate from scene/task objects where possible.

For this repository, external robot assets are referenced rather than copied. The model source, revision and license belong in the task documentation.

## Timestep, control rate and decimation

If physics runs at 500 Hz and the policy acts at 50 Hz, the same policy action is applied for ten physics steps. These rates must be documented separately.

Do not “fix” training speed by increasing MuJoCo timestep without a stability study. MuJoCo's documentation calls timestep one of the most important model parameters and generally recommends `implicitfast` for a good stability/performance tradeoff in typical systems.

## Contacts

Walking and manipulation are contact problems. Check:

- collision geometry is simpler than visual geometry where appropriate;
- feet/fingers and the ground/object have sensible friction;
- initial state is not penetrating;
- contact pairs you use in rewards are named and tested;
- contact-related reward code is robust to multiple simultaneous contacts.

## Actuation

The policy should not directly emit an undefined “action”. Specify whether action means:

- torque;
- motor command;
- joint velocity;
- absolute joint position target;
- residual around a nominal joint pose;
- Cartesian/operational-space command.

Humanoid locomotion commonly benefits from a normalized policy action followed by joint-target scaling and a lower-level PD layer. Bimanual tasks in robosuite can start with its composite controller abstractions before dropping to lower-level actuation.

## First debugging exercise

Before training a custom robot:

1. load the model;
2. print joint/actuator/site names and limits;
3. reset to the nominal pose;
4. simulate zero action for several seconds;
5. apply one actuator at a time;
6. inspect contacts and joint-limit behavior;
7. only then build reward terms.

The Phase 2 G1 work will turn these steps into executable model-inspection tests.
