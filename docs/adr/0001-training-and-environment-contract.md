# ADR 0001 — Gymnasium environment contract + pluggable trainers

- Status: Accepted
- Date: 2026-09-04

## Context

The project needs both humanoid locomotion and bimanual manipulation. MuJoCo, Gymnasium, robosuite, `dm_control`, MJX and multiple RL libraries expose different environment/training abstractions. Letting each tutorial choose its own stack would quickly create two or more incompatible mini-projects.

## Decision

Use Gymnasium's reset/step/spaces semantics as the public environment contract. Keep simulator/task implementations under `envs/` and optimization code under `training/`.

The initial trainer is Stable-Baselines3 PPO. MJX/JAX or RSL-RL can be added later as independent backends without altering task semantics.

## Consequences

Positive:

- simple interface for learners;
- robosuite can use its existing Gym wrapper;
- algorithm swaps do not require robot/task rewrites;
- task tests can run without the RL library.

Tradeoffs:

- some accelerator-native APIs will need adapters;
- the abstraction must not hide MuJoCo-specific quantities needed for robotics debugging.
