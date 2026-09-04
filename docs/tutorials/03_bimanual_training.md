# 03 — Bimanual manipulation

## Why robosuite first

robosuite is MuJoCo-based and already includes coordinated two-arm tasks such as:

- `TwoArmLift` — two arms grasp and lift a pot while keeping it level;
- `TwoArmHandover` — one arm transfers an object to the other;
- `TwoArmPegInHole` — coordinated insertion.

Its `GymWrapper` converts the environment to the Gym/Gymnasium-style API and can flatten proprioceptive/object observations for a standard MLP policy. This lets the tutorial focus first on control/reward/task design rather than reimplementing simulator plumbing.

## First baseline: TwoArmLift

Install the optional dependency and inspect the environment:

```bash
uv sync --extra train --extra bimanual --extra dev
uv run mujoco-lab inspect-env configs/bimanual/two_arm_lift_ppo.yaml
```

Then train:

```bash
uv run mujoco-lab train configs/bimanual/two_arm_lift_ppo.yaml
```

The checked-in config uses two Panda arms in parallel, state observations and shaped reward. Treat its PPO settings as a baseline, not a claim that they are optimal.

## What to inspect before tuning PPO

1. Action dimensions and controller semantics.
2. Whether each gripper can reach its intended handle.
3. Initial object-placement distribution.
4. Reward component progression from reaching → grasp → lift/level.
5. Episode truncation vs true task termination.
6. Success rate, not only scalar return.

## From RL to imitation learning

Bimanual contact tasks can be exploration-heavy. The ALOHA/ACT line of work is useful because its simulation examples include Transfer Cube and Bimanual Insertion with MuJoCo + `dm_control`, scripted/human demonstrations and ACT training.

Planned progression:

```text
robosuite state RL
      ↓
data collection + replay
      ↓
behavior cloning
      ↓
ACT action chunks
      ↓
vision + proprioception
      ↓
Diffusion Policy / hybrid fine-tuning
```

This repository will keep data schema and policy code separate from the simulator adapter so the same demonstrations can be audited and reused.

## Control-space lesson

Two-arm tasks are a good place to compare joint-space and operational-space controllers. A higher-level OSC action can make exploration easier but also hides low-level dynamics. When the goal moves toward real actuator control, make controller assumptions explicit and progressively lower the abstraction rather than silently swapping semantics.
