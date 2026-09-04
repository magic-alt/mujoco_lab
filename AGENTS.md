# AGENTS.md

Instructions for Codex, Copilot, Claude Code and other AI coding agents working in this repository.

## Mission

Build a reproducible, tutorial-quality MuJoCo research platform for two tracks:

1. humanoid locomotion, progressing from Gymnasium Humanoid to Unitree G1/H1 and sim-to-real concerns;
2. bimanual manipulation, progressing from robosuite state-based RL to demonstrations and imitation learning.

A change is successful only when the code, tests, documentation and experiment contract agree.

## Required workflow

1. Read `docs/architecture.md`, `docs/roadmap.md` and the tutorial relevant to the issue.
2. Restate the acceptance criteria in the working plan before editing.
3. Implement one vertical slice. Avoid unrelated refactors.
4. Add or update tests before claiming completion.
5. Run the local gates documented in `CONTRIBUTING.md`.
6. Update `docs/development-log.md` and any affected tutorial/ADR.
7. In the PR, state what was executed locally and what remains unverified.

## Architectural boundaries

- `envs/` may depend on MuJoCo/Gymnasium/robosuite, but **must not import Stable-Baselines3 or a trainer**.
- `training/` may depend on environment factories, never on private simulator internals.
- Experiment behavior belongs in checked-in YAML under `configs/`, not hard-coded magic values in CLIs.
- Robot assets and large datasets are external dependencies. Do not silently vendor Menagerie, ALOHA data, checkpoints or third-party meshes.
- Import optional dependencies lazily and raise actionable installation errors.
- Keep reset/step semantics compatible with Gymnasium.
- Preserve deterministic seeding where the upstream simulator permits it.

## Robotics correctness checklist

For any new robot/task, document:

- source and license of the model;
- generalized coordinates, controlled DoFs and actuator type;
- simulation timestep and policy/control decimation;
- action scaling and saturation;
- observation ordering and normalization;
- termination conditions;
- every reward term, sign, unit/normalization and weight;
- contact geoms/sites used by reward or termination logic;
- initial-state randomization;
- evaluation metrics and success gate.

Do not tune reward weights before adding diagnostics that expose each reward component.

## RL experiment rules

- Never infer convergence from one seed.
- Keep evaluation separate from training rollouts.
- Record resolved config, seed, package versions and checkpoint path.
- Normalize continuous-control observations when appropriate and save normalization statistics with the checkpoint.
- Prefer a small smoke budget in CI; full training belongs in reproducible documented commands, not CI.
- Do not commit a policy because a rendered video looks good without quantitative evaluation.

## AI-generated code rules

- Do not fabricate API names. Verify upstream APIs against official docs/source before integrating them.
- Do not copy large external code blocks; implement against documented interfaces and preserve attribution for adapted ideas.
- Avoid broad dependency additions. Every new dependency requires a short justification in the PR.
- Keep patches reviewable: one issue, one coherent PR, explicit acceptance gates.
- When an experiment fails, preserve the failure as a note or test rather than hiding it with a larger hyperparameter search.
