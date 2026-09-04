# Development log

## 2026-09-04 — Bootstrap architecture

### Goal

Turn the empty repository into a reproducible MuJoCo learning lab for humanoid walking and bimanual manipulation.

### Research performed

Reviewed current MuJoCo Python/model/MJX documentation, MuJoCo Menagerie, Gymnasium Humanoid, Stable-Baselines3 continuous-control guidance, robosuite two-arm environments/GymWrapper, `dm_control`, ALOHA/ACT simulation structure and several Unitree G1 MuJoCo/RL implementations. See `docs/research/2026-09-04-landscape.md`.

### Decisions

- Gymnasium is the public environment contract.
- SB3 PPO is the first trainer, not a permanent framework lock-in.
- Menagerie is the preferred source for G1/H1 assets.
- robosuite is the initial bimanual task framework.
- experiments are config-driven and write resolved config + runtime metadata + normalization state.
- AI coding follows issue/spec → vertical slice → tests → docs → experiment evidence.

### Implemented

- packaging and CLI;
- config validation;
- Humanoid-v5 factory and PPO baseline;
- robosuite two-arm factory and PPO baseline config;
- evaluation path;
- CI and smoke tests;
- architecture/roadmap/ADR/tutorial/research docs.

### Explicitly not claimed yet

No full training run has been executed by this bootstrap change. The checked-in hyperparameters are starting baselines, not benchmark results. Phase 1 requires empirical multi-seed validation; Phase 2 implements the first Unitree G1 custom locomotion task.
