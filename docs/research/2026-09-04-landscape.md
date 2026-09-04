# MuJoCo humanoid + bimanual landscape — 2026-09-04

This note records the sources used to choose the initial architecture. Official documentation is preferred for API/behavior claims; small GitHub projects are treated as implementation references, not authorities.

## Primary sources

### MuJoCo Python and model APIs

- MuJoCo Python bindings: https://mujoco.readthedocs.io/en/stable/python.html
- Model editing / `mjSpec`: https://mujoco.readthedocs.io/en/stable/programming/modeledit.html
- Computation, timestep and integrators: https://mujoco.readthedocs.io/en/latest/computation/
- MJX: https://mujoco.readthedocs.io/en/latest/mjx.html

Decision: use native MuJoCo semantics as the simulator source of truth; keep MJX as a later acceleration backend after task semantics are validated.

### MuJoCo Menagerie

- Repository: https://github.com/google-deepmind/mujoco_menagerie

Menagerie is a curated set of robot models and currently lists Unitree H1 (19 DoF) and G1 (29 DoF), among multiple humanoids. It supports direct loading and `robot_descriptions`, with model-specific licenses.

Decision: do not vendor G1/H1 meshes. Resolve a pinned external model revision and record provenance.

### Gymnasium Humanoid

- Environment: https://gymnasium.farama.org/environments/mujoco/humanoid/

`Humanoid-v5` provides a canonical MuJoCo continuous-control benchmark with a 17-dimensional action space.

Decision: use it only for the first end-to-end PPO tutorial, then move to an explicit G1 task.

### Stable-Baselines3

- RL tips: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
- Documentation: https://stable-baselines3.readthedocs.io/

The project recommends multiple runs because RL results vary by seed, notes the importance of normalization for continuous control, and supports PPO for vectorized continuous-action environments.

Decision: PPO is the readable baseline; every serious result must use multiple seeds and saved normalization state.

### robosuite

- Environments: https://robosuite.ai/docs/modules/environments.html
- Gym wrapper source: https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/wrappers/gym_wrapper.py
- Installation: https://robosuite.ai/docs/installation.html

robosuite supplies `TwoArmLift`, `TwoArmHandover` and `TwoArmPegInHole`, plus a Gym/Gymnasium-compatible wrapper.

Decision: use robosuite rather than rebuilding bimanual benchmark scenes in Phase 1–4.

### ALOHA / ACT

- ACT repository: https://github.com/tonyzhaozh/act

The project includes MuJoCo + `dm_control` simulation for Transfer Cube and Bimanual Insertion, scripted/human demonstrations and ACT training.

Decision: use this as the conceptual bridge from state-based bimanual RL to demonstration-driven imitation learning. Do not copy its code wholesale; define our own dataset and policy adapters with attribution.

### dm_control

- Repository: https://github.com/google-deepmind/dm_control

`dm_control` provides MuJoCo bindings, suite, MJCF composition, Composer and locomotion components.

Decision: study Composer/MJCF patterns, but avoid making both Gymnasium and `dm_env` first-class environment APIs in the bootstrap. One public environment contract reduces accidental complexity.

## Secondary implementation references

- Unitree official MuJoCo RL project: https://github.com/unitreerobotics/unitree_rl_mjlab
- Minimal G1 PPO/MuJoCo example: https://github.com/adamson-gh/g1-locomotion-rl
- G1 locomotion + manipulation study: https://github.com/idan0405/Unitree-G1-RL

These are useful for reward/hyperparameter/experiment ideas, but every borrowed idea must be re-derived against the current robot model and documented before adoption.

## Architecture consequences

1. **Gymnasium is the interoperability boundary**, not the internal simulator truth.
2. **Tasks and trainers are separate** so SB3 can later be replaced by MJX/JAX or another RL stack.
3. **Robot assets stay external and versioned**.
4. **Bimanual manipulation starts from robosuite** to avoid spending the first milestones on scene plumbing.
5. **Humanoid work starts canonical, then becomes robot-specific**: Humanoid-v5 teaches the loop; G1/H1 teaches real task engineering.
6. **Documentation and experiment logs are first-class artifacts**, because reward design and simulator settings are part of the executable specification.
