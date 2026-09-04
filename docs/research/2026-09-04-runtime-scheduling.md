# Runtime scheduling and vectorization notes — 2026-09-04

This note records the upstream evidence behind the workload-aware runtime change. It is about execution architecture, not a claim that one backend is universally faster.

## Stable-Baselines3 VecEnv default

Stable-Baselines3 documents `make_vec_env(..., vec_env_cls=None)` as using `DummyVecEnv` by default and notes that it is usually faster than `SubprocVecEnv`. This is why `mujoco_lab` does not automatically map `n_envs > 1` to subprocesses. Process topology must be measured on the actual task and host.

Source: <https://stable-baselines3.readthedocs.io/en/master/common/env_util.html>

SB3 also documents multiprocessing examples that pair CPU execution with `SubprocVecEnv` for appropriate workloads. This supports exposing subprocess vectorization as a first-class option without treating it as an unconditional optimization.

Source: <https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html>

## PPO/MLP accelerator boundary

The first `mujoco_lab` trainer uses native MuJoCo plus SB3 PPO with `MlpPolicy`. Native MuJoCo physics remains CPU-side. A small MLP can therefore leave most wall-clock work on CPU while adding CPU/GPU transfers and synchronization. The repository treats CUDA availability and current-workload recommendation as separate facts; explicit CUDA remains available for controlled comparison.

Related SB3 discussion: <https://github.com/DLR-RM/stable-baselines3/issues/1245>

## MJX accelerator model

MuJoCo's MJX documentation describes batch dimensions as the natural representation for high-throughput reinforcement-learning simulation. It also warns that single-scene MJX-JAX can be substantially slower than optimized CPU MuJoCo and that the backend is most suitable for thousands or tens of thousands of parallel scenes.

Source: <https://mujoco.readthedocs.io/en/latest/mjx.html>

This distinction motivates the roadmap split:

1. native MuJoCo standing is the correctness/reference environment;
2. an early MJX prototype benchmarks 512/1024/2048+ G1 standing environments on available hardware;
3. command-conditioned walking is added after the execution boundary is measured;
4. later accelerator hardening expands task coverage and semantic regression.

The first target machine for the early benchmark is an RTX 5060 8 GB. That target is useful for engineering the project but any throughput result must be labeled as hardware-, software-version- and batch-size-specific.
