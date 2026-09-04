# ADR 0003 — Separate workload scheduling from accelerator availability

## Status

Accepted.

## Context

A CUDA-capable PyTorch installation does not imply that every robotics RL workload should run its policy network on CUDA. The first `Humanoid-v5` baseline uses native MuJoCo physics on CPU and SB3 PPO with a small `MlpPolicy`. In that workload, simulator stepping and Python/NumPy integration dominate enough of the wall clock that moving only the MLP to a GPU can provide little benefit or add synchronization overhead.

Vectorized simulation is a separate decision. SB3 defaults to `DummyVecEnv`; `SubprocVecEnv` can exploit multiple CPU processes but adds process startup, serialization and IPC costs. The faster choice depends on the task and host machine.

MJX is different again: its accelerator value comes from batching many identical physics scenes on device, not from moving a small policy network while leaving native MuJoCo on CPU.

## Decision

1. `doctor` reports hardware/torch accelerator availability independently from the current trainer recommendation.
2. `runtime.device: auto` is resolved using a named workload profile. For native MuJoCo + SB3 PPO + `MlpPolicy`, `auto` prefers CPU. Explicit `cuda`, `mps` or `cpu` always overrides the automatic policy and fails fast if an explicitly requested accelerator is unavailable.
3. `runtime.vec_env_backend` is independent of torch device selection and accepts `auto`, `dummy` or `subproc`.
4. VecEnv `auto` stays on the conservative `DummyVecEnv` baseline. The repository will not guess that multiprocessing is faster without a machine-specific benchmark.
5. `benchmark-vec-env` measures VecEnv startup cost and steady-state environment transitions/s without policy compute. This keeps simulator parallelism evidence separate from neural-network acceleration.
6. The first G1 standing task remains a native MuJoCo CPU semantic reference. An early MJX milestone will then test batched standing throughput at 512/1024/2048+ environments before command-conditioned walking is scaled up.

## Consequences

- A machine may report `CUDA available: True` while an `auto` native PPO run intentionally selects CPU. This is expected and recorded with a reason.
- Users can still force CUDA for controlled comparisons.
- Dummy/Subproc decisions become reproducible configuration rather than hidden SB3 defaults.
- Performance claims must name the workload, simulator backend, VecEnv backend, environment count and hardware.
- Future MJX/JAX work plugs into an explicit runtime/backend contract instead of overloading the meaning of `device`.
