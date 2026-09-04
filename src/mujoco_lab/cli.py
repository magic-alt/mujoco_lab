from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from mujoco_lab.config import load_config
from mujoco_lab.envs import make_env

app = typer.Typer(no_args_is_help=True, help="MuJoCo humanoid and bimanual learning lab")


def _parse_csv_ints(value: str, label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"{label} must be a comma-separated integer list") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise typer.BadParameter(f"{label} must contain positive integers")
    return parsed


def _parse_backends(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not parsed or any(item not in {"dummy", "subproc"} for item in parsed):
        raise typer.BadParameter("backends must contain only: dummy, subproc")
    return parsed


@app.command()
def doctor() -> None:
    """Print the local runtime, dependency and accelerator status."""
    packages = ["mujoco", "gymnasium", "numpy", "stable-baselines3", "torch", "robosuite"]
    typer.echo("mujoco_lab dependency report")
    for package in packages:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "not installed (may be optional)"
        typer.echo(f"- {package}: {version}")

    try:
        from mujoco_lab.runtime import (
            WORKLOAD_GENERIC,
            WORKLOAD_SB3_PPO_MLP_NATIVE,
            resolve_torch_device,
        )

        hardware_device, accelerator = resolve_torch_device(
            "auto",
            workload=WORKLOAD_GENERIC,
        )
        _, native_training = resolve_torch_device(
            "auto",
            workload=WORKLOAD_SB3_PPO_MLP_NATIVE,
        )
    except RuntimeError as exc:
        typer.echo(f"accelerator: unavailable ({exc})")
        return

    typer.echo("mujoco_lab accelerator report")
    typer.echo(f"- hardware-preferred torch device: {hardware_device}")
    typer.echo(f"- CUDA available: {accelerator['cuda_available']}")
    typer.echo(f"- PyTorch CUDA runtime: {accelerator['torch_cuda_version']}")
    typer.echo(f"- CUDA device count: {accelerator['cuda_device_count']}")
    for device in accelerator["cuda_devices"]:
        gib = float(device["total_memory_bytes"]) / (1024**3)
        capability = ".".join(str(value) for value in device["compute_capability"])
        typer.echo(
            f"- cuda:{device['index']}: {device['name']} "
            f"({gib:.1f} GiB, compute capability {capability})"
        )
    typer.echo(f"- MPS available: {accelerator['mps_available']}")
    typer.echo(f"- native SB3 PPO/MlpPolicy auto device: {native_training['resolved']}")
    typer.echo(f"- workload policy: {native_training['selection_reason']}")


@app.command("inspect-env")
def inspect_env(config: Path) -> None:
    """Reset an environment, take one random action and print its contract."""
    cfg = load_config(config)
    env = make_env(cfg)
    try:
        obs, info = env.reset(seed=cfg.algorithm.seed)
        typer.echo(f"environment: {cfg.task.family}/{cfg.task.env_id}")
        typer.echo(f"observation_space: {env.observation_space}")
        typer.echo(f"action_space: {env.action_space}")
        typer.echo(f"reset observation shape: {np.asarray(obs).shape}")
        typer.echo(f"reset info keys: {sorted(info.keys())}")
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        typer.echo(
            f"sample step: reward={float(reward):.4f}, "
            f"terminated={terminated}, truncated={truncated}"
        )
    finally:
        env.close()


@app.command("inspect-robot")
def inspect_robot(
    robot: Annotated[str, typer.Argument(help="Robot model to inspect; currently: g1")] = "g1",
    cache_root: Annotated[Path | None, typer.Option("--cache-root")] = None,
    offline: Annotated[bool, typer.Option("--offline")] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Resolve, compile and validate a pinned external robot model."""
    from mujoco_lab.robots.g1 import G1_SPEC, inspect_g1_model, inspection_json_path
    from mujoco_lab.robots.resolver import AssetResolutionError, resolve_robot_model

    if robot.lower() != "g1":
        raise typer.BadParameter("supported robot models: g1", param_hint="robot")

    try:
        resolved = resolve_robot_model(
            G1_SPEC,
            cache_root=cache_root,
            offline=offline,
            force=force,
        )
        report = inspect_g1_model(resolved)
    except (AssetResolutionError, ValueError) as exc:
        typer.echo(f"robot inspection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    cache_report = inspection_json_path(resolved)
    cache_report.write_text(serialized, encoding="utf-8")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    typer.echo(serialized, nl=False)
    typer.echo(f"inspection report: {cache_report}", err=True)


@app.command("benchmark-vec-env")
def benchmark_vec_env_command(
    config: Path,
    env_counts: Annotated[
        str,
        typer.Option("--env-counts", help="Comma-separated environment counts"),
    ] = "1,4,8",
    backends: Annotated[
        str,
        typer.Option("--backends", help="Comma-separated backends: dummy,subproc"),
    ] = "dummy,subproc",
    transitions: Annotated[int, typer.Option(min=1)] = 5_000,
    warmup_steps: Annotated[int, typer.Option("--warmup-steps", min=0)] = 20,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Benchmark DummyVecEnv and SubprocVecEnv simulator throughput."""
    from mujoco_lab.benchmarks.vec_env import benchmark_vec_envs

    cfg = load_config(config)
    counts = _parse_csv_ints(env_counts, "env-counts")
    requested_backends = _parse_backends(backends)
    report = benchmark_vec_envs(
        cfg,
        env_counts=counts,
        backends=requested_backends,
        transitions=transitions,
        warmup_steps=warmup_steps,
    )

    destination = output or Path("runs/benchmarks") / f"{cfg.name}-vecenv.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    typer.echo("backend  n_envs  startup_s  transitions/s")
    for result in report["results"]:
        typer.echo(
            f"{result['backend']:<8} {result['n_envs']:>6}  "
            f"{result['startup_seconds']:>9.3f}  "
            f"{result['transitions_per_second']:>13.1f}"
        )
    for skipped in report["skipped"]:
        typer.echo(f"skipped {skipped['backend']} n_envs={skipped['n_envs']}: {skipped['reason']}")
    typer.echo(f"benchmark report: {destination}")


@app.command()
def train(config: Path) -> None:
    """Train the configured PPO baseline."""
    from mujoco_lab.training.sb3 import train_ppo

    cfg = load_config(config)
    model_path = train_ppo(cfg)
    typer.echo(f"saved model: {model_path}")


@app.command()
def evaluate(
    config: Path,
    checkpoint: Path,
    episodes: int = typer.Option(5, min=1),
    render: bool = typer.Option(True, "--render/--no-render"),
) -> None:
    """Evaluate a PPO checkpoint with deterministic actions."""
    from mujoco_lab.evaluation import evaluate_ppo

    cfg = load_config(config)
    returns = evaluate_ppo(cfg, checkpoint, episodes=episodes, render=render)
    typer.echo(f"episode returns: {returns}")
    typer.echo(f"mean return: {float(np.mean(returns)):.3f}")
    typer.echo(f"std return: {float(np.std(returns)):.3f}")
