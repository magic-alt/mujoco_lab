from __future__ import annotations

from importlib import metadata
from pathlib import Path

import numpy as np
import typer

from mujoco_lab.config import load_config
from mujoco_lab.envs import make_env

app = typer.Typer(no_args_is_help=True, help="MuJoCo humanoid and bimanual learning lab")


@app.command()
def doctor() -> None:
    """Print the local runtime and dependency status."""
    packages = ["mujoco", "gymnasium", "numpy", "stable-baselines3", "robosuite"]
    typer.echo("mujoco_lab dependency report")
    for package in packages:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "not installed (may be optional)"
        typer.echo(f"- {package}: {version}")


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
