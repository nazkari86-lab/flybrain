"""Focused Typer commands for training and using game-learning agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from flybrain.games.artifacts import write_json_atomic
from flybrain.games.runner_learning import (
    RunnerCheckpointManifest,
    RunnerTrainingConfig,
    evaluate_runner,
    train_runner,
)
from flybrain.games.runner_viewer import (
    ViewerConfig,
    run_runner_viewer,
    run_runner_viewer_smoke,
)

GameName = Literal["runner"]

games_app = typer.Typer(help="Train, evaluate, watch, and play game-learning agents.")


@games_app.command("train")
def train_game(
    output: Annotated[Path, typer.Option("--output")],
    game: Annotated[GameName, typer.Option("--game")] = "runner",
    steps: Annotated[int, typer.Option("--steps", min=1)] = 100_000,
    checkpoint_every: Annotated[
        int, typer.Option("--checkpoint-every", min=1)
    ] = 10_000,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 7,
    resume: Annotated[Path | None, typer.Option("--resume")] = None,
) -> None:
    """Train or resume one persistent game policy."""

    if game != "runner":
        raise typer.BadParameter(f"unsupported game: {game}")
    try:
        result = train_runner(
            RunnerTrainingConfig(
                total_steps=steps,
                checkpoint_every=min(checkpoint_every, steps),
                seed=seed,
            ),
            output,
            resume=resume,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(result.model_dump_json())


@games_app.command("evaluate")
def evaluate_game(
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    game: Annotated[GameName, typer.Option("--game")] = "runner",
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Evaluate a frozen checkpoint on its declared holdout seeds."""

    if game != "runner":
        raise typer.BadParameter(f"unsupported game: {game}")
    try:
        resolved = checkpoint.resolve()
        manifest = RunnerCheckpointManifest.model_validate_json(
            resolved.joinpath("manifest.json").read_text()
        )
        result = evaluate_runner(resolved, seeds=manifest.seeds.holdout)
        if output is not None:
            write_json_atomic(output, json.loads(result.model_dump_json()))
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(result.model_dump_json())


@games_app.command("watch")
def watch_game(
    game: Annotated[GameName, typer.Option("--game")] = "runner",
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 7,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    steps: Annotated[int, typer.Option("--steps", min=1)] = 120,
) -> None:
    """Watch an autonomous runner policy in a persistent local window."""

    if game != "runner":
        raise typer.BadParameter(f"unsupported game: {game}")
    if dry_run:
        result = run_runner_viewer_smoke(steps=steps, seed=seed, checkpoint=checkpoint)
        typer.echo(result.model_dump_json())
        return
    raise typer.Exit(run_runner_viewer(ViewerConfig(checkpoint=checkpoint, seed=seed)))


@games_app.command("play")
def play_game(
    game: Annotated[GameName, typer.Option("--game")] = "runner",
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 7,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    steps: Annotated[int, typer.Option("--steps", min=1)] = 120,
) -> None:
    """Play the runner manually, with optional instant agent toggle."""

    if game != "runner":
        raise typer.BadParameter(f"unsupported game: {game}")
    if dry_run:
        result = run_runner_viewer_smoke(steps=steps, seed=seed, checkpoint=checkpoint)
        typer.echo(result.model_dump_json())
        return
    raise typer.Exit(
        run_runner_viewer(
            ViewerConfig(checkpoint=checkpoint, seed=seed, human_control=True)
        )
    )
