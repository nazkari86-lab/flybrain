"""Focused Typer commands for training and using game-learning agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from flybrain.games.artifacts import write_json_atomic
from flybrain.games.chess_engine import resolve_stockfish
from flybrain.games.chess_evaluation import (
    evaluate_chess,
    material_opponent,
    random_opponent,
    stockfish_opponent,
)
from flybrain.games.chess_training import ChessTrainingConfig, train_chess
from flybrain.games.chess_viewer import (
    ChessViewerConfig,
    run_chess_viewer,
    run_chess_viewer_smoke,
)
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
from flybrain.games.training_dashboard import run_training_dashboard
from flybrain.games.universal import (
    SnapshotCallback,
    UniversalGameConfig,
    UniversalGameRun,
    env_id_factory,
    evaluate_universal_game,
    load_factory,
    run_universal_dashboard,
    train_universal_game,
)

GameName = Literal["runner", "chess"]

games_app = typer.Typer(help="Train, evaluate, watch, and play game-learning agents.")


@games_app.command("connect")
def connect_game(
    output: Annotated[Path, typer.Option("--output")],
    env_id: Annotated[
        str | None,
        typer.Option("--env-id", help="Registered Gymnasium environment id."),
    ] = None,
    factory: Annotated[
        str | None,
        typer.Option("--factory", help="Local environment factory as module:callable."),
    ] = None,
    steps: Annotated[int, typer.Option("--steps", min=1)] = 100_000,
    checkpoint_every: Annotated[int, typer.Option("--checkpoint-every", min=1)] = 10_000,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 7,
    resume: Annotated[Path | None, typer.Option("--resume")] = None,
    visual: Annotated[
        bool,
        typer.Option("--visual", help="Show the connected game and learning telemetry."),
    ] = False,
    max_frames: Annotated[int | None, typer.Option("--max-frames", min=1)] = None,
    demo_frames: Annotated[int, typer.Option("--demo-frames", min=1)] = 90,
) -> None:
    """Connect, train, and watch any discrete-action Gymnasium game."""

    if (env_id is None) == (factory is None):
        raise typer.BadParameter("provide exactly one of --env-id or --factory")
    try:
        resolved = env_id_factory(env_id) if env_id is not None else load_factory(factory or "")
        config = UniversalGameConfig(
            total_steps=steps,
            checkpoint_every=min(checkpoint_every, steps),
            seed=seed,
        )

        def train(publish: SnapshotCallback) -> UniversalGameRun:
            return train_universal_game(
                resolved.factory,
                env_name=resolved.name,
                config=config,
                output=output,
                resume=resume,
                on_snapshot=publish,
            )

        if visual:

            def visual_train(publish: SnapshotCallback) -> UniversalGameRun:
                return train_universal_game(
                    resolved.factory,
                    env_name=resolved.name,
                    config=config,
                    output=output,
                    resume=resume,
                    on_snapshot=publish,
                    close_env=False,
                )

            result = run_universal_dashboard(
                resolved.visual_factory,
                visual_train,
                env_name=resolved.name,
                seed=seed,
                max_frames=max_frames,
                demo_frames=demo_frames,
            )
        else:
            result = train(lambda _path, _manifest: None)
        typer.echo(result.model_dump_json())
    except (FileExistsError, FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error


@games_app.command("connect-evaluate")
def evaluate_connected_game(
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    env_id: Annotated[
        str | None,
        typer.Option("--env-id", help="Registered Gymnasium environment id."),
    ] = None,
    factory: Annotated[
        str | None,
        typer.Option("--factory", help="Local environment factory as module:callable."),
    ] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Evaluate a connected checkpoint on its frozen holdout seed schedule."""

    if (env_id is None) == (factory is None):
        raise typer.BadParameter("provide exactly one of --env-id or --factory")
    try:
        resolved = env_id_factory(env_id) if env_id is not None else load_factory(factory or "")
        evaluation = evaluate_universal_game(checkpoint, resolved.factory)
        if output is not None:
            write_json_atomic(output, json.loads(evaluation.model_dump_json()))
        typer.echo(evaluation.model_dump_json())
    except (FileExistsError, FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error


@games_app.command("train")
def train_game(
    output: Annotated[Path, typer.Option("--output")],
    game: Annotated[GameName, typer.Option("--game")] = "runner",
    steps: Annotated[int, typer.Option("--steps", min=1)] = 100_000,
    checkpoint_every: Annotated[int, typer.Option("--checkpoint-every", min=1)] = 10_000,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 7,
    resume: Annotated[Path | None, typer.Option("--resume")] = None,
    teacher_positions: Annotated[int, typer.Option("--teacher-positions", min=0)] = 0,
    teacher_nodes: Annotated[int, typer.Option("--teacher-nodes", min=1)] = 2_000,
    self_play_games: Annotated[int, typer.Option("--self-play-games", min=0)] = 0,
    self_play_simulations: Annotated[int, typer.Option("--self-play-simulations", min=1)] = 16,
    max_game_plies: Annotated[int, typer.Option("--max-game-plies", min=2)] = 160,
    epochs: Annotated[int, typer.Option("--epochs", min=0)] = 3,
    stockfish: Annotated[Path | None, typer.Option("--stockfish")] = None,
    visual: Annotated[
        bool, typer.Option("--visual", help="Show learning and checkpoint play in a live window.")
    ] = False,
) -> None:
    """Train or resume one persistent game policy."""

    try:
        if game == "runner":
            runner_config = RunnerTrainingConfig(
                total_steps=steps,
                checkpoint_every=min(checkpoint_every, steps),
                seed=seed,
            )
            if visual:
                dashboard = run_training_dashboard(
                    "runner",
                    lambda publish: train_runner(
                        runner_config,
                        output,
                        resume=resume,
                        on_checkpoint=lambda path, manifest: publish(path, manifest, None),
                    ),
                    seed=seed,
                )
                runner_result = dashboard.training_result
            else:
                runner_result = train_runner(runner_config, output, resume=resume)
            typer.echo(runner_result.model_dump_json())
            return
        else:
            teacher = resolve_stockfish(stockfish)
            if teacher_positions and teacher is None:
                raise ValueError("teacher positions require a verified Stockfish binary")
            chess_config = ChessTrainingConfig(
                seed=seed,
                teacher_positions=teacher_positions,
                teacher_nodes=teacher_nodes,
                self_play_games=self_play_games,
                self_play_simulations=self_play_simulations,
                max_game_plies=max_game_plies,
                epochs=epochs,
            )
            if visual:
                dashboard = run_training_dashboard(
                    "chess",
                    lambda publish: train_chess(
                        chess_config,
                        output,
                        teacher=teacher,
                        resume=resume,
                        on_checkpoint=publish,
                    ),
                    seed=seed,
                )
                chess_result = dashboard.training_result
            else:
                chess_result = train_chess(
                    chess_config,
                    output,
                    teacher=teacher,
                    resume=resume,
                )
            typer.echo(chess_result.model_dump_json())
            return
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error


@games_app.command("evaluate")
def evaluate_game(
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    game: Annotated[GameName, typer.Option("--game")] = "runner",
    output: Annotated[Path | None, typer.Option("--output")] = None,
    games_per_opponent: Annotated[int, typer.Option("--games-per-opponent", min=1)] = 2,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 7,
    search_simulations: Annotated[int, typer.Option("--search-simulations", min=1)] = 16,
    max_game_plies: Annotated[int, typer.Option("--max-game-plies", min=2)] = 160,
    stockfish: Annotated[Path | None, typer.Option("--stockfish")] = None,
    stockfish_nodes: Annotated[int, typer.Option("--stockfish-nodes", min=1)] = 500,
) -> None:
    """Evaluate a frozen checkpoint on its declared holdout seeds."""

    try:
        resolved = checkpoint.resolve()
        if game == "runner":
            manifest = RunnerCheckpointManifest.model_validate_json(
                resolved.joinpath("manifest.json").read_text()
            )
            runner_result = evaluate_runner(resolved, seeds=manifest.seeds.holdout)
            if output is not None:
                write_json_atomic(output, json.loads(runner_result.model_dump_json()))
            typer.echo(runner_result.model_dump_json())
            return
        else:
            opponents = [random_opponent(), material_opponent()]
            engine = resolve_stockfish(stockfish)
            if engine is not None:
                opponents.append(stockfish_opponent(engine, nodes=stockfish_nodes))
            chess_result = evaluate_chess(
                resolved,
                opponents=tuple(opponents),
                seeds=tuple(seed + index for index in range(games_per_opponent)),
                search_simulations=search_simulations,
                max_plies=max_game_plies,
            )
            if output is not None:
                write_json_atomic(output, json.loads(chess_result.model_dump_json()))
            typer.echo(chess_result.model_dump_json())
            return
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error


@games_app.command("watch")
def watch_game(
    game: Annotated[GameName, typer.Option("--game")] = "runner",
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 7,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    steps: Annotated[int, typer.Option("--steps", min=1)] = 120,
    simulations: Annotated[int, typer.Option("--simulations", min=1)] = 16,
) -> None:
    """Watch an autonomous runner policy in a persistent local window."""

    if game == "runner":
        if dry_run:
            result = run_runner_viewer_smoke(steps=steps, seed=seed, checkpoint=checkpoint)
            typer.echo(result.model_dump_json())
            return
        raise typer.Exit(run_runner_viewer(ViewerConfig(checkpoint=checkpoint, seed=seed)))
    if checkpoint is None:
        raise typer.BadParameter("chess watch requires --checkpoint")
    if dry_run:
        chess_result = run_chess_viewer_smoke(
            checkpoint,
            plies=steps,
            seed=seed,
            simulations=simulations,
        )
        typer.echo(chess_result.model_dump_json())
        return
    raise typer.Exit(
        run_chess_viewer(
            ChessViewerConfig(
                checkpoint=checkpoint,
                seed=seed,
                simulations=simulations,
            )
        )
    )


@games_app.command("play")
def play_game(
    game: Annotated[GameName, typer.Option("--game")] = "runner",
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 7,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    steps: Annotated[int, typer.Option("--steps", min=1)] = 120,
    simulations: Annotated[int, typer.Option("--simulations", min=1)] = 16,
) -> None:
    """Play the runner manually, with optional instant agent toggle."""

    if game == "runner":
        if dry_run:
            result = run_runner_viewer_smoke(steps=steps, seed=seed, checkpoint=checkpoint)
            typer.echo(result.model_dump_json())
            return
        raise typer.Exit(
            run_runner_viewer(ViewerConfig(checkpoint=checkpoint, seed=seed, human_control=True))
        )
    if checkpoint is None:
        raise typer.BadParameter("chess play requires --checkpoint")
    if dry_run:
        chess_result = run_chess_viewer_smoke(
            checkpoint,
            plies=steps,
            seed=seed,
            simulations=simulations,
        )
        typer.echo(chess_result.model_dump_json())
        return
    raise typer.Exit(
        run_chess_viewer(
            ChessViewerConfig(
                checkpoint=checkpoint,
                seed=seed,
                simulations=simulations,
                human_play=True,
            )
        )
    )
