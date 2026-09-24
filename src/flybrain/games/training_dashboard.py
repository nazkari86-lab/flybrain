"""Live Pygame dashboard for checkpoint-by-checkpoint game learning."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import chess
import numpy as np
import torch

from flybrain.games.chess_search import SearchConfig, search
from flybrain.games.chess_training import ChessCheckpointManifest, load_chess_policy
from flybrain.games.chess_viewer import render_chess_frame
from flybrain.games.runner_env import RunnerEnv
from flybrain.games.runner_learning import RunnerCheckpointManifest, load_runner_model

GameName = Literal["runner", "chess"]
Manifest = RunnerCheckpointManifest | ChessCheckpointManifest
Publish = Callable[[Path, Manifest, float | None], None]


@dataclass(frozen=True)
class DashboardResult:
    completed: bool
    checkpoints_seen: int
    checkpoints_demonstrated: int
    played_frames: int
    training_result: Any


def _chart(surface: Any, points: list[float], *, area: tuple[int, int, int, int]) -> None:
    import pygame

    x, y, width, height = area
    pygame.draw.rect(surface, (30, 40, 61), pygame.Rect(*area), border_radius=10)
    if not points:
        return
    low = min(0.0, min(points))
    high = max(1.0, max(points)) if max(points) <= 1.0 else max(points) * 1.05
    span = max(1e-6, high - low)
    coordinates = [
        (
            x + 16 + int((width - 32) * index / max(1, len(points) - 1)),
            y + height - 16 - int((height - 32) * (value - low) / span),
        )
        for index, value in enumerate(points)
    ]
    if len(coordinates) > 1:
        pygame.draw.lines(surface, (88, 227, 183), False, coordinates, 3)
    for point in coordinates:
        pygame.draw.circle(surface, (248, 211, 112), point, 4)


def run_training_dashboard(
    game: GameName,
    train: Callable[[Publish], Any],
    *,
    seed: int = 7,
    max_frames: int | None = None,
    demo_frames: int = 60,
) -> DashboardResult:
    """Train in a worker while the main thread shows published policies playing."""

    import pygame

    if demo_frames < 1:
        raise ValueError("demo_frames must be positive")
    pygame.init()
    updates: queue.SimpleQueue[tuple[Path, Manifest, float | None]] = queue.SimpleQueue()
    result: list[Any] = []
    errors: list[BaseException] = []

    def publish(path: Path, manifest: Manifest, loss: float | None = None) -> None:
        updates.put((path, manifest, loss))

    def worker() -> None:
        try:
            result.append(train(publish))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=worker, name=f"flybrain-{game}-training")
    try:
        screen = pygame.display.set_mode((1280, 760))
        pygame.display.set_caption(f"FlyBrain — {game} learning live")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 30)
        small_font = pygame.font.Font(None, 23)
        runner_env = RunnerEnv() if game == "runner" else None
        runner_observation = None
        runner_info: dict[str, Any] = {}
        chess_board = chess.Board()
        model: Any = None
        points: list[float] = []
        current: Manifest | None = None
        seen = 0
        demonstrated = 0
        played_frames = 0
        frames_since_switch = 0
        pending: deque[tuple[Path, Manifest, float | None]] = deque()
        last_move = 0.0
        running = True
        paused = False
        thread.start()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)
                ):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    paused = not paused

            while not updates.empty():
                update = updates.get_nowait()
                pending.append(update)
                _, manifest, loss = update
                seen += 1
                if game == "runner":
                    assert isinstance(manifest, RunnerCheckpointManifest)
                    points.append(manifest.validation_mean_distance)
                elif loss is not None:
                    points.append(loss)

            if pending and (current is None or frames_since_switch >= demo_frames):
                path, current, _ = pending.popleft()
                demonstrated += 1
                frames_since_switch = 0
                if game == "runner":
                    assert runner_env is not None
                    model = load_runner_model(path, runner_env)
                    runner_observation, runner_info = runner_env.reset(
                        seed=seed, options={"mode": "play"}
                    )
                else:
                    model = load_chess_policy(path)
                    chess_board.reset()
                    last_move = 0.0

            screen.fill((18, 24, 37))
            status = "training" if thread.is_alive() else "training finished"
            if errors:
                status = f"training failed: {str(errors[0])[:70]}"
            title = f"{game.upper()}  |  {status}  |  snapshots: {seen}"
            screen.blit(font.render(title, True, (248, 250, 255)), (20, 18))
            screen.blit(
                small_font.render(
                    "P pause playback  |  Q close window (training continues)",
                    True,
                    (178, 195, 216),
                ),
                (20, 55),
            )

            if model is not None and game == "runner":
                assert runner_env is not None and runner_observation is not None
                if not paused:
                    action, _ = model.predict(runner_observation, deterministic=True)
                    runner_observation, _, ended, timed_out, runner_info = runner_env.step(
                        int(np.asarray(action).item())
                    )
                    if ended or timed_out:
                        runner_observation, runner_info = runner_env.reset(
                            seed=seed, options={"mode": "play"}
                        )
                frame = runner_env.render_rgb(width=920, height=620)
                played_frames += 1
                frames_since_switch += 1
                screen.blit(
                    pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), (20, 100)
                )
            elif model is not None:
                if not paused and time.monotonic() - last_move >= 0.45:
                    if chess_board.is_game_over(claim_draw=True) or chess_board.ply() >= 80:
                        chess_board.reset()
                    move = search(
                        chess_board,
                        model,
                        SearchConfig(simulations=2, seed=seed + chess_board.ply()),
                        torch.device("cpu"),
                    ).move
                    if move not in chess_board.legal_moves:
                        raise RuntimeError("dashboard agent selected an illegal move")
                    chess_board.push(move)
                    last_move = time.monotonic()
                frame = render_chess_frame(chess_board, width=920, height=620)
                played_frames += 1
                frames_since_switch += 1
                screen.blit(
                    pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), (20, 100)
                )
            else:
                waiting = "Waiting for first complete checkpoint..."
                screen.blit(font.render(waiting, True, (209, 220, 236)), (50, 150))

            if current is not None:
                if isinstance(current, RunnerCheckpointManifest):
                    lines = (
                        f"steps: {current.completed_steps:,} / {current.config.total_steps:,}",
                        f"validation distance: {current.validation_mean_distance:.3f}",
                        f"play distance: {float(runner_info.get('normalized_distance', 0)):.3f}",
                        "Validation: separate seeds",
                    )
                    chart_title = "Validation distance"
                else:
                    lines = (
                        f"epochs: {current.completed_epochs}",
                        f"examples: {current.examples}",
                        f"self-play games: {current.self_play_games}",
                        f"board ply: {chess_board.ply()}",
                    )
                    chart_title = "Training loss (lower is better)"
                for index, line in enumerate(lines):
                    screen.blit(
                        small_font.render(line, True, (225, 231, 242)), (955, 125 + 34 * index)
                    )
                screen.blit(small_font.render(chart_title, True, (185, 223, 234)), (955, 307))
                _chart(screen, points, area=(955, 340, 300, 240))
                if points:
                    screen.blit(
                        small_font.render(f"latest: {points[-1]:.3f}", True, (248, 211, 112)),
                        (955, 598),
                    )

            pygame.display.flip()
            clock.tick(30)
            if errors and not thread.is_alive():
                running = False
            if (
                max_frames is not None
                and not thread.is_alive()
                and not pending
                and frames_since_switch >= demo_frames
                and played_frames >= max_frames
            ):
                running = False
        thread.join()
        if errors:
            raise RuntimeError("game training failed") from errors[0]
        return DashboardResult(
            completed=bool(result),
            checkpoints_seen=seen,
            checkpoints_demonstrated=demonstrated,
            played_frames=played_frames,
            training_result=result[0] if result else None,
        )
    finally:
        pygame.quit()
