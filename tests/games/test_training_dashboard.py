from pathlib import Path
from typing import Any

import pytest

from flybrain.games.chess_training import ChessTrainingConfig, train_chess
from flybrain.games.runner_learning import RunnerTrainingConfig, train_runner
from flybrain.games.training_dashboard import Publish, run_training_dashboard


def test_visual_runner_training_plays_published_checkpoint(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    output = tmp_path / "runner"
    seen: list[Path] = []

    def train(publish: Publish) -> object:
        def on_checkpoint(checkpoint: Path, manifest: Any) -> None:
            seen.append(checkpoint)
            publish(checkpoint, manifest, None)

        return train_runner(
            RunnerTrainingConfig(total_steps=256, checkpoint_every=128, seed=9),
            output,
            on_checkpoint=on_checkpoint,
        )

    result = run_training_dashboard("runner", train, seed=9, max_frames=12, demo_frames=3)

    assert result.completed is True
    assert result.checkpoints_seen == 2
    assert result.checkpoints_demonstrated == 2
    assert result.played_frames > 0
    assert seen[0].joinpath("manifest.json").is_file()
    assert seen[1].joinpath("manifest.json").is_file()


def test_visual_chess_training_shows_each_epoch(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    def train(publish: Publish) -> object:
        return train_chess(
            ChessTrainingConfig(
                epochs=2,
                self_play_games=1,
                max_game_plies=4,
                self_play_simulations=1,
                seed=9,
            ),
            tmp_path / "chess",
            on_checkpoint=publish,
        )

    result = run_training_dashboard("chess", train, seed=9, max_frames=12, demo_frames=3)

    assert result.completed is True
    assert result.checkpoints_seen == 2
    assert result.checkpoints_demonstrated == 2
    assert result.played_frames >= 12


def test_dashboard_reports_training_failure_without_waiting_for_checkpoint(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    def fail(publish: Publish) -> object:
        raise ValueError("bad training data")

    with pytest.raises(RuntimeError, match="game training failed") as error:
        run_training_dashboard("runner", fail, max_frames=3, demo_frames=3)

    assert isinstance(error.value.__cause__, ValueError)
