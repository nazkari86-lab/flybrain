from pathlib import Path

from typer.testing import CliRunner

from flybrain.cli import app
from flybrain.games.chess_training import (
    ChessTrainingConfig,
    make_training_state,
    save_chess_checkpoint,
)
from flybrain.games.chess_viewer import run_chess_viewer_smoke

runner = CliRunner()


def _checkpoint(tmp_path: Path) -> Path:
    return save_chess_checkpoint(
        tmp_path / "checkpoint",
        make_training_state(ChessTrainingConfig(seed=7)),
    )


def test_chess_viewer_smoke_plays_only_legal_moves(tmp_path: Path) -> None:
    result = run_chess_viewer_smoke(_checkpoint(tmp_path), plies=8, seed=7)

    assert result.plies == 8
    assert result.illegal_moves == 0
    assert result.frame_shape == (720, 960, 3)


def test_chess_play_dry_run(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    result = runner.invoke(
        app,
        [
            "games",
            "play",
            "--game",
            "chess",
            "--checkpoint",
            str(checkpoint),
            "--dry-run",
            "--steps",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"illegal_moves":0' in result.output.replace(" ", "")
