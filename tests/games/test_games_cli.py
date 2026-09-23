from typer.testing import CliRunner

from flybrain.cli import app

runner = CliRunner()


def test_games_watch_dry_run() -> None:
    result = runner.invoke(
        app,
        ["games", "watch", "--game", "runner", "--dry-run", "--steps", "12"],
    )

    assert result.exit_code == 0, result.output
    assert '"frames":12' in result.output.replace(" ", "")


def test_games_reject_unknown_game() -> None:
    result = runner.invoke(
        app,
        ["games", "train", "--game", "unknown", "--output", "unused"],
    )

    assert result.exit_code != 0
