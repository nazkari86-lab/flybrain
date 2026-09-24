from typer.testing import CliRunner

from flybrain.cli import app
from flybrain.games import cli as games_cli
from flybrain.games.training_dashboard import run_training_dashboard

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


def test_games_train_visual_opens_dashboard_and_saves_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    def bounded_dashboard(game, train, *, seed):
        return run_training_dashboard(game, train, seed=seed, max_frames=20, demo_frames=3)

    monkeypatch.setattr(games_cli, "run_training_dashboard", bounded_dashboard)
    output = tmp_path / "visual-runner"
    result = runner.invoke(
        app,
        [
            "games",
            "train",
            "--game",
            "runner",
            "--visual",
            "--steps",
            "128",
            "--checkpoint-every",
            "64",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.joinpath("latest", "manifest.json").is_file()
    assert '"completed_steps":128' in result.output.replace(" ", "")
