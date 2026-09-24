from pathlib import Path

from flybrain.games.chess_training import ChessTrainingConfig, train_chess


def test_chess_visual_training_publishes_each_epoch(tmp_path: Path) -> None:
    published: list[tuple[int, Path]] = []
    train_chess(
        ChessTrainingConfig(
            epochs=2,
            teacher_positions=0,
            self_play_games=1,
            max_game_plies=4,
            self_play_simulations=1,
            seed=9,
        ),
        tmp_path / "chess",
        on_checkpoint=lambda checkpoint, manifest, loss: published.append(
            (manifest.completed_epochs, checkpoint)
        ),
    )

    assert [epoch for epoch, _ in published] == [1, 2]
    assert all(path.joinpath("model.pt").is_file() for _, path in published)
