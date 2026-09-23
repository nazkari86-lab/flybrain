import hashlib
from pathlib import Path

from flybrain.games.chess_evaluation import evaluate_chess, random_opponent
from flybrain.games.chess_training import (
    ChessTrainingConfig,
    make_training_state,
    save_chess_checkpoint,
)


def _hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def test_evaluation_is_immutable_and_legal(tmp_path: Path) -> None:
    checkpoint = save_chess_checkpoint(
        tmp_path / "checkpoint",
        make_training_state(ChessTrainingConfig(seed=7)),
    )
    before = _hash_tree(checkpoint)

    result = evaluate_chess(
        checkpoint,
        opponents=(random_opponent(),),
        seeds=(1, 2),
        search_simulations=2,
        max_plies=12,
    )

    assert result.illegal_moves == 0
    assert result.games == 4
    assert _hash_tree(checkpoint) == before
    assert result.training_mutated is False
