import io

import chess
import chess.pgn
import torch

from flybrain.games.chess_engine import resolve_stockfish
from flybrain.games.chess_model import ChessPolicyValueNet
from flybrain.games.chess_training import (
    ChessTrainingConfig,
    TrainingExample,
    batch_loss,
    generate_self_play_game,
    generate_teacher_examples,
    load_chess_checkpoint,
    make_training_state,
    save_chess_checkpoint,
    train_batches,
)


def _examples() -> tuple[TrainingExample, ...]:
    initial = chess.Board()
    after_e4 = initial.copy()
    after_e4.push_uci("e2e4")
    return (
        TrainingExample(
            fen=initial.fen(),
            policy={"e2e4": 1.0},
            value=0.4,
            source="teacher",
        ),
        TrainingExample(
            fen=after_e4.fen(),
            policy={"e7e5": 1.0},
            value=-0.4,
            source="teacher",
        ),
    )


def test_tiny_batch_update_reduces_loss() -> None:
    state = make_training_state(ChessTrainingConfig(seed=7), examples=_examples())
    before = float(batch_loss(state.model, state.examples).item())

    train_batches(state, epochs=20)
    after = float(batch_loss(state.model, state.examples).item())

    assert after < before


def test_checkpoint_restores_model_optimizer_counters_and_replay(tmp_path) -> None:
    state = make_training_state(ChessTrainingConfig(seed=11), examples=_examples())
    train_batches(state, epochs=2)
    state.self_play_games = 3

    checkpoint = save_chess_checkpoint(tmp_path / "checkpoint", state)
    restored = load_chess_checkpoint(checkpoint)

    assert restored.manifest is not None
    assert restored.completed_epochs == state.completed_epochs
    assert restored.self_play_games == state.self_play_games
    assert restored.manifest.replay_sha256 == state.manifest.replay_sha256
    assert all(
        torch.equal(restored.model.state_dict()[name], state.model.state_dict()[name])
        for name in state.model.state_dict()
    )
    assert restored.optimizer.state_dict()["param_groups"] == state.optimizer.state_dict()[
        "param_groups"
    ]


def test_real_teacher_and_self_play_emit_legal_examples() -> None:
    teacher = resolve_stockfish()
    assert teacher is not None
    teacher_examples = generate_teacher_examples(teacher, count=1, nodes=100, seed=7)
    self_play_examples, pgn = generate_self_play_game(
        ChessPolicyValueNet(seed=7),
        simulations=2,
        max_plies=4,
        seed=7,
    )

    assert len(teacher_examples) == 1
    assert teacher_examples[0].teacher_sha256 == teacher.executable_sha256
    assert len(self_play_examples) == 4
    assert chess.pgn.read_game(io.StringIO(pgn)) is not None
