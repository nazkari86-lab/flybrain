import chess
import torch

from flybrain.games.chess_model import ChessNetConfig, policy_value
from flybrain.games.chess_training import (
    ChessTrainingConfig,
    TrainingExample,
    make_training_state,
    train_batches,
)


def _teacher_fixture() -> tuple[TrainingExample, ...]:
    positions = (
        (chess.STARTING_FEN, "e2e4", 0.3),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "g1f3", 0.2),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "b8c6", -0.2),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f1b5", 0.25),
    )
    return tuple(
        TrainingExample(fen=fen, policy={move: 1.0}, value=value, source="teacher")
        for fen, move, value in positions
    )


def _mean_target_probability(state, examples: tuple[TrainingExample, ...]) -> float:
    probabilities = []
    for example in examples:
        board = chess.Board(example.fen)
        priors, _ = policy_value(state.model, board, torch.device("cpu"))
        target = chess.Move.from_uci(next(iter(example.policy)))
        probabilities.append(priors[target])
    return sum(probabilities) / len(probabilities)


def test_trained_chess_policy_improves_on_fixed_examples() -> None:
    examples = _teacher_fixture()
    state = make_training_state(
        ChessTrainingConfig(
            seed=7,
            epochs=30,
            network=ChessNetConfig(channels=16, residual_blocks=1),
        ),
        examples=examples,
    )
    initial = _mean_target_probability(state, examples)

    train_batches(state, epochs=30)
    trained = _mean_target_probability(state, examples)

    assert trained > initial
    assert trained > 0.5
