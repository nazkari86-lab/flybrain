import chess
import torch

from flybrain.games.chess_model import ChessPolicyValueNet
from flybrain.games.chess_search import SearchConfig, search


def test_search_always_returns_legal_move() -> None:
    board = chess.Board()
    result = search(
        board,
        ChessPolicyValueNet(seed=7),
        SearchConfig(simulations=8, seed=7),
        torch.device("cpu"),
    )

    assert result.move in board.legal_moves
    assert sum(result.visit_counts.values()) == 8


def test_search_takes_forced_mate_in_one() -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    result = search(
        board,
        ChessPolicyValueNet(seed=7),
        SearchConfig(simulations=32, seed=7),
        torch.device("cpu"),
    )

    board.push(result.move)
    assert board.is_checkmate()
