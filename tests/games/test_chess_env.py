import chess
import numpy as np
import pytest

from flybrain.games.chess_env import ChessAdapter, encode_board, promotion_index


def test_initial_board_encoding_is_stable() -> None:
    encoded = encode_board(chess.Board())

    assert encoded.shape == (21, 8, 8)
    assert encoded.dtype == np.float32
    assert int(encoded[:12].sum()) == 32
    assert np.all(encoded[12] == 1.0)


def test_adapter_rejects_illegal_move() -> None:
    adapter = ChessAdapter()
    adapter.reset(seed=7, mode="play")
    illegal = chess.Move.from_uci("e2e5")

    with pytest.raises(ValueError, match="illegal move"):
        adapter.step(illegal)


def test_terminal_reward_is_from_the_movers_perspective() -> None:
    adapter = ChessAdapter()
    adapter.reset(seed=7, mode="play")
    for uci in ("f2f3", "e7e5", "g2g4"):
        adapter.step(chess.Move.from_uci(uci))

    result = adapter.step(chess.Move.from_uci("d8h4"))

    assert result.terminated is True
    assert result.reward == 1.0
    assert result.info["outcome"] == "0-1"


def test_special_state_planes_and_promotion_index() -> None:
    board = chess.Board("8/P7/8/8/8/8/7p/4K2k w - - 99 1")
    encoded = encode_board(board)

    assert np.all(encoded[13:17] == 0.0)
    assert np.allclose(encoded[18], 0.99)
    assert promotion_index(chess.Move.from_uci("a7a8q")) == 4
    assert set(board.legal_moves)
