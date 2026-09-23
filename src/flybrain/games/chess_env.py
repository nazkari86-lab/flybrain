"""Legal chess adapter and compact provenance-stable board encoding."""

from __future__ import annotations

import chess
import numpy as np
from numpy.typing import NDArray

from flybrain.games.contracts import GameMode, StepResult

ChessObservation = NDArray[np.float32]


def _coordinates(square: chess.Square) -> tuple[int, int]:
    return 7 - chess.square_rank(square), chess.square_file(square)


def encode_board(board: chess.Board) -> ChessObservation:
    """Encode one position into 21 finite planes from White's orientation."""

    planes = np.zeros((21, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane = color_offset + piece.piece_type - 1
        row, column = _coordinates(square)
        planes[plane, row, column] = 1.0

    planes[12].fill(float(board.turn == chess.WHITE))
    castling = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    for index, allowed in enumerate(castling, start=13):
        planes[index].fill(float(allowed))
    if board.ep_square is not None:
        row, column = _coordinates(board.ep_square)
        planes[17, row, column] = 1.0
    planes[18].fill(min(board.halfmove_clock, 100) / 100.0)
    planes[19].fill(float(board.is_repetition(2)))
    if board.move_stack:
        row, column = _coordinates(board.peek().to_square)
        planes[20, row, column] = 1.0
    return planes


def promotion_index(move: chess.Move) -> int:
    """Map legal promotion choices to the model's five-way promotion head."""

    mapping = {
        None: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
    }
    try:
        return mapping[move.promotion]
    except KeyError as error:
        raise ValueError(f"unsupported promotion piece: {move.promotion}") from error


class ChessAdapter:
    """Stateful standard-chess game boundary that exposes only legal moves."""

    def __init__(self, starting_fen: str = chess.STARTING_FEN) -> None:
        self.starting_fen = starting_fen
        self.board = chess.Board(starting_fen)
        self._seed = 0
        self._mode: GameMode = "play"

    def reset(self, *, seed: int, mode: GameMode) -> ChessObservation:
        self.board = chess.Board(self.starting_fen)
        self._seed = seed
        self._mode = mode
        return encode_board(self.board)

    def legal_actions(self) -> tuple[chess.Move, ...]:
        return tuple(self.board.legal_moves)

    def step(self, action: chess.Move) -> StepResult[ChessObservation]:
        if action not in self.board.legal_moves:
            raise ValueError(f"illegal move: {action.uci()}")
        mover = self.board.turn
        san = self.board.san(action)
        self.board.push(action)
        outcome = self.board.outcome(claim_draw=True)
        reward = 0.0
        if outcome is not None and outcome.winner is not None:
            reward = 1.0 if outcome.winner == mover else -1.0
        return StepResult(
            observation=encode_board(self.board),
            reward=reward,
            terminated=outcome is not None,
            truncated=False,
            info={
                "seed": self._seed,
                "mode": self._mode,
                "fen": self.board.fen(),
                "uci": action.uci(),
                "san": san,
                "ply": self.board.ply(),
                "outcome": outcome.result() if outcome is not None else "",
                "termination": (
                    outcome.termination.name.lower() if outcome is not None else ""
                ),
            },
        )
