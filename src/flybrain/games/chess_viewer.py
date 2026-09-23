"""Local chess board viewer for autoplay and human-vs-agent games."""

from __future__ import annotations

from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from flybrain.games.chess_search import SearchConfig, search
from flybrain.games.chess_training import load_chess_checkpoint

RgbFrame = NDArray[np.uint8]


class ChessViewerConfig(BaseModel, frozen=True):
    """Local chess-window controls and search budget."""

    model_config = ConfigDict(extra="forbid")

    checkpoint: Path
    seed: int = Field(default=7, ge=0)
    width: int = Field(default=960, ge=720, le=3_840)
    height: int = Field(default=720, ge=640, le=2_160)
    simulations: int = Field(default=16, ge=1, le=10_000)
    human_play: bool = False


class ChessViewerSmokeResult(BaseModel, frozen=True):
    """Headless proof of legal search and board rendering."""

    model_config = ConfigDict(extra="forbid")

    protocol: str = "chess-viewer-smoke-v1"
    plies: int = Field(ge=1)
    illegal_moves: int = Field(ge=0)
    frame_shape: tuple[int, int, int]
    final_fen: str
    pgn: str


_PIECE_LABELS = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}


def render_chess_frame(
    board: chess.Board,
    *,
    width: int = 960,
    height: int = 720,
    flipped: bool = False,
    selected: chess.Square | None = None,
) -> RgbFrame:
    """Render one board and compact status panel into an RGB array."""

    import pygame

    surface = pygame.Surface((width, height))
    surface.fill((20, 25, 39))
    board_size = min(height, width - 240)
    square_size = board_size // 8
    board_size = square_size * 8
    piece_font = pygame.font.Font(None, max(28, int(square_size * 0.68)))
    text_font = pygame.font.Font(None, 24)
    legal_destinations = {
        move.to_square for move in board.legal_moves if move.from_square == selected
    }

    for display_rank in range(8):
        for display_file in range(8):
            file_index = 7 - display_file if flipped else display_file
            rank_index = display_rank if flipped else 7 - display_rank
            square = chess.square(file_index, rank_index)
            color = (235, 218, 181) if (file_index + rank_index) % 2 else (111, 143, 114)
            rectangle = pygame.Rect(
                display_file * square_size,
                display_rank * square_size,
                square_size,
                square_size,
            )
            pygame.draw.rect(surface, color, rectangle)
            if square == selected:
                pygame.draw.rect(surface, (246, 196, 68), rectangle, width=5)
            if square in legal_destinations:
                pygame.draw.circle(surface, (55, 96, 65), rectangle.center, square_size // 9)
            piece = board.piece_at(square)
            if piece is not None:
                label = _PIECE_LABELS[piece.piece_type]
                if piece.color == chess.BLACK:
                    label = label.lower()
                foreground = (248, 248, 244) if piece.color == chess.WHITE else (22, 25, 31)
                glyph = piece_font.render(label, True, foreground)
                surface.blit(glyph, glyph.get_rect(center=rectangle.center))

    panel_x = board_size + 18
    outcome = board.outcome(claim_draw=True)
    status = outcome.result() if outcome is not None else ("White" if board.turn else "Black")
    lines = (
        "FlyBrain Chess",
        f"turn/result: {status}",
        f"ply: {board.ply()}",
        f"check: {board.is_check()}",
        "F flip | N new",
        "U undo | A side",
        "P pause | [ ] search",
        "Q quit",
    )
    for index, line in enumerate(lines):
        surface.blit(text_font.render(line, True, (224, 230, 243)), (panel_x, 24 + index * 30))
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).copy()


def run_chess_viewer_smoke(
    checkpoint: Path,
    *,
    plies: int,
    seed: int = 7,
    simulations: int = 2,
) -> ChessViewerSmokeResult:
    """Play legal model-vs-model plies and render without opening a display."""

    if plies < 1:
        raise ValueError("chess viewer smoke requires at least one ply")
    import pygame

    pygame.init()
    try:
        state = load_chess_checkpoint(checkpoint)
        board = chess.Board()
        played = 0
        for ply in range(plies):
            if board.is_game_over(claim_draw=True):
                break
            result = search(
                board,
                state.model,
                SearchConfig(simulations=simulations, seed=seed + ply),
                torch.device("cpu"),
            )
            if result.move not in board.legal_moves:
                raise RuntimeError(f"viewer search produced illegal move: {result.move.uci()}")
            board.push(result.move)
            played += 1
        frame = render_chess_frame(board)
        return ChessViewerSmokeResult(
            plies=played,
            illegal_moves=0,
            frame_shape=frame.shape,
            final_fen=board.fen(),
            pgn=str(chess.pgn.Game.from_board(board)),
        )
    finally:
        pygame.quit()


def _display_square(
    position: tuple[int, int],
    *,
    square_size: int,
    flipped: bool,
) -> chess.Square | None:
    x, y = position
    if x < 0 or y < 0 or x >= square_size * 8 or y >= square_size * 8:
        return None
    display_file = x // square_size
    display_rank = y // square_size
    file_index = 7 - display_file if flipped else display_file
    rank_index = display_rank if flipped else 7 - display_rank
    return chess.square(file_index, rank_index)


def run_chess_viewer(config: ChessViewerConfig) -> int:
    """Open a persistent click-to-move or autoplay Pygame chess window."""

    import pygame

    pygame.init()
    try:
        screen = pygame.display.set_mode((config.width, config.height))
        pygame.display.set_caption("FlyBrain Game Learning — Chess")
        clock = pygame.time.Clock()
        state = load_chess_checkpoint(config.checkpoint)
        board = chess.Board()
        flipped = False
        paused = False
        selected: chess.Square | None = None
        human_color = chess.WHITE
        simulations = config.simulations
        running = True
        while running:
            board_size = min(config.height, config.width - 240)
            square_size = board_size // 8
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_f:
                        flipped = not flipped
                    elif event.key == pygame.K_n:
                        board.reset()
                        selected = None
                        paused = False
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_a:
                        human_color = not human_color
                        selected = None
                    elif event.key == pygame.K_u and (paused or config.human_play):
                        if board.move_stack:
                            board.pop()
                        if config.human_play and board.move_stack:
                            board.pop()
                        selected = None
                    elif event.key == pygame.K_LEFTBRACKET:
                        simulations = max(1, simulations // 2)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        simulations = min(10_000, simulations * 2)
                elif event.type == pygame.MOUSEBUTTONDOWN and config.human_play:
                    square = _display_square(
                        event.pos,
                        square_size=square_size,
                        flipped=flipped,
                    )
                    if square is None or board.turn != human_color:
                        continue
                    if selected is None:
                        piece = board.piece_at(square)
                        if piece is not None and piece.color == human_color:
                            selected = square
                    else:
                        candidates = [
                            move
                            for move in board.legal_moves
                            if move.from_square == selected and move.to_square == square
                        ]
                        if candidates:
                            move = next(
                                (
                                    candidate
                                    for candidate in candidates
                                    if candidate.promotion == chess.QUEEN
                                ),
                                candidates[0],
                            )
                            board.push(move)
                        selected = None

            agent_turn = not config.human_play or board.turn != human_color
            if not paused and agent_turn and not board.is_game_over(claim_draw=True):
                result = search(
                    board,
                    state.model,
                    SearchConfig(simulations=simulations, seed=config.seed + board.ply()),
                    torch.device("cpu"),
                )
                if result.move not in board.legal_moves:
                    raise RuntimeError("viewer agent produced an illegal move")
                board.push(result.move)

            frame = render_chess_frame(
                board,
                width=config.width,
                height=config.height,
                flipped=flipped,
                selected=selected,
            )
            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(30)
        return 0
    finally:
        pygame.quit()
