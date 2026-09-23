"""Immutable chess matches against frozen reproducible opponents."""

from __future__ import annotations

import hashlib
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from flybrain.games.chess_engine import StockfishInfo, open_stockfish
from flybrain.games.chess_model import ChessPolicyValueNet
from flybrain.games.chess_search import SearchConfig, search
from flybrain.games.chess_training import load_chess_checkpoint

OpponentKind = Literal["random", "material", "checkpoint", "stockfish"]


class OpponentConfig(BaseModel, frozen=True):
    """Complete frozen identity and strength limit for one opponent."""

    model_config = ConfigDict(extra="forbid")

    kind: OpponentKind
    name: str = Field(min_length=1)
    baseline_elo: int
    depth: int = Field(default=2, ge=1, le=6)
    nodes: int = Field(default=1_000, ge=1, le=10_000_000)
    checkpoint: Path | None = None
    stockfish: StockfishInfo | None = None


class ChessGameResult(BaseModel, frozen=True):
    """One game result from the evaluated agent's perspective."""

    model_config = ConfigDict(extra="forbid")

    opponent: str
    seed: int = Field(ge=0)
    agent_color: Literal["white", "black"]
    result: Literal["win", "draw", "loss"]
    plies: int = Field(ge=1)
    termination: str
    pgn: str
    agent_move_seconds: float = Field(ge=0.0)
    search_nodes: int = Field(ge=0)


class ChessEvaluation(BaseModel, frozen=True):
    """Frozen match evidence with opponent-relative rating estimate."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["chess-evaluation-v1"] = "chess-evaluation-v1"
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    games: int = Field(ge=1)
    wins: int = Field(ge=0)
    draws: int = Field(ge=0)
    losses: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    estimated_elo: float
    elo_low: float
    elo_high: float
    illegal_moves: int = Field(ge=0)
    mean_agent_move_seconds: float = Field(ge=0.0)
    search_nodes: int = Field(ge=0)
    training_mutated: bool
    per_game: tuple[ChessGameResult, ...]


def random_opponent(*, baseline_elo: int = 200) -> OpponentConfig:
    return OpponentConfig(kind="random", name="seeded-random", baseline_elo=baseline_elo)


def material_opponent(*, depth: int = 2, baseline_elo: int = 700) -> OpponentConfig:
    return OpponentConfig(
        kind="material",
        name=f"material-depth-{depth}",
        baseline_elo=baseline_elo,
        depth=depth,
    )


def checkpoint_opponent(checkpoint: Path, *, baseline_elo: int = 800) -> OpponentConfig:
    return OpponentConfig(
        kind="checkpoint",
        name=f"checkpoint-{checkpoint.name}",
        baseline_elo=baseline_elo,
        checkpoint=checkpoint,
    )


def stockfish_opponent(
    stockfish: StockfishInfo,
    *,
    nodes: int = 1_000,
    baseline_elo: int = 1_400,
) -> OpponentConfig:
    return OpponentConfig(
        kind="stockfish",
        name=f"{stockfish.name}-nodes-{nodes}",
        baseline_elo=baseline_elo,
        nodes=nodes,
        stockfish=stockfish,
    )


def _hash_tree(path: Path) -> str:
    root = path.resolve()
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(root).as_posix().encode())
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


_PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def _material_score(board: chess.Board, perspective: chess.Color) -> int:
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner is None:
            return 0
        return 100_000 if outcome.winner == perspective else -100_000
    score = 0
    for piece_type, value in _PIECE_VALUES.items():
        score += value * len(board.pieces(piece_type, perspective))
        score -= value * len(board.pieces(piece_type, not perspective))
    return score


def _minimax(board: chess.Board, depth: int, perspective: chess.Color) -> int:
    if depth == 0 or board.is_game_over(claim_draw=True):
        return _material_score(board, perspective)
    values: list[int] = []
    for move in sorted(board.legal_moves, key=lambda item: item.uci()):
        board.push(move)
        values.append(_minimax(board, depth - 1, perspective))
        board.pop()
    return max(values) if board.turn == perspective else min(values)


def _material_move(board: chess.Board, depth: int) -> chess.Move:
    perspective = board.turn
    candidates: list[tuple[int, str, chess.Move]] = []
    for move in board.legal_moves:
        board.push(move)
        score = _minimax(board, depth - 1, perspective)
        board.pop()
        candidates.append((-score, move.uci(), move))
    return min(candidates)[2]


def _opponent_move(
    board: chess.Board,
    opponent: OpponentConfig,
    *,
    generator: random.Random,
    model: ChessPolicyValueNet | None,
    engine: chess.engine.SimpleEngine | None,
    simulations: int,
    seed: int,
) -> chess.Move:
    legal = tuple(board.legal_moves)
    if opponent.kind == "random":
        return generator.choice(legal)
    if opponent.kind == "material":
        return _material_move(board, opponent.depth)
    if opponent.kind == "checkpoint":
        if model is None:
            raise ValueError("checkpoint opponent model is unavailable")
        return search(
            board,
            model,
            SearchConfig(simulations=simulations, seed=seed),
            torch.device("cpu"),
        ).move
    if engine is None:
        raise ValueError("Stockfish opponent engine is unavailable")
    result = engine.play(board, chess.engine.Limit(nodes=opponent.nodes))
    if result.move is None:
        raise RuntimeError("Stockfish returned no move for a nonterminal position")
    return result.move


def _play_game(
    agent: ChessPolicyValueNet,
    opponent: OpponentConfig,
    *,
    agent_color: chess.Color,
    seed: int,
    search_simulations: int,
    max_plies: int,
    opponent_model: ChessPolicyValueNet | None,
    engine: chess.engine.SimpleEngine | None,
) -> ChessGameResult:
    board = chess.Board()
    generator = random.Random(seed)
    agent_seconds = 0.0
    agent_moves = 0
    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        if board.turn == agent_color:
            started = time.perf_counter()
            move = search(
                board,
                agent,
                SearchConfig(simulations=search_simulations, seed=seed + ply),
                torch.device("cpu"),
            ).move
            agent_seconds += time.perf_counter() - started
            agent_moves += 1
        else:
            move = _opponent_move(
                board,
                opponent,
                generator=generator,
                model=opponent_model,
                engine=engine,
                simulations=search_simulations,
                seed=seed + ply,
            )
        if move not in board.legal_moves:
            raise RuntimeError(f"evaluation produced illegal move: {move.uci()}")
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        result: Literal["win", "draw", "loss"] = "draw"
    elif outcome.winner == agent_color:
        result = "win"
    else:
        result = "loss"
    termination = outcome.termination.name.lower() if outcome is not None else "max_plies"
    return ChessGameResult(
        opponent=opponent.name,
        seed=seed,
        agent_color="white" if agent_color == chess.WHITE else "black",
        result=result,
        plies=board.ply(),
        termination=termination,
        pgn=str(chess.pgn.Game.from_board(board)),
        agent_move_seconds=agent_seconds,
        search_nodes=agent_moves * search_simulations,
    )


def estimate_elo(score: float, games: int, baseline_elo: float) -> float:
    """Return a bounded opponent-relative logistic Elo estimate."""

    bounded = min(1.0 - 1.0 / (2 * games), max(1.0 / (2 * games), score))
    return baseline_elo + 400.0 * math.log10(bounded / (1.0 - bounded))


def evaluate_chess(
    checkpoint: Path,
    *,
    opponents: tuple[OpponentConfig, ...],
    seeds: tuple[int, ...],
    search_simulations: int = 16,
    max_plies: int = 160,
) -> ChessEvaluation:
    """Play both colors against each frozen opponent without checkpoint writes."""

    if not opponents or not seeds:
        raise ValueError("chess evaluation requires opponents and seeds")
    if search_simulations < 1 or max_plies < 2:
        raise ValueError("chess evaluation limits must be positive")
    root = checkpoint.resolve()
    before = _hash_tree(root)
    state = load_chess_checkpoint(root)
    if state.manifest is None:
        raise RuntimeError("loaded chess checkpoint lacks manifest")
    games: list[ChessGameResult] = []
    baseline_elos: list[int] = []
    for opponent in opponents:
        opponent_model = None
        if opponent.kind == "checkpoint":
            if opponent.checkpoint is None:
                raise ValueError("checkpoint opponent path is required")
            opponent_model = load_chess_checkpoint(opponent.checkpoint).model
        context = (
            open_stockfish(opponent.stockfish)
            if opponent.kind == "stockfish" and opponent.stockfish is not None
            else nullcontext(None)
        )
        with context as engine:
            for seed in seeds:
                for color in (chess.WHITE, chess.BLACK):
                    games.append(
                        _play_game(
                            state.model,
                            opponent,
                            agent_color=color,
                            seed=seed,
                            search_simulations=search_simulations,
                            max_plies=max_plies,
                            opponent_model=opponent_model,
                            engine=engine,
                        )
                    )
                    baseline_elos.append(opponent.baseline_elo)
    after = _hash_tree(root)
    wins = sum(game.result == "win" for game in games)
    draws = sum(game.result == "draw" for game in games)
    losses = len(games) - wins - draws
    points = np.asarray(
        [1.0 if game.result == "win" else 0.5 if game.result == "draw" else 0.0 for game in games]
    )
    score = float(points.mean())
    baseline = float(np.mean(baseline_elos))
    generator = np.random.default_rng(0)
    bootstraps = np.asarray(
        [
            estimate_elo(
                float(generator.choice(points, size=len(points)).mean()),
                len(points),
                baseline,
            )
            for _ in range(500)
        ]
    )
    total_agent_moves = sum(game.search_nodes for game in games) / search_simulations
    return ChessEvaluation(
        checkpoint_sha256=state.manifest.model_sha256,
        games=len(games),
        wins=wins,
        draws=draws,
        losses=losses,
        score=score,
        estimated_elo=estimate_elo(score, len(games), baseline),
        elo_low=float(np.quantile(bootstraps, 0.025)),
        elo_high=float(np.quantile(bootstraps, 0.975)),
        illegal_moves=0,
        mean_agent_move_seconds=(
            sum(game.agent_move_seconds for game in games) / total_agent_moves
            if total_agent_moves
            else 0.0
        ),
        search_nodes=sum(game.search_nodes for game in games),
        training_mutated=before != after,
        per_game=tuple(games),
    )
