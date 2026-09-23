"""Deterministic legal-move PUCT search guided by the learned chess network."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from flybrain.games.chess_model import ChessPolicyValueNet, policy_value


class SearchConfig(BaseModel, frozen=True):
    """Bounded PUCT parameters for evaluation or self-play."""

    model_config = ConfigDict(extra="forbid")

    simulations: int = Field(default=64, ge=1, le=100_000)
    c_puct: float = Field(default=1.5, gt=0.0, le=100.0)
    temperature: float = Field(default=0.0, ge=0.0, le=10.0)
    seed: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class SearchResult:
    """Selected legal move plus reproducible root evidence."""

    move: chess.Move
    visit_counts: dict[chess.Move, int]
    root_value: float


@dataclass
class _Node:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, _Node] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def _terminal_value(board: chess.Board) -> float | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def _expand(
    node: _Node,
    board: chess.Board,
    model: ChessPolicyValueNet,
    device: torch.device,
) -> float:
    priors, value = policy_value(model, board, device)
    node.children = {
        move: _Node(prior=prior)
        for move, prior in sorted(priors.items(), key=lambda item: item[0].uci())
    }
    return value


def _select(node: _Node, c_puct: float) -> tuple[chess.Move, _Node]:
    parent_scale = math.sqrt(max(1, node.visits))

    def score(item: tuple[chess.Move, _Node]) -> float:
        _, child = item
        exploitation = -child.value
        exploration = c_puct * child.prior * parent_scale / (1 + child.visits)
        return exploitation + exploration

    return min(node.children.items(), key=lambda item: (-score(item), item[0].uci()))


def _immediate_mate(board: chess.Board) -> chess.Move | None:
    for move in sorted(board.legal_moves, key=lambda candidate: candidate.uci()):
        board.push(move)
        checkmate = board.is_checkmate()
        board.pop()
        if checkmate:
            return move
    return None


def _choose_root_move(
    visits: dict[chess.Move, int],
    config: SearchConfig,
) -> chess.Move:
    ordered = sorted(visits, key=lambda move: move.uci())
    if config.temperature <= 1e-9:
        return min(ordered, key=lambda move: (-visits[move], move.uci()))
    counts = np.asarray([visits[move] for move in ordered], dtype=np.float64)
    weights = np.power(np.maximum(counts, 1e-12), 1.0 / config.temperature)
    probabilities = weights / weights.sum()
    generator = np.random.default_rng(config.seed)
    return ordered[int(generator.choice(len(ordered), p=probabilities))]


def search(
    board: chess.Board,
    model: ChessPolicyValueNet,
    config: SearchConfig,
    device: torch.device,
) -> SearchResult:
    """Run PUCT without mutating the caller's board and return a legal move."""

    if board.is_game_over(claim_draw=True):
        raise ValueError("cannot search a terminal chess position")
    mating_move = _immediate_mate(board)
    legal_moves = tuple(board.legal_moves)
    if mating_move is not None:
        counts = {move: config.simulations if move == mating_move else 0 for move in legal_moves}
        return SearchResult(move=mating_move, visit_counts=counts, root_value=1.0)

    root = _Node(prior=1.0)
    _expand(root, board, model, device)
    for _ in range(config.simulations):
        simulation_board = board.copy(stack=True)
        node = root
        path = [root]
        while node.children:
            move, node = _select(node, config.c_puct)
            simulation_board.push(move)
            path.append(node)

        value = _terminal_value(simulation_board)
        if value is None:
            value = _expand(node, simulation_board, model, device)
        for visited in reversed(path):
            visited.visits += 1
            visited.value_sum += value
            value = -value

    visit_counts = {move: child.visits for move, child in root.children.items()}
    selected = _choose_root_move(visit_counts, config)
    return SearchResult(
        move=selected,
        visit_counts=visit_counts,
        root_value=root.value,
    )
