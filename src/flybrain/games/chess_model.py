"""Compact factorized policy/value network for legal chess search."""

from __future__ import annotations

from typing import cast

import chess
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from flybrain.games.chess_env import encode_board, promotion_index

PolicyValueOutput = tuple[Tensor, Tensor, Tensor, Tensor]


class ChessNetConfig(BaseModel, frozen=True):
    """Small CPU-friendly residual network configuration."""

    model_config = ConfigDict(extra="forbid")

    channels: int = Field(default=64, ge=8, le=512)
    residual_blocks: int = Field(default=4, ge=1, le=32)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.first_norm = nn.BatchNorm2d(channels)
        self.second = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.second_norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.activation(self.first_norm(self.first(inputs)))
        hidden = self.second_norm(self.second(hidden))
        return cast(Tensor, self.activation(inputs + hidden))


class ChessPolicyValueNet(nn.Module):
    """Residual encoder with factorized legal-move policy and scalar value."""

    def __init__(self, config: ChessNetConfig | None = None, *, seed: int = 0) -> None:
        super().__init__()
        self.config = config or ChessNetConfig()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            channels = self.config.channels
            self.stem = nn.Sequential(
                nn.Conv2d(21, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            )
            self.blocks = nn.ModuleList(
                _ResidualBlock(channels) for _ in range(self.config.residual_blocks)
            )
            self.from_head = nn.Conv2d(channels, 1, kernel_size=1)
            self.to_head = nn.Conv2d(channels, 1, kernel_size=1)
            self.promotion_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, 5),
            )
            self.value_head = nn.Sequential(
                nn.Conv2d(channels, 8, kernel_size=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(8 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Tanh(),
            )

    def forward(self, inputs: Tensor) -> PolicyValueOutput:
        hidden = self.stem(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        from_logits = self.from_head(hidden).flatten(start_dim=1)
        to_logits = self.to_head(hidden).flatten(start_dim=1)
        promotion_logits = self.promotion_head(hidden)
        value = self.value_head(hidden).squeeze(1)
        return from_logits, to_logits, promotion_logits, value


def score_legal_moves(
    outputs: PolicyValueOutput,
    legal_moves: tuple[chess.Move, ...],
) -> Tensor:
    """Combine factorized heads into one logit for each current legal move."""

    from_logits, to_logits, promotion_logits, _ = outputs
    if from_logits.ndim != 1 or to_logits.ndim != 1 or promotion_logits.ndim != 1:
        raise ValueError("score_legal_moves expects unbatched policy tensors")
    if not legal_moves:
        return torch.empty(0, dtype=from_logits.dtype, device=from_logits.device)
    return torch.stack(
        [
            from_logits[move.from_square]
            + to_logits[move.to_square]
            + promotion_logits[promotion_index(move)]
            for move in legal_moves
        ]
    )


def policy_value(
    model: ChessPolicyValueNet,
    board: chess.Board,
    device: torch.device,
) -> tuple[dict[chess.Move, float], float]:
    """Return a normalized policy over legal moves and side-to-move value."""

    legal_moves = tuple(board.legal_moves)
    if not legal_moves:
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return {}, 0.0
        return {}, 1.0 if outcome.winner == board.turn else -1.0

    was_training = model.training
    model.eval()
    try:
        inputs = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
        with torch.no_grad():
            from_logits, to_logits, promotion_logits, value = model(inputs)
            move_logits = score_legal_moves(
                (from_logits[0], to_logits[0], promotion_logits[0], value[0]),
                legal_moves,
            )
            probabilities = torch.softmax(move_logits, dim=0).cpu().tolist()
        return dict(zip(legal_moves, probabilities, strict=True)), float(value[0].item())
    finally:
        model.train(was_training)
