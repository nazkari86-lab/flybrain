"""Teacher bootstrap, self-play learning, and complete chess checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from flybrain.games.artifacts import publish_directory_atomic, write_json_atomic
from flybrain.games.chess_engine import StockfishInfo, open_stockfish
from flybrain.games.chess_env import encode_board
from flybrain.games.chess_model import (
    ChessNetConfig,
    ChessPolicyValueNet,
    score_legal_moves,
)
from flybrain.games.chess_search import SearchConfig, search


class _BackwardTensor(Protocol):
    def backward(self) -> None: ...


class TrainingExample(BaseModel, frozen=True):
    """One sparse legal policy target and side-to-move value target."""

    model_config = ConfigDict(extra="forbid")

    fen: str = Field(min_length=1)
    policy: dict[str, float]
    value: float = Field(ge=-1.0, le=1.0)
    source: Literal["teacher", "self_play"]
    teacher_sha256: str | None = None
    teacher_nodes: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_policy(self) -> TrainingExample:
        if not self.policy:
            raise ValueError("training policy must contain at least one legal move")
        if any(weight < 0.0 or not math.isfinite(weight) for weight in self.policy.values()):
            raise ValueError("training policy weights must be finite and non-negative")
        if not math.isclose(sum(self.policy.values()), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("training policy weights must sum to one")
        board = chess.Board(self.fen)
        legal = {move.uci() for move in board.legal_moves}
        if not set(self.policy).issubset(legal):
            raise ValueError("training policy contains an illegal move")
        return self


class ChessTrainingConfig(BaseModel, frozen=True):
    """Bounded local teacher and self-play training configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=7, ge=0)
    teacher_positions: int = Field(default=0, ge=0, le=1_000_000)
    teacher_nodes: int = Field(default=2_000, ge=1, le=10_000_000)
    self_play_games: int = Field(default=0, ge=0, le=1_000_000)
    self_play_simulations: int = Field(default=16, ge=1, le=10_000)
    max_game_plies: int = Field(default=160, ge=2, le=2_000)
    epochs: int = Field(default=3, ge=0, le=10_000)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1.0)
    network: ChessNetConfig = Field(default_factory=ChessNetConfig)


class ChessCheckpointManifest(BaseModel, frozen=True):
    """Validated continuation identity for one learned chess checkpoint."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["chess-policy-value-v1"] = "chess-policy-value-v1"
    config: ChessTrainingConfig
    completed_epochs: int = Field(ge=0)
    self_play_games: int = Field(ge=0)
    examples: int = Field(ge=0)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    optimizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rng_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    teacher: StockfishInfo | None
    dependencies: dict[str, str]


@dataclass
class ChessTrainingState:
    """In-memory continuation state reconstructed from one checkpoint."""

    config: ChessTrainingConfig
    model: ChessPolicyValueNet
    optimizer: torch.optim.AdamW
    examples: tuple[TrainingExample, ...]
    teacher: StockfishInfo | None = None
    completed_epochs: int = 0
    self_play_games: int = 0
    manifest: ChessCheckpointManifest | None = None


class ChessRun(BaseModel, frozen=True):
    """Published paths and counters from one training invocation."""

    model_config = ConfigDict(extra="forbid")

    output: Path
    latest: Path
    best: Path
    examples: int = Field(ge=0)
    completed_epochs: int = Field(ge=0)
    self_play_games: int = Field(ge=0)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def make_training_state(
    config: ChessTrainingConfig,
    *,
    examples: tuple[TrainingExample, ...] = (),
    teacher: StockfishInfo | None = None,
) -> ChessTrainingState:
    """Create a deterministically initialized CPU training state."""

    model = ChessPolicyValueNet(config.network, seed=config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    return ChessTrainingState(
        config=config,
        model=model,
        optimizer=optimizer,
        examples=examples,
        teacher=teacher,
    )


def _example_loss(model: ChessPolicyValueNet, example: TrainingExample) -> Tensor:
    board = chess.Board(example.fen)
    legal_moves = tuple(board.legal_moves)
    inputs = torch.from_numpy(encode_board(board)).unsqueeze(0)
    from_logits, to_logits, promotion_logits, values = model(inputs)
    logits = score_legal_moves(
        (from_logits[0], to_logits[0], promotion_logits[0], values[0]),
        legal_moves,
    )
    targets = torch.zeros_like(logits)
    move_indices = {move.uci(): index for index, move in enumerate(legal_moves)}
    for uci, probability in example.policy.items():
        targets[move_indices[uci]] = probability
    policy_loss = -(targets * torch.log_softmax(logits, dim=0)).sum()
    value_target = torch.tensor(example.value, dtype=values.dtype)
    value_loss = torch.square(values[0] - value_target)
    return policy_loss + value_loss


def batch_loss(
    model: ChessPolicyValueNet,
    examples: tuple[TrainingExample, ...],
) -> Tensor:
    """Return mean legal-policy plus value loss for a non-empty batch."""

    if not examples:
        raise ValueError("cannot compute chess loss without examples")
    return torch.stack([_example_loss(model, example) for example in examples]).mean()


def train_batches(
    state: ChessTrainingState,
    *,
    epochs: int | None = None,
    on_epoch: Callable[[ChessTrainingState, float], None] | None = None,
) -> float:
    """Run seeded single-example updates and return the final full-batch loss."""

    count = state.config.epochs if epochs is None else epochs
    if count < 1:
        return float(batch_loss(state.model, state.examples).detach().item())
    if not state.examples:
        raise ValueError("cannot train chess model without examples")
    generator = random.Random(state.config.seed + state.completed_epochs)
    state.model.train()
    for _ in range(count):
        order = list(range(len(state.examples)))
        generator.shuffle(order)
        for index in order:
            state.optimizer.zero_grad(set_to_none=True)
            loss = _example_loss(state.model, state.examples[index])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite chess training loss")
            cast(_BackwardTensor, loss).backward()
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_norm=1.0)
            state.optimizer.step()
        state.completed_epochs += 1
        if on_epoch is not None:
            state.model.eval()
            on_epoch(state, float(batch_loss(state.model, state.examples).detach().item()))
            state.model.train()
    state.model.eval()
    return float(batch_loss(state.model, state.examples).detach().item())


def _random_positions(count: int, seed: int) -> tuple[chess.Board, ...]:
    generator = random.Random(seed)
    positions: list[chess.Board] = []
    seen: set[str] = set()
    attempts = 0
    while len(positions) < count and attempts < max(100, count * 20):
        attempts += 1
        board = chess.Board()
        plies = generator.randint(8, 60)
        for _ in range(plies):
            if board.is_game_over(claim_draw=True):
                break
            board.push(generator.choice(tuple(board.legal_moves)))
        if board.is_game_over(claim_draw=True):
            continue
        key = board.fen()
        if key not in seen:
            seen.add(key)
            positions.append(board)
    if len(positions) != count:
        raise RuntimeError("could not generate enough unique teacher positions")
    return tuple(positions)


def generate_teacher_examples(
    teacher: StockfishInfo,
    *,
    count: int,
    nodes: int,
    seed: int,
) -> tuple[TrainingExample, ...]:
    """Generate sparse policy/value targets from a bounded visible UCI teacher."""

    if count < 1:
        return ()
    examples: list[TrainingExample] = []
    with open_stockfish(teacher) as engine:
        for board in _random_positions(count, seed):
            analyses = engine.analyse(
                board,
                chess.engine.Limit(nodes=nodes),
                multipv=min(5, board.legal_moves.count()),
            )
            scored: list[tuple[str, int]] = []
            for analysis in analyses:
                principal = analysis.get("pv", [])
                score = analysis.get("score")
                if not principal or not isinstance(score, chess.engine.PovScore):
                    continue
                centipawns = score.pov(board.turn).score(mate_score=10_000)
                if centipawns is not None:
                    scored.append((principal[0].uci(), centipawns))
            if not scored:
                raise RuntimeError("Stockfish returned no scored legal move")
            values = np.asarray([score for _, score in scored], dtype=np.float64)
            weights = np.exp((values - values.max()) / 120.0)
            weights /= weights.sum()
            examples.append(
                TrainingExample(
                    fen=board.fen(),
                    policy={
                        move: float(probability)
                        for (move, _), probability in zip(scored, weights, strict=True)
                    },
                    value=float(np.tanh(values[0] / 600.0)),
                    source="teacher",
                    teacher_sha256=teacher.executable_sha256,
                    teacher_nodes=nodes,
                )
            )
    return tuple(examples)


def generate_self_play_game(
    model: ChessPolicyValueNet,
    *,
    simulations: int,
    max_plies: int,
    seed: int,
) -> tuple[tuple[TrainingExample, ...], str]:
    """Generate one legal PUCT self-play game and outcome-labelled examples."""

    board = chess.Board()
    records: list[tuple[str, dict[str, float], chess.Color]] = []
    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        result = search(
            board,
            model,
            SearchConfig(
                simulations=simulations,
                temperature=1.0 if ply < 20 else 0.0,
                seed=seed + ply,
            ),
            torch.device("cpu"),
        )
        total = sum(result.visit_counts.values())
        policy = {
            move.uci(): visits / total for move, visits in result.visit_counts.items() if visits > 0
        }
        records.append((board.fen(), policy, board.turn))
        board.push(result.move)

    outcome = board.outcome(claim_draw=True)
    examples = tuple(
        TrainingExample(
            fen=fen,
            policy=policy,
            value=(
                0.0
                if outcome is None or outcome.winner is None
                else 1.0
                if outcome.winner == side
                else -1.0
            ),
            source="self_play",
        )
        for fen, policy, side in records
    )
    game = chess.pgn.Game.from_board(board)
    return examples, str(game)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependencies() -> dict[str, str]:
    return {package: version(package) for package in ("numpy", "python-chess", "torch")}


def save_chess_checkpoint(target: Path, state: ChessTrainingState) -> Path:
    """Atomically publish model, optimizer, replay, RNG, and validated metadata."""

    holder: list[ChessCheckpointManifest] = []

    def write(stage: Path) -> None:
        torch.save(state.model.state_dict(), stage / "model.pt")
        torch.save(state.optimizer.state_dict(), stage / "optimizer.pt")
        torch.save(torch.get_rng_state(), stage / "rng.pt")
        with stage.joinpath("examples.jsonl").open("x", encoding="utf-8") as stream:
            for example in state.examples:
                stream.write(example.model_dump_json() + "\n")
        manifest = ChessCheckpointManifest(
            config=state.config,
            completed_epochs=state.completed_epochs,
            self_play_games=state.self_play_games,
            examples=len(state.examples),
            model_sha256=_sha256(stage / "model.pt"),
            optimizer_sha256=_sha256(stage / "optimizer.pt"),
            replay_sha256=_sha256(stage / "examples.jsonl"),
            rng_sha256=_sha256(stage / "rng.pt"),
            teacher=state.teacher,
            dependencies=_dependencies(),
        )
        write_json_atomic(stage / "manifest.json", json.loads(manifest.model_dump_json()))
        holder.append(manifest)

    published = publish_directory_atomic(target, write)
    state.manifest = holder[0]
    return published


def load_chess_checkpoint(path: Path) -> ChessTrainingState:
    """Validate hashes and restore a complete CPU chess training state."""

    root = path.resolve()
    manifest = ChessCheckpointManifest.model_validate_json((root / "manifest.json").read_text())
    files = {
        "model.pt": manifest.model_sha256,
        "optimizer.pt": manifest.optimizer_sha256,
        "examples.jsonl": manifest.replay_sha256,
        "rng.pt": manifest.rng_sha256,
    }
    for name, expected in files.items():
        if _sha256(root / name) != expected:
            raise ValueError(f"chess checkpoint hash mismatch: {name}")
    examples = tuple(
        TrainingExample.model_validate_json(line)
        for line in (root / "examples.jsonl").read_text().splitlines()
        if line
    )
    state = make_training_state(manifest.config, examples=examples, teacher=manifest.teacher)
    model_state = cast(dict[str, Tensor], torch.load(root / "model.pt", weights_only=True))
    optimizer_state = cast(dict[str, Any], torch.load(root / "optimizer.pt", weights_only=True))
    rng_state = cast(Tensor, torch.load(root / "rng.pt", weights_only=True))
    state.model.load_state_dict(model_state)
    state.optimizer.load_state_dict(optimizer_state)
    torch.set_rng_state(rng_state)
    state.completed_epochs = manifest.completed_epochs
    state.self_play_games = manifest.self_play_games
    state.manifest = manifest
    return state


def load_chess_policy(path: Path) -> ChessPolicyValueNet:
    """Load a verified frozen policy without restoring optimizer or training RNG."""

    root = path.resolve()
    manifest = ChessCheckpointManifest.model_validate_json((root / "manifest.json").read_text())
    model_file = root / "model.pt"
    if _sha256(model_file) != manifest.model_sha256:
        raise ValueError("chess checkpoint hash mismatch: model.pt")
    model = ChessPolicyValueNet(manifest.config.network, seed=manifest.config.seed)
    model.load_state_dict(cast(dict[str, Tensor], torch.load(model_file, weights_only=True)))
    model.eval()
    return model


def _replace_symlink(link: Path, target: Path) -> None:
    with tempfile.TemporaryDirectory(dir=link.parent) as temporary:
        staged = Path(temporary) / link.name
        staged.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
        os.replace(staged, link)


def train_chess(
    config: ChessTrainingConfig,
    output: Path,
    *,
    teacher: StockfishInfo | None = None,
    resume: Path | None = None,
    on_checkpoint: Callable[[Path, ChessCheckpointManifest, float | None], None] | None = None,
) -> ChessRun:
    """Run bounded teacher/self-play updates and publish a resumable checkpoint."""

    root = output.resolve()
    if resume is None:
        if root.exists():
            raise FileExistsError(root)
        root.mkdir(parents=True)
        state = make_training_state(config, teacher=teacher)
        write_json_atomic(
            root / "run.json",
            {
                "config": json.loads(config.model_dump_json()),
                "protocol": "chess-run-v1",
                "teacher": json.loads(teacher.model_dump_json()) if teacher else None,
            },
        )
    else:
        if not root.is_dir():
            raise FileNotFoundError(root)
        state = load_chess_checkpoint(resume)
        if state.config.model_dump(exclude={"epochs"}) != config.model_dump(exclude={"epochs"}):
            raise ValueError("resume checkpoint is incompatible with requested chess config")
        state.config = config
        state.teacher = teacher or state.teacher

    new_examples: tuple[TrainingExample, ...] = ()
    if config.teacher_positions:
        if state.teacher is None:
            raise ValueError("teacher_positions requires a verified Stockfish engine")
        new_examples += generate_teacher_examples(
            state.teacher,
            count=config.teacher_positions,
            nodes=config.teacher_nodes,
            seed=config.seed + state.completed_epochs,
        )

    replays = root / "replays"
    replays.mkdir(exist_ok=True)
    for index in range(config.self_play_games):
        examples, pgn = generate_self_play_game(
            state.model,
            simulations=config.self_play_simulations,
            max_plies=config.max_game_plies,
            seed=config.seed + state.self_play_games + index,
        )
        new_examples += examples
        replay_path = replays / f"self-play-{state.self_play_games + index + 1:06d}.pgn"
        with replay_path.open("x", encoding="utf-8") as stream:
            stream.write(pgn + "\n")
    state.examples += new_examples
    state.self_play_games += config.self_play_games
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    def publish(progress: ChessTrainingState, loss: float | None) -> Path:
        checkpoint = checkpoints / (
            f"epochs-{progress.completed_epochs:06d}-games-{progress.self_play_games:06d}"
        )
        save_chess_checkpoint(checkpoint, progress)
        _replace_symlink(root / "latest", checkpoint)
        if not root.joinpath("best").exists():
            _replace_symlink(root / "best", checkpoint)
        assert progress.manifest is not None
        if on_checkpoint is not None:
            on_checkpoint(checkpoint, progress.manifest, loss)
        return checkpoint

    if config.epochs and state.examples:
        if on_checkpoint is None:
            train_batches(state, epochs=config.epochs)
        else:

            def publish_epoch(progress: ChessTrainingState, loss: float) -> None:
                publish(progress, loss)

            train_batches(state, epochs=config.epochs, on_epoch=publish_epoch)
    if on_checkpoint is None or not (config.epochs and state.examples):
        publish(state, None)
    assert state.manifest is not None
    return ChessRun(
        output=root,
        latest=root / "latest",
        best=root / "best",
        examples=len(state.examples),
        completed_epochs=state.completed_epochs,
        self_play_games=state.self_play_games,
        model_sha256=state.manifest.model_sha256,
    )
