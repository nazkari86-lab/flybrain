# Chess Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a legal, persistent chess learner that bootstraps from an explicitly recorded Stockfish teacher when available, continues through policy/value-guided self-play, and is evaluated against frozen opponents.

**Architecture:** `python-chess` owns rules and UCI communication. A compact PyTorch residual network predicts factorized from-square, to-square, promotion policy logits plus position value; PUCT scores only current legal moves, avoiding a large mostly invalid global action head. Shared FlyBrain game artifacts store training examples, complete model/optimizer/RNG checkpoints, PGNs, promotion matches, and immutable evaluations.

**Tech Stack:** Python 3.12, NumPy 2.x, Pydantic 2.x, PyTorch 2.14.x, python-chess 1.999, optional Stockfish UCI engine, Pygame, Typer, pytest, Ruff, mypy

**Spec:** `docs/superpowers/specs/2026-09-24-game-learning-platform-design.md`

## Global Constraints

- Complete the game-platform and runner plan first; consume its contracts and artifact helpers without duplicating them.
- Every emitted chess move must come from `board.legal_moves`; an illegal move is an implementation failure.
- Stockfish is an optional, visible teacher and benchmark. It is never bundled into or described as the learned policy.
- Every Stockfish artifact records engine path, version, node limit, and source position FEN.
- Self-play checkpoints replace `best/` only after a fixed promotion match against the frozen incumbent.
- Evaluation uses fixed seeds and node limits and cannot update model, optimizer, replay, or training counters.
- `python-chess` and Stockfish are GPL-licensed dependencies; record that boundary in the reuse document before distribution.
- Preserve unrelated dirty-worktree files and keep engineered chess AI separate from MaleCNS claims.

## File Map

- `pyproject.toml`: adds `python-chess` to the optional `games` group.
- `docs/data/game-learning-reuse-2026-09-24.md`: adds python-chess and Stockfish license/version results.
- `src/flybrain/games/chess_env.py`: board encoding, legal moves, terminal rewards, FEN/PGN state.
- `src/flybrain/games/chess_model.py`: compact residual policy/value network and legal-move scoring.
- `src/flybrain/games/chess_search.py`: deterministic PUCT over legal moves.
- `src/flybrain/games/chess_training.py`: teacher examples, self-play, minibatch updates, promotion, checkpoints.
- `src/flybrain/games/chess_evaluation.py`: fixed opponent ladder, Elo estimate, immutable reports.
- `src/flybrain/games/chess_viewer.py`: local board viewer and human-vs-agent mode.
- `src/flybrain/games/cli.py`: extends the existing games commands with `--game chess`.
- `tests/games/test_chess_*.py`: focused and end-to-end tests.
- `README.md`: exact chess commands and measured evidence.

---

### Task 1: Add Chess Rules and Resolve Stockfish

**Files:**
- Modify: `pyproject.toml`
- Modify: `docs/data/game-learning-reuse-2026-09-24.md`
- Create: `src/flybrain/games/chess_engine.py`
- Create: `tests/games/test_chess_engine.py`

**Interfaces:**
- Consumes: optional `games` extra from the runner plan.
- Produces: `StockfishInfo`, `resolve_stockfish(explicit: Path | None) -> StockfishInfo | None`, and `open_stockfish(info)` context manager.

- [ ] **Step 1: Write failing dependency and resolver tests**

```python
from pathlib import Path

from flybrain.games.chess_engine import resolve_stockfish


def test_explicit_missing_stockfish_returns_none(tmp_path: Path) -> None:
    assert resolve_stockfish(tmp_path / "missing") is None


def test_path_resolution_never_invents_engine_metadata(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "")
    assert resolve_stockfish(None) is None
```

- [ ] **Step 2: Run tests and verify missing module**

Run: `uv run pytest tests/games/test_chess_engine.py -v`

Expected: FAIL importing `chess_engine`.

- [ ] **Step 3: Add dependency and exact resolver behavior**

Add `"python-chess>=1.999,<2"` to the `games` extra and run `uv sync --extra dev --extra games`. Resolve an explicit executable first, then `shutil.which("stockfish")`, then Apple Homebrew paths `/opt/homebrew/bin/stockfish` and `/usr/local/bin/stockfish`. Probe with `chess.engine.SimpleEngine.popen_uci`, call `engine.id`, and return name, author, executable SHA-256, and path. Catch only `FileNotFoundError`, `PermissionError`, `chess.engine.EngineError`, and `chess.engine.EngineTerminatedError` and return `None` with no fabricated version.

If Homebrew is available and Stockfish is absent, install with `brew install stockfish`, then record `stockfish --version` and the GPL license boundary in the reuse document.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/games/test_chess_engine.py -v`

Run: `uv run ruff check src/flybrain/games/chess_engine.py tests/games/test_chess_engine.py`

Run: `uv run mypy src/flybrain/games/chess_engine.py`

```bash
git add pyproject.toml uv.lock docs/data/game-learning-reuse-2026-09-24.md src/flybrain/games/chess_engine.py tests/games/test_chess_engine.py
git commit -m "build: add chess rules and engine discovery"
```

---

### Task 2: Chess Adapter and 21-Plane Encoding

**Files:**
- Create: `src/flybrain/games/chess_env.py`
- Create: `tests/games/test_chess_env.py`

**Interfaces:**
- Consumes: `GameAdapter[np.ndarray, chess.Move]` and python-chess.
- Produces: `encode_board(board) -> NDArray[np.float32]` shape `(21, 8, 8)`, `promotion_index(move) -> int`, and `ChessAdapter` returning legal `chess.Move` actions.

- [ ] **Step 1: Write failing encoding and legality tests**

```python
import chess
import numpy as np

from flybrain.games.chess_env import ChessAdapter, encode_board


def test_initial_board_encoding_is_stable() -> None:
    encoded = encode_board(chess.Board())
    assert encoded.shape == (21, 8, 8)
    assert encoded.dtype == np.float32
    assert int(encoded[:12].sum()) == 32


def test_adapter_rejects_illegal_move() -> None:
    adapter = ChessAdapter()
    adapter.reset(seed=7, mode="play")
    illegal = chess.Move.from_uci("e2e5")
    with pytest.raises(ValueError, match="illegal move"):
        adapter.step(illegal)
```

- [ ] **Step 2: Run tests and verify the adapter is absent**

Run: `uv run pytest tests/games/test_chess_env.py -v`

Expected: FAIL importing `chess_env`.

- [ ] **Step 3: Implement the exact planes and terminal result**

Planes `0..11` represent white then black pawn, knight, bishop, rook, queen, king occupancy. Plane `12` is all ones for white to move and zero for black. Planes `13..16` are white/black king/queen-side castling rights. Plane `17` marks the en-passant square. Plane `18` is `min(halfmove_clock, 100) / 100`. Plane `19` is all ones when the current position has occurred at least twice. Planes `20` marks the destination square of the last move, or zero when no move exists.

`ChessAdapter.step` checks membership in `board.legal_moves`, pushes the move, and returns reward from the mover's perspective: `+1` for a win, `-1` for a loss, `0` for draw/nonterminal. It includes FEN, UCI, ply, outcome, and termination reason in `info`.

- [ ] **Step 4: Verify representative positions and commit**

Add tests for castling rights, en-passant, promotion legal moves, checkmate, stalemate, threefold claim, and encoding finiteness.

Run: `uv run pytest tests/games/test_chess_env.py -v`

Run: `uv run ruff check src/flybrain/games/chess_env.py tests/games/test_chess_env.py`

Run: `uv run mypy src/flybrain/games/chess_env.py`

```bash
git add src/flybrain/games/chess_env.py tests/games/test_chess_env.py
git commit -m "feat: add legal chess adapter and encoding"
```

---

### Task 3: Compact Policy/Value Network

**Files:**
- Create: `src/flybrain/games/chess_model.py`
- Create: `tests/games/test_chess_model.py`

**Interfaces:**
- Consumes: `(batch, 21, 8, 8)` tensors and tuples of legal `chess.Move`.
- Produces: `ChessNetConfig`, `ChessPolicyValueNet.forward`, `score_legal_moves(outputs, legal_moves)`, and `policy_value(model, board, device) -> (dict[chess.Move, float], float)`.

- [ ] **Step 1: Write failing shape and masking tests**

```python
import chess
import torch

from flybrain.games.chess_model import ChessPolicyValueNet, policy_value


def test_network_scores_only_legal_moves() -> None:
    board = chess.Board()
    model = ChessPolicyValueNet()
    priors, value = policy_value(model, board, torch.device("cpu"))
    assert set(priors) == set(board.legal_moves)
    assert abs(sum(priors.values()) - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0
```

- [ ] **Step 2: Run tests and verify missing model**

Run: `uv run pytest tests/games/test_chess_model.py -v`

Expected: FAIL importing `chess_model`.

- [ ] **Step 3: Implement the compact residual network**

Use a `Conv2d(21, 64, 3, padding=1)` stem with batch normalization and ReLU, followed by four residual blocks of two `Conv2d(64, 64, 3, padding=1)` layers. Policy heads produce 64 from-square logits, 64 to-square logits, and five promotion logits (`none`, knight, bishop, rook, queen). The value head uses `Conv2d(64, 8, 1)`, flatten, `Linear(512, 128)`, `Linear(128, 1)`, and `tanh`.

For each legal move, its logit is `from_logits[from_square] + to_logits[to_square] + promotion_logits[promotion_index]`. Apply softmax only across legal move logits. Initialize torch with the recorded seed before construction.

- [ ] **Step 4: Test gradients and deterministic initialization**

Add a test that cross-entropy plus mean-squared value loss produces finite nonzero gradients and that two models initialized with the same seed have identical state dictionaries.

Run: `uv run pytest tests/games/test_chess_model.py -v`

Run: `uv run ruff check src/flybrain/games/chess_model.py tests/games/test_chess_model.py`

Run: `uv run mypy src/flybrain/games/chess_model.py`

- [ ] **Step 5: Commit**

```bash
git add src/flybrain/games/chess_model.py tests/games/test_chess_model.py
git commit -m "feat: add chess policy value network"
```

---

### Task 4: Deterministic PUCT Search

**Files:**
- Create: `src/flybrain/games/chess_search.py`
- Create: `tests/games/test_chess_search.py`

**Interfaces:**
- Consumes: `policy_value(model, board, device)`.
- Produces: `SearchConfig(simulations, c_puct, temperature, seed)`, `SearchResult(move, visit_counts, root_value)`, and `search(board, model, config, device)`.

- [ ] **Step 1: Write failing tactical and legality tests**

```python
import chess
import torch

from flybrain.games.chess_model import ChessPolicyValueNet
from flybrain.games.chess_search import SearchConfig, search


def test_search_always_returns_legal_move() -> None:
    board = chess.Board()
    result = search(board, ChessPolicyValueNet(), SearchConfig(simulations=8, seed=7), torch.device("cpu"))
    assert result.move in board.legal_moves
    assert sum(result.visit_counts.values()) == 8


def test_search_takes_forced_mate_in_one() -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    result = search(board, ChessPolicyValueNet(), SearchConfig(simulations=32, seed=7), torch.device("cpu"))
    board.push(result.move)
    assert board.is_checkmate()
```

- [ ] **Step 2: Run tests and verify missing search module**

Run: `uv run pytest tests/games/test_chess_search.py -v`

Expected: FAIL importing `chess_search`.

- [ ] **Step 3: Implement PUCT with terminal overrides**

Each node stores prior, visits, value sum, children keyed by UCI, and side to move. Selection maximizes `Q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)`. Expansion calls the network once, creates children for every legal move, and backs up value with sign inversion per ply. Terminal checkmate values override the network before expansion; draws return zero. Break exact score ties by sorted UCI order in deterministic evaluation mode.

Root move selection uses visit counts with temperature during self-play and maximum visits during evaluation. Return counts for every legal root move.

- [ ] **Step 4: Run search tests and commit**

Run: `uv run pytest tests/games/test_chess_search.py -v`

Run: `uv run ruff check src/flybrain/games/chess_search.py tests/games/test_chess_search.py`

Run: `uv run mypy src/flybrain/games/chess_search.py`

```bash
git add src/flybrain/games/chess_search.py tests/games/test_chess_search.py
git commit -m "feat: add legal policy guided chess search"
```

---

### Task 5: Teacher Bootstrap, Self-Play, and Complete Checkpoints

**Files:**
- Create: `src/flybrain/games/chess_training.py`
- Create: `tests/games/test_chess_training.py`

**Interfaces:**
- Consumes: chess adapter, model, PUCT, optional Stockfish info, and shared atomic artifacts.
- Produces: `ChessTrainingConfig`, `TrainingExample`, `generate_teacher_examples`, `generate_self_play_game`, `train_chess`, `save_chess_checkpoint`, and `load_chess_checkpoint`.

- [ ] **Step 1: Write failing update and checkpoint tests**

```python
def test_tiny_batch_update_reduces_loss() -> None:
    model, examples = deterministic_tiny_training_fixture(seed=7)
    before = batch_loss(model, examples).item()
    train_batches(model, examples, epochs=20, learning_rate=1e-3, seed=7)
    after = batch_loss(model, examples).item()
    assert after < before


def test_checkpoint_restores_model_optimizer_counters_and_rng(tmp_path) -> None:
    state = deterministic_training_state(seed=11)
    checkpoint = save_chess_checkpoint(tmp_path / "checkpoint", state)
    restored = load_chess_checkpoint(checkpoint)
    assert restored.manifest == state.manifest
    assert equal_state_dicts(restored.model.state_dict(), state.model.state_dict())
    assert restored.replay_hash == state.replay_hash
```

- [ ] **Step 2: Run tests and verify missing training module**

Run: `uv run pytest tests/games/test_chess_training.py -v`

Expected: FAIL importing `chess_training`.

- [ ] **Step 3: Implement teacher examples**

Generate positions from seeded legal random playouts of 8 to 60 plies, rejecting terminal positions and duplicate FENs. Query Stockfish with `chess.engine.Limit(nodes=config.teacher_nodes)` and `multipv=min(5, legal_move_count)`. Convert centipawn scores to move probabilities with softmax temperature 120 cp and position value with `tanh(score_cp / 600)`, using `mate_score=10_000`. Store FEN, sparse UCI policy targets, value, engine metadata hash, and node limit in each example.

- [ ] **Step 4: Implement self-play and minibatch training**

For each self-play turn, run PUCT, normalize root visit counts into sparse legal-move targets, sample while ply is below `temperature_plies`, then choose max visits. At termination, assign `+1/-1/0` values from each recorded side-to-move perspective and write a valid PGN. Train with legal-move cross-entropy plus value MSE, AdamW, gradient clipping at `1.0`, finite-loss checks, and seeded minibatch ordering.

- [ ] **Step 5: Implement full atomic checkpoints and promotion**

Checkpoint `chess-policy-value-v1` includes model and optimizer state, counters, Python/NumPy/Torch RNG states, replay-chunk hashes, training config, dependency versions, teacher provenance, and accepted-incumbent hash. A candidate becomes `best/` only after a deterministic two-sided match scores at least `55%` over the configured number of games against the frozen incumbent; otherwise preserve it under `candidates/` and continue from `latest/`.

- [ ] **Step 6: Verify focused tests and commit**

Run: `uv run pytest tests/games/test_chess_training.py -v`

Run: `uv run ruff check src/flybrain/games/chess_training.py tests/games/test_chess_training.py`

Run: `uv run mypy src/flybrain/games/chess_training.py`

```bash
git add src/flybrain/games/chess_training.py tests/games/test_chess_training.py
git commit -m "feat: add chess teacher and self play learning"
```

---

### Task 6: Frozen Chess Evaluation Ladder

**Files:**
- Create: `src/flybrain/games/chess_evaluation.py`
- Create: `tests/games/test_chess_evaluation.py`

**Interfaces:**
- Consumes: a frozen checkpoint and fixed opponent configurations.
- Produces: `OpponentConfig`, `ChessEvaluation`, `evaluate_chess(checkpoint, opponents, seeds)`, and `estimate_elo(score, games, baseline_elo)`.

- [ ] **Step 1: Write failing evaluation tests**

```python
def test_evaluation_is_immutable_and_legal(tmp_path) -> None:
    checkpoint = tiny_checkpoint(tmp_path)
    before = hash_tree(checkpoint)
    result = evaluate_chess(checkpoint, opponents=(random_opponent(),), seeds=(1, 2))
    assert result.illegal_moves == 0
    assert result.games == 4
    assert hash_tree(checkpoint) == before
    assert result.training_mutated is False
```

- [ ] **Step 2: Run tests and verify missing evaluation module**

Run: `uv run pytest tests/games/test_chess_evaluation.py -v`

Expected: FAIL importing `chess_evaluation`.

- [ ] **Step 3: Implement fixed opponents and reports**

Implement random legal play, a depth-2 negamax opponent with material values `P=100, N=320, B=330, R=500, Q=900`, frozen checkpoint PUCT, and optional Stockfish node-limited opponents. Play both colors for every seed. Record W/D/L, score, legal-move rate, mean move latency, search nodes, termination reasons, PGN paths, opponent identity, and checkpoint hash.

Compute indicative Elo difference as `400 * log10(score / (1 - score))`, clamping score to `[1/(2*games), 1-1/(2*games)]`; report a bootstrap interval and label it an estimate tied to the declared opponent, not an official rating.

- [ ] **Step 4: Verify evaluation and commit**

Run: `uv run pytest tests/games/test_chess_evaluation.py -v`

Run: `uv run ruff check src/flybrain/games/chess_evaluation.py tests/games/test_chess_evaluation.py`

Run: `uv run mypy src/flybrain/games/chess_evaluation.py`

```bash
git add src/flybrain/games/chess_evaluation.py tests/games/test_chess_evaluation.py
git commit -m "feat: add immutable chess evaluation ladder"
```

---

### Task 7: Chess Viewer and Unified CLI

**Files:**
- Create: `src/flybrain/games/chess_viewer.py`
- Modify: `src/flybrain/games/cli.py`
- Create: `tests/games/test_chess_viewer.py`
- Modify: `tests/games/test_games_cli.py`

**Interfaces:**
- Consumes: chess checkpoint, search, and evaluation functions.
- Produces: `run_chess_viewer`, `run_chess_viewer_smoke`, and `--game chess` support in train/evaluate/watch/play.

- [ ] **Step 1: Write failing viewer and CLI tests**

```python
def test_chess_viewer_smoke_plays_only_legal_moves(tiny_checkpoint_path) -> None:
    result = run_chess_viewer_smoke(tiny_checkpoint_path, plies=8, seed=7)
    assert result.plies == 8
    assert result.illegal_moves == 0
    assert result.frame_shape == (720, 960, 3)


def test_chess_play_dry_run(cli_runner, tiny_checkpoint_path) -> None:
    result = cli_runner.invoke(
        app,
        ["games", "play", "--game", "chess", "--checkpoint", str(tiny_checkpoint_path), "--dry-run", "--steps", "4"],
    )
    assert result.exit_code == 0, result.output
```

- [ ] **Step 2: Run tests and verify chess UI is absent**

Run: `SDL_VIDEODRIVER=dummy uv run pytest tests/games/test_chess_viewer.py tests/games/test_games_cli.py -v`

Expected: FAIL importing `chess_viewer` or rejecting `--game chess`.

- [ ] **Step 3: Implement board UI and controls**

Draw an 8x8 board, Unicode or bundled Pygame piece glyphs, selected square, legal destinations, last move, check marker, clocks measured as search time, evaluation, and principal move. Controls: click source/destination, `F = flip`, `N = new game`, `U = undo only in analysis mode`, `A = choose agent side`, `P = pause autoplay`, `[`/`] = search simulations`, `Q/Escape = quit`. Never accept a click-generated move not present in `board.legal_moves`.

The headless smoke renderer uses `pygame.Surface` with no display, plays a fixed number of legal PUCT plies, and returns frame shape, plies, illegal moves, final FEN, and PGN.

- [ ] **Step 4: Extend CLI dispatch**

Change the game type to `Literal["runner", "chess"]`. Chess train accepts `--self-play-games`, `--teacher-positions`, `--stockfish`, `--seed`, `--output`, and `--resume`. Evaluate accepts `--games-per-opponent` and `--stockfish`. Watch/play accept `--simulations` and retain common dry-run options. Game-specific invalid options fail with a Typer message rather than being ignored.

- [ ] **Step 5: Verify and commit**

Run: `SDL_VIDEODRIVER=dummy uv run pytest tests/games/test_chess_viewer.py tests/games/test_games_cli.py -v`

Run: `uv run ruff check src/flybrain/games/chess_viewer.py src/flybrain/games/cli.py tests/games`

Run: `uv run mypy src/flybrain/games`

```bash
git add src/flybrain/games/chess_viewer.py src/flybrain/games/cli.py tests/games/test_chess_viewer.py tests/games/test_games_cli.py
git commit -m "feat: expose playable chess learner"
```

---

### Task 8: End-to-End Chess Evidence and Documentation

**Files:**
- Modify: `README.md`
- Create: `tests/games/test_chess_end_to_end.py`
- Generate: `artifacts/games/chess/seed7-v1/`

**Interfaces:**
- Consumes: all chess commands and a resolved optional Stockfish binary.
- Produces: teacher provenance, trained checkpoint, self-play PGNs, frozen-opponent evaluation, and exact launch commands.

- [ ] **Step 1: Write the bounded acceptance test**

```python
def test_trained_chess_policy_improves_on_fixed_examples(tmp_path) -> None:
    fixture = write_tactical_teacher_fixture(tmp_path, positions=16, seed=7)
    initial, trained = train_tiny_chess_fixture(fixture, epochs=30, seed=7)
    assert trained.mean_teacher_move_probability > initial.mean_teacher_move_probability
    assert trained.illegal_moves == 0
```

- [ ] **Step 2: Run the acceptance test**

Run: `uv run pytest tests/games/test_chess_end_to_end.py -v`

Expected: PASS without network access and without requiring Stockfish at test time.

- [ ] **Step 3: Produce bounded local training evidence**

Run: `uv run flybrain games train --game chess --teacher-positions 5000 --self-play-games 200 --seed 7 --output artifacts/games/chess/seed7-v1`

Run: `uv run flybrain games evaluate --game chess --checkpoint artifacts/games/chess/seed7-v1/best --games-per-opponent 20 --output artifacts/games/chess/seed7-v1/evaluations/final.json`

Run: `SDL_VIDEODRIVER=dummy uv run flybrain games watch --game chess --checkpoint artifacts/games/chess/seed7-v1/best --dry-run --steps 20`

Expected: all moves legal, checkpoint unchanged by evaluation, PGNs parse successfully, and measured W/D/L plus opponent-relative Elo are recorded. If bounded training does not beat the material opponent, report that honestly and retain the strongest verified baseline rather than claiming strong chess.

- [ ] **Step 4: Document commands and evidence boundary**

Add installation, teacher bootstrap, no-Stockfish self-play, resume, evaluation, watch, and human-play commands. State measured strength, exact engine assistance, compute budget, and the distinction between learned policy, PUCT search, and Stockfish teacher.

- [ ] **Step 5: Run complete verification**

Run: `uv run ruff check .`

Run: `uv run mypy src`

Run: `uv run pytest -q`

Expected: all checks PASS and optional Stockfish tests skip only when resolution returns `None`.

- [ ] **Step 6: Commit**

```bash
git add README.md tests/games/test_chess_end_to_end.py artifacts/games/chess/seed7-v1
git commit -m "test: verify chess learning and legal play"
```
