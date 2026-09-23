# Game Learning Platform and Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared game-learning infrastructure and a persistent DQN agent that learns a procedural one-button runner, evaluates on unseen seeds, and can be watched or controlled live.

**Architecture:** A generic game contract and atomic artifact store sit under `flybrain.games`. `RunnerEnv` supplies deterministic Gymnasium dynamics and RGB rendering, while Stable-Baselines3 DQN supplies the maintained learning implementation. A focused Typer sub-application exposes train, evaluate, watch, and play without growing the existing CLI further.

**Tech Stack:** Python 3.12, NumPy 2.x, Pydantic 2.x, Gymnasium 1.3.x, Stable-Baselines3 2.9.x, PyTorch 2.14.x, pygame-ce 2.x, Typer, pytest, Ruff, mypy

**Spec:** `docs/superpowers/specs/2026-09-24-game-learning-platform-design.md`

## Global Constraints

- Keep all engineered game RL separate from `flybrain` biological MaleCNS modules and claims.
- Training, validation, and final holdout seeds must be disjoint and recorded.
- Evaluators must never mutate model weights, replay buffers, optimizer state, or curriculum state.
- New artifacts use atomic publication and refuse accidental overwrite unless `--resume` names a compatible run.
- The runner's initial action space is exactly `0 = wait`, `1 = jump`.
- The first learner is Stable-Baselines3 DQN; switch to PPO only in a separately recorded benchmark if DQN is demonstrably unstable.
- `games` dependencies are optional so existing biological and MuJoCo paths remain independently installable.
- Preserve all unrelated dirty-worktree files; stage and commit only files listed in each task.

## File Map

- `pyproject.toml`: optional game dependencies and mypy overrides for external packages.
- `docs/data/game-learning-reuse-2026-09-24.md`: dated dependency, license, and prior-art decision record.
- `src/flybrain/games/__init__.py`: public game-learning package exports.
- `src/flybrain/games/contracts.py`: generic adapter transition and mode contracts.
- `src/flybrain/games/artifacts.py`: run manifests, disjoint seed schedules, atomic JSON/directory publication.
- `src/flybrain/games/runner_env.py`: deterministic procedural runner physics, Gymnasium API, and RGB renderer.
- `src/flybrain/games/runner_learning.py`: DQN creation, train/resume, checkpoint, immutable evaluation, and result models.
- `src/flybrain/games/runner_viewer.py`: local Pygame human/agent viewer and headless smoke mode.
- `src/flybrain/games/cli.py`: `games train/evaluate/watch/play` Typer commands.
- `src/flybrain/cli.py`: registers `games_app` only.
- `tests/games/`: focused tests for contracts, artifacts, environment, learning, viewer, and CLI.
- `README.md`: verified commands and evidence boundaries.

---

### Task 1: Record Reuse Decisions and Add Optional Dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `docs/data/game-learning-reuse-2026-09-24.md`
- Test: `tests/games/test_dependencies.py`

**Interfaces:**
- Consumes: Python 3.12 and NumPy already declared by the project.
- Produces: installable `games` extra and a machine-readable import smoke test used by every later task.

- [ ] **Step 1: Write the failing dependency test**

```python
from importlib.util import find_spec


def test_game_dependencies_are_importable() -> None:
    for module in ("gymnasium", "stable_baselines3", "torch", "pygame"):
        assert find_spec(module) is not None, module
```

- [ ] **Step 2: Run the test to verify the extra is missing**

Run: `uv run pytest tests/games/test_dependencies.py -v`

Expected: FAIL naming at least `gymnasium` or `stable_baselines3`.

- [ ] **Step 3: Add the optional dependency group**

Add to `pyproject.toml`:

```toml
games = [
  "gymnasium>=1.3,<2",
  "pygame-ce>=2.5,<3",
  "stable-baselines3>=2.9,<3",
  "torch>=2.14,<3",
]
```

Add missing-import overrides only if the installed packages omit typing markers:

```toml
[[tool.mypy.overrides]]
module = ["gymnasium", "gymnasium.*", "pygame", "pygame.*", "stable_baselines3", "stable_baselines3.*"]
ignore_missing_imports = true
```

Write the reuse record with the checked date, package versions, upstream URLs, licenses, Python compatibility, Apple Silicon result, the MIT DashRL comparison, and the exact rejected gaps: one fixed level, no procedural train/holdout split, no atomic full-state resume, and no FlyBrain CLI/artifact contract.

- [ ] **Step 4: Resolve and verify dependencies**

Run: `uv sync --extra dev --extra games`

Run: `uv run pytest tests/games/test_dependencies.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock docs/data/game-learning-reuse-2026-09-24.md tests/games/test_dependencies.py
git commit -m "build: add optional game learning stack"
```

---

### Task 2: Shared Contracts, Seed Isolation, and Atomic Artifacts

**Files:**
- Create: `src/flybrain/games/__init__.py`
- Create: `src/flybrain/games/contracts.py`
- Create: `src/flybrain/games/artifacts.py`
- Create: `tests/games/test_game_artifacts.py`

**Interfaces:**
- Consumes: NumPy arrays and Pydantic models.
- Produces: `GameAdapter[ObservationT, ActionT]`, `StepResult`, `SeedSchedule.build(root_seed, train_count, validation_count, holdout_count)`, `RunManifest`, `write_json_atomic(path, payload)`, and `publish_directory_atomic(target, writer)`.

- [ ] **Step 1: Write failing contract and artifact tests**

```python
from pathlib import Path

import pytest

from flybrain.games.artifacts import SeedSchedule, write_json_atomic


def test_seed_schedule_is_reproducible_and_disjoint() -> None:
    first = SeedSchedule.build(7, train_count=8, validation_count=4, holdout_count=4)
    second = SeedSchedule.build(7, train_count=8, validation_count=4, holdout_count=4)
    assert first == second
    assert not (set(first.training) & set(first.validation))
    assert not (set(first.training) & set(first.holdout))
    assert not (set(first.validation) & set(first.holdout))


def test_atomic_json_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    write_json_atomic(output, {"ok": True})
    with pytest.raises(FileExistsError):
        write_json_atomic(output, {"ok": False})
    assert output.read_text() == '{"ok":true}\n'
```

- [ ] **Step 2: Run tests and verify missing modules**

Run: `uv run pytest tests/games/test_game_artifacts.py -v`

Expected: FAIL with `ModuleNotFoundError: flybrain.games`.

- [ ] **Step 3: Implement exact shared types**

In `contracts.py` define:

```python
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")
GameMode = Literal["training", "validation", "holdout", "play"]


@dataclass(frozen=True)
class StepResult(Generic[ObservationT]):
    observation: ObservationT
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, int | float | str | bool]


class GameAdapter(Protocol[ObservationT, ActionT]):
    def reset(self, *, seed: int, mode: GameMode) -> ObservationT: ...
    def legal_actions(self) -> tuple[ActionT, ...]: ...
    def step(self, action: ActionT) -> StepResult[ObservationT]: ...
```

In `artifacts.py`, use `random.Random(root_seed).sample(range(1, 2**31), total)` for deterministic unique schedules; model them with frozen Pydantic models. `write_json_atomic` serializes with `json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"`, fsyncs a file in a sibling temporary directory, and publishes with `os.link` so an existing destination cannot be replaced.

- [ ] **Step 4: Verify focused quality gates**

Run: `uv run pytest tests/games/test_game_artifacts.py -v`

Run: `uv run ruff check src/flybrain/games tests/games`

Run: `uv run mypy src/flybrain/games`

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flybrain/games tests/games/test_game_artifacts.py
git commit -m "feat: add game learning contracts and artifacts"
```

---

### Task 3: Deterministic Procedural Runner Environment

**Files:**
- Create: `src/flybrain/games/runner_env.py`
- Create: `tests/games/test_runner_env.py`

**Interfaces:**
- Consumes: `GameMode` from `contracts.py`; Gymnasium `Env`; NumPy `Generator`.
- Produces: `RunnerConfig`, `Obstacle`, `RunnerState`, `RunnerEnv(gym.Env[np.ndarray, int])` with observation shape `(9,)`, `Discrete(2)` actions, `render_rgb(width=960, height=540)`, deterministic `level_signature`, and `RunnerGameAdapter` implementing the shared contract by wrapping the Gymnasium tuples.

- [ ] **Step 1: Write failing deterministic physics tests**

```python
import numpy as np

from flybrain.games.runner_env import RunnerConfig, RunnerEnv


def test_same_seed_produces_same_level_and_trace() -> None:
    first = RunnerEnv(RunnerConfig(level_length=12))
    second = RunnerEnv(RunnerConfig(level_length=12))
    first_obs, first_info = first.reset(seed=11, options={"mode": "training"})
    second_obs, second_info = second.reset(seed=11, options={"mode": "training"})
    assert first_info["level_signature"] == second_info["level_signature"]
    np.testing.assert_array_equal(first_obs, second_obs)
    for action in (0, 1, 0, 0, 0, 1):
        left = first.step(action)
        right = second.step(action)
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:] == right[1:]


def test_jump_is_grounded_and_observation_stays_bounded() -> None:
    env = RunnerEnv(RunnerConfig(level_length=4))
    env.reset(seed=3)
    observation, _, _, _, _ = env.step(1)
    assert observation[1] < 0.0
    second, _, _, _, _ = env.step(1)
    assert second[1] > observation[1]
    assert np.all(np.isfinite(second))
    assert env.observation_space.contains(second)
```

- [ ] **Step 2: Run tests and verify the environment is absent**

Run: `uv run pytest tests/games/test_runner_env.py -v`

Expected: FAIL importing `flybrain.games.runner_env`.

- [ ] **Step 3: Implement fixed-timestep state and generation**

Use `dt=1/60`, normalized world units, semi-implicit Euler integration, grounded-only jump impulse, and swept horizontal overlap plus vertical crossing for collision. Generate obstacle kinds from `{"spike", "block", "gap"}` with curriculum parameters derived only from `options["mode"]` and `options["difficulty"]`. Keep the authoritative state in a frozen `RunnerState` replaced on each step.

The nine observation values are:

```python
np.asarray(
    [player_y, player_vy, float(grounded), speed,
     first_distance, first_kind, first_width,
     second_distance, second_kind],
    dtype=np.float32,
)
```

Normalize and clip each element to the declared `Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)`. Rewards are `+delta_progress`, `+5` per passed obstacle, `-25` on collision, `+100` on level completion, and `-0.002` for a jump action. Include `seed`, `level_signature`, `distance`, `obstacles_passed`, `collision`, and `completed` in `info`.

`RunnerGameAdapter.reset(seed, mode)` calls `RunnerEnv.reset(seed=seed, options={"mode": mode})` and returns only the observation; `legal_actions()` returns `(0, 1)`; `step(action)` converts the Gymnasium five-tuple into `StepResult`. This makes the shared contract real without changing the API Stable-Baselines3 requires.

- [ ] **Step 4: Add termination, holdout, and renderer tests**

```python
def test_rgb_render_matches_declared_shape() -> None:
    env = RunnerEnv(RunnerConfig(level_length=4))
    env.reset(seed=5, options={"mode": "holdout"})
    frame = env.render_rgb(width=320, height=180)
    assert frame.shape == (180, 320, 3)
    assert frame.dtype == np.uint8
    assert int(frame.max()) > int(frame.min())
```

Run: `uv run pytest tests/games/test_runner_env.py -v`

Expected: PASS.

- [ ] **Step 5: Run focused static checks and commit**

Run: `uv run ruff check src/flybrain/games/runner_env.py tests/games/test_runner_env.py`

Run: `uv run mypy src/flybrain/games/runner_env.py`

```bash
git add src/flybrain/games/runner_env.py tests/games/test_runner_env.py
git commit -m "feat: add procedural runner environment"
```

---

### Task 4: Persistent DQN Training and Immutable Evaluation

**Files:**
- Create: `src/flybrain/games/runner_learning.py`
- Create: `tests/games/test_runner_learning.py`

**Interfaces:**
- Consumes: `RunnerEnv`, `RunnerConfig`, `SeedSchedule`, atomic artifact helpers, and SB3 `DQN`.
- Produces: `RunnerTrainingConfig`, `RunnerCheckpointManifest`, `RunnerEvaluation`, `train_runner(config, output, resume=None)`, `evaluate_runner(checkpoint, seeds)`, and `load_runner_model(checkpoint, env)`.

- [ ] **Step 1: Write failing train/evaluate tests**

```python
from flybrain.games.runner_learning import (
    RunnerTrainingConfig,
    evaluate_runner,
    train_runner,
)


def test_short_training_writes_resumable_checkpoint(tmp_path) -> None:
    run = train_runner(
        RunnerTrainingConfig(total_steps=256, checkpoint_every=128, seed=7),
        tmp_path / "run",
    )
    assert run.latest.joinpath("manifest.json").is_file()
    assert run.latest.joinpath("model.zip").is_file()
    result = evaluate_runner(run.latest, seeds=(101, 102))
    assert result.episodes == 2
    assert result.model_sha256 == run.model_sha256
    assert result.training_mutated is False
```

- [ ] **Step 2: Run the test and confirm missing learning module**

Run: `uv run pytest tests/games/test_runner_learning.py::test_short_training_writes_resumable_checkpoint -v`

Expected: FAIL importing `runner_learning`.

- [ ] **Step 3: Implement DQN construction and checkpoint schema**

Construct DQN with exact deterministic defaults:

```python
DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=500,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1_000,
    exploration_fraction=0.30,
    exploration_final_eps=0.05,
    policy_kwargs={"net_arch": [128, 128]},
    seed=config.seed,
    device="cpu",
    verbose=0,
)
```

At each checkpoint, write `model.zip`, `replay.pkl`, and `manifest.json` into a sibling staging directory; include config, completed timesteps, seed schedule, algorithm, dependency versions, model SHA-256, and schema version `runner-dqn-v1`; fsync files and atomically rename the directory. Resume loads both model and replay buffer, validates the schema and runner configuration, and continues with `reset_num_timesteps=False`.

- [ ] **Step 4: Implement side-effect-free evaluation and baseline comparison**

Evaluation loads a separate model instance, predicts with `deterministic=True`, and records pre/post hashes of the checkpoint files. It returns completion rate, mean normalized distance, collision count, mean return, action rate, per-seed results, bootstrap confidence bounds generated from a recorded evaluation RNG seed, and `training_mutated=False` only when every hash remains unchanged.

Add `evaluate_random_runner` using `random.Random(seed).randrange(2)` and an untrained DQN baseline with identical architecture. Selection of `best/` uses validation mean normalized distance, with completion rate as the first tie-breaker and fewer collisions as the second.

- [ ] **Step 5: Add deterministic resume and disjoint-seed tests**

```python
def test_resume_increases_steps_without_overwriting_prior_checkpoint(tmp_path) -> None:
    first = train_runner(
        RunnerTrainingConfig(total_steps=128, checkpoint_every=128, seed=13),
        tmp_path / "run",
    )
    old_hash = first.model_sha256
    resumed = train_runner(
        RunnerTrainingConfig(total_steps=256, checkpoint_every=128, seed=13),
        tmp_path / "run",
        resume=first.latest,
    )
    assert resumed.completed_steps >= 256
    assert first.latest.joinpath("model.zip").is_file()
    assert resumed.model_sha256 != old_hash
```

Run: `uv run pytest tests/games/test_runner_learning.py -v`

Expected: PASS.

- [ ] **Step 6: Run static checks and commit**

Run: `uv run ruff check src/flybrain/games/runner_learning.py tests/games/test_runner_learning.py`

Run: `uv run mypy src/flybrain/games/runner_learning.py`

```bash
git add src/flybrain/games/runner_learning.py tests/games/test_runner_learning.py
git commit -m "feat: train and evaluate persistent runner agent"
```

---

### Task 5: Live Runner Viewer and Human Play

**Files:**
- Create: `src/flybrain/games/runner_viewer.py`
- Create: `tests/games/test_runner_viewer.py`

**Interfaces:**
- Consumes: `RunnerEnv` and optional DQN checkpoint.
- Produces: `ViewerConfig`, `run_runner_viewer(config) -> int`, and `run_runner_viewer_smoke(steps, checkpoint=None) -> ViewerSmokeResult`.

- [ ] **Step 1: Write the headless viewer test**

```python
from flybrain.games.runner_viewer import run_runner_viewer_smoke


def test_viewer_smoke_renders_and_steps() -> None:
    result = run_runner_viewer_smoke(steps=20, seed=7)
    assert result.frames == 20
    assert result.frame_shape == (540, 960, 3)
    assert result.actions in {1, 2}
    assert result.finite_observations is True
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `SDL_VIDEODRIVER=dummy uv run pytest tests/games/test_runner_viewer.py -v`

Expected: FAIL importing `runner_viewer`.

- [ ] **Step 3: Implement viewer controls and overlays**

Initialize Pygame only inside the public viewer function. Display `render_rgb()` at 960x540 and draw text for mode, seed, distance, reward, completion count, epsilon/deterministic mode, and speed multiplier. Controls are `Space/Up = jump`, `A = agent/human`, `P = pause`, `R = reset same seed`, `N = next seed`, `[` and `] = speed`, `Q/Escape = quit`. In agent mode load the checkpoint once and call `predict(observation, deterministic=True)`.

The smoke function never opens a window: it advances the environment, calls `render_rgb`, validates frame and observation finiteness, and returns a frozen Pydantic result.

- [ ] **Step 4: Verify viewer tests and commit**

Run: `SDL_VIDEODRIVER=dummy uv run pytest tests/games/test_runner_viewer.py -v`

Run: `uv run ruff check src/flybrain/games/runner_viewer.py tests/games/test_runner_viewer.py`

Run: `uv run mypy src/flybrain/games/runner_viewer.py`

```bash
git add src/flybrain/games/runner_viewer.py tests/games/test_runner_viewer.py
git commit -m "feat: add live runner viewer"
```

---

### Task 6: Unified Games CLI

**Files:**
- Create: `src/flybrain/games/cli.py`
- Modify: `src/flybrain/cli.py`
- Create: `tests/games/test_games_cli.py`

**Interfaces:**
- Consumes: runner train/evaluate/viewer public functions.
- Produces: `games_app` registered as `flybrain games`, with `train`, `evaluate`, `watch`, and `play` commands and `--game runner` dispatch.

- [ ] **Step 1: Write failing CLI smoke tests**

```python
from typer.testing import CliRunner

from flybrain.cli import app

runner = CliRunner()


def test_games_watch_dry_run() -> None:
    result = runner.invoke(app, ["games", "watch", "--game", "runner", "--dry-run", "--steps", "12"])
    assert result.exit_code == 0, result.output
    assert '"frames":12' in result.output.replace(" ", "")


def test_games_reject_unknown_game() -> None:
    result = runner.invoke(app, ["games", "train", "--game", "unknown", "--output", "unused"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run tests and confirm `games` is absent**

Run: `uv run pytest tests/games/test_games_cli.py -v`

Expected: FAIL because no `games` command is registered.

- [ ] **Step 3: Implement focused Typer dispatch**

Define `games_app = typer.Typer(help="Train, evaluate, watch, and play game-learning agents.")`. Use `Literal["runner", "chess"]` only after chess is implemented; for this plan use `Literal["runner"]`. `train` requires `--output`, supports `--steps`, `--seed`, and `--resume`. `evaluate` requires `--checkpoint` and writes an optional atomic `--output`. `watch` and `play` accept `--dry-run` and `--steps` for CI.

Register in the root CLI:

```python
from flybrain.games.cli import games_app

app.add_typer(games_app, name="games")
```

- [ ] **Step 4: Verify CLI commands**

Run: `uv run flybrain games --help`

Run: `uv run flybrain games watch --game runner --dry-run --steps 12`

Run: `uv run pytest tests/games/test_games_cli.py -v`

Expected: commands and tests PASS and JSON identifies `runner-viewer-smoke-v1`.

- [ ] **Step 5: Commit**

```bash
git add src/flybrain/cli.py src/flybrain/games/cli.py tests/games/test_games_cli.py
git commit -m "feat: expose runner learning CLI"
```

---

### Task 7: End-to-End Evidence and Documentation

**Files:**
- Modify: `README.md`
- Create: `tests/games/test_runner_end_to_end.py`
- Generate: `artifacts/games/runner/seed7-v1/`

**Interfaces:**
- Consumes: all runner commands.
- Produces: a reproducible trained checkpoint, immutable holdout evaluation, replay, and documented launch commands.

- [ ] **Step 1: Add the end-to-end acceptance test**

```python
def test_trained_runner_beats_random_on_unseen_seeds(tmp_path) -> None:
    run = train_runner(
        RunnerTrainingConfig(total_steps=4_096, checkpoint_every=2_048, seed=7),
        tmp_path / "run",
    )
    learned = evaluate_runner(run.best, seeds=(7001, 7002, 7003, 7004))
    random = evaluate_random_runner(RunnerConfig(), seeds=(7001, 7002, 7003, 7004))
    assert learned.mean_normalized_distance > random.mean_normalized_distance
    assert learned.training_mutated is False
```

- [ ] **Step 2: Run the acceptance test**

Run: `uv run pytest tests/games/test_runner_end_to_end.py -v`

Expected: PASS. If learning variance causes failure, first increase training steps to `16_384`; do not weaken seed isolation or compare on training seeds.

- [ ] **Step 3: Run the full bounded training and holdout evaluation**

Run: `uv run flybrain games train --game runner --steps 100000 --seed 7 --output artifacts/games/runner/seed7-v1`

Run: `uv run flybrain games evaluate --game runner --checkpoint artifacts/games/runner/seed7-v1/best --output artifacts/games/runner/seed7-v1/evaluations/final-holdout.json`

Run: `uv run flybrain games watch --game runner --checkpoint artifacts/games/runner/seed7-v1/best --dry-run --steps 120`

Expected: checkpoint, evaluation, and dry-run replay succeed; report measured metrics rather than a predeclared success percentage.

- [ ] **Step 4: Document exact commands and claim boundary**

Add README sections for installing `--extra games`, training, resume, evaluate, live watch, human play, artifact locations, and the explicit statement that this is an engineered RL subsystem rather than MaleCNS-generated learning or direct commercial Geometry Dash automation.

- [ ] **Step 5: Run complete verification**

Run: `uv run ruff check .`

Run: `uv run mypy src`

Run: `uv run pytest -q`

Expected: all checks PASS; existing optional skips remain explained.

- [ ] **Step 6: Commit**

```bash
git add README.md tests/games/test_runner_end_to_end.py artifacts/games/runner/seed7-v1
git commit -m "test: verify runner learns on unseen levels"
```
