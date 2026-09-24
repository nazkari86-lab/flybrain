# Practical Autonomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and execute inline because this is a bounded one-hour sprint.

**Goal:** Add a reliable train/evaluate/run command for the practical hybrid autonomous fly.

**Architecture:** Reuse `ArenaWorld`; add one focused hybrid controller module and one CLI entry point. Reflexes and potential fields provide immediate behavior while tabular Q-learning supplies online adaptation.

**Tech Stack:** Python 3.12, NumPy, Pydantic, Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-practical-autonomy-design.md`

## Global Constraints

- Preserve existing dirty-worktree changes.
- No external dependency is required for the fast path.
- Never label the result proof of full biological intelligence.

### Task 1: Hybrid controller and benchmark

**Files:** Create `src/flybrain/practical_autonomy.py`; create `tests/test_practical_autonomy.py`.

- [ ] Write a failing end-to-end behavior test.
- [ ] Run it and confirm failure because the module is absent.
- [ ] Implement sensory discretization, reflexes, potential-field action priors, Q updates, randomized training and held-out evaluation.
- [ ] Run the test and tune only declared controller parameters.
- [ ] Add deterministic replay and validation tests.

### Task 2: Atomic CLI artifact

**Files:** Modify `src/flybrain/cli.py`; create `tests/test_practical_autonomy_cli.py`.

- [ ] Write the failing CLI test.
- [ ] Add `experiment practical-autonomy` with atomic no-overwrite output.
- [ ] Verify JSON output and collision rejection.

### Task 3: Verification and runnable artifact

- [ ] Reuse FlyGym's official 42-DOF hybrid turning controller and six adhesion channels.
- [ ] Add a real MuJoCo movement/stability test and include physical metrics in the artifact.
- [ ] Run focused tests, Ruff and mypy.
- [ ] Run the command with defaults and inspect the generated metrics.
- [ ] Run the broader non-real test suite and report exact results.

### Task 4: Interactive MuJoCo application

- [ ] Add a separately testable keyboard command state machine.
- [ ] Add a headless smoke path using the official 42-DOF controller.
- [ ] Open a passive MuJoCo viewer with command/status overlay and real-time stepping.
- [ ] Relaunch through `mjpython` automatically on macOS.
- [ ] Document keyboard controls and the one-command launcher.
