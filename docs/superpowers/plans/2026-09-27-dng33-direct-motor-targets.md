# DNg33 Direct Motor-Target Readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure all 708 annotated `vnc_motor` cells without changing neural or body dynamics, then test the eight preselected direct DNg33 targets under a prospective typed-DN lesion.

**Architecture:** Add an optional superclass-indexed spike-count readout to the existing embodied episode. It observes emitted neuron IDs, serializes zero-inclusive per-cell counts, and is replayed exactly; it never enters motor decoding or learning. Use the existing typed-DN lesion and unchanged MaleCNS–FlyGym interface for the 16–18 seed panel.

**Tech Stack:** Python 3.12, Pydantic, Typer, NumPy/SciPy, pytest, MaleCNS snapshot, FlyGym.

**Spec:** `docs/superpowers/specs/2026-09-27-dng33-direct-motor-targets-design.md`

## Global Constraints

- No decoder remapping, gain adjustment, reward/action injection, planner, RL policy, or graph mutation.
- `autonomous_behavior_claim_allowed` stays false.
- The eight direct target IDs and all thresholds/seeds are frozen in the spec before outcome runs.
- A requested `vnc_motor` readout with no such graph cells fails before body stepping.
- Preserve the unrelated untracked `docs/data/runner-terminal-curriculum-2026-09-26 (1).md`.

---

### Task 1: Readout-only episode interface

**Files:**
- Modify: `tests/test_autonomous_hexapod_episode.py`
- Modify: `src/flybrain/autonomous_hexapod_episode.py`

**Interfaces:**
- Consumes: `EventConnectome.superclasses`, emitted `simulate_shiu` neuron IDs.
- Produces: `run_autonomous_hexapod_episode(..., spike_readout_superclass="none" | "vnc_motor")`, result fields `spike_readout_superclass` and `spike_readout_counts: dict[int, int]`.

- [ ] Write a synthetic-chain test with one extra unregistered `vnc_motor` cell 42 reached from DNg33. Assert intact count for 42 is positive, DNg33 lesion reduces it to zero, DNg48 lesion retains it, and registered motor behavior remains as before. Assert the readout includes zero counts and replays exactly.
- [ ] Write a fail-closed test requesting `vnc_motor` from a fixture graph with no such superclass.
- [ ] Run `.venv/bin/pytest -q tests/test_autonomous_hexapod_episode.py -k spike_readout`; verify red due to missing argument/field.
- [ ] Implement exact superclass resolution, deterministic sorted ID keys, zero-inclusive counts, and replay propagation without changing the Shiu mask or decoder.
- [ ] Rerun focused synthetic tests and existing typed/all-DN lesions.

### Task 2: Retained assay and CLI

**Files:**
- Modify: `tests/test_autonomous_hexapod_cli_real.py`
- Modify: `src/flybrain/autonomous_hexapod_assay.py`
- Modify: `src/flybrain/cli.py`

**Interfaces:**
- Consumes: Task 1 readout option.
- Produces: `--spike-readout-superclass vnc_motor` in the retained CLI and an output dictionary for all 708 graph `vnc_motor` cells.

- [ ] Add a real-data CLI regression using two reference body steps: assert 708 count keys, inclusion of all eight preregistered target IDs, exact replay, unchanged graph, and unchanged no-autonomy gate.
- [ ] Run `FLYBRAIN_MALECNS_SNAPSHOT=artifacts/male-cns-v1.0-w5 .venv/bin/pytest -q tests/test_autonomous_hexapod_cli_real.py -k spike_readout`; verify red due to absent option.
- [ ] Thread the option through CLI and retained assay; run focused tests, full `.venv/bin/pytest -q`, Ruff, mypy, and `git diff --check`.
- [ ] Commit code, tests, and this plan; push the source revision before running seeds 16–18.

### Task 3: Prospective target-cell assay

**Files:**
- Create: `docs/data/dng33-direct-motor-targets-2026-09-27.md`
- Add: compressed `artifacts/dng33-direct-targets-flygym-seed{16,17,18}-*.json.gz`.

**Interfaces:**
- Consumes: committed Task 2 CLI, eight fixed direct target IDs, `vnc_motor` readout.
- Produces: per-seed direct-target and non-target spike totals, exact primary-gate verdict, checksums, and explicitly limited interpretation.

- [ ] For each seed 16–18, run 100-step FlyGym normal DNg33, DNg33 lesion, and DNg48 lesion with `--spike-readout-superclass vnc_motor --motor-trace`; do not inspect outcomes until all nine finish.
- [ ] Apply all four predeclared primary criteria without changing thresholds or targets. Report null/inconclusive if any fails.
- [ ] Audit replay, graph, revisions, readout cardinality, zero-inclusive counts, body/motor diagnostics, food/threat contacts, DAN events, and compression/hash integrity.
- [ ] Publish the report and all compressed artifacts, preserving any negative result.
