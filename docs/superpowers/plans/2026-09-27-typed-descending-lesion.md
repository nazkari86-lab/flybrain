# Typed Descending Lesion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the DNg33 pair has a reproducible motor-spike effect distinct from the anatomically size/transmitter/outdegree-matched DNg48 pair in the retained embodied model.

**Architecture:** Resolve an optional exact DN `cell_type` against the graph once, pass its index set as a readout and optionally as the Shiu silence mask, and record the resulting spike count and provenance. The existing `none` and `all_annotated` paths remain unchanged; no motor or learning code changes.

**Tech Stack:** Python 3.12, Pydantic, Typer, NumPy/SciPy, pytest, MaleCNS snapshot, FlyGym.

**Spec:** `docs/superpowers/specs/2026-09-27-typed-descending-lesion-design.md`

## Global Constraints

- No hidden RL/planner, decoder replacement, gain adjustment, target-bearing action, or graph mutation.
- `autonomous_behavior_claim_allowed` remains false.
- `descending_type` must resolve only exact `descending_neuron` annotations; missing or incompatible selections fail closed.
- Keep the default episode and CLI behavior unchanged.
- Preserve the unrelated untracked `docs/data/runner-terminal-curriculum-2026-09-26 (1).md`.
- Do not inspect outcome on seeds 13–15 until code and thresholds are committed.

---

### Task 1: Episode-level type selection and efficacy readout

**Files:**
- Modify: `tests/test_autonomous_hexapod_episode.py`
- Modify: `src/flybrain/autonomous_hexapod_episode.py`

**Interfaces:**
- Consumes: `EventConnectome.cell_types`, `EventConnectome.superclasses`, existing `simulate_shiu(..., silenced=...)` path.
- Produces: `run_autonomous_hexapod_episode(..., descending_lesion="none" | "all_annotated" | "annotated_type", descending_type: str | None = None)` and result fields `descending_type`, `target_type_neurons`, `target_type_spikes`.

- [x] Write a synthetic-chain test with one spiking `DNg33` DN on the motor path and one spiking `DNg48` DN off that path. A DNg33 readout-only baseline must have `target_type_spikes > 0` and motor spikes; DNg33 lesion must have zero target spikes and zero motor spikes; DNg48 lesion must preserve DNg33/motor activity. Both lesion runs must replay exactly and preserve the graph.
- [x] Add fail-closed tests: `annotated_type` without a type, a non-DN type, a missing type, and `all_annotated` combined with a type must raise `ValueError` before stepping.
- [x] Run `.venv/bin/pytest -q tests/test_autonomous_hexapod_episode.py -k descending_type`; confirm the first failure is the missing argument or mode, not a broken fixture.
- [x] Implement exact type resolution, validation, readout IDs, same mask in replay, and result fields. Count target spikes from emitted neuron IDs; do not infer them from a selected motor or decoder channel.
- [x] Rerun the focused tests and existing DN/motor-lesion tests.

### Task 2: Retained assay and CLI provenance

**Files:**
- Modify: `tests/test_autonomous_hexapod_cli_real.py`
- Modify: `src/flybrain/autonomous_hexapod_assay.py`
- Modify: `src/flybrain/cli.py`

**Interfaces:**
- Consumes: Task 1's `descending_type` and `annotated_type` mode.
- Produces: `--descending-type DNg33` with `--descending-lesion none|annotated_type`; serialized type, cell count, target spikes, intervention, replay and graph checks.

- [x] Add retained CLI tests using the configured MaleCNS snapshot: readout-only `DNg33` resolves exactly two cells; `annotated_type` silences two and reports zero target spikes; `DNg48` resolves two cells; invalid `vnc_motor` or unknown type fails without writing an output.
- [x] Run `FLYBRAIN_MALECNS_SNAPSHOT=artifacts/male-cns-v1.0-w5 .venv/bin/pytest -q tests/test_autonomous_hexapod_cli_real.py -k descending_type`; confirm red for missing CLI option.
- [x] Thread the option through CLI and retained assay with no change to default behavior; rerun focused tests and the full suite.
- [x] Run `.venv/bin/ruff check .`, `.venv/bin/mypy src/flybrain`, `git diff --check`, and commit code, tests, and this plan before any seed-13–15 outcome run.

### Task 3: Locked comparison and honest report

**Files:**
- Create: `docs/data/typed-descending-lesion-2026-09-27.md`
- Add: compressed JSON artifacts under `artifacts/typed-descending-flygym-seed{13,14,15}-*.json.gz`.

**Interfaces:**
- Consumes: committed Task 2 CLI, snapshot/registries, the spec's locked primary gate.
- Produces: exact per-seed motor spikes, target-type spikes, applied-moment integral, contact/stability diagnostics, hashes, and a positive/negative/inconclusive conclusion.

- [ ] For each seed 13, 14, 15 run four 100-step FlyGym conditions: DNg33 readout-only baseline, DNg33 lesion, DNg48 readout-only baseline, DNg48 lesion. Use `--motor-trace` and unique output files.
- [ ] Check that the two readout-only baselines have identical motor/body traces for each seed; otherwise fail the comparison.
- [ ] Evaluate the exact primary gate in the spec, with no threshold or parameter changes. Record all failures as prominently as successes.
- [ ] Audit all JSON, gzip integrity, checksums, software/snapshot/registry revisions, replay, graph state, contacts, body motion, and phase amplitude; publish the report and compressed artifacts only after verification.
