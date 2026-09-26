# DNg33 Direct-Edge Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the 16 retained DNg33→motor-target edges are necessary for target recruitment under fixed neural input.

**Architecture:** A read-only retained-graph assay resolves frozen biological IDs to sparse CSR positions, applies zero multipliers through the existing Shiu edge overlay, and compares paired source-identical conditions. A standalone module CLI writes one provenance-rich JSON. No embodied episode or canonical graph is modified.

**Tech Stack:** Python 3.12, NumPy, SciPy CSR, Pydantic, pytest, Ruff, mypy, retained MaleCNS snapshot.

**Spec:** `docs/superpowers/specs/2026-09-27-dng33-direct-edge-localization-design.md`

## Global Constraints

- Fixed source IDs 13317/13442, eight target IDs, eight control IDs, 16 edges per intervention, and sums +404/+379.
- Seeds 19/20/21, 2,000 steps, Shiu `dt_ms=0.1`, `refractory_ms=2.0`, `synaptic_delay_ms=1.0`; no retuning after seeing outcomes.
- No graph mutation, FlyGym, learning, decoder, reward, planner, or RL. `autonomous_behavior_claim_allowed=false`.
- Preserve unrelated untracked `docs/data/runner-terminal-curriculum-2026-09-26 (1).md`.

---

### Task 1: Frozen-edge resolver and fixed-source assay

**Files:**
- Create: `src/flybrain/dng33_edge_probe.py`
- Create: `tests/test_dng33_edge_probe.py`

**Interfaces:**
- Consumes: `EventConnectome`, `ShiuParameters`, `poisson_voltage_events`, `simulate_shiu` sparse overlay.
- Produces: `resolve_frozen_edges(graph, source_ids, post_ids) -> tuple[np.ndarray, ...]`, `run_dng33_edge_probe(graph, *, steps, seeds, source_ids, target_ids, control_ids) -> dict`.

- [x] Write synthetic tests first: exactly two present positive source→target and source→control edges per postsynaptic cell; missing/negative edges fail closed; nonmatching biological IDs prove ID→index translation; direct-edge zero changes target response while equal source events and pristine CSR are retained. Use a small fixture and explicit expected spike counts, not mocked Shiu calls.
- [x] Run `.venv/bin/pytest -q tests/test_dng33_edge_probe.py` and confirm failure because production module/API is absent.
- [x] Implement frozen-edge resolution from `outgoing.indptr/indices/data`, sorted overlay positions, duplicate/positive checks, graph digest, and a condition runner that records source-event and source-spike-time digests, all target counts, other-motor sum, trace digest, replay, and graph unchanged. Use `np.zeros(16, dtype=np.float32)` only as edge multipliers; never write `outgoing.data`.
- [x] Rerun focused tests and real-snapshot 10-step smoke (seed 0, not any prospective seed), then run full `.venv/bin/pytest -q`, Ruff, mypy, and `git diff --check`.

### Task 2: Publication-ready CLI and source lock

**Files:**
- Modify: `src/flybrain/dng33_edge_probe.py`
- Modify: `tests/test_dng33_edge_probe.py`

**Interfaces:**
- Consumes: Task 1 assay and `artifacts/male-cns-v1.0-w5`.
- Produces: `python -m flybrain.dng33_edge_probe SNAPSHOT --output FILE --steps 2000 --seeds 19 20 21`.

- [x] Write a failing subprocess CLI test on a tiny snapshot: output contains exact condition names, digests, resolved edges, source revision, snapshot digest, and `autonomous_behavior_claim_allowed=false`; reject missing output path/snapshot or nonpositive steps.
- [x] Implement CLI with `argparse`, JSON output, snapshot digest and clean source revision provenance. Do not inspect seed 19–21 outcomes during this task.
- [ ] Run focused/full tests, Ruff, mypy, `git diff --check`; commit code/tests/plan and push source revision **before** prospective seeds.

### Task 3: Prospective edge intervention and report

**Files:**
- Create: `artifacts/dng33-direct-edge-localization-seeds19-21.json.gz`
- Create: `docs/data/dng33-direct-edge-localization-2026-09-27.md`
- Modify: this plan's checkboxes.

**Interfaces:**
- Consumes: committed Task 2 CLI and locked spec.
- Produces: per-seed full readout and a primary-gate verdict, positive or null.

- [ ] Run all five conditions for each of seeds 19–21, then inspect outcomes. Do not alter thresholds, sources, targets, controls, or horizon.
- [ ] Check each predeclared gate independently, replay, source timing, graph digest, revision/snapshot, artifact gzip integrity and SHA-256; report every failed gate as null/inconclusive.
- [ ] Publish compressed JSON and interpretation on the public branch, and verify remote revision contains both.
