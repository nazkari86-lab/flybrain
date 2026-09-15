# Mushroom-Body Plasticity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Demonstrate persistent dopamine-gated cue-specific memory on real MaleCNS KC→MBON synapses.

**Architecture:** Extract a provenance-rich plastic edge overlay from canonical Parquet, update eligibility and bounded multipliers independently of fixed anatomy, persist state atomically, and run paired real-topology/control benchmarks.

**Tech Stack:** Python 3.12+, NumPy, PyArrow, Pydantic, Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-mb-plasticity-design.md`

### Task 1: Plastic edge rule and persistence

- [x] Write failing locality, no-dopamine, decay, bounds, and save/load tests.
- [x] Implement `PlasticEdgeSet`, `PlasticityParameters`, and atomic state I/O.
- [x] Run focused and full quality gates.
- [x] Commit `feat: add dopamine-gated synaptic plasticity`.

### Task 2: MaleCNS mushroom-body extraction

- [x] Write failing fixture tests for class-based KC/MBON extraction and exact edge accounting.
- [x] Implement streaming `extract_kc_mbon_edges()` with source identity metadata.
- [x] Run focused and full tests.
- [x] Commit `feat: extract MaleCNS mushroom-body plastic edges`.

### Task 3: Real associative-memory benchmark

- [ ] Write a failing deterministic benchmark and CLI test.
- [ ] Implement trained/untrained, no-dopamine, and cleared-eligibility conditions.
- [ ] Run the benchmark on the real MaleCNS snapshot and verify persistence replay.
- [ ] Document measured results and limitations.
- [ ] Commit `feat: validate real-topology associative memory`.

## Global constraints

- Fixed connectome topology is never trained or rewired.
- Plasticity is restricted to measured KC→MBON edges.
- No separate neural policy, classifier, or privileged task-state controller.
- Biological assumptions and fitted parameters are serialized with every result.
