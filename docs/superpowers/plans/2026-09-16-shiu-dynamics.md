# Shiu Reference Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify event-driven Shiu whole-CNS dynamics on the real MaleCNS snapshot.

**Architecture:** Extend the sparse graph with an outgoing event representation, implement analytic alpha-synapse LIF state and delayed spike queues, expose deterministic Poisson drive, then run a measured full-graph smoke experiment.

**Tech Stack:** Python 3.12+, NumPy, SciPy sparse, PyArrow, Pydantic, Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-shiu-dynamics-design.md`

### Task 1: Outgoing event graph

- [x] Write failing tests for outgoing direction, duplicate accumulation, metadata, and linear storage.
- [x] Implement `EventConnectome.from_sparse()` and `propagate_indices()`.
- [x] Run focused and full tests.
- [x] Commit `feat: add sparse event connectome`.

### Task 2: Analytic Shiu dynamics

- [x] Write failing analytic decay, exact-delay, refractory, silencing, and replay tests.
- [x] Implement `ShiuParameters`, `ShiuState`, `poisson_voltage_events()`, and `simulate_shiu()`.
- [x] Run focused and full quality gates.
- [x] Commit `feat: implement Shiu reference dynamics`.

### Task 3: Whole-CNS smoke experiment

- [x] Write a failing CLI fixture test for `experiment shiu-smoke`.
- [x] Implement metrics and CLI without privileged game state.
- [x] Run the full MaleCNS 10 ms sensory experiment under `/usr/bin/time -l`.
- [x] Validate deterministic replay and document measured output/resources.
- [x] Commit `feat: run whole-CNS Shiu smoke experiment`.

## Global constraints

- Preserve source parameter values exactly in the default profile.
- Never scan all edges per simulation step.
- Never assign nonzero fast signs to unresolved canonical edges.
- Full-run peak RSS must remain below 8 GiB.
- Report negative or absent propagation honestly; do not tune parameters to force a dramatic result.
