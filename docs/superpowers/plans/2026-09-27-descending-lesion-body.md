# Annotated DN Lesion Body Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose an evidence-bound all-annotated-DN lesion in retained embodied episodes and measure its motor/body effect.

**Architecture:** Extend the existing Shiu silence mask in `autonomous_hexapod_episode.py`, then pass one explicit mode through the retained assay and CLI. Preserve the default path and serialize the intervention for matched comparisons.

**Tech Stack:** Python 3, Pydantic, Typer, NumPy/SciPy, pytest, MaleCNS snapshot, FlyGym.

**Spec:** `docs/superpowers/specs/2026-09-27-descending-lesion-body-design.md`

## Global Constraints

- No hidden RL/planner, decoder replacement, gain adjustment, or graph mutation.
- Default `none` path must retain prior episode behavior.
- An empty `descending_neuron` selection must fail closed.
- Keep `autonomous_behavior_claim_allowed=False` regardless of lesion effects.
- Preserve the unrelated untracked report copy in `docs/data/`.

---

### Task 1: Episode-level DN lesion

**Files:**
- Modify: `tests/test_autonomous_hexapod_episode.py`
- Modify: `src/flybrain/autonomous_hexapod_episode.py`

**Interfaces:**
- Consumes: `EventConnectome.superclasses`, `simulate_shiu(..., silenced=...)`.
- Produces: `run_autonomous_hexapod_episode(..., descending_lesion="none" | "all_annotated")` and result fields `descending_lesion`, `silenced_descending_neurons`.

- [x] Add `test_all_annotated_descending_lesion_blocks_motor_in_synthetic_chain` using an ORN→KC→MBON→DN→motor graph. Its decisive assertions are:

```python
assert baseline.motor_spikes > 0
assert lesioned.motor_spikes == 0
assert lesioned.silenced_descending_neurons == 1
assert lesioned.replay_exact and lesioned.graph_unchanged
```

- [x] Run `.venv/bin/pytest -q tests/test_autonomous_hexapod_episode.py -k descending_lesion` and confirm it fails because the argument is unsupported.
- [x] Add the mask selection, empty-selection error, result fields, and replay propagation without changing the default path. The only new silence source is:

```python
if descending_lesion == "all_annotated":
    indices = [i for i, kind in enumerate(graph.superclasses) if kind == "descending_neuron"]
    if not indices:
        raise ValueError("no annotated descending neurons")
    silenced[indices] = True
```

- [x] Rerun the focused test and existing motor-lesion tests until green.

### Task 2: Retained launcher, CLI, and real-data comparison

**Files:**
- Modify: `tests/test_autonomous_hexapod_cli_real.py`
- Modify: `src/flybrain/autonomous_hexapod_assay.py`
- Modify: `src/flybrain/cli.py`
- Create: `docs/data/embodied-descending-lesion-2026-09-27.md`

**Interfaces:**
- Consumes: Task 1 `descending_lesion` argument and result fields.
- Produces: `flybrain experiment autonomous-hexapod ... --descending-lesion all_annotated`.

- [x] Add a retained CLI test asserting the output carries `all_annotated`, a positive cell count, exact replay, and zero annotated-DN spikes. Unlike the synthetic chain, the real embodied loop retains residual motor spikes through parallel pathways:

```python
assert episode["descending_lesion"] == "all_annotated"
assert episode["silenced_descending_neurons"] == 1314
assert episode["annotated_descending_spikes"] == 0
assert episode["motor_spikes"] > 0
assert episode["replay_exact"] is True
```

- [x] Run the focused real-data test with `FLYBRAIN_MALECNS_SNAPSHOT` and confirm it fails because the CLI option is absent.
- [x] Thread the option through the retained assay and CLI using `Literal["none", "all_annotated"]`, then rerun the test.
- [x] Run paired FlyGym baseline/all-DN and passive all-motor controls with seed 7 and 100 body steps; inspect joint moments, thorax displacement, contacts, replay, and stability. Repeat the paired active conditions at seeds 8 and 9.
- [ ] Record all conditions, source revision, caveats, and exact artifact paths in the data report; run the full suite and static checks.
