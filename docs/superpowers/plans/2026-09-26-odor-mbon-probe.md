# Odor-to-MBON Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user has directed work to stay in the active session without subagents.

**Goal:** Turn the isolated odor-to-MBON bottleneck measurement into a replay-checked, publicly reproducible artifact without changing the autonomous controller.

**Architecture:** A focused assay module consumes the existing graph, plastic binding, receptor map and Shiu simulator. The retained-snapshot wrapper resolves exact data provenance; a CLI writes the result atomically. All interventions are labeled diagnostics, and the behavioral claim is always false.

**Tech Stack:** Python 3.12, NumPy, SciPy CSR, Pydantic, Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-odor-mbon-probe-design.md`

## Global Constraints

- Do not mutate the MaleCNS graph, main learning rule, motor decoder or embodied benchmark.
- Reuse measured-side ORN_DM1/ORN_VA2 and ORN_DA2 channels and the current Shiu parameters.
- Compare the same threat event train under baseline and artificial 2.0 KC→519128 multipliers.
- Keep `autonomous_behavior_claim_allowed=false`; no body behavior is inferred.
- Do not download data while disk free space is below 10 GiB.

---

### Task 1: Deterministic isolated-neural condition

**Files:**
- Create: `src/flybrain/odor_mbon_probe.py`
- Create: `tests/test_odor_mbon_probe.py`

**Interfaces:**
- Consumes: `EventConnectome`, `PlasticEdgeBinding`, `ShiuParameters`, source and target neuron ID tuples.
- Produces: `run_odor_mbon_condition(graph, binding, *, name, source_ids, target_ids, steps, seed, parameters, max2_target_id=None) -> OdorMbonCondition`.

- [ ] **Step 1: Write a failing signed-graph test.** Use a four-neuron feed-forward fixture with IDs `(1, 2, 3, 4)`, source 1, KC 2, MBONs 3 and 4, edges `1→2`, `2→3`, `2→4`, and a binding only for `2→3`. Assert that no-source has zero source/target spikes; an active source emits events; baseline and max2 share source events; max2 changes only target 3's effective weighted input, not the graph or target 4's input. Add a separate invalid-target test.

```python
baseline = run_odor_mbon_condition(graph, binding, name="threat", source_ids=(1,),
    target_ids=(3, 4), steps=200, seed=7, parameters=params)
boosted = run_odor_mbon_condition(graph, binding, name="threat-max2", source_ids=(1,),
    target_ids=(3, 4), steps=200, seed=7, parameters=params, max2_target_id=3)
assert baseline.source_voltage_events == boosted.source_voltage_events
assert boosted.effective_positive_weighted_spikes[3] > baseline.effective_positive_weighted_spikes[3]
assert boosted.effective_positive_weighted_spikes[4] == baseline.effective_positive_weighted_spikes[4]
```

- [ ] **Step 2: Verify red.** Run `.venv/bin/pytest tests/test_odor_mbon_probe.py -q`; expect failure because `flybrain.odor_mbon_probe` is absent.
- [ ] **Step 3: Implement the minimum condition runner.** Validate IDs and steps; generate source events with `poisson_voltage_events`; use fresh `ShiuState`; copy overlay multipliers and set only declared target edges to 2.0 when requested; run `simulate_shiu`; count source, declared KC and target MBON spikes; sample post-step target voltage; sum active signed presynaptic edge weights, applying the diagnostic multiplier only to the target edge. Return a frozen Pydantic result with explicit `artificial_intervention`.

```python
events = poisson_voltage_events(source_indices, steps=steps, params=parameters, seed=seed)
edge_multipliers = binding.overlay.multipliers.copy()
if max2_target_id is not None:
    edge_multipliers[binding.post_ids == max2_target_id] = 2.0
state = ShiuState.initial(graph.neuron_count, params=parameters, seed=seed)
for batch in simulate_shiu(graph, parameters, steps=steps,
    external_voltage_events=events, state=state,
    plastic_edge_indices=binding.overlay.edge_indices,
    plastic_edge_multipliers=edge_multipliers):
    spike_counts.update(int(value) for value in batch.neuron_ids)
```
- [ ] **Step 4: Verify green and refactor.** Run the same focused pytest command, then `.venv/bin/ruff check src/flybrain/odor_mbon_probe.py tests/test_odor_mbon_probe.py` and `.venv/bin/mypy src/flybrain/odor_mbon_probe.py`.

### Task 2: Retained wrapper, replay, CLI and published artifact

**Files:**
- Modify: `src/flybrain/odor_mbon_probe.py`
- Modify: `src/flybrain/cli.py`
- Modify: `tests/test_odor_mbon_probe.py`
- Modify: `docs/data/odor-to-mbon-bottleneck-2026-09-26.md`
- Create: `artifacts/odor-mbon-probe-seeds7-8-9-steps2000.json.gz`

**Interfaces:**
- Consumes: `run_odor_mbon_condition`, retained snapshot, learning registry and receptor map.
- Produces: `run_retained_odor_mbon_probe(snapshot, *, steps=2000, seeds=(7,8,9)) -> RetainedOdorMbonProbe`; CLI `flybrain experiment odor-mbon SNAPSHOT --output PATH [--steps N] [--seeds CSV]`.

- [ ] **Step 1: Write a failing retained-snapshot test.** Skip only when `FLYBRAIN_MALECNS_SNAPSHOT` is unset; with `steps=20, seeds=(7,)` assert exact 148/41 measured-side food/threat IDs, `519128` DAN route counts `18/0`, five named conditions, exact replay, unchanged graph, and false behavioral claim. Add a CLI existing-output refusal test using a temporary file.
- [ ] **Step 2: Verify red.** Run `FLYBRAIN_MALECNS_SNAPSHOT=artifacts/male-cns-v1.0-w5 .venv/bin/pytest tests/test_odor_mbon_probe.py -q`; expect missing retained runner/CLI failures.
- [ ] **Step 3: Implement wrapper and CLI.** Resolve registry and manifests, bind KC edges, resolve receptor sides and DAN valence routes, run `none/food/threat/both/threat-max2` for each seed, rerun one baseline threat for exact replay, compare graph digest, include snapshot/registry SHA-256 and source revision, and atomically link staged JSON to a non-existing output path. Refuse input/output overlap.

```python
assignment = task_odor_assignment()
food_ids = tuple(sorted(receptors.side_channel_ids(assignment.food_cell_types, "L")
    + receptors.side_channel_ids(assignment.food_cell_types, "R")))
threat_ids = tuple(sorted(receptors.side_channel_ids(assignment.threat_cell_types, "L")
    + receptors.side_channel_ids(assignment.threat_cell_types, "R")))
for seed in seeds:
    for name, source_ids, boost in (
        ("none", (), None), ("food", food_ids, None),
        ("threat", threat_ids, None), ("both", food_ids + threat_ids, None),
        ("threat_max2", threat_ids, 519128),
    ):
        conditions.append(run_odor_mbon_condition(graph, binding, name=name,
            source_ids=source_ids, target_ids=(519128, 524893), steps=steps,
            seed=seed, parameters=parameters, max2_target_id=boost))
```
- [ ] **Step 4: Verify focused tests and static checks.** Run `.venv/bin/pytest tests/test_odor_mbon_probe.py -q`, `.venv/bin/ruff check .`, `.venv/bin/mypy src/flybrain` and `git diff --check`.
- [ ] **Step 5: Run full suite and retained assay.** Run `.venv/bin/pytest -q`, then the 2,000-step CLI assay for seeds `7,8,9` into an ignored local JSON. Confirm output fields, replay, graph identity and false behavioral claim; gzip-copy the JSON, test the gzip and decompressed hash, and add only the compressed artifact and relevant docs to Git.
- [ ] **Step 6: Publish and verify.** Commit exact reviewed files, push `feature/biological-steering`, check remote HEAD, public visibility, and unauthenticated HTTP 200 for the artifact. Leave the persistent autonomy goal active.
