# Shiu Plastic-Memory Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the persisted MaleCNS KC→MBON memory alter live Shiu whole-CNS dynamics under a deterministic paired-cue protocol.

**Architecture:** Centralize immutable snapshot identity, copy and patch only matched KC→MBON values in an outgoing sparse CSR, then run baseline/learned conditions with identical cue events and initial state. Measure actual spike-driven target input and complete target trajectories while preserving the canonical snapshot and baseline graph.

**Tech Stack:** Python 3.12+, NumPy, SciPy sparse arrays, PyArrow, Pydantic 2, Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-shiu-plastic-integration-design.md`

## Global Constraints

- The canonical Parquet snapshot, topology, indices, signs, and baseline event graph are immutable.
- Only measured KC→MBON edges from the identity-bound state may receive multipliers.
- Baseline and learned conditions use identical Shiu parameters, events, seed, and initial state.
- The published `synapse_mv`, delay, refractory period, and time constants are not tuned.
- No dense neuron-by-neuron matrix, policy network, classifier, or privileged task-state controller.
- A result is published only after identity, determinism, specificity, and acceptance checks pass.
- Discrete spike changes are reported as measured results and are never an acceptance target.

---

### Task 1: Shared immutable snapshot identity

**Files:**
- Create: `src/flybrain/provenance.py`
- Modify: `src/flybrain/mushroom_body.py`
- Create: `tests/test_provenance.py`
- Modify: `tests/test_mushroom_body.py`

**Interfaces:**
- Produces: `snapshot_sha256(snapshot: Path) -> str`
- Produces: `validate_state_identity(snapshot: Path, identity: PlasticStateIdentity) -> None`
- Preserves: the exact existing digest algorithm and existing MaleCNS association state identity.

- [ ] **Step 1: Write failing shared-identity tests**

Add tests that construct a minimal snapshot with `metadata.json`, `source-annotations.parquet`, and
`edges.parquet`, then assert:

```python
first = snapshot_sha256(snapshot)
assert first == snapshot_sha256(snapshot)
(snapshot / "metadata.json").write_text('{"changed": true}')
assert snapshot_sha256(snapshot) != first
```

Add an identity-validation test with literal 64-character digests:

```python
identity = PlasticStateIdentity(
    dataset_id="fixture",
    source_manifest_sha256="a" * 64,
    snapshot_sha256=snapshot_sha256(snapshot),
)
validate_state_identity(snapshot, identity)
with pytest.raises(ValueError, match="snapshot identity"):
    validate_state_identity(snapshot, replace(identity, snapshot_sha256="b" * 64))
```

The fixture metadata must contain `dataset_id="fixture"` and `manifest_sha256="a" * 64`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/test_provenance.py tests/test_mushroom_body.py
```

Expected: collection fails because `flybrain.provenance` does not exist.

- [ ] **Step 3: Implement the shared provenance module**

Move the private digest implementation from `mushroom_body.py` without changing byte order:

```python
SNAPSHOT_IDENTITY_FILES = (
    "metadata.json",
    "source-annotations.parquet",
    "edges.parquet",
)

def snapshot_sha256(snapshot: Path) -> str:
    digest = hashlib.sha256()
    for name in SNAPSHOT_IDENTITY_FILES:
        digest.update(name.encode())
        digest.update(b"\0")
        with (snapshot / name).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()
```

`validate_state_identity()` must parse metadata using a strict small Pydantic model, compare
dataset, source-manifest, and computed snapshot digests, and raise one `ValueError` that names the
mismatched fields. Import `snapshot_sha256()` into `mushroom_body.py` from the new provenance
module and delete the old private digest function.

- [ ] **Step 4: Verify the old real state identity remains valid**

Run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from flybrain.plasticity import load_plastic_state
from flybrain.provenance import validate_state_identity
p = Path('/Users/dulatnurlanuly/Downloads/flybrain/artifacts/male-cns-v1.0-w5')
_, _, identity = load_plastic_state(Path('/Users/dulatnurlanuly/Downloads/flybrain/artifacts/mb-association-seed7-state.npz'))
validate_state_identity(p, identity)
print(identity.snapshot_sha256)
PY
```

Expected digest: `900eee63ac8d1e1805479a993735d41c20bdefb96e9c52d968a9db0b34709231`.

- [ ] **Step 5: Run focused and full gates**

Run:

```bash
.venv/bin/pytest -q tests/test_provenance.py tests/test_mushroom_body.py tests/test_mb_association.py
.venv/bin/ruff check .
.venv/bin/mypy src
.venv/bin/pytest -q
```

Expected: all commands exit 0; the optional real-data test may skip without its environment variable.

- [ ] **Step 6: Commit**

```bash
git add src/flybrain/provenance.py src/flybrain/mushroom_body.py tests/test_provenance.py tests/test_mushroom_body.py
git commit -m "refactor: centralize snapshot identity validation"
```

---

### Task 2: Sparse plastic event-graph materialization

**Files:**
- Create: `src/flybrain/plastic_graph.py`
- Create: `tests/test_plastic_graph.py`

**Interfaces:**
- Consumes: `EventConnectome`, `PlasticEdgeSet`
- Produces: `PlasticGraphMetrics(BaseModel, frozen=True)`
- Produces: `materialize_plastic_event_graph(graph: EventConnectome, edges: PlasticEdgeSet) -> tuple[EventConnectome, PlasticGraphMetrics]`

- [ ] **Step 1: Write failing locality and immutability tests**

Build a four-neuron outgoing graph ordered as IDs `[1, 2, 10, 11]`, with signed edges `1→10=5`,
`2→10=7`, and `10→11=-3`.
Create a plastic set for only the first two edges with multipliers `[0.5, 1.0]`. Assert:

```python
baseline_data = graph.outgoing.data.copy()
learned, metrics = materialize_plastic_event_graph(graph, edges)
np.testing.assert_array_equal(graph.outgoing.data, baseline_data)
assert learned.outgoing[0, 2] == 2.5
assert learned.outgoing[1, 2] == 7.0
assert learned.outgoing[2, 3] == -3.0
assert metrics.matched_edges == 2
assert metrics.modified_edges == 1
assert metrics.baseline_absolute_weight == 12.0
assert metrics.effective_absolute_weight == 9.5
```

Use explicit index lookup rather than relying on dense conversion in the production code.

- [ ] **Step 2: Write failing rejection tests**

Add separate tests for duplicate plastic pairs, unknown neuron IDs, missing graph pairs, baseline
weight mismatch, zero-sign graph edges, and nonfinite multipliers. Each must assert a specific
`ValueError` fragment and verify the original graph arrays did not change.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/test_plastic_graph.py
```

Expected: collection fails because `flybrain.plastic_graph` does not exist.

- [ ] **Step 4: Implement one-time CSR pair resolution and patching**

For every plastic pair:

1. map IDs to graph indices;
2. inspect only `outgoing.indices[indptr[pre]:indptr[pre + 1]]`;
3. find `post` with `np.searchsorted()` because CSR indices are sorted;
4. validate one explicit match and `np.isclose(abs(value), baseline, rtol=1e-6, atol=1e-6)`;
5. assign `copied.data[position] = value * multiplier`.

Reject duplicate `(pre_id, post_id)` pairs before patching. Copy the CSR once, retain every neuron
metadata tuple, and return exact metrics including `matched_edges`, `modified_edges`,
`baseline_absolute_weight`, `effective_absolute_weight`, `min_multiplier`, and `max_multiplier`.

- [ ] **Step 5: Run focused and scalability gates**

Run:

```bash
.venv/bin/pytest -q tests/test_plastic_graph.py tests/test_event_graph.py
.venv/bin/ruff check src/flybrain/plastic_graph.py tests/test_plastic_graph.py
.venv/bin/mypy src/flybrain/plastic_graph.py
```

Then run a real materialization smoke command and assert exactly `33_496` matched edges, no dense
allocation, and an unchanged baseline CSR digest.

- [ ] **Step 6: Commit**

```bash
git add src/flybrain/plastic_graph.py tests/test_plastic_graph.py
git commit -m "feat: materialize plastic weights in event graph"
```

---

### Task 3: Deterministic paired Shiu dynamics experiment

**Files:**
- Create: `src/flybrain/shiu_plastic_experiment.py`
- Create: `tests/test_shiu_plastic_experiment.py`

**Interfaces:**
- Consumes: canonical snapshot path, association metrics JSON, plastic NPZ state.
- Produces: `CueConditionMetrics`, `CuePairMetrics`, and `ShiuPlasticIntegrationResult` frozen Pydantic models.
- Produces: `run_shiu_plastic_integration(snapshot: Path, association_path: Path, state_path: Path) -> ShiuPlasticIntegrationResult`

- [ ] **Step 1: Write failing association/state validation tests**

Create a tiny canonical snapshot and a valid association JSON/state pair. Mutate one field at a
time and assert rejection before simulation for:

- state file SHA-256;
- association `snapshot_sha256`;
- state identity;
- target MBON not present in the graph;
- cue ID not present or not represented by a plastic edge to the target;
- association result with `passed=false`.

- [ ] **Step 2: Write failing cue-schedule tests**

Test a public helper:

```python
schedule = build_cue_schedule(graph, cue_ids, ShiuParameters(), batch_size=8)
assert list(schedule.events) == [0, 23]
assert schedule.steps == 23 + 18 + 250 + 1
assert all(amplitudes[0] == np.float32(0.275 * 250.0)
           for _, amplitudes in schedule.events.values())
```

Use 9 cue IDs so the second batch is required. Assert ascending neuron-ID order, no duplicate IDs,
and rejection for `batch_size <= 0`.

- [ ] **Step 3: Write a failing paired-dynamics behavior test**

Use a fixture where cue A has multiplier `0.8`, cue B has `1.0`, and both connect to one target.
Run the paired experiment and assert:

```python
assert result.passed is True
assert result.cue_a.relative_cue_input_decrease >= 0.10
assert result.cue_b.relative_cue_input_drift <= 1e-6
assert result.deterministic_replay_exact is True
assert result.baseline_graph_unchanged is True
assert len(result.cue_a.baseline.target_voltage_mv) == result.protocol.steps
assert len(result.cue_a.learned.target_conductance_mv) == result.protocol.steps
```

Also assert the report contains explicit booleans for target-spike, downstream-spike, and activity
digest changes rather than requiring any of them to be true.

- [ ] **Step 4: Run tests and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/test_shiu_plastic_experiment.py
```

Expected: collection fails because the experiment module does not exist.

- [ ] **Step 5: Implement validation and the exact schedule**

Parse association JSON with `MBAssociationResult`, hash the supplied state file, load the NPZ, call
`validate_state_identity()`, verify association/state identity equality, then build baseline and
learned event graphs. Sort cue IDs, split into batches of eight, schedule batches every
`params.refractory_steps + 1`, and set total steps to:

```python
last_event_step + params.delay_steps + math.ceil(
    5 * params.conductance_tau_ms / params.dt_ms
) + 1
```

- [ ] **Step 6: Implement condition recording without changing Shiu equations**

Pass an explicit `ShiuState` into `simulate_shiu()`. After each yielded batch, read only the target
voltage/conductance scalars from that state. Intersect the batch's emitted IDs with cue IDs scheduled
at that same step; convert those IDs to graph indices and call `graph.propagate_indices()` to record
the actual cue-attributable target arrival amplitude multiplied by `params.synapse_mv`.

Digest each `(step, neuron_ids)` batch exactly as the current Shiu smoke run does. Count target,
total, per-role, and non-cue spikes. Reject missing scheduled cue spikes and nonfinite trajectories.

- [ ] **Step 7: Implement paired acceptance and replay**

Run four primary conditions: cue-A baseline/learned and cue-B baseline/learned. Repeat learned cue A
from a fresh state and require exact digest, voltage trajectory, conductance trajectory, and
cue-attributable input equality. Calculate:

```python
a_decrease = (a_baseline_input - a_learned_input) / a_baseline_input
b_drift = abs(b_learned_input - b_baseline_input) / b_baseline_input
passed = a_decrease >= 0.10 and b_drift <= 1e-6 and replay_exact
```

Require positive denominators and raise `ValueError("Shiu plastic integration acceptance failed")`
before returning if false. Compare baseline CSR array digests before and after all runs.

- [ ] **Step 8: Run focused and full gates**

Run:

```bash
.venv/bin/pytest -q tests/test_shiu_plastic_experiment.py tests/test_shiu_dynamics.py
.venv/bin/ruff check .
.venv/bin/mypy src
.venv/bin/pytest -q
```

Expected: all commands exit 0.

- [ ] **Step 9: Commit**

```bash
git add src/flybrain/shiu_plastic_experiment.py tests/test_shiu_plastic_experiment.py
git commit -m "feat: measure plastic memory in Shiu dynamics"
```

---

### Task 4: Atomic CLI, real MaleCNS gate, and measured documentation

**Files:**
- Modify: `src/flybrain/cli.py`
- Create: `tests/test_shiu_plastic_cli.py`
- Create: `tests/test_shiu_plastic_real.py`
- Create: `docs/data/male-cns-shiu-plastic-memory.md`
- Modify: `docs/superpowers/plans/2026-09-16-shiu-plastic-integration.md`

**Interfaces:**
- Produces CLI: `flybrain experiment shiu-plastic SNAPSHOT --association ASSOCIATION_JSON --state STATE_NPZ --output RESULT_JSON`
- Produces artifact: `artifacts/shiu-plastic-memory-seed7.json`

- [ ] **Step 1: Write failing CLI atomicity tests**

Use `CliRunner` to verify a valid fixture emits JSON and writes one result. Add occupied-output and
failed-acceptance tests that assert the existing file is preserved and no partial result remains.
The command must reject an output path equal to either input artifact after `Path.resolve()`.

- [ ] **Step 2: Implement staged atomic publication**

Validate all input/output paths before simulation. Write the result in a temporary directory under
the output parent, flush and `fsync`, then publish with a no-overwrite hard link as used by the
association CLI. The result JSON must contain its software revision and all acceptance fields.

- [ ] **Step 3: Add the opt-in real-data test**

Read these environment variables:

```text
FLYBRAIN_MALECNS_SNAPSHOT
FLYBRAIN_MB_ASSOCIATION_JSON
FLYBRAIN_MB_STATE_NPZ
```

Skip if any are absent. Otherwise run the complete experiment and assert `166_606` neurons,
`6_240_402` graph edges, `33_496` matched plastic edges, passed acceptance, exact replay, and
unchanged baseline graph.

- [ ] **Step 4: Run the measured real MaleCNS experiment**

Run:

```bash
/usr/bin/time -l .venv/bin/flybrain experiment shiu-plastic \
  /Users/dulatnurlanuly/Downloads/flybrain/artifacts/male-cns-v1.0-w5 \
  --association /Users/dulatnurlanuly/Downloads/flybrain/artifacts/mb-association-seed7.json \
  --state /Users/dulatnurlanuly/Downloads/flybrain/artifacts/mb-association-seed7-state.npz \
  --output /Users/dulatnurlanuly/Downloads/flybrain/artifacts/shiu-plastic-memory-seed7.json
```

Run the deterministic replay verification independently and compare all non-resource fields.

- [ ] **Step 5: Document measured positive and null results**

Record exact identities, schedule, graph/plastic counts, cue-attributable input changes, target and
downstream spike differences, activity digests, trajectory comparison, runtime, peak RSS, and the
remaining scientific limitations. State explicitly that a changed conductance trajectory is not
embodied learning and that an unchanged spike raster is a valid null result.

- [ ] **Step 6: Complete final verification**

Run:

```bash
FLYBRAIN_MALECNS_SNAPSHOT=/Users/dulatnurlanuly/Downloads/flybrain/artifacts/male-cns-v1.0-w5 \
FLYBRAIN_MB_ASSOCIATION_JSON=/Users/dulatnurlanuly/Downloads/flybrain/artifacts/mb-association-seed7.json \
FLYBRAIN_MB_STATE_NPZ=/Users/dulatnurlanuly/Downloads/flybrain/artifacts/mb-association-seed7-state.npz \
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy src
git diff --check
```

Expected: all tests, including the real-data gate, pass with zero failures.

- [ ] **Step 7: Commit**

```bash
git add src/flybrain/cli.py tests/test_shiu_plastic_cli.py tests/test_shiu_plastic_real.py docs/data/male-cns-shiu-plastic-memory.md docs/superpowers/plans/2026-09-16-shiu-plastic-integration.md
git commit -m "feat: validate plastic memory in whole-CNS dynamics"
```

---

## Completion gate

- [ ] Independent code review has no unresolved Critical or Important findings.
- [ ] The feature branch is fast-forward merged into local `master`.
- [ ] Full Ruff, mypy, unit, fixture, and real MaleCNS tests pass again on merged `master`.
- [ ] The worktree is cleanly removed only after merged verification.
- [ ] The project documentation still states that online reward, action selection, and embodied closed-loop behavior remain future gates.
