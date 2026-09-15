# Connectome Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a provenance-aware sparse connectome pipeline and deterministic LIF experiment runner that can acquire verified data, simulate a sensory-to-motor circuit, apply a lesion, restore a checkpoint, and export a reproducible experiment bundle.

**Architecture:** Source manifests feed a guarded content-addressed downloader and canonical Parquet importer. A CPU-first sparse/event dynamics engine consumes the canonical snapshot, while declarative experiments produce inspectable metrics, checkpoints, and provenance bundles.

**Tech Stack:** Python 3.12+, uv, NumPy, SciPy, PyArrow, Pydantic 2, Typer, pytest, Hypothesis, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-16-flybrain-platform-design.md`

## Global Constraints

- Never allocate a dense neuron-by-neuron matrix.
- Raw downloads are immutable and accepted only after SHA-256 verification.
- Refuse downloads that would leave less than 10 GiB free.
- Unknown biological parameters remain explicit and are not silently defaulted.
- Primary behavior is generated only by connectome dynamics; no CNN, PPO, A*, or hard-coded task policy.
- Every experiment records dataset IDs, configuration hash, software revision, seed, runtime, and peak memory.
- Tests use committed miniature fixtures and do not require network access.

---

## File map

- `pyproject.toml`: package metadata, dependencies, CLI entry point, and tool configuration.
- `src/flybrain/manifest.py`: validated source and artifact declarations.
- `src/flybrain/acquire.py`: disk guard, resumable retrieval, and checksum promotion.
- `src/flybrain/schema.py`: canonical neuron/edge Arrow schemas and snapshot metadata.
- `src/flybrain/importers/csv_edges.py`: first source adapter for tabular neuron/edge exports.
- `src/flybrain/graph.py`: compact index and directed sparse adjacency representation.
- `src/flybrain/dynamics.py`: deterministic event-oriented LIF reference engine.
- `src/flybrain/interventions.py`: neuron/population silencing masks.
- `src/flybrain/checkpoint.py`: atomic state save and validated restore.
- `src/flybrain/experiment.py`: declarative execution and metric calculation.
- `src/flybrain/bundle.py`: immutable experiment-bundle writer.
- `src/flybrain/cli.py`: user-facing validation, acquisition, import, and run commands.
- `data/registry/sources.yaml`: curated public-source registry with explicit availability state.
- `tests/fixtures/`: tiny data whose expected graph and spikes can be hand-calculated.

### Task 1: Package skeleton and manifest validation

**Files:**
- Create: `pyproject.toml`
- Create: `src/flybrain/__init__.py`
- Create: `src/flybrain/manifest.py`
- Create: `tests/test_manifest.py`
- Create: `.gitignore`

**Interfaces:**
- Consumes: JSON-compatible dictionaries loaded by callers.
- Produces: `SourceManifest`, `Artifact`, and `load_manifest(path: Path) -> SourceManifest`.

- [x] **Step 1: Write a failing manifest test**

```python
def test_manifest_rejects_artifact_without_sha256(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"dataset_id": "tiny-v1", "artifacts": [{"url": "https://example.invalid/a.csv", "bytes": 12}]}))
    with pytest.raises(ValueError, match="sha256"):
        load_manifest(path)
```

- [x] **Step 2: Verify the red state**

Run: `uv run pytest tests/test_manifest.py::test_manifest_rejects_artifact_without_sha256 -v`

Expected: import failure because `flybrain.manifest` does not exist.

- [x] **Step 3: Add package configuration and minimal validated models**

```python
class Artifact(BaseModel):
    url: HttpUrl
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

class SourceManifest(BaseModel):
    dataset_id: str = Field(min_length=1)
    source_publication: HttpUrl
    license: str = Field(min_length=1)
    artifacts: tuple[Artifact, ...]

def load_manifest(path: Path) -> SourceManifest:
    return SourceManifest.model_validate_json(path.read_text())
```

- [x] **Step 4: Add valid-manifest and duplicate-URL tests, then run the file**

Run: `uv run pytest tests/test_manifest.py -v`

Expected: all manifest tests pass.

- [x] **Step 5: Commit the package and contract**

```bash
git add pyproject.toml .gitignore src/flybrain tests/test_manifest.py
git commit -m "feat: define source manifest contract"
```

### Task 2: Guarded content-addressed acquisition

**Files:**
- Create: `src/flybrain/acquire.py`
- Create: `tests/test_acquire.py`

**Interfaces:**
- Consumes: `Artifact`, destination root, injectable free-space and stream functions.
- Produces: `DiskBudgetError`, `ChecksumMismatch`, `artifact_path(root: Path, artifact: Artifact) -> Path`, and `acquire_artifact(...) -> Path`.

- [x] **Step 1: Write a failing disk-reserve test**

```python
def test_acquire_refuses_when_ten_gib_reserve_would_be_crossed(tmp_path: Path) -> None:
    artifact = artifact_for(b"abc")
    with pytest.raises(DiskBudgetError):
        acquire_artifact(artifact, tmp_path, free_bytes=lambda _: TEN_GIB + 2)
```

- [x] **Step 2: Verify it fails because acquisition is absent**

Run: `uv run pytest tests/test_acquire.py::test_acquire_refuses_when_ten_gib_reserve_would_be_crossed -v`

Expected: import failure for `flybrain.acquire`.

- [x] **Step 3: Implement the space gate and content address**

```python
TEN_GIB = 10 * 1024**3

def artifact_path(root: Path, artifact: Artifact) -> Path:
    return root / "sha256" / artifact.sha256[:2] / artifact.sha256

def ensure_space(root: Path, expected_bytes: int, free_bytes: Callable[[Path], int]) -> None:
    if free_bytes(root) - expected_bytes < TEN_GIB:
        raise DiskBudgetError(expected_bytes)
```

- [x] **Step 4: Test checksum mismatch, atomic promotion, cache hit, and partial-file quarantine**

Run: `uv run pytest tests/test_acquire.py -v`

Expected: all acquisition tests pass using local byte streams only.

- [x] **Step 5: Commit verified acquisition**

```bash
git add src/flybrain/acquire.py tests/test_acquire.py
git commit -m "feat: add verified dataset acquisition"
```

### Task 3: Canonical sparse snapshot import

**Files:**
- Create: `src/flybrain/schema.py`
- Create: `src/flybrain/importers/__init__.py`
- Create: `src/flybrain/importers/csv_edges.py`
- Create: `src/flybrain/graph.py`
- Create: `tests/fixtures/tiny/neurons.csv`
- Create: `tests/fixtures/tiny/edges.csv`
- Create: `tests/test_import_csv.py`
- Create: `tests/test_graph.py`

**Interfaces:**
- Consumes: source neuron and edge CSV paths.
- Produces: `SnapshotMetadata`, `import_csv_snapshot(neurons, edges, output, metadata) -> Path`, and `SparseConnectome.from_snapshot(path) -> SparseConnectome` with CSR incoming/outgoing arrays.

- [x] **Step 1: Write a failing referential-integrity test**

```python
def test_import_rejects_edge_to_unknown_neuron(tmp_path: Path) -> None:
    with pytest.raises(SnapshotIntegrityError, match="unknown neuron 99"):
        import_csv_snapshot(fixture("neurons.csv"), fixture("bad_edges.csv"), tmp_path, metadata())
```

- [x] **Step 2: Verify the importer test is red**

Run: `uv run pytest tests/test_import_csv.py::test_import_rejects_edge_to_unknown_neuron -v`

Expected: import failure for the missing importer.

- [x] **Step 3: Define exact Arrow fields and write canonical Parquet files**

```python
NEURON_SCHEMA = pa.schema([
    ("neuron_id", pa.uint64()), ("source_dataset", pa.string()),
    ("cell_type", pa.string()), ("side", pa.string()),
    ("transmitter", pa.string()), ("role", pa.string()),
    ("annotation_confidence", pa.float32()),
])
EDGE_SCHEMA = pa.schema([
    ("pre_id", pa.uint64()), ("post_id", pa.uint64()),
    ("synapse_count", pa.uint32()), ("sign", pa.int8()),
    ("sign_provenance", pa.string()), ("confidence", pa.float32()),
])
```

- [x] **Step 4: Write graph tests for direction, index stability, edge-order invariance, and `O(N+E)` storage**

Run: `uv run pytest tests/test_import_csv.py tests/test_graph.py -v`

Expected: fixture imports and sparse invariants pass.

- [x] **Step 5: Commit the canonical snapshot path**

```bash
git add src/flybrain/schema.py src/flybrain/importers src/flybrain/graph.py tests/fixtures tests/test_import_csv.py tests/test_graph.py
git commit -m "feat: import canonical sparse connectomes"
```

### Task 4: Deterministic LIF engine

**Files:**
- Create: `src/flybrain/dynamics.py`
- Create: `tests/test_dynamics.py`

**Interfaces:**
- Consumes: `SparseConnectome`, `LIFParameters`, initial state, and timestamped input currents.
- Produces: `LIFState`, `SpikeBatch`, and `simulate_lif(...) -> Iterator[SpikeBatch]`.

- [x] **Step 1: Write a failing hand-calculated threshold test**

```python
def test_single_neuron_spikes_at_analytic_threshold() -> None:
    params = LIFParameters(dt_ms=1.0, tau_ms=10.0, rest_mv=0.0, reset_mv=0.0, threshold_mv=1.0)
    spikes = list(simulate_lif(single_neuron_graph(), params, constant_current(2.0, steps=8), seed=7))
    assert [batch.step for batch in spikes if batch.neuron_ids.size] == [6]
```

- [x] **Step 2: Verify the dynamics test is red**

Run: `uv run pytest tests/test_dynamics.py::test_single_neuron_spikes_at_analytic_threshold -v`

Expected: import failure for `flybrain.dynamics`.

- [x] **Step 3: Implement vector state with sparse event propagation**

```python
voltage += (-(voltage - params.rest_mv) + current) * (params.dt_ms / params.tau_ms)
fired = voltage >= params.threshold_mv
voltage[fired] = params.reset_mv
current_next = graph.propagate(fired.astype(np.float32))
```

- [x] **Step 4: Add tests for inhibitory sign, one-step delay, reset, identical-seed replay, and no dense allocation API**

Run: `uv run pytest tests/test_dynamics.py -v`

Expected: analytic and determinism tests pass.

- [x] **Step 5: Commit reference dynamics**

```bash
git add src/flybrain/dynamics.py tests/test_dynamics.py
git commit -m "feat: add deterministic LIF reference engine"
```

### Task 5: Lesions and atomic checkpoints

**Files:**
- Create: `src/flybrain/interventions.py`
- Create: `src/flybrain/checkpoint.py`
- Create: `tests/test_interventions.py`
- Create: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: neuron metadata predicates and `LIFState`.
- Produces: `silence_mask(graph, predicate) -> NDArray[np.bool_]`, `save_checkpoint(path, state, metadata)`, and `load_checkpoint(path, expected_config_hash) -> tuple[LIFState, CheckpointMetadata]`.

- [x] **Step 1: Write a failing causal-lesion test**

```python
def test_silencing_relay_removes_motor_spikes() -> None:
    normal = run_tiny_pathway(silenced=np.zeros(3, dtype=bool))
    lesioned = run_tiny_pathway(silenced=np.array([False, True, False]))
    assert normal.motor_spikes > 0
    assert lesioned.motor_spikes == 0
```

- [x] **Step 2: Verify the lesion test is red**

Run: `uv run pytest tests/test_interventions.py::test_silencing_relay_removes_motor_spikes -v`

Expected: missing intervention API.

- [x] **Step 3: Apply masks at spike emission and persist complete state atomically**

```python
fired &= ~silenced
temporary = path.with_suffix(path.suffix + ".partial")
np.savez_compressed(temporary, voltage=state.voltage, step=state.step, rng_state=json.dumps(state.rng_state))
temporary.replace(path)
```

- [x] **Step 4: Test config-hash mismatch, corrupt checkpoint rejection, and split-run replay equivalence**

Run: `uv run pytest tests/test_interventions.py tests/test_checkpoint.py -v`

Expected: lesion and restoration behavior passes.

- [x] **Step 5: Commit intervention and replay support**

```bash
git add src/flybrain/interventions.py src/flybrain/checkpoint.py tests/test_interventions.py tests/test_checkpoint.py
git commit -m "feat: support lesions and deterministic checkpoints"
```

### Task 6: Declarative experiments and reproducibility bundles

**Files:**
- Create: `src/flybrain/experiment.py`
- Create: `src/flybrain/bundle.py`
- Create: `tests/test_experiment.py`
- Create: `tests/test_bundle.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, canonical snapshot, stimulus, and optional lesion.
- Produces: `ExperimentResult`, `run_experiment(config) -> ExperimentResult`, and `write_bundle(result, output) -> Path`.

- [x] **Step 1: Write a failing bundle-completeness test**

```python
def test_bundle_contains_required_reproducibility_records(tmp_path: Path) -> None:
    bundle = write_bundle(tiny_result(), tmp_path / "run")
    assert required_files(bundle) == {
        "config.json", "metrics.json", "provenance.json", "events.jsonl", "environment.json"
    }
```

- [x] **Step 2: Verify the bundle test is red**

Run: `uv run pytest tests/test_bundle.py::test_bundle_contains_required_reproducibility_records -v`

Expected: import failure for the bundle module.

- [x] **Step 3: Implement config hashing, causal metric, timing, and peak-memory capture**

```python
class ExperimentResult(BaseModel):
    config_hash: str
    dataset_id: str
    seed: int
    motor_spikes: int
    lesion_effect: float | None
    runtime_seconds: float
    peak_memory_bytes: int
    software_revision: str
```

- [x] **Step 4: Test normal/lesioned paired execution and deterministic observable output**

Run: `uv run pytest tests/test_experiment.py tests/test_bundle.py -v`

Expected: metrics and bundle tests pass.

- [x] **Step 5: Commit experiment outputs**

```bash
git add src/flybrain/experiment.py src/flybrain/bundle.py tests/test_experiment.py tests/test_bundle.py
git commit -m "feat: export reproducible lesion experiments"
```

### Task 7: CLI, source registry, and scaling gate

**Files:**
- Create: `src/flybrain/cli.py`
- Create: `data/registry/sources.yaml`
- Create: `tests/test_cli.py`
- Create: `tests/test_scaling.py`
- Create: `README.md`

**Interfaces:**
- Consumes: manifests, fixture/source paths, and experiment configuration.
- Produces: `flybrain manifest validate`, `flybrain data acquire`, `flybrain snapshot import-csv`, and `flybrain experiment run` commands.

- [x] **Step 1: Write a failing CLI smoke test**

```python
def test_cli_runs_tiny_lesion_experiment(tmp_path: Path) -> None:
    result = runner.invoke(app, ["experiment", "run", "tests/fixtures/tiny/experiment.json", "--output", str(tmp_path)])
    assert result.exit_code == 0
    assert "lesion_effect" in result.stdout
```

- [x] **Step 2: Verify the CLI test is red**

Run: `uv run pytest tests/test_cli.py::test_cli_runs_tiny_lesion_experiment -v`

Expected: import failure for `flybrain.cli`.

- [x] **Step 3: Wire thin CLI commands to tested library interfaces and document exact commands**

```python
@experiment_app.command("run")
def run_command(config: Path, output: Path) -> None:
    result = run_experiment(load_experiment(config))
    bundle = write_bundle(result, output)
    typer.echo(result.model_dump_json())
    typer.echo(f"bundle={bundle}")
```

- [x] **Step 4: Add a scaling regression that compares 1,000-edge and 10,000-edge sparse graphs**

Run: `uv run pytest tests/test_cli.py tests/test_scaling.py -v`

Expected: stored sparse array bytes grow linearly within a 12x upper bound; no dense matrix exists.

- [x] **Step 5: Run the complete quality gate**

Run: `uv run ruff check . && uv run mypy src && uv run pytest -q`

Expected: zero lint errors, zero type errors, and zero failed tests.

- [x] **Step 6: Run the real tiny experiment and inspect its bundle**

Run: `uv run flybrain experiment run tests/fixtures/tiny/experiment.json --output artifacts/tiny-run`

Expected: exit code 0, a nonzero normal motor spike count, zero lesioned motor spikes, and all five reproducibility records.

- [x] **Step 7: Commit the milestone**

```bash
git add src/flybrain/cli.py data/registry/sources.yaml tests/test_cli.py tests/test_scaling.py README.md
git commit -m "feat: deliver connectome core milestone"
```

## Plan self-review record

- Spec coverage: all first-milestone acceptance items map to Tasks 1 through 7.
- Deferred platform stages remain outside this plan as required by the approved design.
- Type names and signatures are consistent across acquisition, snapshot, dynamics, intervention, checkpoint, experiment, and CLI boundaries.
- Network-independent fixtures provide deterministic tests; live source acquisition remains an explicit post-test operation.
