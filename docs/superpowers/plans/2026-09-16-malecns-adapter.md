# MaleCNS v1.0 Streaming Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Convert verified official MaleCNS v1.0 Feather exports into a canonical sparse snapshot using the published filtering mechanism and bounded memory.

**Architecture:** A manifest resolver locates immutable artifacts, a metadata stage joins selected annotations and consensus neurotransmitters, and a streaming Arrow stage filters and writes weight batches into an atomic Parquet snapshot. Metrics record every source and retained count plus unresolved biological assumptions.

**Tech Stack:** Python 3.12+, PyArrow, NumPy, Pydantic 2, Typer, pytest, Hypothesis, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-16-malecns-adapter-design.md`

## Global constraints

- Reproduce the published valid-superclass and default `weight >= 5` policy.
- Never load the full weight table or create a dense matrix.
- Preserve unresolved transmitter sign as `0`; never invent a fast excitatory sign for modulators or unclear annotations.
- Refuse output replacement and preserve failed work only under a `.partial` path.
- Validate all real inputs against the committed SHA-256 manifest before conversion.
- Keep peak RSS below 8 GiB and free disk above 10 GiB.

### Task 1: Evolve canonical schemas for source-rich and unresolved metadata

**Files:**
- Modify: `src/flybrain/schema.py`
- Modify: `src/flybrain/importers/csv_edges.py`
- Modify: `tests/fixtures/tiny/neurons.csv`
- Modify: `tests/fixtures/tiny/edges.csv`
- Modify: `tests/fixtures/tiny/bad_edges.csv`
- Modify: `tests/test_import_csv.py`
- Modify: `tests/test_graph.py`

**Interfaces:**
- Produces canonical neuron fields `superclass`, `annotation_status`, `status_label`, and `transmitter_provenance`.
- Accepts canonical edge signs in `{-1, 0, 1}` while retaining zero-sign rows in Parquet.

- [x] Write failing schema and zero-sign preservation tests.
- [x] Run the focused tests and confirm contract failures.
- [x] Implement the schema evolution and importer validation.
- [x] Run import and graph tests; confirm all pass.
- [x] Commit as `feat: preserve unresolved connectome metadata`.

### Task 2: Resolve verified MaleCNS artifacts and metadata joins

**Files:**
- Create: `src/flybrain/importers/malecns.py`
- Create: `tests/fixtures/malecns/`
- Create: `tests/test_malecns_metadata.py`

**Interfaces:**
- Produces `MaleCNSSources.from_manifest(manifest_path, cache_root)`.
- Produces `select_neurons(annotations, neurotransmitters) -> tuple[pa.Table, MaleCNSSelectionMetrics]`.
- Produces `transmitter_sign(name: str) -> tuple[int, str]`.

- [x] Write fixture-based failing tests for superclass selection, metadata fallbacks, and transmitter signs.
- [x] Verify the tests fail because the adapter is absent.
- [x] Implement verified source resolution and deterministic metadata selection.
- [x] Run metadata tests and all pre-existing tests.
- [x] Commit as `feat: select canonical MaleCNS neurons`.

### Task 3: Stream weights into an atomic canonical snapshot

**Files:**
- Modify: `src/flybrain/importers/malecns.py`
- Create: `tests/test_malecns_streaming.py`

**Interfaces:**
- Produces `import_malecns(sources, output, min_weight=5) -> MaleCNSImportMetrics`.
- Produces `validate_malecns_snapshot(path) -> MaleCNSImportMetrics`.

- [x] Write failing tests for thresholding, endpoint filtering, unresolved signs, batch-order invariance, output refusal, and partial-output behavior.
- [x] Verify focused tests fail for missing streaming behavior.
- [x] Implement record-batch filtering, Parquet row-group writing, metrics, and atomic promotion.
- [x] Run streaming tests and full suite.
- [x] Commit as `feat: stream MaleCNS weights into snapshots`.

### Task 4: CLI and real-data acceptance run

**Files:**
- Modify: `src/flybrain/cli.py`
- Create: `tests/test_malecns_cli.py`
- Modify: `README.md`
- Create: `docs/data/male-cns-v1.0-import.md`

**Interfaces:**
- Produces `flybrain snapshot import-malecns MANIFEST --cache-root PATH --output PATH --min-weight 5`.
- Produces JSON metrics on stdout and canonical snapshot files on disk.

- [x] Write a failing CLI fixture import test.
- [x] Implement the thin CLI command and exact README workflow.
- [x] Run Ruff, mypy, and the complete pytest suite.
- [x] Run the adapter against all verified v1.0 source batches while measuring `/usr/bin/time -l`.
- [x] Validate the produced real snapshot, document measured counts/resources, and verify free-space reserve.
- [x] Commit as `feat: import verified MaleCNS v1.0 snapshot`.

## Self-review record

- The plan covers every design acceptance criterion and keeps later dynamics/plasticity work out of this adapter.
- Interfaces use one sign policy and one selection rule throughout.
- Real-data claims require measured output from the official verified artifacts, not fixture extrapolation.
