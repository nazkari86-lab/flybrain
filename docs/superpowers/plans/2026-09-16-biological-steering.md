# Biological Interface Registry and Causal Steering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace arbitrary embodied neuron mappings with snapshot-bound biological populations and publish a causal, replayable MaleCNS steering and retreat benchmark.

**Architecture:** A strict registry resolves exact annotation predicates into provenance-bound neuron populations. Focused retinal and descending-neuron interfaces feed a new multi-condition benchmark runner that keeps direct DN calibration, open-loop sensory response, and closed-loop behavior separate. A dedicated CLI validates the snapshot and registry before loading the sparse graph and atomically publishes evidence, interventions, digests, and claim classifications.

**Tech Stack:** Python 3.12, Pydantic 2, PyArrow/Parquet, NumPy, SciPy sparse CSR, Typer, pytest, Ruff, mypy

**Spec:** `docs/superpowers/specs/2026-09-16-biological-steering-design.md`

## Global Constraints

- Canonical topology and weights remain immutable; no dense neuron-by-neuron matrix is allowed.
- The primary agent may not use CNN, PPO, LLM, A*, hard-coded target seeking, target coordinates in the decoder, or any hidden learned policy.
- Every claim is labeled `dataset_measurement`, `experimental_result`, `model_assumption`, or `simulation_observation`.
- Population selection uses allowlisted exact predicates, never row order, graph degree, ID ranges, or arbitrary truncation.
- Direct DN calibration is never reported as visually evoked steering.
- The retained visual annotations support hemispheric photoreceptor banks only; do not claim an ommatidial or retinotopic map.
- R1-R6 receives luminance/contrast only; direct HS optic-flow and LC16 looming drive is calibration that explicitly bypasses upstream visual processing.
- Real sensory outcomes may be positive, null, directionally wrong, or underpowered; thresholds are fixed before the run.
- Outputs refuse overwrite, are staged and fsynced, and are linked atomically only after every gate passes.
- Python code remains Ruff-clean and mypy-strict; all feature work follows RED-GREEN-REFACTOR.

## File structure

- Create `src/flybrain/biological_registry.py`: registry schemas, strict annotation resolution, evidence validation, and graph binding.
- Create `data/registry/biological-interface-registry-v1.json`: reviewed selectors, exact expected MaleCNS IDs/counts, and literature records.
- Create `src/flybrain/retinal_interface.py`: label-free angular projection, motion/looming observation, and hemispheric visual encoding.
- Create `src/flybrain/descending_interface.py`: evidence-bound DN maps, rate summaries, causal masks, and bounded motor decoding.
- Create `src/flybrain/steering_benchmark.py`: protocol definitions, condition execution, comparisons, classifications, and result model.
- Modify `src/flybrain/cli.py`: add `experiment biological-steering` with fail-closed atomic publication.
- Modify `src/flybrain/embodied_episode.py`: label the legacy arbitrary mapping result as smoke-only.
- Create `tests/fixtures/biological_registry_fixture.json`: small exact registry for unit and synthetic-graph tests.
- Create `tests/test_biological_registry.py`: schema, resolution, identity, and failure tests.
- Create `tests/test_retinal_interface.py`: geometry, mirroring, monotonic looming, and encoding tests.
- Create `tests/test_descending_interface.py`: laterality, cancellation, retreat, and lesion tests.
- Create `tests/test_steering_benchmark.py`: synthetic positive/null protocols, replay, intervention, restoration, and integrity tests.
- Create `tests/test_biological_steering_cli.py`: output safety and serialized-evidence tests.
- Create `tests/test_biological_steering_real.py`: opt-in full MaleCNS gate.
- Create `docs/data/male-cns-biological-steering.md`: exact real-run command, resources, outcomes, nulls, and claim boundary.

---

### Task 1: Strict biological registry and resolver

**Files:**
- Create: `src/flybrain/biological_registry.py`
- Create: `tests/fixtures/biological_registry_fixture.json`
- Create: `tests/test_biological_registry.py`

**Interfaces:**
- Consumes: canonical `metadata.json`, `source-annotations.parquet`, and `EventConnectome.neuron_ids`.
- Produces: `load_biological_registry(path: Path) -> BiologicalInterfaceRegistry`, `resolve_biological_registry(registry: BiologicalInterfaceRegistry, snapshot: Path) -> ResolvedRegistry`, and `ResolvedRegistry.validate_graph(graph: EventConnectome) -> None`.

- [ ] **Step 1: Write failing model and selector tests**

```python
def test_registry_resolves_exact_sorted_population(tmp_path: Path) -> None:
    snapshot = annotation_snapshot(
        tmp_path,
        [
            {"bodyId": 30, "type": "DNa02", "superclass": "descending_neuron", "somaSide": "R", "rootSide": None, "class": None},
            {"bodyId": 10, "type": "DNa02", "superclass": "descending_neuron", "somaSide": "L", "rootSide": None, "class": None},
        ],
    )
    registry = load_biological_registry(FIXTURE_REGISTRY)

    resolved = resolve_biological_registry(registry, snapshot)

    assert resolved.population("d_na02_left").neuron_ids == (10,)
    assert resolved.population("d_na02_left").selector.equals["somaSide"] == "L"
    assert len(resolved.annotation_sha256) == 64


@pytest.mark.parametrize("column", ["unknown", "bodyId", "assignedOlHex1"])
def test_selector_rejects_non_allowlisted_columns(column: str) -> None:
    with pytest.raises(ValueError, match="selector column"):
        AnnotationSelector(equals={column: "x"})
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `.venv/bin/pytest tests/test_biological_registry.py -q`

Expected: collection fails because `flybrain.biological_registry` does not exist.

- [ ] **Step 3: Implement immutable registry schemas**

```python
EvidenceKind = Literal[
    "dataset_measurement",
    "experimental_result",
    "model_assumption",
    "simulation_observation",
]
ALLOWED_SELECTOR_COLUMNS = frozenset(
    {"type", "superclass", "class", "subclass", "somaSide", "rootSide", "entryNerve", "exitNerve", "receptorType"}
)


class EvidenceRecord(BaseModel, frozen=True):
    evidence_id: str = Field(min_length=1)
    kind: EvidenceKind
    source_url: AnyHttpUrl
    claim: str = Field(min_length=1)
    pmid: str | None = None
    doi: str | None = None
    confidence: Literal["measured", "high", "moderate", "assumption"]


class AnnotationSelector(BaseModel, frozen=True):
    equals: dict[str, str]
    in_values: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    expected_ids: tuple[int, ...] = ()
    expected_count: int | None = Field(default=None, gt=0)


class PopulationDeclaration(BaseModel, frozen=True):
    name: str
    role: Literal["sensory", "steering", "retreat", "future_interface"]
    selector: AnnotationSelector
    evidence_ids: tuple[str, ...]


class BiologicalInterfaceRegistry(BaseModel, frozen=True):
    registry_version: str
    dataset_id: str
    populations: tuple[PopulationDeclaration, ...]
    evidence: tuple[EvidenceRecord, ...]
```

Add model validators that reject duplicate names/evidence IDs, absent evidence references, empty predicates, conflicting `equals`/`in_values` keys, non-allowlisted columns, duplicate/unsorted expected IDs, and evidence-kind/confidence contradictions. `experimental_result` requires a PMID or DOI; `model_assumption` requires `confidence="assumption"`.

- [ ] **Step 4: Implement exact Arrow resolution and identity binding**

```python
class ResolvedPopulation(BaseModel, frozen=True):
    name: str
    role: str
    selector: AnnotationSelector
    neuron_ids: tuple[int, ...]
    id_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: tuple[EvidenceRecord, ...]


class ResolvedRegistry(BaseModel, frozen=True):
    registry_version: str
    registry_sha256: str
    dataset_id: str
    source_manifest_sha256: str
    snapshot_content_sha256: str
    annotation_sha256: str
    populations: tuple[ResolvedPopulation, ...]

    def population(self, name: str) -> ResolvedPopulation:
        matches = tuple(item for item in self.populations if item.name == name)
        if len(matches) != 1:
            raise ValueError(f"resolved population not found exactly once: {name}")
        return matches[0]

    def validate_graph(self, graph: EventConnectome) -> None:
        available = {int(value) for value in graph.neuron_ids}
        required = {
            neuron_id
            for population in self.populations
            for neuron_id in population.neuron_ids
        }
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"resolved population IDs absent from graph: {missing}")
```

Resolve one predicate at a time with `pyarrow.compute.equal` and `pyarrow.compute.is_in`, combine masks with `and_`, select `bodyId`, convert to sorted unique positive integers, and compare exact IDs/counts when declared. Hash canonical registry bytes, annotation file bytes, and `b"\n".join(str(value).encode("ascii") for value in neuron_ids)`. Validate `dataset_id` plus both current snapshot identities before returning.

- [ ] **Step 5: Add fail-closed tests**

```python
def test_registry_rejects_expected_id_drift(tmp_path: Path) -> None:
    snapshot = annotation_snapshot(
        tmp_path,
        [{"bodyId": 11, "type": "DNa02", "superclass": "descending_neuron", "somaSide": "L", "rootSide": None, "class": None}],
    )
    with pytest.raises(ValueError, match="expected IDs"):
        resolve_biological_registry(load_biological_registry(FIXTURE_REGISTRY), snapshot)


def test_resolved_registry_rejects_graph_missing_population_id(
    tmp_path: Path,
) -> None:
    snapshot = annotation_snapshot(
        tmp_path,
        [{"bodyId": 10, "type": "DNa02", "superclass": "descending_neuron", "somaSide": "L", "rootSide": None, "class": None}],
    )
    resolved = resolve_biological_registry(
        load_biological_registry(FIXTURE_REGISTRY), snapshot
    )
    graph_without_id_10 = EventConnectome(
        neuron_ids=np.array([99], dtype=np.uint64),
        cell_types=("fixture",),
        roles=("interneuron",),
        transmitters=("acetylcholine",),
        superclasses=("cb_intrinsic",),
        outgoing=csr_array((1, 1), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="absent from graph"):
        resolved.validate_graph(graph_without_id_10)
```

Also cover duplicate source `bodyId`, missing annotation columns, empty matches, metadata mismatch, registry hash stability, evidence validation, and unexpected laterality.

- [ ] **Step 6: Run quality gates and commit**

Run: `.venv/bin/pytest tests/test_biological_registry.py -q`

Run: `.venv/bin/ruff check src/flybrain/biological_registry.py tests/test_biological_registry.py`

Run: `.venv/bin/mypy src/flybrain/biological_registry.py`

Expected: all pass.

```bash
git add src/flybrain/biological_registry.py tests/fixtures/biological_registry_fixture.json tests/test_biological_registry.py
git commit -m "feat: resolve evidence-bound neuron populations"
```

### Task 2: Reviewed MaleCNS registry artifact

**Files:**
- Create: `data/registry/biological-interface-registry-v1.json`
- Modify: `tests/test_biological_registry.py`

**Interfaces:**
- Consumes: Task 1 registry schema and the retained `source-annotations.parquet`.
- Produces: the canonical v1 declarations used by the CLI and full-data test.

- [ ] **Step 1: Add a failing retained-snapshot resolution test**

```python
def test_v1_registry_declares_exact_documented_populations() -> None:
    registry = load_biological_registry(Path("data/registry/biological-interface-registry-v1.json"))
    by_name = {item.name: item for item in registry.populations}

    assert by_name["d_na02_left"].selector.expected_ids == (523769,)
    assert by_name["d_na02_right"].selector.expected_ids == (10360,)
    assert by_name["d_ng13_left"].selector.expected_ids == (11074,)
    assert by_name["d_ng13_right"].selector.expected_ids == (512006,)
    assert by_name["mdn_left"].selector.expected_ids == (11288, 12348)
    assert by_name["mdn_right"].selector.expected_ids == (10763, 11332)
    assert by_name["visual_all"].selector.expected_count == 6091
    assert by_name["visual_r1_r6_left"].selector.expected_count == 1112
    assert by_name["visual_r1_r6_right"].selector.expected_count == 2265
    assert by_name["hs_left"].selector.expected_ids == (10034, 10181, 10419, 11793)
    assert by_name["hs_right"].selector.expected_ids == (10015, 10016, 10023, 12521)
    assert by_name["lc16_left"].selector.expected_count == 88
    assert by_name["lc16_right"].selector.expected_count == 94
```

- [ ] **Step 2: Confirm RED**

Run: `.venv/bin/pytest tests/test_biological_registry.py::test_v1_registry_declares_exact_documented_populations -q`

Expected: FAIL because the registry JSON is absent.

- [ ] **Step 3: Write the complete reviewed registry**

Include declarations for `visual_all`, `visual_r1_r6_left`, `visual_r1_r6_right`, bilateral `hs_visual_projection` cells (`HSE`, `HSN`, `HSS`, `HST`), bilateral `lc16_visual_projection`, `olfactory_cb`, `tactile_vnc`, `proprioceptive_vnc`, both DNa02, both DNg13, and both MDN populations. Use exact predicates and these source records:

```json
{
  "evidence_id": "yang-2024-steering",
  "kind": "experimental_result",
  "source_url": "https://pubmed.ncbi.nlm.nih.gov/39293446/",
  "pmid": "39293446",
  "doi": "10.1016/j.cell.2024.08.033",
  "claim": "DNa02 and DNg13 activity is ipsiversive; unilateral activation drives ipsiversive steering through distinct stride effects.",
  "confidence": "high"
}
```

Add equivalent complete records for PMID 28238656, PMID 30528583, the official MaleCNS release, and each explicit decoder/hemispheric-interface assumption. Do not copy claims broader than the cited evidence.

- [ ] **Step 4: Resolve against the retained full snapshot**

Run:

```bash
.venv/bin/python -c 'from pathlib import Path; from flybrain.biological_registry import load_biological_registry, resolve_biological_registry; r=load_biological_registry(Path("data/registry/biological-interface-registry-v1.json")); x=resolve_biological_registry(r, Path("artifacts/male-cns-v1.0-w5")); print([(p.name, len(p.neuron_ids)) for p in x.populations])'
```

Expected: exact DN and HS IDs, `visual_all=6091`, R1-R6 side counts `1112/2265`, LC16 side
counts `88/94`, and no identity error. Any difference is a snapshot/selector drift failure: inspect
the snapshot identity and selector rather than changing the expected value during this run.

- [ ] **Step 5: Run tests and commit**

Run: `.venv/bin/pytest tests/test_biological_registry.py -q`

Run: `.venv/bin/ruff check src/flybrain/biological_registry.py tests/test_biological_registry.py`

Expected: all pass.

```bash
git add data/registry/biological-interface-registry-v1.json tests/test_biological_registry.py
git commit -m "data: register MaleCNS steering populations"
```

### Task 3: Label-free retinal observation and explicit visual feature boundary

**Files:**
- Create: `src/flybrain/retinal_interface.py`
- Create: `tests/test_retinal_interface.py`

**Interfaces:**
- Consumes: `FlyBody`, anonymous `VisualDisc` geometry, resolved left/right R1-R6 IDs, HS IDs, and LC16 IDs.
- Produces: `observe_retina(body: FlyBody, previous: tuple[VisualDisc, ...], current: tuple[VisualDisc, ...]) -> RetinalObservation`, `VisualInterfaceEncoder.encode_photoreceptors(observation: RetinalObservation, *, step: int) -> tuple[ExternalEvent, ...]`, and `VisualInterfaceEncoder.encode_feature_calibration(observation: RetinalObservation, *, step: int) -> tuple[ExternalEvent, ...]`.

- [ ] **Step 1: Write failing geometry and causality tests**

```python
def test_mirroring_scene_swaps_motion_banks() -> None:
    body = FlyBody(5.0, 5.0, 0.0, 0.0, 0.0, 1.0, (False,) * 6)
    previous = (VisualDisc(8.0, 5.5, 0.4),)
    current = (VisualDisc(8.0, 6.0, 0.4),)

    left = observe_retina(body, previous, current)
    right = observe_retina(body, mirror_y(previous, 5.0), mirror_y(current, 5.0))

    assert left.left_motion == pytest.approx(right.right_motion)
    assert left.right_motion == pytest.approx(right.left_motion)


def test_looming_is_monotonic_with_angular_expansion() -> None:
    far = observe_retina(BODY, (VisualDisc(9.0, 5.0, 0.2),), (VisualDisc(8.0, 5.0, 0.2),))
    near = observe_retina(BODY, (VisualDisc(8.0, 5.0, 0.2),), (VisualDisc(6.0, 5.0, 0.2),))
    assert 0.0 < far.looming < near.looming <= 1.0


def test_photoreceptor_events_never_contain_motion_or_looming_channels() -> None:
    events = ENCODER.encode_photoreceptors(OBSERVATION, step=3)
    assert {event.channel for event in events} <= {
        "luminance_left", "luminance_right", "contrast_left", "contrast_right"
    }
```

- [ ] **Step 2: Confirm RED**

Run: `.venv/bin/pytest tests/test_retinal_interface.py -q`

Expected: import failure for `flybrain.retinal_interface`.

- [ ] **Step 3: Implement bounded angular projection**

```python
@dataclass(frozen=True)
class VisualDisc:
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class RetinalObservation:
    left_luminance: float
    right_luminance: float
    left_contrast: float
    right_contrast: float
    left_motion: float
    right_motion: float
    looming: float


def observe_retina(
    body: FlyBody,
    previous: tuple[VisualDisc, ...],
    current: tuple[VisualDisc, ...],
) -> RetinalObservation:
    """Project anonymous discs; return no object identity, coordinates, or desired action."""
```

Require parallel scene tuples so temporal correspondences are explicit inside the sensor only. Compute wrapped body-relative bearing, angular half-width `asin(min(1, radius / distance))`, signed bearing delta, and positive half-width growth. Aggregate and clip each channel to `[0, 1]`. Reject nonfinite values, nonpositive radii, body-disc overlap, tuple-length mismatch, and ambiguous wrapped jumps at exactly pi.

- [ ] **Step 4: Implement hemispheric event encoding**

```python
@dataclass(frozen=True)
class VisualInterfaceMap:
    left_r1_r6_ids: tuple[int, ...]
    right_r1_r6_ids: tuple[int, ...]
    left_hs_ids: tuple[int, ...]
    right_hs_ids: tuple[int, ...]
    left_lc16_ids: tuple[int, ...]
    right_lc16_ids: tuple[int, ...]


class VisualInterfaceEncoder:
    version = "hemispheric-visual-interface-v1"

    def __init__(self, mapping: VisualInterfaceMap, *, total_voltage: float = 68.75) -> None:
        if not math.isfinite(total_voltage) or total_voltage <= 0:
            raise ValueError("total_voltage must be finite and positive")
        mapping.validate_disjoint_nonempty()
        self.mapping = mapping
        self.total_voltage = total_voltage

    def encode_photoreceptors(
        self, observation: RetinalObservation, *, step: int
    ) -> tuple[ExternalEvent, ...]:
        if step < 0:
            raise ValueError("step must be non-negative")
        channels = (
            ("luminance_left", self.mapping.left_r1_r6_ids, observation.left_luminance),
            ("luminance_right", self.mapping.right_r1_r6_ids, observation.right_luminance),
            ("contrast_left", self.mapping.left_r1_r6_ids, observation.left_contrast),
            ("contrast_right", self.mapping.right_r1_r6_ids, observation.right_contrast),
        )
        return self._events(channels, step=step)

    def encode_feature_calibration(
        self, observation: RetinalObservation, *, step: int
    ) -> tuple[ExternalEvent, ...]:
        channels = (
            ("hs_optic_flow_left", self.mapping.left_hs_ids, observation.left_motion),
            ("hs_optic_flow_right", self.mapping.right_hs_ids, observation.right_motion),
            ("lc16_looming_left", self.mapping.left_lc16_ids, observation.looming),
            ("lc16_looming_right", self.mapping.right_lc16_ids, observation.looming),
        )
        return self._events(channels, step=step)

    def _events(
        self,
        channels: tuple[tuple[str, tuple[int, ...], float], ...],
        *,
        step: int,
    ) -> tuple[ExternalEvent, ...]:
        return tuple(
            ExternalEvent(
                step=step,
                neuron_ids=neuron_ids,
                voltages=tuple(
                    self.total_voltage * value / len(neuron_ids) for _ in neuron_ids
                ),
                channel=channel,
            )
            for channel, neuron_ids, value in channels
            if value > 0.0
        )
```

Normalize by bank size so total injected drive does not grow with population count. Encode only non-negative voltages. The API split is a hard boundary: photoreceptor mode cannot emit feature labels, and feature-calibration mode records that HS/LC16 activation bypasses upstream retina and optic-lobe computation.

- [ ] **Step 5: Add encoder tests and run gates**

Cover bank reversal, bilateral looming symmetry, exact replay, bank overlap rejection, unknown IDs, and absence of labels/coordinates from serialized `RetinalObservation`.

Run: `.venv/bin/pytest tests/test_retinal_interface.py -q`

Run: `.venv/bin/ruff check src/flybrain/retinal_interface.py tests/test_retinal_interface.py`

Run: `.venv/bin/mypy src/flybrain/retinal_interface.py`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/flybrain/retinal_interface.py tests/test_retinal_interface.py
git commit -m "feat: encode hemispheric optic flow"
```

### Task 4: Evidence-bound descending decoder and interventions

**Files:**
- Create: `src/flybrain/descending_interface.py`
- Create: `tests/test_descending_interface.py`

**Interfaces:**
- Consumes: registry-resolved DNa02, DNg13, and MDN IDs plus a tuple of emitted neuron IDs.
- Produces: `DescendingMap`, `DescendingDecoder.decode`, `DescendingActivity`, and `population_silence_mask`.

- [ ] **Step 1: Write failing laterality, cancellation, and retreat tests**

```python
def test_left_steering_is_ipsiversive_and_bilateral_activity_cancels() -> None:
    decoder = DescendingDecoder(DN_MAP, walking_drive=0.2)
    left = decoder.decode((10, 20))
    symmetric = decoder.decode((10, 11, 20, 21))
    assert left.command.turn > 0
    assert symmetric.command.turn == pytest.approx(0.0)


def test_bilateral_mdn_activity_reverses_without_turning() -> None:
    result = DescendingDecoder(DN_MAP, walking_drive=0.2).decode((30, 31))
    assert result.command.forward < 0
    assert result.command.turn == pytest.approx(0.0)
```

- [ ] **Step 2: Confirm RED**

Run: `.venv/bin/pytest tests/test_descending_interface.py -q`

Expected: import failure for `flybrain.descending_interface`.

- [ ] **Step 3: Implement maps, summaries, and bounded decoder**

```python
@dataclass(frozen=True)
class DescendingMap:
    d_na02_left: tuple[int, ...]
    d_na02_right: tuple[int, ...]
    d_ng13_left: tuple[int, ...]
    d_ng13_right: tuple[int, ...]
    mdn_left: tuple[int, ...]
    mdn_right: tuple[int, ...]


@dataclass(frozen=True)
class DescendingActivity:
    command: MotorCommand
    d_na02_left_rate: float
    d_na02_right_rate: float
    d_ng13_left_rate: float
    d_ng13_right_rate: float
    mdn_left_rate: float
    mdn_right_rate: float


class DescendingDecoder:
    version = "evidence-dn-v1"

    def decode(self, spikes: tuple[int, ...]) -> DescendingActivity:
        counts = Counter(spikes)
        rate = lambda ids: sum(counts[value] for value in ids) / len(ids)
        na_l, na_r = rate(self.mapping.d_na02_left), rate(self.mapping.d_na02_right)
        ng_l, ng_r = rate(self.mapping.d_ng13_left), rate(self.mapping.d_ng13_right)
        mdn_l, mdn_r = rate(self.mapping.mdn_left), rate(self.mapping.mdn_right)
        steering = 0.5 * ((na_l - na_r) + (ng_l - ng_r))
        retreat = 0.5 * (mdn_l + mdn_r)
        retreat_turn = 0.25 * (mdn_l - mdn_r)
        return DescendingActivity(
            command=MotorCommand(
                forward=float(np.clip(self.walking_drive - 2.0 * retreat, -1.0, 1.0)),
                turn=float(np.clip(steering + retreat_turn, -1.0, 1.0)),
            ),
            d_na02_left_rate=na_l,
            d_na02_right_rate=na_r,
            d_ng13_left_rate=ng_l,
            d_ng13_right_rate=ng_r,
            mdn_left_rate=mdn_l,
            mdn_right_rate=mdn_r,
        )
```

Normalize counts by population size, compute steering as `0.5 * ((DNa02_L-DNa02_R) + (DNg13_L-DNg13_R))`, retreat as the bilateral MDN mean, and retreat yaw as `0.25 * (MDN_L-MDN_R)`. Set forward command to `clip(walking_drive - 2 * retreat, -1, 1)` and turn to the clipped sum. Record the walking drive and all coefficients as `model_assumption` in benchmark output.

- [ ] **Step 4: Implement named silence masks**

```python
def population_silence_mask(
    graph: EventConnectome,
    mapping: DescendingMap,
    names: frozenset[str],
) -> NDArray[np.bool_]:
    populations = mapping.named_populations()
    unknown = sorted(names - populations.keys())
    if unknown:
        raise ValueError(f"unknown descending populations: {unknown}")
    selected = {value for name in names for value in populations[name]}
    if names and not selected:
        raise ValueError("requested descending population is empty")
    return np.fromiter(
        (int(neuron_id) in selected for neuron_id in graph.neuron_ids),
        dtype=np.bool_,
        count=graph.neuron_count,
    )
```

Reject unknown names and empty requested populations. Tests must prove that silencing does not inject opposite activity and cannot select neurons outside the declared map.

- [ ] **Step 5: Run gates and commit**

Run: `.venv/bin/pytest tests/test_descending_interface.py -q`

Run: `.venv/bin/ruff check src/flybrain/descending_interface.py tests/test_descending_interface.py`

Run: `.venv/bin/mypy src/flybrain/descending_interface.py`

Expected: all pass.

```bash
git add src/flybrain/descending_interface.py tests/test_descending_interface.py
git commit -m "feat: decode causal descending commands"
```

### Task 5: Multi-condition causal calibration benchmark

**Files:**
- Create: `src/flybrain/steering_benchmark.py`
- Create: `tests/test_steering_benchmark.py`

**Interfaces:**
- Consumes: `EventConnectome`, `ResolvedRegistry`, `DescendingMap`, Shiu parameters, and condition protocols.
- Produces: `run_causal_steering_benchmark(graph: EventConnectome, registry: ResolvedRegistry, *, steps: int, seed: int, snapshot: str = "unbound") -> SteeringBenchmarkResult` with exact replay, graph integrity, and causal comparisons.

- [ ] **Step 1: Write a failing synthetic calibration test**

```python
def test_dn_calibration_reverses_lesions_restores_and_replays() -> None:
    graph, registry = calibration_fixture()
    result = run_causal_steering_benchmark(
        graph, registry, steps=12, seed=7
    )
    by_name = {condition.name: condition for condition in result.conditions}
    assert by_name["d_na02_left"].turn_integral > 0
    assert by_name["d_na02_right"].turn_integral < 0
    assert by_name["d_na02_left_silenced"].turn_integral == 0
    assert by_name["d_na02_left_restored"].trace_digest == by_name["d_na02_left"].trace_digest
    assert result.replay_exact is True
    assert result.graph_unchanged is True
    assert result.calibration_passed is True
```

- [ ] **Step 2: Confirm RED**

Run: `.venv/bin/pytest tests/test_steering_benchmark.py::test_dn_calibration_reverses_lesions_restores_and_replays -q`

Expected: import failure for `flybrain.steering_benchmark`.

- [ ] **Step 3: Define explicit protocol and result models**

```python
class ProtocolName(StrEnum):
    DN_CALIBRATION = "dn_calibration"
    VISUAL_FEATURE_OPEN_LOOP = "visual_feature_open_loop"
    PHOTORECEPTOR_OPEN_LOOP = "photoreceptor_open_loop"
    VISUAL_FEATURE_CLOSED_LOOP = "visual_feature_closed_loop"


class ConditionResult(BaseModel, frozen=True):
    name: str
    protocol: ProtocolName
    stimulus_digest: str
    spike_digest: str
    command_digest: str
    trace_digest: str
    relevant_spike_counts: dict[str, int]
    turn_integral: float
    reverse_integral: float
    final_body: FlyBody
    silenced_populations: tuple[str, ...]
    upstream_visual_processing_bypassed: bool


class CausalComparison(BaseModel, frozen=True):
    name: str
    normal_condition: str
    intervention_condition: str
    metric: Literal["turn_integral", "reverse_integral", "relevant_spikes"]
    normal_value: float
    intervention_value: float
    absolute_effect: float
    relative_effect: float | None


class ClaimClassification(BaseModel, frozen=True):
    classification: Literal["positive", "null", "directionally_wrong", "underpowered"]
    evidence_kind: Literal["simulation_observation"]
    numerator: float
    denominator: float
    threshold: float
    lesion_effect: float
    upstream_visual_processing_bypassed: bool
    reasons: tuple[str, ...]


class SteeringBenchmarkResult(BaseModel, frozen=True):
    benchmark: Literal["biological-steering-v1"]
    snapshot: str
    graph_neurons: int
    graph_edges: int
    registry: ResolvedRegistry
    conditions: tuple[ConditionResult, ...]
    comparisons: tuple[CausalComparison, ...]
    calibration_passed: bool
    sensory_claims: dict[str, ClaimClassification]
    replay_exact: bool
    graph_unchanged: bool
    software_revision: str
    runtime_seconds: float
    peak_rss_bytes: int
```

In the test module, `calibration_fixture()` constructs one zero-edge `EventConnectome` with IDs
`(1, 2, 3, 4, 5, 6, 10, 11, 20, 21, 30, 31)` and a `ResolvedRegistry` whose populations are
R1-R6 left/right `1/2`, HS left/right `3/4`, LC16 left/right `5/6`, DNa02 left/right `10/11`,
DNg13 left/right `20/21`, and MDN left/right `30/31`. Use
`csr_array((12, 12), dtype=np.float32)`; direct calibration must therefore succeed without any
synaptic path and cannot be mistaken for visual propagation.

`ClaimClassification` is exactly `positive`, `null`, `directionally_wrong`, or `underpowered`; include the predeclared numerator, denominator, threshold, and evidence kind in each classification record.

- [ ] **Step 4: Implement deterministic direct-DN schedules**

Build schedules before simulation. Use the published Shiu voltage jump already exposed by `ShiuParameters`, repeat each target at fixed `refractory_steps + 1` intervals, and exempt only directly stimulated DN indices from refractory input suppression. Run DNa02 left/right, DNg13 left/right, MDN bilateral, MDN left/right, corresponding named silence conditions, restorations, and exact replays from independent initial state.

Never mutate the graph or reuse mutable `ShiuState` across conditions. Hash step, neuron IDs, voltages, relevant spikes, commands, and body states in canonical order.

- [ ] **Step 5: Implement causal calibration gates**

```python
calibration_passed = all(
    (
        left.turn_integral > 0,
        right.turn_integral < 0,
        abs(bilateral.turn_integral) <= 1e-9,
        abs(left_silenced.turn_integral) <= 1e-9,
        restored.trace_digest == left.trace_digest,
        mdn_bilateral.reverse_integral > 0,
        mdn_silenced.reverse_integral <= 1e-9,
        replay_exact,
        graph_unchanged,
    )
)
```

Store each individual gate and measured value, not only the aggregate Boolean. Add tests for mirrored signs, DNa02/DNg13 channel separation, MDN retreat, unknown lesions, graph mutation detection, and same-seed replay.

- [ ] **Step 6: Run gates and commit**

Run: `.venv/bin/pytest tests/test_steering_benchmark.py -q`

Run: `.venv/bin/ruff check src/flybrain/steering_benchmark.py tests/test_steering_benchmark.py`

Run: `.venv/bin/mypy src/flybrain/steering_benchmark.py`

Expected: all pass.

```bash
git add src/flybrain/steering_benchmark.py tests/test_steering_benchmark.py
git commit -m "feat: add causal DN calibration benchmark"
```

### Task 6: Photoreceptor and visual-feature protocols

**Files:**
- Modify: `src/flybrain/steering_benchmark.py`
- Modify: `tests/test_steering_benchmark.py`

**Interfaces:**
- Consumes: Task 3 retinal observations/encoder and Task 5 condition runner.
- Produces: R1-R6 luminance/contrast responses plus separately labeled HS optic-flow, LC16 looming, perturbation, holdout, and lesion outcomes.

- [ ] **Step 1: Add failing positive, null, mirrored, and holdout tests**

```python
def test_open_loop_positive_circuit_requires_expected_path() -> None:
    graph, registry = sensory_fixture(connected=True)
    result = run_causal_steering_benchmark(graph, registry, steps=40, seed=7)
    assert result.sensory_claims["hs_optic_flow"].classification == "positive"
    assert result.sensory_claims["hs_optic_flow"].lesion_effect > 0


def test_open_loop_null_is_reported_not_forced() -> None:
    graph, registry = sensory_fixture(connected=False)
    result = run_causal_steering_benchmark(graph, registry, steps=40, seed=7)
    assert result.sensory_claims["hs_optic_flow"].classification == "null"
    assert result.calibration_passed is True
```

`sensory_fixture(connected=True)` uses the calibration fixture IDs and a sparse graph with positive
edges `3 -> 10`, `4 -> 11`, and `5,6 -> 30,31`, each weight
`40.0`; the false variant uses identical metadata with zero edges. This makes the only difference
an inspectable feature-cell-to-DN path. Separate fixture edges from R1-R6 IDs `1/2` exercise the
photoreceptor protocol without relabeling it as optic flow.

- [ ] **Step 2: Confirm RED**

Run: `.venv/bin/pytest tests/test_steering_benchmark.py -q`

Expected: new sensory tests fail because only calibration protocols exist.

- [ ] **Step 3: Implement immutable stimulus families**

Create predeclared 40-step sequences for hemispheric luminance/contrast, left HS optic flow,
mirrored right HS optic flow, bilateral LC16 looming, bounded ±10% timing/amplitude perturbations
derived from the seed, and at least two holdout visual-disc geometries. Serialize each sequence and
hash it before neural execution. Mirroring must swap only homologous hemispheric banks; it must not
change the decoder or graph.

- [ ] **Step 4: Implement open-loop execution and classification**

Use `encode_photoreceptors` only for R1-R6 luminance/contrast and
`encode_feature_calibration` only for HS/LC16 protocols. Compare normal, matching-side lesion,
opposite-side lesion, bilateral lesion, restoration, replay, perturbation, and holdout conditions.
Every result includes `upstream_visual_processing_bypassed: bool`. Predeclare:

```python
MIN_RELEVANT_SPIKES = 2
MIN_DIRECTIONAL_EFFECT = 1e-6
MIN_LESION_FRACTION = 0.25
```

Classify `underpowered` when the normal relevant-DN spike count is below two; `null` when effect
magnitude is below threshold; `directionally_wrong` when mirror or expected sign is wrong; and
`positive` only when direction, matching lesion, restoration, replay, and every holdout agree. A
positive HS/LC16 path is named a feature-path result, never a retina-to-behavior result.

- [ ] **Step 5: Implement closed-loop execution**

For each world step: derive an anonymous-disc retinal observation from consecutive geometry, inject
either photoreceptor events or explicitly labeled HS/LC16 feature-calibration events, run one bounded
neural chunk, decode only DN spikes, step a fresh `ArenaWorld`, and feed the new body state into the
next retinal projection. The decoder receives no world object or coordinate. Start with the
declared target-independent walking drive from Task 4 and embed both that assumption and the selected
visual boundary in the result.

- [ ] **Step 6: Add perturbation and integrity tests**

Prove same seed reproduces stimulus and trace digests, different seed changes only the declared perturbation, lesions alter neural output without injecting an opposite command, holdout worlds were not used in thresholds, and all canonical CSR arrays plus neuron IDs retain their digest.

- [ ] **Step 7: Run gates and commit**

Run: `.venv/bin/pytest tests/test_retinal_interface.py tests/test_descending_interface.py tests/test_steering_benchmark.py -q`

Run: `.venv/bin/ruff check src/flybrain/retinal_interface.py src/flybrain/descending_interface.py src/flybrain/steering_benchmark.py tests/test_retinal_interface.py tests/test_descending_interface.py tests/test_steering_benchmark.py`

Run: `.venv/bin/mypy src/flybrain/retinal_interface.py src/flybrain/descending_interface.py src/flybrain/steering_benchmark.py`

Expected: all pass.

```bash
git add src/flybrain/steering_benchmark.py tests/test_steering_benchmark.py
git commit -m "feat: test visual steering causally"
```

### Task 7: Safe CLI and smoke-claim boundary

**Files:**
- Modify: `src/flybrain/cli.py`
- Modify: `src/flybrain/embodied_episode.py`
- Create: `tests/test_biological_steering_cli.py`
- Modify: `tests/test_embodied_episode.py`
- Modify: `tests/test_embodied_cli.py`

**Interfaces:**
- Consumes: `run_causal_steering_benchmark` and the v1 registry JSON.
- Produces: `flybrain experiment biological-steering SNAPSHOT --registry REGISTRY.json --output RESULT.json` and explicit smoke-only metadata for the legacy command.

- [ ] **Step 1: Write failing CLI safety and serialization tests**

```python
def test_cli_publishes_provenance_bound_biological_result(tmp_path: Path) -> None:
    snapshot_path = biological_snapshot(tmp_path)
    output = tmp_path / "result.json"
    result = runner.invoke(
        app,
        [
            "experiment", "biological-steering", str(snapshot_path),
            "--registry", str(FIXTURE_REGISTRY), "--steps", "12",
            "--output", str(output),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["benchmark"] == "biological-steering-v1"
    assert payload["registry"]["populations"]
    assert payload["calibration_passed"] is True
    assert set(payload["sensory_claims"]) == {
        "photoreceptor_response", "hs_optic_flow", "lc16_looming", "feature_closed_loop"
    }


def test_legacy_embodied_result_is_labeled_arbitrary_smoke(tmp_path: Path) -> None:
    snapshot_path = snapshot(tmp_path)
    output = tmp_path / "legacy.json"
    invocation = runner.invoke(
        app,
        ["experiment", "embodied-loop", str(snapshot_path), "--max-steps", "8", "--output", str(output)],
    )
    assert invocation.exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["interface_evidence"] == "arbitrary-smoke-only"
    assert payload["behavior_claim_allowed"] is False
```

- [ ] **Step 2: Confirm RED**

Run: `.venv/bin/pytest tests/test_biological_steering_cli.py tests/test_embodied_cli.py -q`

Expected: biological command missing and legacy fields absent.

- [ ] **Step 3: Add the dedicated CLI command**

```python
@experiment_app.command("biological-steering")
def biological_steering_command(
    snapshot: Path,
    registry: Annotated[Path, typer.Option("--registry")],
    output: Annotated[Path, typer.Option("--output")],
    steps: Annotated[int, typer.Option("--steps", min=1)] = 40,
    seed: Annotated[int, typer.Option("--seed")] = 7,
) -> None:
    output_final = output.resolve()
    snapshot_final = snapshot.resolve()
    registry_final = registry.resolve()
    if output_final in {snapshot_final, registry_final} or output_final.is_relative_to(snapshot_final):
        raise typer.BadParameter("--output must differ from inputs and be outside snapshot")
    if output_final.exists():
        raise typer.BadParameter(f"output already exists: {output_final}")
    declared = load_biological_registry(registry_final)
    resolved = resolve_biological_registry(declared, snapshot_final)
    graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(snapshot_final))
    resolved.validate_graph(graph)
    result = run_causal_steering_benchmark(
        graph, resolved, steps=steps, seed=seed, snapshot=str(snapshot_final)
    )
    output_final.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_final.parent) as temporary:
        stage = Path(temporary) / output_final.name
        with stage.open("xb") as stream:
            stream.write((result.model_dump_json(indent=2) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(stage, output_final)
    typer.echo(result.model_dump_json())
```

Resolve and validate the registry before loading the full graph. Reject output equal to or inside snapshot/registry paths, occupied output, missing identity files, and registry mismatch. Stage bytes in a temporary directory next to the destination, open with `xb`, flush, `os.fsync`, then `os.link`; emit JSON only after publication.

- [ ] **Step 4: Make the legacy claim boundary machine-readable**

Add `interface_evidence: Literal["arbitrary-smoke-only"]` and `behavior_claim_allowed: Literal[False]` to `EmbodiedEpisodeResult`; retain the old command for regression fixtures but make misuse visible in every artifact.

- [ ] **Step 5: Test failure atomicity**

Cover occupied output, snapshot alias, output inside snapshot, registry alias, malformed registry, expected-ID drift, graph missing a resolved neuron, execution exception, and no `.partial`/destination artifact after each failure.

- [ ] **Step 6: Run gates and commit**

Run: `.venv/bin/pytest tests/test_biological_steering_cli.py tests/test_embodied_cli.py tests/test_embodied_episode.py -q`

Run: `.venv/bin/ruff check src/flybrain/cli.py src/flybrain/embodied_episode.py tests/test_biological_steering_cli.py tests/test_embodied_cli.py tests/test_embodied_episode.py`

Run: `.venv/bin/mypy src/flybrain`

Expected: all pass.

```bash
git add src/flybrain/cli.py src/flybrain/embodied_episode.py tests/test_biological_steering_cli.py tests/test_embodied_cli.py tests/test_embodied_episode.py
git commit -m "feat: publish biological steering assays"
```

### Task 8: Full MaleCNS execution, documentation, and final verification

**Files:**
- Create: `tests/test_biological_steering_real.py`
- Create: `docs/data/male-cns-biological-steering.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: retained MaleCNS snapshot and all preceding tasks.
- Produces: a retained real benchmark JSON and an evidence-bounded report.

- [ ] **Step 1: Write the opt-in real-data test**

```python
def test_real_malecns_biological_steering_gate(tmp_path: Path) -> None:
    snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    output = tmp_path / "biological-steering.json"
    result = runner.invoke(
        app,
        [
            "experiment", "biological-steering", snapshot,
            "--registry", "data/registry/biological-interface-registry-v1.json",
            "--steps", "40", "--seed", "7", "--output", str(output),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["registry"]["dataset_id"] == "male-cns-v1.0-essential"
    assert payload["graph_neurons"] == 166_606
    assert payload["graph_edges"] == 6_240_402
    assert payload["calibration_passed"] is True
    assert payload["replay_exact"] is True
    assert payload["graph_unchanged"] is True
```

- [ ] **Step 2: Run the complete fast suite**

Run: `.venv/bin/pytest -q`

Run: `.venv/bin/ruff check .`

Run: `.venv/bin/mypy src`

Run: `git diff --check`

Expected: all pass; the real-data test may skip only when its environment variable is absent.

- [ ] **Step 3: Run the full retained MaleCNS benchmark**

Run:

```bash
FLYBRAIN_MALECNS_SNAPSHOT="$PWD/artifacts/male-cns-v1.0-w5" \
  .venv/bin/pytest tests/test_biological_steering_real.py -q

.venv/bin/flybrain experiment biological-steering \
  artifacts/male-cns-v1.0-w5 \
  --registry data/registry/biological-interface-registry-v1.json \
  --steps 40 \
  --seed 7 \
  --output artifacts/biological-steering-malecns-seed7.json
```

Expected: the exact registry, calibration, replay, and graph-integrity gates pass. Sensory claims may be any valid predeclared classification and must not be rewritten by hand.

- [ ] **Step 4: Audit the retained artifact**

Use a read-only verification script to assert every declared population ID, registry/annotation/snapshot digest, condition name, lesion, restoration pair, holdout, effect denominator, finite metric, and graph-integrity field. Recompute the output SHA-256 and record it. Confirm peak RSS remains sparse-scale and no dense `166606 x 166606` allocation exists in source or runtime paths.

- [ ] **Step 5: Write the evidence report**

Document exact command, git revision, snapshot and registry hashes, neuron/edge counts, runtime,
peak RSS, calibration effects, each real visual classification, holdout outcomes, lesions, null or
wrong-direction results, and the limitations: hemispheric rather than ommatidial photoreception,
explicit HS/LC16 upstream bypass, reduced body, tonic walking assumption, and no
intelligence/learning claim from this assay.

Update `README.md` with a short command and link to the report; keep the old embodied loop explicitly labeled as a smoke test.

- [ ] **Step 6: Re-run documentation-sensitive gates and commit**

Run: `.venv/bin/pytest -q`

Run: `.venv/bin/ruff check .`

Run: `.venv/bin/mypy src`

Run: `git diff --check`

Expected: all pass with the real test enabled in the recorded full run.

```bash
git add tests/test_biological_steering_real.py docs/data/male-cns-biological-steering.md README.md artifacts/biological-steering-malecns-seed7.json
git commit -m "docs: validate MaleCNS biological steering"
```

### Task 9: Independent review and completion audit

**Files:**
- Modify only files implicated by validated review findings.

**Interfaces:**
- Consumes: the complete branch diff, test evidence, specification, and retained artifact.
- Produces: reviewed code with no unresolved correctness or claim-boundary findings.

- [ ] **Step 1: Request a focused code and scientific-claim review**

Ask reviewers to check selector drift, evidence overreach, sensory label leakage, decoder access to world state, lesion validity, replay independence, graph mutation, threshold post-selection, dense allocation, and atomic publication.

- [ ] **Step 2: Reproduce each actionable finding**

For every finding, write or identify a failing test before changing production code. Reject findings only with concrete source/test evidence.

- [ ] **Step 3: Fix validated findings with focused commits**

Use the smallest behavior-preserving change that closes the reproduced defect. Do not weaken acceptance criteria or change real outcomes to obtain a positive claim.

- [ ] **Step 4: Perform requirement-by-requirement final verification**

Map every section of `docs/superpowers/specs/2026-09-16-biological-steering-design.md` to code, a test, and artifact evidence. Re-run the fast suite, Ruff, mypy, `git diff --check`, the real MaleCNS test, and the artifact audit. Record any remaining scientific stage as future work rather than claiming the overall fly-brain objective is complete.

- [ ] **Step 5: Commit review fixes and the final audit evidence**

```bash
git add src/flybrain/biological_registry.py src/flybrain/retinal_interface.py src/flybrain/descending_interface.py src/flybrain/steering_benchmark.py src/flybrain/cli.py tests/test_biological_registry.py tests/test_retinal_interface.py tests/test_descending_interface.py tests/test_steering_benchmark.py tests/test_biological_steering_cli.py tests/test_biological_steering_real.py docs/data/male-cns-biological-steering.md README.md
git commit -m "fix: close biological steering review findings"
```

Skip this commit only when review finds no actionable issue and the worktree is already clean.
