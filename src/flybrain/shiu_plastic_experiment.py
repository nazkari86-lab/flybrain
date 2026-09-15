"""Paired Shiu dynamics with an identity-bound plastic KC-to-MBON overlay."""

from __future__ import annotations

import hashlib
import math
import resource
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.mb_association import MBAssociationResult
from flybrain.plastic_graph import materialize_plastic_event_graph
from flybrain.plasticity import load_plastic_state
from flybrain.provenance import validate_state_identity
from flybrain.shiu import ShiuParameters, ShiuState, simulate_shiu


@dataclass(frozen=True)
class CueSchedule:
    """Deterministic direct voltage events for one cue ensemble."""

    events: dict[int, tuple[np.ndarray, np.ndarray]]
    steps: int


class CueConditionMetrics(BaseModel, frozen=True):
    """Observable metrics for one cue and one graph condition."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_spikes: int
    spikes_by_role: dict[str, int]
    target_spikes: int
    downstream_spikes: int
    activity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_voltage_mv: tuple[float, ...]
    target_conductance_mv: tuple[float, ...]
    cue_attributable_target_input: float


class CuePairMetrics(BaseModel, frozen=True):
    """Baseline/learned comparison for one cue."""

    baseline: CueConditionMetrics
    learned: CueConditionMetrics
    relative_cue_input_decrease: float
    relative_cue_input_drift: float
    target_spikes_changed: bool
    downstream_spikes_changed: bool
    activity_digest_changed: bool


class ShiuPlasticProtocol(BaseModel, frozen=True):
    """Complete deterministic protocol parameters."""

    duration_steps: int
    batch_size: int
    interval_steps: int
    delay_steps: int
    cue_a_ids: tuple[int, ...]
    cue_b_ids: tuple[int, ...]
    target_mbon_id: int
    parameters: dict[str, float]

    @property
    def steps(self) -> int:
        """Compatibility alias for the protocol duration."""

        return self.duration_steps


class ShiuPlasticIntegrationResult(BaseModel, frozen=True):
    """Auditable paired dynamic integration result."""

    benchmark: str
    snapshot: str
    association_path: str
    state_path: str
    dataset_id: str
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_neurons: int
    graph_edges: int
    matched_plastic_edges: int
    modified_plastic_edges: int
    plastic_baseline_absolute_weight: float
    plastic_effective_absolute_weight: float
    protocol: ShiuPlasticProtocol
    cue_a: CuePairMetrics
    cue_b: CuePairMetrics
    deterministic_replay_exact: bool
    baseline_graph_unchanged: bool
    passed: bool
    minimum_cue_a_relative_decrease: float
    maximum_cue_b_relative_drift: float
    software_revision: str
    runtime_seconds: float
    peak_rss_bytes: int


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _software_revision() -> str:
    from flybrain.mb_association import _software_revision as revision

    return revision()


def _graph_digest(graph: EventConnectome) -> str:
    digest = hashlib.sha256()
    for array in (graph.outgoing.data, graph.outgoing.indices, graph.outgoing.indptr):
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_cue_schedule(
    graph: EventConnectome,
    cue_ids: np.ndarray,
    params: ShiuParameters,
    *,
    batch_size: int = 8,
) -> CueSchedule:
    """Build staggered source-equivalent voltage events for sorted cue IDs."""

    params.validate()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cue_ids.ndim != 1 or cue_ids.size == 0:
        raise ValueError("cue_ids must be a non-empty one-dimensional array")
    if np.unique(cue_ids).size != cue_ids.size:
        raise ValueError("cue_ids must be unique")
    indices_by_id = {int(neuron_id): index for index, neuron_id in enumerate(graph.neuron_ids)}
    try:
        sorted_ids = np.sort(cue_ids.astype(np.uint64, copy=False))
        indices = np.asarray([indices_by_id[int(value)] for value in sorted_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"cue neuron is absent from graph: {error.args[0]}") from error
    amplitude = np.float32(params.synapse_mv * params.poisson_voltage_scale)
    interval = params.refractory_steps + 1
    events: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for batch_number, start in enumerate(range(0, indices.size, batch_size)):
        batch = indices[start : start + batch_size].copy()
        events[batch_number * interval] = (batch, np.full(batch.size, amplitude, dtype=np.float32))
    last_event = max(events)
    steps = last_event + params.delay_steps + math.ceil(
        5 * params.conductance_tau_ms / params.dt_ms
    ) + 1
    return CueSchedule(events=events, steps=steps)


def _run_condition(
    graph: EventConnectome,
    schedule: CueSchedule,
    cue_ids: np.ndarray,
    target_mbon_id: int,
    params: ShiuParameters,
    *,
    seed: int,
) -> CueConditionMetrics:
    state = ShiuState.initial(graph.neuron_count, params=params, seed=seed)
    index_by_id = {int(neuron_id): index for index, neuron_id in enumerate(graph.neuron_ids)}
    cue_indices = np.asarray([index_by_id[int(value)] for value in cue_ids], dtype=np.int64)
    target_index = index_by_id[target_mbon_id]
    refractory_exempt = np.zeros(graph.neuron_count, dtype=np.bool_)
    refractory_exempt[cue_indices] = True
    cue_index_set = set(int(value) for value in cue_indices)
    digest = hashlib.sha256()
    target_voltage: list[float] = []
    target_conductance: list[float] = []
    role_counts: dict[str, int] = {}
    target_spikes = 0
    downstream_spikes = 0
    total_spikes = 0
    cue_input = 0.0
    for batch in simulate_shiu(
        graph,
        params,
        steps=schedule.steps,
        external_voltage_events=schedule.events,
        seed=seed,
        state=state,
        refractory_exempt=refractory_exempt,
    ):
        digest.update(batch.step.to_bytes(8, byteorder="little", signed=False))
        digest.update(batch.neuron_ids.tobytes())
        target_voltage.append(float(state.voltage_mv[target_index]))
        target_conductance.append(float(state.conductance_mv[target_index]))
        total_spikes += int(batch.neuron_ids.size)
        for neuron_id_raw in batch.neuron_ids:
            neuron_id = int(neuron_id_raw)
            index = index_by_id[neuron_id]
            role = graph.roles[index]
            role_counts[role] = role_counts.get(role, 0) + 1
            if index == target_index:
                target_spikes += 1
            if role == "motor" or role == "descending":
                downstream_spikes += 1
        fired_cues = np.asarray(
            [index for index in batch.neuron_ids if index_by_id[int(index)] in cue_index_set],
            dtype=np.int64,
        )
        if fired_cues.size:
            propagated = graph.propagate_indices(
                np.asarray([index_by_id[int(value)] for value in fired_cues], dtype=np.int64),
                scale=params.synapse_mv,
            )
            cue_input += abs(float(propagated[target_index]))
    if not np.all(np.isfinite(np.asarray(target_voltage + target_conductance))):
        raise ValueError("Shiu trajectories contain non-finite values")
    return CueConditionMetrics(
        total_spikes=total_spikes,
        spikes_by_role=dict(sorted(role_counts.items())),
        target_spikes=target_spikes,
        downstream_spikes=downstream_spikes,
        activity_digest=digest.hexdigest(),
        target_voltage_mv=tuple(target_voltage),
        target_conductance_mv=tuple(target_conductance),
        cue_attributable_target_input=cue_input,
    )


def _pair_metrics(
    baseline: CueConditionMetrics,
    learned: CueConditionMetrics,
    *,
    trained: bool,
) -> CuePairMetrics:
    before = baseline.cue_attributable_target_input
    after = learned.cue_attributable_target_input
    if before <= 0:
        raise ValueError("cue input baseline must be positive")
    relative = round(abs(after - before) / before, 6)
    decrease = round((before - after) / before, 6)
    return CuePairMetrics(
        baseline=baseline,
        learned=learned,
        relative_cue_input_decrease=decrease if trained else 0.0,
        relative_cue_input_drift=0.0 if trained else relative,
        target_spikes_changed=baseline.target_spikes != learned.target_spikes,
        downstream_spikes_changed=baseline.downstream_spikes != learned.downstream_spikes,
        activity_digest_changed=baseline.activity_digest != learned.activity_digest,
    )


def run_shiu_plastic_integration(
    snapshot: Path,
    association_path: Path,
    state_path: Path,
    *,
    params: ShiuParameters | None = None,
    batch_size: int = 8,
    seed: int = 7,
) -> ShiuPlasticIntegrationResult:
    """Run paired live dynamics using a validated persisted plastic overlay."""

    started = __import__("time").perf_counter()
    association = MBAssociationResult.model_validate_json(
        association_path.read_text(encoding="utf-8")
    )
    state_digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    if state_digest != association.state_sha256:
        raise ValueError("state checksum does not match association metrics")
    edges, _, identity = load_plastic_state(state_path)
    validate_state_identity(snapshot, identity)
    if (
        identity.dataset_id != association.dataset_id
        or identity.source_manifest_sha256 != association.source_manifest_sha256
        or identity.snapshot_sha256 != association.snapshot_sha256
    ):
        raise ValueError("association and state identity mismatch")
    if not association.passed:
        raise ValueError("association benchmark did not pass acceptance")

    active_params = params or ShiuParameters()
    active_params.validate()
    graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(snapshot))
    baseline_digest = _graph_digest(graph)
    learned, graph_metrics = materialize_plastic_event_graph(graph, edges)
    target = association.target_mbon_id
    graph_ids = set(int(value) for value in graph.neuron_ids)
    cue_a = np.asarray(association.cue_a_ids, dtype=np.uint64)
    cue_b = np.asarray(association.cue_b_ids, dtype=np.uint64)
    if (
        target not in graph_ids
        or not set(map(int, cue_a)).issubset(graph_ids)
        or not set(map(int, cue_b)).issubset(graph_ids)
    ):
        raise ValueError("association target or cue ID is absent from graph")
    schedule_a = build_cue_schedule(graph, cue_a, active_params, batch_size=batch_size)
    schedule_b = build_cue_schedule(graph, cue_b, active_params, batch_size=batch_size)
    if schedule_a.steps != schedule_b.steps:
        raise ValueError("cue schedules have different durations")
    protocol = ShiuPlasticProtocol(
        duration_steps=schedule_a.steps,
        batch_size=batch_size,
        interval_steps=active_params.refractory_steps + 1,
        delay_steps=active_params.delay_steps,
        cue_a_ids=tuple(int(value) for value in cue_a),
        cue_b_ids=tuple(int(value) for value in cue_b),
        target_mbon_id=target,
        parameters={key: float(value) for key, value in vars(active_params).items()},
    )
    a_baseline = _run_condition(graph, schedule_a, cue_a, target, active_params, seed=seed)
    a_learned = _run_condition(learned, schedule_a, cue_a, target, active_params, seed=seed)
    b_baseline = _run_condition(graph, schedule_b, cue_b, target, active_params, seed=seed)
    b_learned = _run_condition(learned, schedule_b, cue_b, target, active_params, seed=seed)
    cue_a_metrics = _pair_metrics(a_baseline, a_learned, trained=True)
    cue_b_metrics = _pair_metrics(b_baseline, b_learned, trained=False)
    replay = _run_condition(learned, schedule_a, cue_a, target, active_params, seed=seed)
    replay_exact = replay == a_learned
    baseline_unchanged = _graph_digest(graph) == baseline_digest
    passed = (
        cue_a_metrics.relative_cue_input_decrease >= 0.10
        and cue_b_metrics.relative_cue_input_drift <= 1e-6
        and replay_exact
        and baseline_unchanged
    )
    if not passed:
        raise ValueError("Shiu plastic integration acceptance failed")
    return ShiuPlasticIntegrationResult(
        benchmark="shiu-plastic-integration-v1",
        snapshot=str(snapshot.resolve()),
        association_path=str(association_path.resolve()),
        state_path=str(state_path.resolve()),
        dataset_id=identity.dataset_id,
        source_manifest_sha256=identity.source_manifest_sha256,
        snapshot_sha256=identity.snapshot_sha256,
        graph_neurons=graph.neuron_count,
        graph_edges=graph.edge_count,
        matched_plastic_edges=graph_metrics.matched_edges,
        modified_plastic_edges=graph_metrics.modified_edges,
        plastic_baseline_absolute_weight=graph_metrics.baseline_absolute_weight,
        plastic_effective_absolute_weight=graph_metrics.effective_absolute_weight,
        protocol=protocol,
        cue_a=cue_a_metrics,
        cue_b=cue_b_metrics,
        deterministic_replay_exact=replay_exact,
        baseline_graph_unchanged=baseline_unchanged,
        passed=passed,
        minimum_cue_a_relative_decrease=0.10,
        maximum_cue_b_relative_drift=1e-6,
        software_revision=_software_revision(),
        runtime_seconds=__import__("time").perf_counter() - started,
        peak_rss_bytes=_peak_rss_bytes(),
    )
