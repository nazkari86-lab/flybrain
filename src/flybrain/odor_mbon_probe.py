"""Isolated, non-behavioral odor-to-MBON recruitment diagnostics."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from flybrain.biological_registry import load_biological_registry, resolve_biological_registry
from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.mb_association import _software_revision
from flybrain.olfactory_interface import OlfactoryReceptorMap, task_odor_assignment
from flybrain.plastic_edge_binding import PlasticEdgeBinding, bind_manifest_to_graph
from flybrain.plastic_edge_registry import resolve_plastic_edge_manifests
from flybrain.provenance import snapshot_content_sha256
from flybrain.reinforcement_interface import ReinforcementInterface
from flybrain.shiu import ShiuParameters, ShiuState, poisson_voltage_events, simulate_shiu


class OdorMbonCondition(BaseModel, frozen=True):
    """Neural observations for one fixed source and plastic-overlay condition."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    source_voltage_events: int = Field(ge=0)
    source_event_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    neural_trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_spikes: int = Field(ge=0)
    kc_input_spike_counts: dict[int, int]
    mbon_spike_counts: dict[int, int]
    peak_post_step_voltage_mv: dict[int, float]
    effective_positive_weighted_spikes: dict[int, float]
    effective_negative_weighted_spikes: dict[int, float]
    artificial_intervention: str | None = None


def run_odor_mbon_condition(
    graph: EventConnectome,
    binding: PlasticEdgeBinding,
    *,
    name: str,
    source_ids: tuple[int, ...],
    target_ids: tuple[int, ...],
    steps: int,
    seed: int,
    parameters: ShiuParameters,
    max2_target_id: int | None = None,
) -> OdorMbonCondition:
    """Measure source-to-MBON recruitment; never interpret it as body behavior."""

    if not name or type(steps) is not int or steps <= 0:
        raise ValueError("condition name and positive steps are required")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not target_ids or len(target_ids) != len(set(target_ids)):
        raise ValueError("target IDs must be nonempty and unique")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source IDs must be unique")
    index_by_id = {int(value): index for index, value in enumerate(graph.neuron_ids)}
    if any(value not in index_by_id for value in target_ids):
        raise ValueError("target ID absent from graph")
    if any(value not in index_by_id for value in source_ids):
        raise ValueError("source ID absent from graph")
    if max2_target_id is not None and (
        max2_target_id not in target_ids or not np.any(binding.post_ids == max2_target_id)
    ):
        raise ValueError("max2 target must have declared KC edges and be measured")
    if binding.pre_ids.shape != binding.post_ids.shape or (
        binding.pre_ids.shape != binding.overlay.multipliers.shape
    ):
        raise ValueError("plastic binding endpoint and multiplier shapes differ")

    edge_multipliers = binding.overlay.multipliers.copy()
    if max2_target_id is not None:
        edge_multipliers[binding.post_ids == max2_target_id] = 2.0
    source_indices = np.asarray([index_by_id[value] for value in source_ids], dtype=np.int64)
    events = poisson_voltage_events(source_indices, steps=steps, params=parameters, seed=seed)
    source_digest = hashlib.sha256()
    for event_step, (indices, voltages) in sorted(events.items()):
        source_digest.update(np.asarray([event_step, len(indices)], dtype="<i8").tobytes())
        source_digest.update(indices.astype("<i8", copy=False).tobytes())
        source_digest.update(voltages.astype("<f4", copy=False).tobytes())
    state = ShiuState.initial(graph.neuron_count, params=parameters, seed=seed)
    counts: Counter[int] = Counter()
    peak = {target: parameters.rest_mv for target in target_ids}
    trace_digest = hashlib.sha256()
    for batch in simulate_shiu(
        graph,
        parameters,
        steps=steps,
        external_voltage_events=events,
        state=state,
        plastic_edge_indices=binding.overlay.edge_indices,
        plastic_edge_multipliers=edge_multipliers,
    ):
        trace_digest.update(np.asarray([batch.step, len(batch.neuron_ids)], dtype="<i8").tobytes())
        trace_digest.update(batch.neuron_ids.astype("<u8", copy=False).tobytes())
        counts.update(int(value) for value in batch.neuron_ids)
        for target in target_ids:
            peak[target] = max(peak[target], float(state.voltage_mv[index_by_id[target]]))

    pair_multipliers = {
        (int(pre), int(post)): float(multiplier)
        for pre, post, multiplier in zip(
            binding.pre_ids, binding.post_ids, edge_multipliers, strict=True
        )
    }
    incoming = graph.outgoing.transpose().tocsr()
    kc_input_spike_counts: dict[int, int] = {}
    positive: dict[int, float] = {}
    negative: dict[int, float] = {}
    for target in target_ids:
        kc_input_spike_counts[target] = sum(
            counts[int(pre)]
            for pre, post in zip(binding.pre_ids, binding.post_ids, strict=True)
            if int(post) == target
        )
        row = incoming[index_by_id[target]].tocoo()
        values = tuple(
            float(weight)
            * counts[int(graph.neuron_ids[pre_index])]
            * pair_multipliers.get((int(graph.neuron_ids[pre_index]), target), 1.0)
            for pre_index, weight in zip(row.col, row.data, strict=True)
        )
        positive[target] = float(sum(value for value in values if value > 0.0))
        negative[target] = float(sum(value for value in values if value < 0.0))
    return OdorMbonCondition(
        name=name,
        seed=seed,
        source_voltage_events=sum(len(indices) for indices, _ in events.values()),
        source_event_digest=source_digest.hexdigest(),
        neural_trace_digest=trace_digest.hexdigest(),
        source_spikes=sum(counts[value] for value in source_ids),
        kc_input_spike_counts=kc_input_spike_counts,
        mbon_spike_counts={target: counts[target] for target in target_ids},
        peak_post_step_voltage_mv=peak,
        effective_positive_weighted_spikes=positive,
        effective_negative_weighted_spikes=negative,
        artificial_intervention=(
            f"max2_kc_to_mbon_{max2_target_id}" if max2_target_id is not None else None
        ),
    )


class RetainedOdorMbonProbe(BaseModel, frozen=True):
    """Provenance and exact replay for an isolated retained-graph assay."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["retained-odor-mbon-probe-v1"] = "retained-odor-mbon-probe-v1"
    evidence_kind: Literal["isolated_neural_simulation"] = "isolated_neural_simulation"
    snapshot_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    learning_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    software_revision: str = Field(min_length=1)
    graph_neurons: int = Field(gt=0)
    graph_edges: int = Field(gt=0)
    graph_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    steps: int = Field(gt=0)
    seeds: tuple[int, ...]
    shiu_parameters: dict[str, float]
    source_ids: dict[str, tuple[int, ...]]
    target_ids: tuple[int, ...]
    dan_route_counts: dict[int, dict[str, int]]
    conditions: tuple[OdorMbonCondition, ...]
    replay_exact: bool
    paired_source_events_exact: bool
    graph_unchanged: bool
    overlay_unchanged: bool
    autonomous_behavior_claim_allowed: Literal[False] = False


def _graph_digest(graph: EventConnectome) -> str:
    digest = hashlib.sha256()
    for array in (
        graph.neuron_ids,
        graph.outgoing.data,
        graph.outgoing.indices,
        graph.outgoing.indptr,
    ):
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_retained_odor_mbon_probe(
    snapshot: Path,
    *,
    learning_registry_path: Path = Path("data/registry/autonomous-learning-registry-v1.json"),
    steps: int = 2_000,
    seeds: tuple[int, ...] = (7, 8, 9),
) -> RetainedOdorMbonProbe:
    """Replay-check a fixed odor panel while leaving embodied claims disabled."""

    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonempty non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    learning_registry = load_biological_registry(learning_registry_path)
    populations = resolve_biological_registry(learning_registry, snapshot)
    manifests = {
        manifest.name: manifest
        for manifest in resolve_plastic_edge_manifests(learning_registry, populations, snapshot)
    }
    graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(snapshot))
    before = _graph_digest(graph)
    binding = bind_manifest_to_graph(graph, manifests["kc_to_mbon"])
    initial_multipliers = binding.overlay.multipliers.copy()
    reinforcement = ReinforcementInterface.from_resolved_registry(populations)
    receptors = OlfactoryReceptorMap.from_snapshot(
        snapshot,
        graph=graph,
        olfactory_neuron_ids=populations.population("olfactory_sensory").neuron_ids,
    )
    assignment = task_odor_assignment()
    food_ids = tuple(
        sorted(
            receptors.side_channel_ids(assignment.food_cell_types, "L")
            + receptors.side_channel_ids(assignment.food_cell_types, "R")
        )
    )
    threat_ids = tuple(
        sorted(
            receptors.side_channel_ids(assignment.threat_cell_types, "L")
            + receptors.side_channel_ids(assignment.threat_cell_types, "R")
        )
    )
    if not food_ids or not threat_ids or set(food_ids) & set(threat_ids):
        raise ValueError("food and threat receptor sources must be nonempty and disjoint")
    target_ids = (519128, 524893)
    if any(target not in set(int(value) for value in graph.neuron_ids) for target in target_ids):
        raise ValueError("target MBON absent from graph")
    dan_pairs = manifests["dan_to_mbon"].edge_pairs
    appetitive = set(reinforcement.appetitive_dan_ids)
    aversive = set(reinforcement.aversive_dan_ids)
    dan_route_counts = {
        target: {
            "appetitive": sum(pre in appetitive for pre, post in dan_pairs if post == target),
            "aversive": sum(pre in aversive for pre, post in dan_pairs if post == target),
        }
        for target in target_ids
    }
    parameters = ShiuParameters(dt_ms=0.1, refractory_ms=2.0, synaptic_delay_ms=1.0)
    panel = (
        ("none", (), None),
        ("food", food_ids, None),
        ("threat", threat_ids, None),
        ("both", tuple(sorted(food_ids + threat_ids)), None),
        ("threat_max2", threat_ids, 519128),
    )
    conditions = tuple(
        run_odor_mbon_condition(
            graph,
            binding,
            name=name,
            source_ids=source_ids,
            target_ids=target_ids,
            steps=steps,
            seed=seed,
            parameters=parameters,
            max2_target_id=boost,
        )
        for seed in seeds
        for name, source_ids, boost in panel
    )
    replay = run_odor_mbon_condition(
        graph,
        binding,
        name="threat",
        source_ids=threat_ids,
        target_ids=target_ids,
        steps=steps,
        seed=seeds[0],
        parameters=parameters,
    )
    return RetainedOdorMbonProbe(
        snapshot_content_sha256=snapshot_content_sha256(snapshot),
        learning_registry_sha256=populations.registry_sha256,
        software_revision=_software_revision(),
        graph_neurons=graph.neuron_count,
        graph_edges=graph.edge_count,
        graph_digest=before,
        steps=steps,
        seeds=seeds,
        shiu_parameters={
            "dt_ms": parameters.dt_ms,
            "refractory_ms": parameters.refractory_ms,
            "synaptic_delay_ms": parameters.synaptic_delay_ms,
            "poisson_rate_hz": parameters.poisson_rate_hz,
            "poisson_voltage_scale": parameters.poisson_voltage_scale,
            "synapse_mv": parameters.synapse_mv,
            "threshold_mv": parameters.threshold_mv,
        },
        source_ids={"food": food_ids, "threat": threat_ids},
        target_ids=target_ids,
        dan_route_counts=dan_route_counts,
        conditions=conditions,
        replay_exact=replay == conditions[2],
        paired_source_events_exact=all(
            conditions[5 * index + 2].source_event_digest
            == conditions[5 * index + 4].source_event_digest
            for index in range(len(seeds))
        ),
        graph_unchanged=_graph_digest(graph) == before,
        overlay_unchanged=np.array_equal(binding.overlay.multipliers, initial_multipliers),
    )
