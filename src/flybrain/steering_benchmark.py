"""Deterministic causal calibration of the biological descending interface."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from dataclasses import asdict
from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from flybrain.biological_registry import ResolvedRegistry
from flybrain.descending_interface import (
    DescendingDecoder,
    DescendingMap,
    population_silence_mask,
)
from flybrain.embodied_world import ArenaConfig, ArenaWorld, FlyBody
from flybrain.graph import EventConnectome
from flybrain.shiu import ShiuParameters, simulate_shiu


class ProtocolName(StrEnum):
    DN_CALIBRATION = "dn_calibration"
    VISUAL_FEATURE_OPEN_LOOP = "visual_feature_open_loop"
    PHOTORECEPTOR_OPEN_LOOP = "photoreceptor_open_loop"
    VISUAL_FEATURE_CLOSED_LOOP = "visual_feature_closed_loop"


class ConditionResult(BaseModel, frozen=True):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    protocol: ProtocolName
    stimulus_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    spike_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    command_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    relevant_spike_counts: dict[str, int]
    turn_integral: float
    reverse_integral: float
    final_body: FlyBody
    silenced_populations: tuple[str, ...]
    upstream_visual_processing_bypassed: bool


class CausalComparison(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    name: str
    normal_condition: str
    intervention_condition: str
    metric: Literal["turn_integral", "reverse_integral", "relevant_spikes"]
    normal_value: float
    intervention_value: float
    absolute_effect: float
    relative_effect: float | None


class ClaimClassification(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    classification: Literal["positive", "null", "directionally_wrong", "underpowered"]
    evidence_kind: Literal["simulation_observation"]
    numerator: float
    denominator: float
    threshold: float
    lesion_effect: float
    upstream_visual_processing_bypassed: bool
    reasons: tuple[str, ...]


class SteeringBenchmarkResult(BaseModel, frozen=True):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    benchmark: Literal["biological-steering-v1"]
    snapshot: str
    graph_neurons: int
    graph_edges: int
    registry: ResolvedRegistry
    conditions: tuple[ConditionResult, ...]
    comparisons: tuple[CausalComparison, ...]
    calibration_gates: dict[str, bool]
    calibration_passed: bool
    sensory_claims: dict[str, ClaimClassification]
    replay_exact: bool
    graph_unchanged: bool
    software_revision: str
    runtime_seconds: float
    peak_rss_bytes: int


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


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


def _descending_map(registry: ResolvedRegistry) -> DescendingMap:
    return DescendingMap(
        *(registry.population(name).neuron_ids for name in (
            "d_na02_left", "d_na02_right", "d_ng13_left", "d_ng13_right",
            "mdn_left", "mdn_right",
        ))
    )


def _condition(
    graph: EventConnectome,
    mapping: DescendingMap,
    name: str,
    targets: tuple[int, ...],
    silenced: frozenset[str],
    *,
    steps: int,
    seed: int,
) -> ConditionResult:
    if steps < 1:
        raise ValueError("steps must be positive")
    index_by_id = {int(value): index for index, value in enumerate(graph.neuron_ids)}
    available = set(index_by_id)
    if any(value not in available for value in targets):
        raise ValueError("direct calibration target is absent from graph")
    decoder = DescendingDecoder(mapping)
    params = ShiuParameters()
    interval = params.refractory_steps + 1
    target_indices = np.array([index_by_id[value] for value in targets], dtype=np.int64)
    amplitude = np.float32(params.synapse_mv * params.poisson_voltage_scale)
    voltage_events = {
        step: (
            target_indices.copy(),
            np.full(target_indices.size, amplitude, dtype=np.float32),
        )
        for step in range(0, steps, interval)
    }
    refractory_exempt = np.zeros(graph.neuron_count, dtype=np.bool_)
    refractory_exempt[target_indices] = True
    silence_mask = population_silence_mask(graph, mapping, silenced)
    spike_steps: list[tuple[int, tuple[int, ...]]] = []
    commands: list[tuple[float, float]] = []
    bodies: list[FlyBody] = []
    world = ArenaWorld(
        ArenaConfig(10.0, 10.0, 0.1, 0.2),
        FlyBody(5.0, 5.0, 0.0, 0.0, 0.0, 1.0, (False,) * 6),
        food=(1.0, 1.0),
        threat=(9.0, 9.0),
    )
    batches = simulate_shiu(
        graph,
        params,
        steps=steps,
        external_voltage_events=voltage_events,
        seed=seed,
        silenced=silence_mask,
        refractory_exempt=refractory_exempt,
    )
    for batch in batches:
        step = batch.step
        spikes = tuple(int(value) for value in batch.neuron_ids)
        activity = decoder.decode(spikes)
        world_step = world.step(activity.command)
        spike_steps.append((step, spikes))
        commands.append((activity.command.forward, activity.command.turn))
        bodies.append(world_step.body)
    counts = {
        population: sum(
            neuron_id in spikes
            for _, spikes in spike_steps
            for neuron_id in mapping.named_populations()[population]
        )
        for population in mapping.named_populations()
    }
    return ConditionResult(
        name=name,
        protocol=ProtocolName.DN_CALIBRATION,
        stimulus_digest=_digest(
            [
                (step, targets, [float(amplitude)] * len(targets))
                for step in range(0, steps, interval)
            ]
        ),
        spike_digest=_digest(spike_steps),
        command_digest=_digest(commands),
        trace_digest=_digest(
            {
                "spikes": spike_steps,
                "commands": commands,
                "bodies": [asdict(body) for body in bodies],
            }
        ),
        relevant_spike_counts=counts,
        turn_integral=sum(command[1] for command in commands),
        reverse_integral=sum(max(0.0, -command[0]) for command in commands),
        final_body=bodies[-1],
        silenced_populations=tuple(sorted(silenced)),
        upstream_visual_processing_bypassed=True,
    )


def run_causal_steering_benchmark(
    graph: EventConnectome,
    registry: ResolvedRegistry,
    *,
    steps: int,
    seed: int,
    snapshot: str = "unbound",
) -> SteeringBenchmarkResult:
    """Run direct DN calibration with lesions, restoration, and exact replay."""

    started = time.perf_counter()
    registry.validate_graph(graph)
    before = _graph_digest(graph)
    mapping = _descending_map(registry)
    schedules: tuple[tuple[str, tuple[int, ...], frozenset[str]], ...] = (
        ("d_na02_left", mapping.d_na02_left, frozenset()),
        ("d_na02_right", mapping.d_na02_right, frozenset()),
        ("d_ng13_left", mapping.d_ng13_left, frozenset()),
        ("d_ng13_right", mapping.d_ng13_right, frozenset()),
        (
            "d_na02_bilateral",
            mapping.d_na02_left + mapping.d_na02_right,
            frozenset(),
        ),
        ("mdn_bilateral", mapping.mdn_left + mapping.mdn_right, frozenset()),
        ("mdn_left", mapping.mdn_left, frozenset()),
        ("mdn_right", mapping.mdn_right, frozenset()),
        ("d_na02_left_silenced", mapping.d_na02_left, frozenset({"d_na02_left"})),
        (
            "mdn_bilateral_silenced",
            mapping.mdn_left + mapping.mdn_right,
            frozenset({"mdn_left", "mdn_right"}),
        ),
        ("d_na02_left_restored", mapping.d_na02_left, frozenset()),
    )
    conditions = tuple(
        _condition(graph, mapping, name, targets, silenced, steps=steps, seed=seed)
        for name, targets, silenced in schedules
    )
    replay = _condition(
        graph,
        mapping,
        "replay",
        mapping.d_na02_left,
        frozenset(),
        steps=steps,
        seed=seed,
    )
    baseline = next(item for item in conditions if item.name == "d_na02_left")
    restored = next(item for item in conditions if item.name == "d_na02_left_restored")
    mdn = next(item for item in conditions if item.name == "mdn_bilateral")
    mdn_silenced = next(item for item in conditions if item.name == "mdn_bilateral_silenced")
    left_silenced = next(item for item in conditions if item.name == "d_na02_left_silenced")

    def metric_value(result: ConditionResult, metric: str) -> float:
        if metric == "relevant_spikes":
            return float(sum(result.relevant_spike_counts.values()))
        return float(getattr(result, metric))

    comparison_inputs: tuple[
        tuple[
            str,
            ConditionResult,
            ConditionResult,
            Literal["turn_integral", "reverse_integral", "relevant_spikes"],
        ],
        ...,
    ] = (
        ("left_steering_silence", baseline, left_silenced, "turn_integral"),
        ("mdn_retreat_silence", mdn, mdn_silenced, "reverse_integral"),
    )
    comparisons = tuple(
        CausalComparison(
            name=name,
            normal_condition=normal.name,
            intervention_condition=intervention.name,
            metric=metric,
            normal_value=metric_value(normal, metric),
            intervention_value=metric_value(intervention, metric),
            absolute_effect=metric_value(normal, metric) - metric_value(intervention, metric),
            relative_effect=None,
        )
        for name, normal, intervention, metric in comparison_inputs
    )
    graph_unchanged = before == _graph_digest(graph)
    right = next(item for item in conditions if item.name == "d_na02_right")
    bilateral = next(item for item in conditions if item.name == "d_na02_bilateral")
    calibration_gates = {
        "left_ipsiversive": baseline.turn_integral > 0,
        "right_ipsiversive": right.turn_integral < 0,
        "bilateral_steering_cancels": abs(bilateral.turn_integral) <= 1e-9,
        "left_silence_removes_turn": abs(left_silenced.turn_integral) <= 1e-9,
        "restoration_exact": restored.trace_digest == baseline.trace_digest,
        "bilateral_mdn_retreats": mdn.reverse_integral > 0,
        "mdn_silence_removes_retreat": mdn_silenced.reverse_integral <= 1e-9,
        "replay_exact": replay.trace_digest == baseline.trace_digest,
        "graph_unchanged": graph_unchanged,
    }
    calibration_passed = all(calibration_gates.values())
    return SteeringBenchmarkResult(
        benchmark="biological-steering-v1",
        snapshot=snapshot,
        graph_neurons=graph.neuron_count,
        graph_edges=graph.edge_count,
        registry=registry,
        conditions=conditions,
        comparisons=comparisons,
        calibration_gates=calibration_gates,
        calibration_passed=calibration_passed,
        sensory_claims={},
        replay_exact=replay.trace_digest == baseline.trace_digest,
        graph_unchanged=graph_unchanged,
        software_revision="source",
        runtime_seconds=time.perf_counter() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
