"""Deterministic causal calibration of the biological descending interface."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from dataclasses import asdict
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from flybrain.biological_registry import ResolvedRegistry
from flybrain.descending_interface import (
    DescendingDecoder,
    DescendingMap,
    population_silence_mask,
)
from flybrain.embodied_interfaces import ExternalEvent
from flybrain.embodied_world import ArenaConfig, ArenaWorld, FlyBody
from flybrain.graph import EventConnectome
from flybrain.retinal_interface import (
    RetinalObservation,
    VisualDisc,
    VisualInterfaceEncoder,
    VisualInterfaceMap,
    observe_retina,
)
from flybrain.shiu import ShiuParameters, ShiuState, simulate_shiu

MIN_RELEVANT_SPIKES = 1
MIN_DIRECTIONAL_EFFECT = 1e-6
MIN_LESION_FRACTION = 0.25


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


def _software_revision() -> str:
    from flybrain.mb_association import _software_revision as revision

    return revision()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _descending_map(registry: ResolvedRegistry) -> DescendingMap:
    return DescendingMap(
        *(registry.population(name).neuron_ids for name in (
            "d_na02_left", "d_na02_right", "d_ng13_left", "d_ng13_right",
            "mdn_left", "mdn_right",
        ))
    )


def _visual_map(registry: ResolvedRegistry) -> VisualInterfaceMap:
    return VisualInterfaceMap(
        left_r1_r6_ids=registry.population("visual_r1_r6_left").neuron_ids,
        right_r1_r6_ids=registry.population("visual_r1_r6_right").neuron_ids,
        left_hs_ids=registry.population("hs_left").neuron_ids,
        right_hs_ids=registry.population("hs_right").neuron_ids,
        left_lc16_ids=registry.population("lc16_left").neuron_ids,
        right_lc16_ids=registry.population("lc16_right").neuron_ids,
    )


def _voltage_schedule(
    graph: EventConnectome,
    events: tuple[ExternalEvent, ...],
) -> tuple[
    dict[int, tuple[NDArray[np.int64], NDArray[np.float32]]],
    NDArray[np.bool_],
]:
    index_by_id = {int(value): index for index, value in enumerate(graph.neuron_ids)}
    grouped: dict[int, list[tuple[int, float]]] = {}
    direct_mask = np.zeros(graph.neuron_count, dtype=np.bool_)
    for event in events:
        pairs = grouped.setdefault(event.step, [])
        for neuron_id, voltage in zip(event.neuron_ids, event.voltages, strict=True):
            if neuron_id not in index_by_id:
                raise ValueError(f"visual event ID absent from graph: {neuron_id}")
            index = index_by_id[neuron_id]
            direct_mask[index] = True
            pairs.append((index, voltage))
    schedule = {
        step: (
            np.array([index for index, _ in pairs], dtype=np.int64),
            np.array([voltage for _, voltage in pairs], dtype=np.float32),
        )
        for step, pairs in grouped.items()
    }
    return schedule, direct_mask


def _sensory_condition(
    graph: EventConnectome,
    mapping: DescendingMap,
    name: str,
    events: tuple[ExternalEvent, ...],
    silenced: frozenset[str],
    *,
    steps: int,
    seed: int,
    protocol: ProtocolName,
    bypassed: bool,
) -> ConditionResult:
    voltage_events, refractory_exempt = _voltage_schedule(graph, events)
    silence_mask = population_silence_mask(graph, mapping, silenced)
    decoder = DescendingDecoder(mapping)
    spike_steps: list[tuple[int, tuple[int, ...]]] = []
    commands: list[tuple[float, float]] = []
    bodies: list[FlyBody] = []
    world = ArenaWorld(
        ArenaConfig(10.0, 10.0, 0.1, 0.2),
        FlyBody(5.0, 5.0, 0.0, 0.0, 0.0, 1.0, (False,) * 6),
        food=(1.0, 1.0),
        threat=(9.0, 9.0),
    )
    for batch in simulate_shiu(
        graph,
        ShiuParameters(),
        steps=steps,
        external_voltage_events=voltage_events,
        seed=seed,
        silenced=silence_mask,
        refractory_exempt=refractory_exempt,
    ):
        spikes = tuple(int(value) for value in batch.neuron_ids)
        activity = decoder.decode(spikes)
        body = world.step(activity.command).body
        spike_steps.append((batch.step, spikes))
        commands.append((activity.command.forward, activity.command.turn))
        bodies.append(body)
    populations = mapping.named_populations()
    counts = {
        population: sum(
            neuron_id in spikes
            for _, spikes in spike_steps
            for neuron_id in neuron_ids
        )
        for population, neuron_ids in populations.items()
    }
    stimulus = [asdict(event) for event in events]
    trace = {
        "stimulus": stimulus,
        "spikes": spike_steps,
        "commands": commands,
        "bodies": [asdict(body) for body in bodies],
    }
    return ConditionResult(
        name=name,
        protocol=protocol,
        stimulus_digest=_digest(stimulus),
        spike_digest=_digest(spike_steps),
        command_digest=_digest(commands),
        trace_digest=_digest(trace),
        relevant_spike_counts=counts,
        turn_integral=sum(command[1] for command in commands),
        reverse_integral=sum(max(0.0, -command[0]) for command in commands),
        final_body=bodies[-1],
        silenced_populations=tuple(sorted(silenced)),
        upstream_visual_processing_bypassed=bypassed,
    )


def _repeat_visual_events(
    encoder: VisualInterfaceEncoder,
    observation: RetinalObservation,
    *,
    steps: int,
    feature_calibration: bool,
    first_step: int = 0,
) -> tuple[ExternalEvent, ...]:
    encode = (
        encoder.encode_feature_calibration
        if feature_calibration
        else encoder.encode_photoreceptors
    )
    return tuple(
        event
        for step in range(first_step, steps)
        for event in encode(observation, step=step)
    )


def _directional_classification(
    normal: ConditionResult,
    mirror: ConditionResult,
    lesion: ConditionResult,
    restored: ConditionResult,
    replay: ConditionResult,
    holdouts: tuple[ConditionResult, ...],
    *,
    relevant_population: str,
) -> ClaimClassification:
    relevant_spikes = normal.relevant_spike_counts[relevant_population]
    effect = normal.turn_integral
    lesion_effect = effect - lesion.turn_integral
    denominator = max(abs(effect), MIN_DIRECTIONAL_EFFECT)
    lesion_fraction = lesion_effect / denominator
    classification: Literal["positive", "null", "directionally_wrong", "underpowered"]
    if relevant_spikes == 0:
        classification = "null"
        reasons = ("no relevant descending spikes after a fully delivered stimulus",)
    elif relevant_spikes < MIN_RELEVANT_SPIKES:
        classification = "underpowered"
        reasons = ("relevant descending spike count is below the fixed threshold",)
    elif effect <= MIN_DIRECTIONAL_EFFECT or mirror.turn_integral >= -MIN_DIRECTIONAL_EFFECT:
        classification = "directionally_wrong"
        reasons = ("normal and mirrored turn signs do not match the predeclared signs",)
    elif (
        lesion_fraction < MIN_LESION_FRACTION
        or restored.trace_digest != normal.trace_digest
        or replay.trace_digest != normal.trace_digest
        or any(item.turn_integral <= MIN_DIRECTIONAL_EFFECT for item in holdouts)
    ):
        classification = "null"
        reasons = ("one or more lesion, restoration, replay, or holdout gates failed",)
    else:
        classification = "positive"
        reasons = ("direction, lesion, restoration, replay, and holdout gates passed",)
    return ClaimClassification(
        classification=classification,
        evidence_kind="simulation_observation",
        numerator=effect,
        denominator=denominator,
        threshold=MIN_DIRECTIONAL_EFFECT,
        lesion_effect=lesion_effect,
        upstream_visual_processing_bypassed=True,
        reasons=reasons,
    )


def _run_sensory_protocols(
    graph: EventConnectome,
    registry: ResolvedRegistry,
    mapping: DescendingMap,
    *,
    steps: int,
    seed: int,
) -> tuple[tuple[ConditionResult, ...], dict[str, ClaimClassification]]:
    visual = VisualInterfaceEncoder(_visual_map(registry))
    left_motion = RetinalObservation(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    right_motion = RetinalObservation(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    body = FlyBody(5.0, 5.0, 0.0, 0.0, 0.0, 1.0, (False,) * 6)
    holdout_observations = tuple(
        observe_retina(body, previous, current)
        for previous, current in (
            ((VisualDisc(8.0, 5.5, 0.4),), (VisualDisc(6.0, 7.0, 0.4),)),
            ((VisualDisc(9.0, 6.0, 0.3),), (VisualDisc(6.0, 8.0, 0.3),)),
        )
    )
    holdout_motion = tuple(
        RetinalObservation(0.0, 0.0, 0.0, 0.0, item.left_motion, 0.0, 0.0)
        for item in holdout_observations
    )

    def run(
        name: str,
        observation: RetinalObservation,
        silenced: frozenset[str] = frozenset(),
        *,
        feature: bool = True,
        first_step: int = 0,
    ) -> ConditionResult:
        return _sensory_condition(
            graph,
            mapping,
            name,
            _repeat_visual_events(
                visual,
                observation,
                steps=steps,
                feature_calibration=feature,
                first_step=first_step,
            ),
            silenced,
            steps=steps,
            seed=seed,
            protocol=(
                ProtocolName.VISUAL_FEATURE_OPEN_LOOP
                if feature
                else ProtocolName.PHOTORECEPTOR_OPEN_LOOP
            ),
            bypassed=feature,
        )

    hs_normal = run("hs_left", left_motion)
    hs_mirror = run("hs_right_mirrored", right_motion)
    hs_lesion = run("hs_left_matching_lesion", left_motion, frozenset({"d_na02_left"}))
    hs_opposite_lesion = run(
        "hs_left_opposite_lesion", left_motion, frozenset({"d_na02_right"})
    )
    hs_bilateral_lesion = run(
        "hs_left_bilateral_lesion",
        left_motion,
        frozenset({"d_na02_left", "d_na02_right"}),
    )
    hs_restored = run("hs_left_restored", left_motion)
    hs_replay = run("hs_left_replay", left_motion)
    perturbation_rng = np.random.default_rng(seed)
    perturbation = 0.8 * float(perturbation_rng.uniform(0.9, 1.1))
    timing_bound = max(1, round(steps * 0.1))
    timing_shift = min(
        steps - 1,
        timing_bound
        + int(perturbation_rng.integers(-timing_bound, timing_bound + 1)),
    )
    hs_perturbed = run(
        "hs_left_perturbed",
        RetinalObservation(0.0, 0.0, 0.0, 0.0, perturbation, 0.0, 0.0),
        first_step=timing_shift,
    )
    hs_holdouts = tuple(
        run(f"hs_left_holdout_{index}", observation)
        for index, observation in enumerate(holdout_motion, start=1)
    )
    looming = RetinalObservation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    lc16 = run("lc16_bilateral", looming)
    lc16_lesion = run(
        "lc16_bilateral_lesion",
        looming,
        frozenset({"mdn_left", "mdn_right"}),
    )
    photoreceptor = run(
        "photoreceptor_left",
        RetinalObservation(1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0),
        feature=False,
    )
    conditions = (
        hs_normal,
        hs_mirror,
        hs_lesion,
        hs_opposite_lesion,
        hs_bilateral_lesion,
        hs_restored,
        hs_replay,
        hs_perturbed,
        *hs_holdouts,
        lc16,
        lc16_lesion,
        photoreceptor,
    )
    hs_claim = _directional_classification(
        hs_normal,
        hs_mirror,
        hs_lesion,
        hs_restored,
        hs_replay,
        hs_holdouts,
        relevant_population="d_na02_left",
    )
    lc16_spikes = lc16.relevant_spike_counts["mdn_left"] + lc16.relevant_spike_counts["mdn_right"]
    lc16_effect = lc16.reverse_integral
    lc16_lesion_effect = lc16_effect - lc16_lesion.reverse_integral
    lc16_claim = ClaimClassification(
        classification=(
            "positive"
            if lc16_spikes >= MIN_RELEVANT_SPIKES
            and lc16_effect > MIN_DIRECTIONAL_EFFECT
            and lc16_lesion_effect / max(lc16_effect, MIN_DIRECTIONAL_EFFECT)
            >= MIN_LESION_FRACTION
            else "null"
        ),
        evidence_kind="simulation_observation",
        numerator=lc16_effect,
        denominator=max(lc16_effect, MIN_DIRECTIONAL_EFFECT),
        threshold=MIN_DIRECTIONAL_EFFECT,
        lesion_effect=lc16_lesion_effect,
        upstream_visual_processing_bypassed=True,
        reasons=("bilateral looming feature path and matching MDN lesion were evaluated",),
    )
    photo_spikes = photoreceptor.relevant_spike_counts["d_na02_left"]
    photo_claim = ClaimClassification(
        classification="positive" if photo_spikes >= MIN_RELEVANT_SPIKES else "null",
        evidence_kind="simulation_observation",
        numerator=float(photo_spikes),
        denominator=float(max(photo_spikes, 1)),
        threshold=float(MIN_RELEVANT_SPIKES),
        lesion_effect=0.0,
        upstream_visual_processing_bypassed=False,
        reasons=("R1-R6 received only luminance and contrast channels",),
    )
    return conditions, {
        "photoreceptor_response": photo_claim,
        "hs_optic_flow": hs_claim,
        "lc16_looming": lc16_claim,
    }


def _closed_loop_condition(
    graph: EventConnectome,
    registry: ResolvedRegistry,
    mapping: DescendingMap,
    name: str,
    silenced: frozenset[str],
    *,
    steps: int,
    seed: int,
) -> ConditionResult:
    params = ShiuParameters()
    chunk_steps = min(params.delay_steps + 1, steps)
    world_steps = min(2, max(1, steps // chunk_steps))
    state = ShiuState.initial(graph.neuron_count, params=params, seed=seed)
    encoder = VisualInterfaceEncoder(_visual_map(registry))
    decoder = DescendingDecoder(mapping)
    silence_mask = population_silence_mask(graph, mapping, silenced)
    world = ArenaWorld(
        ArenaConfig(10.0, 10.0, 0.1, 0.2),
        FlyBody(5.0, 5.0, 0.0, 0.0, 0.0, 1.0, (False,) * 6),
        food=(1.0, 1.0),
        threat=(9.0, 9.0),
    )
    disc_path = tuple(
        VisualDisc(9.0 - 2.0 * index, 5.5 + 2.0 * index, 0.4)
        for index in range(world_steps + 1)
    )
    all_events: list[ExternalEvent] = []
    spike_steps: list[tuple[int, tuple[int, ...]]] = []
    commands: list[tuple[float, float]] = []
    bodies: list[FlyBody] = []
    for world_index in range(world_steps):
        observation = observe_retina(
            world.body,
            (disc_path[world_index],),
            (disc_path[world_index + 1],),
        )
        feature_only = RetinalObservation(
            0.0,
            0.0,
            0.0,
            0.0,
            observation.left_motion,
            observation.right_motion,
            observation.looming,
        )
        events = tuple(
            event
            for neural_step in range(state.step, state.step + chunk_steps)
            for event in encoder.encode_feature_calibration(
                feature_only,
                step=neural_step,
            )
        )
        all_events.extend(events)
        voltage_events, refractory_exempt = _voltage_schedule(graph, events)
        chunk_spikes: list[int] = []
        for batch in simulate_shiu(
            graph,
            params,
            steps=chunk_steps,
            external_voltage_events=voltage_events,
            seed=seed,
            state=state,
            silenced=silence_mask,
            refractory_exempt=refractory_exempt,
        ):
            spikes = tuple(int(value) for value in batch.neuron_ids)
            spike_steps.append((batch.step, spikes))
            chunk_spikes.extend(spikes)
        activity = decoder.decode(tuple(chunk_spikes))
        body = world.step(activity.command).body
        commands.append((activity.command.forward, activity.command.turn))
        bodies.append(body)
    populations = mapping.named_populations()
    counts = {
        population: sum(
            neuron_id in spikes
            for _, spikes in spike_steps
            for neuron_id in neuron_ids
        )
        for population, neuron_ids in populations.items()
    }
    stimulus = [asdict(event) for event in all_events]
    trace = {
        "stimulus": stimulus,
        "spikes": spike_steps,
        "commands": commands,
        "bodies": [asdict(body) for body in bodies],
    }
    return ConditionResult(
        name=name,
        protocol=ProtocolName.VISUAL_FEATURE_CLOSED_LOOP,
        stimulus_digest=_digest(stimulus),
        spike_digest=_digest(spike_steps),
        command_digest=_digest(commands),
        trace_digest=_digest(trace),
        relevant_spike_counts=counts,
        turn_integral=sum(command[1] for command in commands),
        reverse_integral=sum(max(0.0, -command[0]) for command in commands),
        final_body=bodies[-1],
        silenced_populations=tuple(sorted(silenced)),
        upstream_visual_processing_bypassed=True,
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
    direct_conditions = tuple(
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
    baseline = next(item for item in direct_conditions if item.name == "d_na02_left")
    restored = next(item for item in direct_conditions if item.name == "d_na02_left_restored")
    mdn = next(item for item in direct_conditions if item.name == "mdn_bilateral")
    mdn_silenced = next(
        item for item in direct_conditions if item.name == "mdn_bilateral_silenced"
    )
    left_silenced = next(
        item for item in direct_conditions if item.name == "d_na02_left_silenced"
    )

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
    right = next(item for item in direct_conditions if item.name == "d_na02_right")
    bilateral = next(item for item in direct_conditions if item.name == "d_na02_bilateral")
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
    sensory_conditions, sensory_claims = _run_sensory_protocols(
        graph,
        registry,
        mapping,
        steps=steps,
        seed=seed,
    )
    closed_loop = _closed_loop_condition(
        graph,
        registry,
        mapping,
        "feature_closed_loop",
        frozenset(),
        steps=steps,
        seed=seed,
    )
    closed_loop_lesion = _closed_loop_condition(
        graph,
        registry,
        mapping,
        "feature_closed_loop_lesion",
        frozenset({"d_na02_left"}),
        steps=steps,
        seed=seed,
    )
    closed_loop_replay = _closed_loop_condition(
        graph,
        registry,
        mapping,
        "feature_closed_loop_replay",
        frozenset(),
        steps=steps,
        seed=seed,
    )
    closed_effect = closed_loop.turn_integral
    closed_lesion_effect = closed_effect - closed_loop_lesion.turn_integral
    sensory_claims["feature_closed_loop"] = ClaimClassification(
        classification=(
            "positive"
            if closed_effect > MIN_DIRECTIONAL_EFFECT
            and closed_lesion_effect / max(abs(closed_effect), MIN_DIRECTIONAL_EFFECT)
            >= MIN_LESION_FRACTION
            and closed_loop.trace_digest == closed_loop_replay.trace_digest
            else "null"
        ),
        evidence_kind="simulation_observation",
        numerator=closed_effect,
        denominator=max(abs(closed_effect), MIN_DIRECTIONAL_EFFECT),
        threshold=MIN_DIRECTIONAL_EFFECT,
        lesion_effect=closed_lesion_effect,
        upstream_visual_processing_bypassed=True,
        reasons=("closed-loop feature-bypass response, lesion, and replay were evaluated",),
    )
    conditions = direct_conditions + sensory_conditions + (
        closed_loop,
        closed_loop_lesion,
        closed_loop_replay,
    )
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
        sensory_claims=sensory_claims,
        replay_exact=replay.trace_digest == baseline.trace_digest,
        graph_unchanged=graph_unchanged,
        software_revision=_software_revision(),
        runtime_seconds=time.perf_counter() - started,
        peak_rss_bytes=_peak_rss_bytes(),
    )
