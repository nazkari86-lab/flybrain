"""Sparse neural-path protocols connecting descending and hexapod motor populations."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flybrain.descending_interface import DescendingMap, population_silence_mask
from flybrain.embodied_interfaces import ExternalEvent
from flybrain.graph import EventConnectome
from flybrain.hexapod_body import (
    HexapodBody,
    HexapodParameters,
    LegName,
    ReferenceHexapod,
)
from flybrain.hexapod_motor import (
    HexapodMotorDecoder,
    HexapodMotorMap,
    MotorGroup,
    motor_population_silence_mask,
)
from flybrain.proprioceptive_interface import (
    ProprioceptiveBank,
    ProprioceptiveCalibration,
    ProprioceptiveEncoder,
    ProprioceptiveMap,
    observe_proprioception,
)
from flybrain.shiu import ShiuParameters, ShiuState, simulate_shiu

DN_NAMES = (
    "d_na02_left",
    "d_na02_right",
    "d_ng13_left",
    "d_ng13_right",
    "mdn_left",
    "mdn_right",
)
_MIRROR_DN = {
    "d_na02_left": "d_na02_right",
    "d_na02_right": "d_na02_left",
    "d_ng13_left": "d_ng13_right",
    "d_ng13_right": "d_ng13_left",
    "mdn_left": "mdn_right",
    "mdn_right": "mdn_left",
}
_MIRROR_LEG: dict[LegName, LegName] = {
    "left_fore": "right_fore",
    "left_middle": "right_middle",
    "left_hind": "right_hind",
    "right_fore": "left_fore",
    "right_middle": "left_middle",
    "right_hind": "left_hind",
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
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


class NeuralConditionResult(BaseModel, frozen=True):
    """Open-loop neural condition with no body-level command or behavior claim."""

    model_config = ConfigDict(extra="forbid")

    name: str
    protocol: Literal["dn_to_motor_open_loop"] = "dn_to_motor_open_loop"
    input_population: str
    silenced_populations: tuple[str, ...]
    event_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    spike_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_spikes: int = Field(ge=0)
    motor_spikes: int = Field(ge=0)
    motor_spikes_by_population: dict[str, int]


class NeuralClaimClassification(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    classification: Literal["positive", "null", "directionally_wrong", "underpowered"]
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    motor_spikes: int = Field(ge=0)
    threshold: int = Field(gt=0)
    dn_lesion_fraction: float
    motor_lesion_fraction: float
    reasons: tuple[str, ...]

    @model_validator(mode="after")
    def validate_finite(self) -> Self:
        if not math.isfinite(self.dn_lesion_fraction) or not math.isfinite(
            self.motor_lesion_fraction
        ):
            raise ValueError("neural lesion fractions must be finite")
        return self


class DNMotorProtocolResult(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    protocol: Literal["dn_to_motor-open-loop-v1"] = "dn_to_motor-open-loop-v1"
    graph_neurons: int
    graph_edges: int
    conditions: tuple[NeuralConditionResult, ...]
    claims: dict[str, NeuralClaimClassification]
    replay_exact: bool
    graph_unchanged: bool


class ProprioceptiveConditionResult(BaseModel, frozen=True):
    """Open-loop proprioceptive condition without world or body-command fields."""

    model_config = ConfigDict(extra="forbid")

    name: str
    protocol: Literal["proprio_to_motor_open_loop"] = "proprio_to_motor_open_loop"
    input_bank: str
    event_target_ids: tuple[int, ...]
    silenced_banks: tuple[str, ...]
    motor_silenced: bool
    event_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    spike_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_spikes: int = Field(ge=0)
    input_spikes_by_bank: dict[str, int]
    motor_spikes: int = Field(ge=0)
    motor_spikes_by_population: dict[str, int]


class ProprioceptiveClaimClassification(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    classification: Literal["positive", "null", "directionally_wrong", "underpowered"]
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    motor_spikes: int = Field(ge=0)
    threshold: int = Field(gt=0)
    sensory_lesion_fraction: float
    motor_lesion_fraction: float
    reasons: tuple[str, ...]

    @model_validator(mode="after")
    def validate_finite(self) -> Self:
        if not math.isfinite(self.sensory_lesion_fraction) or not math.isfinite(
            self.motor_lesion_fraction
        ):
            raise ValueError("proprioceptive lesion fractions must be finite")
        return self


class ProprioMotorProtocolResult(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    protocol: Literal["proprio-to-motor-open-loop-v1"] = (
        "proprio-to-motor-open-loop-v1"
    )
    graph_neurons: int
    graph_edges: int
    conditions: tuple[ProprioceptiveConditionResult, ...]
    claims: dict[str, ProprioceptiveClaimClassification]
    replay_exact: bool
    graph_unchanged: bool


class ClosedLoopConditionResult(BaseModel, frozen=True):
    """Stateful neural-body condition using proprioception as its only feedback."""

    model_config = ConfigDict(extra="forbid")

    name: str
    protocol: Literal["closed_loop_hexapod"] = "closed_loop_hexapod"
    proprio_silenced: bool
    motor_silenced: bool
    neural_chunk_steps: int = Field(gt=0)
    neural_state_steps: tuple[int, ...]
    proprioceptive_event_steps: tuple[int, ...]
    proprioceptive_input_mapping: dict[str, str]
    motor_output_mapping: dict[str, str]
    proprioceptive_spikes: int = Field(ge=0)
    motor_spikes: int = Field(ge=0)
    neural_joint_motion_rad: float = Field(ge=0.0)
    final_body: HexapodBody
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class ClosedLoopClaimClassification(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    classification: Literal["positive", "null", "directionally_wrong", "underpowered"]
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    proprioceptive_spikes: int = Field(ge=0)
    motor_spikes: int = Field(ge=0)
    motor_threshold: int = Field(gt=0)
    proprio_lesion_fraction: float
    motor_lesion_fraction: float
    reasons: tuple[str, ...]

    @model_validator(mode="after")
    def validate_finite(self) -> Self:
        if not math.isfinite(self.proprio_lesion_fraction) or not math.isfinite(
            self.motor_lesion_fraction
        ):
            raise ValueError("closed-loop lesion fractions must be finite")
        return self


class ClosedLoopHexapodResult(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    protocol: Literal["closed-loop-hexapod-v1"] = "closed-loop-hexapod-v1"
    conditions: tuple[ClosedLoopConditionResult, ...]
    claim: ClosedLoopClaimClassification
    replay_exact: bool
    mirror_exact: bool
    graph_unchanged: bool
    sparse_storage_unchanged: bool


def _validate_interfaces(
    graph: EventConnectome,
    descending: DescendingMap,
    motor: HexapodMotorMap,
) -> None:
    motor.validate_graph(graph)
    available = {int(value) for value in graph.neuron_ids}
    required = {
        neuron_id
        for population in descending.named_populations().values()
        for neuron_id in population
    }
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"descending population IDs absent from graph: {missing}")


def _voltage_events(
    graph: EventConnectome,
    neuron_ids: tuple[int, ...],
    *,
    steps: int,
    interval_steps: int,
    amplitude_mv: float,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if interval_steps <= 0:
        raise ValueError("drive interval must be positive")
    if not math.isfinite(amplitude_mv) or amplitude_mv <= 0.0:
        raise ValueError("drive amplitude must be finite and positive")
    index_by_id = {int(value): index for index, value in enumerate(graph.neuron_ids)}
    indices = np.array([index_by_id[value] for value in neuron_ids], dtype=np.int64)
    amplitudes = np.full(indices.size, amplitude_mv, dtype=np.float32)
    return {
        step: (indices.copy(), amplitudes.copy())
        for step in range(0, steps, interval_steps)
    }


def _run_dn_condition(
    graph: EventConnectome,
    descending: DescendingMap,
    motor: HexapodMotorMap,
    *,
    name: str,
    input_population: str,
    silenced_dn: frozenset[str],
    silence_all_motor: bool,
    steps: int,
    seed: int,
    params: ShiuParameters,
    interval_steps: int,
    amplitude_mv: float,
) -> NeuralConditionResult:
    dn_populations = descending.named_populations()
    input_ids = dn_populations[input_population]
    events = _voltage_events(
        graph,
        input_ids,
        steps=steps,
        interval_steps=interval_steps,
        amplitude_mv=amplitude_mv,
    )
    dn_mask = population_silence_mask(graph, descending, silenced_dn)
    motor_names = (
        frozenset(motor.named_populations()) if silence_all_motor else frozenset()
    )
    motor_mask = motor_population_silence_mask(graph, motor, motor_names)
    silence_mask = np.logical_or(dn_mask, motor_mask)
    spike_trace: list[tuple[int, tuple[int, ...]]] = []
    input_id_set = set(input_ids)
    motor_populations = motor.named_populations()
    motor_id_set = {
        neuron_id for ids in motor_populations.values() for neuron_id in ids
    }
    counts = dict.fromkeys(motor_populations, 0)
    input_spikes = 0
    motor_spikes = 0
    for batch in simulate_shiu(
        graph,
        params,
        steps=steps,
        external_voltage_events=events,
        seed=seed,
        silenced=silence_mask,
    ):
        spikes = tuple(int(value) for value in batch.neuron_ids)
        spike_trace.append((batch.step, spikes))
        input_spikes += sum(value in input_id_set for value in spikes)
        motor_spikes += sum(value in motor_id_set for value in spikes)
        for population, ids in motor_populations.items():
            selected = set(ids)
            counts[population] += sum(value in selected for value in spikes)
    event_payload = {
        step: (indices.tolist(), amplitudes.tolist())
        for step, (indices, amplitudes) in sorted(events.items())
    }
    return NeuralConditionResult(
        name=name,
        input_population=input_population,
        silenced_populations=tuple(
            sorted((*silenced_dn, *(motor_names if silence_all_motor else ())))
        ),
        event_digest=_digest(event_payload),
        spike_digest=_digest(spike_trace),
        trace_digest=_digest({"events": event_payload, "spikes": spike_trace}),
        input_spikes=input_spikes,
        motor_spikes=motor_spikes,
        motor_spikes_by_population=counts,
    )


def _lesion_fraction(normal: int, lesion: int) -> float:
    return (normal - lesion) / max(normal, 1)


def _classify(
    normal: NeuralConditionResult,
    dn_lesion: NeuralConditionResult,
    motor_lesion: NeuralConditionResult,
    restored: NeuralConditionResult,
    replay: NeuralConditionResult,
    mirror: NeuralConditionResult,
    *,
    minimum_motor_spikes: int,
    minimum_lesion_fraction: float,
) -> NeuralClaimClassification:
    dn_fraction = _lesion_fraction(normal.motor_spikes, dn_lesion.motor_spikes)
    motor_fraction = _lesion_fraction(normal.motor_spikes, motor_lesion.motor_spikes)
    if normal.input_spikes == 0:
        classification: Literal[
            "positive", "null", "directionally_wrong", "underpowered"
        ] = "underpowered"
        reasons = ("driven descending population emitted no spikes",)
    elif normal.motor_spikes < minimum_motor_spikes:
        classification = "null"
        reasons = ("no sufficient motor recruitment through the canonical graph",)
    elif mirror.motor_spikes < minimum_motor_spikes:
        classification = "directionally_wrong"
        reasons = ("homologous mirrored descending drive failed to recruit motor neurons",)
    elif (
        dn_fraction < minimum_lesion_fraction
        or motor_fraction < minimum_lesion_fraction
    ):
        classification = "null"
        reasons = ("matching DN or motor lesion effect is below threshold",)
    elif (
        restored.trace_digest != normal.trace_digest
        or replay.trace_digest != normal.trace_digest
    ):
        classification = "null"
        reasons = ("restoration or exact replay failed",)
    else:
        classification = "positive"
        reasons = (
            "canonical recruitment, lesions, mirror, restoration, and replay passed",
        )
    return NeuralClaimClassification(
        classification=classification,
        motor_spikes=normal.motor_spikes,
        threshold=minimum_motor_spikes,
        dn_lesion_fraction=dn_fraction,
        motor_lesion_fraction=motor_fraction,
        reasons=reasons,
    )


def run_dn_to_motor_protocols(
    graph: EventConnectome,
    descending: DescendingMap,
    motor: HexapodMotorMap,
    *,
    steps: int,
    seed: int,
    params: ShiuParameters | None = None,
    drive_interval_steps: int = 25,
    drive_amplitude_mv: float = 10.0,
    minimum_motor_spikes: int = 1,
    minimum_lesion_fraction: float = 0.9,
) -> DNMotorProtocolResult:
    """Measure DN-to-motor recruitment without translating DN spikes into body commands."""

    if type(steps) is not int or steps <= 0:
        raise ValueError("DN-to-motor steps must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("DN-to-motor seed must be a non-negative integer")
    if type(minimum_motor_spikes) is not int or minimum_motor_spikes <= 0:
        raise ValueError("minimum motor spikes must be a positive integer")
    if (
        not math.isfinite(minimum_lesion_fraction)
        or not 0.0 <= minimum_lesion_fraction <= 1.0
    ):
        raise ValueError("minimum lesion fraction must lie in [0, 1]")
    _validate_interfaces(graph, descending, motor)
    shiu_params = params or ShiuParameters()
    before_digest = _graph_digest(graph)
    conditions: list[NeuralConditionResult] = []
    claims: dict[str, NeuralClaimClassification] = {}

    def run(
        condition_name: str,
        input_name: str,
        *,
        silenced_dn: frozenset[str] = frozenset(),
        silence_all_motor: bool = False,
    ) -> NeuralConditionResult:
        return _run_dn_condition(
            graph,
            descending,
            motor,
            name=condition_name,
            input_population=input_name,
            silenced_dn=silenced_dn,
            silence_all_motor=silence_all_motor,
            steps=steps,
            seed=seed,
            params=shiu_params,
            interval_steps=drive_interval_steps,
            amplitude_mv=drive_amplitude_mv,
        )

    for population in DN_NAMES:
        normal = run(f"{population}:normal", population)
        dn_lesion = run(
            f"{population}:dn_lesion",
            population,
            silenced_dn=frozenset({population}),
        )
        motor_lesion = run(
            f"{population}:motor_lesion",
            population,
            silence_all_motor=True,
        )
        restored = run(f"{population}:restored", population)
        replay = run(f"{population}:replay", population)
        mirror_name = _MIRROR_DN[population]
        mirror = run(f"{population}:mirror", mirror_name)
        conditions.extend(
            (normal, dn_lesion, motor_lesion, restored, replay, mirror)
        )
        claims[population] = _classify(
            normal,
            dn_lesion,
            motor_lesion,
            restored,
            replay,
            mirror,
            minimum_motor_spikes=minimum_motor_spikes,
            minimum_lesion_fraction=minimum_lesion_fraction,
        )
    replay_exact = all(
        next(item for item in conditions if item.name == f"{name}:replay").trace_digest
        == next(item for item in conditions if item.name == f"{name}:normal").trace_digest
        for name in DN_NAMES
    )
    return DNMotorProtocolResult(
        graph_neurons=graph.neuron_count,
        graph_edges=graph.edge_count,
        conditions=tuple(conditions),
        claims=claims,
        replay_exact=replay_exact,
        graph_unchanged=_graph_digest(graph) == before_digest,
    )


def _validate_proprio_interfaces(
    graph: EventConnectome,
    proprio: ProprioceptiveMap,
    motor: HexapodMotorMap,
) -> None:
    proprio.validate_graph(graph)
    motor.validate_graph(graph)
    proprio_ids = {
        neuron_id for bank in proprio.banks for neuron_id in bank.neuron_ids
    }
    motor_ids = {
        neuron_id for group in motor.groups for neuron_id in group.neuron_ids
    }
    overlap = sorted(proprio_ids & motor_ids)
    if overlap:
        raise ValueError(f"proprioceptive and motor populations overlap: {overlap}")


def _population_mask(
    graph: EventConnectome,
    populations: dict[str, tuple[int, ...]],
    selected: frozenset[str],
) -> np.ndarray:
    unknown = sorted(selected - populations.keys())
    if unknown:
        raise ValueError(f"unknown populations: {unknown}")
    selected_ids = {
        neuron_id for name in selected for neuron_id in populations[name]
    }
    return np.isin(graph.neuron_ids, tuple(selected_ids))


def _run_proprio_condition(
    graph: EventConnectome,
    proprio: ProprioceptiveMap,
    motor: HexapodMotorMap,
    *,
    name: str,
    input_bank: str,
    silenced_banks: frozenset[str],
    silence_all_motor: bool,
    steps: int,
    seed: int,
    params: ShiuParameters,
    interval_steps: int,
    amplitude_mv: float,
) -> ProprioceptiveConditionResult:
    bank_populations = {bank.name: bank.neuron_ids for bank in proprio.banks}
    input_ids = bank_populations[input_bank]
    events = _voltage_events(
        graph,
        input_ids,
        steps=steps,
        interval_steps=interval_steps,
        amplitude_mv=amplitude_mv,
    )
    proprio_mask = _population_mask(graph, bank_populations, silenced_banks)
    motor_names = (
        frozenset(motor.named_populations()) if silence_all_motor else frozenset()
    )
    motor_mask = motor_population_silence_mask(graph, motor, motor_names)
    silence_mask = np.logical_or(proprio_mask, motor_mask)
    motor_populations = motor.named_populations()
    input_sets = {key: set(value) for key, value in bank_populations.items()}
    motor_sets = {key: set(value) for key, value in motor_populations.items()}
    input_counts = dict.fromkeys(bank_populations, 0)
    motor_counts = dict.fromkeys(motor_populations, 0)
    spike_trace: list[tuple[int, tuple[int, ...]]] = []
    for batch in simulate_shiu(
        graph,
        params,
        steps=steps,
        external_voltage_events=events,
        seed=seed,
        silenced=silence_mask,
    ):
        spikes = tuple(int(value) for value in batch.neuron_ids)
        spike_trace.append((batch.step, spikes))
        for population, selected in input_sets.items():
            input_counts[population] += sum(value in selected for value in spikes)
        for population, selected in motor_sets.items():
            motor_counts[population] += sum(value in selected for value in spikes)
    event_payload = {
        step: (indices.tolist(), amplitudes.tolist())
        for step, (indices, amplitudes) in sorted(events.items())
    }
    return ProprioceptiveConditionResult(
        name=name,
        input_bank=input_bank,
        event_target_ids=input_ids,
        silenced_banks=tuple(sorted(silenced_banks)),
        motor_silenced=silence_all_motor,
        event_digest=_digest(event_payload),
        spike_digest=_digest(spike_trace),
        trace_digest=_digest({"events": event_payload, "spikes": spike_trace}),
        input_spikes=input_counts[input_bank],
        input_spikes_by_bank=input_counts,
        motor_spikes=sum(motor_counts.values()),
        motor_spikes_by_population=motor_counts,
    )


def _classify_proprio(
    normal: ProprioceptiveConditionResult,
    sensory_lesion: ProprioceptiveConditionResult,
    motor_lesion: ProprioceptiveConditionResult,
    restored: ProprioceptiveConditionResult,
    replay: ProprioceptiveConditionResult,
    mirror: ProprioceptiveConditionResult,
    *,
    minimum_motor_spikes: int,
    minimum_lesion_fraction: float,
) -> ProprioceptiveClaimClassification:
    sensory_fraction = _lesion_fraction(
        normal.motor_spikes, sensory_lesion.motor_spikes
    )
    motor_fraction = _lesion_fraction(normal.motor_spikes, motor_lesion.motor_spikes)
    if normal.input_spikes == 0:
        classification: Literal[
            "positive", "null", "directionally_wrong", "underpowered"
        ] = "underpowered"
        reasons = ("driven proprioceptive bank emitted no spikes",)
    elif normal.motor_spikes < minimum_motor_spikes:
        classification = "null"
        reasons = ("no sufficient motor recruitment through the canonical graph",)
    elif mirror.motor_spikes < minimum_motor_spikes:
        classification = "directionally_wrong"
        reasons = ("homologous mirrored bank failed to recruit motor neurons",)
    elif (
        sensory_fraction < minimum_lesion_fraction
        or motor_fraction < minimum_lesion_fraction
    ):
        classification = "null"
        reasons = ("sensory or motor lesion effect is below threshold",)
    elif (
        restored.trace_digest != normal.trace_digest
        or replay.trace_digest != normal.trace_digest
    ):
        classification = "null"
        reasons = ("restoration or exact replay failed",)
    else:
        classification = "positive"
        reasons = (
            "canonical recruitment, lesions, mirror, restoration, and replay passed",
        )
    return ProprioceptiveClaimClassification(
        classification=classification,
        motor_spikes=normal.motor_spikes,
        threshold=minimum_motor_spikes,
        sensory_lesion_fraction=sensory_fraction,
        motor_lesion_fraction=motor_fraction,
        reasons=reasons,
    )


def run_proprio_to_motor_protocols(
    graph: EventConnectome,
    proprio: ProprioceptiveMap,
    motor: HexapodMotorMap,
    *,
    steps: int,
    seed: int,
    params: ShiuParameters | None = None,
    drive_interval_steps: int = 25,
    drive_amplitude_mv: float = 10.0,
    minimum_motor_spikes: int = 1,
    minimum_lesion_fraction: float = 0.9,
) -> ProprioMotorProtocolResult:
    """Measure exact proprioceptive-bank recruitment of motor populations."""

    if type(steps) is not int or steps <= 0:
        raise ValueError("proprio-to-motor steps must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("proprio-to-motor seed must be a non-negative integer")
    if type(minimum_motor_spikes) is not int or minimum_motor_spikes <= 0:
        raise ValueError("minimum motor spikes must be a positive integer")
    if (
        not math.isfinite(minimum_lesion_fraction)
        or not 0.0 <= minimum_lesion_fraction <= 1.0
    ):
        raise ValueError("minimum lesion fraction must lie in [0, 1]")
    _validate_proprio_interfaces(graph, proprio, motor)
    shiu_params = params or ShiuParameters()
    before_digest = _graph_digest(graph)
    conditions: list[ProprioceptiveConditionResult] = []
    claims: dict[str, ProprioceptiveClaimClassification] = {}

    def run(
        condition_name: str,
        input_name: str,
        *,
        silenced_banks: frozenset[str] = frozenset(),
        silence_all_motor: bool = False,
    ) -> ProprioceptiveConditionResult:
        return _run_proprio_condition(
            graph,
            proprio,
            motor,
            name=condition_name,
            input_bank=input_name,
            silenced_banks=silenced_banks,
            silence_all_motor=silence_all_motor,
            steps=steps,
            seed=seed,
            params=shiu_params,
            interval_steps=drive_interval_steps,
            amplitude_mv=drive_amplitude_mv,
        )

    for bank in proprio.banks:
        name = bank.name
        normal = run(f"{name}:normal", name)
        sensory_lesion = run(
            f"{name}:side_lesion", name, silenced_banks=frozenset({name})
        )
        motor_lesion = run(f"{name}:motor_lesion", name, silence_all_motor=True)
        restored = run(f"{name}:restored", name)
        replay = run(f"{name}:replay", name)
        mirror_name = f"{_MIRROR_LEG[bank.leg]}_proprioception"
        mirror = run(f"{name}:mirror", mirror_name)
        conditions.extend(
            (normal, sensory_lesion, motor_lesion, restored, replay, mirror)
        )
        claims[name] = _classify_proprio(
            normal,
            sensory_lesion,
            motor_lesion,
            restored,
            replay,
            mirror,
            minimum_motor_spikes=minimum_motor_spikes,
            minimum_lesion_fraction=minimum_lesion_fraction,
        )
    replay_exact = all(
        next(item for item in conditions if item.name == f"{name}:replay").trace_digest
        == next(item for item in conditions if item.name == f"{name}:normal").trace_digest
        for name in claims
    )
    return ProprioMotorProtocolResult(
        graph_neurons=graph.neuron_count,
        graph_edges=graph.edge_count,
        conditions=tuple(conditions),
        claims=claims,
        replay_exact=replay_exact,
        graph_unchanged=_graph_digest(graph) == before_digest,
    )


def _external_event_schedule(
    graph: EventConnectome,
    events: tuple[ExternalEvent, ...],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    index_by_id = {int(value): index for index, value in enumerate(graph.neuron_ids)}
    grouped: dict[int, list[tuple[int, float]]] = {}
    for event in events:
        pairs = grouped.setdefault(event.step, [])
        for neuron_id, voltage in zip(
            event.neuron_ids, event.voltages, strict=True
        ):
            if neuron_id not in index_by_id:
                raise ValueError(f"proprioceptive event ID absent from graph: {neuron_id}")
            pairs.append((index_by_id[neuron_id], voltage))
    return {
        event_step: (
            np.array([index for index, _ in pairs], dtype=np.int64),
            np.array([voltage for _, voltage in pairs], dtype=np.float32),
        )
        for event_step, pairs in grouped.items()
    }


def _bodies_close(left: HexapodBody, right: HexapodBody) -> bool:
    left_values = left.model_dump(mode="json")
    right_values = right.model_dump(mode="json")
    for body_values in (left_values, right_values):
        for leg in body_values["legs"]:
            leg.pop("phase")

    def close(a: object, b: object) -> bool:
        if isinstance(a, float) and isinstance(b, float):
            return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(
                close(x, y) for x, y in zip(a, b, strict=True)
            )
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(close(a[key], b[key]) for key in a)
        return a == b

    return close(left_values, right_values)


def _mirror_interfaces(
    proprio: ProprioceptiveMap,
    motor: HexapodMotorMap,
    *,
    mirrored: bool,
) -> tuple[ProprioceptiveMap, HexapodMotorMap, dict[str, str], dict[str, str]]:
    """Transform only exact left/right interface populations around a fixed graph."""

    if not mirrored:
        return (
            proprio,
            motor,
            {bank.leg: bank.name for bank in proprio.banks},
            {group.name: group.name for group in motor.groups},
        )
    banks = []
    proprio_mapping: dict[str, str] = {}
    for bank in proprio.banks:
        source = proprio.bank(_MIRROR_LEG[bank.leg])
        banks.append(
            ProprioceptiveBank(
                name=bank.name,
                leg=bank.leg,
                neuron_ids=source.neuron_ids,
            )
        )
        proprio_mapping[bank.leg] = source.name
    groups = []
    motor_mapping: dict[str, str] = {}
    for group in motor.groups:
        source_name = (
            f"{_MIRROR_LEG[group.leg]}_{group.joint}_{group.direction}"
        )
        motor_source = motor.group(source_name)
        groups.append(
            MotorGroup(
                name=group.name,
                leg=group.leg,
                joint=group.joint,
                direction=group.direction,
                neuron_ids=motor_source.neuron_ids,
            )
        )
        motor_mapping[group.name] = motor_source.name
    return (
        ProprioceptiveMap(banks=tuple(banks)),
        HexapodMotorMap(groups=tuple(groups)),
        proprio_mapping,
        motor_mapping,
    )


def _run_closed_loop_condition(
    graph: EventConnectome,
    proprio: ProprioceptiveMap,
    motor: HexapodMotorMap,
    *,
    name: str,
    body_steps: int,
    seed: int,
    params: ShiuParameters,
    body_parameters: HexapodParameters,
    calibration: ProprioceptiveCalibration,
    proprio_silenced: bool,
    motor_silenced: bool,
    interface_mirrored: bool,
) -> ClosedLoopConditionResult:
    execution_proprio, execution_motor, proprio_mapping, motor_mapping = (
        _mirror_interfaces(proprio, motor, mirrored=interface_mirrored)
    )
    simulator = ReferenceHexapod(body_parameters)
    initial_body = simulator.observe()
    encoder = ProprioceptiveEncoder(execution_proprio, calibration)
    decoder = HexapodMotorDecoder(execution_motor)
    state = ShiuState.initial(graph.neuron_count, params=params, seed=seed)
    chunk_steps = round(body_parameters.dt_s * 1000.0 / params.dt_ms)
    if chunk_steps < 1:
        raise ValueError("body step is shorter than one Shiu integration step")
    proprio_names = frozenset(bank.name for bank in execution_proprio.banks)
    proprio_mask = _population_mask(
        graph,
        {bank.name: bank.neuron_ids for bank in execution_proprio.banks},
        proprio_names if proprio_silenced else frozenset(),
    )
    motor_names = (
        frozenset(execution_motor.named_populations()) if motor_silenced else frozenset()
    )
    motor_mask = motor_population_silence_mask(graph, execution_motor, motor_names)
    silence_mask = np.logical_or(proprio_mask, motor_mask)
    motor_ids = {
        neuron_id for group in execution_motor.groups for neuron_id in group.neuron_ids
    }
    proprio_ids = {
        neuron_id for bank in execution_proprio.banks for neuron_id in bank.neuron_ids
    }
    neural_state_steps: list[int] = []
    event_steps: list[int] = []
    trace: list[object] = []
    total_motor_spikes = 0
    total_proprio_spikes = 0
    for _ in range(body_steps):
        body = simulator.observe()
        observation = observe_proprioception(body, body_parameters, calibration)
        events = encoder.encode(observation, step=state.step)
        event_steps.append(state.step)
        schedule = _external_event_schedule(graph, events)
        chunk_spikes: list[int] = []
        neural_trace: list[tuple[int, tuple[int, ...]]] = []
        for batch in simulate_shiu(
            graph,
            params,
            steps=chunk_steps,
            external_voltage_events=schedule,
            state=state,
            silenced=silence_mask,
        ):
            spikes = tuple(int(value) for value in batch.neuron_ids)
            neural_trace.append((batch.step, spikes))
            selected = [value for value in spikes if value in motor_ids]
            chunk_spikes.extend(selected)
            total_motor_spikes += len(selected)
            total_proprio_spikes += sum(value in proprio_ids for value in spikes)
        neural_state_steps.append(state.step)
        phases = (
            body.legs[0].phase,
            body.legs[1].phase,
            body.legs[2].phase,
            body.legs[3].phase,
            body.legs[4].phase,
            body.legs[5].phase,
        )
        activation = decoder.decode(
            tuple(chunk_spikes),
            window_s=body_parameters.dt_s,
            leg_phases=phases,
        )
        next_body = simulator.step(activation.torques)
        trace.append(
            {
                "events": [
                    {
                        "step": event.step,
                        "ids": event.neuron_ids,
                        "voltages": event.voltages,
                        "channel": event.channel,
                    }
                    for event in events
                ],
                "spikes": neural_trace,
                "body": next_body.model_dump(mode="json"),
            }
        )
    final_body = simulator.observe()
    neural_joint_motion = sum(
        abs(final.joint_angles_rad[joint] - initial.joint_angles_rad[joint])
        for initial, final in zip(initial_body.legs, final_body.legs, strict=True)
        for joint in (1, 2)
    )
    return ClosedLoopConditionResult(
        name=name,
        proprio_silenced=proprio_silenced,
        motor_silenced=motor_silenced,
        neural_chunk_steps=chunk_steps,
        neural_state_steps=tuple(neural_state_steps),
        proprioceptive_event_steps=tuple(event_steps),
        proprioceptive_input_mapping=proprio_mapping,
        motor_output_mapping=motor_mapping,
        proprioceptive_spikes=total_proprio_spikes,
        motor_spikes=total_motor_spikes,
        neural_joint_motion_rad=neural_joint_motion,
        final_body=final_body,
        trace_digest=_digest(trace),
    )


def run_closed_loop_hexapod_protocol(
    graph: EventConnectome,
    proprio: ProprioceptiveMap,
    motor: HexapodMotorMap,
    *,
    body_steps: int,
    seed: int,
    params: ShiuParameters | None = None,
    body_parameters: HexapodParameters | None = None,
    calibration: ProprioceptiveCalibration | None = None,
    minimum_motor_spikes: int = 1,
    minimum_lesion_fraction: float = 0.9,
) -> ClosedLoopHexapodResult:
    """Run a stateful sparse neural-body-proprioceptive feedback loop."""

    if type(body_steps) is not int or body_steps <= 0:
        raise ValueError("closed-loop body steps must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("closed-loop seed must be a non-negative integer")
    if type(minimum_motor_spikes) is not int or minimum_motor_spikes <= 0:
        raise ValueError("minimum motor spikes must be a positive integer")
    if (
        not math.isfinite(minimum_lesion_fraction)
        or not 0.0 <= minimum_lesion_fraction <= 1.0
    ):
        raise ValueError("minimum lesion fraction must lie in [0, 1]")
    _validate_proprio_interfaces(graph, proprio, motor)
    shiu_params = params or ShiuParameters()
    shiu_params.validate()
    body_config = body_parameters or HexapodParameters()
    sensory_calibration = calibration or ProprioceptiveCalibration()
    before_digest = _graph_digest(graph)
    before_storage = graph.storage_items

    def run(
        name: str,
        *,
        proprio_silenced: bool = False,
        motor_silenced: bool = False,
        interface_mirrored: bool = False,
    ) -> ClosedLoopConditionResult:
        return _run_closed_loop_condition(
            graph,
            proprio,
            motor,
            name=name,
            body_steps=body_steps,
            seed=seed,
            params=shiu_params,
            body_parameters=body_config,
            calibration=sensory_calibration,
            proprio_silenced=proprio_silenced,
            motor_silenced=motor_silenced,
            interface_mirrored=interface_mirrored,
        )

    normal = run("normal")
    motor_lesion = run("motor_lesion", motor_silenced=True)
    proprio_lesion = run("proprio_lesion", proprio_silenced=True)
    restored = run("restored")
    replay = run("replay")
    mirror = run("mirror", interface_mirrored=True)
    conditions = (
        normal,
        motor_lesion,
        proprio_lesion,
        restored,
        replay,
        mirror,
    )
    proprio_fraction = _lesion_fraction(
        normal.motor_spikes, proprio_lesion.motor_spikes
    )
    motor_fraction = _lesion_fraction(normal.motor_spikes, motor_lesion.motor_spikes)
    replay_exact = (
        restored.trace_digest == normal.trace_digest
        and replay.trace_digest == normal.trace_digest
    )
    mirror_exact = _bodies_close(mirror.final_body.mirror(), normal.final_body)
    if normal.proprioceptive_spikes == 0:
        classification: Literal[
            "positive", "null", "directionally_wrong", "underpowered"
        ] = "underpowered"
        reasons = ("proprioceptive banks emitted no spikes",)
    elif normal.motor_spikes < minimum_motor_spikes:
        classification = "null"
        reasons = ("no sufficient closed-loop motor recruitment",)
    elif normal.neural_joint_motion_rad == 0.0:
        classification = "null"
        reasons = ("motor spikes caused no neurally driven joint motion",)
    elif not mirror_exact:
        classification = "directionally_wrong"
        reasons = ("homologous mirror mechanics failed",)
    elif (
        proprio_fraction < minimum_lesion_fraction
        or motor_fraction < minimum_lesion_fraction
    ):
        classification = "null"
        reasons = ("proprioceptive or motor lesion effect is below threshold",)
    elif not replay_exact:
        classification = "null"
        reasons = ("restoration or exact replay failed",)
    else:
        classification = "positive"
        reasons = (
            "stateful feedback, lesions, motion, mirror, restoration, and replay passed",
        )
    claim = ClosedLoopClaimClassification(
        classification=classification,
        proprioceptive_spikes=normal.proprioceptive_spikes,
        motor_spikes=normal.motor_spikes,
        motor_threshold=minimum_motor_spikes,
        proprio_lesion_fraction=proprio_fraction,
        motor_lesion_fraction=motor_fraction,
        reasons=reasons,
    )
    return ClosedLoopHexapodResult(
        conditions=conditions,
        claim=claim,
        replay_exact=replay_exact,
        mirror_exact=mirror_exact,
        graph_unchanged=_graph_digest(graph) == before_digest,
        sparse_storage_unchanged=graph.storage_items == before_storage,
    )
