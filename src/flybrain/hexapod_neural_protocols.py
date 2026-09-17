"""Sparse neural-path protocols connecting descending and hexapod motor populations."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flybrain.descending_interface import DescendingMap, population_silence_mask
from flybrain.graph import EventConnectome
from flybrain.hexapod_motor import HexapodMotorMap, motor_population_silence_mask
from flybrain.shiu import ShiuParameters, simulate_shiu

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
