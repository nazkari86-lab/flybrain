"""Calibration-only sensory-to-plasticity-to-hexapod closed neural loop."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from flybrain.autonomous_learning_benchmark import (
    AssociativeCalibrationConfig,
    _clone_binding,
    _effective_graph,
)
from flybrain.conditioning_world import ConditioningSchedule
from flybrain.graph import EventConnectome
from flybrain.hexapod_backend import (
    HexapodBackend,
    ReferenceHexapodBackend,
)
from flybrain.hexapod_body import HexapodParameters
from flybrain.hexapod_motor import (
    HexapodMotorDecoder,
    HexapodMotorMap,
    motor_population_silence_mask,
)
from flybrain.mushroom_body_learning import MushroomBodyLearning
from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.proprioceptive_interface import (
    ProprioceptiveCalibration,
    ProprioceptiveEncoder,
    ProprioceptiveMap,
    observe_proprioception,
)
from flybrain.reinforcement_interface import AnonymousContact, ReinforcementInterface
from flybrain.shiu import ShiuState, poisson_voltage_events, simulate_shiu

BackendFactory = Callable[[HexapodParameters], HexapodBackend]


class AssociativeMotorCalibrationResult(BaseModel, frozen=True):
    """A reproducible full-loop calibration, explicitly not an autonomous behavior result."""

    model_config = ConfigDict(extra="forbid")

    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    autonomous_behavior_claim_allowed: Literal[False] = False
    replay_exact: bool
    graph_unchanged: bool
    motor_spikes: int = Field(ge=0)
    mbon_spikes: int = Field(ge=0)
    mbon_spike_counts: tuple[tuple[int, int], ...]
    sensory_voltage_events: int = Field(ge=0)
    proprioceptive_events: int = Field(ge=0)
    final_multipliers: tuple[float, ...]
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class AssociativeMotorLearningProbeResult(BaseModel, frozen=True):
    """Training/probe calibration that attributes a probe only to persistent weights."""

    model_config = ConfigDict(extra="forbid")

    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    autonomous_behavior_claim_allowed: Literal[False] = False
    classification: Literal["motor_difference", "null", "underpowered"]
    replay_exact: bool
    graph_unchanged: bool
    training_final_multipliers: tuple[float, ...]
    baseline_probe_mbon_spikes: int = Field(ge=0)
    learned_probe_mbon_spikes: int = Field(ge=0)
    probe_mbon_spike_difference: int
    baseline_probe_mbon_spike_counts: tuple[tuple[int, int], ...]
    learned_probe_mbon_spike_counts: tuple[tuple[int, int], ...]
    baseline_probe_motor_spikes: int = Field(ge=0)
    learned_probe_motor_spikes: int = Field(ge=0)
    probe_motor_spike_difference: int
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class _LoopTrace:
    motor_spikes: int
    mbon_spikes: int
    mbon_spike_counts: tuple[tuple[int, int], ...]
    sensory_voltage_events: int
    proprioceptive_events: int
    final_multipliers: tuple[float, ...]
    body_digest: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()


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


def _run(
    graph: EventConnectome,
    binding: PlasticEdgeBinding,
    *,
    learning: AssociativeCalibrationConfig,
    motor: HexapodMotorMap,
    proprio: ProprioceptiveMap,
    schedule: ConditioningSchedule,
    reinforcement: ReinforcementInterface,
    body_parameters: HexapodParameters,
    proprioceptive_calibration: ProprioceptiveCalibration,
    backend_factory: BackendFactory,
    motor_silenced: bool,
    proprioceptive_silenced: bool,
) -> _LoopTrace:
    motor.validate_graph(graph)
    proprio.validate_graph(graph)
    if len(schedule.events) == 0:
        raise ValueError("associative motor schedule must contain events")
    chunk_steps = round(body_parameters.dt_s * 1000.0 / learning.shiu_parameters.dt_ms)
    if chunk_steps != learning.neural_chunk_steps:
        raise ValueError("body timestep must equal declared neural calibration chunk")
    index_by_id = {int(neuron_id): index for index, neuron_id in enumerate(graph.neuron_ids)}
    source_ids = (
        learning.cue_ids
        if learning.input_mode == "direct_kc"
        else learning.sensory_input_ids
    )
    required = {
        *source_ids,
        *learning.cue_ids,
        *learning.mbon_ids,
        *(dan_id for dan_id, _ in learning.dan_to_mbon_pairs),
        *binding.pre_ids.tolist(),
        *binding.post_ids.tolist(),
    }
    missing = sorted(required - set(index_by_id))
    if missing:
        raise ValueError(f"associative motor IDs absent from graph: {missing}")
    state = ShiuState.initial(
        graph.neuron_count,
        params=learning.shiu_parameters,
        seed=schedule.seed,
    )
    learner = MushroomBodyLearning(
        overlay=binding.overlay,
        edge_pre_ids=binding.pre_ids,
        edge_post_ids=binding.post_ids,
        dan_ids=np.asarray([item[0] for item in learning.dan_to_mbon_pairs], dtype=np.uint64),
        dan_post_ids=np.asarray([item[1] for item in learning.dan_to_mbon_pairs], dtype=np.uint64),
        parameters=learning.learning_parameters,
    )
    body = backend_factory(body_parameters)
    encoder = ProprioceptiveEncoder(proprio, proprioceptive_calibration)
    decoder = HexapodMotorDecoder(motor)
    motor_ids = {neuron_id for group in motor.groups for neuron_id in group.neuron_ids}
    motor_mask = motor_population_silence_mask(
        graph,
        motor,
        frozenset(motor.named_populations()) if motor_silenced else frozenset(),
    )
    proprio_ids = {
        neuron_id for bank in proprio.banks for neuron_id in bank.neuron_ids
    }
    proprio_mask = np.isin(graph.neuron_ids, tuple(proprio_ids))
    silence_mask = np.logical_or(
        motor_mask,
        proprio_mask if proprioceptive_silenced else np.zeros_like(proprio_mask),
    )
    source_indices = np.asarray(
        [index_by_id[item] for item in source_ids], dtype=np.int64
    )
    motor_spikes = 0
    mbon_spikes = 0
    mbon_spike_counts = {neuron_id: 0 for neuron_id in learning.mbon_ids}
    sensory_voltage_events = 0
    proprioceptive_events = 0
    body_trace = []
    for event in schedule.events:
        current_body = body.observe()
        proprio_events = (
            ()
            if proprioceptive_silenced
            else encoder.encode(
                observe_proprioception(
                    current_body,
                    body_parameters,
                    proprioceptive_calibration,
                ),
                step=state.step,
            )
        )
        proprioceptive_events += len(proprio_events)
        scheduled: dict[int, list[tuple[int, float]]] = {}
        if event.odor_intensity and learning.input_mode == "sensory_path":
            sensory_params = replace(
                learning.shiu_parameters,
                poisson_rate_hz=(
                    learning.shiu_parameters.poisson_rate_hz * event.odor_intensity
                ),
            )
            for relative_step, (indices, voltages) in poisson_voltage_events(
                source_indices,
                steps=chunk_steps,
                params=sensory_params,
                seed=schedule.seed + state.step,
            ).items():
                target = scheduled.setdefault(state.step + relative_step, [])
                target.extend(zip(indices.tolist(), voltages.tolist(), strict=True))
                sensory_voltage_events += int(indices.size)
        elif event.odor_intensity:
            scheduled[state.step] = list(
                zip(
                    source_indices.tolist(),
                    [learning.cue_voltage_mv * event.odor_intensity] * source_indices.size,
                    strict=True,
                )
            )
        for proprio_event in proprio_events:
            scheduled.setdefault(state.step, []).extend(
                zip(
                    (index_by_id[item] for item in proprio_event.neuron_ids),
                    proprio_event.voltages,
                    strict=True,
                )
            )
        external = {
            step: (
                np.asarray([index for index, _ in pairs], dtype=np.int64),
                np.asarray([voltage for _, voltage in pairs], dtype=np.float32),
            )
            for step, pairs in scheduled.items()
        }
        batches = tuple(
            simulate_shiu(
                _effective_graph(graph, binding),
                learning.shiu_parameters,
                steps=chunk_steps,
                external_voltage_events=external,
                state=state,
                silenced=silence_mask,
                presynaptic_transmitter_multipliers=(
                    learning.presynaptic_transmitter_multipliers
                ),
            )
        )
        fired_ids = tuple(
            int(neuron_id) for batch in batches for neuron_id in batch.neuron_ids
        )
        motor_spikes += sum(neuron_id in motor_ids for neuron_id in fired_ids)
        mbon_spikes += sum(neuron_id in learning.mbon_ids for neuron_id in fired_ids)
        for neuron_id in fired_ids:
            if neuron_id in mbon_spike_counts:
                mbon_spike_counts[neuron_id] += 1
        recruitment = reinforcement.recruit(
            AnonymousContact(
                appetitive_intensity=event.appetitive_contact_intensity,
                aversive_intensity=event.aversive_contact_intensity,
            )
        )
        fired_array = np.asarray(fired_ids, dtype=np.uint64)
        learner.step(
            active_kc_ids=fired_array[np.isin(fired_array, learning.cue_ids)],
            active_mbon_ids=fired_array[np.isin(fired_array, learning.mbon_ids)],
            routed_dan_ids=np.asarray(recruitment.dan_ids, dtype=np.uint64),
            dt_ms=chunk_steps * learning.shiu_parameters.dt_ms,
        )
        phases = cast(
            tuple[float, float, float, float, float, float],
            tuple(leg.phase for leg in current_body.legs),
        )
        activation = decoder.decode(
            fired_ids,
            window_s=body_parameters.dt_s,
            leg_phases=phases,
        )
        body_trace.append(body.step(activation.torques).model_dump(mode="json"))
    return _LoopTrace(
        motor_spikes=motor_spikes,
        mbon_spikes=mbon_spikes,
        mbon_spike_counts=tuple(
            (neuron_id, count)
            for neuron_id, count in sorted(mbon_spike_counts.items())
            if count
        ),
        sensory_voltage_events=sensory_voltage_events,
        proprioceptive_events=proprioceptive_events,
        final_multipliers=tuple(float(value) for value in binding.overlay.multipliers),
        body_digest=hashlib.sha256(repr(body_trace).encode("utf-8")).hexdigest(),
    )


def run_associative_motor_calibration(
    graph: EventConnectome,
    binding: PlasticEdgeBinding,
    *,
    learning: AssociativeCalibrationConfig,
    motor: HexapodMotorMap,
    proprio: ProprioceptiveMap,
    schedule: ConditioningSchedule,
    reinforcement: ReinforcementInterface,
    body_parameters: HexapodParameters,
    proprioceptive_calibration: ProprioceptiveCalibration | None = None,
    backend_factory: BackendFactory = ReferenceHexapodBackend,
    motor_silenced: bool = False,
    proprioceptive_silenced: bool = False,
) -> AssociativeMotorCalibrationResult:
    """Run a stateful sensory-learning-motor-body loop and exact independent replay."""

    before = _graph_digest(graph)
    active_calibration = proprioceptive_calibration or ProprioceptiveCalibration()
    first = _run(
        graph,
        _clone_binding(binding),
        learning=learning,
        motor=motor,
        proprio=proprio,
        schedule=schedule,
        reinforcement=reinforcement,
        body_parameters=body_parameters,
        proprioceptive_calibration=active_calibration,
        backend_factory=backend_factory,
        motor_silenced=motor_silenced,
        proprioceptive_silenced=proprioceptive_silenced,
    )
    replay = _run(
        graph,
        _clone_binding(binding),
        learning=learning,
        motor=motor,
        proprio=proprio,
        schedule=schedule,
        reinforcement=reinforcement,
        body_parameters=body_parameters,
        proprioceptive_calibration=active_calibration,
        backend_factory=backend_factory,
        motor_silenced=motor_silenced,
        proprioceptive_silenced=proprioceptive_silenced,
    )
    return AssociativeMotorCalibrationResult(
        replay_exact=first == replay,
        graph_unchanged=_graph_digest(graph) == before,
        motor_spikes=first.motor_spikes,
        mbon_spikes=first.mbon_spikes,
        mbon_spike_counts=first.mbon_spike_counts,
        sensory_voltage_events=first.sensory_voltage_events,
        proprioceptive_events=first.proprioceptive_events,
        final_multipliers=first.final_multipliers,
        trace_digest=first.digest,
    )


def run_associative_motor_learning_probe(
    graph: EventConnectome,
    binding: PlasticEdgeBinding,
    *,
    learning: AssociativeCalibrationConfig,
    motor: HexapodMotorMap,
    proprio: ProprioceptiveMap,
    training_schedule: ConditioningSchedule,
    probe_schedule: ConditioningSchedule,
    reinforcement: ReinforcementInterface,
    body_parameters: HexapodParameters,
    proprioceptive_calibration: ProprioceptiveCalibration | None = None,
    backend_factory: BackendFactory = ReferenceHexapodBackend,
    motor_silenced: bool = False,
    proprioceptive_silenced: bool = False,
) -> AssociativeMotorLearningProbeResult:
    """Compare an odor-only probe before/after training with reset dynamic state.

    The graph, body, and Shiu state restart for each probe. Only the learned sparse
    KC-to-MBON multipliers persist into the learned condition.
    """

    if any(
        event.appetitive_contact_intensity or event.aversive_contact_intensity
        for event in probe_schedule.events
    ):
        raise ValueError("learning probe must not contain a contact signal")
    before = _graph_digest(graph)
    active_calibration = proprioceptive_calibration or ProprioceptiveCalibration()

    def sequence() -> tuple[_LoopTrace, _LoopTrace, _LoopTrace]:
        trained_binding = _clone_binding(binding)
        training = _run(
            graph,
            trained_binding,
            learning=learning,
            motor=motor,
            proprio=proprio,
            schedule=training_schedule,
            reinforcement=reinforcement,
            body_parameters=body_parameters,
            proprioceptive_calibration=active_calibration,
            backend_factory=backend_factory,
            motor_silenced=motor_silenced,
            proprioceptive_silenced=proprioceptive_silenced,
        )
        learned_probe = _run(
            graph,
            trained_binding,
            learning=learning,
            motor=motor,
            proprio=proprio,
            schedule=probe_schedule,
            reinforcement=reinforcement,
            body_parameters=body_parameters,
            proprioceptive_calibration=active_calibration,
            backend_factory=backend_factory,
            motor_silenced=motor_silenced,
            proprioceptive_silenced=proprioceptive_silenced,
        )
        baseline_probe = _run(
            graph,
            _clone_binding(binding),
            learning=learning,
            motor=motor,
            proprio=proprio,
            schedule=probe_schedule,
            reinforcement=reinforcement,
            body_parameters=body_parameters,
            proprioceptive_calibration=active_calibration,
            backend_factory=backend_factory,
            motor_silenced=motor_silenced,
            proprioceptive_silenced=proprioceptive_silenced,
        )
        return training, learned_probe, baseline_probe

    first_training, first_learned, first_baseline = sequence()
    replay_training, replay_learned, replay_baseline = sequence()
    difference = first_learned.motor_spikes - first_baseline.motor_spikes
    mbon_difference = first_learned.mbon_spikes - first_baseline.mbon_spikes
    if first_baseline.motor_spikes == 0 and first_learned.motor_spikes == 0:
        classification: Literal["motor_difference", "null", "underpowered"] = (
            "underpowered"
        )
    elif difference == 0:
        classification = "null"
    else:
        classification = "motor_difference"
    trace_payload = (
        first_training.digest,
        first_learned.digest,
        first_baseline.digest,
    )
    return AssociativeMotorLearningProbeResult(
        classification=classification,
        replay_exact=(
            first_training == replay_training
            and first_learned == replay_learned
            and first_baseline == replay_baseline
        ),
        graph_unchanged=_graph_digest(graph) == before,
        training_final_multipliers=first_training.final_multipliers,
        baseline_probe_mbon_spikes=first_baseline.mbon_spikes,
        learned_probe_mbon_spikes=first_learned.mbon_spikes,
        probe_mbon_spike_difference=mbon_difference,
        baseline_probe_mbon_spike_counts=first_baseline.mbon_spike_counts,
        learned_probe_mbon_spike_counts=first_learned.mbon_spike_counts,
        baseline_probe_motor_spikes=first_baseline.motor_spikes,
        learned_probe_motor_spikes=first_learned.motor_spikes,
        probe_motor_spike_difference=difference,
        trace_digest=hashlib.sha256(repr(trace_payload).encode("utf-8")).hexdigest(),
    )
