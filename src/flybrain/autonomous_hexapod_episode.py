"""Contact-driven associative learning on the complete hexapod motor interface."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Literal, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flybrain.associative_motor_loop import BackendFactory
from flybrain.autonomous_learning_benchmark import (
    AssociativeCalibrationConfig,
    _effective_graph,
)
from flybrain.graph import EventConnectome
from flybrain.hexapod_backend import BackendIdentity, ReferenceHexapodBackend
from flybrain.hexapod_body import HexapodBody, HexapodParameters
from flybrain.hexapod_motor import HexapodMotorDecoder, HexapodMotorMap
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
from flybrain.slow_memory import SlowMemoryParameters, SlowMemoryState


class HexapodArenaConfig(BaseModel, frozen=True):
    """Physical source locations kept inside the environment boundary."""

    model_config = ConfigDict(extra="forbid")

    food_position_m: tuple[float, float]
    threat_position_m: tuple[float, float]
    contact_radius_m: float = Field(gt=0.0)
    odor_length_scale_m: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_finite(self) -> Self:
        values = (
            *self.food_position_m,
            *self.threat_position_m,
            self.contact_radius_m,
            self.odor_length_scale_m,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("hexapod arena values must be finite")
        return self


class AutonomousHexapodConfig(BaseModel, frozen=True):
    """Bounded autonomous episode inputs without schedules or desired actions."""

    model_config = ConfigDict(extra="forbid")

    body_steps: int = Field(gt=0)
    learning: AssociativeCalibrationConfig
    arena: HexapodArenaConfig
    seed: int = Field(ge=0)
    nitric_oxide_dan_ids: tuple[int, ...] = ()
    slow_memory_parameters: SlowMemoryParameters = Field(
        default_factory=SlowMemoryParameters
    )

    @model_validator(mode="after")
    def validate_sensory_path(self) -> Self:
        if self.learning.input_mode != "sensory_path":
            raise ValueError("autonomous hexapod requires the canonical sensory path")
        if not self.learning.sensory_input_ids:
            raise ValueError("autonomous hexapod requires declared sensory input IDs")
        if any(neuron_id <= 0 for neuron_id in self.nitric_oxide_dan_ids):
            raise ValueError("nitric-oxide DAN IDs must be positive")
        if len(set(self.nitric_oxide_dan_ids)) != len(self.nitric_oxide_dan_ids):
            raise ValueError("nitric-oxide DAN IDs must be unique")
        self.slow_memory_parameters.validate()
        return self


class AutonomousHexapodResult(BaseModel, frozen=True):
    """Auditable closed-loop trace summary with conservative claim gating."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["autonomous-hexapod-contact-learning-v1"] = (
        "autonomous-hexapod-contact-learning-v1"
    )
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    autonomous_behavior_claim_allowed: Literal[False] = False
    backend: BackendIdentity
    replay_exact: bool
    graph_unchanged: bool
    body_steps: int = Field(gt=0)
    neural_steps: int = Field(gt=0)
    appetitive_contacts: int = Field(ge=0)
    aversive_contacts: int = Field(ge=0)
    dan_events: int = Field(ge=0)
    motor_spikes: int = Field(ge=0)
    active_motor_groups: int = Field(ge=0, le=24)
    all_motor_groups_active: bool
    proprioceptive_events: int = Field(ge=0)
    slow_memory_enabled: bool
    slow_memory_edges: int = Field(ge=0)
    slow_memory_dopamine_effect_max: float = Field(ge=0.0, le=1.0)
    slow_memory_nitric_oxide_effect_max: float = Field(ge=0.0, le=1.0)
    final_multipliers: tuple[float, ...]
    final_body: HexapodBody
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class _Trace:
    backend: BackendIdentity
    appetitive_contacts: int
    aversive_contacts: int
    dan_events: int
    motor_spikes: int
    active_motor_groups: tuple[str, ...]
    proprioceptive_events: int
    slow_memory_enabled: bool
    slow_memory_edges: int
    slow_memory_dopamine_effect_max: float
    slow_memory_nitric_oxide_effect_max: float
    final_multipliers: tuple[float, ...]
    final_body: HexapodBody
    body_digest: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()


def _clone_binding(binding: PlasticEdgeBinding) -> PlasticEdgeBinding:
    return PlasticEdgeBinding(
        overlay=binding.overlay.copy(),
        pre_ids=binding.pre_ids.copy(),
        post_ids=binding.post_ids.copy(),
    )


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


def _distance(body: HexapodBody, point: tuple[float, float]) -> float:
    return math.hypot(
        body.thorax_position_m[0] - point[0],
        body.thorax_position_m[1] - point[1],
    )


def _odor_intensity(body: HexapodBody, arena: HexapodArenaConfig) -> float:
    distance = min(
        _distance(body, arena.food_position_m),
        _distance(body, arena.threat_position_m),
    )
    return math.exp(-distance / arena.odor_length_scale_m)


def _run(
    graph: EventConnectome,
    binding: PlasticEdgeBinding,
    config: AutonomousHexapodConfig,
    *,
    motor: HexapodMotorMap,
    proprio: ProprioceptiveMap,
    reinforcement: ReinforcementInterface,
    body_parameters: HexapodParameters,
    backend_factory: BackendFactory,
    proprioceptive_calibration: ProprioceptiveCalibration,
) -> _Trace:
    motor.validate_graph(graph)
    proprio.validate_graph(graph)
    learning = config.learning
    neural_window_s = (
        learning.neural_chunk_steps * learning.shiu_parameters.dt_ms / 1000.0
    )
    if not math.isclose(neural_window_s, body_parameters.dt_s, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("neural chunk duration must equal the backend timestep")
    index_by_id = {int(neuron_id): index for index, neuron_id in enumerate(graph.neuron_ids)}
    required = {
        *learning.sensory_input_ids,
        *learning.cue_ids,
        *learning.mbon_ids,
        *(pre for pre, _ in learning.dan_to_mbon_pairs),
        *(post for _, post in learning.dan_to_mbon_pairs),
        *binding.pre_ids.tolist(),
        *binding.post_ids.tolist(),
    }
    missing = sorted(required - set(index_by_id))
    if missing:
        raise ValueError(f"autonomous hexapod IDs absent from graph: {missing}")
    state = ShiuState.initial(
        graph.neuron_count,
        params=learning.shiu_parameters,
        seed=config.seed,
    )
    learner = MushroomBodyLearning(
        overlay=(
            binding.overlay.copy()
            if config.nitric_oxide_dan_ids
            else binding.overlay
        ),
        edge_pre_ids=binding.pre_ids,
        edge_post_ids=binding.post_ids,
        dan_ids=np.asarray([item[0] for item in learning.dan_to_mbon_pairs], dtype=np.uint64),
        dan_post_ids=np.asarray(
            [item[1] for item in learning.dan_to_mbon_pairs], dtype=np.uint64
        ),
        parameters=learning.learning_parameters,
    )
    slow_state: SlowMemoryState | None = None
    if config.nitric_oxide_dan_ids:
        no_dan_ids = set(config.nitric_oxide_dan_ids)
        no_target_ids = np.asarray(
            [
                post_id
                for dan_id, post_id in learning.dan_to_mbon_pairs
                if dan_id in no_dan_ids
            ],
            dtype=np.uint64,
        )
        competent = np.isin(binding.post_ids, no_target_ids)
        if not np.any(competent):
            raise ValueError(
                "nitric-oxide DANs do not route to any bound KC-to-MBON edge"
            )
        slow_state = SlowMemoryState.initial(nitric_oxide_competent=competent)
    backend = backend_factory(body_parameters)
    decoder = HexapodMotorDecoder(motor)
    proprio_encoder = ProprioceptiveEncoder(proprio, proprioceptive_calibration)
    source_indices = np.asarray(
        [index_by_id[item] for item in learning.sensory_input_ids], dtype=np.int64
    )
    motor_owner = {
        neuron_id: group.name for group in motor.groups for neuron_id in group.neuron_ids
    }
    motor_spikes = 0
    active_groups: set[str] = set()
    proprioceptive_events = 0
    appetitive_contacts = 0
    aversive_contacts = 0
    dan_events = 0
    body_trace = []
    for _ in range(config.body_steps):
        body = backend.observe()
        intensity = _odor_intensity(body, config.arena)
        sensory_params = replace(
            learning.shiu_parameters,
            poisson_rate_hz=learning.shiu_parameters.poisson_rate_hz * intensity,
        )
        scheduled: dict[int, list[tuple[int, float]]] = {}
        for relative_step, (indices, voltages) in poisson_voltage_events(
            source_indices,
            steps=learning.neural_chunk_steps,
            params=sensory_params,
            seed=config.seed + state.step,
        ).items():
            scheduled[state.step + relative_step] = list(
                zip(indices.tolist(), voltages.tolist(), strict=True)
            )
        proprio_events = proprio_encoder.encode(
            observe_proprioception(body, body_parameters, proprioceptive_calibration),
            step=state.step,
        )
        proprioceptive_events += len(proprio_events)
        for event in proprio_events:
            scheduled.setdefault(state.step, []).extend(
                zip(
                    (index_by_id[item] for item in event.neuron_ids),
                    event.voltages,
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
                steps=learning.neural_chunk_steps,
                external_voltage_events=external,
                state=state,
                presynaptic_transmitter_multipliers=(
                    learning.presynaptic_transmitter_multipliers
                ),
            )
        )
        fired_ids = tuple(
            int(neuron_id) for batch in batches for neuron_id in batch.neuron_ids
        )
        for neuron_id in fired_ids:
            group = motor_owner.get(neuron_id)
            if group is not None:
                motor_spikes += 1
                active_groups.add(group)
        phases = cast(
            tuple[float, float, float, float, float, float],
            tuple(leg.phase for leg in body.legs),
        )
        activation = decoder.decode(
            fired_ids,
            window_s=body_parameters.dt_s,
            leg_phases=phases,
        )
        next_body = backend.step(activation.torques)
        body_trace.append(next_body.model_dump(mode="json"))
        food_contact = (
            _distance(next_body, config.arena.food_position_m)
            <= config.arena.contact_radius_m
        )
        threat_contact = (
            _distance(next_body, config.arena.threat_position_m)
            <= config.arena.contact_radius_m
        )
        appetitive_contacts += int(food_contact)
        aversive_contacts += int(threat_contact)
        recruitment = reinforcement.recruit(
            AnonymousContact(
                appetitive_intensity=1.0 if food_contact else 0.0,
                aversive_intensity=1.0 if threat_contact else 0.0,
            )
        )
        dan_events += int(bool(recruitment.dan_ids))
        fired = np.asarray(fired_ids, dtype=np.uint64)
        learner.step(
            active_kc_ids=fired[np.isin(fired, binding.pre_ids)],
            active_mbon_ids=fired[np.isin(fired, binding.post_ids)],
            routed_dan_ids=np.asarray(recruitment.dan_ids, dtype=np.uint64),
            dt_ms=learning.neural_chunk_steps * learning.shiu_parameters.dt_ms,
        )
        if slow_state is not None:
            routed_dans = set(recruitment.dan_ids)
            routed_posts = np.asarray(
                [
                    post_id
                    for dan_id, post_id in learning.dan_to_mbon_pairs
                    if dan_id in routed_dans
                ],
                dtype=np.uint64,
            )
            active_kc_edges = np.isin(binding.pre_ids, fired)
            routed_edges = np.isin(binding.post_ids, routed_posts)
            slow_state.step(
                paired=active_kc_edges & routed_edges,
                dan_unpaired=(~active_kc_edges) & routed_edges,
                dt_seconds=neural_window_s,
                parameters=config.slow_memory_parameters,
            )
            slow_multipliers = np.asarray(
                learner.overlay.multipliers * slow_state.weight_multipliers,
                dtype=np.float32,
            )
            binding.overlay.set_multipliers(
                slow_multipliers,
                minimum=0.0,
                maximum=2.0,
            )
    final_body = backend.observe()
    dopamine_effect_max = (
        float(np.max(slow_state.dopamine_effect)) if slow_state is not None else 0.0
    )
    nitric_oxide_effect_max = (
        float(np.max(slow_state.nitric_oxide_effect)) if slow_state is not None else 0.0
    )
    return _Trace(
        backend=backend.identity,
        appetitive_contacts=appetitive_contacts,
        aversive_contacts=aversive_contacts,
        dan_events=dan_events,
        motor_spikes=motor_spikes,
        active_motor_groups=tuple(sorted(active_groups)),
        proprioceptive_events=proprioceptive_events,
        slow_memory_enabled=slow_state is not None,
        slow_memory_edges=(
            int(np.count_nonzero(slow_state.nitric_oxide_competent))
            if slow_state is not None
            else 0
        ),
        slow_memory_dopamine_effect_max=dopamine_effect_max,
        slow_memory_nitric_oxide_effect_max=nitric_oxide_effect_max,
        final_multipliers=tuple(float(value) for value in binding.overlay.multipliers),
        final_body=final_body,
        body_digest=hashlib.sha256(repr(body_trace).encode("utf-8")).hexdigest(),
    )


def run_autonomous_hexapod_episode(
    graph: EventConnectome,
    binding: PlasticEdgeBinding,
    config: AutonomousHexapodConfig,
    *,
    motor: HexapodMotorMap,
    proprio: ProprioceptiveMap,
    reinforcement: ReinforcementInterface,
    body_parameters: HexapodParameters,
    backend_factory: BackendFactory = ReferenceHexapodBackend,
    proprioceptive_calibration: ProprioceptiveCalibration | None = None,
) -> AutonomousHexapodResult:
    """Run and replay a schedule-free physical-contact learning episode."""

    before = _graph_digest(graph)
    calibration = proprioceptive_calibration or ProprioceptiveCalibration()
    first = _run(
        graph,
        _clone_binding(binding),
        config,
        motor=motor,
        proprio=proprio,
        reinforcement=reinforcement,
        body_parameters=body_parameters,
        backend_factory=backend_factory,
        proprioceptive_calibration=calibration,
    )
    replay = _run(
        graph,
        _clone_binding(binding),
        config,
        motor=motor,
        proprio=proprio,
        reinforcement=reinforcement,
        body_parameters=body_parameters,
        backend_factory=backend_factory,
        proprioceptive_calibration=calibration,
    )
    return AutonomousHexapodResult(
        backend=first.backend,
        replay_exact=first == replay,
        graph_unchanged=_graph_digest(graph) == before,
        body_steps=config.body_steps,
        neural_steps=config.body_steps * config.learning.neural_chunk_steps,
        appetitive_contacts=first.appetitive_contacts,
        aversive_contacts=first.aversive_contacts,
        dan_events=first.dan_events,
        motor_spikes=first.motor_spikes,
        active_motor_groups=len(first.active_motor_groups),
        all_motor_groups_active=len(first.active_motor_groups) == 24,
        proprioceptive_events=first.proprioceptive_events,
        slow_memory_enabled=first.slow_memory_enabled,
        slow_memory_edges=first.slow_memory_edges,
        slow_memory_dopamine_effect_max=first.slow_memory_dopamine_effect_max,
        slow_memory_nitric_oxide_effect_max=(
            first.slow_memory_nitric_oxide_effect_max
        ),
        final_multipliers=first.final_multipliers,
        final_body=first.final_body,
        trace_digest=first.digest,
    )
