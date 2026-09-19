"""Autonomous sensory-to-body episodes with contact-gated local learning."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flybrain.autonomous_learning_benchmark import _effective_graph
from flybrain.embodied_interfaces import (
    ExternalEvent,
    MotorDecoder,
    MotorMap,
    SensoryEncoder,
    SensoryMap,
)
from flybrain.embodied_world import ArenaWorld, FlyBody, MotorCommand
from flybrain.graph import EventConnectome
from flybrain.mushroom_body_learning import (
    MushroomBodyLearning,
    MushroomBodyLearningParameters,
)
from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.reinforcement_interface import AnonymousContact, ReinforcementInterface
from flybrain.shiu import ShiuParameters, ShiuState, simulate_shiu


class AutonomousEpisodeConfig(BaseModel, frozen=True):
    """Bounded episode configuration with no target or action input."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    max_steps: int = Field(gt=0)
    chunk_steps: int = Field(gt=0)
    seed: int = Field(ge=0)
    sensory_map: SensoryMap
    motor_map: MotorMap
    shiu_parameters: ShiuParameters = ShiuParameters()
    learning_parameters: MushroomBodyLearningParameters = MushroomBodyLearningParameters()

    @model_validator(mode="after")
    def validate_parameters(self) -> AutonomousEpisodeConfig:
        self.shiu_parameters.validate()
        self.learning_parameters.validate()
        return self


class AutonomousEpisodeResult(BaseModel, frozen=True):
    """Auditable episode trace; behavior claims remain gated until benchmarks pass."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["autonomous-contact-learning-v1"] = "autonomous-contact-learning-v1"
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    autonomous_behavior_claim_allowed: Literal[False] = False
    replay_exact: bool
    graph_unchanged: bool
    steps: int = Field(gt=0)
    neural_steps: int = Field(gt=0)
    contact_events: int = Field(ge=0)
    dan_events: int = Field(ge=0)
    motor_spikes: int = Field(ge=0)
    active_motor_ids: tuple[int, ...]
    motor_population_coverage: float = Field(ge=0.0, le=1.0)
    all_declared_motors_active: bool
    sensory_events: tuple[ExternalEvent, ...]
    actions: tuple[MotorCommand, ...]
    body_trace: tuple[FlyBody, ...]
    final_multipliers: tuple[float, ...]
    learning_applied: bool
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class _Trace:
    sensory_events: tuple[ExternalEvent, ...]
    actions: tuple[MotorCommand, ...]
    body_trace: tuple[FlyBody, ...]
    contact_events: int
    dan_events: int
    motor_spikes: int
    active_motor_ids: tuple[int, ...]
    final_multipliers: tuple[float, ...]

    @property
    def digest(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()


def _clone_binding(binding: PlasticEdgeBinding) -> PlasticEdgeBinding:
    return PlasticEdgeBinding(
        overlay=binding.overlay.copy(),
        pre_ids=binding.pre_ids.copy(),
        post_ids=binding.post_ids.copy(),
    )


def _clone_world(world: ArenaWorld) -> ArenaWorld:
    return ArenaWorld(
        world.config,
        world.body,
        food=world.food,
        threat=world.threat,
        wall_segments=world.wall_segments,
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


def _validate_interfaces(
    graph: EventConnectome,
    config: AutonomousEpisodeConfig,
    binding: PlasticEdgeBinding,
    dan_to_mbon_pairs: tuple[tuple[int, int], ...],
    reinforcement: ReinforcementInterface,
) -> None:
    declared_interfaces = (
        *config.sensory_map.visual_ids,
        *config.sensory_map.odor_ids,
        *config.sensory_map.touch_ids,
        *config.sensory_map.proprioception_ids,
        *config.motor_map.left_ids,
        *config.motor_map.right_ids,
        *config.motor_map.forward_ids,
    )
    available = {int(value) for value in graph.neuron_ids}
    missing = sorted(set(declared_interfaces) - available)
    missing += sorted(set(binding.pre_ids.tolist()) - available)
    missing += sorted(set(binding.post_ids.tolist()) - available)
    missing += sorted({pre for pre, _ in dan_to_mbon_pairs} - available)
    missing += sorted({post for _, post in dan_to_mbon_pairs} - available)
    if missing:
        raise ValueError(f"autonomous episode IDs absent from graph: {sorted(set(missing))}")
    if len(set(dan_to_mbon_pairs)) != len(dan_to_mbon_pairs):
        raise ValueError("DAN-to-MBON routes must be unique")
    if any(post not in set(binding.post_ids.tolist()) for _, post in dan_to_mbon_pairs):
        raise ValueError("DAN-to-MBON routes must target plastic MBON endpoints")
    permitted = set(reinforcement.appetitive_dan_ids) | set(reinforcement.aversive_dan_ids)
    if any(pre not in permitted for pre, _ in dan_to_mbon_pairs):
        raise ValueError("DAN-to-MBON routes must use declared reinforcement DANs")
    declared_motor_ids = {
        *config.motor_map.left_ids,
        *config.motor_map.right_ids,
        *config.motor_map.forward_ids,
    }
    if not declared_motor_ids:
        raise ValueError("autonomous episode requires at least one declared motor neuron")


def _execute(
    graph: EventConnectome,
    config: AutonomousEpisodeConfig,
    *,
    binding: PlasticEdgeBinding,
    dan_to_mbon_pairs: tuple[tuple[int, int], ...],
    reinforcement: ReinforcementInterface,
    world: ArenaWorld,
) -> _Trace:
    _validate_interfaces(graph, config, binding, dan_to_mbon_pairs, reinforcement)
    neural_window_s = (
        config.chunk_steps * config.shiu_parameters.dt_ms / 1000.0
    )
    if not math.isclose(neural_window_s, world.config.dt_s, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("neural chunk duration must equal the body timestep")
    index_by_id = {int(neuron_id): index for index, neuron_id in enumerate(graph.neuron_ids)}
    state = ShiuState.initial(graph.neuron_count, params=config.shiu_parameters, seed=config.seed)
    encoder = SensoryEncoder(
        config.sensory_map,
        arena_width=world.config.width,
        arena_height=world.config.height,
    )
    decoder = MotorDecoder(config.motor_map)
    learner = MushroomBodyLearning(
        overlay=binding.overlay,
        edge_pre_ids=binding.pre_ids,
        edge_post_ids=binding.post_ids,
        dan_ids=np.asarray([pre for pre, _ in dan_to_mbon_pairs], dtype=np.uint64),
        dan_post_ids=np.asarray([post for _, post in dan_to_mbon_pairs], dtype=np.uint64),
        parameters=config.learning_parameters,
    )
    sensory_ids = (
        *config.sensory_map.visual_ids,
        *config.sensory_map.odor_ids,
        *config.sensory_map.touch_ids,
        *config.sensory_map.proprioception_ids,
    )
    sensory_indices = np.asarray(
        [index_by_id[neuron_id] for neuron_id in sensory_ids], dtype=np.int64
    )
    refractory_exempt = np.zeros(graph.neuron_count, dtype=np.bool_)
    refractory_exempt[sensory_indices] = True
    sensory_events: list[ExternalEvent] = []
    actions: list[MotorCommand] = []
    body_trace: list[FlyBody] = []
    contact_events = 0
    dan_events = 0
    declared_motor_ids = {
        *config.motor_map.left_ids,
        *config.motor_map.right_ids,
        *config.motor_map.forward_ids,
    }
    motor_spikes = 0
    active_motor_ids: set[int] = set()
    for step in range(config.max_steps):
        encoded = encoder.encode(world.observe(), step=step)
        sensory_events.extend(encoded)
        indices = [index_by_id[neuron_id] for event in encoded for neuron_id in event.neuron_ids]
        voltages = [voltage for event in encoded for voltage in event.voltages]
        external = (
            {
                state.step + offset: (
                    np.asarray(indices, dtype=np.int64),
                    np.asarray(voltages, dtype=np.float32),
                )
                for offset in range(config.chunk_steps)
            }
            if indices
            else {}
        )
        active_graph = _effective_graph(graph, binding)
        batches = simulate_shiu(
            active_graph,
            config.shiu_parameters,
            steps=config.chunk_steps,
            external_voltage_events=external,
            state=state,
            refractory_exempt=refractory_exempt,
        )
        fired_ids = tuple(int(value) for batch in batches for value in batch.neuron_ids)
        fired_motor_ids = tuple(
            neuron_id for neuron_id in fired_ids if neuron_id in declared_motor_ids
        )
        motor_spikes += len(fired_motor_ids)
        active_motor_ids.update(fired_motor_ids)
        command = decoder.decode(fired_ids)
        actions.append(command)
        result = world.step(command)
        body_trace.append(result.body)
        contact = AnonymousContact(
            appetitive_intensity=1.0 if result.food_contact else 0.0,
            aversive_intensity=1.0 if result.threat_contact else 0.0,
        )
        if result.food_contact or result.threat_contact:
            contact_events += 1
        recruitment = reinforcement.recruit(contact)
        if recruitment.dan_ids:
            dan_events += 1
        fired = np.asarray(fired_ids, dtype=np.uint64)
        learner.step(
            active_kc_ids=fired[np.isin(fired, binding.pre_ids)],
            active_mbon_ids=fired[np.isin(fired, binding.post_ids)],
            routed_dan_ids=np.asarray(recruitment.dan_ids, dtype=np.uint64),
            dt_ms=config.chunk_steps * config.shiu_parameters.dt_ms,
        )
    return _Trace(
        sensory_events=tuple(sensory_events),
        actions=tuple(actions),
        body_trace=tuple(body_trace),
        contact_events=contact_events,
        dan_events=dan_events,
        motor_spikes=motor_spikes,
        active_motor_ids=tuple(sorted(active_motor_ids)),
        final_multipliers=tuple(float(value) for value in binding.overlay.multipliers),
    )


def run_autonomous_episode(
    graph: EventConnectome,
    config: AutonomousEpisodeConfig,
    *,
    binding: PlasticEdgeBinding,
    dan_to_mbon_pairs: tuple[tuple[int, int], ...],
    reinforcement: ReinforcementInterface,
    world: ArenaWorld,
) -> AutonomousEpisodeResult:
    """Run an autonomous closed loop and an independent exact replay.

    World contact is the only source of reinforcement. The neural system receives
    anonymous sensory events and proprioceptive channels, never a target or action.
    """

    before = _graph_digest(graph)
    first_binding = _clone_binding(binding)
    first = _execute(
        graph,
        config,
        binding=first_binding,
        dan_to_mbon_pairs=dan_to_mbon_pairs,
        reinforcement=reinforcement,
        world=_clone_world(world),
    )
    replay = _execute(
        graph,
        config,
        binding=_clone_binding(binding),
        dan_to_mbon_pairs=dan_to_mbon_pairs,
        reinforcement=reinforcement,
        world=_clone_world(world),
    )
    declared_motor_ids = {
        *config.motor_map.left_ids,
        *config.motor_map.right_ids,
        *config.motor_map.forward_ids,
    }
    motor_population_coverage = (
        len(first.active_motor_ids) / len(declared_motor_ids)
        if declared_motor_ids
        else 1.0
    )
    return AutonomousEpisodeResult(
        replay_exact=first == replay,
        graph_unchanged=_graph_digest(graph) == before,
        steps=config.max_steps,
        neural_steps=config.max_steps * config.chunk_steps,
        contact_events=first.contact_events,
        dan_events=first.dan_events,
        motor_spikes=first.motor_spikes,
        active_motor_ids=first.active_motor_ids,
        motor_population_coverage=motor_population_coverage,
        all_declared_motors_active=set(first.active_motor_ids) == declared_motor_ids,
        sensory_events=first.sensory_events,
        actions=first.actions,
        body_trace=first.body_trace,
        final_multipliers=first.final_multipliers,
        learning_applied=any(value != 1.0 for value in first.final_multipliers),
        trace_digest=first.digest,
    )
